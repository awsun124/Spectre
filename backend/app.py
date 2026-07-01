import os
import tempfile
from collections import Counter

import librosa
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from similar_songs import SongLibrary, build_uploaded_identity


SAMPLE_RATE = 22050
CHUNK_SECONDS = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
N_MELS = 128
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pth")


class GenreCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenreCNN(n_classes=len(GENRES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
song_library = SongLibrary(model, audio_to_chunks=lambda filepath: audio_to_chunks(filepath), device=device)

app = FastAPI(title="Spectre Genre Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def audio_to_chunks(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    chunks = []
    total_chunks = max(1, int(np.ceil(len(y) / CHUNK_SAMPLES)))
    for index in range(total_chunks):
        start = index * CHUNK_SAMPLES
        chunk = y[start : start + CHUNK_SAMPLES]

        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        chunks.append(mel_db)

    return torch.tensor(np.stack(chunks), dtype=torch.float32).unsqueeze(1)


dataset_dir = os.getenv("GTZAN_DATASET_DIR")
if dataset_dir and not song_library.ready:
    song_library.build_from_directory(dataset_dir)


@app.get("/health")
def health():
    return {"ok": True, "device": str(device), "songLibraryReady": song_library.ready}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        chunks = audio_to_chunks(temp_path).to(device)
        with torch.no_grad():
            logits = model(chunks)
            probabilities = torch.softmax(logits, dim=1)
            chunk_predictions = probabilities.argmax(dim=1).cpu().tolist()
            predicted_index = Counter(chunk_predictions).most_common(1)[0][0]
            confidence = probabilities[:, predicted_index].mean().item()

        identity = build_uploaded_identity(model, chunks, device)
        return {
            "genre": GENRES[predicted_index],
            "confidence": round(confidence, 4),
            "chunks": len(chunk_predictions),
            "similarSongs": song_library.most_similar_to_identity(identity),
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        os.unlink(temp_path)


@app.post("/similar-songs")
async def similar_songs(file: UploadFile = File(...), k: int = 5):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        return {"similarSongs": song_library.most_similar_to_audio(temp_path, k=k)}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        os.unlink(temp_path)
