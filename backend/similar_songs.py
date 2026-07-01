import glob
import os
from collections import defaultdict

import numpy as np
import torch


LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "model", "song_library.npz")


def extract_embedding(model, chunks):
    features = model.conv(chunks)
    pooled_features = model.pool(features)
    return pooled_features.flatten(1)


def build_song_embedding(model, chunks, device):
    model.eval()
    with torch.no_grad():
        chunk_embeddings = extract_embedding(model, chunks.to(device)).cpu().numpy()
    return chunk_embeddings.mean(axis=0)


def cosine_similarity(vector, matrix):
    vector_norm = np.linalg.norm(vector) + 1e-8
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
    return np.dot(matrix, vector) / (matrix_norm * vector_norm)


class SongLibrary:
    def __init__(self, model, audio_to_chunks, device):
        self.model = model
        self.audio_to_chunks = audio_to_chunks
        self.device = device
        self.song_files = []
        self.song_genres = []
        self.song_identity = None
        self.load()

    @property
    def ready(self):
        return self.song_identity is not None and len(self.song_files) > 0

    def load(self):
        if not os.path.exists(LIBRARY_PATH):
            return

        library = np.load(LIBRARY_PATH, allow_pickle=True)
        self.song_files = library["song_files"].tolist()
        self.song_genres = library["song_genres"].tolist()
        self.song_identity = library["song_identity"]

    def save(self):
        os.makedirs(os.path.dirname(LIBRARY_PATH), exist_ok=True)
        np.savez_compressed(
            LIBRARY_PATH,
            song_files=np.array(self.song_files, dtype=object),
            song_genres=np.array(self.song_genres, dtype=object),
            song_identity=self.song_identity,
        )

    def build_from_directory(self, dataset_dir):
        song_paths = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True))
        if not song_paths:
            raise ValueError(f"No .wav files found in {dataset_dir}")

        embeddings_by_song = defaultdict(list)
        genre_by_song = {}

        self.model.eval()
        for filepath in song_paths:
            genre = os.path.basename(os.path.dirname(filepath))
            try:
                chunks = self.audio_to_chunks(filepath)
            except Exception:
                continue

            with torch.no_grad():
                chunk_embeddings = extract_embedding(self.model, chunks.to(self.device)).cpu().numpy()

            embeddings_by_song[filepath].extend(chunk_embeddings)
            genre_by_song[filepath] = genre

        if not embeddings_by_song:
            raise ValueError(f"No readable .wav files found in {dataset_dir}")

        self.song_files = list(embeddings_by_song.keys())
        self.song_genres = [genre_by_song[filepath] for filepath in self.song_files]
        self.song_identity = np.stack(
            [np.mean(embeddings_by_song[filepath], axis=0) for filepath in self.song_files]
        )
        self.save()

    def most_similar_to_embedding(self, embedding, k=5):
        if not self.ready:
            return []

        similarities = cosine_similarity(embedding, self.song_identity)
        order = np.argsort(similarities)[::-1][:k]

        return [
            {
                "title": os.path.basename(self.song_files[index]),
                "genre": self.song_genres[index],
                "similarity": round(float(similarities[index]), 4),
            }
            for index in order
        ]

    def most_similar_to_audio(self, filepath, k=5):
        chunks = self.audio_to_chunks(filepath)
        embedding = build_song_embedding(self.model, chunks, self.device)
        return self.most_similar_to_embedding(embedding, k=k)
