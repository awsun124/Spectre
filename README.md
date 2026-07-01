# Spectre

Website URL: https://www.spectremlmusic.com/

Spectre is a music genre classifier built around a CNN trained on the GTZAN dataset.

The project has two main parts. The frontend is a React app with a circular upload/scanning interface, the predicted genre, model confidence, and a similar-song leaderboard. The backend is a FastAPI server that loads our trained PyTorch CNN weights from `best_model.pth`, turns uploaded audio into mel spectrogram chunks, and runs the model to classify the song.

It takes embeddings from the model's learned feature layers, compares the uploaded song against cached embeddings from the GTZAN library, and returns the closest matches by cosine similarity.
