# Spectre

Spectre is a React front end for the GTZAN music genre classifier notebook.

The notebook trains a CNN on mel spectrogram chunks, predicts one of 10 genres, and compares learned song embeddings to find similar tracks. This UI turns that idea into a single Shazam-style screen:

- uses the GTZAN music genre classification dataset
- upload a song
- show a scanning animation
- display the predicted genre and confidence
- show a top 5 similar-song leaderboard

The current prediction logic is a front-end demo so the app is usable without a running Python server. The next step is to expose the trained `best_model.pth` through a small API and replace the mock result in `src/main.jsx`.

## Run

```bash
npm install
npm run dev
```
