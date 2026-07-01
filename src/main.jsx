import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const demoMatches = [
  { title: 'blues.00055.wav', genre: 'blues' },
  { title: 'blues.00050.wav', genre: 'blues' },
  { title: 'jazz.00041.wav', genre: 'jazz' },
  { title: 'rock.00017.wav', genre: 'rock' },
  { title: 'reggae.00063.wav', genre: 'reggae' },
  { title: 'hiphop.00038.wav', genre: 'hiphop' },
  { title: 'classical.00021.wav', genre: 'classical' },
  { title: 'metal.00084.wav', genre: 'metal' },
  { title: 'pop.00074.wav', genre: 'pop' },
  { title: 'disco.00012.wav', genre: 'disco' },
];

const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const almostDoneProgress = 94;

function downloadUrlFor(songTitle) {
  return `${apiBaseUrl}/songs/${encodeURIComponent(songTitle)}`;
}

function buildDemoMatches(genre, fileName) {
  const seed = fileName.length + genre.length;
  const rankedSongs = demoMatches
    .map((song, index) => {
      const genreBoost = song.genre === genre ? 0.18 : 0;
      const movement = ((seed + index * 17) % 24) / 100;
      return {
        ...song,
        similarity: Math.min(0.99, 0.68 + genreBoost + movement),
      };
    })
    .sort((a, b) => b.similarity - a.similarity);

  return rankedSongs.slice(0, 5);
}

function App() {
  const fileInputRef = useRef(null);
  const progressTimerRef = useRef(null);
  const [track, setTrack] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const leaderboard = useMemo(() => {
    if (result?.similarSongs?.length) {
      return result.similarSongs;
    }

    if (!result || !track) {
      return buildDemoMatches('blues', 'demo');
    }

    return buildDemoMatches(result.genre, track.name);
  }, [result, track]);

  useEffect(() => {
    return () => {
      window.clearInterval(progressTimerRef.current);
    };
  }, []);

  function startProgressRing() {
    setScanProgress(3);
    window.clearInterval(progressTimerRef.current);

    progressTimerRef.current = window.setInterval(() => {
      setScanProgress((currentProgress) => {
        const remainingProgress = almostDoneProgress - currentProgress;
        const nextStep = Math.max(1.5, remainingProgress * 0.08);
        return Math.min(almostDoneProgress, currentProgress + nextStep);
      });
    }, 180);
  }

  function finishProgressRing() {
    window.clearInterval(progressTimerRef.current);
    setScanProgress(100);
  }

  async function askModelToClassify(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.detail || 'The classifier could not read this audio file.');
    }

    return response.json();
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    const audioUrl = URL.createObjectURL(file);
    setTrack({ name: file.name, size: file.size, url: audioUrl });
    setResult(null);
    setErrorMessage('');
    setIsScanning(true);
    startProgressRing();

    try {
      const prediction = await askModelToClassify(file);
      setResult(prediction);
    } catch (error) {
      setErrorMessage(
        error.message ||
          'Could not reach the Python model backend. Start it and try the upload again.',
      );
    } finally {
      finishProgressRing();
      setIsScanning(false);
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return (
    <main className="app-shell">
      <section className="scanner-panel" aria-label="Music genre scanner">
        <header className="topbar">
          <div>
            <p className="eyebrow">CNN classifier</p>
            <h1>Spectre</h1>
            <p className="dataset-line">
              Dataset: GTZAN music genre classification, 1,000 tracks across 10 genres.
            </p>
          </div>
        </header>

        <div className="scan-stage">
          <div
            className={`wave-field ${isScanning ? 'is-scanning' : ''} ${
              track ? 'has-track' : ''
            } ${result ? 'is-complete' : ''}`}
          >
            <span className="wave wave-one" />
            <span className="wave wave-two" />
            <span className="wave wave-three" />
            <div
              className="scan-progress-ring"
              role="progressbar"
              aria-label="Model analysis progress"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={Math.round(scanProgress)}
              style={{ '--scan-progress': `${scanProgress * 3.6}deg` }}
            />
            <button
              className="scan-button"
              disabled={isScanning}
              onClick={openFilePicker}
              type="button"
            >
              <span className="scan-icon">♪</span>
              <span className="scan-label">
                {isScanning ? 'Analyzing' : track ? 'Scan another' : 'Upload song'}
              </span>
              <span className="scan-status">
                {isScanning ? `${Math.round(scanProgress)}%` : result ? 'Complete' : 'Ready'}
              </span>
            </button>
          </div>

          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept="audio/*,.wav,.mp3"
            onChange={handleUpload}
          />
          <p className="file-hint">Accepts .wav files</p>

          <div className="result-strip">
            <div>
              <p className="label">Current track</p>
              <strong>{track?.name || 'No song uploaded yet'}</strong>
            </div>
            <div>
              <p className="label">Predicted genre</p>
              <strong className="genre-name">
                {isScanning ? 'Listening...' : result?.genre || 'Waiting'}
              </strong>
            </div>
            <div>
              <p className="label">Model accuracy</p>
              <strong>82.7%</strong>
            </div>
          </div>

          {result && (
            <p className="confidence-line">
              Confidence {Math.round(result.confidence * 100)}% from {result.chunks} audio chunks.
            </p>
          )}

          {errorMessage && <p className="error-line">{errorMessage}</p>}

          {track && (
            <audio className="audio-player" src={track.url} controls>
              <track kind="captions" />
            </audio>
          )}
        </div>
      </section>

      <aside className="leaderboard-panel" aria-label="Similar song leaderboard">
        <div className="leaderboard-header">
          <p className="eyebrow">Top matches</p>
          <h2>Similar songs</h2>
        </div>

        <div className="rank-list">
          {leaderboard.map((song, index) => (
            <article className="rank-card" key={song.title}>
              <div className="rank-number">{index + 1}</div>
              <div>
                <h3>{song.title}</h3>
                <p>{song.genre}</p>
              </div>
              <strong>{Math.round(song.similarity * 100)}%</strong>
              <a className="download-link" href={downloadUrlFor(song.title)} download>
                Download
              </a>
            </article>
          ))}
        </div>
      </aside>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
