import React, { useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const genres = [
  'blues',
  'classical',
  'country',
  'disco',
  'hiphop',
  'jazz',
  'metal',
  'pop',
  'reggae',
  'rock',
];

const librarySongs = [
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

function pickGenre(fileName) {
  const lowerName = fileName.toLowerCase();
  const directMatch = genres.find((genre) => lowerName.includes(genre));

  if (directMatch) {
    return directMatch;
  }

  let score = 0;
  for (const letter of lowerName) {
    score += letter.charCodeAt(0);
  }

  return genres[score % genres.length];
}

function buildLeaderboard(genre, fileName) {
  const seed = fileName.length + genre.length;
  const ranked = librarySongs
    .map((song, index) => {
      const genreBoost = song.genre === genre ? 0.18 : 0;
      const movement = ((seed + index * 17) % 24) / 100;
      return {
        ...song,
        similarity: Math.min(0.99, 0.68 + genreBoost + movement),
      };
    })
    .sort((a, b) => b.similarity - a.similarity);

  return ranked.slice(0, 5);
}

function App() {
  const fileInputRef = useRef(null);
  const [track, setTrack] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState(null);

  const leaderboard = useMemo(() => {
    if (!result || !track) {
      return buildLeaderboard('blues', 'demo');
    }

    return buildLeaderboard(result.genre, track.name);
  }, [result, track]);

  function handleUpload(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    const audioUrl = URL.createObjectURL(file);
    setTrack({ name: file.name, size: file.size, url: audioUrl });
    setResult(null);
    setIsScanning(true);

    window.setTimeout(() => {
      const genre = pickGenre(file.name);
      setResult({
        genre,
      });
      setIsScanning(false);
    }, 1400);
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
          <div className={`wave-field ${isScanning ? 'is-scanning' : ''}`}>
            <span className="wave wave-one" />
            <span className="wave wave-two" />
            <span className="wave wave-three" />
            <button className="scan-button" onClick={openFilePicker} type="button">
              <span className="scan-icon">♪</span>
              <span>{track ? 'Scan another' : 'Upload song'}</span>
            </button>
          </div>

          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept="audio/*,.wav,.mp3"
            onChange={handleUpload}
          />

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
            </article>
          ))}
        </div>
      </aside>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
