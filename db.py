"""
db.py — SQLite schema and helpers for the artwork auction database.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("auction_data.db")


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH):
    """Create tables if they don't exist."""
    conn = get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS artists (
            id              INTEGER PRIMARY KEY,   -- matches id from artists_scored.json
            qid             TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            score           REAL NOT NULL,
            score_band      INTEGER NOT NULL       -- floor(score), clamped 1–9
        );

        CREATE INDEX IF NOT EXISTS idx_artists_band ON artists(score_band);
        CREATE INDEX IF NOT EXISTS idx_artists_score ON artists(score);

        -- Normalised medium vocabulary (populated from raw strings)
        CREATE TABLE IF NOT EXISTS mediums (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            label   TEXT UNIQUE NOT NULL   -- e.g. "oil on canvas", "bronze", "watercolor"
        );

        CREATE TABLE IF NOT EXISTS artworks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            qid             TEXT UNIQUE,            -- Wikidata QID of the artwork
            title           TEXT,
            artist_id       INTEGER NOT NULL REFERENCES artists(id),
            score_band      INTEGER NOT NULL,       -- denormalised for fast band queries

            -- Temporal
            decade          INTEGER,                -- e.g. 1880, 1920, 2000

            -- Dimensions (all in cm; depth only for 3D works)
            is_3d           INTEGER NOT NULL DEFAULT 0,  -- 0=2D, 1=3D
            width_cm        REAL,
            height_cm       REAL,
            depth_cm        REAL,                   -- NULL for 2D works

            -- Medium
            medium_id       INTEGER REFERENCES mediums(id),
            medium_raw      TEXT,                   -- original string from source

            -- Sale data
            sale_price_usd  REAL,                   -- NULL if unsold / unknown
            sale_year       INTEGER,

            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_artworks_band      ON artworks(score_band);
        CREATE INDEX IF NOT EXISTS idx_artworks_artist    ON artworks(artist_id);
        CREATE INDEX IF NOT EXISTS idx_artworks_decade    ON artworks(decade);
        CREATE INDEX IF NOT EXISTS idx_artworks_price     ON artworks(sale_price_usd);

        -- Saved regression models (one per score band)
        CREATE TABLE IF NOT EXISTS regression_models (
            score_band      INTEGER PRIMARY KEY,
            model_blob      BLOB NOT NULL,          -- pickle of fitted sklearn pipeline
            n_samples       INTEGER,
            r2_score        REAL,
            mae_usd         REAL,
            trained_at      TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
    print(f"Database initialised at {db_path.resolve()}")


if __name__ == "__main__":
    init_db()
