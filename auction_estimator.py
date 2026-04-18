r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ARTWORK AUCTION PRICE ESTIMATOR                         ║
║                                                                              ║
║  Paste a Wikipedia URL for any artwork → get an estimated auction price.     ║
║                                                                              ║
║  HOW IT WORKS                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. Parses the Wikipedia page to extract: artist, medium, dimensions,        ║
║     creation date.                                                           ║
║  2. Looks up the artist in the local SQLite DB (built by the data pipeline). ║
║     If not found, scores them live from Wikidata + Wikipedia APIs.           ║
║  3. Loads the regression model for that artist's score band (1–9).           ║
║  4. Predicts log(price), back-transforms to USD, returns estimate + CI.      ║
║                                                                              ║
║  FIRST-TIME SETUP (run once, takes 2–6 hours for full 100k artist dataset)   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  pip install requests tqdm scikit-learn pandas numpy beautifulsoup4 lxml     ║
║                                                                              ║
║  python auction_estimator.py --setup            # fetch artists + artworks   ║
║  python auction_estimator.py --setup --limit 500  # quick test (5 min)       ║
║  python auction_estimator.py --train            # fit regression models      ║
║                                                                              ║
║  DAILY USE                                                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  python auction_estimator.py \                                               ║
║      --url "https://en.wikipedia.org/wiki/The_Starry_Night"                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import io
import json
import math
import pickle
import re
import sqlite3
import time
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DB_PATH         = Path("auction_data.db")
ARTISTS_FILE    = Path("artists_scored.json")
CHECKPOINT_FILE = Path("checkpoint.json")
LOG_FILE        = Path("auction_estimator.log")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKI_PAGEVIEWS  = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
WIKI_API        = "https://en.wikipedia.org/w/api.php"

HEADERS = {"User-Agent": "ArtworkAuctionEstimator/2.0 (research; contact@example.com)"}

# Score weights for artist fame (must sum to 1.0)
W_PAGEVIEWS   = 0.38
W_WORKS       = 0.37
W_PRICE       = 0.13
W_WIKI_LENGTH = 0.12

# Regression weights
W_SEARCH_VOL   = 0.38   # Wikipedia pageviews proxy
W_NUM_WORKS    = 0.37
W_AVG_PRICE    = 0.13
W_WIKI_LEN     = 0.12

MIN_SAMPLES_TO_TRAIN = 30

NUMERIC_FEATURES = ["artist_score", "decade", "width_cm", "height_cm", "depth_cm"]
BINARY_FEATURES  = ["is_3d"]
CAT_FEATURES     = ["medium"]
ALL_FEATURES     = NUMERIC_FEATURES + BINARY_FEATURES + CAT_FEATURES

THREED_KEYWORDS = {
    "bronze", "marble", "stone", "terracotta", "clay", "plaster",
    "wood", "cast", "sculpture", "ceramic", "granite", "alabaster",
    "stainless steel", "iron", "copper", "resin", "fiberglass",
}

MEDIUM_MAP = {
    "oil on canvas": "oil on canvas", "oil on panel": "oil on panel",
    "oil on board": "oil on panel",   "oil on wood": "oil on panel",
    "oil on paper": "oil on paper",   "watercolor": "watercolor",
    "watercolour": "watercolor",      "gouache": "gouache",
    "ink": "ink",                     "pencil": "pencil",
    "charcoal": "charcoal",           "pastel": "pastel",
    "chalk": "chalk",                 "engraving": "engraving",
    "etching": "etching",             "lithograph": "lithograph",
    "screenprint": "screenprint",     "woodcut": "woodcut",
    "print": "print",                 "acrylic": "acrylic",
    "tempera": "tempera",             "fresco": "fresco",
    "mixed media": "mixed media",     "collage": "collage",
    "photograph": "photograph",       "bronze": "bronze",
    "marble": "marble",               "terracotta": "terracotta",
    "clay": "clay",                   "plaster": "plaster",
    "ceramic": "ceramic",             "stone": "stone",
    "wood": "carved wood",            "iron": "iron",
    "steel": "steel",                 "copper": "copper",
    "resin": "resin",
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH):
    conn = get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS artists (
            id          INTEGER PRIMARY KEY,
            qid         TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            score       REAL NOT NULL,
            score_band  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_artists_band  ON artists(score_band);
        CREATE INDEX IF NOT EXISTS idx_artists_score ON artists(score);
        CREATE INDEX IF NOT EXISTS idx_artists_qid   ON artists(qid);

        CREATE TABLE IF NOT EXISTS mediums (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artworks (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            qid            TEXT UNIQUE,
            title          TEXT,
            artist_id      INTEGER NOT NULL REFERENCES artists(id),
            score_band     INTEGER NOT NULL,
            decade         INTEGER,
            is_3d          INTEGER NOT NULL DEFAULT 0,
            width_cm       REAL,
            height_cm      REAL,
            depth_cm       REAL,
            medium_id      INTEGER REFERENCES mediums(id),
            medium_raw     TEXT,
            sale_price_usd REAL,
            sale_year      INTEGER,
            created_at     TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_artworks_band   ON artworks(score_band);
        CREATE INDEX IF NOT EXISTS idx_artworks_artist ON artworks(artist_id);
        CREATE INDEX IF NOT EXISTS idx_artworks_price  ON artworks(sale_price_usd);

        CREATE TABLE IF NOT EXISTS regression_models (
            score_band INTEGER PRIMARY KEY,
            model_blob BLOB NOT NULL,
            n_samples  INTEGER,
            r2_score   REAL,
            mae_usd    REAL,
            trained_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_medium(raw: str) -> str:
    if not raw:
        return "unknown"
    lower = raw.lower()
    for key, label in MEDIUM_MAP.items():
        if key in lower:
            return label
    return raw.lower().strip()[:80]


def is_3d_medium(medium: str) -> bool:
    return any(kw in medium.lower() for kw in THREED_KEYWORDS)


_medium_cache: dict[str, int] = {}

def get_or_create_medium(conn: sqlite3.Connection, label: str) -> int:
    if label in _medium_cache:
        return _medium_cache[label]
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO mediums (label) VALUES (?)", (label,))
    conn.commit()
    cur.execute("SELECT id FROM mediums WHERE label = ?", (label,))
    mid = cur.fetchone()[0]
    _medium_cache[label] = mid
    return mid


# ═══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA / WIKIDATA API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_monthly_pageviews(wiki_title: str) -> int:
    if not wiki_title:
        return 0
    encoded = requests.utils.quote(wiki_title.replace(" ", "_"), safe="")
    url = f"{WIKI_PAGEVIEWS}/en.wikipedia/all-access/all-agents/{encoded}/monthly/20230101/20231201"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 404:
            return 0
        r.raise_for_status()
        items = r.json().get("items", [])
        return int(sum(x["views"] for x in items) / len(items)) if items else 0
    except Exception:
        return 0


def get_wiki_length(wiki_title: str) -> int:
    if not wiki_title:
        return 0
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "titles": wiki_title,
            "prop": "revisions", "rvprop": "size", "format": "json",
        }, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        revisions = page.get("revisions", [])
        return revisions[0].get("size", 0) if revisions else 0
    except Exception:
        return 0


def get_avg_price_for_artist(qid: str) -> float:
    query = f"""
    SELECT ?price WHERE {{
      ?work wdt:P170 wd:{qid} .
      ?work wdt:P1028 ?price .
    }} LIMIT 50
    """
    try:
        r = requests.get(WIKIDATA_SPARQL,
                         params={"query": query, "format": "json"},
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        items = r.json()["results"]["bindings"]
        prices = [float(i["price"]["value"]) for i in items if "price" in i]
        return sum(prices) / len(prices) if prices else 0.0
    except Exception:
        return 0.0


def sparql(query: str, timeout: int = 60) -> list[dict]:
    try:
        r = requests.get(WIKIDATA_SPARQL,
                         params={"query": query, "format": "json"},
                         headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    except Exception as e:
        log.warning(f"SPARQL error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH & SCORE ARTISTS  (run once)
# ═══════════════════════════════════════════════════════════════════════════════

ARTIST_SPARQL = """
SELECT DISTINCT ?artist ?artistLabel ?wikiarticle ?works WHERE {
  ?artist wdt:P31 wd:Q5 .
  { ?artist wdt:P106 wd:Q1028181 . }
  UNION { ?artist wdt:P106 wd:Q1281618 . }
  UNION { ?artist wdt:P106 wd:Q329439 . }
  UNION { ?artist wdt:P106 wd:Q3391743 . }
  UNION { ?artist wdt:P106 wd:Q1925963 . }
  OPTIONAL {
    ?wikiarticle schema:about ?artist ;
                 schema:isPartOf <https://en.wikipedia.org/> .
  }
  OPTIONAL {
    SELECT ?artist (COUNT(?work) AS ?works) WHERE {
      ?work wdt:P170 ?artist .
    } GROUP BY ?artist
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
ORDER BY DESC(?works)
LIMIT {LIMIT}
"""


def log_normalize(values: list[float]) -> list[float]:
    logged = [math.log1p(v) for v in values]
    max_val = max(logged) if logged else 1
    return [0.0] * len(values) if max_val == 0 else [v / max_val for v in logged]


def fetch_and_score_artists(limit: int = 100000, skip_prices: bool = False):
    """Fetch artists from Wikidata, enrich with Wikipedia signals, score 1–10."""
    # Resume from checkpoint if available
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        artists, start_idx = data["artists"], data["idx"]
        log.info(f"Resuming from checkpoint at index {start_idx:,}")
    else:
        log.info(f"Querying Wikidata for up to {limit:,} artists…")
        query = ARTIST_SPARQL.replace("{LIMIT}", str(limit))
        bindings = sparql(query, timeout=300)
        log.info(f"  → {len(bindings):,} artists returned")
        artists = []
        for i, b in enumerate(bindings):
            qid = b["artist"]["value"].split("/")[-1]
            name = b.get("artistLabel", {}).get("value", "")
            wiki_title = None
            if "wikiarticle" in b:
                wiki_title = b["wikiarticle"]["value"] \
                    .replace("https://en.wikipedia.org/wiki/", "") \
                    .replace("_", " ")
            artists.append({
                "id": i + 1, "qid": qid, "name": name,
                "wiki_title": wiki_title,
                "raw_works": int(b["works"]["value"]) if "works" in b else 0,
                "raw_pageviews": 0, "raw_wiki_length": 0,
                "raw_avg_price_usd": 0, "score": 0.0,
            })
        start_idx = 0

    total = len(artists)
    log.info(f"Enriching {total:,} artists…")
    for i in tqdm(range(start_idx, total), desc="Enriching"):
        a = artists[i]
        if a["wiki_title"]:
            a["raw_pageviews"]   = get_monthly_pageviews(a["wiki_title"])
            a["raw_wiki_length"] = get_wiki_length(a["wiki_title"])
            time.sleep(0.05)
        if not skip_prices:
            a["raw_avg_price_usd"] = get_avg_price_for_artist(a["qid"])
            time.sleep(0.1)
        if (i + 1) % 500 == 0:
            CHECKPOINT_FILE.write_text(
                json.dumps({"idx": i + 1, "artists": artists}, indent=2))

    # Compute composite scores
    pv = log_normalize([a["raw_pageviews"]    for a in artists])
    wk = log_normalize([a["raw_works"]        for a in artists])
    pr = log_normalize([a["raw_avg_price_usd"] for a in artists])
    wl = log_normalize([a["raw_wiki_length"]  for a in artists])
    for i, a in enumerate(artists):
        c = W_PAGEVIEWS * pv[i] + W_WORKS * wk[i] + W_PRICE * pr[i] + W_WIKI_LENGTH * wl[i]
        a["score"] = round(1.0 + c * 9.0, 1)
    artists.sort(key=lambda x: x["score"], reverse=True)
    for i, a in enumerate(artists):
        a["id"] = i + 1

    ARTISTS_FILE.write_text(json.dumps(artists, indent=2, ensure_ascii=False))
    log.info(f"✓ Saved {total:,} artists to {ARTISTS_FILE}")
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    return artists


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FETCH ARTWORKS INTO SQLITE  (run once)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_year(s: str) -> Optional[int]:
    try:
        return int(s[:4])
    except Exception:
        return None


def fetch_artworks_for_band(band: int, limit: Optional[int] = None):
    """Pull artwork records from Wikidata for one score band, store in SQLite."""
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, qid, name, score, score_band FROM artists WHERE score_band = ?",
        (band,)
    ).fetchall()
    if not rows:
        log.info(f"Band {band}: no artists, skipping")
        conn.close()
        return
    if limit:
        rows = rows[:limit]

    artist_map = {r["qid"]: dict(r) for r in rows}
    qids = list(artist_map.keys())
    log.info(f"Band {band}: fetching artworks for {len(qids):,} artists")

    BATCH = 50
    total_inserted = 0
    for i in tqdm(range(0, len(qids), BATCH), desc=f"Band {band}"):
        batch = qids[i:i + BATCH]
        values = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT DISTINCT ?work ?workLabel ?artist ?inception
            ?width ?height ?depth ?material ?materialLabel ?price
        WHERE {{
            VALUES ?artist {{ {values} }}
            ?work wdt:P170 ?artist .
            OPTIONAL {{ ?work wdt:P571 ?inception . }}
            OPTIONAL {{ ?work wdt:P2049 ?width . }}
            OPTIONAL {{ ?work wdt:P2048 ?height . }}
            OPTIONAL {{ ?work wdt:P2610 ?depth . }}
            OPTIONAL {{ ?work wdt:P186 ?material . }}
            OPTIONAL {{ ?work wdt:P1028 ?price . }}
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
            }}
        }} LIMIT 5000
        """
        bindings = sparql(query)
        for b in bindings:
            try:
                work_qid   = b["work"]["value"].split("/")[-1]
                artist_qid = b["artist"]["value"].split("/")[-1]
                artist_row = artist_map.get(artist_qid)
                if not artist_row:
                    continue

                year   = parse_year(b.get("inception", {}).get("value", ""))
                decade = ((year // 10) * 10) if year else None

                def dim(k):
                    v = b.get(k, {}).get("value")
                    return float(v) if v else None

                width_cm  = dim("width")
                height_cm = dim("height")
                depth_cm  = dim("depth")
                medium_raw  = b.get("materialLabel", {}).get("value", "")
                medium_norm = normalise_medium(medium_raw)
                three_d = 1 if (is_3d_medium(medium_norm) or depth_cm is not None) else 0
                if three_d == 0:
                    depth_cm = None
                medium_id = get_or_create_medium(conn, medium_norm)
                price_raw = b.get("price", {}).get("value")
                price_usd = float(price_raw) if price_raw else None
                title = b.get("workLabel", {}).get("value", "")

                conn.execute("""
                    INSERT OR IGNORE INTO artworks
                        (qid, title, artist_id, score_band, decade,
                         is_3d, width_cm, height_cm, depth_cm,
                         medium_id, medium_raw, sale_price_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    work_qid, title, artist_row["id"], artist_row["score_band"],
                    decade, three_d, width_cm, height_cm, depth_cm,
                    medium_id, medium_raw or None, price_usd,
                ))
                total_inserted += 1
            except Exception as e:
                log.debug(f"Row error: {e}")
        conn.commit()
        time.sleep(0.5)

    log.info(f"Band {band}: inserted {total_inserted:,} artworks")
    conn.close()


def load_artists_into_db():
    if not ARTISTS_FILE.exists():
        raise FileNotFoundError(f"{ARTISTS_FILE} not found. Run --setup first.")
    artists = json.loads(ARTISTS_FILE.read_text())
    conn = get_conn()
    inserted = 0
    for a in artists:
        band = max(1, min(9, int(a["score"])))
        try:
            conn.execute(
                "INSERT OR IGNORE INTO artists (id, qid, name, score, score_band) VALUES (?,?,?,?,?)",
                (a["id"], a["qid"], a["name"], a["score"], band)
            )
            inserted += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    log.info(f"Loaded {inserted:,} artists into DB")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN REGRESSION MODELS  (run once, re-run to refresh)
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", cat_pipe,      CAT_FEATURES),
    ])
    return Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))])


def train_all_models():
    init_db()
    for band in range(1, 10):
        conn = get_conn()
        df = pd.read_sql_query("""
            SELECT ar.score AS artist_score, aw.decade,
                   aw.width_cm, aw.height_cm, aw.depth_cm, aw.is_3d,
                   m.label AS medium, aw.sale_price_usd
            FROM artworks aw
            JOIN artists ar  ON ar.id = aw.artist_id
            LEFT JOIN mediums m ON m.id = aw.medium_id
            WHERE aw.score_band = ? AND aw.sale_price_usd > 0
        """, conn, params=(band,))
        conn.close()

        log.info(f"Band {band}: {len(df):,} labelled samples")
        if len(df) < MIN_SAMPLES_TO_TRAIN:
            log.warning(f"Band {band}: skipping — need {MIN_SAMPLES_TO_TRAIN} samples")
            continue

        df["depth_cm"] = df["depth_cm"].fillna(0)
        X = df[ALL_FEATURES].copy()
        y = np.log1p(df["sale_price_usd"].values)
        pipeline = build_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_r2 = cross_val_score(
                pipeline, X, y, cv=min(5, len(df) // 10 + 1), scoring="r2"
            ).mean()
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r2  = r2_score(y, y_pred)
        mae = mean_absolute_error(np.expm1(y), np.expm1(np.clip(y_pred, 0, None)))
        log.info(f"Band {band}: R²={r2:.3f}  CV-R²={cv_r2:.3f}  MAE=${mae:,.0f}")

        buf = io.BytesIO()
        pickle.dump(pipeline, buf)
        conn = get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO regression_models
                (score_band, model_blob, n_samples, r2_score, mae_usd)
            VALUES (?, ?, ?, ?, ?)
        """, (band, buf.getvalue(), len(df), r2, mae))
        conn.commit()
        conn.close()
    log.info("✓ All models trained")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — WIKIPEDIA PAGE PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_wikipedia_artwork(url: str) -> dict:
    """
    Scrape an artwork's Wikipedia page to extract:
      title, artist_name, artist_wiki_title, decade,
      medium, width_cm, height_cm, depth_cm, wikidata_qid
    """
    log.info(f"Fetching Wikipedia page: {url}")
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    result = {
        "title": "", "artist_name": "", "artist_wiki_title": None,
        "decade": None, "medium": "unknown",
        "width_cm": None, "height_cm": None, "depth_cm": None,
        "wikidata_qid": None,
    }

    # Title
    h1 = soup.find("h1", id="firstHeading")
    if h1:
        result["title"] = h1.get_text(strip=True)

    # Wikidata QID from sidebar link
    wd_link = soup.find("a", href=re.compile(r"wikidata\.org/wiki/(Q\d+)"))
    if wd_link:
        m = re.search(r"(Q\d+)", wd_link["href"])
        if m:
            result["wikidata_qid"] = m.group(1)

    # Infobox
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if not infobox:
        infobox = soup.find("table", class_="wikitable")

    if infobox:
        rows = infobox.find_all("tr")
        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            label = th.get_text(strip=True).lower()
            value = td.get_text(" ", strip=True)

            # Artist
            if any(k in label for k in ["artist", "painter", "sculptor", "creator", "author"]):
                result["artist_name"] = value.split("\n")[0].strip()
                a_tag = td.find("a")
                if a_tag and a_tag.get("href", "").startswith("/wiki/"):
                    result["artist_wiki_title"] = (
                        a_tag["href"].replace("/wiki/", "").replace("_", " ")
                    )

            # Date / year
            elif any(k in label for k in ["year", "date", "created", "painted", "completed"]):
                years = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", value)
                if years:
                    year = int(years[0])
                    result["decade"] = (year // 10) * 10

            # Medium
            elif any(k in label for k in ["medium", "material", "technique", "media"]):
                result["medium"] = normalise_medium(value)

            # Dimensions
            elif any(k in label for k in ["dimension", "size", "measurement"]):
                _parse_dimensions(value, result)

    # Fallback: try to find dimensions in page text
    if result["width_cm"] is None:
        text = soup.get_text(" ")
        _parse_dimensions(text, result)

    return result


def _parse_dimensions(text: str, result: dict):
    """Extract width/height/depth from a free text string (cm or in)."""
    # Pattern: 81.3 cm × 65.2 cm  or  81.3 × 65.2 cm  or  32 in × 25.6 in
    pat_cm = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:cm)?\s*[×xX]\s*(\d+(?:\.\d+)?)\s*(?:cm)?(?:\s*[×xX]\s*(\d+(?:\.\d+)?)\s*(?:cm)?)?",
        text
    )
    pat_in = re.findall(
        r"(\d+(?:\.\d+)?)\s*in(?:ch(?:es)?)?\s*[×xX]\s*(\d+(?:\.\d+)?)\s*in",
        text
    )

    if pat_cm and result["width_cm"] is None:
        g = pat_cm[0]
        result["width_cm"]  = float(g[0])
        result["height_cm"] = float(g[1])
        if g[2]:
            result["depth_cm"] = float(g[2])
    elif pat_in and result["width_cm"] is None:
        result["width_cm"]  = float(pat_in[0][0]) * 2.54
        result["height_cm"] = float(pat_in[0][1]) * 2.54


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ARTIST LOOKUP / LIVE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_artist_in_db(artist_name: str, wiki_title: Optional[str]) -> Optional[dict]:
    """Check local DB for artist by wiki title or name."""
    conn = get_conn()
    row = None
    if wiki_title:
        # Try to match via Wikidata QID
        encoded = wiki_title.replace(" ", "_")
        wikidata_url = f"https://en.wikipedia.org/wiki/{encoded}"
        wd_query = f"""
        SELECT ?artist WHERE {{
          ?article schema:about ?artist ;
                   schema:isPartOf <https://en.wikipedia.org/> ;
                   schema:name "{wiki_title}"@en .
        }} LIMIT 1
        """
        bindings = sparql(wd_query, timeout=15)
        if bindings:
            qid = bindings[0]["artist"]["value"].split("/")[-1]
            row = conn.execute(
                "SELECT * FROM artists WHERE qid = ?", (qid,)
            ).fetchone()

    if row is None and artist_name:
        row = conn.execute(
            "SELECT * FROM artists WHERE name LIKE ? LIMIT 1",
            (f"%{artist_name}%",)
        ).fetchone()

    conn.close()
    return dict(row) if row else None


def score_artist_live(artist_name: str, wiki_title: Optional[str],
                      wikidata_qid: Optional[str] = None) -> dict:
    """Score an artist on-the-fly if not in the local DB."""
    log.info(f"Artist '{artist_name}' not in DB — scoring live…")

    # Try to find QID from Wikipedia title
    if not wikidata_qid and wiki_title:
        try:
            r = requests.get(WIKI_API, params={
                "action": "query", "titles": wiki_title,
                "prop": "pageprops", "ppprop": "wikibase_item", "format": "json",
            }, headers=HEADERS, timeout=15)
            pages = r.json()["query"]["pages"]
            page = next(iter(pages.values()))
            wikidata_qid = page.get("pageprops", {}).get("wikibase_item")
        except Exception:
            pass

    pageviews  = get_monthly_pageviews(wiki_title or artist_name)
    wiki_len   = get_wiki_length(wiki_title or artist_name)
    avg_price  = get_avg_price_for_artist(wikidata_qid) if wikidata_qid else 0.0

    # Works count from Wikidata
    works_count = 0
    if wikidata_qid:
        q = f"SELECT (COUNT(?w) AS ?c) WHERE {{ ?w wdt:P170 wd:{wikidata_qid} . }}"
        b = sparql(q, timeout=20)
        if b:
            try:
                works_count = int(b[0]["c"]["value"])
            except Exception:
                pass

    # Score using log-normalisation with reasonable reference maxima
    # (approximated from the full dataset distribution)
    REF_PAGEVIEWS = 500_000
    REF_WORKS     = 1_000
    REF_PRICE     = 100_000_000
    REF_WIKILEN   = 200_000

    def lnorm(v, ref):
        return math.log1p(v) / math.log1p(ref) if ref > 0 else 0

    c = (
        W_PAGEVIEWS   * lnorm(pageviews, REF_PAGEVIEWS) +
        W_WORKS       * lnorm(works_count, REF_WORKS) +
        W_PRICE       * lnorm(avg_price, REF_PRICE) +
        W_WIKI_LENGTH * lnorm(wiki_len, REF_WIKILEN)
    )
    score = round(max(1.0, min(10.0, 1.0 + c * 9.0)), 1)
    log.info(f"  Live score for '{artist_name}': {score}")

    return {
        "name": artist_name,
        "qid":  wikidata_qid or "",
        "score": score,
        "score_band": max(1, min(9, int(score))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL LOADER & PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

_model_cache: dict[int, tuple] = {}


def load_model(band: int) -> Optional[tuple]:
    if band in _model_cache:
        return _model_cache[band]
    conn = get_conn()
    row = conn.execute(
        "SELECT model_blob, n_samples, r2_score, mae_usd FROM regression_models WHERE score_band = ?",
        (band,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    pipeline = pickle.loads(row["model_blob"])
    entry = (pipeline, dict(row))
    _model_cache[band] = entry
    return entry


def find_nearest_model_band(target: int) -> Optional[int]:
    conn = get_conn()
    trained = [r[0] for r in conn.execute(
        "SELECT score_band FROM regression_models ORDER BY score_band"
    ).fetchall()]
    conn.close()
    return min(trained, key=lambda b: abs(b - target)) if trained else None


def predict_price(
    artist_score: float,
    decade: Optional[int],
    medium: str,
    width_cm: Optional[float],
    height_cm: Optional[float],
    depth_cm: Optional[float],
) -> dict:
    band = max(1, min(9, int(artist_score)))
    entry = load_model(band)
    model_band = band

    if entry is None:
        model_band = find_nearest_model_band(band)
        if model_band is None:
            raise RuntimeError(
                "No trained models found. Run:  python auction_estimator.py --train"
            )
        entry = load_model(model_band)
        log.warning(f"No model for band {band}, using band {model_band}")

    pipeline, meta = entry
    three_d  = 1 if (is_3d_medium(medium) or depth_cm is not None) else 0
    depth_val = depth_cm if (three_d and depth_cm is not None) else 0.0

    row = pd.DataFrame([{
        "artist_score": artist_score,
        "decade":       decade,
        "width_cm":     width_cm,
        "height_cm":    height_cm,
        "depth_cm":     depth_val,
        "is_3d":        three_d,
        "medium":       medium.lower().strip() if medium else "unknown",
    }])

    log_pred = float(pipeline.predict(row)[0])
    log_pred = max(log_pred, 0)
    price    = float(np.expm1(log_pred))

    # Apply artist score multiplier — exponential boost for high-fame artists
    # Score 10 -> 6x, Score 9 -> 3x, Score 8 -> 1.5x, Score 5 -> 1x, below 5 -> discount
    score_multiplier = math.exp((artist_score - 5.0) * 0.45)
    price = price * score_multiplier
    mae      = meta["mae_usd"] or 0

    return {
        "estimated_price_usd":  round(price, 2),
        "confidence_interval":  (round(max(0, price - mae), 2), round(price + mae, 2)),
        "score_band":           band,
        "model_band":           model_band,
        "n_training_samples":   meta["n_samples"],
        "model_r2":             round(meta["r2_score"] or 0, 3),
        "model_mae_usd":        round(mae, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def check_museum_class(title: str) -> Optional[dict]:
    """Check if an artwork is museum-class by title match. Returns row or None."""
    if not title:
        return None
    conn = get_conn()
    row = conn.execute("""
        SELECT aw.title, aw.insurance_value_usd, aw.museum_name,
               ar.name as artist_name, aw.decade, aw.medium_raw
        FROM artworks aw
        JOIN artists ar ON ar.id = aw.artist_id
        WHERE aw.museum_class = 1
          AND lower(aw.title) LIKE lower(?)
        LIMIT 1
    """, (f"%{title[:30]}%",)).fetchone()
    conn.close()
    return dict(row) if row else None


def estimate_from_url(url: str) -> dict:
    """Full pipeline: Wikipedia URL → auction price estimate."""

    # 1. Parse Wikipedia page
    artwork = parse_wikipedia_artwork(url)
    log.info(f"Parsed artwork: '{artwork['title']}' by '{artwork['artist_name']}'")

    # 2. Check museum class before doing anything else
    museum_row = check_museum_class(artwork["title"])
    if museum_row and museum_row["insurance_value_usd"]:
        ins = museum_row["insurance_value_usd"]
        log.info(f"Museum-class work detected: '{artwork['title']}' — returning insurance valuation")
        return {
            "artwork": {
                "title":    artwork["title"],
                "url":      url,
                "artist":   museum_row["artist_name"],
                "decade":   artwork["decade"],
                "medium":   artwork["medium"],
                "width_cm": artwork["width_cm"],
                "height_cm":artwork["height_cm"],
                "depth_cm": artwork["depth_cm"],
                "is_3d":    False,
            },
            "artist_score":        10.0,
            "estimated_price_usd": ins,
            "confidence_interval": [ins * 0.7, ins * 1.5],
            "museum_class":        True,
            "museum_name":         museum_row["museum_name"],
            "model_info": {
                "score_band":         9,
                "model_band":         9,
                "n_training_samples": None,
                "model_r2":           None,
                "model_mae_usd":      None,
            }
        }

    # 3. Look up or score artist
    artist = lookup_artist_in_db(artwork["artist_name"], artwork["artist_wiki_title"])
    if artist is None:
        artist = score_artist_live(
            artwork["artist_name"],
            artwork["artist_wiki_title"],
            artwork.get("wikidata_qid"),
        )

    # 4. Predict
    result = predict_price(
        artist_score = artist["score"],
        decade       = artwork["decade"],
        medium       = artwork["medium"],
        width_cm     = artwork["width_cm"],
        height_cm    = artwork["height_cm"],
        depth_cm     = artwork["depth_cm"],
    )

    # 5. Assemble full output
    return {
        "artwork": {
            "title":    artwork["title"],
            "url":      url,
            "artist":   artist["name"],
            "decade":   artwork["decade"],
            "medium":   artwork["medium"],
            "width_cm": artwork["width_cm"],
            "height_cm":artwork["height_cm"],
            "depth_cm": artwork["depth_cm"],
            "is_3d":    bool(result.get("score_band") and is_3d_medium(artwork["medium"])),
        },
        "museum_class":         False,
        "museum_name":          None,
        "artist_score":         artist["score"],
        "estimated_price_usd":  result["estimated_price_usd"],
        "confidence_interval":  result["confidence_interval"],
        "model_info": {
            "score_band":        result["score_band"],
            "model_band":        result["model_band"],
            "n_training_samples":result["n_training_samples"],
            "model_r2":          result["model_r2"],
            "model_mae_usd":     result["model_mae_usd"],
        }
    }


def print_result(out: dict):
    lo, hi = out["confidence_interval"]
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           AUCTION PRICE ESTIMATE                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Artwork  : {out['artwork']['title']:<44} ║")
    print(f"║  Artist   : {out['artwork']['artist']:<44} ║")
    print(f"║  Medium   : {out['artwork']['medium']:<44} ║")
    decade_str = str(out['artwork']['decade']) + "s" if out['artwork']['decade'] else "unknown"
    print(f"║  Decade   : {decade_str:<44} ║")
    dims = ""
    if out['artwork']['width_cm']:
        dims = f"{out['artwork']['width_cm']}w × {out['artwork']['height_cm']}h"
        if out['artwork']['depth_cm']:
            dims += f" × {out['artwork']['depth_cm']}d"
        dims += " cm"
    print(f"║  Dims     : {dims:<44} ║")
    print(f"║  Artist score : {out['artist_score']:<40} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    price_str = f"${out['estimated_price_usd']:>15,.2f}"
    print(f"║  ESTIMATE : {price_str:<44} ║")
    range_str = f"${lo:,.0f}  –  ${hi:,.0f}"
    print(f"║  ±1 MAE   : {range_str:<44} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    mi = out['model_info']
    print(f"║  Band {mi['score_band']} model  │  n={mi['n_training_samples']:,}  │  R²={mi['model_r2']:.3f}  │  MAE=${mi['model_mae_usd']:,.0f}  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Artwork Auction Price Estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup (small test, ~5 min):
  python auction_estimator.py --setup --limit 500

  # Full setup (~4-6 hours):
  python auction_estimator.py --setup

  # Train models after setup:
  python auction_estimator.py --train

  # Estimate from a Wikipedia URL:
  python auction_estimator.py --url "https://en.wikipedia.org/wiki/The_Starry_Night"
  python auction_estimator.py --url "https://en.wikipedia.org/wiki/David_(Michelangelo)"
        """
    )
    parser.add_argument("--url",    type=str,  help="Wikipedia URL of the artwork to estimate")
    parser.add_argument("--setup",  action="store_true",
                        help="Run full data pipeline (fetch artists + artworks)")
    parser.add_argument("--train",  action="store_true",
                        help="Fit regression models from local DB")
    parser.add_argument("--limit",  type=int,  default=None,
                        help="Limit artists per band during --setup (for testing)")
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip price enrichment during --setup (faster)")
    parser.add_argument("--json",   action="store_true",
                        help="Output result as raw JSON instead of formatted table")
    args = parser.parse_args()

    init_db()

    if args.setup:
        log.info("=== SETUP: Fetching & scoring artists ===")
        fetch_and_score_artists(
            limit=args.limit or 100000,
            skip_prices=args.skip_prices,
        )
        load_artists_into_db()
        log.info("=== SETUP: Fetching artworks per band ===")
        for band in range(1, 10):
            fetch_artworks_for_band(band, limit=args.limit)
        log.info("✓ Setup complete. Now run:  python auction_estimator.py --train")
        return

    if args.train:
        log.info("=== TRAINING regression models ===")
        train_all_models()
        return

    if args.url:
        out = estimate_from_url(args.url)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print_result(out)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
