"""
scrape_prices.py — Mine Wikipedia for artwork sale prices and seed auction_data.db.

Runs indefinitely (Ctrl+C to stop). Safe to restart — skips already-priced works
and logs everything to scrape_prices.log.

Strategy:
  1. For every artist in the DB (highest score band first), fetch their Wikipedia
     page and find links to artwork articles.
  2. For each linked artwork page, parse the infobox and body text for sale price
     mentions (e.g. "$82.5 million", "sold for £44 million").
  3. If a price is found, upsert it into the artworks table (matching on title +
     artist_id, or inserting a new row if not present).
  4. Also re-scrapes artworks already in the DB that have no sale price yet.

Converts GBP → USD at a fixed 1.27 rate (good enough for historical estimates).
Converts EUR → USD at 1.09.
"""

import re
import time
import logging
import sqlite3
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH  = Path("auction_data.db")
LOG_PATH = Path("scrape_prices.log")

GBP_TO_USD = 1.27
EUR_TO_USD = 1.09

HEADERS = {"User-Agent": "ArtworkAuctionEstimator/2.0 (research; contact@example.com)"}

SLEEP_BETWEEN_PAGES   = 1.2   # seconds — be polite to Wikipedia
SLEEP_BETWEEN_ARTISTS = 2.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── DB ────────────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

# ── Price parsing ─────────────────────────────────────────────────────────────

# Matches things like:
#   $82.5 million   $104,000,000   £44 million   €30.5 million
#   USD 95.2 million   US$71.5 million
PRICE_PATTERNS = [
    # $X million / $X,000,000
    r'(?:US\$|\$|USD\s*)\s*([\d,]+(?:\.\d+)?)\s*million',
    r'(?:US\$|\$|USD\s*)\s*([\d,]+(?:\.\d+)?)\s*billion',
    r'(?:US\$|\$|USD\s*)\s*([\d]{1,3}(?:,\d{3})+)',
    # £X million
    r'£\s*([\d,]+(?:\.\d+)?)\s*million',
    r'£\s*([\d]{1,3}(?:,\d{3})+)',
    # €X million
    r'€\s*([\d,]+(?:\.\d+)?)\s*million',
    r'€\s*([\d]{1,3}(?:,\d{3})+)',
]

SALE_CONTEXT = re.compile(
    r'(?:sold|auction|christie|sotheby|phillips|bonham|hammer|fetch|realise|realize|purchase)',
    re.I
)

def extract_price_usd(text: str) -> float | None:
    """
    Scan text for sale price mentions near auction-related words.
    Returns USD float or None.
    """
    # Split into sentences for context checking
    sentences = re.split(r'[.!?\n]', text)

    candidates = []
    for sentence in sentences:
        if not SALE_CONTEXT.search(sentence):
            continue
        for i, pat in enumerate(PRICE_PATTERNS):
            for m in re.finditer(pat, sentence, re.I):
                raw = float(m.group(1).replace(',', ''))
                # Determine currency and scale
                snippet = sentence[max(0, m.start()-5):m.start()+2]
                if 'billion' in pat:
                    raw *= 1_000_000_000
                elif 'million' in pat:
                    raw *= 1_000_000
                # Convert currency
                if '£' in snippet or pat.startswith(r'£'):
                    raw *= GBP_TO_USD
                elif '€' in snippet or pat.startswith(r'€'):
                    raw *= EUR_TO_USD
                # Sanity check: between $10k and $5bn
                if 10_000 <= raw <= 5_000_000_000:
                    candidates.append(raw)

    if not candidates:
        return None
    # Return the largest plausible price found (most likely the headline sale)
    return max(candidates)


def extract_sale_year(text: str) -> int | None:
    """Look for a 4-digit year near sale/auction context."""
    sentences = re.split(r'[.!?\n]', text)
    for sentence in sentences:
        if not SALE_CONTEXT.search(sentence):
            continue
        years = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', sentence)
        if years:
            return int(years[-1])  # take last year mentioned (most likely sale year)
    return None

# ── Wikipedia fetching ────────────────────────────────────────────────────────

def fetch_page(url: str) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        log.warning(f"Fetch failed {url}: {e}")
        return None


def title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def get_artist_wikipedia_title(artist_name: str, qid: str) -> str | None:
    """Resolve artist's Wikipedia title via the MediaWiki API."""
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "titles": artist_name,
            "prop": "pageprops", "ppprop": "wikibase_item", "format": "json",
        }, headers=HEADERS, timeout=15)
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" not in page:
            return page["title"]
    except Exception:
        pass
    return None


def get_artwork_links_from_artist_page(soup: BeautifulSoup) -> list[str]:
    """
    Extract /wiki/ links from an artist's page that are likely artwork articles.
    Heuristic: links in the body that don't point to meta pages.
    """
    skip_prefixes = (
        "Wikipedia:", "File:", "Category:", "Help:", "Portal:",
        "Template:", "Talk:", "Special:", "User:",
    )
    links = []
    for a in soup.select("div#mw-content-text a[href^='/wiki/']"):
        href = a["href"]
        title = href[6:]  # strip /wiki/
        if any(title.startswith(p) for p in skip_prefixes):
            continue
        if "#" in title:
            continue
        links.append(title.replace("_", " "))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for l in links:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return unique


def scrape_artwork_page(title: str) -> dict | None:
    """
    Fetch a Wikipedia artwork page and extract price + metadata.
    Returns dict or None if no price found.
    """
    url = title_to_url(title)
    soup = fetch_page(url)
    if not soup:
        return None

    full_text = soup.get_text(" ")

    price = extract_price_usd(full_text)
    if price is None:
        return None  # Don't bother storing if no price

    year  = extract_sale_year(full_text)

    # Try to get decade from infobox
    decade = None
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if infobox:
        for row in infobox.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            label = th.get_text(strip=True).lower()
            value = td.get_text(" ", strip=True)
            if any(k in label for k in ["year", "date", "created", "painted"]):
                years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', value)
                if years:
                    decade = (int(years[0]) // 10) * 10

    return {
        "title": title,
        "url":   url,
        "price": price,
        "sale_year": year,
        "decade": decade,
    }

# ── DB upsert ─────────────────────────────────────────────────────────────────

def upsert_price(conn, artist_id: int, score_band: int, data: dict) -> bool:
    """
    Update existing artwork row or insert new one with the sale price.
    Returns True if a price was saved.
    """
    title = data["title"]
    price = data["price"]

    # Try to find existing row by title + artist
    row = conn.execute(
        "SELECT id, sale_price_usd FROM artworks WHERE artist_id=? AND title LIKE ? LIMIT 1",
        (artist_id, f"%{title[:40]}%")
    ).fetchone()

    if row:
        if row["sale_price_usd"] and row["sale_price_usd"] > 0:
            return False  # already has a price, skip
        conn.execute(
            "UPDATE artworks SET sale_price_usd=?, sale_year=? WHERE id=?",
            (price, data.get("sale_year"), row["id"])
        )
        log.info(f"  ✓ Updated price for '{title}': ${price:,.0f}")
    else:
        # Insert new artwork row
        conn.execute("""
            INSERT OR IGNORE INTO artworks
                (title, artist_id, score_band, decade, sale_price_usd, sale_year, qid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            title, artist_id, score_band,
            data.get("decade"), price, data.get("sale_year"),
            f"WP_{title[:30].replace(' ','_')}",
        ))
        log.info(f"  ✓ Inserted new artwork '{title}': ${price:,.0f}")

    conn.commit()
    return True

# ── Existing unpriced artworks ────────────────────────────────────────────────

def scrape_unpriced_artworks(conn):
    """Re-scrape artworks already in DB that have no sale price."""
    rows = conn.execute("""
        SELECT aw.id, aw.title, aw.artist_id, aw.score_band
        FROM artworks aw
        WHERE aw.sale_price_usd IS NULL OR aw.sale_price_usd = 0
        ORDER BY aw.score_band DESC
    """).fetchall()

    log.info(f"Scraping {len(rows)} unpriced artworks already in DB…")
    found = 0
    for row in rows:
        title = row["title"]
        if not title or title.startswith("Q"):
            continue
        url = title_to_url(title)
        soup = fetch_page(url)
        if not soup:
            time.sleep(SLEEP_BETWEEN_PAGES)
            continue
        text  = soup.get_text(" ")
        price = extract_price_usd(text)
        if price:
            year = extract_sale_year(text)
            conn.execute(
                "UPDATE artworks SET sale_price_usd=?, sale_year=? WHERE id=?",
                (price, year, row["id"])
            )
            conn.commit()
            log.info(f"  ✓ Found price for existing '{title}': ${price:,.0f}")
            found += 1
        time.sleep(SLEEP_BETWEEN_PAGES)

    log.info(f"Found prices for {found} previously unpriced artworks.")

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Wikipedia price scraper starting — Ctrl+C to stop")
    log.info("=" * 60)

    conn = get_conn()

    # Phase 1: scrape unpriced artworks already in DB
    scrape_unpriced_artworks(conn)

    # Phase 2: walk every artist page, find artwork links, scrape each
    artists = conn.execute(
        "SELECT id, qid, name, score_band, score FROM artists ORDER BY score DESC"
    ).fetchall()

    total_found = 0
    for artist in artists:
        artist_name = artist["name"]
        artist_id   = artist["id"]
        score_band  = artist["score_band"]

        log.info(f"Artist: {artist_name} (band {score_band})")

        wiki_title = get_artist_wikipedia_title(artist_name, artist["qid"])
        if not wiki_title:
            log.info(f"  No Wikipedia page found, skipping")
            time.sleep(SLEEP_BETWEEN_ARTISTS)
            continue

        soup = fetch_page(title_to_url(wiki_title))
        if not soup:
            time.sleep(SLEEP_BETWEEN_ARTISTS)
            continue

        artwork_titles = get_artwork_links_from_artist_page(soup)
        log.info(f"  Found {len(artwork_titles)} linked pages to check")

        artist_found = 0
        for title in artwork_titles:
            try:
                data = scrape_artwork_page(title)
                if data:
                    saved = upsert_price(conn, artist_id, score_band, data)
                    if saved:
                        artist_found += 1
                        total_found  += 1
                time.sleep(SLEEP_BETWEEN_PAGES)
            except KeyboardInterrupt:
                log.info(f"\nStopped by user. Total prices found this session: {total_found}")
                # Retrain before exiting
                log.info("Running --train on exit…")
                import subprocess
                subprocess.run(["python3", "auction_estimator.py", "--train"])
                return
            except Exception as e:
                log.warning(f"  Error on '{title}': {e}")
                continue

        log.info(f"  → {artist_found} prices found for {artist_name}")
        time.sleep(SLEEP_BETWEEN_ARTISTS)

    log.info(f"\nAll artists processed. Total prices found: {total_found}")
    log.info("Running final --train…")
    import subprocess
    subprocess.run(["python3", "auction_estimator.py", "--train"])
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nStopped by user.")
        import subprocess
        log.info("Running --train on exit…")
        subprocess.run(["python3", "auction_estimator.py", "--train"])
