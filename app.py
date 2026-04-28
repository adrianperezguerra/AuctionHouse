"""
app.py — AuctionHouse backend for Render.com
"""

import os
import json
import sqlite3
import secrets
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import wraps

import bcrypt
import jwt
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from auction_estimator import estimate_from_url, init_db as init_auction_db
    ESTIMATOR_AVAILABLE = True
except ImportError:
    ESTIMATOR_AVAILABLE = False
    def init_auction_db(): pass

# ── Config ────────────────────────────────────────────────────────────────────

JWT_SECRET       = os.environ.get("JWT_SECRET", "de5db8e2b2a890547023651d946cf02048d7a8d7f3536c10d31cb44119704822")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRY_HOURS = 12
AUTH_DB_PATH     = Path("users.db")

DEV_USERNAME = "adrianperez"
DEV_PASSWORD = "xbox5678"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ── User DB ───────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_auth_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password   TEXT NOT NULL,
            role       TEXT NOT NULL DEFAULT 'user',
            created_at TEXT DEFAULT (datetime('now')),
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            title       TEXT NOT NULL,
            artist      TEXT,
            medium      TEXT,
            decade      INTEGER,
            width_cm    REAL,
            height_cm   REAL,
            wiki_url    TEXT,
            estimated_price_usd REAL,
            confidence_low      REAL,
            confidence_high     REAL,
            artist_score        REAL,
            notes       TEXT,
            added_at    TEXT DEFAULT (datetime("now")),
            updated_at  TEXT DEFAULT (datetime("now"))
        );
        CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id);
    """)
    conn.commit()
    # Ensure dev account always exists
    hashed = bcrypt.hashpw(DEV_PASSWORD.encode(), bcrypt.gensalt()).decode()
    conn.execute("""
        INSERT INTO users (username, password, role) VALUES (?, ?, 'admin')
        ON CONFLICT(username) DO UPDATE SET password=excluded.password, role='admin'
    """, (DEV_USERNAME, hashed))
    conn.commit()
    conn.close()


# ── JWT ───────────────────────────────────────────────────────────────────────

def make_token(uid, username, role):
    return jwt.encode({
        "sub": uid, "username": username, "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


def get_token():
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        p = decode_token(get_token() or "")
        if not p:
            return jsonify({"error": "Authentication required."}), 401
        request.user = p
        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        p = decode_token(get_token() or "")
        if not p:
            return jsonify({"error": "Authentication required."}), 401
        if p.get("role") != "admin":
            return jsonify({"error": "Admin access required."}), 403
        request.user = p
        return f(*args, **kwargs)
    return wrapper


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def login():
    d  = request.get_json(force=True) or {}
    un = (d.get("username") or "").strip()
    pw = d.get("password") or ""
    if not un or not pw:
        return jsonify({"error": "Username and password required."}), 400
    conn = get_conn()
    row  = conn.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (un,)).fetchone()
    if not row or not bcrypt.checkpw(pw.encode(), row["password"].encode()):
        conn.close()
        return jsonify({"error": "Invalid username or password."}), 401
    conn.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()
    return jsonify({
        "token":    make_token(row["id"], row["username"], row["role"]),
        "username": row["username"],
        "role":     row["role"],
    })


@app.route("/api/me", methods=["GET"])
@require_auth
def me():
    return jsonify({"username": request.user["username"], "role": request.user["role"]})


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
@require_admin
def list_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, role, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/users", methods=["POST"])
@require_admin
def create_user():
    d  = request.get_json(force=True) or {}
    un = (d.get("username") or "").strip()
    pw = d.get("password") or ""
    if not un or not pw:
        return jsonify({"error": "Username and password required."}), 400
    if len(pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    try:
        conn = get_conn()
        conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'user')",
                     (un, hashed))
        conn.commit()
        row = conn.execute(
            "SELECT id, username, role, created_at FROM users WHERE username=?", (un,)
        ).fetchone()
        conn.close()
        return jsonify(dict(row)), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": f"Username '{un}' already exists."}), 409


@app.route("/api/users/<int:uid>", methods=["DELETE"])
@require_admin
def delete_user(uid):
    if uid == request.user["sub"]:
        return jsonify({"error": "You cannot delete your own account."}), 400
    conn = get_conn()
    cur  = conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True}) if cur.rowcount else (jsonify({"error": "User not found."}), 404)


@app.route("/api/users/<int:uid>/password", methods=["PATCH"])
@require_admin
def reset_password(uid):
    d  = request.get_json(force=True) or {}
    pw = d.get("password") or ""
    if len(pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    conn   = get_conn()
    cur    = conn.execute("UPDATE users SET password=? WHERE id=?", (hashed, uid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True}) if cur.rowcount else (jsonify({"error": "User not found."}), 404)


# ── Estimator ─────────────────────────────────────────────────────────────────

@app.route("/api/estimate", methods=["POST"])
@require_auth
def estimate():
    if not ESTIMATOR_AVAILABLE:
        return jsonify({"error": "Estimator not set up yet."}), 503
    d   = request.get_json(force=True) or {}
    url = (d.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400
    if "wikipedia.org/wiki/" not in url:
        return jsonify({"error": "Please provide a valid Wikipedia article URL."}), 400
    try:
        return jsonify(estimate_from_url(url))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/estimate-manual", methods=["POST"])
@require_auth
def estimate_manual():
    if not ESTIMATOR_AVAILABLE:
        return jsonify({"error": "Estimator not set up yet."}), 503
    d = request.get_json(force=True) or {}

    artist_name       = (d.get("artist_name") or "").strip()
    artist_wiki_title = (d.get("artist_wiki_title") or "").strip() or None
    if not artist_name:
        return jsonify({"error": "artist_name is required."}), 400

    try:
        from auction_estimator import (
            lookup_artist_in_db, score_artist_live, predict_price,
            normalise_medium, is_3d_medium
        )

        artist = lookup_artist_in_db(artist_name, artist_wiki_title)
        if artist is None:
            artist = score_artist_live(artist_name, artist_wiki_title)

        medium    = normalise_medium(d.get("medium") or "unknown")
        width_cm  = d.get("width_cm")
        height_cm = d.get("height_cm")
        depth_cm  = d.get("depth_cm")
        decade    = d.get("decade")

        result = predict_price(
            artist_score = artist["score"],
            decade       = decade,
            medium       = medium,
            width_cm     = width_cm,
            height_cm    = height_cm,
            depth_cm     = depth_cm,
        )

        return jsonify({
            "artwork": {
                "title":    d.get("title") or "Untitled",
                "url":      None,
                "artist":   artist["name"],
                "decade":   decade,
                "medium":   medium,
                "width_cm": width_cm,
                "height_cm":height_cm,
                "depth_cm": depth_cm,
                "is_3d":    bool(is_3d_medium(medium) or depth_cm),
            },
            "artist_score":        artist["score"],
            "estimated_price_usd": result["estimated_price_usd"],
            "confidence_interval": result["confidence_interval"],
            "model_info": {
                "score_band":         result["score_band"],
                "model_band":         result["model_band"],
                "n_training_samples": result["n_training_samples"],
                "model_r2":           result["model_r2"],
                "model_mae_usd":      result["model_mae_usd"],
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/api/comparables", methods=["POST"])
@require_auth
def get_comparables():
    try:
        data = request.get_json()
        score_band = data.get("score_band", 5)
        medium = data.get("medium", "")
        decade = data.get("decade")
        title = data.get("title", "")

        conn = get_auction_conn()
        # Find similar artworks by band, prefer same medium and nearby decade
        rows = conn.execute("""
            SELECT aw.title, ar.name as artist, aw.sale_price_usd,
                   aw.sale_year, aw.medium_raw, aw.decade, aw.score_band
            FROM artworks aw
            JOIN artists ar ON ar.id = aw.artist_id
            WHERE aw.score_band BETWEEN ? AND ?
              AND aw.sale_price_usd > 0
              AND aw.sale_price_usd < 5000000000
              AND lower(aw.title) != lower(?)
              AND aw.title NOT LIKE '%%27%'
              AND aw.title NOT LIKE '%%25%'
              AND aw.title NOT LIKE '%NBA%'
              AND aw.title NOT LIKE '%Facebook%'
              AND aw.title NOT LIKE '%Google%'
              AND length(aw.title) > 3
            ORDER BY
                CASE WHEN lower(aw.medium_raw) LIKE ? THEN 0 ELSE 1 END,
                CASE WHEN aw.decade BETWEEN ? AND ? THEN 0 ELSE 1 END,
                ABS(aw.sale_price_usd - (
                    SELECT AVG(sale_price_usd) FROM artworks
                    WHERE score_band = ? AND sale_price_usd > 0
                ))
            LIMIT 5
        """, (
            max(1, score_band - 1), min(9, score_band + 1),
            title,
            f"%{medium[:20]}%" if medium else "%",
            (decade or 1900) - 20, (decade or 1900) + 20,
            score_band
        )).fetchall()
        conn.close()

        comparables = []
        for r in rows:
            comparables.append({
                "title": r["title"],
                "artist": r["artist"],
                "sale_price_usd": r["sale_price_usd"],
                "sale_year": r["sale_year"],
                "medium": r["medium_raw"],
                "decade": r["decade"],
            })
        return jsonify({"comparables": comparables})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio", methods=["GET"])
@require_auth
def get_portfolio():
    try:
        user_id = request.user["sub"]
        conn = get_conn()
        rows = conn.execute("""
            SELECT * FROM portfolio WHERE user_id=? ORDER BY added_at DESC
        """, (user_id,)).fetchall()
        conn.close()
        items = [dict(r) for r in rows]
        total = sum(r["estimated_price_usd"] or 0 for r in items)
        return jsonify({"portfolio": items, "total_value": total})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/portfolio", methods=["POST"])
@require_auth
def add_to_portfolio():
    try:
        user_id = request.user["sub"]
        data = request.get_json()
        conn = get_conn()
        conn.execute("""
            INSERT INTO portfolio
            (user_id, title, artist, medium, decade, width_cm, height_cm,
             wiki_url, estimated_price_usd, confidence_low, confidence_high, artist_score, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            user_id,
            data.get("title"), data.get("artist"), data.get("medium"),
            data.get("decade"), data.get("width_cm"), data.get("height_cm"),
            data.get("wiki_url"), data.get("estimated_price_usd"),
            data.get("confidence_low"), data.get("confidence_high"),
            data.get("artist_score"), data.get("notes", "")
        ))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/portfolio/<int:item_id>", methods=["DELETE"])
@require_auth
def delete_portfolio_item(item_id):
    try:
        user_id = request.user["sub"]
        conn = get_conn()
        conn.execute("DELETE FROM portfolio WHERE id=? AND user_id=?", (item_id, user_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/identify-image", methods=["POST"])
@require_auth
def identify_image():
    try:
        import base64, json as json_mod
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return jsonify({"error": "Gemini API key not configured"}), 500

        data = request.get_json()
        image_b64 = data.get("image_b64")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        prompt = (
            "You are an expert art historian. Look at this artwork and identify it. "
            "Search your knowledge to determine: the exact title, the artist full name, "
            "and the Wikipedia URL for this specific artwork. "
            "Reply ONLY as JSON like: "
            '{"title": "The Starry Night", "artist": "Vincent van Gogh", ' 
            '"wiki_url": "https://en.wikipedia.org/wiki/The_Starry_Night", ' 
            '"confidence": "high"} '
            "If you cannot identify it, reply: "
            '{"title": null, "artist": null, "wiki_url": null, "confidence": "low"}' 
        )

        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                        {"text": prompt}
                    ]
                }]
            },
            timeout=30,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Extract JSON from response
        import re as re_mod
        match = re_mod.search(r'\{.*\}', text, re_mod.DOTALL)
        if match:
            result = json_mod.loads(match.group())
            return jsonify(result)
        return jsonify({"error": "Could not parse Gemini response", "raw": text}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})



# ── Startup ───────────────────────────────────────────────────────────────────

init_auth_db()
init_auction_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
