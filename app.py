import os
import json
import sqlite3
import re
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
USER_CONTEXT = os.getenv("USER_CONTEXT", "I am a learner who wants practical, actionable insights.")
DB_PATH = os.getenv("DB_PATH", "ytlearn.db")
PORT = int(os.getenv("PORT", 5050))

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            video_title TEXT,
            channel_name TEXT,
            thumbnail TEXT,
            duration TEXT,
            summary TEXT,
            key_takeaways TEXT,
            how_to_apply TEXT,
            raw_transcript TEXT,
            notion_page_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS channel_cache (
            channel_url TEXT PRIMARY KEY,
            videos TEXT,
            categories TEXT,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_duration(seconds):
    if not seconds:
        return "?"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def fetch_channel_videos(channel_url, max_videos=50):
    import yt_dlp
    clean_url = channel_url.strip()
    if not clean_url.startswith("http"):
        if clean_url.startswith("@"):
            clean_url = "https://www.youtube.com/" + clean_url
        else:
            clean_url = "https://www.youtube.com/@" + clean_url

    if "/videos" not in clean_url and "playlist" not in clean_url and "watch" not in clean_url:
        clean_url = clean_url.rstrip("/") + "/videos"

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "playlist_items": f"1:{max_videos}",
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=False)

    videos = []
    if info and "entries" in info:
        for entry in info["entries"]:
            if entry and entry.get("id"):
                videos.append({
                    "id": entry.get("id"),
                    "title": entry.get("title", "Untitled"),
                    "duration": format_duration(entry.get("duration")),
                    "thumbnail": entry.get("thumbnail") or f"https://i.ytimg.com/vi/{entry.get('id')}/mqdefault.jpg",
                    "channel": info.get("channel") or info.get("uploader") or "",
                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                    "category": None,
                })
    return videos

def auto_categorise(videos):
    """Send video titles to GPT and get back category assignments."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    titles_payload = [{"id": v["id"], "title": v["title"]} for v in videos]

    prompt = f"""You are a content categorisation assistant.

Given these YouTube video titles from a single channel, identify 3-6 topic categories that best describe the channel's content. Then assign each video to exactly one category.

Videos:
{json.dumps(titles_payload, indent=2)}

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{
  "categories": ["Category Name 1", "Category Name 2", "Category Name 3"],
  "assignments": {{
    "video_id_here": "Category Name 1",
    "video_id_here2": "Category Name 2"
  }}
}}

Rules:
- Categories should be short (1-3 words), descriptive topic names
- Every video must be assigned to exactly one category
- Use the exact category names from your categories list in assignments
- Aim for roughly even distribution — avoid putting 90% of videos in one category"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    result = json.loads(raw)

    # Apply categories to videos
    assignments = result.get("assignments", {})
    categories = result.get("categories", [])
    for v in videos:
        v["category"] = assignments.get(v["id"], categories[0] if categories else "General")

    return videos, categories

def fetch_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join([t["text"] for t in transcript_list])
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, None
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No English transcript found."
    except Exception as e:
        return None, str(e)

def generate_notes(title, transcript, user_context):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    truncated = transcript[:12000]
    prompt = f"""You are an expert learning assistant. Analyse this YouTube video transcript and produce structured notes.

Video Title: {title}
Learner Context: {user_context}

Transcript:
{truncated}

Respond ONLY with this JSON (no markdown):
{{
  "summary": "2-3 sentence overview",
  "key_takeaways": [
    "Specific actionable insight 1",
    "Specific actionable insight 2",
    "Specific actionable insight 3",
    "Specific actionable insight 4",
    "Specific actionable insight 5"
  ],
  "how_to_apply": [
    "Concrete action step tailored to: {user_context}",
    "Concrete action step tailored to: {user_context}",
    "Concrete action step tailored to: {user_context}"
  ]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

def save_to_notion(title, channel, video_url, summary, takeaways, apply_steps):
    from notion_client import Client
    notion = Client(auth=NOTION_API_KEY)
    takeaway_text = "\n".join([f"• {t}" for t in takeaways])
    apply_text = "\n".join([f"• {a}" for a in apply_steps])
    notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties={
            "Name": {"title": [{"text": {"content": title}}]},
            "Channel": {"rich_text": [{"text": {"content": channel}}]},
            "URL": {"url": video_url},
        },
        children=[
            {"object": "block", "type": "heading_2",
             "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}},
            {"object": "block", "type": "paragraph",
             "paragraph": {"rich_text": [{"text": {"content": summary}}]}},
            {"object": "block", "type": "heading_2",
             "heading_2": {"rich_text": [{"text": {"content": "Key Takeaways"}}]}},
            {"object": "block", "type": "paragraph",
             "paragraph": {"rich_text": [{"text": {"content": takeaway_text}}]}},
            {"object": "block", "type": "heading_2",
             "heading_2": {"rich_text": [{"text": {"content": "How to Apply"}}]}},
            {"object": "block", "type": "paragraph",
             "paragraph": {"rich_text": [{"text": {"content": apply_text}}]}},
        ],
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/config")
def get_config():
    return jsonify({
        "openai_configured": bool(OPENAI_API_KEY),
        "notion_configured": bool(NOTION_API_KEY and NOTION_DATABASE_ID),
        "user_context": USER_CONTEXT,
    })

@app.route("/api/channel", methods=["POST"])
def get_channel():
    data = request.json
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server."}), 400

    try:
        videos = fetch_channel_videos(url, max_videos=50)
        if not videos:
            return jsonify({"error": "No videos found. Check the channel URL and try again."}), 404

        # Auto-categorise
        videos, categories = auto_categorise(videos)

        return jsonify({"videos": videos, "categories": categories})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch channel: {str(e)}"}), 500

@app.route("/api/process", methods=["POST"])
def process_videos():
    data = request.json
    video_ids = data.get("video_ids", [])
    video_map = data.get("video_map", {})
    user_context = data.get("user_context", USER_CONTEXT).strip() or USER_CONTEXT

    if not video_ids:
        return jsonify({"error": "No videos selected"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured."}), 400

    results = []
    errors = []
    conn = get_db()

    for vid_id in video_ids:
        info = video_map.get(vid_id, {})
        title = info.get("title", vid_id)
        channel = info.get("channel", "")
        thumbnail = info.get("thumbnail", "")
        video_url = info.get("url", f"https://www.youtube.com/watch?v={vid_id}")
        category = info.get("category", "General")

        # Cache hit
        cached = conn.execute("SELECT * FROM notes WHERE video_id = ?", (vid_id,)).fetchone()
        if cached:
            results.append({
                "video_id": vid_id,
                "title": cached["video_title"],
                "channel": cached["channel_name"],
                "thumbnail": cached["thumbnail"],
                "category": category,
                "summary": cached["summary"],
                "key_takeaways": json.loads(cached["key_takeaways"] or "[]"),
                "how_to_apply": json.loads(cached["how_to_apply"] or "[]"),
                "cached": True,
                "video_url": video_url,
            })
            continue

        transcript, err = fetch_transcript(vid_id)
        if err:
            errors.append({"video_id": vid_id, "title": title, "error": err})
            continue

        try:
            notes = generate_notes(title, transcript, user_context)
        except Exception as e:
            errors.append({"video_id": vid_id, "title": title, "error": f"AI error: {str(e)}"})
            continue

        summary = notes.get("summary", "")
        takeaways = notes.get("key_takeaways", [])
        apply_steps = notes.get("how_to_apply", [])

        notion_page_id = None
        if NOTION_API_KEY and NOTION_DATABASE_ID:
            try:
                save_to_notion(title, channel, video_url, summary, takeaways, apply_steps)
            except Exception:
                pass

        conn.execute("""
            INSERT OR REPLACE INTO notes
            (video_id, video_title, channel_name, thumbnail, summary, key_takeaways, how_to_apply, raw_transcript, notion_page_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (vid_id, title, channel, thumbnail, summary,
              json.dumps(takeaways), json.dumps(apply_steps),
              transcript[:5000], notion_page_id))
        conn.commit()

        results.append({
            "video_id": vid_id,
            "title": title,
            "channel": channel,
            "thumbnail": thumbnail,
            "category": category,
            "summary": summary,
            "key_takeaways": takeaways,
            "how_to_apply": apply_steps,
            "cached": False,
            "video_url": video_url,
        })

    conn.close()
    return jsonify({"results": results, "errors": errors})

@app.route("/api/library")
def get_library():
    conn = get_db()
    rows = conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify({"notes": [{
        "video_id": r["video_id"],
        "title": r["video_title"],
        "channel": r["channel_name"],
        "thumbnail": r["thumbnail"],
        "summary": r["summary"],
        "key_takeaways": json.loads(r["key_takeaways"] or "[]"),
        "how_to_apply": json.loads(r["how_to_apply"] or "[]"),
        "created_at": r["created_at"],
    } for r in rows]})

@app.route("/api/library/<video_id>", methods=["DELETE"])
def delete_note(video_id):
    conn = get_db()
    conn.execute("DELETE FROM notes WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV") != "production", host="0.0.0.0", port=PORT)
