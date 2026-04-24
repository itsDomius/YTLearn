import os
import json
import sqlite3
import re
from flask import Flask, request, jsonify, send_from_directory, make_response
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
USER_CONTEXT = os.getenv("USER_CONTEXT", "I am a learner who wants practical, actionable insights.")
DB_PATH = os.getenv("DB_PATH", "ytlearn.db")
PORT = int(os.getenv("PORT", 5050))

# ── Note styles ───────────────────────────────────────────────────────────────

NOTE_STYLES = {
    "study": {
        "label": "Study Notes",
        "emoji": "📚",
        "desc": "Dense & comprehensive",
        "model": "gpt-4o",
        "instruction": """Extract dense, comprehensive notes for mastering this topic.
- Summary: 3-4 sentences capturing the core thesis, key arguments, and most important points
- Key Takeaways: 6 specific insights including data points, frameworks, and concrete examples actually mentioned
- How to Apply: 3 practical steps directly tied to the learner's context, grounded in what was taught""",
        "max_tokens": 1500,
    },
    "brief": {
        "label": "Executive Brief",
        "emoji": "⚡",
        "desc": "Ruthlessly condensed",
        "model": "gpt-4o-mini",
        "instruction": """Strip this to the absolute highest-signal information only.
- Summary: 2 sentences max — what is this and why does it matter right now
- Key Takeaways: 5 points, each under 15 words, zero filler, zero repetition
- How to Apply: 3 things the learner can do THIS week, ultra-specific""",
        "max_tokens": 900,
    },
    "founder": {
        "label": "Founder Lens",
        "emoji": "🚀",
        "desc": "Reframed for builders",
        "model": "gpt-4o",
        "instruction": """Reframe everything specifically for someone building a product or startup.
- Summary: 2-3 sentences on what founders should take from this — focus on growth, product, or mindset
- Key Takeaways: 5 insights rewritten as founder lessons tied to growth, product, hiring, revenue, or resilience
- How to Apply: 3 concrete moves for someone building right now, directly tailored to the learner's context""",
        "max_tokens": 1400,
    },
    "tweets": {
        "label": "Tweet Thread",
        "emoji": "🐦",
        "desc": "Ready to post",
        "model": "gpt-4o-mini",
        "instruction": """Turn the key ideas into a tweet thread ready to publish.
- Summary: one hook tweet under 280 chars that makes someone stop scrolling — bold, specific, intriguing
- Key Takeaways: 6 standalone tweets each under 260 chars, each a complete insight, no hashtags
- How to Apply: 2 closing tweets as a strong CTA — what the reader should do or think next""",
        "max_tokens": 1000,
    },
    "actions": {
        "label": "Action Plan",
        "emoji": "✅",
        "desc": "Pure next steps",
        "model": "gpt-4o",
        "instruction": """Remove all theory. Extract ONLY what can be acted on immediately.
- Summary: what specific problem does this solve and for whom
- Key Takeaways: 5 named techniques or frameworks explicitly mentioned — use their exact names
- How to Apply: 5 ordered action steps from easiest to hardest, each tied directly to the learner's context""",
        "max_tokens": 1400,
    },
}

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
            video_id TEXT NOT NULL,
            style TEXT NOT NULL DEFAULT 'study',
            video_title TEXT,
            channel_name TEXT,
            thumbnail TEXT,
            summary TEXT,
            key_takeaways TEXT,
            how_to_apply TEXT,
            raw_transcript TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_id, style)
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
    ydl_opts = {"quiet": True, "extract_flat": True,
                "playlist_items": f"1:{max_videos}", "no_warnings": True}
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
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    titles_payload = [{"id": v["id"], "title": v["title"]} for v in videos]
    prompt = f"""Categorise these YouTube videos into 3-6 topic groups.

Videos:
{json.dumps(titles_payload, indent=2)}

Respond ONLY with JSON (no markdown):
{{
  "categories": ["Category 1", "Category 2"],
  "assignments": {{"video_id": "Category 1"}}
}}

Rules: categories 1-3 words, every video assigned, roughly even distribution."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    result = json.loads(raw)
    assignments = result.get("assignments", {})
    categories = result.get("categories", [])
    for v in videos:
        v["category"] = assignments.get(v["id"], categories[0] if categories else "General")
    return videos, categories

def fetch_transcript(video_id):
    """
    Fetch transcript using yt-dlp — handles cloud IP blocks better than
    youtube-transcript-api by mimicking a real browser request.
    Falls back to youtube-transcript-api if yt-dlp fails.
    """
    import yt_dlp
    import urllib.request
    import json as _json

    url = f"https://www.youtube.com/watch?v={video_id}"

    # ── Primary: yt-dlp subtitle extraction ──────────────────────────────────
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
            # Mimic a real browser
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # Prefer manual English subtitles, fall back to auto-generated
        subs = info.get("subtitles", {})
        auto = info.get("automatic_captions", {})

        chosen = None
        for lang in ["en", "en-US", "en-GB", "en-orig"]:
            if lang in subs:
                chosen = subs[lang]
                break
        if not chosen:
            for lang in ["en", "en-US", "en-GB", "en-orig"]:
                if lang in auto:
                    chosen = auto[lang]
                    break

        if chosen:
            # Find JSON3 format (easiest to parse), fall back to srv1/vtt
            fmt_url = None
            for fmt in chosen:
                if fmt.get("ext") == "json3":
                    fmt_url = fmt["url"]
                    break
            if not fmt_url:
                for fmt in chosen:
                    if fmt.get("ext") in ("srv1", "vtt", "srv3"):
                        fmt_url = fmt["url"]
                        break

            if fmt_url:
                req = urllib.request.Request(
                    fmt_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode("utf-8")

                # Parse JSON3 format
                if "json3" in fmt_url or raw.strip().startswith("{"):
                    try:
                        data = _json.loads(raw)
                        events = data.get("events", [])
                        parts = []
                        for event in events:
                            for seg in event.get("segs", []):
                                t = seg.get("utf8", "").strip()
                                if t and t != "\n":
                                    parts.append(t)
                        text = " ".join(parts)
                    except Exception:
                        # Fall through to VTT parsing
                        text = _parse_vtt(raw)
                else:
                    text = _parse_vtt(raw)

                text = re.sub(r"\[.*?\]", "", text)
                text = re.sub(r"<[^>]+>", "", text)   # strip HTML tags from VTT
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 100:
                    return text, None

    except Exception as e:
        pass  # Fall through to backup method

    # ── Fallback: youtube-transcript-api ────────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join([t.text for t in fetched])
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, None
    except Exception as e2:
        err = str(e2)
        if "disabled" in err.lower():
            return None, "Transcripts are disabled for this video."
        if "no transcript" in err.lower() or "Could not find" in err.lower():
            return None, "No English transcript found."
        return None, "Could not fetch transcript — YouTube may be blocking this server. Try a different video or contact support."


def _parse_vtt(vtt_text):
    """Extract plain text from WebVTT subtitle format."""
    lines = []
    for line in vtt_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("NOTE") or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        lines.append(line)
    return " ".join(lines)

def generate_notes(title, transcript, user_context, style="study"):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    cfg = NOTE_STYLES.get(style, NOTE_STYLES["study"])
    truncated = transcript[:14000]
    system = """You are an elite learning assistant used by high-performers to extract maximum value from video content.
Your notes are specific, dense, and directly useful — never vague, never padded, never generic.
Every insight must be grounded in something actually said in the transcript.
Never invent information not present in the source material.
Output only valid JSON with no markdown fences."""
    prompt = f"""Video Title: {title}
Learner Context: {user_context}
Note Style: {cfg["label"]}

Instructions:
{cfg["instruction"]}

Transcript:
{truncated}

Return this exact JSON:
{{
  "summary": "...",
  "key_takeaways": ["...", "...", "...", "...", "...", "..."],
  "how_to_apply": ["...", "...", "..."]
}}"""
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=cfg["max_tokens"],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

def build_markdown(title, channel, video_url, style, summary, takeaways, apply_steps):
    style_label = NOTE_STYLES.get(style, {}).get("label", style)
    lines = [
        f"# {title}",
        f"**Channel:** {channel}  |  **Style:** {style_label}",
        f"**Source:** {video_url}",
        "",
        "## Summary",
        summary,
        "",
        "## Key Takeaways",
    ]
    for t in takeaways:
        lines.append(f"- {t}")
    lines += ["", "## How to Apply"]
    for a in apply_steps:
        lines.append(f"- [ ] {a}")
    return "\n".join(lines)

def build_notion_blocks(title, channel, video_url, style, summary, takeaways, apply_steps):
    style_label = NOTE_STYLES.get(style, {}).get("label", style)
    blocks = [
        {"object": "block", "type": "heading_1",
         "heading_1": {"rich_text": [{"text": {"content": title}}]}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": [
             {"text": {"content": f"Channel: {channel}  |  Style: {style_label}  |  "}},
             {"text": {"content": video_url, "link": {"url": video_url}}},
         ]}},
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"text": {"content": "📋 Summary"}}]}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": [{"text": {"content": summary}}]}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"text": {"content": "⚡ Key Takeaways"}}]}},
    ]
    for t in takeaways:
        blocks.append({"object": "block", "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": t}}]}})
    blocks.append({"object": "block", "type": "heading_2",
                   "heading_2": {"rich_text": [{"text": {"content": "🚀 How to Apply"}}]}})
    for a in apply_steps:
        blocks.append({"object": "block", "type": "to_do",
                        "to_do": {"rich_text": [{"text": {"content": a}}], "checked": False}})
    return blocks

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
        "note_styles": [
            {"id": k, "label": v["label"], "emoji": v["emoji"], "desc": v["desc"]}
            for k, v in NOTE_STYLES.items()
        ],
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
    style = data.get("style", "study")

    if not video_ids:
        return jsonify({"error": "No videos selected"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured."}), 400
    if style not in NOTE_STYLES:
        style = "study"

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

        cached = conn.execute(
            "SELECT * FROM notes WHERE video_id = ? AND style = ?", (vid_id, style)
        ).fetchone()
        if cached:
            results.append({
                "video_id": vid_id, "title": cached["video_title"],
                "channel": cached["channel_name"], "thumbnail": cached["thumbnail"],
                "category": category, "style": style,
                "summary": cached["summary"],
                "key_takeaways": json.loads(cached["key_takeaways"] or "[]"),
                "how_to_apply": json.loads(cached["how_to_apply"] or "[]"),
                "cached": True, "video_url": video_url,
            })
            continue

        transcript, err = fetch_transcript(vid_id)
        if err:
            errors.append({"video_id": vid_id, "title": title, "error": err})
            continue

        try:
            notes = generate_notes(title, transcript, user_context, style)
        except Exception as e:
            errors.append({"video_id": vid_id, "title": title, "error": f"AI error: {str(e)}"})
            continue

        summary = notes.get("summary", "")
        takeaways = notes.get("key_takeaways", [])
        apply_steps = notes.get("how_to_apply", [])

        conn.execute("""
            INSERT OR REPLACE INTO notes
            (video_id, style, video_title, channel_name, thumbnail, summary, key_takeaways, how_to_apply, raw_transcript)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (vid_id, style, title, channel, thumbnail, summary,
              json.dumps(takeaways), json.dumps(apply_steps), transcript[:5000]))
        conn.commit()

        results.append({
            "video_id": vid_id, "title": title, "channel": channel,
            "thumbnail": thumbnail, "category": category, "style": style,
            "summary": summary, "key_takeaways": takeaways,
            "how_to_apply": apply_steps, "cached": False, "video_url": video_url,
        })

    conn.close()
    return jsonify({"results": results, "errors": errors})

@app.route("/api/export", methods=["POST"])
def export_note():
    data = request.json
    fmt = data.get("format", "md")
    title = data.get("title", "Note")
    channel = data.get("channel", "")
    video_url = data.get("video_url", "")
    style = data.get("style", "study")
    summary = data.get("summary", "")
    takeaways = data.get("key_takeaways", [])
    apply_steps = data.get("how_to_apply", [])

    md = build_markdown(title, channel, video_url, style, summary, takeaways, apply_steps)
    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()

    if fmt == "md":
        resp = make_response(md)
        resp.headers["Content-Type"] = "text/markdown"
        resp.headers["Content-Disposition"] = f'attachment; filename="{safe_title}.md"'
        return resp

    elif fmt == "txt":
        resp = make_response(md)
        resp.headers["Content-Type"] = "text/plain"
        resp.headers["Content-Disposition"] = f'attachment; filename="{safe_title}.txt"'
        return resp

    elif fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            import io

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                leftMargin=22*mm, rightMargin=22*mm,
                topMargin=22*mm, bottomMargin=22*mm)

            style_label = NOTE_STYLES.get(style, {}).get("label", style)
            title_s = ParagraphStyle("T", fontSize=20, fontName="Helvetica-Bold",
                spaceAfter=4, textColor=colors.HexColor("#0a0a0a"))
            meta_s = ParagraphStyle("M", fontSize=9, fontName="Helvetica",
                spaceAfter=14, textColor=colors.HexColor("#888888"))
            h2_s = ParagraphStyle("H2", fontSize=12, fontName="Helvetica-Bold",
                spaceBefore=18, spaceAfter=6, textColor=colors.HexColor("#111111"))
            body_s = ParagraphStyle("B", fontSize=10, fontName="Helvetica",
                spaceAfter=8, textColor=colors.HexColor("#333333"), leading=16)
            bullet_s = ParagraphStyle("BL", fontSize=10, fontName="Helvetica",
                spaceAfter=5, textColor=colors.HexColor("#333333"), leading=15,
                leftIndent=14)

            story = [
                Paragraph(title, title_s),
                Paragraph(f"{channel}  ·  {style_label}", meta_s),
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=14),
                Paragraph("Summary", h2_s),
                Paragraph(summary, body_s),
                Paragraph("Key Takeaways", h2_s),
            ]
            for t in takeaways:
                story.append(Paragraph(f"→  {t}", bullet_s))
            story.append(Paragraph("How to Apply", h2_s))
            for a in apply_steps:
                story.append(Paragraph(f"☐  {a}", bullet_s))

            doc.build(story)
            buf.seek(0)
            resp = make_response(buf.read())
            resp.headers["Content-Type"] = "application/pdf"
            resp.headers["Content-Disposition"] = f'attachment; filename="{safe_title}.pdf"'
            return resp
        except ImportError:
            resp = make_response(md)
            resp.headers["Content-Type"] = "text/plain"
            resp.headers["Content-Disposition"] = f'attachment; filename="{safe_title}.txt"'
            return resp

    elif fmt == "notion":
        blocks = build_notion_blocks(title, channel, video_url, style, summary, takeaways, apply_steps)
        return jsonify({"blocks": blocks, "json": json.dumps(blocks, indent=2)})

    return jsonify({"error": "Unknown format"}), 400

@app.route("/api/library")
def get_library():
    conn = get_db()
    rows = conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify({"notes": [{
        "video_id": r["video_id"], "style": r["style"],
        "title": r["video_title"], "channel": r["channel_name"],
        "thumbnail": r["thumbnail"], "summary": r["summary"],
        "key_takeaways": json.loads(r["key_takeaways"] or "[]"),
        "how_to_apply": json.loads(r["how_to_apply"] or "[]"),
        "created_at": r["created_at"],
    } for r in rows]})

@app.route("/api/library/<video_id>", methods=["DELETE"])
def delete_note(video_id):
    style = request.args.get("style")
    conn = get_db()
    if style:
        conn.execute("DELETE FROM notes WHERE video_id = ? AND style = ?", (video_id, style))
    else:
        conn.execute("DELETE FROM notes WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV") != "production", host="0.0.0.0", port=PORT)
