# YTLearn

Extract structured learning notes from any YouTube channel. Videos are **auto-categorised by topic** — no manual searching. Get AI-generated summaries, key takeaways, and personalised action steps.

**[Live demo →](https://your-app.railway.app)**

---

## What makes it different

- **Auto-categorisation** — paste a channel, AI groups videos into topics automatically
- **Filter by category** — browse "Nutrition", "Sleep", "Exercise" instead of scrolling 50 titles
- **Personalised apply steps** — set your context once, every note tells you *specifically* how to use it
- **No install for users** — hosted, share a link, it just works
- **SQLite cache** — same video processed twice = instant, no extra API cost

## Deploy in 5 minutes (Railway)

1. Fork this repo on GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Select your fork
4. Go to Variables and add:
   ```
   OPENAI_API_KEY=sk-...
   USER_CONTEXT=I am a solo founder building AI products...
   ```
5. Click Deploy — Railway gives you a public URL instantly

That's it. Share the URL.

## Deploy on Render (alternative, also free)

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. Render auto-detects `render.yaml` — just add your `OPENAI_API_KEY` in Environment
4. Deploy → get a public URL

## Run locally (Mac/Linux)

```bash
git clone https://github.com/yourusername/ytlearn
cd ytlearn
pip3 install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python3 app.py
# open http://localhost:5050
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Your OpenAI key — users never see this |
| `USER_CONTEXT` | Recommended | Describes your audience for personalised notes |
| `NOTION_API_KEY` | Optional | Auto-push notes to Notion |
| `NOTION_DATABASE_ID` | Optional | Target Notion database |

## Cost

You pay for OpenAI. Users pay nothing.

- ~$0.001 per video processed
- 100 videos/day = ~$0.10/day
- $5 OpenAI credit lasts months at typical usage

## Stack

Python · Flask · yt-dlp · youtube-transcript-api · OpenAI gpt-4o-mini · SQLite · Vanilla JS
