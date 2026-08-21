# backend/main.py - FastAPI Backend for William Tanna Website Chatbot

import os
import re
from typing import Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---------------------------
# Config
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"
KB_PATH = BASE_DIR / "knowledge_base.txt"

# Load .env from project root or backend folder
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv(dotenv_path=BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Add it to .env in the project root.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="William Tanna Portfolio API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models
# ---------------------------

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    raw_response: str
    used_kb: bool

# ---------------------------
# Helpers
# ---------------------------

# Cache knowledge base in memory (avoid disk read every request)
_KB_CACHE: Optional[str] = None
_KB_MTIME: Optional[float] = None

def load_knowledge_base() -> str:
    global _KB_CACHE, _KB_MTIME
    if KB_PATH.exists():
        try:
            mtime = KB_PATH.stat().st_mtime
            if _KB_CACHE is not None and _KB_MTIME == mtime:
                return _KB_CACHE
            _KB_CACHE = KB_PATH.read_text(encoding="utf-8").strip()
            _KB_MTIME = mtime
            return _KB_CACHE
        except OSError:
            pass
        return KB_PATH.read_text(encoding="utf-8").strip()
    return (
        "NAME: William Tanna\n"
        "NOTE: Knowledge base file is not added yet.\n"
        "If asked for details, say you don't have enough information."
    )

def highlight_text(text: str) -> str:
    text = re.sub(r'(\d+(\.\d+)?)%', r'<span class="highlight-stat">\1%</span>', text)
    text = re.sub(r'(\$[\d,]+(\.\d+)?\s*[KMB]?)', r'<span class="highlight-money">\1</span>', text)
    text = re.sub(r'(₹\s?[\d,]+(\.\d+)?\s*Cr)', r'<span class="highlight-money">\1</span>', text)
    text = re.sub(r'(\d{2,})\+', r'<span class="highlight-number">\1+</span>', text)
    keywords = [
        "improved", "decreased", "increased", "analyzed", "managed", "led",
        "built", "developed", "created", "delivered", "automated", "verified",
        "ensured", "authored", "proposed", "mentored", "coordinated"
    ]
    for kw in keywords:
        text = re.sub(rf"\b({kw})\b", r'<strong class="highlight-keyword">\1</strong>', text, flags=re.IGNORECASE)
    return text

def build_system_prompt(kb: str) -> str:
    now = datetime.now()
    today = now.strftime("%B %d, %Y")
    # Graduation is always a completed fact (May 2026). Spell out timeline vs "today".
    return f"""
You ARE William Tanna. You speak in first person as William — use "I", "my", "me". Never refer to yourself as "William" or "he" unless the visitor explicitly asks for a third-person bio.

TODAY IS: {today} (server clock — treat this as the real current date).
- It is AFTER May 2026. You already graduated. Graduation is in the PAST (about 3 months before August 2026).
- Never say you are "expected to graduate," "currently pursuing a degree," or still in school.

HARD FACTS (never contradict these):
- You GRADUATED from UIC in May 2026 with Cum Laude honors. Completed. Done.
- Dual degrees: Finance and Information & Decision Sciences.
- Concentrations: Business Analytics and Supply Chain & Operations.
- CFA Level I candidate — exam November 2026.
- FORBIDDEN phrases: "aspiring", "currently pursuing a B.S.", "expected to graduate", "expecting to graduate", "I have not graduated yet", "I am expected to graduate".

If someone asks for a short bio (first or third person), use this shape:
- Graduated Cum Laude from UIC (May 2026) with dual degrees in Finance and IDS (Business Analytics + Supply Chain & Operations concentrations).
- CFA Level I candidate (Nov 2026 exam).
- Builder + finance analyst: live fund work (Gap thesis / allocation), internships (Sharekhan, Michael B. Michael Fund), and shipped products (Tech Circle 1st place, ClearSpend, Chargeback Risk Intelligence).

VOICE & TONE:
- Confident, direct, no fluff. Sound human and a little sharp — not corporate-speak.
- Emphasize builder identity: you ship things end to end rather than just theorizing.
- When asked about skills, experience, or achievements: be proud, lead with impact and numbers.
- Keep responses conversational and under 300 words unless they ask for more.
- Job search: stay forward-looking; you can say you're actively interviewing across finance/fintech/IB/analytics/startup ops — don't invent specific companies or interview stages.

RULES:
- For questions about you (background, experience, skills, projects, contact): use the knowledge base below. Answer as William in first person.
- For other questions: answer helpfully using your general knowledge, still as William when relevant.
- Don't invent facts. Stick to the knowledge base for William-specific info.
- Never say "William" when referring to yourself in first person — always "I".

KNOWLEDGE BASE (your info — use for self-related questions):
{kb}
""".strip()

# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    now = datetime.now()
    return {
        "service": "William Tanna Portfolio Chat API",
        "status": "ok",
        "server_time": now.strftime("%B %d, %Y %H:%M"),
        "endpoints": {
            "health": "/api/health",
            "chat": "POST /api/chat",
        },
        "note": "This is an API, not a website. Use /api/health to check status.",
    }

@app.get("/api/health")
def health():
    now = datetime.now()
    return {
        "status": "healthy",
        "message": "Backend is running",
        "knowledge_base_loaded": KB_PATH.exists(),
        "server_time": now.strftime("%B %d, %Y %H:%M"),
        "prompt_version": "2026-08-graduated",
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="No message provided")

    kb = load_knowledge_base()
    system_prompt = build_system_prompt(kb)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        max_tokens=450,
    )

    raw = completion.choices[0].message.content.strip()
    highlighted = highlight_text(raw)

    return ChatResponse(
        success=True,
        response=highlighted,
        raw_response=raw,
        used_kb=bool(kb and "Knowledge base file is not added yet" not in kb)
    )
