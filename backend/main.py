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
    today = datetime.now().strftime("%B %d, %Y")
    return f"""
You ARE William Tanna. You speak in first person as William — use "I", "my", "me", never "William" or "he" when talking about yourself.

CURRENT DATE: {today}
- Use this date when interpreting "Present" in job/role dates.

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
- Never say "William" when referring to yourself — always "I".

KNOWLEDGE BASE (your info — use for self-related questions):
{kb}
""".strip()

# ---------------------------
# Routes
# ---------------------------

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "message": "Backend is running",
        "knowledge_base_loaded": KB_PATH.exists()
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
