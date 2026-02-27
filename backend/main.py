# backend/main.py - FastAPI Backend for William Tanna Website Chatbot

import os
import re
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

def load_knowledge_base() -> str:
    if KB_PATH.exists():
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
    return f"""
You are a helpful AI assistant on William Tanna's personal portfolio website.

Rules:
- For questions about William (background, experience, skills, projects, contact), use the knowledge base below.
- For any other questions, answer helpfully using your general knowledge.
- Keep responses professional, conversational, and concise.
- When using the knowledge base, include specific details and metrics. Don't invent facts about William.
- Keep responses under 300 words unless the user asks for more detail.

KNOWLEDGE BASE (use for William-related questions):
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
        temperature=0.55,
        max_tokens=800,
    )

    raw = completion.choices[0].message.content.strip()
    highlighted = highlight_text(raw)

    return ChatResponse(
        success=True,
        response=highlighted,
        raw_response=raw,
        used_kb=bool(kb and "Knowledge base file is not added yet" not in kb)
    )
