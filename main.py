import os, sqlite3, datetime
from contextlib import contextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

DB_PATH = os.getenv("DATABASE_PATH", "edusmart.db")
app = FastAPI(title="EduSmart AI API", version="1.0.0", description="Student progress and learning tutor API")

origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins or ["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_name TEXT NOT NULL,
            course TEXT DEFAULT '', subject TEXT NOT NULL, correct INTEGER NOT NULL,
            total INTEGER NOT NULL, score INTEGER NOT NULL, created_at TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_name TEXT UNIQUE NOT NULL,
            course TEXT DEFAULT '', updated_at TEXT NOT NULL
        )""")
init_db()

class Attempt(BaseModel):
    student_name: str = Field(default="Student", max_length=50)
    course: str = Field(default="", max_length=80)
    subject: str = Field(min_length=1, max_length=40)
    correct: int = Field(ge=0)
    total: int = Field(gt=0)
    score: int = Field(ge=0, le=100)
    date: Optional[str] = None

class Profile(BaseModel):
    name: str = Field(default="Student", max_length=50)
    course: str = Field(default="", max_length=80)

class TutorRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    student_name: str = Field(default="Student", max_length=50)
    course: str = Field(default="", max_length=80)

@app.get("/")
def root():
    return {"message": "EduSmart AI API is running", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ai_configured": bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/api/attempts")
def save_attempt(a: Attempt):
    now = a.date or datetime.datetime.now(datetime.timezone.utc).isoformat()
    with db() as c:
        cur = c.execute("INSERT INTO attempts(student_name,course,subject,correct,total,score,created_at) VALUES(?,?,?,?,?,?,?)",
            (a.student_name, a.course, a.subject, a.correct, a.total, a.score, now))
        new_id = cur.lastrowid
    return {"saved": True, "id": new_id}

@app.get("/api/attempts")
def list_attempts(student_name: str = "Student", limit: int = 100):
    limit = max(1, min(limit, 500))
    with db() as c:
        rows = c.execute("SELECT id,student_name,course,subject,correct,total,score,created_at FROM attempts WHERE student_name=? ORDER BY id DESC LIMIT ?", (student_name, limit)).fetchall()
    return {"attempts": [dict(r) for r in rows]}

@app.post("/api/profile")
def save_profile(p: Profile):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with db() as c:
        c.execute("""INSERT INTO profiles(student_name,course,updated_at) VALUES(?,?,?)
          ON CONFLICT(student_name) DO UPDATE SET course=excluded.course, updated_at=excluded.updated_at""",
          (p.name, p.course, now))
    return {"saved": True, "profile": p.model_dump()}

@app.get("/api/profile")
def get_profile(name: str = "Student"):
    with db() as c:
        row = c.execute("SELECT student_name,course,updated_at FROM profiles WHERE student_name=?", (name,)).fetchone()
    return {"profile": dict(row) if row else None}

@app.post("/api/tutor")
async def tutor(req: TutorRequest):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return {"answer": demo_answer(req.question), "mode": "demo",
                "note": "Set OPENAI_API_KEY in the backend hosting environment to enable live AI."}
    # Uses the OpenAI Responses API. API key stays on the backend and is never sent to the browser.
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                      "instructions": "You are EduSmart, a friendly tutor for college and school learners. Explain clearly, step by step, with simple examples. Encourage learning. If uncertain, say so. Do not do harmful or unrelated requests.",
                      "input": req.question}
            )
        response.raise_for_status()
        data = response.json()
        answer = data.get("output_text")
        if not answer:
            bits = []
            for item in data.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") == "output_text": bits.append(part.get("text", ""))
            answer = "\n".join(bits).strip()
        return {"answer": answer or "I couldn't create an answer. Please try asking in a different way.", "mode": "live-ai"}
    except Exception:
        return {"answer": "The live AI service is temporarily unavailable. Check the server's AI API configuration and try again.", "mode": "error"}

def demo_answer(q: str) -> str:
    s = q.lower()
    if "photosynthesis" in s:
        return "Photosynthesis is how green plants make food. Leaves use sunlight, water, and carbon dioxide to make sugar and release oxygen. Think of a leaf as a tiny solar-powered kitchen!"
    if "fraction" in s:
        return "A fraction represents part of a whole. In 3/4, the denominator 4 means the whole is split into four equal parts; the numerator 3 means you have three of those parts."
    if "english" in s:
        return "Try a daily 15-minute routine: read a short paragraph aloud, note three new words, write three sentences using them, and listen to clear English audio."
    return "Demo tutor: split the question into smaller parts, identify the main concept, and work through one example. For live AI explanations, configure OPENAI_API_KEY in the backend host's environment variables."
