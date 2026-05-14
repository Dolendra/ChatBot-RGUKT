"""
HTTP API for the RAG chatbot. Run from the project root:

    uvicorn api_server:app --reload --host 127.0.0.1 --port 8000

Then start the Vite UI: cd frontend && npm run dev
"""

import logging
import os
import traceback
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import AuthenticationError
from pydantic import BaseModel, Field

from project_env import (
    PROJECT_ROOT,
    ensure_dotenv,
    get_app_subtitle,
    get_app_title,
    groq_key_configured,
)

ensure_dotenv()

app = FastAPI(title="Organisation RAG API", version="1.0.0")

_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(3, ge=1, le=12)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=32)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class AppConfigResponse(BaseModel):
    app_title: str
    app_subtitle: str


@app.get("/api/config", response_model=AppConfigResponse)
def get_app_config():
    return AppConfigResponse(app_title=get_app_title(), app_subtitle=get_app_subtitle())


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "groq_key_present": groq_key_configured(),
        "env_file": str(PROJECT_ROOT / ".env"),
    }


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest):
    try:
        from chat import run_rag

        hist = [h.model_dump() for h in body.history[-32:]]
        answer, sources = run_rag(body.message.strip(), top_k=body.top_k, history=hist or None)
        return ChatResponse(answer=answer or "", sources=sources)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Search index not found. Run embed.py and vector_store.py from the project root.",
        ) from e
    except AuthenticationError as e:
        logging.getLogger("uvicorn.error").warning("Groq authentication failed: %s", e)
        raise HTTPException(
            status_code=401,
            detail=(
                "Groq rejected the API key (401 invalid_api_key). "
                "Create a new key at https://console.groq.com/keys (revoke any key you pasted in chat or logs). "
                "In `.env` use exactly: GROQ_API_KEY=gsk_... on one line, no spaces around '='. "
                "If you ever ran $env:GROQ_API_KEY=... in PowerShell, close that terminal or unset it — "
                "this app loads `.env` with override so the file should win after restart. "
                "Save `.env` as UTF-8 and restart uvicorn. GET /api/health → groq_key_present should be true."
            ),
        ) from e
    except RuntimeError as e:
        if "GROQ_API_KEY" in str(e):
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logging.getLogger("uvicorn.error").error("POST /api/chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e
