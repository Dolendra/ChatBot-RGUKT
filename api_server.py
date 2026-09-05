"""
HTTP API for the RAG chatbot. Run from the project root:

    uvicorn api_server:app --reload --host 127.0.0.1 --port 8000

Then start the Vite UI: cd frontend && npm run dev
"""

import logging
import os
import traceback
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field, field_validator

from project_env import (
    PROJECT_ROOT,
    ensure_dotenv,
    get_app_subtitle,
    get_app_title,
    groq_key_configured,
)

ensure_dotenv()

app = FastAPI(title="RGUKT Academic Assistant API", version="1.1.0")

_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(3, ge=1, le=12)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=32)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        return cleaned


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    metrics: dict = Field(default_factory=dict)


class AppConfigResponse(BaseModel):
    app_title: str
    app_subtitle: str


_USER_UNAVAILABLE = "The assistant is temporarily unavailable. Please try again in a moment."
_USER_CONFIG = "The assistant is not configured yet. Please try again later."
_USER_BUSY = "The assistant is busy right now. Please wait a moment and try again."
_USER_TIMEOUT = "The request took too long. Please try again."
_USER_BAD_REQUEST = "Please check your message and try again."


@app.get("/")
def root():
    """Browser-friendly landing for the API port (UI runs on Vite :5173)."""
    return {
        "service": "RGUKT Academic Assistant API",
        "status": "ok",
        "ui": "http://127.0.0.1:5173/",
        "docs": "/docs",
        "health": "/api/health",
        "chat": "POST /api/chat",
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.getLogger("uvicorn.error").info("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": _USER_BAD_REQUEST})


@app.get("/api/config", response_model=AppConfigResponse)
def get_app_config():
    return AppConfigResponse(app_title=get_app_title(), app_subtitle=get_app_subtitle())


@app.get("/api/health")
def health():
    index_ok = os.path.exists("faiss_index.index") and os.path.exists("embeddings.pkl")
    return {
        "ok": True,
        "groq_key_present": groq_key_configured(),
        "index_present": index_ok,
    }


@app.get("/api/evaluate")
def get_evaluate():
    import json

    results_path = os.path.join("data", "evaluation_results.json")
    if not os.path.exists(results_path):
        return {"has_run": False, "msg": "Evaluation has not been run yet."}
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/evaluate/run")
def post_evaluate_run():
    from evaluate_rag import run_evaluation

    summary = run_evaluation()
    return summary


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest):
    log = logging.getLogger("uvicorn.error")
    try:
        from chat import run_rag

        hist = [h.model_dump() for h in body.history[-32:]]
        # Do not log message content (privacy)
        log.info("POST /api/chat top_k=%s history_turns=%s", body.top_k, len(hist))
        answer, sources, metrics = run_rag(body.message, top_k=body.top_k, history=hist or None)
        return ChatResponse(answer=answer or "", sources=sources, metrics=metrics)
    except FileNotFoundError as e:
        log.error("Search index missing: %s", e)
        raise HTTPException(status_code=503, detail=_USER_UNAVAILABLE) from e
    except AuthenticationError as e:
        log.warning("Groq authentication failed")
        raise HTTPException(status_code=503, detail=_USER_CONFIG) from e
    except RateLimitError as e:
        log.warning("Groq rate limit")
        raise HTTPException(status_code=503, detail=_USER_BUSY) from e
    except APITimeoutError as e:
        log.warning("Groq timeout")
        raise HTTPException(status_code=504, detail=_USER_TIMEOUT) from e
    except APIConnectionError as e:
        log.warning("Groq connection error: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail=_USER_UNAVAILABLE) from e
    except APIStatusError as e:
        log.warning("Groq API status error: %s", getattr(e, "status_code", "?"))
        raise HTTPException(status_code=503, detail=_USER_UNAVAILABLE) from e
    except RuntimeError as e:
        if "GROQ_API_KEY" in str(e):
            log.warning("Missing Groq API key")
            raise HTTPException(status_code=503, detail=_USER_CONFIG) from e
        log.error("POST /api/chat RuntimeError:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=_USER_UNAVAILABLE) from e
    except Exception:
        log.error("POST /api/chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=_USER_UNAVAILABLE) from e
