"""FastAPI application for the SHL Assessment Recommender."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import agent
import retriever
from embedder import get_embedder
from models import ChatRequest, ChatResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading embedder and ChromaDB index...")
    get_embedder()
    retriever.init_retriever()
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY is not set. Chat requests will return a fallback error until configured."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    message_dicts = [m.model_dump() for m in body.messages]
    result = agent.run(message_dicts)
    return ChatResponse(**result)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "Internal error",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
