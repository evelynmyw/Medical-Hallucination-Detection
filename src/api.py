"""FastAPI service exposing the hallucination detector.

Run locally (from repo root):
    MODEL_DIR=/path/to/seed44 uvicorn src.api:app --port 8000
Then:
    curl -X POST localhost:8000/detect -H "Content-Type: application/json" \
         -d '{"question": "Does X reduce Y?", "answer": "X significantly reduces Y."}'

MODEL_DIR must contain best/, config.json and threshold.json (one training seed).
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, StringConstraints

# Whitespace-only input is rejected (422) instead of being sent to the model.
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Answer = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class DetectRequest(BaseModel):
    question: Question
    answer: Answer


class DetectResponse(BaseModel):
    label: Literal["hallucinated", "ground_truth"]
    p_hallucinated: float
    threshold: float


def create_app(detector=None) -> FastAPI:
    """Build the app. Tests pass a fake detector; in production it is loaded from MODEL_DIR."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.detector is None:
            model_dir = os.environ.get("MODEL_DIR")
            if not model_dir:
                raise RuntimeError("Set MODEL_DIR to a seed directory with best/ and threshold.json")
            from src.predict import HallucinationDetector  # loads torch only when serving
            app.state.detector = HallucinationDetector(model_dir)
        yield  # model is loaded once at startup, not per request

    app = FastAPI(title="MedHallu hallucination detector", version="1.0", lifespan=lifespan)
    app.state.detector = detector

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # Plain `def` (not async): FastAPI runs it in a thread pool, so CPU-bound
    # inference does not block the event loop.
    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest, request: Request) -> dict:
        return request.app.state.detector.predict(req.question, req.answer)

    return app


app = create_app()
