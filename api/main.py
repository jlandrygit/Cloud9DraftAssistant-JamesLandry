"""FastAPI entry point for the C9 Draft Assistant service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from api.routes import draft, opponent
from core.config import MODEL_PATCH_END, MODEL_PATCH_START
from data.ingestion.grid_api import get_ingested_match_count

app: FastAPI = FastAPI(title="C9 Draft Assistant API")
app.include_router(draft.router, prefix="/draft", tags=["draft"])
app.include_router(opponent.router, prefix="/opponent", tags=["opponent"])


@app.get("/health")
def health_check() -> dict[str, str]:
    """Service-level health check endpoint."""
    return {"status": "ok"}


@app.get("/health/demo")
def demo_health() -> dict[str, Any]:
    """Demo transparency endpoint with runtime stats."""
    # TODO (Post-Hackathon): Source the active model patch range from a model registry.
    # Missing: authoritative metadata for the currently loaded model (patch start/end).
    # Out of scope: hackathon demo used env vars to avoid building a registry/config service.
    # Production: register model artifacts with versioned metadata and expose it via config/health.
    return {
        "active_sessions": draft.get_active_session_count(),
        "session_ttl_seconds": draft.get_session_ttl_seconds(),
        "grid_matches_ingested": get_ingested_match_count(),
        "model_patch_range": {"start": MODEL_PATCH_START, "end": MODEL_PATCH_END},
    }
