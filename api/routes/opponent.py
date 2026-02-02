"""Opponent-related HTTP endpoints."""

from fastapi import APIRouter

router: APIRouter = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Health check for opponent endpoints."""
    return {"status": "ok"}
