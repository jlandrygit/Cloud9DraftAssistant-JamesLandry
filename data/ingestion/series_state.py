"""Ingest series state data (players/champions) from GRID Live Data Feed."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

GRID_LIVE_ENDPOINT = "https://api-op.grid.gg/live-data-feed/series-state/graphql"

SERIES_STATE_QUERY = """
query SeriesState($id: ID!) {
  seriesState(id: $id) {
    id
    teams {
      name
      won
    }
    games {
      teams {
        name
        players {
          name
          character {
            name
          }
        }
      }
    }
  }
}
"""


def load_series(path: str | Path) -> list[dict[str, Any]]:
    """Load series IDs from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch_series_state(api_key: str, series_id: str) -> dict[str, Any] | None:
    """Fetch series state for a single series ID."""
    headers = {"x-api-key": api_key}
    variables = {"id": series_id}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            GRID_LIVE_ENDPOINT,
            json={"query": SERIES_STATE_QUERY, "variables": variables},
            headers=headers,
        )
        if response.status_code != 200:
            raise RuntimeError(f"GRID API HTTP {response.status_code}: {response.text}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("GRID API returned invalid JSON.") from exc
        if "errors" in payload:
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        data = payload.get("data")
        if data is None or not isinstance(data, dict):
            raise RuntimeError("GRID response missing 'data' object.")
        series_state = data.get("seriesState")
        if series_state is None:
            # Some series IDs may not have live data; skip them gracefully.
            return None
        return series_state


def save_series_states(path: str | Path, series_states: list[dict[str, Any]]) -> None:
    """Save series state data to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(series_states, indent=2), encoding="utf-8")


def ingest_series_states(
    api_key: str,
    series_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch series state data for all series IDs and save to disk."""
    series = load_series(series_path)
    series_ids = [item["id"] for item in series if "id" in item]
    if limit is not None:
        series_ids = series_ids[:limit]

    results: list[dict[str, Any]] = []
    for series_id in series_ids:
        series_state = fetch_series_state(api_key, series_id)
        if series_state is not None:
            results.append(series_state)

    save_series_states(output_path, results)
    return results


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GRID_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GRID_API_KEY environment variable.")

    ingest_series_states(
        api_key=api_key,
        series_path="data/ingestion/series.json",
        output_path="data/ingestion/series_states.json",
    )
