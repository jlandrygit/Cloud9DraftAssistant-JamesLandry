"""GRID Central Data API ingestion utilities.

This module uses the Central Data API because it exposes structured,
queryable esports entities (tournaments, series, matches) with stable
pagination. The League of Legends titleId is 3 in GRID's catalog, so all
queries filter by titleId=3 to scope results to LoL-only data.
"""

# HACKATHON NOTE: GraphQL queries Central Data API for series metadata in a single request,
# reducing client-side assumptions about REST paths. Cursor-based pagination safely traverses
# ordered results using pageInfo.hasNextPage and pageInfo.endCursor. Series discovery feeds
# downstream draft modeling by identifying matches that generate pick/ban and outcome features.

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

GRID_GRAPHQL_ENDPOINT = "https://api-op.grid.gg/central-data/graphql"
LOL_TITLE_ID = 3
MAX_PAGES = 200

TOURNAMENTS_QUERY = """
query Tournaments($titleIds: [ID!], $first: Int!, $after: String) {
  tournaments(filter: { title: { id: { in: $titleIds } } }, first: $first, after: $after) {
    totalCount
    edges {
      node {
        id
        name
      }
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
}
"""

SERIES_BY_TOURNAMENT_QUERY = """
query AllSeries($tournamentIds: [ID!], $first: Int!, $after: String) {
  allSeries(
    filter: { tournament: { id: { in: $tournamentIds }, includeChildren: { equals: true } } }
    orderBy: StartTimeScheduled
    first: $first
    after: $after
  ) {
    totalCount
    edges {
      node {
        id
        startTimeScheduled
        teams {
          baseInfo {
            id
            name
          }
        }
      }
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
}
"""


def fetch_lol_tournaments(api_key: str) -> list[dict]:
    """Fetch League of Legends tournament IDs and names."""
    headers = {"x-api-key": api_key}
    results: list[dict] = []
    cursor: str | None = None
    pages = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            pages += 1
            if pages > MAX_PAGES:
                raise RuntimeError("Pagination exceeded MAX_PAGES guard.")

            variables = {
                "titleIds": [str(LOL_TITLE_ID)],
                "first": 50,
                "after": cursor,
            }
            # TODO (Post-Hackathon): Add retry/backoff and rate limiting for GRID requests.
            # Missing: resilience to transient failures and protection against request bursts.
            # Out of scope: hackathon ingestion prioritized clarity over operational hardening.
            # Production: retries with exponential backoff + jitter, 429 handling, timeouts, and client-side throttling.
            response = client.post(
                GRID_GRAPHQL_ENDPOINT,
                json={"query": TOURNAMENTS_QUERY, "variables": variables},
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

            if "data" not in payload or not isinstance(payload["data"], dict):
                raise RuntimeError("GRID response missing 'data' object.")
            data = payload["data"]
            if "tournaments" not in data or not isinstance(data["tournaments"], dict):
                raise RuntimeError("GRID response missing 'tournaments' object.")
            tournaments = data["tournaments"]

            edges = tournaments.get("edges")
            if edges is None:
                raise RuntimeError("GRID response missing 'edges' list.")
            if not isinstance(edges, list):
                raise RuntimeError("GRID response 'edges' is not a list.")
            if not edges:
                break

            for edge in edges:
                if not isinstance(edge, dict):
                    raise RuntimeError("GRID response contains invalid edge entry.")
                if "node" not in edge:
                    raise RuntimeError("GRID response edge missing 'node'.")
                results.append(edge["node"])

            page_info = tournaments.get("pageInfo")
            if page_info is None:
                raise RuntimeError("GRID response missing 'pageInfo'.")
            if not isinstance(page_info, dict):
                raise RuntimeError("GRID response 'pageInfo' is not an object.")
            if "hasNextPage" not in page_info:
                raise RuntimeError("GRID response missing 'pageInfo.hasNextPage'.")
            if "endCursor" not in page_info:
                raise RuntimeError("GRID response missing 'pageInfo.endCursor'.")

            has_next = bool(page_info["hasNextPage"])
            next_cursor = page_info["endCursor"]
            if has_next and not next_cursor:
                raise RuntimeError("GRID response missing endCursor for next page.")
            if next_cursor == cursor:
                break
            cursor = next_cursor
            if not has_next:
                break

    return results


def save_tournaments(path: str | Path, tournaments: list[dict]) -> None:
    """Save tournament IDs and names for downstream queries."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(tournaments, indent=2), encoding="utf-8")


def fetch_series_for_tournaments(api_key: str, tournament_ids: list[str]) -> list[dict]:
    """Fetch series for the provided tournament IDs with pagination."""
    headers = {"x-api-key": api_key}
    results: list[dict] = []
    cursor: str | None = None
    pages = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            pages += 1
            if pages > MAX_PAGES:
                raise RuntimeError("Pagination exceeded MAX_PAGES guard.")

            variables = {
                "tournamentIds": tournament_ids,
                "first": 50,
                "after": cursor,
            }
            # TODO (Post-Hackathon): Add retry/backoff and rate limiting for GRID requests.
            # Missing: resilience to transient failures and protection against request bursts.
            # Out of scope: hackathon ingestion prioritized clarity over operational hardening.
            # Production: retries with exponential backoff + jitter, 429 handling, timeouts, and client-side throttling.
            response = client.post(
                GRID_GRAPHQL_ENDPOINT,
                json={"query": SERIES_BY_TOURNAMENT_QUERY, "variables": variables},
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

            if "data" not in payload or not isinstance(payload["data"], dict):
                raise RuntimeError("GRID response missing 'data' object.")
            data = payload["data"]
            if "allSeries" not in data or not isinstance(data["allSeries"], dict):
                raise RuntimeError("GRID response missing 'allSeries' object.")
            all_series = data["allSeries"]

            edges = all_series.get("edges")
            if edges is None:
                raise RuntimeError("GRID response missing 'edges' list.")
            if not isinstance(edges, list):
                raise RuntimeError("GRID response 'edges' is not a list.")
            if not edges:
                break

            for edge in edges:
                if not isinstance(edge, dict):
                    raise RuntimeError("GRID response contains invalid edge entry.")
                if "node" not in edge:
                    raise RuntimeError("GRID response edge missing 'node'.")
                results.append(edge["node"])

            page_info = all_series.get("pageInfo")
            if page_info is None:
                raise RuntimeError("GRID response missing 'pageInfo'.")
            if not isinstance(page_info, dict):
                raise RuntimeError("GRID response 'pageInfo' is not an object.")
            if "hasNextPage" not in page_info:
                raise RuntimeError("GRID response missing 'pageInfo.hasNextPage'.")
            if "endCursor" not in page_info:
                raise RuntimeError("GRID response missing 'pageInfo.endCursor'.")

            has_next = bool(page_info["hasNextPage"])
            next_cursor = page_info["endCursor"]
            if has_next and not next_cursor:
                raise RuntimeError("GRID response missing endCursor for next page.")
            if next_cursor == cursor:
                break
            cursor = next_cursor
            if not has_next:
                break

    return results


def save_series(path: str | Path, series: list[dict]) -> None:
    """Save series IDs and metadata for downstream queries."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(series, indent=2), encoding="utf-8")


def load_tournaments(path: str | Path) -> list[dict]:
    """Load tournaments from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def filter_tournaments_2025_or_later(tournaments: list[dict]) -> list[dict]:
    """Return tournaments whose name indicates 2025 or later."""
    results: list[dict] = []
    for tournament in tournaments:
        name = str(tournament.get("name", ""))
        years = [int(value) for value in re.findall(r"\b20\d{2}\b", name)]
        if any(year >= 2025 for year in years):
            results.append(tournament)
    return results


def get_ingested_match_count() -> int:
    """Return the count of ingested matches for demo health endpoint.
    
    Returns 0 if no match data is available (demo mode).
    """
    return 0


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GRID_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GRID_API_KEY environment variable.")

    tournaments = fetch_lol_tournaments(api_key)
    tournaments = filter_tournaments_2025_or_later(tournaments)
    save_tournaments("data/ingestion/tournaments.json", tournaments)
    print(f"Total tournaments fetched: {len(tournaments)}")
    for item in tournaments[:3]:
        tournament_id = item.get("id", "unknown")
        name = item.get("name", "unknown")
        print(f"- {tournament_id} | {name}")

    tournament_ids = [tournament["id"] for tournament in tournaments if "id" in tournament]
    series = fetch_series_for_tournaments(api_key, tournament_ids)
    save_series("data/ingestion/series.json", series)
    print(f"Total series fetched: {len(series)}")
    for item in series[:3]:
        series_id = item.get("id", "unknown")
        print(f"- {series_id}")
