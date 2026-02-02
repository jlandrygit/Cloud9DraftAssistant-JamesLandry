"""Load and validate series JSON data for downstream transformations.

NOTE: This file is kept because it's actively used by scripts/build_dataset.py.
While it could be moved to data/ingestion/, it's kept at root level for
script convenience. Consider consolidating into data/ingestion/ in future refactoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)


class Character(BaseModel):
    """Champion selection for a player."""

    name: str

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value:
            raise ValueError("character.name must be non-empty")
        return value


class Player(BaseModel):
    """Player with selected champion."""

    name: str
    character: Character

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value:
            raise ValueError("player.name must be non-empty")
        return value


class Team(BaseModel):
    """Team of players in a game."""

    name: str | None = None
    players: list[Player]

    @field_validator("players")
    @classmethod
    def _validate_players(cls, value: list[Player]) -> list[Player]:
        if len(value) != 5:
            raise ValueError("team must have exactly 5 players")
        return value


class SeriesTeam(BaseModel):
    """Series-level team metadata (no players at this level)."""

    name: str
    won: bool | None = None


class Game(BaseModel):
    """Single game within a series."""

    teams: list[Team]

    @field_validator("teams")
    @classmethod
    def _validate_teams(cls, value: list[Team]) -> list[Team]:
        if len(value) != 2:
            raise ValueError("game must have exactly 2 teams")
        return value


class Series(BaseModel):
    """Series containing games and team metadata."""

    id: str
    teams: list[SeriesTeam]
    games: list[Game]
    winner: str | None = None
    startTimeScheduled: str | None = None

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, value: str) -> str:
        if not value:
            raise ValueError("series.id must be non-empty")
        return value


def load_series_json(path: str | Path) -> list[Series]:
    """Load and validate a list of series from a JSON file.

    Malformed entries are logged and skipped to keep ingestion resilient.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a list of series objects in JSON.")

    validated: list[Series] = []
    for idx, raw in enumerate(payload):
        try:
            validated.append(Series.model_validate(raw))
        except ValidationError as exc:
            logger.warning("Skipping invalid series at index %s: %s", idx, exc)
    return validated


def flatten_matches(
    series_list: list[Series],
    *,
    champion_roles_path: str | Path | None = None,
) -> "pd.DataFrame":
    """Flatten series into one row per player per game.

    If champion_roles_path is provided, roles are inferred and added to the
    output DataFrame.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for series in series_list:
        winner_name = _series_winner_name(series)
        series_date = series.startTimeScheduled
        for game_index, game in enumerate(series.games):
            if len(game.teams) != 2:
                continue
            team_a, team_b = game.teams
            rows.extend(
                _flatten_team_rows(
                    series_id=series.id,
                    game_index=game_index,
                    team=team_a,
                    opponent=team_b,
                    winner_name=winner_name,
                    match_date=series_date,
                )
            )
            rows.extend(
                _flatten_team_rows(
                    series_id=series.id,
                    game_index=game_index,
                    team=team_b,
                    opponent=team_a,
                    winner_name=winner_name,
                    match_date=series_date,
                )
            )

    df = pd.DataFrame(rows)
    df = _set_column_types(df)
    if champion_roles_path:
        df = add_roles(df, champion_roles_path)
    return df


def _flatten_team_rows(
    *,
    series_id: str,
    game_index: int,
    team: Team,
    opponent: Team,
    winner_name: str | None,
    match_date: str | None,
) -> list[dict[str, Any]]:
    """Build rows for all players on a team."""
    rows: list[dict[str, Any]] = []
    for player in team.players:
        rows.append(
            {
                "series_id": series_id,
                "game_index": game_index,
                "team_name": team.name or "",
                "opponent_team_name": opponent.name or "",
                "player_name": player.name,
                "champion_name": player.character.name,
                "game_win": _resolve_game_win(team.name, winner_name),
                "match_date": match_date or "",
            }
        )
    return rows


def _resolve_game_win(team_name: str | None, winner_name: str | None) -> bool:
    """Resolve game win from series winner name."""
    if not team_name or not winner_name:
        return False
    return team_name == winner_name


def _series_winner_name(series: Series) -> str | None:
    """Resolve series winner name from series-level teams metadata."""
    if series.winner:
        return series.winner
    for team in series.teams:
        if team.won:
            return team.name
    return None


def _set_column_types(df: "pd.DataFrame") -> "pd.DataFrame":
    """Set explicit column types for downstream modeling."""
    import pandas as pd

    if df.empty:
        return df
    df["series_id"] = df["series_id"].astype("string")
    df["game_index"] = df["game_index"].astype("int64")
    df["team_name"] = df["team_name"].astype("string")
    df["opponent_team_name"] = df["opponent_team_name"].astype("string")
    df["player_name"] = df["player_name"].astype("string")
    df["champion_name"] = df["champion_name"].astype("string")
    df["game_win"] = df["game_win"].astype("bool")
    return df


def aggregate_player_champion_stats(
    df: "pd.DataFrame",
    *,
    half_life_days: int = 90,
    output_csv: str | Path = "data/processed/player_champion_stats.csv",
    output_parquet: str | Path = "data/processed/player_champion_stats.parquet",
) -> "pd.DataFrame":
    """Aggregate flat rows into player/champion/role comfort stats."""
    import pandas as pd

    _ensure_dataframe_columns(
        df,
        ["player_name", "champion_name", "game_win"],
    )

    result = df.copy()
    if "role" not in result.columns:
        result["role"] = "UNKNOWN"
    if "match_date" not in result.columns:
        result["match_date"] = ""
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    if result["match_date"].isna().all():
        result["recency_weight"] = 1.0
    else:
        reference_date = result["match_date"].max()
        age_days = (reference_date - result["match_date"]).dt.days
        result["recency_weight"] = 0.5 ** (age_days / float(half_life_days))

    grouped = (
        result.groupby(["player_name", "champion_name", "role"], as_index=False)
        .agg(
            games_played=("champion_name", "size"),
            wins=("game_win", "sum"),
            win_rate=("game_win", "mean"),
            recency_weighted_games=("recency_weight", "sum"),
        )
        .sort_values("games_played", ascending=False)
        .reset_index(drop=True)
    )

    _write_dataframe(grouped, output_csv, output_parquet)
    return grouped


def aggregate_player_vs_opponent_champion_stats(
    df: "pd.DataFrame",
    *,
    output_csv: str | Path = "data/processed/player_vs_opponent_champion_stats.csv",
    output_parquet: str | Path = "data/processed/player_vs_opponent_champion_stats.parquet",
) -> "pd.DataFrame":
    """Aggregate player vs opponent champion stats for ban/punish modeling."""
    import pandas as pd

    _ensure_dataframe_columns(
        df,
        ["player_name", "opponent_team_name", "champion_name", "game_win"],
    )

    grouped = (
        df.groupby(["player_name", "opponent_team_name", "champion_name"], as_index=False)
        .agg(
            games_played=("champion_name", "size"),
            wins=("game_win", "sum"),
            win_rate=("game_win", "mean"),
        )
        .sort_values("games_played", ascending=False)
        .reset_index(drop=True)
    )

    _write_dataframe(grouped, output_csv, output_parquet)
    return grouped


def export_player_champion_matrix(
    df: "pd.DataFrame",
    *,
    output_dir: str | Path = "models",
    value_column: str = "games_played",
) -> None:
    """Export sparse player×champion matrix and encoders for modeling."""
    import pandas as pd
    from scipy import sparse
    from sklearn.preprocessing import LabelEncoder

    _ensure_dataframe_columns(df, ["player_name", "champion_name", value_column])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    players = sorted(df["player_name"].dropna().unique().tolist())
    champions = sorted(df["champion_name"].dropna().unique().tolist())

    player_encoder = LabelEncoder()
    champion_encoder = LabelEncoder()
    player_encoder.fit(players)
    champion_encoder.fit(champions)

    row_idx = player_encoder.transform(df["player_name"])
    col_idx = champion_encoder.transform(df["champion_name"])
    data = df[value_column].astype(float).to_numpy()

    matrix = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(players), len(champions)),
    )

    # Deterministic ordering ensures reproducible matrix construction.
    joblib.dump(player_encoder, output_path / "player_encoder.pkl")
    joblib.dump(champion_encoder, output_path / "champion_encoder.pkl")
    sparse.save_npz(output_path / "player_champion_matrix.npz", matrix)


def _write_dataframe(df: "pd.DataFrame", csv_path: str | Path, parquet_path: str | Path) -> None:
    """Persist DataFrame to CSV and Parquet."""
    import pandas as pd

    csv_target = Path(csv_path)
    parquet_target = Path(parquet_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    parquet_target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_target, index=False)
    df.to_parquet(parquet_target, index=False)


def _ensure_dataframe_columns(df: "pd.DataFrame", columns: list[str]) -> None:
    """Raise ValueError when required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_roles(df: "pd.DataFrame", champion_roles_path: str | Path) -> "pd.DataFrame":
    """Infer player roles using a champion->roles mapping."""
    import pandas as pd

    if df.empty:
        return df
    role_map = load_champion_roles(champion_roles_path)
    result = df.copy()
    result["role"] = ""

    grouped = result.groupby(["series_id", "game_index", "team_name"], sort=False)
    for (series_id, game_index, team_name), group in grouped:
        roles = _infer_team_roles(group, role_map)
        for idx, role in roles.items():
            result.at[idx, "role"] = role

        unresolved = [idx for idx, role in roles.items() if not role]
        if unresolved:
            logger.warning(
                "Unresolved roles for series=%s game=%s team=%s players=%s",
                series_id,
                game_index,
                team_name,
                result.loc[unresolved, "player_name"].tolist(),
            )

    result["role"] = result["role"].astype("string")
    return result


def load_champion_roles(path: str | Path) -> dict[str, list[str]]:
    """Load champion role mappings from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Champion role mapping must be a JSON object.")
    role_map: dict[str, list[str]] = {}
    for champion, roles in payload.items():
        if isinstance(roles, str):
            role_map[champion] = [roles]
        elif isinstance(roles, list):
            role_map[champion] = [str(role) for role in roles]
        else:
            raise ValueError(f"Invalid roles for champion {champion}.")
    return role_map


def _infer_team_roles(
    team_df: "pd.DataFrame", role_map: dict[str, list[str]]
) -> dict[int, str]:
    """Infer team roles using elimination when multiple roles exist."""
    candidates: dict[int, set[str]] = {}
    for idx, row in team_df.iterrows():
        champion = str(row["champion_name"])
        roles = role_map.get(champion, [])
        candidates[idx] = set(roles)

    assigned: dict[int, str] = {idx: "" for idx in candidates}
    remaining_roles = {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}

    # Assign single-role champions first, then eliminate until stable.
    progress = True
    while progress:
        progress = False
        for idx, options in list(candidates.items()):
            if assigned[idx]:
                continue
            if len(options) == 1:
                role = next(iter(options))
                assigned[idx] = role
                remaining_roles.discard(role)
                progress = True
        for idx, options in list(candidates.items()):
            if assigned[idx]:
                continue
            filtered = options & remaining_roles
            if filtered != options:
                candidates[idx] = filtered
                progress = True

    # Fill any still-unassigned entries with empty string.
    return assigned
