"""End-to-end dataset build script for judges and demo runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_ingestion import (
    aggregate_player_champion_stats,
    aggregate_player_vs_opponent_champion_stats,
    export_player_champion_matrix,
    flatten_matches,
    load_series_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed datasets.")
    parser.add_argument("--input", required=True, help="Path to raw series JSON")
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Minimum games threshold for aggregated stats",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("build_dataset")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw series JSON from %s", input_path)
    series = load_series_json(input_path)
    logger.info("Validated series count: %s", len(series))

    logger.info("Flattening matches into player-level rows")
    flat_df = flatten_matches(series)
    logger.info("Flat rows: %s", len(flat_df))

    logger.info("Aggregating player_champion_stats")
    player_champ_stats = aggregate_player_champion_stats(
        flat_df,
        output_csv=output_dir / "player_champion_stats.csv",
        output_parquet=output_dir / "player_champion_stats.parquet",
    )
    if args.min_games > 1:
        before = len(player_champ_stats)
        player_champ_stats = player_champ_stats[
            player_champ_stats["games_played"] >= args.min_games
        ].reset_index(drop=True)
        logger.info(
            "Filtered player_champion_stats by min-games=%s: %s -> %s",
            args.min_games,
            before,
            len(player_champ_stats),
        )
        player_champ_stats.to_csv(output_dir / "player_champion_stats.csv", index=False)
        player_champ_stats.to_parquet(
            output_dir / "player_champion_stats.parquet", index=False
        )

    logger.info("Aggregating player_vs_opponent_champion_stats")
    player_vs_opponent = aggregate_player_vs_opponent_champion_stats(
        flat_df,
        output_csv=output_dir / "player_vs_opponent_champion_stats.csv",
        output_parquet=output_dir / "player_vs_opponent_champion_stats.parquet",
    )
    if args.min_games > 1:
        before = len(player_vs_opponent)
        player_vs_opponent = player_vs_opponent[
            player_vs_opponent["games_played"] >= args.min_games
        ].reset_index(drop=True)
        logger.info(
            "Filtered player_vs_opponent_champion_stats by min-games=%s: %s -> %s",
            args.min_games,
            before,
            len(player_vs_opponent),
        )
        player_vs_opponent.to_csv(
            output_dir / "player_vs_opponent_champion_stats.csv", index=False
        )
        player_vs_opponent.to_parquet(
            output_dir / "player_vs_opponent_champion_stats.parquet", index=False
        )

    logger.info("Exporting player×champion matrix and encoders")
    export_player_champion_matrix(
        player_champ_stats, output_dir=output_dir
    )
    logger.info("Dataset build complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
