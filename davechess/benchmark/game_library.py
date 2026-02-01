"""GM game library for agent study."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

from davechess.data.storage import load_game
from davechess.game.notation import game_to_dcn

logger = logging.getLogger("davechess.benchmark")


class GameLibrary:
    """Library of grandmaster-level games available for agent study.

    Games are loaded from DCN files and served without replacement.
    """

    def __init__(self, games_dir: str, max_games: int = 200):
        self.games_dir = games_dir
        self.max_games = max_games
        self.games: list[str] = []  # Pre-formatted DCN strings
        self.served_indices: set[int] = set()

    def load(self) -> int:
        """Load GM games from disk.

        Returns:
            Number of games loaded.
        """
        if not os.path.isdir(self.games_dir):
            logger.warning(f"Game library directory not found: {self.games_dir}")
            return 0

        dcn_files = sorted(
            f for f in os.listdir(self.games_dir) if f.endswith(".dcn")
        )

        for fname in dcn_files[: self.max_games]:
            path = os.path.join(self.games_dir, fname)
            try:
                headers, moves, result = load_game(path)
                # Format as a readable DCN game record
                dcn_text = _format_game_record(headers, moves, result)
                self.games.append(dcn_text)
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")

        logger.info(f"Loaded {len(self.games)} GM games from {self.games_dir}")
        return len(self.games)

    def get_games(self, n: int) -> list[str]:
        """Return N unserved games.

        Args:
            n: Number of games to retrieve.

        Returns:
            List of game records in DCN format.

        Raises:
            ValueError: If not enough unserved games remain.
        """
        available = [
            i for i in range(len(self.games)) if i not in self.served_indices
        ]
        if len(available) < n:
            raise ValueError(
                f"Requested {n} games but only {len(available)} remain unserved "
                f"(total: {len(self.games)}, served: {len(self.served_indices)})"
            )

        selected = random.sample(available, n)
        for idx in selected:
            self.served_indices.add(idx)

        return [self.games[i] for i in selected]

    @property
    def total_games(self) -> int:
        return len(self.games)

    @property
    def remaining(self) -> int:
        return len(self.games) - len(self.served_indices)

    def reset(self) -> None:
        """Reset served tracking (for new benchmark run)."""
        self.served_indices.clear()


def _format_game_record(
    headers: dict[str, str], moves: list, result: Optional[str]
) -> str:
    """Format a game record as readable DCN text."""
    lines = []
    for key, val in headers.items():
        lines.append(f'[{key} "{val}"]')
    lines.append("")

    # Format moves as numbered pairs
    for i in range(0, len(moves), 2):
        move_num = i // 2 + 1
        white_move = moves[i]
        if i + 1 < len(moves):
            black_move = moves[i + 1]
            lines.append(f"{move_num}. {white_move} {black_move}")
        else:
            lines.append(f"{move_num}. {white_move}")

    if result:
        lines.append(result)

    return "\n".join(lines)
