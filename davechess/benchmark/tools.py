"""Tool definitions and execution for the agentic benchmark."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from davechess.benchmark.game_manager import GameManager
from davechess.benchmark.game_library import GameLibrary

logger = logging.getLogger("davechess.benchmark")

# Tool definitions in OpenAI function-calling schema
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "study_games",
            "description": (
                "Retrieve N grandmaster-level DaveChess games from the game "
                "library for study. Games are returned in DCN (DaveChess "
                "Notation). Each call returns different games."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of games to retrieve (1-20 per call)",
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_practice_game",
            "description": (
                "Start a new practice game against an opponent at the "
                "specified ELO rating. Returns the game ID and initial "
                "board state with your legal moves."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "opponent_elo": {
                        "type": "integer",
                        "description": "Target ELO for opponent (400-2700)",
                        "minimum": 400,
                        "maximum": 2700,
                    }
                },
                "required": ["opponent_elo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_move",
            "description": (
                "Make a move in an active game. The move must be in DCN "
                "notation (e.g., 'Wc1-c2', 'Rb1xd3', '+W@c2', 'Bc3~e3'). "
                "After your move, the opponent responds immediately and the "
                "updated board state is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "The game ID from start_practice_game",
                    },
                    "move_dcn": {
                        "type": "string",
                        "description": "Your move in DCN notation",
                    },
                },
                "required": ["game_id", "move_dcn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_state",
            "description": (
                "Get the current state of a game, including the board "
                "position, legal moves, move history, and game status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "The game ID",
                    }
                },
                "required": ["game_id"],
            },
        },
    },
]


def tools_to_anthropic_format() -> list[dict]:
    """Convert tool definitions to Anthropic tool-use format."""
    anthropic_tools = []
    for tool in TOOL_DEFINITIONS:
        func = tool["function"]
        anthropic_tools.append({
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        })
    return anthropic_tools


@dataclass
class ToolCall:
    """A parsed tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecutor:
    """Executes tool calls against the game manager and library."""

    def __init__(self, game_manager: GameManager, game_library: GameLibrary):
        self.game_manager = game_manager
        self.game_library = game_library
        self.stats = {
            "games_studied": 0,
            "practice_games_started": 0,
            "moves_played": 0,
            "state_queries": 0,
        }

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a JSON string."""
        name = tool_call.name
        args = tool_call.arguments

        try:
            if name == "study_games":
                result = self._study_games(args)
            elif name == "start_practice_game":
                result = self._start_practice_game(args)
            elif name == "play_move":
                result = self._play_move(args)
            elif name == "get_game_state":
                result = self._get_game_state(args)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Tool execution error ({name}): {e}")
            result = {"error": str(e)}

        return json.dumps(result, indent=2)

    def _study_games(self, args: dict) -> dict:
        n = args.get("n", 1)
        n = max(1, min(20, n))
        try:
            games = self.game_library.get_games(n)
            self.stats["games_studied"] += n
            return {
                "games": games,
                "count": len(games),
                "remaining_in_library": self.game_library.remaining,
            }
        except ValueError as e:
            return {"error": str(e)}

    def _start_practice_game(self, args: dict) -> dict:
        opponent_elo = args.get("opponent_elo", 1000)
        opponent_elo = max(400, min(2700, opponent_elo))
        result = self.game_manager.start_game(opponent_elo)
        if "error" not in result:
            self.stats["practice_games_started"] += 1
        return result

    def _play_move(self, args: dict) -> dict:
        game_id = args.get("game_id", "")
        move_dcn = args.get("move_dcn", "")
        result = self.game_manager.play_move(game_id, move_dcn)
        if "error" not in result:
            self.stats["moves_played"] += 1
        return result

    def _get_game_state(self, args: dict) -> dict:
        game_id = args.get("game_id", "")
        self.stats["state_queries"] += 1
        return self.game_manager.get_state(game_id)
