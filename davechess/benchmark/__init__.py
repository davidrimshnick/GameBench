"""DaveChess benchmark: LLM evaluation pipeline.

Includes both the legacy static benchmark (show N games, measure ELO)
and the agentic benchmark (token budget, tool-use, autonomous learning).
"""

from davechess.benchmark.token_tracker import TokenTracker, TokenUsage
from davechess.benchmark.game_library import GameLibrary
from davechess.benchmark.opponent_pool import OpponentPool, CalibratedLevel
from davechess.benchmark.game_manager import GameManager
from davechess.benchmark.tools import ToolExecutor, ToolCall, TOOL_DEFINITIONS
from davechess.benchmark.sequential_eval import SequentialEvaluator, EvalConfig, EvalResult
from davechess.benchmark.agentic_protocol import AgenticBenchmarkRunner, AgenticRunResult

__all__ = [
    "TokenTracker",
    "TokenUsage",
    "GameLibrary",
    "OpponentPool",
    "CalibratedLevel",
    "GameManager",
    "ToolExecutor",
    "ToolCall",
    "TOOL_DEFINITIONS",
    "SequentialEvaluator",
    "EvalConfig",
    "EvalResult",
    "AgenticBenchmarkRunner",
    "AgenticRunResult",
]
