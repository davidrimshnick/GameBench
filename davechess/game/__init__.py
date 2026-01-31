"""DaveChess game engine: state, rules, board, notation."""

from davechess.game.state import GameState, Player, PieceType, Piece
from davechess.game.rules import generate_legal_moves, apply_move, check_winner, get_resource_income
from davechess.game.board import RESOURCE_NODES, STARTING_POSITIONS, render_board
from davechess.game.notation import move_to_dcn, dcn_to_move, game_to_dcn, dcn_to_game

__all__ = [
    "GameState", "Player", "PieceType", "Piece",
    "generate_legal_moves", "apply_move", "check_winner", "get_resource_income",
    "RESOURCE_NODES", "STARTING_POSITIONS", "render_board",
    "move_to_dcn", "dcn_to_move", "game_to_dcn", "dcn_to_game",
]
