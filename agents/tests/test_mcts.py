import numpy as np
from agents.new_agent import mcts_agent
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2
from agents.common import pretty_print_board, string_to_board, initialize_game_state, apply_player_action


def test_untried_actions():
