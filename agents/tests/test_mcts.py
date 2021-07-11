import numpy as np
from agents.new_agent import mcts_agent
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2
from agents.common import pretty_print_board, string_to_board, initialize_game_state, apply_player_action


def test_find_remaining_actions():
    """
    Finds whether the function returns the valid columns = [0,2,4,5,6] after 1 and 3 columns
    are fully filled with 1
    """
    test_board = initialize_game_state()
    test_board[:,1] = 1
    test_board[:, 3] = 1
    test_node = mcts_agent.MonteCarloTreeSearchNode(state=test_board,player = PLAYER2)
    test_remain_actions = test_node.find_remaining_actions()
    actual_actions_remaining =[0,2,4,5,6]
    assert (test_remain_actions == actual_actions_remaining)

def expand():
