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
    test_board[:, 1] = 1
    test_board[:, 3] = 1
    test_node = mcts_agent.MonteCarloTreeSearchNode(state=test_board, player=PLAYER2)
    test_remain_actions = test_node.find_remaining_actions()
    actual_actions_remaining = [0, 2, 4, 5, 6]
    assert (test_remain_actions == actual_actions_remaining)


def test_expand():
    """
    Compares the child state in child node with a test_next_state
    and remaining actions before and after expansion
    of current state given by test board
    """
    test_board = initialize_game_state()
    test_board[0:2, 1] = 1
    test_board[0:3, 3] = 2
    test_board[:, 6] = 1
    test_node = mcts_agent.MonteCarloTreeSearchNode(state=test_board, player=PLAYER2)
    assert (len(test_node.remaining_actions) == 6) # remaining actions before expansion
    assert (test_node.remaining_actions[-1] == 5)
    test_action = test_node.remaining_actions[-1]
    test_child = test_node.expand()
    test_next_state = apply_player_action(test_board, test_action, PLAYER2)
    test_action = test_node.remaining_actions[-1] #after expansion
    assert (test_action == 4) # taking the last remaining action
    assert (test_child.state == test_next_state).all()


# def test_rollout():

def test_game_result():
    """
    Checks if the game_result_function returns 1,0,-1 based on win,draw or lose for the AI
    player
    """
    #Checking win condition
    test_board = initialize_game_state()
    test_board[0:2, 1] = 1
    test_board[0:4, 3] = 2
    test_board[1:2, 6] = 1
    test_node = mcts_agent.MonteCarloTreeSearchNode(state=test_board, player=PLAYER2)
    ret = test_node.game_result(test_board, PLAYER2)
    assert (ret == 1)

    # Checking lose condition
    test_board = initialize_game_state()
    test_board[0:2, 1] = 1
    test_board[0:4, 3] = 1
    test_board[1:3, 6] = 2
    test_node = mcts_agent.MonteCarloTreeSearchNode(state=test_board, player=PLAYER1)
    ret = test_node.game_result(test_board, PLAYER1)
    assert (ret == -1)
