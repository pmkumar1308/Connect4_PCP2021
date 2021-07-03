import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2
from agents.common import pretty_print_board, string_to_board, initialize_game_state, apply_player_action


def test_initialize_game_state():

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_print_board() -> str:

    test_board = initialize_game_state()
    ret = pretty_print_board(test_board)

    test_board[0, 1] = PLAYER1
    test_board[1, 1] = PLAYER2
    test_board[0, 2] = PLAYER2
    test_board[2, 2] = PLAYER2
    test_board[1, 3] = PLAYER2
    test_board[0, 0] = PLAYER1

    assert isinstance(ret, str)


def test_string_to_board() -> str:

    board = initialize_game_state()
    board_test = pretty_print_board(board)
    ret = string_to_board(board_test)

    assert isinstance(ret,np.ndarray)

def test_apply_player_action:

    test_board = initialize_game_state()
    ret = pretty_print_board(test_board)



