import numpy as np
from agents.common import BoardPiece, NO_PLAYER

def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_print_board(test_board: np.ndarray) -> str:
    from agents.common import pretty_print_board
    test_board = np.zeros(6, 7)
    ret = pretty_print_board()

    assert ret.dtype == str
