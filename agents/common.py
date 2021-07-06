from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple
from scipy.signal.sigtools import _convolve2d

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

ROWS = 6
COLUMNS = 7
CONNECT_N = 4

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    initial_state = np.empty((ROWS, COLUMNS), dtype=BoardPiece) * NO_PLAYER

    return initial_state
    # raise NotImplementedError


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    """

    board = np.flip(board, 0)  # flip vertically to get board[x][y] in lower left
    print('|===============|')
    pp_string = str('')
    for y in range(ROWS):
        board_2_str = pp_string + ' '.join(
            NO_PLAYER_PRINT if board[y][x] == NO_PLAYER else PLAYER1_PRINT if board[y][x] == PLAYER1 else PLAYER2_PRINT
            for x in range(COLUMNS))

        print(str('| ') + ' '.join(
            NO_PLAYER_PRINT if board[y][x] == NO_PLAYER else PLAYER1_PRINT if board[y][x] == PLAYER1 else PLAYER2_PRINT
            for x in range(COLUMNS)) + str(' |'))
    print('|===============|')
    # print(str('| ') + ' '.join(map(str, range(COLUMNS))) + str(' |'))
    print()

    return board_2_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    str_2_board = np.array(pp_board).reshape((ROWS,COLUMNS))

    return str_2_board

    # raise NotImplementedError()


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    board = np.flip(board, 0)

    if copy:
        board_copy = board

    for row in range(board.shape[0]):
        if board[row][action] == 0:
            lowest_open_row = row

    board[lowest_open_row, action] = player

    return board
    # raise NotImplementedError()


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1

    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j + CONNECT_N] == player):
                return True

    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i + CONNECT_N, j] == player):
                return True

    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i + CONNECT_N, j:j + CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True

    return False


col_kernel = np.ones((4, 1), dtype=BoardPiece)
row_kernel = np.ones((1, 4), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(4, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(4, dtype=BoardPiece))[::-1, :])

def connected_four_convolve(
        board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = NO_PLAYER
    board[board == player] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == 4):
            return True
    return False


    # raise NotImplementedError()


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    state = connected_four(board, player)

    if state:
        game_state = GameState.IS_WIN
    else:
        if NO_PLAYER in board:
            game_state = GameState.STILL_PLAYING
        else:
            game_state = GameState.IS_DRAW

    return game_state

    # raise NotImplementedError()

def get_valid_columns(board:np.ndarray):
    # print('s')
    valid_columns = []
    for col in range(COLUMNS):
        if board[ROWS - 1][col] == 0:
            valid_columns.append(col)
    return valid_columns

class SavedState:
    pass




GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

