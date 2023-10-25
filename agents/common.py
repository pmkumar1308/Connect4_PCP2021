from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple
from scipy.signal import convolve2d

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

    board = np.flipud(board)  # flip vertically to get board[x][y] in lower left
    print('|===============|')
    pp_string = str('')
    for y in range(ROWS):
        # board_2_str = pp_string + ' '.join(
        #     NO_PLAYER_PRINT if board[y][x] == NO_PLAYER else PLAYER1_PRINT if board[y][x] == PLAYER1 else PLAYER2_PRINT
        #     for x in range(COLUMNS))
        board_2_str = ''
        # board_2_str = pp_string + str('| ') + ' '.join(
        #     NO_PLAYER_PRINT if board[y][x] == NO_PLAYER else PLAYER1_PRINT if board[y][x] == PLAYER1 else PLAYER2_PRINT
        #     for x in range(COLUMNS)) + str(' |')

        print(str('| ') + ' '.join(
            NO_PLAYER_PRINT if board[y][x] == NO_PLAYER else PLAYER1_PRINT if board[y][x] == PLAYER1 else PLAYER2_PRINT
            for x in range(COLUMNS)) + str(' |'))
    print('|===============|')
    print(str('| ') + ' '.join(map(str, range(COLUMNS))) + str(' |'))

    return board_2_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    str_2_board = np.array(pp_board).reshape((ROWS,COLUMNS))

    return str_2_board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    board_copy = board
    if copy:
        board_copy = board.copy()
    lowest_open_row = np.min(np.where(board_copy[:, action] == NO_PLAYER))
    board_copy[lowest_open_row, action] = player
    return board_copy

col_kernel = np.ones((4, 1), dtype=BoardPiece)
row_kernel = np.ones((1, 4), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(4, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(4, dtype=BoardPiece))[::-1, :])

def connected_four(
        board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    """
   Returns True if there are four adjacent pieces equal to `player` arranged
   in either a horizontal, vertical, or diagonal line. Returns False otherwise.
   If desired, the last action taken (i.e. last column played) can be provided
   for potential speed optimisation.
    """
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = NO_PLAYER
    board[board == player] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = convolve2d(board, kernel, mode='valid', boundary='fill', fillvalue=BoardPiece(0))
        if np.any(result == 4):
            return True
    return False


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

def get_valid_columns(board:np.ndarray):

    valid_columns = []
    for col in range(COLUMNS):
        if board[ROWS - 1, col] == 0:
            valid_columns.append(col)
    return valid_columns

class SavedState:
    pass


def get_opponent_player(player: BoardPiece) -> BoardPiece:
    if player == PLAYER1:
        return PLAYER2
    else:
        return PLAYER1

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

