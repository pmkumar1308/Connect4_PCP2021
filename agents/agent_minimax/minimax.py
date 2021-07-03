import numpy as np
import math
from typing import Optional, Tuple
from agents.common import PlayerAction, SavedState, BoardPiece, PLAYER1, PLAYER2, connected_four_convolve, connected_four,ROWS, \
    COLUMNS,apply_player_action, \
    check_end_state, GameState
import random

WINDOW_LENGTH = 4


def window_value(window, player: BoardPiece):
    """

    :param window: The window in which the heuristic value of the board is calculated
    :param player: Current player playing the game of type BoardPiece
    :return: heuristic_value: heuristic value of the board position in the given window of type float

    """
    heuristic_value = 0

    if player == BoardPiece(1):
        opp_player = BoardPiece(2)
    else:
        opp_player = BoardPiece(1)

        if window.count(player) == 4:
            heuristic_value += 1000
        elif window.count(player) == 3 and window.count(BoardPiece(0)) == 1:
            heuristic_value += 10
        elif window.count(player) == 2 and window.count(BoardPiece(0)) == 2:
            heuristic_value += 5
        if window.count(opp_player) == 3 and window.count(BoardPiece(0)) == 1:
            heuristic_value -= 90
        elif window.count(opp_player) == 2 and window.count(BoardPiece(0)) == 2:
            heuristic_value -= 20

        return heuristic_value


def board_heuristic(board: np.ndarray, player: BoardPiece):
    """

    :param board: Contains current state of the board an ndarray, shape (ROWS, COLUMNS) and data type (dtype) BoardPiece
    :param player: Current player playing the game of type BoardPiece
    :return: heuristic_value: heuristic value of the column of type float

    """
    heuristic_value = 0

    # Horizontal
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMNS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            heuristic_value += window_value(window, player)

    # Vertical
    for c in range(COLUMNS):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            heuristic_value += window_value(window, player)

    # center column
    center_array = [int(i) for i in list(board[:, COLUMNS // 2])]
    center_count = center_array.count(player)
    heuristic_value += center_count * 3

    # Diagonal element
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            heuristic_value += window_value(window, player)

    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            heuristic_value += window_value(window, player)

    return heuristic_value


def minimax(depth: int, board: np.ndarray, player: BoardPiece, alpha, beta, maximizing=True):
    """

    :param depth: depth of the tree search of type int
    :param board: Contains current state of the board an ndarray, shape (ROWS, COLUMNS) and data type (dtype) BoardPiece
    :param player: Current player playing the game of type BoardPiece
    :param alpha: Alpha value for alpha-beta pruning of type float
    :param beta: Beta value for alpha-beta pruning of type float
    :param maximizing: A boolean value to switch between maximising and minimising heuristic_value
    :return: column : the column to be played by the agent of type int
            value : the heuristic value of the board

    """
    board_copy = np.copy(board)
    valid_columns = []
    for col in range(COLUMNS):
        if board[ROWS - 1][col] == 0:
            valid_columns.append(col)

    if depth == 0 or check_end_state(board,player).name == GameState.IS_WIN or len(valid_columns) == 0:
        if check_end_state(board,player).name == GameState.IS_WIN or len(valid_columns) == 0:
            if connected_four(board_copy, BoardPiece(2)):
                return None, math.inf
            elif connected_four(board_copy, BoardPiece(1)):
                return None, -math.inf
            else:
                return None, 0
        else: 
            return None, board_heuristic(board_copy, BoardPiece(2))

    if maximizing:
        value = -math.inf
        column = random.choice(valid_columns)
        for col in valid_columns:
            board_copy = np.copy(board)
            value_temp = minimax(depth - 1, board_copy, player, alpha, beta, False)[1]
            if value_temp > value:
                value = value_temp
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_columns)
        for col in valid_columns:
            board_copy = np.copy(board)
            value_temp = minimax(depth - 1, board_copy, player, alpha, beta, True)[1]
            if value_temp < value:
                value = value_temp
                column = col
            beta = min(beta, value)
            if beta <= alpha:
                break
        return column, value


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                          ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    :param board:   np.ndarray
                    Contains current state of the board an ndarray, shape (ROWS, COLUMNS) and data type (dtype) BoardPiece
    :param player:  BoardPiece
                    Current player playing the game
    :param saved_state: Saved state of the game
    :return: action:    PlayerAction (np.int8)
                        The column to be played
            saved_state: The saved state of the game

    """
    col_, val = minimax(4, board, player, math.inf, -math.inf, True)
    action = PlayerAction(int(col_))
    return action, saved_state
