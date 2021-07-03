import numpy as np
import random
from typing import Optional,Tuple
from agents.common import PlayerAction, BoardPiece, SavedState,ROWS,COLUMNS


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    action = PlayerAction(-1)
    # Choose a valid, non-full column randomly and return it as `action`
    if player == BoardPiece(2):
        valid_columns = []
        for col in range(COLUMNS):
            if board[ROWS - 1][col] == 0:
                valid_columns.append(col)
        action = PlayerAction(random.sample(valid_columns, 1))
    return action, saved_state