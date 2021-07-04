import numpy as np
from collections import defaultdict
from typing import Optional, Tuple
from agents.common import PlayerAction, SavedState, BoardPiece, PLAYER1, PLAYER2, connected_four_convolve, connected_four,ROWS, \
    COLUMNS,apply_player_action, \
    check_end_state, GameState

# MCTS Steps

# Selection
# Selection based on the upper confidence bound (UCB)

# Expansion
# Expansion based on the tree state if it has not visited all the child nodes yet

# Simulation
# Randomly move along the nodes from current node till game ends

# Backpropagation
# Based on the end state of the game when travelling from a node transfer a score
# the current node and backpropagation to the parent nodes as well

# UCB
# helps in selection of the next node based on the simulation. Used to balance the exploration
# and exploitation from existing information




class MCTSNode:
    def __init__(self,board:np.ndarray, parent = None, move = None,player: BoardPiece ):

        self.game_state = board
        self.parent = parent
        self.exploration_param = np.sqrt(2)
        self.num_simulations = 200
        self.player = player
        self._visits = 0
        self._results = 0



    def next_best_move(self):

    def simulation(self):
        while check_end_state(self.game_state,self.player) == GameState.STILL_PLAYING:



    def get_valid_columns(self,board : np.ndarray):
        valid_columns = []
        for col in range(COLUMNS):
            if board[ROWS - 1][col] == 0:
                valid_columns.append(col)
        return valid_columns
