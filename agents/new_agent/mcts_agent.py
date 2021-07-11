import numpy as np
from collections import defaultdict
from typing import Optional, Tuple
from agents.common import PlayerAction, SavedState, BoardPiece, PLAYER1, PLAYER2, \
    connected_four, ROWS, \
    COLUMNS, apply_player_action, get_valid_columns, \
    check_end_state, GameState, get_opponent


# MCTS Steps

# Selection
# Selection based on the upper confidence bound (UCB)

# Expansion
# Expansion based on the tree state if it has not visited all the child nodes yet

# Simulation
# Randomly move along the nodes from current node till game ends

# Backpropagation(update)
# Based on the end state of the game when travelling from a node transfer a score
# the current node and backpropagation to the parent nodes as well

# UCB
# helps in selection of the next node based on the simulation. Used to balance the exploration
# and exploitation from existing information

class MonteCarloTreeSearchNode():

    def __init__(self, state, player: BoardPiece, parent=None, parent_action=None):
        self.initial_state = state.copy()
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.player = player
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()

        return

    def untried_actions(self):
        self._untried_actions = get_valid_columns(self.state)
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.move(self.state, action, self.player)  # Should I be using self.state
        child_node = MonteCarloTreeSearchNode(
            next_state, player=self.player, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self, curr_state,player):
        return self.is_game_over(curr_state,player)

    def rollout(self):
        current_rollout_state = self.state
        player_ = self.player
        while not self.is_game_over(current_rollout_state, player_):
            possible_moves = get_valid_columns(current_rollout_state)

            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.move(current_rollout_state, action, player_)
            if player_ == PLAYER2:
                player_ = PLAYER1
            else:
                player_ = PLAYER2

        return self.game_result(current_rollout_state, player_)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=1.414):

        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return np.random.choice(possible_moves)

    def _tree_policy(self):

        current_node = self
        while current_node.is_fully_expanded():
            current_node = current_node.best_child()

        # fully_expanded_node = current_node
        if current_node._number_of_visits != 0:
            current_node = current_node.expand()

        reward = current_node.rollout()
        current_node.backpropagate(reward)

    def best_action(self):
        simulation_no = 1000

        for i in range(simulation_no):
            self._tree_policy()

        return self.best_child()


    def is_game_over(self, curr_state, player):
        if check_end_state(curr_state,player) == GameState.STILL_PLAYING or check_end_state(curr_state,get_opponent(player)) == GameState.STILL_PLAYING:
            return False
        else :
            return True

    def move(self, board, action, player_playing):



        b = apply_player_action(board, action, player_playing, True)
        return b

    def game_result(self, curr_state, player_r):
        """

        :param curr_state:
        :type curr_state:
        :param player_r:
        :type player_r:
        :return:
        :rtype:
        """
        curr_player = player_r
        game_state = check_end_state(curr_state, curr_player).name
        if game_state == 'IS_WIN' and self.player == PLAYER2:
            return 1
        if game_state == 'IS_DRAW':
            return 0
        if game_state == 'IS_WIN' and self.player == PLAYER1:
            return -1

    def get_legal_actions(self):
        return get_valid_columns(self.board)


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                       ) -> Tuple[PlayerAction, Optional[SavedState]]:
    root = MonteCarloTreeSearchNode(state=board, player=player)
    selected_node = root.best_action()
    action = PlayerAction(int(selected_node.parent_action))
    return action, saved_state
