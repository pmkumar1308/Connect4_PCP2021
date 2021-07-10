import numpy as np
from collections import defaultdict
from typing import Optional, Tuple
from agents.common import PlayerAction, SavedState, BoardPiece, PLAYER1, PLAYER2, connected_four_convolve, \
    connected_four, ROWS, \
    COLUMNS, apply_player_action, get_valid_columns, \
    check_end_state, GameState


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


# class MCTSNode:
#
#     def __init__(self, parent, move):
#         # self, board: np.ndarray, player: BoardPiece,
#         # self.game_state = board
#         self.parent = parent
#         self.exploration_param = np.sqrt(2)
#         self.num_simulations = 200
#         # self.player = player
#         self._visits = 0
#         self._wins = 0
#         self.move = move
#         self.child_states = []
#
#     def expand(self, game_state, player):
#         valid_cols = get_valid_columns(game_state)
#         while check_end_state(game_state, player) == GameState.STILL_PLAYING:
#             for col in valid_cols:
#                 new_child = MCTSNode(col, self)
#                 self.child_states.append(new_child)
#
#     def update(self, results):
#         self._visits += 1
#         if results == GameState.IS_WIN:
#             self.wins +=1
#
#     def is_terminal(self):
#         return len(self.child_states) == 0
#
#     def has_parent(self):
#         if self.parent is not None:
#             return True
#         return False
#
#     def best_child_state(self,exploration_param = 1.414):
#         choices_weights = [(c.q() / c.n()) + exploration_param * np.sqrt((2 * np.log(self.n()) / c.self._visits)) for c in self.child_states]
#         return self.child_states[np.argmax(choices_weights)]
#
# def tree_policy():
#
#
# def gen_mcts_best_action(board: np.ndarray,player: BoardPiece):
#     state = board
#     root_node = MCTSNode(None, None)
#     while time remains:
#         n, s = root_node, state.copy
#         while not n.is_terminal():  # select leaf
#             n = tree_policy_child(n)
#             s.addmove(n.move)
#         n.expand(s,player)  # expand
#         n = tree_policy(n)
#
#         while not check_end_state(s, player) == GameState.STILL_PLAYING:  # simulate
#             s = simulation_policy_child(s)
#
#         result = evaluate(s)
#
#         while n.has_parent():  # propagate
#             n.update(result)
#             n = n.parent
#
#         action = PlayerAction(int(col_))
#         return action

class MonteCarloTreeSearchNode():

    def __init__(self, state, player, parent=None, parent_action=None):
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
        next_state = self.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, player=self.player, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.is_game_over()

    def rollout(self):
        current_rollout_state = self.state

        while not self.is_game_over():
            possible_moves = get_valid_columns(current_rollout_state)

            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.move(action)
        return self.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param = 1.414):

        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.choice(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 1000

        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child()


    def is_game_over(self):
        return connected_four(self.state, self.player)

    def move(self, action):
        b = apply_player_action(self.state, action, self.player)
        return b

    def game_result(self):
        if check_end_state(self.state, self.player).name == GameState.IS_WIN:
            return 1
        if check_end_state(self.state, self.player).name == GameState.IS_DRAW:
            return 0


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                       ) -> Tuple[PlayerAction, Optional[SavedState]]:
    root = MonteCarloTreeSearchNode(state=board, player=player)
    # print('a')
    selected_node = root.best_action()
    # print('The selected node is',selected_node)
    action = PlayerAction(int(selected_node.parent_action))
    return action, saved_state

