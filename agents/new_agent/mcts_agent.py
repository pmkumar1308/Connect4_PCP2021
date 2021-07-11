import numpy as np
from collections import defaultdict
from typing import Optional, Tuple
from agents.common import PlayerAction, SavedState, BoardPiece, PLAYER1, PLAYER2, \
    connected_four, ROWS, \
    COLUMNS, apply_player_action, get_valid_columns, \
    check_end_state, GameState, get_opponent_player


# MCTS Steps

# Selection
# Selection based on the upper confidence bound (UCB)

# Expansion
# Expansion based on the tree state if it has not visited all the child nodes yet

# Simulation/Rollout
# Randomly move along the nodes from current node till game ends

# Backpropagation(update)
# Based on the end state of the game when travelling from a node transfer a score
# the current node and backpropagation to the parent nodes as well

# UCB
# helps in selection of the next node based on the simulation. Used to balance the exploration
# and exploitation from existing information

class MonteCarloTreeSearchNode():

    def __init__(self, state, player: BoardPiece, parent=None, parent_action=None):
        """
        Initialises (constructs) the class object taking in a given state (board) and the player

        :param state: np.ndarray representing the current board
        :param player: int value defining the player playing the game
        """
        self.initial_state = state.copy()
        self.state = state
        self.parent = parent
        self.remaining_actions = None
        self.remaining_actions = self.find_remaining_actions()
        self.parent_action = parent_action
        self.children = []
        self.player = player
        self.number_of_visits = 0
        self.results = defaultdict(int)
        self.results[1] = 0
        self.results[-1] = 0

        return

    def expand(self):
        """
        Takes the self attributes and methods and uses them to expand the current state to a new child state
        and append it to the existing children states.
        """
        action = self.remaining_actions.pop()
        next_state = self.apply_move(self.state, action, self.player)

        child_node = MonteCarloTreeSearchNode(
            next_state, player=self.player, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def find_remaining_actions(self):
        """
        Returns the remaining actions (valid columns) given the current board state.
        """
        self.remaining_actions = get_valid_columns(self.state)
        return self.remaining_actions

    def score(self):
        """
        Returns a win score for the current node based on the wins and losses simulated from
        a current board state.
        """
        wins = self.results[1]
        loses = self.results[-1]
        return wins - loses

    def num_visits(self):
        """
        Returns the number of times a the current board state was visited.
        """
        return self.number_of_visits

    def rollout(self):
        """
        Simulates the current state using a rollout policy which randomly selects a child until
        the game ends and returns the result of the game (1,0,-1) corresponding to the win, draw or
        loss for the AI player.
        """
        current_rollout_state = self.state
        player_ = self.player
        while not self.is_game_over(current_rollout_state, player_):
            remaining_moves = get_valid_columns(current_rollout_state)
            action = self.rollout_policy(remaining_moves)
            current_rollout_state = self.apply_move(current_rollout_state, action, player_)
            player_ = get_opponent_player(player_)
        return self.game_result(current_rollout_state, player_)

    def backpropagate(self, result):
        """
        Takes in the game result from the rollout and backpropagates the value to the parent node.
        """
        self.number_of_visits += 1.
        self.results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        """
        Checks if a current board state is a leaf node that can not be fully expanded
        """
        return len(self.remaining_actions) == 0

    def best_child(self, exploration_param=1.414):
        """
        Returns the best child to the current state with the highest UCB score among them.
        """
        ucb_scores = [(c.score() / c.num_visits()) + exploration_param * np.sqrt((2 * np.log(self.num_visits()) / c.num_visits())) for c in self.children]
        return self.children[np.argmax(ucb_scores)]

    def rollout_policy(self, remaining_moves):
        """
        Takes in the possible remaining moves in the current board state and returns a random move.
        """
        return np.random.choice(remaining_moves)

    def tree_policy(self):
        """
        Traverses through the game tree gets the rewards for the nodes, expand nodes if they were
        visited before and then rolls out(simulates) otherwise just rolls out the current node.
        """

        current_node = self
        while current_node.is_fully_expanded():
            current_node = current_node.best_child()

        # fully_expanded_node = current_node
        if current_node.number_of_visits != 0:
            current_node = current_node.expand()

        reward = current_node.rollout()
        current_node.backpropagate(reward)

    def best_simulated_action(self):
        """
        Runs the Monte Carlo simulation through the game tree for the specified number of times
        and returns the best move.
        """
        simulation_no = 3000

        for i in range(simulation_no):
            self.tree_policy()
        return self.best_child()

    def game_result(self, curr_state, player_r):
        """
        Takes in the current game state and the player playing and returns the value based on
        win, draw or loss for the AI player (PLAYER2)

        """
        curr_player = player_r
        game_state = check_end_state(curr_state, curr_player).name
        if game_state == 'IS_WIN' and player_r == PLAYER2:
            return 1
        if game_state == 'IS_DRAW':
            return 0
        if game_state == 'IS_WIN' and player_r == PLAYER1:
            return -1

    def is_game_over(self, curr_state, player):
        """
        Checks if game is over based on the current board state and the player.
        """
        if check_end_state(curr_state, player) == GameState.STILL_PLAYING or check_end_state(curr_state, get_opponent_player(player)) == GameState.STILL_PLAYING:
            return False
        else:
            return True

    def apply_move(self, board:np.ndarray, action: PlayerAction, player_playing:BoardPiece):
        """
        Applies the action selected for a given player to the current board state and returns the
        modified board.
        """
        b = apply_player_action(board, action, player_playing, True)
        return b


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                       ) -> Tuple[PlayerAction, Optional[SavedState]]:
    root = MonteCarloTreeSearchNode(state=board, player=player)
    best_node = root.best_simulated_action()
    action = PlayerAction(int(best_node.parent_action))
    return action, saved_state
