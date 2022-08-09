from email.policy import default
import numpy as np
import random

"""The Connect Four game environment.

This module contains a class for the Connect Four Gym
environment. This environment mimics the dynamics of
a Connect Four like game defined by a rectagular grid
of rows and columns. The objective of each player is
to obtain "n" contiguous pieces in a horizontal, vertical,
or diagonal placement on the board.

"""

from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from typing import DefaultDict, Union
from utils import check_connect_four
from stable_baselines3.dqn.dqn import DQN


class ConnectFour(Env):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        move_first: Union[None, bool],
        deterministic_opponent: bool,
        opponent_models: Union[None, DQN],
        max_model_history: Union[None, int],
        probability_switch_model: float,
        switch_method: Union[None, str],
        id: int,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.deterministic_opponent = deterministic_opponent
        self.probability_switch_model = probability_switch_model
        self.switch_method = switch_method
        self.id = id
        if move_first == None:
            self.move_first = bool(random.randint(0, 1))
        else:
            self.move_first = move_first
        self.action_space = Discrete(self.n_cols)
        self.observation_space = MultiDiscrete(
            [3 for i in np.arange(self.n_rows * self.n_cols)]
        )
        # States:
        # 0: Empty place
        # 1: Self piece
        # 2: Opponent piece
        self.state = np.zeros((self.n_rows, self.n_cols))
        self.available_columns = set(np.arange(self.n_cols))
        self.piece = 1
        self.opponent_piece = 2

        # Limit opponent models to max history length
        self.recent_models = None
        self.sample_weights = None
        self.opponent_model = None
        self.policy_idx = None
        self.total_reward = None
        self.num_recent_models = None
        if opponent_models:
            if max_model_history:
                self.num_recent_models = int(
                    min(len(opponent_models), max_model_history)
                )
                self.recent_models = opponent_models[: self.num_recent_models]
            else:
                self.recent_models = opponent_models
            self.num_recent_models = len(self.recent_models)

            if self.switch_method == "inverse_history_length":
                # Compute model sampling weights
                sample_weights = [
                    self.num_recent_models - i
                    for i in np.arange(self.num_recent_models)
                ]
                sample_weights = sample_weights / np.sum(sample_weights)
                self.sample_weights = sample_weights

            if self.switch_method == "ucb":
                # Initialize action-reward estimates and counts
                self.Q_t = np.ones(self.num_recent_models) * np.inf
                self.N_t = np.zeros(self.num_recent_models)
                # Todo: Allow user to define these parameters
                self.alpha = 0.1
                self.ucb_c = 2.0

            self.opponent_map = DefaultDict(str)
            for i, model_name in enumerate(self.recent_models):
                self.opponent_map[model_name] = i

        # Instantiate the opponent model
        self.select_opponent_model()

        # Take initial opponent move (only occurs if 'self.move_first==False')
        self.initial_opponent_move()

    def step(self, col):
        """Steps the environment state.

        Updates the environment's state according to the player's move, defined
        by the column into which the player's piece is begin dropped. After the
        player's move the function determines whether (1) the player won, or
        (2) the board is full (tied game). If neither condition is met, the
        opponent policy is interrogated (if it exists) or a column is randomly
        selected for the opponent's move. After the opponent's move, the function
        determines (1) whether the opponent won, or (2) the board is full.

        Args:
            col: An integer that defines the column into which the player's piece
            is dropped.

        Returns:
            state: A flattened np.ndarray of the environment's updated state.
            reward: A float of the step reward.
            done: A Boolean that denotes whether the game has ended
            info: A dictionary of game information.
        """
        reward = 0
        if not col in self.available_columns:
            # If row is full, give penality and randomly choose available column
            reward -= 0.1
            col = np.random.choice(list(self.available_columns), 1)[0]
        # Place player piece
        self.drop_piece(col=col, player=self.piece)

        # Determine whether player won game
        selfWon = check_connect_four(state=self.state, player=self.piece)

        if selfWon:
            # Return win reward
            reward += 1
            done = True
            self.total_reward = reward
            return self.state.flatten(), reward, done, {"done_state": 1}

        if self.all_positions_full():
            # Return tie reward
            reward += 0
            done = True
            self.total_reward = reward
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If player has not won and board is not full, opponent moves
        if self.opponent_model == None:
            # Opponent picks random column
            col = np.random.choice(list(self.available_columns))
        else:
            # Interrogate model to find best move
            col = int(
                self.opponent_model.predict(
                    self.invert_state().flatten(),
                    deterministic=self.deterministic_opponent,
                )[0]
            )
            # If column not valid, choose random column
            if not col in self.available_columns:
                col = np.random.choice(list(self.available_columns))

        self.drop_piece(col=col, player=self.opponent_piece)

        # Check whether opponent won
        opponentWon = check_connect_four(state=self.state, player=self.opponent_piece)

        if opponentWon:
            # return lose reward
            reward += -1
            done = True
            self.total_reward = reward
            return self.state.flatten(), reward, done, {"done_state": -1}

        if self.all_positions_full():
            # Return tie reward
            reward += 0
            done = True
            self.total_reward = reward
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If no conditions met, game continues
        done = False
        return self.state.flatten(), reward, done, {}

    def reset(self):
        """Resets the environment's state.

        Resets the environment's state to the beginning of a new game.
        Determines whether to select a new opponent model probabilistically
        according to user-specified probability and selection method. Randomly
        selects whether the "player" or "opponent" has the first move, and
        implements the opponent's move accordingly. Returns the flattened 
        environment state.

        Args:
            None.

        Returns:
            A flattened np.ndarray of the environment state.

        """
        self.state = np.zeros((self.n_rows, self.n_cols))
        self.available_columns = set(np.arange(self.n_cols))
        if random.random() < self.probability_switch_model:
            self.select_opponent_model()
        self.move_first = bool(random.randint(0, 1))
        # Take initial opponent move (only occurs if 'self.move_first==False')
        self.initial_opponent_move()

        return self.state.flatten()

    def all_positions_full(self):
        """Determines whether the the board is full.

        Determines whether all board positions are occupied by determing
        whether any available columns remain. If no available columns
        remain, the board is full.

        Args:
            None.

        Returns:
            A boolean value that is "True" if all positions are occupied.

        """
        return len(self.available_columns) == 0

    def drop_piece(self, *, col: int, player: int):
        """Updates the game board according to a player's move.

        Models a player dropping a piece into one of the columns of the
        Connect Four board. The piece falls to the first available position. 
        If a row becomes completely occupied, it is removed from the list of
        available columns.

        Args:
            col: An integer for the column into which the piece is dropped.
            player: An integer ('1' or '2') that represents the player's piece.

        Returns:
            None.

        """
        # Determine which positions in column are filled
        filled_positions = np.abs(self.state[:, col])
        row = -1
        if np.sum(filled_positions) == 0:
            # If column is empty, lands at last row
            row = self.n_rows - 1
            self.state[self.n_rows - 1, col] = player
        else:
            # Otherwise lands at first unfilled position
            row = np.argmax(
                [
                    filled_positions[i] - filled_positions[i - 1]
                    for i in np.arange(1, self.n_rows)
                ]
            )
            self.state[row, col] = player

        if row == 0:
            # If column full, remove from set of available columns
            self.available_columns.remove(col)

    def initial_opponent_move(self):
        """Performs the initial opponent move.

        When the opponent has the first move, this function
        either interrogates the opponent's policy (when available)
        or selects a random column for the opponent's move. It then
        updates the board state.

        Args:
            None

        Returns:
            None.
        """
        # If player is going second, opponent makes move
        if not self.move_first:
            if self.opponent_model == None:
                # Take random move
                col = np.random.choice(list(self.available_columns))
            else:
                # Interrogate model to determine first move
                # Note: All columns are available for first move
                col = int(
                    self.opponent_model.predict(
                        self.invert_state().flatten(),
                        deterministic=self.deterministic_opponent,
                    )[0]
                )

            # Opponent makes first move
            self.drop_piece(col=col, player=self.opponent_piece)

    def select_opponent_model(self):
        """Selectes a random opponent policy according to a user-specified
        method.

        Selects a random file path to a existing policy to be used by the
        opponent model. The selection is performed according to one of three
        user-specified methods:
            (1) Random selection
            (2) Random selection weighted inversly proportional to age
            (3) Upper-confidence bound algorithm for multi-bandit problem

        Args:
            None

        Returns:
            A file path to an existing policy.
        """
        if self.recent_models:
            if self.num_recent_models > 1:
                if self.opponent_model:
                    del self.opponent_model

                if self.switch_method == None:
                    load_model_path = self.random_model_choice()
                elif self.switch_method == "inverse_history_length":
                    load_model_path = self.weighted_random_model_choice()
                elif self.switch_method == "ucb":
                    load_model_path = self.ucb_model_choice()
                else:
                    print("Error: Opponent policy selection method not recognized.")
                    exit()
                self.opponent_model = DQN.load(load_model_path)
                self.policy_idx = self.opponent_map[load_model_path]
            else:
                # Only load model once
                if not self.opponent_model:
                    load_model_path = self.recent_models[0]
                    self.opponent_model = DQN.load(load_model_path)
                    self.policy_idx = self.opponent_map[load_model_path]

    def random_model_choice(self):
        """Selectes a random policy to serve as an opponent.

        Selects a random file path for an opponent policy from a 
        list of recent policies that do not exeed the maximum age.

        Args:
            None

        Returns:
            A file path to an existing policy.
        """
        return np.random.choice(self.recent_models)

    def weighted_random_model_choice(self):
        """Selectes a policy to serve as an opponent according
        to a random weighting.

        Selects a random file path for an opponent policy of a list
        of recent policies that do no exceed the maximum age. The 
        selection is weighted.

        Args:
            None

        Returns:
            A file path to an existing policy.
        """
        return np.random.choice(self.recent_models, p=self.sample_weights)

    def ucb_model_choice(self):
        """Selectes a policy according to an upper confidence bound
        for the mutli-armed bandit problem.

        Updates the estimated policy-reward values and counts. Then
        selects a policy with least expected reward according to
        an upper confidence bound.

        Args:
            None

        Returns:
            A file path to an existing policy.
        """
        # If first selection, select a random model
        if self.policy_idx == None or self.total_reward == None:
            return self.random_model_choice()

        idx = self.policy_idx
        # Incentivize opponents that result in lowest reward
        reward = -1.0 * self.total_reward
        self.Q_t[idx] = (
            reward
            if self.Q_t[idx] == np.inf
            else self.Q_t[idx] + self.alpha * (reward - self.Q_t[idx])
        )
        self.N_t[idx] += 1
        ucb = self.Q_t + self.ucb_c * np.sqrt(
            np.log(np.sum(self.N_t)) * 1.0 / (self.N_t + 1e-9)
        )
        select_idx = np.argmax(ucb)
        
        return self.recent_models[select_idx]

    def invert_state(self):
        """Reverses the occupied positions of the board.

        Policies are trained with a board state "1" denoting the positions of
        its pieces. When an opponent policy is interrogated for the best action,
        the board positions need to be inverted so that the opponent will correctly
        observe a state "1" as defining its positions.

        Args:
            None

        Returns:
            A np.ndarray of the board state with board positions containing
            "1" and "2" swapped.
        """
        inv_part_1 = 2 * np.ones((self.n_rows, self.n_cols))
        inv_part_2 = np.ones((self.n_rows, self.n_cols))
        self.inv_state = (self.state == 1).astype(int) * inv_part_1 + (
            self.state == 2
        ).astype(int) * inv_part_2
        return self.inv_state
