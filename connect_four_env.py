from tabnanny import check
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import random
from typing import Union

from utils import check_connect_four
from stable_baselines3.dqn.dqn import DQN


class ConnectFour(Env):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        opponent_models: Union[None, DQN],
        max_model_history: Union[None, int],
        probability_switch_model: float,
        id: int,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.probability_switch_model = probability_switch_model
        self.id = id
        self.move_first = bool(random.randint(0, 1))
        self.action_space = Discrete(self.n_cols)
        self.observation_space = MultiDiscrete(
            [3 for i in np.arange(self.n_rows * self.n_cols)]
        )
        # States:
        # 0: Empty place
        # 1: Player piece
        # 2: Opponent piece
        self.state = np.zeros((self.n_rows, self.n_cols))
        self.available_columns = set(np.arange(self.n_cols))
        self.piece = 1
        self.opponent_piece = 2

        # Limit opponent models to max history length
        self.recent_models = None
        self.sample_weights = None
        self.opponent_model = None
        self.num_recent_models = None
        if opponent_models:
            if max_model_history:
                num_recent_models = int(min(len(opponent_models), max_model_history))
                self.recent_models = opponent_models[:num_recent_models]
            else:
                self.recent_models = opponent_models
            self.num_recent_models = len(self.recent_models)
            # Compute model sampling weights
            sample_weights = [
                num_recent_models - i for i in np.arange(num_recent_models)
            ]
            sample_weights = sample_weights / np.sum(sample_weights)
            self.sample_weights = sample_weights

        # Instantiate the opponent model
        self.select_opponent_model()

        # Take initial opponent move (only occurs if 'self.move_first==False')
        self.initial_opponent_move()

    def step(self, col):
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
            return self.state.flatten(), reward, done, {"done_state": 1}

        if self.all_positions_full():
            # Return tie reward
            reward += 0
            done = True
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If player has not won and board is not full, opponent moves
        if self.opponent_model == None:
            # Opponent picks random column
            col = np.random.choice(list(self.available_columns))
        else:
            # Interrogate model to find best move
            col, _ = self.opponent_model.predict(
                self.state.flatten(), deterministic=True
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
            return self.state.flatten(), reward, done, {"done_state": -1}

        if self.all_positions_full():
            # Return tie reward
            reward += 0
            done = True
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If no conditions met, game continues
        done = False
        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = np.zeros((self.n_rows, self.n_cols))
        self.available_columns = set(np.arange(self.n_cols))
        if random.random() < self.probability_switch_model:
            self.select_opponent_model()
        self.move_first = bool(random.randint(0, 1))
        # Take initial opponent move (only occurs if 'self.move_first==False')
        self.initial_opponent_move()

        return self.state.flatten()

    def all_positions_full(self):
        return len(self.available_columns) == 0

    def drop_piece(self, *, col: int, player: int):
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
        # If player is going second, opponent makes move
        if not self.move_first:
            if self.opponent_model == None:
                # Take random move
                col = np.random.choice(list(self.available_columns))
            else:
                # Interrogate model to determine first move
                # Note: All columns are available for first move
                col, _ = self.opponent_model.predict(
                    self.state.flatten(), deterministic=True
                )
            # Opponent makes first move
            self.drop_piece(col=col, player=self.opponent_piece)

    def select_opponent_model(self):
        if self.recent_models:
            if self.num_recent_models > 1:
                if self.opponent_model:
                    del self.opponent_model
                load_model_path = np.random.choice(
                    self.recent_models, p=self.sample_weights
                )
                load_model_path + "_" + str(id)
                self.opponent_model = DQN.load(load_model_path)
