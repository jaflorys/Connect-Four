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
        move_first: bool,
        opponent_model: Union[None, DQN],
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.move_first = move_first
        self.opponent_model = opponent_model
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
        playerWon = check_connect_four(state=self.state, player=self.piece)

        if playerWon:
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
            col = np.random.choice(list(self.available_columns), 1)[0]
        else:
            # Interrogate model to find best move
            col, _ = self.opponent_model.predict(
                self.state.flatten(), deterministic=True
            )
            # If column not valid, choose random column
            if not col in self.available_columns:
                col = np.random.choice(list(self.available_columns), 1)[0]

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
                col = np.random.choice(list(self.available_columns), 1)[0]
            else:
                # Interrogate model to determine first move
                # Note: All columns are available for first move
                col, _ = self.opponent_model.predict(
                    self.state.flatten(), deterministic=True
                )
            # Opponent makes first move
            self.drop_piece(col=col, player=self.opponent_piece)
