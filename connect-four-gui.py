import numpy as np
import os
import time

from connect_four_env import ConnectFourEnv
from kivy.app import App
from kivy.graphics import Ellipse
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.properties import ListProperty, NumericProperty
from kivy.graphics import Color


class BoardSpace(Button):
    def __init__(self, *, piece_size: float, piece_offset: float, **kwargs):
        super().__init__(**kwargs)
        self.piece_size = (piece_size, piece_size)
        self.piece_offset = piece_offset

        self._N = 0

    def on_press(self):
        if self._N % 2 == 0:
            self.parent._update_map(id=self.id)
            self._N += 1
            print(self._N)

    def _update_state(self, state: int):
        self.canvas.clear()
        with self.canvas:
            if state == 0:
                Color(1, 1, 1)
            elif state == 1:
                Color(1, 0, 0)
            elif state == 2:
                Color(0, 1, 0)
            elif state == -1:
                Color(0, 0, 0)
            Ellipse(
                pos=(self.x + self.piece_offset, self.y + self.piece_offset),
                size=self.piece_size,
            )


class GameBoard(GridLayout):
    def __init__(self, *, rows: int, cols: int, env: ConnectFourEnv):
        super().__init__(rows=rows, cols=cols)
        self.rows = rows
        self.cols = cols
        self.env = env
        self.space_map = {}
        env.reset()

    def _update_map(self, *, id: int):
        row, col = id.split("-")
        row = int(row) - 1
        col = int(col) - 1
        _, _, done, info = env.step(col=col)
        state = env.state
        self._draw_board(state=state)
        print(state)
        if done:
            done_state = info["done_state"]
            if done_state == 1:
                print("YOU WIN!!!")
            elif done_state == -1:
                print("YOU LOSE!!!")
            else:
                print("TIE GAME!!!")

    def _draw_board(self, state: np.ndarray):
        for row in np.arange(self.rows):
            for col in np.arange(self.cols):
                val = state[row, col]
                id = str(row + 1) + "-" + str(col + 1)
                self.space_map[id]._update_state(state=val)


class ConnectFourApp(App):
    def __init__(self, *, rows: int, cols: int, env: ConnectFourEnv, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.env = env

    def build(self):
        rows = self.rows
        cols = self.cols
        self.board_space_size = int(min(100, 500 / max(self.rows, self.cols)))
        self.piece_size = int(0.9 * self.board_space_size)
        self.piece_offset = int((self.board_space_size - self.piece_size) / 2)
        game_board = GameBoard(rows=rows, cols=cols, env=env)
        for row in np.arange(rows):
            for col in np.arange(cols):
                board_space = BoardSpace(
                    piece_size=self.piece_size, piece_offset=self.piece_offset
                )
                id = str(row + 1) + "-" + str(col + 1)
                board_space.id = id
                game_board.add_widget(board_space)
                game_board.space_map[id] = board_space
        return game_board


def get_move_first_input():
    while True:
        print("Do you want to go first? (y/n/q)")
        usr_input = input().lower()
        if usr_input[0] == "q":
            exit()
        elif usr_input[0] == "y":
            return True
        elif usr_input[0] == "n":
            return False


def get_training_epoch(*, max_epoch: int, model_files: list):
    while True:
        print("Which training epoch? (0-" + str(max_epoch) + ")")
        prefix = input().lower()
        if prefix[0] == "q":
            exit()
        if prefix in model_files:
            return prefix


if __name__ == "__main__":
    rows = 6
    cols = 7
    # move_first = get_move_first_input()

    # # Determine available training epochs
    models_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_path):
        print("No available models.")
        exit()
    model_files = os.listdir(models_path)
    available_epochs = list([int(f) for f in model_files])
    available_epochs.sort()
    max_epoch = np.max(available_epochs)

    # # Get which degree of trained model to load
    # prefix = get_training_epoch(max_epoch=max_epoch, model_files=model_files)
    model_file_path = os.path.join(models_path, "5", "end_model")

    # Instantiate environment
    env = ConnectFourEnv(
        rows=6,
        cols=7,
        move_first=True,
        deterministic_opponent=True,
        opponent_models=[model_file_path],
        max_model_history=None,
        probability_switch_model=0,
        switch_method=None,
        id=1,
    )
    # env = None
    cx4 = ConnectFourApp(rows=rows, cols=cols, env=env)
    cx4.run()
