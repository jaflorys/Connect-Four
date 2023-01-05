import numpy as np
import os
import time

from connect_four_env import ConnectFourEnv
from kivy.app import App
from kivy.graphics import Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

# from kivy.properties import ListProperty, NumericProperty
from kivy.graphics import Color


class BoardSpace(Button):
    def __init__(self, *, piece_size: float, piece_offset: float, **kwargs):
        super().__init__(**kwargs)
        self.piece_size = (piece_size, piece_size)
        self.piece_offset = piece_offset
        self.background_color = [0, 0, 0, 0]

    def on_press(self):
        self.parent._update_map(id=self.id)

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
    def __init__(self, *, rows: int, cols: int):
        super().__init__(rows=rows, cols=cols)
        self.rows = rows
        self.cols = cols
        self.env = None
        self.space_map = {}
        self._initialized = False
        self._move_first = None

    def _update_map(self, *, id: int):
        if not self._initialized:
            self._initialized = True
            if self.parent.children[0].children[1].state == "down":
                move_first = True
            else:
                move_first = False

            if not move_first or not (move_first == self._move_first):
                self._move_first = move_first
                self._instantiate_env()
            self.env.reset()
            self.parent.children[0].children[0].text = ""
            self._draw_board(state=self.env.state)
        else:
            row, col = id.split("-")
            row = int(row) - 1
            col = int(col) - 1
            # Check that col is not full
            is_valid_move = np.sum(self.env.state[:, col] == np.zeros(self.rows)) > 0
            if is_valid_move:
                _, _, done, info = self.env.step(col=col)
                self._draw_board(state=self.env.state)
                if done:
                    done_state = info["done_state"]
                    if done_state == 1:
                        self.parent.children[0].children[0].text = "YOU WIN!!!"
                    elif done_state == -1:
                        self.parent.children[0].children[0].text = "YOU LOSE!!!"
                    else:
                        self.parent.children[0].children[0].text = "TIE GAME..."
                    self._initialized = False

    def _instantiate_env(self):
        # Instantiate environment
        self.env = ConnectFourEnv(
            rows=6,
            cols=7,
            move_first=self._move_first,
            deterministic_opponent=True,
            opponent_models=[
                os.path.join(os.getcwd(), "gui_opponent_policy", "policy")
            ],
            max_model_history=None,
            probability_switch_model=0,
            switch_method=None,
            id=1,
        )

    def _draw_board(self, state: np.ndarray):
        for row in np.arange(self.rows):
            for col in np.arange(self.cols):
                val = state[row, col]
                id = str(row + 1) + "-" + str(col + 1)
                self.space_map[id]._update_state(state=val)

    def _reset_game(self):
        self._initialized = False
        self._update_map(id=None)


class GameScreen(BoxLayout):
    pass


class GameButtons(FloatLayout):
    def reset_game(self):
        self.parent.children[1]._reset_game()


class ConnectFourApp(App):
    def __init__(self, *, rows: int, cols: int, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def build(self):
        rows = self.rows
        cols = self.cols
        self.board_space_size = int(min(100, 500 / max(self.rows, self.cols)))
        self.piece_size = int(0.9 * self.board_space_size)
        self.piece_offset = int((self.board_space_size - self.piece_size) / 2)
        game_board = GameBoard(rows=rows, cols=cols)
        for row in np.arange(rows):
            for col in np.arange(cols):
                board_space = BoardSpace(
                    piece_size=self.piece_size, piece_offset=self.piece_offset
                )
                id = str(row + 1) + "-" + str(col + 1)
                board_space.id = id
                game_board.add_widget(board_space)
                game_board.space_map[id] = board_space
        game_screen = GameScreen()
        game_buttons = GameButtons()
        game_screen.add_widget(game_board)
        game_screen.add_widget(game_buttons)
        return game_screen


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

    # # Get which degree of trained model to load
    # prefix = get_training_epoch(max_epoch=max_epoch, model_files=model_files)
    # model_file_path = os.path.join(models_path, "3", "end_model")

    cx4 = ConnectFourApp(rows=rows, cols=cols)
    cx4.run()
