from optparse import check_choice
import numpy as np


def check_connect_four(state: np.ndarray, player: int):
    n_rows, n_cols = state.shape

    # Check whether either player won
    match_pattern = np.asarray([1 for i in np.arange(4)])
    match_value = np.sum(match_pattern)
    match_state = (state == player).astype(int)

    # Check rows
    for i in np.arange(n_rows):
        row = match_state[i, :]
        values = np.convolve(row, match_pattern)
        if np.sum(values == match_value) > 0:
            print("row match: ", i)
            return True

    # Check cols
    for i in np.arange(n_cols):
        col = match_state[:, i]
        values = np.convolve(col, match_pattern)
        if np.sum(values == match_value) > 0:
            print("col match: ", i)
            return True

    # Check diagonals
    for i in np.arange(-n_rows, n_cols):
        diag = np.diagonal(match_state)
        values = np.convolve(diag, match_value)
        if np.sum(values == match_value) > 0:
            print("diag match: ", i)
            return True

    return False


A = np.asarray([[0, 1, 0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1, 0, 1]])

B = np.asarray([[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1]])

C = np.asarray(
    [
        [1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1],
    ]
)

check_connect_four(state=A, player=1)
check_connect_four(state=B, player=1)
check_connect_four(state=C, player=1)
check_connect_four(state=A * -1, player=-1)
check_connect_four(state=B * -1, player=-1)
check_connect_four(state=C * -1, player=-1)
