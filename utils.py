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
            return True

    # Check cols
    for i in np.arange(n_cols):
        col = match_state[:, i]
        values = np.convolve(col, match_pattern)
        if np.sum(values == match_value) > 0:
            return True

    # Check diagonals
    for i in np.arange(-n_rows, n_cols):
        diag = np.diagonal(match_state)
        values = np.convolve(diag, match_pattern)
        if np.sum(values == match_value) > 0:
            return True

    return False
