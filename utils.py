"""Utilities used for training Connect-Four.

This module contains utility functions that are used to train and evaluate
policies for Connect-Four agents.

"""

import numpy as np
from gym import Env
from stable_baselines3.dqn.dqn import DQN


def check_connect_four(state: np.ndarray, player: int):
    """Determines whether a player has won a Connect Four game.

    Uses convolutional filters find a match of four sequential
    player pieces in a horizontal, vertical or diagonal arrangement.

    Args:
        state: An array that defines the current state of the Connect Four game
        player: An integer ('1' or '2') that represents the player for which
        a connect four match is determined.

    Returns:
        A boolean value, where 'True' indicates that the player has a
        connect four match (thereby winning the game).

    """
    n_rows, n_cols = state.shape

    # Create match pattern of four sequential player pieces
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
    for i in np.arange(-n_rows + 1, n_cols):
        diag = np.diagonal(match_state, i)
        values = np.convolve(diag, match_pattern)
        if np.sum(values == match_value) > 0:
            return True
    # Rotate clockwise by 270-degrees to check other diagonal direction
    match_state_rot = np.rot90(match_state, k=3)
    for i in np.arange(-n_cols + 1, n_rows):
        diag = np.diagonal(match_state_rot, i)
        values = np.convolve(diag, match_pattern)
        if np.sum(values == match_value) > 0:
            return True

    return False


def highlight_match(state: np.ndarray):
    """Returns all board spaces where connect-fours occur.

    Uses convolutional filters find a match of four sequential
    player pieces in a horizontal, vertical or diagonal arrangement.

    Args:
        state: An array that defines the current state of the Connect Four game
        player: An integer ('1' or '2') that represents the player for which
        a connect four match is determined.

    Returns:
        Array with 1s in locations where connect-fours occur.

    """
    # Determine whether any player has a connect four.
    player = None
    highlight = np.zeros(state.shape)
    if check_connect_four(state=state, player=1):
        player = 1
    elif check_connect_four(state=state, player=2):
        player = 2
    else:
        return highlight

    n_rows, n_cols = state.shape

    # Create match pattern of four sequential player pieces
    match_pattern = np.asarray([1 for i in np.arange(4)])
    match_value = np.sum(match_pattern)
    match_state = (state == player).astype(int)

    # Check rows
    for i in np.arange(n_rows):
        row = match_state[i, :]
        values = np.convolve(row, match_pattern)
        if np.sum(values == match_value) > 0:
            idx = np.argmax((values == match_value).astype(int))
            highlight[i, idx - 3 : idx + 1] = match_pattern

    # Check cols
    for i in np.arange(n_cols):
        col = match_state[:, i]
        values = np.convolve(col, match_pattern)
        if np.sum(values == match_value) > 0:
            idx = np.argmax((values == match_value).astype(int))
            highlight[idx - 3 : idx + 1, i] = match_pattern

    # Check diagonals
    for i in np.arange(-n_rows + 1, n_cols):
        diag = np.diagonal(match_state, i)
        values = np.convolve(diag, match_pattern)
        if np.sum(values == match_value) > 0:
            idx = np.argmax((values == match_value).astype(int))
            start_row = int(abs(min(i, 0)))
            start_col = int(max(i, 0))
            for k in np.arange(4):
                highlight[start_row - 3 + k + idx, start_col - 3 + k + idx] = 1
    # Rotate clockwise by 270-degrees to check other diagonal direction
    match_state_rot = np.rot90(match_state, k=3)
    for i in np.arange(-n_cols + 1, n_rows):
        diag = np.diagonal(match_state_rot, i)
        values = np.convolve(diag, match_pattern)
        if np.sum(values == match_value) > 0:
            idx = np.argmax((values == match_value).astype(int))
            start_row = int(abs(min(i, 0)))
            start_col = int(max(i, 0))
            # Initialize temporary rotated array to store match pattern.
            rot_highlight = np.zeros((n_cols, n_rows))
            for k in np.arange(4):
                rot_highlight[start_row - 3 + k + idx, start_col - 3 + k + idx] = 1
            # Rotate temp array to normal position and include in result.
            highlight = highlight + np.rot90(rot_highlight, k=1)

    highlight = player * np.minimum(highlight, np.ones(highlight.shape))
    return highlight


def play_matches(
    *, player_policy: DQN, deterministic_player: bool, env: Env, num_games: int
):
    num_wins = np.zeros(2)
    for _ in np.arange(num_games):
        state = env.reset()
        done = False
        while not done:
            move = int(
                player_policy.predict(state, deterministic=deterministic_player)[0]
            )
            state, _, done, info = env.step(move)
        done_state = info["done_state"]
        if done_state == 1:
            num_wins[0] += 1
        if done_state == -1:
            num_wins[1] += 1

    return num_wins
