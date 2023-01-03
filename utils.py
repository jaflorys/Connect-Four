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
    match_state_rot = np.rot90(np.rot90(np.rot90(match_state)))
    for i in np.arange(-n_cols + 1, n_rows):
        diag = np.diagonal(match_state_rot, i)
        values = np.convolve(diag, match_pattern)
        if np.sum(values == match_value) > 0:
            return True

    return False


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
