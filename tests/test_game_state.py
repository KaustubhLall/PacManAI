from pathlib import Path

from game.game_state import GameState

MAZE_1 = Path(__file__).resolve().parents[1] / "mazes" / "1.txt"


def test_reset_replaces_episode_state_without_duplicates():
    state = GameState(str(MAZE_1), pacman_lives=3, ghost_difficulty=0)
    initial_pellets = len(state.pellets)
    initial_ghosts = len(state.ghosts)

    state.pacman.score = 5
    state.remove_pellet(state.pellets[0])
    state.reset()

    assert state.pacman.score == 0
    assert state.pacman.lives == 3
    assert len(state.pellets) == initial_pellets
    assert len(state.ghosts) == initial_ghosts


def test_encoding_shape_matches_loaded_maze_dimensions():
    state = GameState(str(MAZE_1), pacman_lives=3, ghost_difficulty=0)

    assert state.get_encoding().shape == (state.board_height * state.board_width,)
    assert state.get_encoding_ql().shape == (state.board_height, state.board_width, 1)


def test_removed_pellet_disappears_from_q_learning_encoding():
    state = GameState(str(MAZE_1), pacman_lives=3, ghost_difficulty=0)
    pellet = state.pellets[0]

    state.remove_pellet(pellet)

    assert state.get_encoding_ql()[pellet.y, pellet.x, 0] != 1


def test_game_over_covers_loss_and_win_conditions():
    state = GameState(str(MAZE_1), pacman_lives=3, ghost_difficulty=0)
    assert not state.is_game_over()

    state.pacman.lives = 0
    assert state.is_game_over()

    state = GameState(str(MAZE_1), pacman_lives=3, ghost_difficulty=0)
    state.pellets.clear()
    assert state.is_game_over()
