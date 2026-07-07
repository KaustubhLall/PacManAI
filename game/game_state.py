'''game_state.py'''
import warnings

import numpy as np

from game.entities import Pacman, Ghost, Pellet


class GameState:
    """Mutable Pac-Man board state loaded from a text maze."""

    def __init__(self, filename, pacman_lives, ghost_difficulty):
        self.filename = filename
        self.lives = pacman_lives
        self.ghost_difficulty = ghost_difficulty
        self.pacman = None
        self.ghosts = []
        self.pellets = []
        self.board = []
        self.board_width = 0
        self.board_height = 0
        self.load_from_file(filename, pacman_lives, ghost_difficulty)

    def load_from_file(self, filename, pacman_lives, ghost_difficulty):
        """Load a maze and replace any previous episode state."""
        self.filename = filename
        self.lives = pacman_lives
        self.ghost_difficulty = ghost_difficulty
        self.pacman = None
        self.ghosts = []
        self.pellets = []

        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line.rstrip('\n') for line in file if line.rstrip('\n')]

        if not lines:
            raise ValueError(f"Maze file is empty: {filename}")

        self.board_height = len(lines)
        self.board_width = max(len(line) for line in lines)
        self.board = [[' ' for _ in range(self.board_width)] for _ in range(self.board_height)]

        for y, line in enumerate(lines):
            for x, char in enumerate(line.ljust(self.board_width)):
                if char == 'P':
                    self.add_pacman(Pacman(x, y, pacman_lives))
                elif char == 'G':
                    self.add_ghost(Ghost(x, y, ghost_difficulty))
                elif char == '#':
                    self.board[y][x] = '#'
                elif char == '.':
                    self.add_pellet(Pellet(x, y))

        if self.pacman is None:
            raise ValueError("No Pacman character found in the maze file.")
        if not self.ghosts:
            warnings.warn("No Ghost characters found in the maze file.", stacklevel=2)

    def add_pacman(self, pacman):
        self.pacman = pacman

    def add_ghost(self, ghost):
        self.ghosts.append(ghost)

    def add_pellet(self, pellet):
        self.pellets.append(pellet)

    def remove_pellet(self, pellet):
        self.pellets.remove(pellet)

    def _compose_board(self, include_entities=True):
        """Return a current board copy with remaining pellets and live entities."""
        board = [row.copy() for row in self.board]

        for pellet in self.pellets:
            board[pellet.y][pellet.x] = '.'

        if include_entities and self.pacman is not None:
            board[self.pacman.y][self.pacman.x] = 'P'
            for ghost in self.ghosts:
                board[ghost.y][ghost.x] = 'G'

        return board

    def get_encoding(self):
        """Return a flat numeric encoding sized to the loaded maze."""
        mapping = {' ': 0, '#': 1, '.': 2, 'P': 3, 'G': 4}
        encoded_board = np.zeros((self.board_height, self.board_width))

        for y, row in enumerate(self._compose_board(include_entities=True)):
            for x, cell in enumerate(row):
                encoded_board[y, x] = mapping.get(cell, 0)

        return encoded_board.flatten()

    def get_encoding_ql(self):
        """Return a channel-last board encoding for Q-learning models."""
        mapping = {' ': -1, '#': 0, '.': 1, 'P': 2, 'G': -2}
        encoded_board = np.zeros((self.board_height, self.board_width, 1))

        for y, row in enumerate(self._compose_board(include_entities=True)):
            for x, cell in enumerate(row):
                encoded_board[y, x, 0] = mapping.get(cell, -1)

        return encoded_board

    def get_board(self, include_entities=True):
        return np.array(self._compose_board(include_entities=include_entities))

    def reset(self):
        self.load_from_file(self.filename, self.lives, self.ghost_difficulty)
        return self.get_current_state()

    def is_game_over(self):
        return self.pacman.lives <= 0 or len(self.pellets) == 0

    def get_current_state(self):
        return self.get_encoding()

    def get_score(self):
        return self.pacman.score


def print_board(gamestate):
    board_copy = gamestate._compose_board(include_entities=True)

    emoji_board = []
    for row in board_copy:
        emoji_row = []
        for cell in row:
            if cell == 'P':
                emoji_row.append('😮')
            elif cell == 'G':
                emoji_row.append('👻')
            elif cell == '.':
                emoji_row.append('🍒')
            elif cell == '#':
                emoji_row.append('🟦')
            else:
                emoji_row.append('⬛')
        emoji_board.append(emoji_row)

    return '\n'.join([' '.join(row) for row in emoji_board])
