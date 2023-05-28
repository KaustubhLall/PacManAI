'''game_state.py'''
import warnings

import numpy as np

from game.entities import Pacman, Ghost, Pellet


class GameState:
    def __init__(self, filename, pacman_lives, ghost_difficulty):
        self.pacman = None
        self.ghosts = []
        self.pellets = []
        self.lives = pacman_lives
        self.ghost_difficulty = ghost_difficulty
        self.load_from_file(filename, pacman_lives, ghost_difficulty)

    def load_from_file(self, filename, pacman_lives, ghost_difficulty):
        self.filename = filename
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.board_height = len(lines)
        self.board_width = len(lines[0].strip())

        self.board = [[' ' for _ in range(self.board_width)] for _ in range(self.board_height)]

        for y, line in enumerate(lines):
            for x, char in enumerate(line.strip()):
                if char == 'P':
                    self.add_pacman(Pacman(x, y, pacman_lives))
                    self.board[y][x] = '.'
                elif char == 'G':
                    self.add_ghost(Ghost(x, y, ghost_difficulty))
                    self.board[y][x] = '.'
                elif char == '#':
                    self.board[y][x] = '#'
                elif char == '.':
                    self.board[y][x] = '.'
                    self.add_pellet(Pellet(x, y))

        if self.pacman is None:
            raise Exception("No Pacman character found in the file.")
        if not self.ghosts:
            warnings.warn("No Ghost characters found in the file.")

    def add_pacman(self, pacman):
        self.pacman = pacman

    def add_ghost(self, ghost):
        self.ghosts.append(ghost)

    def add_pellet(self, pellet):
        self.pellets.append(pellet)

    def remove_pellet(self, pellet):
        self.pellets.remove(pellet)

    def get_encoding(self):
        encoded_board = np.zeros((self.board_height, self.board_width))

        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell == ' ':
                    encoded_board[y, x] = 0  # Empty cell
                elif cell == '#':
                    encoded_board[y, x] = 1  # Wall
                elif cell == '.':
                    encoded_board[y, x] = 2  # Pellet
                elif cell == 'P':
                    encoded_board[y, x] = 3  # Pacman
                elif cell == 'G':
                    encoded_board[y, x] = 4  # Ghost

        return encoded_board

    def get_board(self):
        return np.array(self.board)

    def reset(self):
        self.load_from_file(self.filename, self.lives, self.ghost_difficulty)

    def is_game_over(self):
        return self.pacman.lives <= 0

    def get_current_state(self):
        return self.get_encoding()

    def get_score(self):
        return self.pacman.score


def print_board(gamestate):
    # Create a copy of the board matrix
    board_copy = [row.copy() for row in gamestate.board]

    # Place entities on the board
    board_copy[gamestate.pacman.y][gamestate.pacman.x] = '😮'  # Pacman

    for pellet in gamestate.pellets:
        board_copy[pellet.y][pellet.x] = '🍚'  # Pellets

    for ghost in gamestate.ghosts:
        board_copy[ghost.y][ghost.x] = '👻'  # Ghosts

    # Translate the maze symbols to emojis
    emoji_board = []
    for row in board_copy:
        emoji_row = []
        for cell in row:
            if cell == ' ':
                emoji_row.append('⬛')
            elif cell == '#':
                emoji_row.append('🟦')
            elif cell == '.':
                emoji_row.append('⬛')  # Replacing consumed pellets with empty space
            else:
                emoji_row.append(cell)
        emoji_board.append(emoji_row)

    # Convert the emoji board to a string
    emoji_board_str = '\n'.join([' '.join(row) for row in emoji_board])
    return emoji_board_str
