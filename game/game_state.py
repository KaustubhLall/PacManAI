'''game_state.py'''
import warnings
from game.entities import Pacman, Ghost, Pellet


class GameState:
    def __init__(self, filename, pacman_lives, ghost_difficulty):
        self.pacman = None
        self.ghosts = []
        self.pellets = []
        self.load_from_file(filename, pacman_lives, ghost_difficulty)

    def load_from_file(self, filename, pacman_lives, ghost_difficulty):
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
