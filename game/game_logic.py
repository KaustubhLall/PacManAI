'''game_logic.py'''
from collections import deque
from random import choice

import numpy as np

from game.game_state import print_board


class GameLogic:
    def __init__(self, game_state):
        self.game_state = game_state

    def move_pacman(self, dx, dy):
        next_x = self.game_state.pacman.x + dx
        next_y = self.game_state.pacman.y + dy

        # Check if next cell is valid and not a wall
        if self.is_valid_cell(next_x, next_y):
            self.game_state.pacman.move(dx, dy)

            # Check for pellet collision
            for pellet in self.game_state.pellets:
                if next_x == pellet.x and next_y == pellet.y:
                    self.game_state.pacman.score += 1
                    self.game_state.remove_pellet(pellet)
                    break

            # Check for ghost collision
            for ghost in self.game_state.ghosts:
                if next_x == ghost.x and next_y == ghost.y:
                    self.game_state.pacman.lives -= 1
                    break

    def ghost_next_move(self, ghost):
        if ghost.difficulty == 0:
            return 0, 0  # static ghost, does not move
        elif ghost.difficulty == 1:
            return choice([(0, 1), (1, 0), (0, -1), (-1, 0)])  # randomly moving ghost
        elif ghost.difficulty == 2:
            return self.dfs_next_move(ghost)  # dfs-based ghost
        elif ghost.difficulty == 3:
            path = self.bfs((ghost.x, ghost.y), (self.game_state.pacman.x, self.game_state.pacman.y))
            if path:
                return path[0]
        return 0, 0  # default, does not move

    def dfs_next_move(self, ghost):
        visited = np.zeros((self.game_state.board_height, self.game_state.board_width))
        stack = [(ghost.x, ghost.y)]
        parents = {(ghost.x, ghost.y): None}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        pacman = (self.game_state.pacman.x, self.game_state.pacman.y)

        while stack:
            x, y = stack.pop()
            if (x, y) == pacman:
                # Found pacman, get next move
                parent_x, parent_y = parents[(x, y)]
                dx, dy = x - parent_x, y - parent_y
                return dx, dy
            if visited[y][x] == 0:
                visited[y][x] = 1
                # Add all unvisited neighbors to the stack
                # Sort directions based on the heuristic
                directions.sort(key=lambda d: self.heuristic((x + d[0], y + d[1]), pacman))
                for dx, dy in directions:
                    next_x, next_y = x + dx, y + dy
                    if self.is_valid_cell(next_x, next_y) and visited[next_y][next_x] == 0:
                        stack.append((next_x, next_y))
                        parents[(next_x, next_y)] = (x, y)
        return 0, 0  # default, does not move

    def heuristic(self, cell, target):
        # Manhattan distance
        return abs(target[0] - cell[0]) + abs(target[1] - cell[1])

    def bfs(self, start, target):
        visited = np.zeros((self.game_state.board_height, self.game_state.board_width))
        queue = deque([start])
        paths = {start: []}

        while queue:
            x, y = queue.popleft()

            if (x, y) == target:
                return paths[(x, y)]

            if visited[y][x] == 0:
                visited[y][x] = 1

                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    next_x, next_y = x + dx, y + dy

                    if self.is_valid_cell(next_x, next_y) and visited[next_y][next_x] == 0:
                        queue.append((next_x, next_y))
                        paths[(next_x, next_y)] = paths[(x, y)] + [(dx, dy)]

        return None

    def move_ghosts(self):
        for ghost in self.game_state.ghosts:
            dx, dy = self.ghost_next_move(ghost)
            next_x, next_y = ghost.x + dx, ghost.y + dy

            # Check for collision with walls and other ghosts
            if not self.is_valid_cell(next_x, next_y) or self.is_collision_with_ghosts(next_x, next_y):
                continue

            ghost.x += dx
            ghost.y += dy
            self.check_ghost_collision(ghost)

    def is_collision_with_ghosts(self, x, y):
        for ghost in self.game_state.ghosts:
            if x == ghost.x and y == ghost.y:
                return True
        return False

    # Call this function in the main game loop
    def update(self, pacman_dx, pacman_dy):
        self.move_pacman(pacman_dx, pacman_dy)
        self.move_ghosts()

    def check_ghost_collision(self, ghost):
        if self.game_state.pacman.x == ghost.x and self.game_state.pacman.y == ghost.y:
            self.game_state.pacman.lives -= 1
            ghost.x, ghost.y = ghost.start_x, ghost.start_y  # move ghost back to starting position

    def is_valid_cell(self, x, y):
        if x < 0 or y < 0 or x >= self.game_state.board_width or y >= self.game_state.board_height:
            return False
        if self.game_state.board[y][x] == '#':
            return False
        return True

    def get_game_state(self):
        return self.game_state
