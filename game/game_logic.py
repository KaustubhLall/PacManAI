'''game_logic.py'''
from random import choice

import numpy as np


def _heuristic(cell, target, metric="manhattan"):
    dx = abs(target[0] - cell[0])
    dy = abs(target[1] - cell[1])

    if metric == "manhattan":
        return dx + dy
    elif metric == "euclidean":
        return (dx ** 2 + dy ** 2) ** 0.5
    elif metric == "chebyshev":
        return max(dx, dy)
    elif metric == "octile":
        return max(dx, dy) + 0.414 * min(dx, dy)
    elif metric == "euclidean_squared":
        return dx ** 2 + dy ** 2
    else:
        raise ValueError("Invalid metric: " + metric)


class GameLogic:
    # public interface
    def __init__(self, game_state):
        self.game_state = game_state
        self.default_dx, self.default_dy = 1, 0  # Initialize default direction

    def update(self, pacman_dx, pacman_dy):
        # If no input is given (i.e., pacman_dx, pacman_dy == 0, 0), use the default direction
        if pacman_dx == 0 and pacman_dy == 0:
            pacman_dx, pacman_dy = self.default_dx, self.default_dy

        next_x = (self.game_state.pacman.x + pacman_dx) % self.game_state.board_width
        next_y = (self.game_state.pacman.y + pacman_dy) % self.game_state.board_height

        # If new move is invalid, use the default move
        if not self._is_valid_cell(next_x, next_y):
            pacman_dx, pacman_dy = self.default_dx, self.default_dy

        next_x = (self.game_state.pacman.x + pacman_dx) % self.game_state.board_width
        next_y = (self.game_state.pacman.y + pacman_dy) % self.game_state.board_height

        # If the default move is invalid, stand still
        if not self._is_valid_cell(next_x, next_y):
            pacman_dx, pacman_dy = 0, 0

        self._move_pacman(pacman_dx, pacman_dy)
        self._move_ghosts()

    def get_game_state(self):
        return self.game_state

    # logic helpers
    # in GameLogic
    def _move_pacman(self, dx, dy):
        next_x = (self.game_state.pacman.x + dx) % self.game_state.board_width
        next_y = (self.game_state.pacman.y + dy) % self.game_state.board_height

        # Check if next cell is valid and not a wall
        if self._is_valid_cell(next_x, next_y):
            # Update the default direction with the current move direction
            self.default_dx, self.default_dy = dx, dy
            self.game_state.pacman.move(dx, dy)

            # Wrap Pacman's coordinates
            self.game_state.pacman.x = next_x
            self.game_state.pacman.y = next_y

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

    def _move_ghosts(self):
        for ghost in self.game_state.ghosts:
            dx, dy = self._ghost_next_move(ghost)
            next_x = (ghost.x + dx) % self.game_state.board_width
            next_y = (ghost.y + dy) % self.game_state.board_height

            # Check for collision with walls and other ghosts
            if not self._is_valid_cell(next_x, next_y) or self._is_collision_with_ghosts(next_x, next_y):
                continue

            ghost.move(dx, dy)

            # Wrap Ghost's coordinates
            ghost.x = next_x
            ghost.y = next_y

            self._check_ghost_collision(ghost)

    def _is_collision_with_ghosts(self, x, y):
        # Wrap coordinates
        x = x % self.game_state.board_width
        y = y % self.game_state.board_height

        for ghost in self.game_state.ghosts:
            if x == ghost.x and y == ghost.y:
                return True
        return False

    def _check_ghost_collision(self, ghost):
        if self.game_state.pacman.x == ghost.x and self.game_state.pacman.y == ghost.y:
            self.game_state.pacman.lives -= 1
            ghost.x, ghost.y = ghost.start_x, ghost.start_y  # move ghost back to starting position

    def _is_valid_cell(self, x, y):
        # Wrap coordinates
        x = x % self.game_state.board_width
        y = y % self.game_state.board_height

        if self.game_state.board[y][x] == '#':
            return False
        return True

    # GHOST AI
    def _ghost_next_move(self, ghost):
        if ghost.difficulty == 0:
            return 0, 0  # static ghost, does not move
        elif ghost.difficulty == 1:
            return self.alg_dbfs(ghost)  # randomly moving ghost
        elif ghost.difficulty == 2:
            return self.alg_dfs(ghost, max_depth=15)  # dfs-based ghost
        elif ghost.difficulty == 3:
            return self.alg_a_star(ghost)
        return 0, 0  # default, does not move

    def get_next_moves(self, x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        next_moves = []

        for dx, dy in directions:
            next_x, next_y = (x + dx) % self.game_state.board_width, (y + dy) % self.game_state.board_height
            if self._is_valid_cell(next_x, next_y):
                next_moves.append((dx, dy))

        return next_moves

    def alg_dfs(self, ghost, max_depth=10):
        stack = [(ghost.x, ghost.y, 0)]  # store nodes with their depth
        visited = np.zeros((self.game_state.board_height, self.game_state.board_width))
        paths = {(ghost.x, ghost.y): []}
        fallback_path = None  # Added fallback_path to store a valid path

        while stack:
            x, y, depth = stack[-1]  # Check the top of the stack without popping
            x, y = x % self.game_state.board_width, y % self.game_state.board_height  # Wrap coordinates

            if (x, y) == (self.game_state.pacman.x, self.game_state.pacman.y):
                return paths[(x, y)][0] if paths[(x, y)] else (0, 0)  # return first step or stay still

            if visited[y][x] == 0 and depth <= max_depth:
                visited[y][x] = 1
                stack.pop()  # Now pop the node since it's visited

                for dx, dy in self.get_next_moves(x, y):
                    next_x, next_y = (x + dx) % self.game_state.board_width, (y + dy) % self.game_state.board_height
                    if visited[next_y][next_x] == 0:
                        stack.append((next_x, next_y, depth + 1))
                        paths[(next_x, next_y)] = paths[(x, y)] + [(dx, dy)]

                        # Store the first valid move as a fallback path
                        if fallback_path is None:
                            fallback_path = paths[(next_x, next_y)][0]
            else:
                stack.pop()  # Pop the node if it is visited or the depth is exceeded

        # If DFS didn't find a path within max_depth, return the fallback_path
        if fallback_path is not None:
            return fallback_path

        raise ValueError("No valid path found in DFS")

    def alg_a_star(self, ghost):
        start = (ghost.x, ghost.y)
        target = (self.game_state.pacman.x, self.game_state.pacman.y)
        visited = np.zeros((self.game_state.board_height, self.game_state.board_width))
        queue = [(0, start)]  # store nodes with their priorities
        paths = {start: []}

        while queue:
            priority, (x, y) = min(queue)  # Check node with lowest priority
            x, y = x % self.game_state.board_width, y % self.game_state.board_height  # Wrap coordinates
            queue.remove((priority, (x, y)))  # Remove this node from queue

            if (x, y) == target:
                return paths[(x, y)][0] if paths[(x, y)] else (0, 0)  # return first step or stay still

            if visited[y][x] == 0:
                visited[y][x] = 1

                for dx, dy in self.get_next_moves(x, y):
                    next_x, next_y = (x + dx) % self.game_state.board_width, (y + dy) % self.game_state.board_height
                    if visited[next_y][next_x] == 0:
                        new_priority = priority + 1 + _heuristic((next_x, next_y), target)
                        queue.append((new_priority, (next_x, next_y)))
                        paths[(next_x, next_y)] = paths[(x, y)] + [(dx, dy)]

        raise ValueError("No valid path found in A*")

    def alg_dbfs(self, ghost):
        # Get pacman's position
        pacman_x, pacman_y = self.game_state.pacman.x, self.game_state.pacman.y

        # Check current direction towards pacman
        direction = (np.sign(pacman_x - ghost.x), np.sign(pacman_y - ghost.y))

        # Get valid moves
        moves = self.get_next_moves(ghost.x, ghost.y)

        # If the ghost has no possible moves, it should remain stationary
        if len(moves) == 0:
            return 0, 0

        # If the ghost's current direction towards pacman is a valid move, choose it
        if direction in moves:
            return direction

        # Otherwise, try a move perpendicular to the preferred direction
        perpendicular_moves = [(direction[1], direction[0]), (-direction[1], -direction[0])]
        for move in perpendicular_moves:
            if move in moves:
                return move

        # If neither the preferred nor perpendicular directions are available, choose a valid random move
        return choice(moves)
