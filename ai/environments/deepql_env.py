import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Lambda, concatenate
from keras.models import Model
from keras.optimizers import Nadam

from ai.environments.sumtree import SumTree
from game.game_logic import GameLogic, _heuristic
from game.game_state import GameState, print_board

VERBOSITY = 0
EPSILON = 1e-5  # small constant to prevent division by zero


class PacmanEnv:
    def __init__(self, filename, pacman_lives, ghost_difficulty):
        self.filename = filename
        self.pacman_lives = pacman_lives
        self.ghost_difficulty = ghost_difficulty
        self.game_state = None
        self.game_logic = None
        self.prev_score = 0
        self.prev_lives = pacman_lives
        self.time_alive = 0
        self.time_since_last_pellet = 5
        self.ghost_distance_threshold = 15
        self.pellet_distance_threshold = 5
        self.closest_pellet = None
        self.reset()

    def get_distance(self, start, end):
        """Return path distance between two cells, or a large fallback if unreachable."""
        visited = np.zeros((self.game_state.board_height, self.game_state.board_width))
        queue = [(0, start)]
        paths = {start: []}

        while queue:
            priority, (x, y) = min(queue)
            x, y = x % self.game_state.board_width, y % self.game_state.board_height
            queue.remove((priority, (x, y)))

            if (x, y) == end:
                return len(paths[(x, y)])

            if visited[y][x] == 0:
                visited[y][x] = 1

                for dx, dy in self.game_logic.get_next_moves(x, y):
                    next_x, next_y = (x + dx) % self.game_state.board_width, (y + dy) % self.game_state.board_height
                    if visited[next_y][next_x] == 0:
                        new_priority = priority + 1 + _heuristic((next_x, next_y), end)
                        queue.append((new_priority, (next_x, next_y)))
                        paths[(next_x, next_y)] = paths[(x, y)] + [(dx, dy)]

        return self.game_state.board_width + self.game_state.board_height

    def step(self, action):
        self.game_logic.update(*action)
        current_score = self.game_state.get_score()
        current_lives = self.game_state.pacman.lives
        pacman_position = (self.game_state.pacman.x, self.game_state.pacman.y)

        score_reward = current_score - self.prev_score
        lives_penalty = -1 * max(0, self.prev_lives - current_lives)

        if self.game_state.pellets:
            closest_pellet = self._closest(self.game_state.pellets)
            self.closest_pellet = closest_pellet
            pellet_distance = self.get_distance(pacman_position, (closest_pellet.x, closest_pellet.y))
            pellet_reward = 1 - pellet_distance / self.pellet_distance_threshold
        else:
            pellet_distance = 0
            pellet_reward = 1.0
            self.closest_pellet = None

        ghost_penalty = 0
        for ghost in self.game_state.ghosts:
            distance = self.get_distance(pacman_position, (ghost.x, ghost.y))
            if distance < self.ghost_distance_threshold:
                penalty = np.log(self.ghost_distance_threshold) - np.log(distance + EPSILON)
                penalty = penalty / np.log(self.ghost_distance_threshold)
                ghost_penalty -= penalty

        time_alive_reward = 0.01
        if self.prev_lives == current_lives:
            self.time_alive += 1
            time_alive_reward = self.time_alive / 200

        if current_score > self.prev_score:
            time_since_last_pellet_penalty = 0
            self.time_since_last_pellet = 0
        else:
            self.time_since_last_pellet += 1
            time_since_last_pellet_penalty = self.time_since_last_pellet / 5

        reward = (
            ghost_penalty * 4
            + pellet_reward
            + lives_penalty * 10
            + time_alive_reward
            - time_since_last_pellet_penalty
            + score_reward
        )

        done = self.game_state.is_game_over()
        if done and current_lives > 0 and not self.game_state.pellets:
            reward += 10

        reward_info = {
            'score_reward': score_reward,
            'ghost_penalty': ghost_penalty,
            'pellet_distance': pellet_distance,
            'pellet_reward': pellet_reward,
            'lives_penalty': lives_penalty,
            'time_alive_reward': time_alive_reward,
            'time_since_last_pellet_penalty': time_since_last_pellet_penalty,
            'total_reward': reward
        }

        self.prev_score = current_score
        self.prev_lives = current_lives

        next_state = self.game_state.get_encoding_ql()
        return (next_state, self._get_extra_features()), reward, done, reward_info

    def reset(self):
        self.game_state = GameState(self.filename, self.pacman_lives, self.ghost_difficulty)
        self.game_logic = GameLogic(self.game_state)
        self.prev_score = 0
        self.prev_lives = self.pacman_lives
        self.time_alive = 0
        self.time_since_last_pellet = 20
        self.closest_pellet = None
        return self.game_state.get_encoding_ql(), self._get_extra_features()

    def _closest(self, entities):
        return min(
            entities,
            key=lambda entity: abs(entity.x - self.game_state.pacman.x) + abs(entity.y - self.game_state.pacman.y)
        )

    def _direction_away_from(self, entity):
        if entity is None:
            return 0, 0
        return (
            np.sign(self.game_state.pacman.x - entity.x),
            np.sign(self.game_state.pacman.y - entity.y),
        )

    def _get_extra_features(self):
        closest_ghost = self._closest(self.game_state.ghosts) if self.game_state.ghosts else None
        closest_pellet = self._closest(self.game_state.pellets) if self.game_state.pellets else None
        self.closest_pellet = closest_pellet

        dir_ghost = self._direction_away_from(closest_ghost)
        dir_pellet = self._direction_away_from(closest_pellet)

        return np.array([self.game_state.pacman.lives, self.game_state.pacman.score,
                         dir_ghost[0], dir_ghost[1], dir_pellet[0], dir_pellet[1],
                         self.time_alive / 1000, self.time_since_last_pellet / 100])

    def render(self):
        sep = '='
        return f'{sep * 128}\nScore: {self.game_state.pacman.score}, Lives: {self.game_state.pacman.lives}\n' \
            + print_board(self.game_state)


class DQNAgent:
    def __init__(self, grid_size, num_channels, num_extra_features, actions, load=None):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.num_extra_features = num_extra_features
        self.action_size = len(actions)
        self.actions = actions
        self.memory = SumTree(75000)
        self.alpha = 0.95
        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9
        self.lr = 1e-3
        self.absolute_error_upper = 1.

        self.model = self._build_model()
        self.model_target = self._build_model()

        if load is not None:
            self.load(load)

    def _build_model(self):
        grid_input = Input(shape=(self.grid_size[0], self.grid_size[1], self.num_channels))
        extra_input = Input(shape=(self.num_extra_features,))

        act_fn = 'relu'

        conv1 = Conv2D(32, kernel_size=3, activation=act_fn, padding='same')(grid_input)
        conv1 = BatchNormalization()(conv1)

        conv2 = Conv2D(32, kernel_size=3, activation=act_fn, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)

        flat = Flatten()(conv2)
        merged = concatenate([flat, extra_input])

        hidden = Dense(128, activation=act_fn)(merged)
        hidden = BatchNormalization()(hidden)
        hidden = Dense(64, activation=act_fn)(hidden)
        hidden = BatchNormalization()(hidden)

        state_value = Dense(1)(hidden)
        action_advantages = Dense(self.action_size)(hidden)

        output = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)),
                        output_shape=(self.action_size,))([state_value, action_advantages])

        model = Model(inputs=[grid_input, extra_input], outputs=output)
        model.compile(optimizer=Nadam(learning_rate=self.lr), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done, reward_info):
        experience = (state, action, reward, next_state, done, reward_info)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.memory.add(max_priority, experience)

    def act(self, state):
        grid_state, extra_features = state
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        act_values = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]],
                                        verbose=VERBOSITY)
        act_idx = np.argmax(act_values[0])
        return self.actions[act_idx]

    def replay(self, batch_size):
        if len(self.memory) < batch_size or self.memory.total() <= 0:
            return

        minibatch = []
        segment = self.memory.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, _priority, data = self.memory.get(s)
            if data is None:
                continue
            priorities.append(idx)
            minibatch.append(data)

        for idx, (state, action, reward, next_state, done, _reward_info) in zip(priorities, minibatch):
            target = reward
            grid_state, extra_features = state
            if not done:
                grid_next_state, extra_next_features = next_state

                act_values = self.model.predict(
                    [grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]], verbose=VERBOSITY)
                action_max = np.argmax(act_values[0])

                act_values_target = self.model_target.predict(
                    [grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]], verbose=VERBOSITY)
                target += self.gamma * act_values_target[0][action_max]

            target_f = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]],
                                          verbose=VERBOSITY)
            action_index = self.actions.index(action)
            old_value = target_f[0][action_index]
            target_f[0][action_index] = target

            self.model.fit([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]], target_f, epochs=1,
                           verbose=VERBOSITY)

            self.memory.update(idx, abs(target - old_value))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name + '_weights.h5')

        try:
            with open(name + '_metadata.json', 'r', encoding='utf-8') as json_file:
                metadata = json.load(json_file)
        except (OSError, json.JSONDecodeError):
            print("Couldn't read metadata.")
            metadata = None
        return metadata

    def save(self, name, score, episode_number, final=False):
        path_prefix = Path(name)
        path_prefix.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'score': score,
            'episode_number': episode_number,
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'epsilon': self.epsilon,
                'gamma': self.gamma
            },
            'final_model': final
        }

        if final:
            self.model.save(str(path_prefix) + '.h5')
        else:
            self.model.save_weights(str(path_prefix) + '_weights.h5')

        model_json = self.model.to_json()
        with open(str(path_prefix) + '_architecture.json', 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)

        with open(str(path_prefix) + '_metadata.json', 'w', encoding='utf-8') as json_file:
            json.dump(metadata, json_file)
