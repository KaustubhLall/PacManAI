import random
from collections import deque

import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from game.game_logic import GameLogic
from game.game_state import GameState, print_board


class PacmanEnv:
    def __init__(self, filename, pacman_lives, ghost_difficulty):
        self.filename = filename
        self.pacman_lives = pacman_lives
        self.ghost_difficulty = ghost_difficulty
        self.game_state = None
        self.game_logic = None
        self.prev_score = 0
        self.reset()

    def step(self, action):
        self.game_logic.update(*action)
        current_score = self.game_state.get_score()
        reward = (current_score - self.prev_score) - (self.pacman_lives - self.game_state.pacman.lives) * 100
        self.prev_score = current_score
        done = self.game_state.is_game_over()
        next_state = self.game_state.get_encoding_ql()
        return (next_state, self._get_extra_features()), reward, done

    def reset(self):
        self.game_state = GameState(self.filename, self.pacman_lives, self.ghost_difficulty)
        self.game_logic = GameLogic(self.game_state)
        self.prev_score = 0
        return self.game_state.get_encoding_ql(), self._get_extra_features()

    def _get_extra_features(self):
        closest_ghost = min(self.game_state.ghosts,
                            key=lambda g: abs(g.x - self.game_state.pacman.x) + abs(g.y - self.game_state.pacman.y))
        return np.array([self.game_state.pacman.x, self.game_state.pacman.y, closest_ghost.x, closest_ghost.y])

    def render(self):
        print_board(self.game_state)


class DQNAgent:
    def __init__(self, grid_size, num_channels, num_extra_features, actions):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.num_extra_features = num_extra_features
        self.action_size = len(actions)
        self.actions = actions
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        grid_input = Input(shape=(self.grid_size[0], self.grid_size[1], self.num_channels))
        conv = Conv2D(32, kernel_size=3, activation='relu')(grid_input)
        conv = Conv2D(64, kernel_size=3, activation='relu')(conv)
        flat = Flatten()(conv)

        extra_input = Input(shape=(self.num_extra_features,))
        concat = Concatenate()([flat, extra_input])

        hidden = Dense(256, activation='relu')(concat)
        output = Dense(self.action_size, activation='linear')(hidden)

        model = Model(inputs=[grid_input, extra_input], outputs=output)
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        grid_state, extra_features = state
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        act_values = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]])
        act_idx = np.argmax(act_values[0])
        return self.actions[act_idx]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            grid_state, extra_features = state
            if not done:
                grid_next_state, extra_next_features = next_state
                target += self.gamma * np.amax(
                    self.model.predict([grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]])[0])
            target_f = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]])
            action_index = self.actions.index(action)  # find the index of action
            target_f[0][action_index] = target  # update target at index of action
            self.model.fit([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]], target_f, epochs=1,
                           verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
