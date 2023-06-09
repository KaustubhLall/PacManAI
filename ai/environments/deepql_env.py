import json
import random
from datetime import datetime

import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Lambda, concatenate
from keras.models import Model
from keras.optimizers import Nadam

from ai.environments.sumtree import SumTree
from game.game_logic import GameLogic, _heuristic
from game.game_state import GameState, print_board

VERBOSITY = 0

EPSILON = 1e-5  # small constant to prevent division by zero
MAX_GHOST_PENALTY = 1  # This value might need to be adjusted based on your observations


# todo implement double Q-learning
# todo switch to prioritized experience replay


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
        self.time_since_last_pellet = 5  # Initialize to a larger value
        self.ghost_distance_threshold = 15
        self.pellet_distance_threshold = 5
        self.reset()

    def get_distance(self, start, end):
        visited = np.zeros((self.game_state.board_height, self.game_state.board_width))
        queue = [(0, start)]  # store nodes with their priorities
        paths = {start: []}

        while queue:
            priority, (x, y) = min(queue)  # Check node with lowest priority
            x, y = x % self.game_state.board_width, y % self.game_state.board_height  # Wrap coordinates
            queue.remove((priority, (x, y)))  # Remove this node from queue

            if (x, y) == end:
                return len(paths[(x, y)])  # Return length of the path

            if visited[y][x] == 0:
                visited[y][x] = 1

                for dx, dy in self.game_logic.get_next_moves(x, y):
                    next_x, next_y = (x + dx) % self.game_state.board_width, (y + dy) % self.game_state.board_height
                    if visited[next_y][next_x] == 0:
                        new_priority = priority + 1 + _heuristic((next_x, next_y), end)
                        queue.append((new_priority, (next_x, next_y)))
                        paths[(next_x, next_y)] = paths[(x, y)] + [(dx, dy)]

        raise ValueError("No valid path found in A*")

    def step(self, action):
        self.game_logic.update(*action)
        current_score = self.game_state.get_score()
        current_lives = self.game_state.pacman.lives

        # Calculate each reward component
        score_reward = (1 if (current_score - self.prev_score) else 0)

        if self.prev_lives > current_lives:  # if a life was lost
            lives_penalty = ((self.prev_lives - current_lives) * -1)
        else:
            lives_penalty = 0

        # Using A* to calculate pellet distance
        pellet_distance = self.get_distance(
            (self.game_state.pacman.x, self.game_state.pacman.y),
            (self.closest_pellet.x, self.closest_pellet.y)
        )
        pellet_reward = 1 - pellet_distance / self.pellet_distance_threshold  # Scales from 0 to 1 as distance decreases

        ghost_penalty = 0
        for ghost in self.game_state.ghosts:
            # Using A* to calculate ghost distance
            distance = self.get_distance(
                (self.game_state.pacman.x, self.game_state.pacman.y),
                (ghost.x, ghost.y)
            )
            if distance < self.ghost_distance_threshold:
                penalty = np.log(self.ghost_distance_threshold) - np.log(distance + EPSILON)
                penalty = penalty / np.log(self.ghost_distance_threshold)
                ghost_penalty -= penalty

        time_alive_reward = 0.01
        if self.prev_lives == current_lives:  #
            self.time_alive += 1
            time_alive_reward = self.time_alive / 200

            # Calculate time since last pellet was eaten
        if current_score > self.prev_score:
            time_since_last_pellet_penalty = 0
            self.time_since_last_pellet = 0
        else:
            self.time_since_last_pellet += 1
            time_since_last_pellet_penalty = self.time_since_last_pellet / 5  # Normalize penalty

        # Combine rewards
        reward = ghost_penalty * 4 + pellet_reward + lives_penalty * 10 + time_alive_reward - \
                 time_since_last_pellet_penalty + score_reward

        reward_info = {
            'score_reward': score_reward,
            'ghost_penalty': ghost_penalty,
            'pellet_reward': pellet_reward,
            'lives_penalty': lives_penalty,
            'time_alive_reward': time_alive_reward,
            'time_since_last_pellet_penalty': time_since_last_pellet_penalty,
            'total_reward': reward
        }

        self.prev_score = current_score
        self.prev_lives = current_lives

        done = self.game_state.is_game_over()
        next_state = self.game_state.get_encoding_ql()
        return (next_state, self._get_extra_features()), reward, done, reward_info

    def reset(self):
        self.game_state = GameState(self.filename, self.pacman_lives, self.ghost_difficulty)
        self.game_logic = GameLogic(self.game_state)
        self.prev_score = 0
        self.time_alive = 0
        self.time_since_last_pellet = 20
        return self.game_state.get_encoding_ql(), self._get_extra_features()

    def _get_extra_features(self):
        closest_ghost = min(self.game_state.ghosts,
                            key=lambda g: abs(g.x - self.game_state.pacman.x) + abs(g.y - self.game_state.pacman.y))

        closest_pellet = min(self.game_state.pellets,
                             key=lambda p: abs(p.x - self.game_state.pacman.x) + abs(p.y - self.game_state.pacman.y))
        self.closest_pellet = closest_pellet
        # Calculate relative directions (opposite of dx, dy)
        dir_ghost = (np.sign(self.game_state.pacman.x - closest_ghost.x),
                     np.sign(self.game_state.pacman.y - closest_ghost.y))
        dir_pellet = (np.sign(self.game_state.pacman.x - closest_pellet.x),
                      np.sign(self.game_state.pacman.y - closest_pellet.y))

        return np.array([self.game_state.pacman.lives, self.game_state.pacman.score,
                         dir_ghost[0], dir_ghost[1], dir_pellet[0], dir_pellet[1],
                         self.time_alive / 1000, self.time_since_last_pellet / 100])  # Normalize time features

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
        self.memory = SumTree(75000)  # Experience replay memory with SumTree
        self.alpha = 0.95  # control how much prioritization is used
        self.gamma = 0.6  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # 0.01 default
        self.epsilon_decay = 0.9
        self.lr = 1e-3
        self.absolute_error_upper = 1.

        self.model = self._build_model()
        self.model_target = self._build_model()

        if load is not None:
            self.load(load)

    def _build_model(self):
        # Input layers
        grid_input = Input(shape=(self.grid_size[0], self.grid_size[1], self.num_channels))
        extra_input = Input(shape=(self.num_extra_features,))

        # Convolution layers with batch normalization
        act_fn = 'relu'

        conv1 = Conv2D(32, kernel_size=3, activation=act_fn, padding='same')(grid_input)
        conv1 = BatchNormalization()(conv1)

        conv2 = Conv2D(32, kernel_size=3, activation=act_fn, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)

        flat = Flatten()(conv2)

        # Concatenate grid and extra features
        merged = concatenate([flat, extra_input])

        # Fully connected layers with batch normalization
        hidden = Dense(128, activation=act_fn)(merged)
        hidden = BatchNormalization()(hidden)
        hidden = Dense(64, activation=act_fn)(hidden)
        hidden = BatchNormalization()(hidden)

        # Dueling DQN architecture
        state_value = Dense(1)(hidden)
        action_advantages = Dense(self.action_size)(hidden)

        # Combine streams into final Q Values
        output = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)),
                        output_shape=(self.action_size,))([state_value, action_advantages])

        model = Model(inputs=[grid_input, extra_input], outputs=output)

        # Using a dynamic learning rate

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
        minibatch = []
        segment = self.memory.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            priorities.append(idx)
            minibatch.append(data)

        for idx, (state, action, reward, next_state, done, reward_info) in zip(priorities, minibatch):
            target = reward
            grid_state, extra_features = state
            if not done:
                grid_next_state, extra_next_features = next_state

                # get the action with max Q-value in the current network
                act_values = self.model.predict(
                    [grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]], verbose=VERBOSITY)
                action_max = np.argmax(act_values[0], )

                # get the Q-value for the selected action from the target network
                act_values_target = self.model_target.predict(
                    [grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]], verbose=VERBOSITY)
                target += self.gamma * act_values_target[0][action_max]

            target_f = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]],
                                          verbose=VERBOSITY)
            action_index = self.actions.index(action)  # find the index of action
            target_f[0][action_index] = target  # update target at index of action

            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            self.model.fit([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]], target_f, epochs=1,
                           verbose=VERBOSITY,
                           # callbacks=[tensorboard_callback]
                           )

            self.memory.update(idx, abs(target - target_f[0][action_index]))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # copy weights from model to model_target
        self.model_target.set_weights(self.model.get_weights())

    def load(self, name):
        # Load the weights into the model
        self.model.load_weights(name + '_weights.h5')

        # Load and return the metadata
        try:
            with open(name + '_metadata.json', 'r') as json_file:
                metadata = json.load(json_file)
        except:
            print("Couldn't read metadata.")
            metadata = None
        return metadata

    def save(self, name, score, episode_number, final=False):
        # Create a metadata dictionary
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

        # Save the model weights and architecture
        if final:
            self.model.save(name + '.h5')
        else:
            self.model.save_weights(name + '_weights.h5')

        # Save the model architecture to JSON
        model_json = self.model.to_json()
        with open(name + '_architecture.json', 'w') as json_file:
            json_file.write(model_json)

        # Save the metadata
        with open(name + '_metadata.json', 'w') as json_file:
            json.dump(metadata, json_file)
