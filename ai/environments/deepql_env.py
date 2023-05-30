import json
import random
from datetime import datetime

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Conv2D, Concatenate, BatchNormalization, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Nadam

from game.game_logic import GameLogic
from game.game_state import GameState, print_board

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
        self.reset()
        self.ghost_distance_threshold = 20
        self.pellet_distance_threshold = 10
        self.prev_lives = pacman_lives
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

        pellet_distance = abs(self.closest_pellet.x - self.game_state.pacman.x) + abs(
            self.closest_pellet.y - self.game_state.pacman.y)
        pellet_reward = 1 - pellet_distance / self.pellet_distance_threshold  # Scales from 0 to 1 as distance decreases

        ghost_penalty = 0
        for ghost in self.game_state.ghosts:
            distance = abs(self.game_state.pacman.x - ghost.x) + abs(self.game_state.pacman.y - ghost.y)
            if distance < self.ghost_distance_threshold:
                # Calculate penalty as a scaled logarithmic function
                penalty = np.log(self.ghost_distance_threshold) - np.log(distance + EPSILON)
                penalty = penalty / np.log(self.ghost_distance_threshold)  # Normalize to 0-1 range
                ghost_penalty -= penalty  # Subtract penalty to introduce the danger of ghosts

        # Combine rewards
        reward = ghost_penalty + pellet_reward + lives_penalty
        # print(f'score_reward :{score_reward:4.2f}, lives_penalty :{lives_penalty:4.2f}, ghost_penalty :'
        #     f'{ghost_penalty:4.2f}, pellet_reward :{pellet_reward:4.2f}, total : {reward:4.2f}')

        reward_info = {
            'score_reward': score_reward,
            'ghost_penalty': ghost_penalty,
            'pellet_reward': pellet_reward,
            'lives_penalty': lives_penalty,
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
        return self.game_state.get_encoding_ql(), self._get_extra_features()

    def _get_extra_features(self):
        closest_ghost = min(self.game_state.ghosts,
                            key=lambda g: abs(g.x - self.game_state.pacman.x) + abs(g.y - self.game_state.pacman.y))

        closest_pellet = min(self.game_state.pellets,
                             key=lambda p: abs(p.x - self.game_state.pacman.x) + abs(p.y - self.game_state.pacman.y))
        self.closest_pellet = closest_pellet
        # Calculate relative positions
        dx_ghost = closest_ghost.x - self.game_state.pacman.x
        dy_ghost = closest_ghost.y - self.game_state.pacman.y

        dx_pellet = closest_pellet.x - self.game_state.pacman.x
        dy_pellet = closest_pellet.y - self.game_state.pacman.y

        return np.array([self.game_state.pacman.lives, self.game_state.pacman.score,
                         dx_ghost, dy_ghost, dx_pellet, dy_pellet])

    def render(self):
        sep = '='
        return f'{sep * 128}\nScore: {self.game_state.pacman.score}, Lives: {self.game_state.pacman.lives}\n' \
               + print_board(self.game_state)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def __len__(self):
        return len(self.data)


class DQNAgent:
    def __init__(self, grid_size, num_channels, num_extra_features, actions, load=None):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.num_extra_features = num_extra_features
        self.action_size = len(actions)
        self.actions = actions
        self.memory = SumTree(50000)  # Experience replay memory with SumTree
        self.epsilon = 0.01  # small epsilon to ensure no zero priority
        self.alpha = 0.6  # control how much prioritization is used
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.2  # 0.01 default
        self.epsilon_decay = 0.995
        self.lr = 1e-4
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
        act_fn = 'selu'
        conv = Conv2D(16, kernel_size=3, activation=act_fn, padding='same')(grid_input)
        conv = BatchNormalization()(conv)
        conv = Conv2D(32, kernel_size=3, activation=act_fn, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(64, kernel_size=3, activation=act_fn, padding='same')(conv)
        conv = BatchNormalization()(conv)

        flat = Flatten()(conv)

        # Concatenation with extra features
        concat = Concatenate()([flat, extra_input])

        # Fully connected layers with batch normalization and dropout
        hidden = Dense(128, activation=act_fn)(concat)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.2)(hidden)
        hidden = Dense(64, activation=act_fn)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.2)(hidden)
        hidden = Dense(32, activation=act_fn)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.2)(hidden)

        # Dueling DQN architecture
        # Split into value and advantage streams
        hidden1 = Dense(32, activation=act_fn)(hidden)
        hidden2 = Dense(32, activation=act_fn)(hidden)
        state_value = Dense(1)(hidden1)
        action_advantages = Dense(self.action_size)(hidden2)

        # Combine streams into final Q Values
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
        act_values = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]])
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
                    [grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]])
                action_max = np.argmax(act_values[0])

                # get the Q-value for the selected action from the target network
                act_values_target = self.model_target.predict(
                    [grid_next_state[np.newaxis, ...], extra_next_features[np.newaxis, ...]])
                target += self.gamma * act_values_target[0][action_max]

            target_f = self.model.predict([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]])
            action_index = self.actions.index(action)  # find the index of action
            target_f[0][action_index] = target  # update target at index of action
            self.model.fit([grid_state[np.newaxis, ...], extra_features[np.newaxis, ...]], target_f, epochs=1,
                           verbose=0)
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
            self.model.save(name + '.h5')  # Saves both weights and architecture
        else:
            self.model.save_weights(name + '_weights.h5')

        # Save the model architecture to JSON
        model_json = self.model.to_json()
        with open(name + '_architecture.json', 'w') as json_file:
            json_file.write(model_json)

        # Save the metadata
        with open(name + '_metadata.json', 'w') as json_file:
            json.dump(metadata, json_file)
