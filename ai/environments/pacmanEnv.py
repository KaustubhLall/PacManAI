import numpy as np
import tensorflow as tf
from tfne.environments.base_environment import BaseEnvironment
from tfne.helper_functions import read_option_from_config

from game.game_logic import GameLogic
from game.game_state import GameState


class PacmanCoDeepNEATEnv(BaseEnvironment):
    def __init__(self, game_map, pacman_lives, ghost_difficulty, weight_training, config=None, verbosity=0, **kwargs):

        self.game_state = GameState(game_map, pacman_lives, ghost_difficulty)
        self.game_logic = GameLogic(self.game_state)

        self.weight_training = weight_training
        self.config = config
        self.verbosity = verbosity

        if weight_training:
            if config is not None:
                self.eval_genome_fitness = self._eval_genome_fitness_weight_training
                self.epochs = read_option_from_config(config, 'EVALUATION', 'epochs')
                self.batch_size = read_option_from_config(config, 'EVALUATION', 'batch_size')
            else:
                self.eval_genome_fitness = self._eval_genome_fitness_weight_training
                self.epochs = kwargs['epochs']
                self.batch_size = kwargs['batch_size']
        else:
            raise NotImplementedError("Pacman environment non-weight training evaluation not yet implemented")

        self.accuracy_metric = tf.keras.metrics.Accuracy()
        self.actions = [(-1, 0), (0, 0), (0, 1), (0, -1), (1, 0), ]  # possible actions

    def reset(self):
        self.game_state.reset()

    def step(self, action):
        self.game_logic.update(*action)
        next_state = self.game_state.get_current_state()
        done = self.game_state.is_game_over()
        reward = self.game_state.get_score() + self.game_state.pacman.lives * 10 - (25 if done else 0)
        return next_state, reward, done

    def eval_genome_fitness(self, genome) -> float:
        raise RuntimeError()

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        model = genome.get_model()
        optimizer = genome.get_optimizer()

        # model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))

        self.reset()
        state = self.game_state.get_current_state()
        done = False
        total_reward = 0
        i = 0
        while not done:
            prediction = model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(prediction)
            action = self.actions[action_index]
            next_state, reward, done = self.step(action)
            total_reward += reward
            state = next_state
            print(f'Iter: {i} reward: {reward}')
            i += 1

        return total_reward

    def replay_genome(self, genome):
        print("Replaying Genome #{}:".format(genome.get_id()))
        model = genome.get_model()

        self.reset()
        state = self.game_state.get_current_state()
        done = False
        total_reward = 0
        while not done:
            prediction = model.predict(np.array([state]))
            action_index = np.argmax(prediction)
            action = self.actions[action_index]
            next_state, reward, done = self.step(action)
            total_reward += reward
            state = next_state

        print("Achieved Fitness:\t{}\n".format(total_reward))

    def get_input_shape(self):
        return 32, 32

    def get_output_shape(self) -> (int,):
        return (5,)

    def duplicate(self):
        return PacmanCoDeepNEATEnv(self.game_state.filename, self.game_state.lives,
                                   self.game_state.ghost_difficulty, self.weight_training, config=self.config,
                                   verbosity=self.verbosity, epochs=self.epochs, batch_size=self.batch_size)
