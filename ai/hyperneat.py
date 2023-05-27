import itertools
import os

import neat
import numpy as np

from game.game_logic import GameLogic
from game.game_state import GameState, print_board
from pytorch_neat.activations import tanh_activation
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter

DEBUG = True




def make_net(genome, config, _batch_size):
    board_height = 15  # adjust these values based on your game's configuration
    board_width = 15

    # Create input and output coordinates for the network
    input_coords = list(itertools.product(np.linspace(-1, 1, board_height), np.linspace(-1, 1, board_width)))
    output_coords = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

    return AdaptiveLinearNet.create(
        genome,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        weight_threshold=0.4,
        batch_size=_batch_size,
        activation=tanh_activation,
        output_activation=tanh_activation,
        device="cpu",
    )


def activate_net(net, states, debug=False, step_num=0):
    outputs = net.activate(states).numpy()
    return np.argmax(outputs, axis=1)


def run(n_generations, n_processes):
    difficulty = 3
    lives = 1
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    envs = [PacmanEnv("./mazes/2.txt", lives, difficulty) for _ in range(n_processes)]

    evaluator = MultiEnvEvaluator(make_net, activate_net, envs=envs, batch_size=n_processes, max_env_steps=1000)

    def eval_genomes(genomes, config):
        for i, (_, genome) in enumerate(genomes):
            try:
                genome.fitness = evaluator.eval_genome(genome, config, debug=DEBUG and i % 100 == 0)
            except Exception as e:
                print(genome)
                raise e

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("log.json", evaluator.eval_genome)
    pop.add_reporter(logger)

    winner = pop.run(eval_genomes, n_generations)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1
    return generations


if __name__ == "__main__":
    run(10, 1)
