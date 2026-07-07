import argparse
import configparser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from neat import Config, DefaultGenome, DefaultStagnation, DefaultReproduction, DefaultSpeciesSet, StatisticsReporter, \
    StdOutReporter, Population
from neat.nn import FeedForwardNetwork

from game.game_logic import GameLogic
from game.game_state import GameState, print_board

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("neat.cfg")
DEFAULT_MAZE = PROJECT_ROOT / "mazes" / "2.txt"
DEFAULT_MAX_STEPS = 2_000


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max to avoid overflow
    return e_x / e_x.sum(axis=0)


def eval_genome(args):
    genome_id, genome, config = args
    net = FeedForwardNetwork.create(genome, config)
    game_state = GameState(str(DEFAULT_MAZE), pacman_lives=3, ghost_difficulty=3)
    game_logic = GameLogic(game_state)
    actions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]  # stay, right, down, left, up

    for _ in range(DEFAULT_MAX_STEPS):
        outputs = net.activate(np.concatenate([game_state.get_encoding(),
                                               [game_state.pacman.x / game_state.board_width,
                                                game_state.pacman.y / game_state.board_height]]))
        softmax_outputs = softmax(outputs)
        action = actions[np.argmax(softmax_outputs)]

        dx, dy = action
        game_logic.update(dx, dy)
        fitness = game_state.pacman.score
        lives = game_state.pacman.lives

        if game_state.pacman.lives <= 0:
            print(f'{genome_id} ENDED game with:\n {print_board(game_state)}')
            return fitness

        if len(game_state.pellets) == 0:
            print(f'{genome_id} WON game with:\n {print_board(game_state)}')
            return fitness + lives * 10

    # Prevent stalled genomes from running forever. Keeping some score reward still
    # distinguishes agents that made progress before timing out.
    return game_state.pacman.score - 1


def eval_fitness(genomes, config):
    with Pool() as pool:
        results = pool.map(eval_genome, [(genome_id, genome, config) for genome_id, genome in genomes])
    for (genome_id, genome), fitness in zip(genomes, results):
        genome.fitness = fitness


def update_config_file(config_file, input_size):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)

    config_parser['DefaultGenome']['num_inputs'] = str(input_size)

    with open(config_file, 'w', encoding='utf-8') as file:
        config_parser.write(file)


def run_neat(config_file=DEFAULT_CONFIG, generations=100):
    game_state = GameState(str(DEFAULT_MAZE), 1, ghost_difficulty=3)
    expected_inputs = len(game_state.get_encoding()) + 2

    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet,
                    DefaultStagnation, str(config_file))

    if config.genome_config.num_inputs != expected_inputs:
        raise ValueError(
            f"NEAT config expects {config.genome_config.num_inputs} inputs, "
            f"but {DEFAULT_MAZE} produces {expected_inputs}. "
            "Update [DefaultGenome].num_inputs or use the matching maze."
        )

    population = Population(config)

    population.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_fitness, generations)

    print('\nBest genome:\n{!s}'.format(winner))
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NEAT agent for the Pac-Man grid environment.")
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG, help="path to neat-python config")
    parser.add_argument("--generations", type=int, default=100, help="number of generations to run")
    args = parser.parse_args()

    run_neat(args.config_file, args.generations)
