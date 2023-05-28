import configparser
from multiprocessing import Pool

import numpy as np
from neat import Config, DefaultGenome, DefaultStagnation, DefaultReproduction, DefaultSpeciesSet, StatisticsReporter, \
    StdOutReporter, Population
from neat.nn import FeedForwardNetwork

from game.game_logic import GameLogic
from game.game_state import GameState, print_board


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max to avoid overflow
    return e_x / e_x.sum(axis=0)


def ic(fitness, lives):
    pass


def eval_genome(args):
    genome_id, genome, config = args
    net = FeedForwardNetwork.create(genome, config)
    game_state = GameState('../mazes/1.txt', 1, 3)
    game_logic = GameLogic(game_state)
    actions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]  # Actions corresponding to stay, right, up, left, down

    while True:
        outputs = net.activate(np.concatenate([game_state.get_encoding(),
                                               [game_state.pacman.x / game_state.board_width,
                                                game_state.pacman.y / game_state.board_height]]))
        softmax_outputs = softmax(outputs)  # Apply softmax
        action = actions[np.argmax(softmax_outputs)]  # Select the action corresponding to the maximum softmax output

        dx, dy = action
        game_logic.update(dx, dy)
        fitness = game_state.pacman.score
        lives = game_state.pacman.lives
        # print(f"fitness, lives = {fitness, lives}")
        if game_state.pacman.lives <= 0:
            # print(f'ended game with:\n {print_board(game_state)}')
            return fitness - 25

        if len(game_state.pellets) == 0:
            return fitness + lives * 10



def eval_fitness(genomes, config):
    with Pool() as pool:
        results = pool.map(eval_genome, [(genome_id, genome, config) for genome_id, genome in genomes])
    for (genome_id, genome), fitness in zip(genomes, results):
        genome.fitness = fitness






def update_config_file(config_file, input_size):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)

    config_parser['DefaultGenome']['num_inputs'] = str(input_size)

    with open(config_file, 'w') as file:
        config_parser.write(file)


def run_neat(config_file):
    game_state = GameState('../mazes/2.txt', 1, ghost_difficulty=3)
    input_size = len(game_state.get_encoding())

    # update_config_file(config_file, input_size)

    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet,
                    DefaultStagnation, config_file)

    population = Population(config)

    population.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_fitness, 100)  # Run the NEAT algorithm for 100 generations

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    run_neat("./neat.cfg")
