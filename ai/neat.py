from multiprocessing import Pool

import neat

from game.game_logic import GameLogic
from game.game_state import GameState


def eval_genome(genome_id, genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game_state = GameState('../mazes/2.txt', 1, 0)
    game_logic = GameLogic(game_state)

    while True:
        outputs = net.activate(game_state.get_encoding())
        dx = int(outputs[0] * 2) - 1  # Map the output to -1, 0, or 1
        dy = int(outputs[1] * 2) - 1  # Map the output to -1, 0, or 1

        game_logic.update(dx, dy)

        if game_state.pacman.lives <= 0:
            genome.fitness = game_state.pacman.score
            break

        if len(game_state.pellets) == 0:
            genome.fitness = game_state.pacman.score * 10
            break

    return genome_id, genome


def eval_fitness(genomes, config):
    with Pool() as pool:
        results = pool.starmap(eval_genome, [(genome_id, genome, config) for genome_id, genome in genomes])
    for genome_id, genome in results:
        genomes[genome_id] = genome


import configparser


def update_config_file(config_file, input_size):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)

    config_parser['DefaultGenome']['num_inputs'] = str(input_size)

    with open(config_file, 'w') as file:
        config_parser.write(file)


def run_neat(config_file):
    game_state = GameState('../mazes/2.txt', 1, ghost_difficulty=3)
    input_size = len(game_state.get_encoding())

    update_config_file(config_file, input_size)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_fitness, 100)  # Run the NEAT algorithm for 100 generations

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    run_neat("../neat-configs/config-feedforward")
