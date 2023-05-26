import neat

from game.game_logic import GameLogic
from game.game_state import GameState


def eval_fitness(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game_state = GameState('./mazes/2.txt', 1, 3)
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


def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_fitness, 100)  # Run the NEAT algorithm for 100 generations

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    run_neat("./neat-configs/config-feedforward.txt")
