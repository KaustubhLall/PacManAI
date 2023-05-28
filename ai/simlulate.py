from game.game_logic import GameLogic
from game.game_state import GameState, print_board


def ai(game_state):
    # Placeholder AI function - goes right always
    return 1, 0


def print_board_basic(gamestate):
    for row in gamestate.board:
        print(''.join(row))




def simulate(maze_file='../mazes/2.txt', pacman_lives=1, ghost_difficulty=3, verbose=True, prefix=''):
    game_state = GameState(maze_file, pacman_lives, ghost_difficulty)
    game_logic = GameLogic(game_state)

    while True:
        # Get the AI's decision for the next move
        dx, dy = ai(game_state)

        game_logic.update(dx, dy)

        # game lost
        if game_state.pacman.lives <= 0:
            print(f'Game OVER - {game_state.pacman.score}')

            return game_state.pacman.score, game_state.pacman.lives
        # game completed
        if len(game_state.pellets) == 0:
            print('CONVERGED')
            return game_state.pacman.score * 10, game_state.pacman.lives * 1000
        if verbose: print(prefix + f"Move - ({dx}, {dy})\n---------\nBoard:\n{print_board(game_state)}")
        print(game_state.get_encoding())


if __name__ == "__main__":
    simulate()
