from game.game_logic import GameLogic
from game.game_state import GameState


def ai(game_state):
    # Placeholder AI function
    return 1, 0


def print_board_basic(gamestate):
    for row in gamestate.board:
        print(''.join(row))


def print_board(gamestate):
    # Create a copy of the board matrix
    board_copy = [row.copy() for row in gamestate.board]

    # Place entities on the board
    board_copy[gamestate.pacman.y][gamestate.pacman.x] = '😮'  # Pacman

    for pellet in gamestate.pellets:
        board_copy[pellet.y][pellet.x] = '🍚'  # Pellets

    for ghost in gamestate.ghosts:
        board_copy[ghost.y][ghost.x] = '👻'  # Ghosts

    # Translate the maze symbols to emojis
    emoji_board = []
    for row in board_copy:
        emoji_row = []
        for cell in row:
            if cell == ' ':
                emoji_row.append('⬛')
            elif cell == '#':
                emoji_row.append('🟦')
            elif cell == '.':
                emoji_row.append('⬛')  # Replacing consumed pellets with empty space
            else:
                emoji_row.append(cell)
        emoji_board.append(emoji_row)

    # Convert the emoji board to a string
    emoji_board_str = '\n'.join([' '.join(row) for row in emoji_board])
    return emoji_board_str


def simulate(maze_file='./mazes/1.txt', pacman_lives=1, ghost_difficulty=3, verbose=True, prefix=''):
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


if __name__ == "__main__":
    simulate()
