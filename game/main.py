import argparse

import pygame

from game.display import Display
from game.game_logic import GameLogic
from game.game_state import GameState
from game.input_handler import InputHandler


def main(maze_file='./mazes/1.txt', pacman_lives=3, ghost_difficulty=0, manual_mode=True, target_fps=10):
    game_state = GameState(maze_file, pacman_lives, ghost_difficulty)
    game_logic = GameLogic(game_state)
    display = Display(game_state)
    input_handler = InputHandler(manual_mode=manual_mode)

    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(target_fps)  # Limit the frame rate

        dx, dy = input_handler.get_input()
        game_logic.update(dx, dy)
        display.draw()
        display.handle_events()

        if game_state.pacman.lives <= 0 or len(game_state.pellets) == 0:
            print(f"Game over! Final score: {game_state.pacman.score}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pacman game')
    parser.add_argument('--maze-file', type=str, default='../mazes/1.txt', help='maze file path')
    parser.add_argument('--pacman-lives', type=int, default=3, help='number of lives for Pacman')
    parser.add_argument('--ghost-difficulty', type=int, default=0, help='difficulty level for ghosts')
    parser.add_argument('--manual-mode', type=bool, default=True, help='enable manual mode')
    parser.add_argument('--target-fps', type=int, default=15, help='target frames per second')

    args = parser.parse_args()

    main(
        maze_file=args.maze_file,
        pacman_lives=args.pacman_lives,
        ghost_difficulty=args.ghost_difficulty,
        manual_mode=args.manual_mode,
        target_fps=args.target_fps
    )
