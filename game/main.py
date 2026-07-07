import argparse
from pathlib import Path

import pygame

from game.display import Display
from game.game_logic import GameLogic
from game.game_state import GameState
from game.input_handler import InputHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAZE = PROJECT_ROOT / "mazes" / "1.txt"


def main(maze_file=DEFAULT_MAZE, pacman_lives=3, ghost_difficulty=0, manual_mode=True, target_fps=10):
    game_state = GameState(str(maze_file), pacman_lives, ghost_difficulty)
    game_logic = GameLogic(game_state)
    display = Display(game_state)
    input_handler = InputHandler(manual_mode=manual_mode)

    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            clock.tick(target_fps)

            dx, dy = input_handler.get_input()
            game_logic.update(dx, dy)
            display.draw()
            running = display.handle_events()

            if game_state.is_game_over():
                print(f"Game over! Final score: {game_state.pacman.score}")
                break
    finally:
        pygame.quit()

    return game_state.pacman.score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Pac-Man environment.')
    parser.add_argument('--maze-file', type=Path, default=DEFAULT_MAZE, help='maze file path')
    parser.add_argument('--pacman-lives', type=int, default=3, help='number of lives for Pac-Man')
    parser.add_argument('--ghost-difficulty', type=int, choices=range(4), default=0,
                        help='ghost AI level: 0=static, 1=greedy, 2=DFS, 3=A*')
    parser.add_argument('--manual-mode', action=argparse.BooleanOptionalAction, default=True,
                        help='use keyboard controls; pass --no-manual-mode for AI/no-input stepping')
    parser.add_argument('--target-fps', type=int, default=15, help='target frames per second')

    args = parser.parse_args()

    main(
        maze_file=args.maze_file,
        pacman_lives=args.pacman_lives,
        ghost_difficulty=args.ghost_difficulty,
        manual_mode=args.manual_mode,
        target_fps=args.target_fps
    )
