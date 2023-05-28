# create a simluation of the game where an ai can play step-by-step
from game.game_logic import GameLogic
from game.game_state import GameState, print_board


class PacmanEnv:
    def __init__(self, game_map, pacman_lives, ghost_difficulty):
        self.game_state = GameState(game_map, pacman_lives, ghost_difficulty)
        self.game_logic = GameLogic(self.game_state)

    def reset(self):
        self.game_state.reset()
        return self.game_state.get_current_state()

    def step(self, action):
        self.game_logic.update(*action)
        next_state = self.game_state.get_current_state()
        done = self.game_state.is_game_over()
        reward = self.game_state.get_score() + self.game_state.pacman.lives * 10 - (25 if done else 0)
        return next_state, reward, done

    def render(self):
        print(print_board(self.game_state))

