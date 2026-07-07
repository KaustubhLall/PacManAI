"""Command-line training loop for the experimental Deep Q-Learning agent.

The original version of this module started a 10,000-episode training run at
import time. Keeping training behind a `main()` guard makes the module safer to
import, easier to test, and friendlier for portfolio review.
"""

import argparse
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAZE = PROJECT_ROOT / "mazes" / "1.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "dqn"
DEFAULT_ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def lr_scheduler(epoch, initial_learning_rate=1e-2, decay_rate=0.1, decay_steps=200):
    return initial_learning_rate * decay_rate ** (epoch / decay_steps)


def _set_learning_rate(agent, learning_rate, keras_backend):
    optimizer = agent.model.optimizer
    lr_var = getattr(optimizer, "learning_rate", None) or getattr(optimizer, "lr", None)
    if lr_var is None:
        return

    if hasattr(lr_var, "assign"):
        lr_var.assign(learning_rate)
    else:
        keras_backend.set_value(lr_var, learning_rate)


def _set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _save_replay(replay_path, replay_states, replay_actions):
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    with open(replay_path, "wb") as f:
        pickle.dump((replay_states, replay_actions), f)


def train_dqn(
    maze_file=DEFAULT_MAZE,
    output_dir=DEFAULT_OUTPUT_DIR,
    episodes=100,
    max_steps=2_000,
    pacman_lives=2,
    ghost_difficulty=3,
    batch_size=32,
    target_update_interval=3,
    checkpoint_interval=100,
    file_prefix="pacmanDQL-light",
    seed=None,
    load_checkpoint=None,
):
    """Train the DQN agent and return the best score observed."""
    _set_seed(seed)

    # Heavy ML imports are intentionally lazy so `python -m ai.deepQL --help`
    # and plain module imports do not require TensorFlow/Keras to be installed.
    from keras import backend as K

    from ai.environments.deepql_env import DQNAgent, PacmanEnv

    output_dir = Path(output_dir)
    checkpoint_root = output_dir / "checkpoints"
    replay_root = output_dir / "replays"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    replay_root.mkdir(parents=True, exist_ok=True)

    env = PacmanEnv(str(maze_file), pacman_lives=pacman_lives, ghost_difficulty=ghost_difficulty)
    grid_state, extra_features = env.reset()
    agent = DQNAgent(
        grid_size=grid_state.shape[:2],
        num_channels=grid_state.shape[2],
        num_extra_features=len(extra_features),
        actions=DEFAULT_ACTIONS,
        load=load_checkpoint,
    )

    high_score = 0
    run_name = datetime.now().strftime(f"{file_prefix}-%Y-%m-%d")

    with tqdm(total=episodes, desc="Episodes", position=0) as pbar:
        for episode in range(episodes):
            state = env.reset()
            done = False
            replay_states = []
            replay_actions = []
            reward_info = {}

            for _step in range(max_steps):
                action = agent.act(state)
                replay_states.append(state)
                replay_actions.append(action)

                next_state, reward, done, reward_info = env.step(action)
                agent.remember(state, action, reward, next_state, done, reward_info)
                state = next_state

                if done:
                    break

            score = env.game_state.get_score()
            should_checkpoint = checkpoint_interval > 0 and episode % checkpoint_interval == 0

            if score > high_score or should_checkpoint:
                high_score = max(score, high_score)
                checkpoint_dir = checkpoint_root / run_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"score-{score}" 
                if should_checkpoint:
                    checkpoint_path = checkpoint_dir / f"score-{score}-ep-{episode}"
                agent.save(str(checkpoint_path), score, episode)

                replay_path = replay_root / run_name / f"score-{score}-ep-{episode}-replay.pkl"
                _save_replay(replay_path, replay_states, replay_actions)
                tqdm.write(f"Ep: {episode}, Score: {score}, Replay saved at: {replay_path}")
            else:
                tqdm.write(f"Ep: {episode}, Score: {score}")

            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

            if target_update_interval > 0 and episode % target_update_interval == 0:
                agent.update_target_model()
                _set_learning_rate(agent, lr_scheduler(episode), K)

            pbar.set_postfix({"High Score": high_score, **reward_info}, refresh=True)
            pbar.update(1)

    return high_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train the experimental DQN Pac-Man agent.")
    parser.add_argument("--maze-file", type=Path, default=DEFAULT_MAZE, help="maze file used for training")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for checkpoints/replays")
    parser.add_argument("--episodes", type=int, default=100, help="number of training episodes")
    parser.add_argument("--max-steps", type=int, default=2_000, help="maximum steps per episode")
    parser.add_argument("--pacman-lives", type=int, default=2, help="Pac-Man lives per episode")
    parser.add_argument("--ghost-difficulty", type=int, choices=range(4), default=3, help="ghost AI difficulty")
    parser.add_argument("--batch-size", type=int, default=32, help="experience replay batch size")
    parser.add_argument("--target-update-interval", type=int, default=3, help="episodes between target model syncs")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="episodes between forced checkpoints")
    parser.add_argument("--file-prefix", type=str, default="pacmanDQL-light", help="checkpoint/replay run prefix")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="checkpoint path prefix to load")
    return parser.parse_args()


def main():
    args = parse_args()
    high_score = train_dqn(**vars(args))
    print(f"Training complete. High score: {high_score}")


if __name__ == "__main__":
    main()
