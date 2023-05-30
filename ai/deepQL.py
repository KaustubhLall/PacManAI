import os
import pickle
from datetime import datetime

from ai.environments.deepql_env import DQNAgent, PacmanEnv


# Define the directories for checkpoints and replays
CHECKPOINT_DIR = './DQL/checkpoints'
REPLAY_DIR = './DQL/replays'
EPISODES = 1000
TARGET_UPDATE_INTERVAL = 10
CHECKPOINT_INTERVAL = 5

# Make the directories if they do not exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)

actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
grid_height = 31
grid_width = 28
num_channels = 5
num_extra_features = 6
agent = DQNAgent((grid_height, grid_width), num_channels, num_extra_features, actions,)
# agent.load('C:/Users/kaus/PycharmProjects/PacManAI/ai/DQL/checkpoints/pacmanDQL - 2023-05-29/score-38')
batch_size = 64
env = PacmanEnv('../mazes/1.txt', pacman_lives=3, ghost_difficulty=3)
high_score = 0
file_prefix = 'pacmanDQL'

for e in range(EPISODES):
    state = env.reset()
    done = False
    # Initialize the lists to store states and actions for the replay
    replay_states = []
    replay_actions = []

    while not done:
        action = agent.act(state)
        # Append the state and action to the replay lists
        replay_states.append(state)
        replay_actions.append(action)
        next_state, reward, done, r_info = env.step(action)
        agent.remember(state, action, reward, next_state, done, r_info)
        state = next_state
        if done:
            score = env.game_state.get_score()
            print(f"episode: {e}/{EPISODES}, score: {score}")
            chk = e % CHECKPOINT_INTERVAL == 0
            if score > high_score or chk:
                high_score = score
                timestamp = datetime.now().strftime(f'{file_prefix} - %Y-%m-%d')
                checkpoint_dir = os.path.join(CHECKPOINT_DIR, timestamp)
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'score-{score}' + (f'-ep-{e}' if chk else ''))
                agent.save(checkpoint_path, score, e)
                print(f"New high score: {score}, checkpoint saved at {checkpoint_path}")

                replay_dir = os.path.join(REPLAY_DIR, timestamp)
                os.makedirs(replay_dir, exist_ok=True)
                replay_path = os.path.join(replay_dir, f'score-{score} %s-replay.pkl' % (f'ep {e}' if chk else ''))
                with open(replay_path, 'wb') as f:
                    pickle.dump((replay_states, replay_actions), f)
                print(f"Replay saved at: {replay_path}")

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % TARGET_UPDATE_INTERVAL == 0:
        agent.update_target_model()
