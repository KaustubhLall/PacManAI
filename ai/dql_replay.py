import pickle

from ai.environments.deepql_env import PacmanEnv

# Load the replay data
# Replace with your replay file below
replay_file = 'C:/Users/kaus/PycharmProjects/PacManAI/ai/DQL/replays/pacmanDQL - 2023-05-31/score-28 -replay.pkl'
with open(replay_file, 'rb') as f:
    replay_states, replay_actions = pickle.load(f)

# Initialize the environment -- need to manually set the correct maze
env = PacmanEnv('../mazes/1.txt', pacman_lives=3, ghost_difficulty=3)

# Reset the environment to the initial state
env.reset()

# Apply the actions sequentially
for state, action in zip(replay_states, replay_actions):
    _, reward, _, reward_info = env.step(action)
    print(env.render())  # Display the game state
    for title, info in reward_info.items():
        print(f"{title.replace('_', ' ').title()}:", f"{info:4.2f}", end=' | ')
    i = input('press any to continue (n to terminate)...')
    if i == 'n':
        break
    # sleep(0.5)  # Sleep for half a second to slow down the replay

print("Replay completed.")
