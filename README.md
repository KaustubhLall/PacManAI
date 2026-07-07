# PacManAI

PacManAI is a Python Pac-Man sandbox for experimenting with search-based ghost behavior and learning agents. The repository includes a playable `pygame` grid game, reusable game-state/step logic, a NEAT training configuration, and early Deep Q-Learning experiments.

This is best read as an AI/game-simulation project rather than a polished product: the core environment is small and hackable, while the learning agents are intentionally experimental.

## What is implemented

- **Playable Pac-Man loop** with keyboard controls and difficulty buttons.
- **Text-maze loader** for editable levels in `mazes/`.
- **Ghost policies** ranging from static ghosts to greedy movement, bounded DFS, and A* pathing.
- **Simulation state encodings** for NEAT and Q-learning style agents.
- **NEAT training scaffold** using `neat-python` and `ai/neat.cfg`.
- **Experimental DQN code** with prioritized replay and saved checkpoint/replay support.

## Repository layout

```text
ai/                 Training scripts and experimental agents
ai/environments/   RL environment wrappers and replay data structures
game/               Pac-Man game state, rules, renderer, and input handling
mazes/              Text-file maze definitions
pytorch_neat/       Imported/experimental PyTorch NEAT utilities
tests/              Regression tests for game-state behavior
```

> Note: `pytorch_neat/` appears to be exploratory/vendor-style research code. The maintained path for this project is currently the `game/` package plus `ai/ai_neat.py`.

## Setup

Python 3.10 or 3.11 is recommended. The core game and NEAT path are lightweight; the DQN path has heavier TensorFlow/Keras dependencies.

```bash
git clone https://github.com/KaustubhLall/PacManAI.git
cd PacManAI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

For the experimental Deep Q-Learning scripts:

```bash
pip install -r requirements-ml.txt
```

## Run the game

From the repository root:

```bash
python -m game.main
```

Useful options:

```bash
python -m game.main --maze-file mazes/2.txt --ghost-difficulty 3 --target-fps 15
python -m game.main --no-manual-mode
```

Ghost difficulty levels:

| Level | Behavior |
| --- | --- |
| `0` | Static ghosts |
| `1` | Greedy/direct movement |
| `2` | Bounded DFS pathing |
| `3` | A* pathing |

## Train a NEAT agent

```bash
python -m ai.ai_neat --config-file ai/neat.cfg --generations 100
```

`ai/neat.cfg` is currently matched to `mazes/2.txt`, which is a 32×32 board. The NEAT input size is the flattened board encoding plus Pac-Man's normalized `(x, y)` coordinates.

## Test

```bash
python -m pytest
```

The regression tests currently cover the most important environment correctness issues: reset behavior, dynamic encodings, pellet removal, and game-over semantics.

## Current limitations

- The renderer is functional, not a finished visual design.
- DQN training is experimental and may require dependency/version tuning.
- Maze symbols beyond `#`, `.`, `P`, and `G` are treated as walkable floor.
- There is no packaged model artifact or benchmark score yet.

## Next improvements

- Add a deterministic CLI simulation mode for fast headless evaluation.
- Record per-episode metrics and replays for comparing agents.
- Split experimental/vendor code from maintained project code.
- Improve the visual presentation layer for screenshots and demos.
- Add a portfolio demo GIF/video once the visual design is upgraded.

## License

MIT. See `LICENSE` for details.
