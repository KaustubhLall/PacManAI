# PacManAI audit notes

This document tracks the first cleanup pass for turning PacManAI into a stronger public project artifact.

## Issues fixed in this pass

1. **README overclaimed the state of the project.** The previous README used placeholder repository URLs, referenced a missing `requirements.txt`, and described the project as a polished portfolio showcase. The new README states what exists, what is experimental, and how to actually run it.
2. **Episode reset was not safe.** `GameState.reset()` called `load_from_file()` without clearing the previous Pac-Man, ghost, and pellet collections. Training episodes could accumulate stale entities.
3. **State encodings were stale.** The board stored pellets and entities statically, so consumed pellets could remain visible to learning agents. Encodings now compose the live board from base walls, remaining pellets, Pac-Man, and ghosts.
4. **NEAT input dimensions were fragile.** `get_encoding()` was hard-coded to reshape to `1024`, which only works for 32×32 boards. The encoding now matches the loaded maze dimensions, and the NEAT runner validates its config against `mazes/2.txt`.
5. **CLI defaults were misleading.** The previous default maze path depended on the current working directory, and boolean parsing used `type=bool`, which makes `--manual-mode False` evaluate as true. The CLI now uses repo-root-relative defaults and `BooleanOptionalAction`.
6. **The game window could quit incorrectly.** The Exit button called `pygame.quit()` directly while the main loop continued. The renderer now reports a running flag back to the loop.
7. **No regression tests existed.** Added tests for reset behavior, encoding shape, consumed-pellet encoding, and win/loss game-over conditions.

## Remaining high-value work

- Add a headless simulation CLI for fast agent benchmarking without `pygame`.
- Split exploratory `pytorch_neat/` code from maintained project code or clearly document its provenance.
- Add deterministic seeds and per-episode metrics for NEAT/DQN comparisons.
- Refactor DQN training into a callable CLI instead of import-time execution.
- Improve the visual layer enough to produce strong screenshots and a short demo GIF.
