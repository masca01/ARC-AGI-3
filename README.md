# ARC-AGI-3
The code aims to be designed to generalize in novel, unseen environments, using AI tools. This code will be tested through ARC-AGI-3 Interactive Reasoning Benchmark.

## Repository structure

```
ARC-AGI-3/
├── agent/               the agent itself (this is what will be submitted to Kaggle)
│   ├── my_agent.py      MyAgent: decides which action to take each step (strategy below)
│   ├── model.py         the CNN ("brain") and how it learns during a game
│   ├── base.py          minimal Agent base class, so the agent runs without the official framework
│   └── __init__.py      marks agent/ as a Python package
├── run_local.py         plays games locally, prints what the agent learns, saves replay GIFs
├── test.py              first toolkit test: random actions on one game
├── requirements.txt     Python packages needed
├── .env                 your ARC_API_KEY (private, never committed)
├── outputs/             replay GIFs from run_local.py (not committed)
└── environment_files/   game files downloaded by the toolkit (not committed)
```

## Strategy: neural networks only

The agent follows the team's three-step idea. Everything is learned inside one game from the agent's own moves. There is no pre-training and no dataset.

1. **Random tries.** The first 20 actions are random, to collect first experience. After that, 10% of actions stay random so the agent never gets stuck.
2. **Learn what changes the screen.** A CNN predicts, for every action and for every pixel that could be clicked, whether the move will change the screen *and* lead to a screen the agent has rarely seen. Moving back and forth changes the screen but teaches nothing, so it scores low. The agent picks actions with probability proportional to that score.
3. **Learn what wins.** When a level is completed, the moves of the winning attempt get credit: the last move gets 1.0 and each earlier move 10% less. The CNN learns these patterns and adds them to the score (`score = P(change) + 5 x P(win)`). The network is kept when a new level starts, so patterns from one level carry over to the next.

### The network (`agent/model.py`)

One CNN with three heads. The input is the 64x64 frame as 16 one-hot colour planes.

| Head | Output | Question it answers |
|---|---|---|
| change | 6 values (ACTION1-5, 7) + one 64x64 map (clicks) | Will this move lead somewhere new? |
| win | 6 values + one 64x64 map | Did moves like this lead to completing a level? |
| static | one 64x64 map | Which pixels change no matter what I do (timers, step counters)? |

The static head exists because many games have a step counter that changes on every move. Without it, every move would look useful and every screen would look new.

### Settings you can tune

| Setting | File | Meaning |
|---|---|---|
| `WARMUP_STEPS = 20` | my_agent.py | purely random actions at the start |
| `EPSILON = 0.1` | my_agent.py | share of random actions after the warm-up |
| `WIN_WEIGHT = 5.0` | my_agent.py | weight of the win head against the change head |
| `CLICK_SHARPNESS = 3.0` | my_agent.py | higher means clicks concentrate on the best pixels |
| `GAMMA = 0.9` | model.py | how fast win credit shrinks for earlier moves |
| `TRAIN_EVERY = 4` | model.py | one training round every 4 moves |
| `UNEXPECTED_PIXELS = 2.0` | model.py | minimum unexpected pixel change that counts as "something happened" |

## How to run

```bash
pip3 install -r requirements.txt
echo "ARC_API_KEY=<your key from arcprize.org/api-keys>" > .env

python3 run_local.py --list                       # public games
python3 run_local.py --game ls20 --gif            # play one game, save outputs/ls20_replay.gif
python3 run_local.py --game ls20,ft09             # several games
python3 run_local.py --all --steps 200            # every public game, like Kaggle
python3 run_local.py --game ls20 --render human --steps 60   # live window (slow)
```

Log columns: `lvl` levels completed, `screens` distinct screens reached, `counter px` pixels the static head flags as counters, `change acc` how often the change head was right over the last 100 moves, `loss` training loss, then the action, its predicted probability and the reason for choosing it.

The replay GIF shows the game on the left and, on the right, what the CNN believed at that step: where clicking looks useful, or which pixels it considers counters.

## Results

Latest version, 300 actions per game, one run each, laptop CPU (about 55 s per game).

| Game | Levels | Distinct screens | Change accuracy | Level solved at step |
|---|---|---|---|---|
| ls20 | 0 / 7 | 292 | 0.69 | - |
| ft09 | 0 / 6 | 88 | 0.64 | - |
| r11l | 1 / 6 | 288 | 0.84 | 258 |
| sb26 | 0 / 8 | 108 | 0.42 | - |
| bp35 | 0 / 9 | 277 | 0.84 | - |
| m0r0 | 0 / 6 | 236 | 0.81 | - |

Comparison on the same 6 games and budget (one run each, so a difference of one level can be luck):

| Version | Levels solved |
|---|---|
| v1: graph search + CNN (replaced) | 2 |
| v2: neural networks only, first try | 0 |
| v2: neural networks only, current | 1 |

### Known limitations
- On ls20 the step counter still makes almost every screen look new, so the novelty signal does not help there yet.
- The win head can only learn after a level is solved by exploration. In 300 moves that happened once, so step 3 has barely been used.
- For Kaggle, `model.py`, `base.py` and `my_agent.py` must be merged into one `my_agent.py`, because the Kaggle notebook only copies that file.

## Changelog

### 2026-09-10 - v2: neural-network-only agent
- Replaced the graph search and hand-made perception with one CNN with three heads (change, win, static).
- Removed `agent/perception.py` and `agent/effect_model.py`. The network now lives in `agent/model.py`.
- Change head target is "meaningful change x novelty of the screen reached", because in movement games every move changes the screen.
- Fixed a deadlock where the agent clicked the same dead pixel forever: training now runs every 4 moves even when moves repeat, and 10% of actions stay random.
- Pixels the static head ever flags as counters are ignored when screens are compared.
- Simplified `run_local.py`: removed `--plot` and `--offline`. The GIF now shows the CNN's click map or counter map.
- Result: 1 level on 6 games (v1 had 2).

### 2026-09-10 - v1: graph search + online CNN (replaced by v2)
- State graph that prefers untried actions, CNN that ranks them, border rows detection for step counters.
- Added `run_local.py` and `requirements.txt`. `outputs/` added to `.gitignore`.
- Result: 2 levels on 6 games.

### 2026-09-10 - repository setup
- API key moved out of `test.py` into `.env`. Added `.gitignore`.
- Personal branch `valeri/dev` created.

## Branches
Each teammate works on their own branch (for example `valeri/dev`) and merges into `main` through a pull request after discussion.
