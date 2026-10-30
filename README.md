# ARC-AGI-3
The code aims to be designed to generalize in novel, unseen environments, using AI tools. This code will be tested through ARC-AGI-3 Interactive Reasoning Benchmark.

## Repository structure

```
ARC-AGI-3/
├── agent/               the agent itself (this is what will be submitted to Kaggle)
│   ├── my_agent.py      MyAgent: decides which action to take each step (strategy below)
│   ├── model.py         the CNN ("brain") and how it learns during a game
│   ├── causal_agent.py  CausalAgent: teammate's causal planner + optional CNN perception
│   ├── ls20_specialist.py  teammate's ls20 causal planner, ported (see below)
│   ├── control_net.py   CNN that learns what the agent controls, from its own moves
│   ├── slide_net.py     CNN that learns from one or two surprises which icons push the player, and where
│   ├── base.py          minimal Agent base class, so the agent runs without the official framework
│   └── __init__.py      marks agent/ as a Python package
├── tests/               regression tests (python3 tests/test_ls20_regression.py)
├── arc-special-tile-agent/  teammate's original causal agent, kept as delivered
├── run_local.py         plays games locally, prints what the agent learns, saves replay GIFs
├── replay.py            writes the interactive HTML replay (--html); page layout in replay_template.html
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

## Causal agent with neural perception (`agent/causal_agent.py`)

A second agent built on a teammate's idea (see `ARC-AGI-3 Active Causal Agent ... .md`): play like a
human. Look for unusual tiles, walk to them on purpose, write down what they do to the carried key
(rotate, recolour, reshape, refill), then plan a route through the right tiles to the door with BFS,
and replan after every move. The planner is the teammate's code (`ls20_specialist.py`), ported
unchanged. It is specialised to the look of `ls20` and only uses ACTION1-4.

`CausalAgent` wraps it in the official contract and adds two things:

- **Retry after GAME_OVER.** RESET restarts the same level, so the agent keeps playing with everything it
  learned about that level (the original script stopped at the first GAME_OVER).
- **Neural perception (`PERCEPTION = "neural"`).** The hand-made player detector ("colour 12, 5x5,
  above row 55") is replaced by a CNN. Nothing about the player is hard-coded:
  1. *Motion labels.* After each move, find the shift that best explains the changed pixels. The
     pixels that moved are what I control; changed pixels the shift does not explain changed by
     themselves (the resource bar). Colours are weighted by rarity, so the floor never counts as the
     moving object.
  2. *CNN.* A fully convolutional net (5 dilated 3x3 conv layers, 31 px receptive field, 16 one-hot
     colour planes in, 2 maps out: "what I control" and "changes by itself") is trained online on
     those labels. It finds the player in any frame from appearance alone, so the agent knows where it
     is at the start of a new level or after a reset without moving first.
  3. *Learned grid.* The grid step is the length of one move (5 px). The grid origin is where moved
     objects usually land. The CNN picks the most probable object-sized window, and the agent snaps it
     to the nearest cell.
  4. *Self-check before trust.* Before learning from each labelled move, the CNN must already have found
     the player's cell. The planner uses it only after 5 correct checks in a row. Until then the agent
     makes warm-up moves. Once the CNN is right 10 times in a row it trains on every 10th label only.

The hand-made detector is still run with `ORACLE = True`, only to measure the CNN's accuracy. It is
never used for decisions.

### Learning slides fast: SlideNet (`agent/slide_net.py`, `SLIDES = True`)

In ls20 some cells have a **pusher**: a short bar on one edge. Entering the cell slides the player
away from the bar until the next wall. The original planner stored each slide as an exact
`(position, action, configuration of every moving block)` entry. On level 5 the blocks move every step,
so the same pusher looked new again and again: 101 stored "non-local moves" in one run, and the
resource ran out. Checked on 122 observed slides: every one is "enter a cell, then a straight slide",
and "slide until the visible wall" predicts the end cell 121 times out of 122.

SlideNet learns the pusher from its look:
- **Input:** a 15x15 one-hot patch around a cell. The player is masked out, so the net learns the icon,
  not "the player was next to it".
- **Output:** P(no push), P(up), P(down), P(left), P(right). That is the slide's icon and its
  direction. The end position is "slide until the next wall" in the visible map. A slide that ended
  elsewhere is remembered as an exception.
- **Fast learning from surprises:** when a move ends somewhere the net did not predict, it takes 40
  gradient steps at once on that move. Each step includes the move's 8 rotations/reflections, with the
  direction label rotated too, so one pusher teaches all four orientations, plus earlier examples so
  nothing is forgotten. Ordinary moves are cheap "no push" examples.
- **In planning:** every step the net scores every visible cell. The BFS slides through predicted
  pushers it has never touched, and confidently predicted pushers are not re-tested.

Encounters per pusher on ls20 (`Y` = predicted correctly before it happened, `n` = surprise):
the first two pushers ever met (level 3) are surprises. After that, **11 of 16 pushers were predicted
the first time the agent met them**, and every pusher that surprised it once was predicted the next
time (`nY`). Over the game: 37 of 42 pushes predicted before they happened, 5 surprises, 0 false alarms,
and 1 stored non-local move instead of 101.

### Results on ls20 (deterministic, one run)

| Agent | Levels | Actions on levels 1 / 2 / 3 / 4 | CNN finds the player's cell |
|---|---|---|---|
| human baseline (`metadata.json`) | 7 | 22 / 123 / 73 / 84 | - |
| `cnn` (MyAgent v2, 300 actions) | 0 | - | - |
| `causal` (hand-made perception) | 4 | 13 / 45 / 41 / 148 | - |
| `causal-neural` (CNN perception) | 4 | 25 / 45 / 41 / 148 | 100% of 449 frames, levels 1-5 |
| `causal-slides` (CNN perception + SlideNet) | **5** | 25 / 45 / 41 / **86** / **126** (level 5) | 100% |

With the CNN, levels 2-4 are solved in exactly the same moves as with the hand-made detector. Level 1
costs the 12 warm-up moves the CNN needs to learn what it controls. Without SlideNet, level 5 is
unsolved: with 1000 actions the agent retries it 4 times (751 actions) and runs out of resource each
time. With SlideNet it is solved in 126 actions (human baseline 96) and level 4 drops from 148 to 86.
Level 6 is not solved yet. In 673 actions and 5 attempts the agent takes the same (correctly
predicted) pusher 16 times, so the planner is going round in a loop there.

Things that went wrong on the way (useful for the report):
- Handing the CNN to the planner after 2 labelled moves: its blob still covered floor pixels, the grid
  was shifted by 3 px, and level 1 took 63 actions. Fixed by the self-check.
- 97-98% per-frame accuracy was not enough. A single wrong cell makes the planner learn a fake
  teleport or a blocked edge, and it got stuck on level 2-3. Fixed by snapping to a grid learned from
  motion, choosing the best object-sized window instead of a blob centre, and taking the origin from
  the most common clean label instead of the last one.

## How to run

```bash
pip3 install -r requirements.txt
echo "ARC_API_KEY=<your key from arcprize.org/api-keys>" > .env

python3 run_local.py --list                       # public games
python3 run_local.py --game ls20 --gif            # play one game, save outputs/ls20_replay.gif
python3 run_local.py --game ls20,ft09             # several games
python3 run_local.py --all --steps 200            # every public game, like Kaggle
python3 run_local.py --game ls20 --render human --steps 60   # live window (slow)
python3 run_local.py --game ls20 --agent causal --steps 1000          # teammate's causal planner
python3 run_local.py --game ls20 --agent causal-neural --steps 1000 --gif   # same, CNN finds the player
python3 run_local.py --game ls20 --agent causal-slides --steps 1000 --html  # + SlideNet (best: 5 levels)
python3 tests/test_ls20_regression.py             # regression tests (about 2 minutes)
```

With `--agent causal-neural` the GIF shows the CNN's "what I control" map next to the game.

### Watching a run: interactive HTML replay

```bash
python3 run_local.py --game ls20 --agent causal-neural --steps 1000 --html   # outputs/ls20_causal-neural_replay.html
python3 run_local.py --game ls20 --agent cnn --html                          # outputs/ls20_cnn_replay.html
```

`--html` writes one self-contained file. Double-click it to open it in a browser; it works offline. For a
1000-step run it is about 5 MB. Per step it shows:

- **Game**, with overlays you can switch on and off: the planned path and its target, the learned tile
  roles (`rot+90`, `colour`, `shape`, `refill`, `?`), SlideNet's predicted pushes (pink arrows from the
  cell to where the slide ends), the CNN's player box (cyan), the hand-made
  detector's box (dashed white), the last motion label (green), and the grid. Hover a pixel to read
  every network output at that pixel.
- **Network outputs**: each CNN probability map, with a colour bar.
- **Inside the CNN**: input → conv1 ... conv5 → outputs, with each layer's mean activation and receptive
  field. You can see the receptive field grow and the player get singled out.
- **Action probabilities** (`cnn` agent): P(change) and P(win) for every action. The chosen one is
  highlighted.
- **Decision and state**: the action, the planner's reason, the carried key next to the door goal,
  resource, lives, what the CNN has learned (grid, trust, training steps), and events at this step.
- **Timeline**: levels, warm-up moves and events (level solved, game over, tile rule learned,
  non-local move learned, push predicted, surprise push, CNN miss, CNN milestone). Below it are curves for CNN confidence, resource and
  loss. Click anywhere to jump there, or use the event buttons or <kbd>N</kbd>/<kbd>P</kbd> to jump from
  event to event.

Keys: ←/→ step (Shift = 10), space play/pause, N/P next/previous event, Home/End.

What the replay already showed on level 5: every tile rule is learned exactly once (rot+90, refill,
colour, shape), but the planner records about 70 separate "non-local moves" on level 5. Those are stored
per exact configuration of the moving blocks, so each new phase looks like a new rule, and the agent
uses up its resource exploring them. This matches the handoff's "dynamic configuration is too brittle"
diagnosis.

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
- For Kaggle, `model.py`, `base.py` and `my_agent.py` (and `causal_agent.py`, `ls20_specialist.py`, `control_net.py` if the causal agent is used) must be merged into one `my_agent.py`, because the Kaggle notebook only copies that file.
- The causal agent only works on `ls20`: everything except the player detector still assumes ls20's colours, 5 px grid and HUD layout, and it never clicks.

## Changelog

### 2026-10-29 - SlideNet: learning slides from one or two surprises
- Added `agent/slide_net.py`. A CNN predicts from a cell's look whether entering it pushes the player and in which direction. The end is "until the next wall", plus remembered exceptions. Surprises trigger 40 immediate gradient steps with rotation/reflection augmentation.
- The planner uses predicted pushes wherever it has no exact observation and stops re-testing predicted pushers (`slide_model` hook in `ls20_specialist.py`, `CausalAgent.SLIDES`).
- New agent `run_local.py --agent causal-slides`. Replay shows predicted pushes and surprise/predicted-push events.
- Result on ls20: 5 levels (level 5 for the first time, in 126 actions; level 4 in 86 instead of 148). 37/42 pushes predicted before they happened, 11/16 pushers at first sight. Tests added; earlier results unchanged.

### 2026-10-29 - interactive replay viewer
- `run_local.py --html` writes `outputs/<game>_<agent>_replay.html`, a scrubbable replay with overlays (plan, target, tiles, CNN vs hand-made player box, motion label), CNN output maps, per-layer activations, action probabilities, state, and an event timeline.
- `CausalAgent` and `MyAgent` record what they saw, believed and planned at every step (`trace`). The planner keeps its whole plan (`last_plan`). Behaviour is unchanged: the regression tests still pass.

### 2026-10-29 - causal agent + neural perception
- Ported the teammate's ls20 causal planner into `agent/ls20_specialist.py` (unchanged except a pluggable player locator and a retry flag) behind `CausalAgent` in `agent/causal_agent.py`.
- Added `agent/control_net.py`: motion labels, an online CNN that learns what the agent controls, a learned grid, and a self-check before the planner trusts it.
- `run_local.py --agent cnn|causal|causal-neural`.
- Added `tests/test_ls20_regression.py`: the port must reproduce 13 / 45 / 41 / 148, and the neural version must match it with 100% player accuracy.
- Result on ls20: 4 levels with both perceptions. The CNN found the player's cell on 100% of frames.

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
