# Integrated temporal ls20 agent

This `agent.py` is a direct replacement for the uploaded 37,325-byte version.

## Run

```bash
python agent.py --game ls20 --episodes 1 --max-actions 1000
```

Use `--render` for terminal rendering.

## Main changes

- Tracks moving mechanisms by visual signature.
- Infers periodic trajectories from observations.
- Removes stale positions of moving mechanisms from the static tile map.
- Adds temporal phase to BFS state.
- Simulates mobile contact effects only at the predicted position and phase.
- Plans explicit interception of mobile mechanisms.
- Freezes a solved key by rejecting later transformations.
- Re-aligns temporal phase and removes weak dynamic edges after a life reset.
- Replans after every real action, as in the original control loop.

The implementation contains no hard-coded level number, coordinate sequence,
target key, or source-code inspection.

## Expected diagnostics

Look for output such as:

```text
mobile_periods={'...': 4}
plan=active-test:rotate@mobile:...
plan=execute:temporal-state-plan
```

The file was syntax-checked and its transformation, periodic-tracking, phase-
realignment and stale-position-removal components were unit-tested. The ARC
runtime itself was unavailable in the build sandbox, so full gameplay must be
verified in your installed competition environment.
