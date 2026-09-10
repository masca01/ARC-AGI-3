import os
import random

from arcengine import GameAction, GameState
import arc_agi

# API key is read from the ARC_API_KEY environment variable (see .env.example).
# Never commit a real key: this repository is public.
api_key = os.environ.get("ARC_API_KEY")
if not api_key:
    raise SystemExit("Set ARC_API_KEY in your environment (or a .env file) before running.")

arc = arc_agi.Arcade(arc_api_key=api_key)


# Create an environment with terminal rendering
env = arc.make("ft09",
               render_mode="human"
               )
if env is None:
    print("Failed to create environment")
    exit(1)

print(env.action_space)

# Play the game
for step in range(100):
    # Choose a random action
    action = random.choice(env.action_space)
    action_data = {}
    if action.is_complex():
        action_data = {
            "x": random.randint(0, 63),
            "y": random.randint(0, 63),
        }        
        
    # Perform the action (rendering happens automatically)
    obs = env.step(action, data=action_data)
    
    # Check game state
    if obs and obs.state == GameState.WIN:
        print(f"Game won at step {step}!")
        break
    elif obs and obs.state == GameState.GAME_OVER:
        env.reset()

# Get and display scorecard
scorecard = arc.get_scorecard()
if scorecard:
    print(f"Final Score: {scorecard.score}")
