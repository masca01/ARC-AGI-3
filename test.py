import random

from arcengine import GameAction, GameState
import arc_agi

# Override specific parameters
arc = arc_agi.Arcade(
    arc_api_key="7b2f2846-152b-457f-86c1-01a442572897"
)


# Create an environment with terminal rendering
env = arc.make("ft09",
               render_mode="human",
               scorecard_id="1465f951-f75c-40cd-8f2d-d6456f91246d"
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
