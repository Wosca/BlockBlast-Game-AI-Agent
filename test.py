from game_env import BlockGameEnv
import numpy as np

# Create a test environment
env = BlockGameEnv()
obs, _ = env.reset()

# Print initial shapes
print("Initial shapes:")
print(obs['shapes'])

# Try a valid action
action = env.get_valid_actions()[0]  # Get first valid action
shape_idx, row, col = env._decode_action(action)
print(f"Placing shape {shape_idx}")

# Apply action
new_obs, reward, term, trunc, info = env.step(action)

# Print shapes after action
print("\nShapes after placement:")
print(new_obs['shapes'])

# Check if the used shape is now all zeros
is_zeros = np.all(new_obs['shapes'][shape_idx] == 0)
print(f"Shape {shape_idx} is all zeros: {is_zeros}")

