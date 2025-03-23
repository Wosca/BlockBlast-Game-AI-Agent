"""
Human play mode for testing the block game environment.
This allows you to play the game using the separated game logic + environment.
"""

import pygame
import time
from game_env import BlockGameEnv


def main():
    """Run the game in human-playable mode using the environment wrapper."""
    # Create the environment with human rendering
    env = BlockGameEnv(render_mode="human")

    # Reset to get the initial state
    obs, _ = env.reset()

    # Game loop
    running = True
    while running:
        # Render the current state
        env.render()

        # Process human input through the renderer
        action = env.renderer.process_human_events()

        # Handle special actions
        if action == "RESET":
            obs, _ = env.reset()
            continue

        # If a valid action was taken, apply it
        if action:
            shape_idx, row, col = action
            # Convert to flat action index
            flat_action = shape_idx * 64 + row * 8 + col

            # Apply the action
            obs, reward, terminated, truncated, info = env.step(flat_action)

            # Print reward information
            print(f"Action: Shape {shape_idx} at ({row}, {col})")
            print(f"Reward: {reward}")
            print(f"Score: {info['score']}")
            print(f"Lines cleared: {info['lines_cleared']}")

            # If the game is over, pause briefly
            if terminated:
                print("Game Over!")
                time.sleep(2)
                obs, _ = env.reset()

        # Maintain frame rate
        pygame.time.delay(30)


if __name__ == "__main__":
    main()
