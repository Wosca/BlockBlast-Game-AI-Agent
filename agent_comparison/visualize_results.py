import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the results CSV
project_root = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(project_root, "results.csv")
df = pd.read_csv(csv_path)

# List of metrics to visualize
metrics = [
    ("score", "Average Score"),
    ("reward", "Average Reward"),
    ("steps", "Average Steps"),
    ("valid_moves", "Average Valid Moves"),
    ("invalid_attempts", "Average Invalid Attempts"),
]

# Compute group means
grouped = df.groupby("agent").mean()

# Bar charts for average metrics across agents
for metric, title in metrics:
    plt.figure()
    grouped[metric].plot(kind="bar")
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = f"average_{metric}.png"
    plt.savefig(fname)
    print(f"Saved {fname}")

# Optional: line plots of each metric over episodes for each agent
for metric, title in metrics:
    plt.figure()
    for agent in df["agent"].unique():
        agent_df = df[df["agent"] == agent]
        plt.plot(agent_df["episode"], agent_df[metric], label=agent)
    plt.title(f"{title} per Episode")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    fname = f"per_episode_{metric}.png"
    plt.savefig(fname)
    print(f"Saved {fname}")
