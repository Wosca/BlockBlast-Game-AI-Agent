import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")

# Read the results CSV
df = pd.read_csv(CSV_PATH)

# Ensure output folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Metrics for separate plots
metrics = [
    ("score", "Average Score"),
    ("reward", "Average Reward"),
]

# Compute group means
grouped = df.groupby("agent").mean()

# Bar charts for average score and reward
for metric, title in metrics:
    plt.figure()
    grouped[metric].plot(kind="bar")
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"average_{metric}.png")
    plt.savefig(fname)
    print(f"Saved {fname}")

# Combined bar chart for steps, valid_moves, invalid_attempts
combined_metrics = ["steps", "valid_moves", "invalid_attempts"]
combined_means = grouped[combined_metrics]

plt.figure()
# Number of agents and metrics
ingent_count = len(combined_means)
metrics_count = len(combined_metrics)
indices = np.arange(combined_means.shape[0])

# Width and positions for grouped bars
bar_width = 0.8 / metrics_count
for i, m in enumerate(combined_metrics):
    positions = indices - 0.4 + i * bar_width + bar_width / 2
    plt.bar(positions, combined_means[m], bar_width, label=m)
    # Annotate bars with values
    for x, y in zip(positions, combined_means[m]):
        plt.text(x, y, f"{y:.1f}", ha="center", va="bottom")

plt.title("Average Steps / Valid / Invalid Movements")
plt.xlabel("Agent")
plt.ylabel("Count")
plt.xticks(indices, combined_means.index, rotation=45)
plt.legend()
plt.tight_layout()
combined_fname = os.path.join(RESULTS_DIR, "combined_moves.png")
plt.savefig(combined_fname)
print(f"Saved {combined_fname}")

# Line plots over episodes for each metric
for metric_label, title in [("score", "Score"), ("reward", "Reward")] + [
    (m, m.replace("_", " ").title()) for m in combined_metrics
]:
    plt.figure()
    for agent in df["agent"].unique():
        agent_df = df[df["agent"] == agent]
        plt.plot(agent_df["episode"], agent_df[metric_label], label=agent)
    plt.title(f"{title} per Episode")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"per_episode_{metric_label}.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
