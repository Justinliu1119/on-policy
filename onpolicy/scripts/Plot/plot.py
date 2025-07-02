import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define paths
base_path = "results/MPE/simple_spread/mappo/test_run/run1/logs"
agent0_path = os.path.join(base_path, "agent0/individual_rewards/agent0/individual_rewards")
agent1_path = os.path.join(base_path, "agent1/individual_rewards/agent1/individual_rewards")
system_path = os.path.join(base_path, "average_episode_rewards/average_episode_rewards")

def load_scalar_from_event(path):
    ea = EventAccumulator(path)
    ea.Reload()
    scalar_events = ea.Scalars("individual_rewards" if "agent" in path else "average_episode_rewards")
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    return steps, values

# Load reward data
steps0, rewards0 = load_scalar_from_event(agent0_path)
steps1, rewards1 = load_scalar_from_event(agent1_path)
steps_avg, avg_rewards = load_scalar_from_event(system_path)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps0, rewards0, label="Agent 0 Reward")
plt.plot(steps1, rewards1, label="Agent 1 Reward")
plt.plot(steps_avg, avg_rewards, label="Average Episode Reward", linestyle='--', linewidth=2)

plt.xlabel("Environment Steps")
plt.ylabel("Reward")
plt.title("Training Rewards Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()