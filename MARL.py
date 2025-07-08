import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Define parameters
alpha = 0.8  # Fairness trade-off coefficient
n = 4  # Number of agents
# Construct agent sequence: first half are type 0, second half type 1
sequence = [0 if i < n // 2 else 1 for i in range(n)]
# Decompose each goal type into individual goal instances
num_goals_per_type = [2, 2]  # A1, B1 (type 0), A2, B2 (type 1)
goal_types = [0]*num_goals_per_type[0] + [1]*num_goals_per_type[1]
# np.random.seed(42)  # Commented out to allow random behavior
agent_positions = np.random.uniform(low=0.0, high=1.0, size=(n, 2))  # Each row is (x, y) for an agent
goal_positions = np.random.uniform(low=0.0, high=1.0, size=(sum(num_goals_per_type), 2))
m = len(goal_positions)
w_k = np.array([1, 2])  # Budgets per agent type
preference = np.array([[2.0, 1], [1, 2.0]])  # Preference coefficients per goal type
total_capacity = np.array([2] * m, dtype=float)
u_kj = np.zeros((n, m))

# Compute distance matrix (agents × decomposed goals)
distances = np.linalg.norm(agent_positions[:, np.newaxis, :] - goal_positions[np.newaxis, :, :], axis=2)

# Compute utility = preference / distance
for i in range(n):
    agent_type = sequence[i]
    for j in range(m):
        goal_type = goal_types[j]
        if distances[i, j] > 0:
            u_kj[i, j] = preference[agent_type, goal_type] / distances[i, j]
        else:
            u_kj[i, j] = 1e6  # Avoid division by zero

print("Agent positions:\n", agent_positions)
print("Goal positions:\n", goal_positions)
print("Distance matrix (agents to goals):\n", distances)


# Comparison logic for social vs individual optimization
def compare_social_individual_allocations(social_allocation, sequence, w_k, u_kj, total_capacity):
    n, m = social_allocation.shape
    # Step 1: Get the dual prices from solving social optimization again
    x = cp.Variable((n, m), nonneg=True)
    objective = cp.Maximize(
        cp.sum([w_k[sequence[t]] * cp.log(cp.sum(cp.multiply(u_kj[t], x[t]))) for t in range(n)])
    )
    constraints = [cp.sum(x[t]) <= 1 for t in range(n)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    if result is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Solver failed. Status: {prob.status}")
        return

    lambda_j = np.array([constraints[j].dual_value for j in range(m)])
    print("Dual variable values (prices):", lambda_j)

    individual_x = np.zeros((n, m))
    for t in range(n):
        agent_type = sequence[t]
        x_indiv = cp.Variable(m, nonneg=True)
        utility = cp.sum(cp.multiply(u_kj[t], x_indiv))
        budget_constraint = cp.sum(cp.multiply(lambda_j, x_indiv)) <= w_k[agent_type]
        prob_indiv = cp.Problem(cp.Maximize(utility), [budget_constraint])
        prob_indiv.solve()
        individual_x[t] = x_indiv.value

    print("Social allocation matrix:")
    print(social_allocation)
    print("Individual allocation matrix (using dual prices):")
    print(individual_x)

    allocation_diff = np.linalg.norm(social_allocation - individual_x, ord='fro')
    print(f"Frobenius norm of allocation difference: {allocation_diff}")
    if allocation_diff <= 1:
        print("✅ The social and individual allocations match within tolerance.")
    else:
        print("❌ The social and individual allocations differ beyond tolerance.")


        
def main():
    x = cp.Variable((n, m), nonneg=True)
    alpha = 0.8
    variance_terms = []
    for agent_type in [0, 1]:
        indices = [i for i in range(n) if sequence[i] == agent_type]
        if indices:
            agent_distances = cp.vstack([cp.sum(cp.multiply(distances[i], x[i])) for i in indices])
            mean_dist = cp.sum(agent_distances) / len(indices)
            variance_terms.append(cp.sum_squares(agent_distances - mean_dist))

    variance = cp.sum(variance_terms)
    objective = cp.Maximize(
        cp.sum([w_k[sequence[t]] * cp.log(cp.sum(cp.multiply(u_kj[t], x[t]))) for t in range(n)])
        
    )
    constraints = [cp.sum(x[t]) == 1 for t in range(n)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    if result is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Social problem solver failed. Status: {prob.status}")
        return

    social_allocation = x.value
    compare_social_individual_allocations(social_allocation, sequence, w_k, u_kj, total_capacity)

    # Detect overlaps
    top_choices = np.argmax(social_allocation, axis=1)
    goal_assignment_counts = np.bincount(top_choices, minlength=m)

    # Check for any goals assigned to multiple agents
    conflicted_goals = np.where(goal_assignment_counts > 1)[0]
    for g in conflicted_goals:
        agents_for_g = np.where(top_choices == g)[0]
        utility_distance_ratios = [u_kj[i, g] for i in agents_for_g]
        winner_idx = agents_for_g[np.argmax(utility_distance_ratios)]
        for i in agents_for_g:
            if i != winner_idx:
                # Reassign to best available goal of same type based on utility
                agent_type = sequence[i]
                goal_type = goal_types[g]
                available_goals = [j for j in range(m) if goal_types[j] == goal_type and goal_assignment_counts[j] == 0]
                if available_goals:
                    best = max(available_goals, key=lambda j: u_kj[i, j])
                    top_choices[i] = best
                    goal_assignment_counts[best] += 1
        goal_assignment_counts[g] = 1  # mark this goal as occupied by winner only

    # Update social_allocation to reflect new choices
    social_allocation = np.zeros_like(social_allocation)
    for i in range(n):
        social_allocation[i, top_choices[i]] = 1

    # Visualization of assignments
    fig, ax = plt.subplots()
    for i in range(n):
        agent_pos = agent_positions[i]
        agent_type = sequence[i]
        color = 'blue' if agent_type == 0 else 'orange'
        marker = 'o' if agent_type == 0 else 's'
        ax.scatter(agent_pos[0], agent_pos[1], c=color, marker=marker, s=100, label=f'Agent Type {agent_type}' if i == sequence.index(agent_type) else "")

    goal_labels = ['A0', 'B0', 'A1', 'B1']
    goal_colors = ['red', 'red', 'green', 'green']
    for idx, (xg, yg) in enumerate(goal_positions):
        label = ""
        if goal_types[idx] == 0 and goal_labels[idx] == 'A0':
            label = "Goal Type 0 (A)"
        elif goal_types[idx] == 1 and goal_labels[idx] == 'B0':
            label = "Goal Type 1 (B)"
        ax.scatter(xg, yg, marker='X', color=goal_colors[idx], s=100, label=label)
        ax.text(xg + 0.01, yg + 0.01, goal_labels[idx], fontsize=9, color='black')

    for i in range(n):
        j = np.argmax(social_allocation[i])  # Assigned goal index
        agent_pos = agent_positions[i]
        goal_pos = goal_positions[j]
        ax.plot([agent_pos[0], goal_pos[0]], [agent_pos[1], goal_pos[1]], 'gray', linestyle='--')

    ax.set_title("Agent-Goal Assignment (Social Optimization)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()