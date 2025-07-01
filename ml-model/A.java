import numpy as np
import pickle

# Grid size
n_rows, n_cols = 6, 6

# Initialize rewards grid with -1
rewards = np.full((n_rows, n_cols), -1)

# Set goal and obstacles
goal = (5, 5)
obstacles = [(1, 1), (2, 2), (3, 3)]

# Set obstacle and goal rewards
for r, c in obstacles:
    rewards[r, c] = -100
rewards[goal] = 100

# Initialize Q-values: shape = (rows, cols, actions)
q_values = np.zeros((n_rows, n_cols, 4))  # 4 actions: up, right, down, left

# Hyperparameters
learning_rate = 0.8
discount = 0.9
epsilon = 0.2
episodes = 1000

# Action mapping: 0-up, 1-right, 2-down, 3-left
actions = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}

# Validate position: within bounds & not an obstacle
def is_valid(pos):
    r, c = pos
    return 0 <= r < n_rows and 0 <= c < n_cols and rewards[r, c] != -100

# Compute next state
def next_state(state, action):
    r, c = state
    dr, dc = actions[action]
    new_pos = (r + dr, c + dc)
    return new_pos if is_valid(new_pos) else state

# Q-learning loop
for _ in range(episodes):
    state = (0, 0)
    while state != goal:
        r, c = state
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_values[r, c])

        next_s = next_state(state, action)
        nr, nc = next_s
        reward = rewards[nr, nc]

        # Q-learning update
        q_values[r, c, action] += learning_rate * (
            reward + discount * np.max(q_values[nr, nc]) - q_values[r, c, action]
        )

        state = next_s

# Save Q-values model
with open("q_model.pkl", "wb") as f:
    pickle.dump(q_values, f)

print("✅ q_model.pkl created successfully.")
