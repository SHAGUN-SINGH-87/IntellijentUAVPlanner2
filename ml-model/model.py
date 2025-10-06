import numpy as np
import pickle

# ==============================
# Grid + Q-learning Parameters
# ==============================
n_rows, n_cols = 6, 6   # Grid size
goal = (5, 5)           # Target goal position
obstacles = [(1, 1), (2, 2), (3, 3)]  # Obstacles

# Initialize rewards grid
rewards = np.full((n_rows, n_cols), -1)
for r, c in obstacles:
    rewards[r, c] = -100  # Obstacles penalty
rewards[goal] = 100       # Goal reward

# Initialize Q-values (rows, cols, actions)
q_values = np.zeros((n_rows, n_cols, 4))  # 4 actions: up, right, down, left

# Hyperparameters
learning_rate = 0.8
discount = 0.9
epsilon = 0.2
episodes = 2000  # Train for more episodes for convergence

# Actions mapping
actions = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1)   # Left
}

# ==============================
# Helper Functions
# ==============================
def is_valid(pos):
    """Check if position is within bounds and not an obstacle."""
    r, c = pos
    return 0 <= r < n_rows and 0 <= c < n_cols and rewards[r, c] != -100

def next_state(state, action):
    """Compute next state given a state and an action."""
    r, c = state
    dr, dc = actions[action]
    new_pos = (r + dr, c + dc)
    return new_pos if is_valid(new_pos) else state

# ==============================
# Q-learning Training Loop
# ==============================
def train_q_learning():
    global q_values
    for _ in range(episodes):
        state = (0, 0)  # Start from top-left corner
        while state != goal:
            r, c = state
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(4)  # Explore
            else:
                action = np.argmax(q_values[r, c])  # Exploit

            next_s = next_state(state, action)
            nr, nc = next_s
            reward = rewards[nr, nc]

            # Q-learning update rule
            q_values[r, c, action] += learning_rate * (
                reward + discount * np.max(q_values[nr, nc]) - q_values[r, c, action]
            )

            state = next_s
    return q_values

# ==============================
# Train & Save Model
# ==============================
if __name__ == "__main__":
    trained_q = train_q_learning()

    # Save Q-values
    with open("q_model.pkl", "wb") as f:
        pickle.dump(trained_q, f)

    print("✅ Q-learning model trained and saved as q_model.pkl")
    print("\nSample Q-values for start (0,0):")
    print(trained_q[0, 0])
