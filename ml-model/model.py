import numpy as np
import pickle

n_rows, n_cols = 6, 6
rewards = np.full((n_rows, n_cols), -1)
goal = (5, 5)
obstacles = [(1, 1), (2, 2), (3, 3)]

rewards = np.full((n_rows, n_cols), -1)
for r, c in obstacles:
    rewards[r, c] = -100
rewards[goal] = 100

q_values = np.zeros((n_rows, n_cols, 4))

learning_rate = 0.8
discount = 0.9
epsilon = 0.2
episodes = 1000

actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

def is_valid(pos):
    r, c = pos
    return 0 <= r < n_rows and 0 <= c < n_cols and rewards[r, c] != -100

def next_state(state, action):
    r, c = state
    dr, dc = actions[action]
    new_pos = (r + dr, c + dc)
    return new_pos if is_valid(new_pos) else state

for _ in range(episodes):
    state = (0, 0)
    while state != goal:
        r, c = state
        action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_values[r, c])
        next_s = next_state(state, action)
        nr, nc = next_s
        reward = rewards[nr, nc]
        q_values[r, c, action] += learning_rate * (
            reward + discount * np.max(q_values[nr, nc]) - q_values[r, c, action]
        )
        state = next_s

with open("q_model.pkl", "wb") as f:
    pickle.dump(q_values, f)

print("✅ q_model.pkl created successfully.")
