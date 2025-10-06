from flask import Flask, request, jsonify
import numpy as np
import pickle
import heapq
import os

app = Flask(__name__)

# Load the Q-learning model
with open("q_model.pkl", "rb") as f:
    q_values = pickle.load(f)

# Grid settings (must match training)
n_rows, n_cols = 6, 6
obstacles = [(1, 1), (2, 2), (3, 3)]
goal = (5, 5)

rewards = np.full((n_rows, n_cols), -1)
for r, c in obstacles:
    rewards[r, c] = -100
rewards[goal] = 100

actions = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1)   # left
}

def is_valid(pos):
    r, c = pos
    return 0 <= r < n_rows and 0 <= c < n_cols and rewards[r, c] != -100

def next_state(state, action):
    r, c = state
    dr, dc = actions[action]
    new = (r + dr, c + dc)
    return new if is_valid(new) else state

def predict_path(origin, destination):
    path = [origin]
    state = origin
    for _ in range(100):
        r,c = state
        action = np.argmax(q_values[r,c])
        next_s = next_state(state, action)
        if next_s == state:
            break
        path.append(next_s)
        state = next_s
        if state == destination:
            break
    return path

def shortest_path(origin, destination):
    heap = [(0, origin)]
    visited = set()
    parent = {origin: None}
    while heap:
        cost, node= heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            break
        for action in actions.values():
            nr, nc = node[0] + action[0], node[0]+ action[1]
            neighbor = (nr,nc)
            if is_valid(neighbor) and neighbor not in visited:
                heapq.heappush(heap, (cost + 1, neighbor))
                if neighbor not in parent:
                    parent[neighbor] = node

    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1]


@app.route('/')
def home():
    return "✅ Flask running!"

@app.route('/predict-path', methods=['POST'])
def predict_path_route():
    try:
        data = request.json
        origin = tuple(map(int, data['origin'].split(',')))
        destination = tuple(map(int, data['destination'].split(',')))
        terrain = data.get("terrainType", "default")

        path = [origin]
        state = origin

        max_steps = 100  # prevent infinite loops
        for _ in range(max_steps):
            r, c = state
            action = np.argmax(q_values[r, c])
            next_s = next_state(state, action)
            if next_s == state:
                break  # no progress (stuck)
            path.append(next_s)
            state = next_s
            if state == destination:
                break

        predicted = predict_path(origin,destination)
        shortest = shortest_path(origin,destination)

        formatted_predicted = ' -> '.join(str(p) for p in predicted)
        formatted_shortest = ' -> '.join(str(p) for p in shortest)
        return jsonify({
        "predictedPath": formatted_predicted ,
        "shortestPath": formatted_shortest
        })

    except Exception as e:
        print("❌ Error in prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


