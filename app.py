from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
import random
import math
from typing import Dict, List, Tuple, Any

app = Flask(__name__)
CORS(app)

class GraphData:
    def __init__(self):
        self.data = {
            'nodes': [
                {'index': 0, 'x': 448.3075371591334, 'y': 108.85981125642901, 'label': 'A', 'isStartNode': False, 'isGoalNode': False, 'links': 3},
                {'index': 1, 'x': 300, 'y': 200, 'label': 'B', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
                {'index': 2, 'x': 600, 'y': 200, 'label': 'C', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
                {'index': 3, 'x': 50, 'y': 300, 'label': 'D', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
                {'index': 4, 'x': 750, 'y': 300, 'label': 'E', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
                {'index': 5, 'x': 300, 'y': 400, 'label': 'F', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
                {'index': 6, 'x': 600, 'y': 400, 'label': 'G', 'isStartNode': False, 'isGoalNode': False, 'links': 3},
                {'index': 7, 'x': 448.3075371591334, 'y': 500, 'label': 'H', 'isStartNode': False, 'isGoalNode': False, 'links': 3}
            ],
            'links': [
                {'source': 0, 'target': 1, 'distance': 20},
                {'source': 1, 'target': 0, 'distance': 20},
                {'source': 0, 'target': 2, 'distance': 23.58884702259604},
                {'source': 2, 'target': 0, 'distance': 23.58884702259604},
                {'source': 0, 'target': 3, 'distance': 32},
                {'source': 3, 'target': 0, 'distance': 32},
                {'source': 1, 'target': 2, 'distance': 15},
                {'source': 2, 'target': 1, 'distance': 15},
                {'source': 2, 'target': 5, 'distance': 18},
                {'source': 5, 'target': 2, 'distance': 18},
                {'source': 2, 'target': 4, 'distance': 13.892122440081096},
                {'source': 4, 'target': 2, 'distance': 13.892122440081096},
                {'source': 3, 'target': 7, 'distance': 42.461598428328124},
                {'source': 7, 'target': 3, 'distance': 42.461598428328124},
                {'source': 5, 'target': 6, 'distance': 13.892122440081096},
                {'source': 6, 'target': 5, 'distance': 13.892122440081096},
                {'source': 6, 'target': 7, 'distance': 9},
                {'source': 7, 'target': 6, 'distance': 9}
            ],
            'startNodeLabel': '',
            'goalNodeLabel': ''
        }
        
    def reset_graph(self) -> Dict:
        self.__init__()
        return self.data

    def generate_new_graph(self, node_count: int) -> Dict:
        # Ensure node count is reasonable
        node_count = max(2, min(node_count, 26))  # Limit to alphabet size
        
        # Create nodes in a more structured layout
        nodes = []
        rows = math.ceil(math.sqrt(node_count))
        spacing_x = 800 / (rows + 1)
        spacing_y = 600 / (rows + 1)
        
        for i in range(node_count):
            row = i // rows
            col = i % rows
            node = {
                'index': i,
                'x': 50 + spacing_x * (col + 1) + random.uniform(-20, 20),  # Add small random offset
                'y': 50 + spacing_y * (row + 1) + random.uniform(-20, 20),  # Add small random offset
                'label': chr(65 + i),
                'isStartNode': False,
                'isGoalNode': False,
                'links': 0
            }
            nodes.append(node)
        
        # Generate links with controlled connectivity
        links = []
        for i in range(node_count):
            # Determine how many connections this node should have (0-3)
            desired_connections = random.randint(1, min(3, node_count - 1))
            current_connections = sum(1 for link in links if link['source'] == i)
            
            # Get potential neighbors (prefer closer nodes)
            potential_neighbors = []
            for j in range(node_count):
                if i != j:
                    x1, y1 = nodes[i]['x'], nodes[i]['y']
                    x2, y2 = nodes[j]['x'], nodes[j]['y']
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    potential_neighbors.append((j, distance))
            
            # Sort by distance and filter out already connected nodes
            potential_neighbors.sort(key=lambda x: x[1])
            potential_neighbors = [
                (j, dist) for j, dist in potential_neighbors
                if not any(
                    (link['source'] == i and link['target'] == j) or
                    (link['source'] == j and link['target'] == i)
                    for link in links
                )
            ]
            
            # Add new connections up to desired amount
            for _ in range(current_connections, desired_connections):
                if not potential_neighbors:
                    break
                    
                target, dist = potential_neighbors.pop(0)
                if nodes[target]['links'] < 3:  # Ensure target node doesn't exceed 3 links
                    distance = max(10, dist / 10)  # Scale distance to reasonable values
                    
                    links.append({'source': i, 'target': target, 'distance': distance})
                    links.append({'source': target, 'target': i, 'distance': distance})
                    
                    nodes[i]['links'] += 1
                    nodes[target]['links'] += 1
        
        self.data = {
            'nodes': nodes,
            'links': links,
            'startNodeLabel': '',
            'goalNodeLabel': ''
        }
        return self.data    

class SearchAlgorithms:
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        self.iteration_counts = {
            'enqueues': 0,
            'extensions': 0,
            'queue_size': 0,
            'path_nodes': 0,
            'path_cost': 0.0
        }

    def dfs(self, start_label: str, goal_label: str) -> Tuple[List, List, Dict]:
        visited_labels = []
        paths = []
        # Track both the node and its accumulated cost
        stack = [(start_label, None, 0.0)]
        self._reset_counts()
        path_cost = 0.0

        while stack:
            current_label, parent_label, current_cost = stack.pop()

            if current_label not in visited_labels:
                visited_labels.append(current_label)

                if parent_label is not None:
                    paths.append((parent_label, current_label))
                    self.iteration_counts['extensions'] += 1
                    path_cost = current_cost
                    self.iteration_counts['path_cost'] = path_cost

                if current_label == goal_label:
                    break

                neighbors = self._get_unvisited_neighbors(current_label, visited_labels)
                neighbors.sort(reverse=True)
                for neighbor_label, distance in neighbors:
                    stack.append((neighbor_label, current_label, current_cost + distance))
                
                self._update_counts(len(neighbors), len(stack), len(paths))

        self.iteration_counts['path_nodes'] = len(paths) + 1  # Add 1 to include start node
        return visited_labels, paths, self.iteration_counts

    def bfs(self, start_label: str, goal_label: str) -> Tuple[List, List, Dict]:
        visited_labels = []
        paths = []
        queue = deque([(start_label, None, 0.0)])
        self._reset_counts()
        path_cost = 0.0

        while queue:
            current_label, parent_label, current_cost = queue.popleft()

            if current_label not in visited_labels:
                visited_labels.append(current_label)

                if parent_label is not None:
                    paths.append((parent_label, current_label))
                    self.iteration_counts['extensions'] += 1
                    path_cost = current_cost
                    self.iteration_counts['path_cost'] = path_cost

                if current_label == goal_label:
                    break

                neighbors = self._get_unvisited_neighbors(current_label, visited_labels)
                neighbors.sort()
                for neighbor_label, distance in neighbors:
                    queue.append((neighbor_label, current_label, current_cost + distance))
                
                self._update_counts(len(neighbors), len(queue), len(paths))

        self.iteration_counts['path_nodes'] = len(paths) + 1  # Add 1 to include start node
        return visited_labels, paths, self.iteration_counts

    def hill_climb(self, start_label: str, goal_label: str) -> Tuple[List, List, Dict]:
        visited_labels = []
        paths = []
        current_label = start_label
        self._reset_counts()
        total_distance = 0.0

        while current_label != goal_label:
            visited_labels.append(current_label)
            
            neighbors = self._get_unvisited_neighbors(current_label, visited_labels)
            if not neighbors:
                break

            next_label, distance = min(neighbors, key=lambda x: x[1])
            paths.append((current_label, next_label))
            
            self.iteration_counts['extensions'] += 1
            total_distance += distance
            self.iteration_counts['path_cost'] = total_distance
            
            current_label = next_label
            self._update_counts(len(neighbors), 1, len(paths))

        self.iteration_counts['path_nodes'] = len(paths) + 1  # Add 1 to include start node
        return visited_labels, paths, self.iteration_counts

    def _get_unvisited_neighbors(self, current_label: str, visited_labels: List[str]) -> List[Tuple[str, float]]:
        current_node = next(node for node in self.graph_data['nodes'] if node['label'] == current_label)
        return [(self.graph_data['nodes'][link['target']]['label'], link['distance'])
                for link in self.graph_data['links']
                if link['source'] == current_node['index'] 
                and self.graph_data['nodes'][link['target']]['label'] not in visited_labels]

    def _reset_counts(self):
        self.iteration_counts = {
            'enqueues': 0,
            'extensions': 0,
            'queue_size': 0,
            'path_nodes': 0,
            'path_cost': 0.0
        }

    def _update_counts(self, neighbors_count: int, queue_size: int, paths_count: int):
        self.iteration_counts['enqueues'] += neighbors_count
        self.iteration_counts['queue_size'] = max(self.iteration_counts['queue_size'], queue_size)

# Initialize global instances
graph_data = GraphData()
search_algorithms = SearchAlgorithms(graph_data.data)

@app.route('/api/reset_graph', methods=['POST'])
def reset_graph():
    try:
        new_graph = graph_data.reset_graph()
        search_algorithms.graph_data = new_graph
        return jsonify(new_graph)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph_data', methods=['GET'])
def get_graph_data():
    return jsonify(graph_data.data)

@app.route('/api/generate_nodes', methods=['POST'])
def generate_nodes():
    data = request.get_json()
    node_count = int(data['node_count'])
    new_graph = graph_data.generate_new_graph(node_count)
    search_algorithms.graph_data = new_graph
    return jsonify(new_graph)

@app.route('/api/search', methods=['POST'])
def run_search():
    data = request.get_json()
    start_label = data['start_label']
    goal_label = data['goal_label']
    algorithm = data['algorithm']
    
    algorithms = {
        'dfs': search_algorithms.dfs,
        'bfs': search_algorithms.bfs,
        'hill_climb': search_algorithms.hill_climb
    }
    
    try:
        visited_labels, paths, iteration_counts = algorithms[algorithm](start_label, goal_label)
        return jsonify({
            'visited_labels': visited_labels,
            'paths': paths,
            'iteration_counts': iteration_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)