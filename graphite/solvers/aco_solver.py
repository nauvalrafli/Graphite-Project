from typing import List, Union
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.solvers.base_solver import BaseSolver
import numpy as np
import random


class MySolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    def ant_colony_optimization(self, distance_matrix, num_ants=10, num_iterations=100, alpha=1, beta=2, evaporation_rate=0.5, Q=1):
        num_nodes = len(distance_matrix)
        pheromone_matrix = np.ones((num_nodes, num_nodes)) / num_nodes  # Initialize pheromone levels
        best_path = None
        best_path_length = float('inf')

        for iteration in range(num_iterations):
            paths = []
            for ant in range(num_ants):
                current_node = random.randint(0, num_nodes - 1)
                visited = [current_node]
                path_length = 0

                for _ in range(num_nodes - 1):
                    probabilities = []
                    unvisited = [node for node in range(num_nodes) if node not in visited]

                    for next_node in unvisited:
                        pheromone = pheromone_matrix[current_node, next_node]
                        distance = distance_matrix[current_node, next_node]
                        if distance == 0:
                            probability = 0
                        else:
                            probability = (pheromone ** alpha) * ((1 / distance) ** beta)
                        probabilities.append(probability)
                    
                    total_probability = sum(probabilities)
                    if total_probability == 0:
                        next_node = random.choice(unvisited)
                    else:
                        probabilities = [p / total_probability for p in probabilities]
                        next_node = random.choices(unvisited, weights=probabilities, k=1)[0]


                    visited.append(next_node)
                    path_length += distance_matrix[current_node, next_node]
                    current_node = next_node

                # Complete the cycle (return to the starting node)
                path_length += distance_matrix[visited[-1], visited[0]]
                visited.append(visited[0])  # Return to start
                paths.append((visited, path_length))


                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = visited


            # Update pheromone levels
            delta_pheromone_matrix = np.zeros((num_nodes, num_nodes))
            for path, path_length in paths:
                for i in range(len(path) - 1):
                    delta_pheromone_matrix[path[i], path[i + 1]] += Q / path_length

            pheromone_matrix = (1 - evaporation_rate) * pheromone_matrix + delta_pheromone_matrix

        return best_path, best_path_length

    async def solve(self, formatted_problem, future_id:int)->List[int]:
        distance_matrix = formatted_problem
        if not self.is_solvable(distance_matrix):
            return False
        
        return ant_colony_optimization(distance_matrix)


    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges
    
if __name__=='__main__':
    pass

