from typing import Optional
import numpy as np
import pyvista as pv
from queue import PriorityQueue
from visualize.planners.planner import Planner

class JPSPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a path using the Jump Point Search (JPS) algorithm on a mesh.

        Args:
            start_point (np.ndarray): Starting point coordinates [x, y, z]
            goal_point (np.ndarray): Goal point coordinates [x, y, z]
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
            mesh (pv.DataSet): Input mesh to plan on
            time_horizon (float): Maximum time horizon for planning
            max_iterations (int): Maximum iterations for search

        Returns:
            Optional[dict]: Dictionary containing path information or None if path not found
        """

        start_point = np.asarray(start_point).reshape(3)
        goal_point = np.asarray(goal_point).reshape(3)

        start_idx = mesh.find_closest_point(start_point)
        goal_idx = mesh.find_closest_point(goal_point)

        # Priority queue for open nodes
        open_set = PriorityQueue()
        open_set.put((0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: np.linalg.norm(start_point - goal_point)}

        def jump(current_idx, direction):
            """
            Jumps in a given direction until a forced neighbor is found.
            """
            while True:
                neighbors = mesh.cell_neighbors(current_idx)
                if not neighbors:
                    return None  # No valid movement

                for neighbor in neighbors:
                    if neighbor == goal_idx:
                        return neighbor  # Goal reached

                    # Check if this is a forced neighbor
                    if is_forced_neighbor(current_idx, neighbor):
                        return neighbor
                
                # Continue moving in the given direction
                current_idx = neighbors[0]  # Assuming simple movement for now

        def is_forced_neighbor(current_idx, neighbor):
            """
            Checks if a neighbor is a forced neighbor, which requires expansion.
            """
            # This is a simplified check for forced neighbors based on connectivity.
            # Implement a more complex check if needed.
            return len(mesh.cell_neighbors(neighbor)) > 1

        def reconstruct_path(came_from, current):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        iterations = 0
        while not open_set.empty() and iterations < max_iterations:
            _, current = open_set.get()

            if current == goal_idx:
                path_indices = reconstruct_path(came_from, current)
                path_points = np.array([mesh.points[i] for i in path_indices])
                path_length = sum(np.linalg.norm(path_points[i] - path_points[i + 1]) for i in range(len(path_points) - 1))

                speed = path_length / time_horizon if time_horizon > 0 else 1.0
                travel_time = path_length / speed

                result = {
                    'path_points': path_points,
                    'path_length': path_length,
                    'travel_time': travel_time,
                    'start_idx': start_idx,
                    'goal_idx': goal_idx,
                    'success': True
                }

                if plotter is not None:
                    plotter.add_mesh(mesh, opacity=0.5)
                    plotter.add_points(start_point, color='red', point_size=10)
                    plotter.add_points(goal_point, color='green', point_size=10)
                    plotter.add_mesh(pv.PolyData(path_points), color='blue', line_width=3)
                    plotter.show_axes()
                    plotter.show_bounds()

                return result

            for direction in [-1, 1]:  # Simulated directional movement
                jump_point = jump(current, direction)
                if jump_point and jump_point not in g_score:
                    g_score[jump_point] = g_score[current] + np.linalg.norm(mesh.points[current] - mesh.points[jump_point])
                    f_score[jump_point] = g_score[jump_point] + np.linalg.norm(mesh.points[jump_point] - goal_point)
                    open_set.put((f_score[jump_point], jump_point))
                    came_from[jump_point] = current

            iterations += 1

        return None  # No path found
