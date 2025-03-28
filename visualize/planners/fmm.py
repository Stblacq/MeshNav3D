from typing import Optional

import numpy as np
import pyvista as pv
from heapq import heappush, heappop

from visualize.planners.planner import Planner


class FastMarchingPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a path between start and goal points on a mesh using Fast-Marching Method.

        Args:
            start_point (np.ndarray): Starting point coordinates [x, y, z]
            goal_point (np.ndarray): Goal point coordinates [x, y, z]
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
            mesh (pv.DataSet): Input mesh to plan on
            time_horizon (float): Maximum time horizon for planning (default: 10.0)
            max_iterations (int): Maximum number of iterations (default: 1000)

        Returns:
            Optional[dict]: Dictionary containing path information or None if path not found
        """
        start_point = np.asarray(start_point).reshape(3)
        goal_point = np.asarray(goal_point).reshape(3)

        edges_polydata = mesh.extract_all_edges()
        edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]
        vertices = edges_polydata.points

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))
        start_vertex = vertices[start_idx]
        goal_vertex = vertices[goal_idx]

        adj_list = {i: [] for i in range(len(vertices))}
        for p1_idx, p2_idx in edge_cells:
            dist = np.linalg.norm(vertices[p1_idx] - vertices[p2_idx])
            adj_list[p1_idx].append((p2_idx, dist))
            adj_list[p2_idx].append((p1_idx, dist))

        distances = np.full(len(vertices), np.inf)
        distances[start_idx] = 0.0
        heap = [(0.0, start_idx)]
        visited = set()

        while heap:
            dist, current_idx = heappop(heap)

            if current_idx in visited:
                continue
            visited.add(current_idx)

            for next_idx, edge_dist in adj_list[current_idx]:
                if next_idx in visited:
                    continue

                new_dist = dist + edge_dist
                if new_dist < distances[next_idx]:
                    distances[next_idx] = new_dist
                    heappush(heap, (new_dist, next_idx))

        if np.isinf(distances[goal_idx]):
            result = {
                'path_points': None,
                'path_length': 0.0,
                'travel_time': 0.0,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': False
            }
            if plotter is not None:
                plotter.add_mesh(mesh, opacity=0.5)
                plotter.add_points(start_point, color='red', point_size=10)
                plotter.add_points(goal_point, color='green', point_size=10)
                plotter.show_axes()
                plotter.show_bounds()
            return result

        path = [goal_vertex]
        current_point = goal_vertex
        current_idx = goal_idx
        path_length = 0.0
        iteration = 0

        while current_idx != start_idx and iteration < max_iterations:
            min_dist = np.inf
            next_idx = None
            for neighbor_idx, edge_dist in adj_list[current_idx]:
                if distances[neighbor_idx] < min_dist:
                    min_dist = distances[neighbor_idx]
                    next_idx = neighbor_idx

            if next_idx is None or min_dist >= distances[current_idx]:
                break

            path_length += np.linalg.norm(vertices[next_idx] - current_point)
            current_point = vertices[next_idx]
            path.append(current_point)
            current_idx = next_idx
            iteration += 1

        if current_idx != start_idx:
            result = {
                'path_points': None,
                'path_length': 0.0,
                'travel_time': 0.0,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': False
            }
        else:
            path_points = np.array(path[::-1])
            travel_time = path_length / time_horizon if time_horizon > 0 else 0.0
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
            if result['success']:
                plotter.add_points(path_points, color='blue', point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return result