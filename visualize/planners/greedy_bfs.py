from typing import Optional, Dict, Set
import heapq

import numpy as np
import pyvista as pv

from visualize.planners.planner import Planner


class GreedyBFSPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a geodesic path between start and goal points on a mesh using Greedy Best-First Search.

        Args:
            start_point (np.ndarray): Starting point coordinates [x, y, z]
            goal_point (np.ndarray): Goal point coordinates [x, y, z]
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
            mesh (pv.DataSet): Input mesh to plan on (must be a triangular mesh)
            time_horizon (float): Maximum time horizon for planning (default: 10.0)
            max_iterations (int): Maximum number of iterations (default: 1000)

        Returns:
            Optional[dict]: Dictionary containing path information or None if path not found
        """
        start_point = np.asarray(start_point).reshape(3)
        goal_point = np.asarray(goal_point).reshape(3)

        vertices = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))

        try:
            adjacency = self._build_adjacency_list(vertices, faces)

            path_indices = self._greedy_bfs(
                start_idx, goal_idx, vertices, adjacency, max_iterations
            )

            if path_indices is None or len(path_indices) == 0:
                raise ValueError("Path not found")

            path_points = vertices[path_indices]

            path_length = float(np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))) if len(path_points) > 1 else 0.0

            travel_time = path_length / time_horizon if time_horizon > 0 else 0.0

            result = {
                'path_points': path_points,
                'path_length': path_length,
                'travel_time': travel_time,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': True
            }

        except Exception as e:
            print(f"Greedy BFS planning failed: {e}")
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
            if result['success'] and result['path_points'] is not None:
                plotter.add_points(result['path_points'], color='blue', point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return result

    def _build_adjacency_list(self, vertices: np.ndarray, faces: np.ndarray) -> Dict[int, Set[int]]:
        """
        Build an adjacency list representation of the mesh based on face connectivity.

        Args:
            vertices (np.ndarray): Mesh vertices
            faces (np.ndarray): Mesh faces

        Returns:
            Dict[int, Set[int]]: Adjacency list mapping vertex indices to their neighbors
        """
        adjacency = {i: set() for i in range(len(vertices))}
        for face in faces:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                adjacency[face[i]].add(face[j])
                adjacency[face[j]].add(face[i])
        return adjacency

    def _greedy_bfs(self, start_idx: int, goal_idx: int, vertices: np.ndarray,
                    adjacency: Dict[int, Set[int]], max_iterations: int) -> Optional[list]:
        """
        Perform Greedy Best-First Search to find a path from start to goal.

        Args:
            start_idx (int): Starting vertex index
            goal_idx (int): Goal vertex index
            vertices (np.ndarray): Mesh vertices
            adjacency (Dict[int, Set[int]]): Adjacency list of the mesh
            max_iterations (int): Maximum number of iterations

        Returns:
            Optional[list]: List of vertex indices forming the path, or None if not found
        """
        goal_pos = vertices[goal_idx]

        frontier = [(np.linalg.norm(vertices[start_idx] - goal_pos), 0, start_idx)]
        came_from = {start_idx: None}
        visited = set()
        iteration = 0

        while frontier and iteration < max_iterations:
            _, _, current_idx = heapq.heappop(frontier)

            if current_idx == goal_idx:
                path = []
                while current_idx is not None:
                    path.append(current_idx)
                    current_idx = came_from[current_idx]
                return path[::-1]

            if current_idx in visited:
                continue

            visited.add(current_idx)

            for neighbor_idx in adjacency[current_idx]:
                if neighbor_idx not in visited:
                    heuristic = np.linalg.norm(vertices[neighbor_idx] - goal_pos)
                    heapq.heappush(frontier, (heuristic, iteration + 1, neighbor_idx))
                    if neighbor_idx not in came_from:
                        came_from[neighbor_idx] = current_idx

            iteration += 1

        return None
