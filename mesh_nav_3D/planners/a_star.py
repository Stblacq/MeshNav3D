from typing import Optional

import numpy as np
import pyvista as pv
from heapq import heappush, heappop

from mesh_nav_3D.planners.planner import Planner


class AStarPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             color="blue",
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a path between start and goal points on a mesh using A* algorithm.

        Args:
            start_point (np.ndarray): Starting point coordinates [x, y, z]
            goal_point (np.ndarray): Goal point coordinates [x, y, z]
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
            mesh (pv.DataSet): Input mesh to plan on
            color: "blue",
            time_horizon (float): Maximum time horizon for planning (default: 10.0)
            max_iterations (int): Maximum number of iterations (default: 1000)

        Returns:
            Optional[dict]: Dictionary containing path information or None if path not found
        """
        start_point = np.asarray(start_point).reshape(3)
        goal_point = np.asarray(goal_point).reshape(3)

        # Extract edges and vertices from the mesh
        edges_polydata = mesh.extract_all_edges()
        edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]
        vertices = edges_polydata.points

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))
        start_vertex = vertices[start_idx]
        goal_vertex = vertices[goal_idx]

        # Build adjacency list
        adj_list = {i: [] for i in range(len(vertices))}
        for p1_idx, p2_idx in edge_cells:
            dist = np.linalg.norm(vertices[p1_idx] - vertices[p2_idx])
            adj_list[p1_idx].append((p2_idx, dist))
            adj_list[p2_idx].append((p1_idx, dist))

        def heuristic(vertex_idx):
            return np.linalg.norm(vertices[vertex_idx] - goal_vertex)

        # A* implementation
        open_set = [(heuristic(start_idx), start_idx, 0, [start_vertex])]  # (f_score, vertex_idx, g_score, path)
        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: heuristic(start_idx)}
        visited = set()
        iteration = 0

        while open_set and iteration < max_iterations:
            f, current_idx, g, path = heappop(open_set)
            iteration += 1

            if current_idx in visited:
                continue
            visited.add(current_idx)

            if current_idx == goal_idx:
                path_points = np.array(path)
                path_length = g

                result = {
                    'path_points': path_points,
                    'path_length': path_length,
                    'start_idx': start_idx,
                    'goal_idx': goal_idx,
                    'success': True
                }

                if plotter is not None:
                    plotter.add_mesh(mesh, opacity=0.5)
                    plotter.add_points(start_point, color='red', point_size=10)
                    plotter.add_points(goal_point, color='green', point_size=10)
                    plotter.add_points(path_points, color=color, point_size=5)
                    plotter.show_axes()
                    plotter.show_bounds()

                return result

            for next_idx, dist in adj_list[current_idx]:
                if next_idx in visited:
                    continue

                tentative_g = g + dist

                if next_idx not in g_score or tentative_g < g_score[next_idx]:
                    came_from[next_idx] = current_idx
                    g_score[next_idx] = tentative_g
                    f_score[next_idx] = tentative_g + heuristic(next_idx)
                    new_path = path + [vertices[next_idx]]
                    heappush(open_set, (f_score[next_idx], next_idx, tentative_g, new_path))

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