from typing import Optional

import numpy as np
import pyvista as pv
import potpourri3d as pp3d

from mesh_nav_3D.planners.planner import Planner


class HeatMethodPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             color="blue",
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a geodesic path between start and goal points on a mesh using the heat method.

        Args:
            start_point (np.ndarray): Starting point coordinates [x, y, z]
            goal_point (np.ndarray): Goal point coordinates [x, y, z]
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
            mesh (pv.DataSet): Input mesh to plan on (must be a triangular mesh)
            time_horizon (float): Maximum time horizon for planning (default: 10.0)
            max_iterations (int): Maximum number of iterations for path tracing (default: 1000)
            color:

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
            solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces, t_coef=1.0, use_robust=True)

            distances = solver.compute_distance(goal_idx)

            path_points = self._trace_geodesic_path(
                vertices, faces, distances, start_idx, goal_idx, max_iterations
            )

            if path_points is not None and len(path_points) > 1:
                path_length = float(np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))
            else:
                path_length = 0.0

            travel_time = path_length / time_horizon if time_horizon > 0 else 0.0

            result = {
                'path_points': path_points,
                'path_length': path_length,
                'travel_time': travel_time,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': path_points is not None
            }

        except Exception as e:
            print(f"Heat method computation failed: {e}")
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
                plotter.add_points(result['path_points'], color=color, point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return result

    def _trace_geodesic_path(self, vertices: np.ndarray, faces: np.ndarray,
                             distances: np.ndarray, start_idx: int, goal_idx: int,
                             max_iterations: int) -> Optional[np.ndarray]:
        """
        Trace the geodesic path from start to goal by following the negative gradient
        of the heat method distance field.

        Args:
            vertices (np.ndarray): Mesh vertices
            faces (np.ndarray): Mesh faces
            distances (np.ndarray): Geodesic distances from the goal
            start_idx (int): Starting vertex index
            goal_idx (int): Goal vertex index
            max_iterations (int): Maximum number of iterations

        Returns:
            Optional[np.ndarray]: Array of points along the geodesic path or None if failed
        """
        current_pos = vertices[start_idx].copy()
        path = [current_pos]
        step_size = np.mean(np.linalg.norm(np.diff(vertices, axis=0), axis=1)) * 0.1
        tolerance = step_size * 0.1

        for _ in range(max_iterations):
            current_vertex_idx = np.argmin(np.linalg.norm(vertices - current_pos, axis=1))
            if current_vertex_idx == goal_idx or np.linalg.norm(current_pos - vertices[goal_idx]) < tolerance:
                path.append(vertices[goal_idx])
                break

            face_indices = np.where(np.any(faces == current_vertex_idx, axis=1))[0]
            neighbor_vertices = np.unique(faces[face_indices])
            neighbor_vertices = neighbor_vertices[neighbor_vertices != current_vertex_idx]

            if len(neighbor_vertices) == 0:
                return None

            gradient_dir = np.zeros(3)
            for neighbor_idx in neighbor_vertices:
                direction = vertices[neighbor_idx] - current_pos
                weight = distances[current_vertex_idx] - distances[neighbor_idx]
                if weight > 0:
                    gradient_dir += weight * direction / (np.linalg.norm(direction) + 1e-6)

            if np.linalg.norm(gradient_dir) < 1e-6:
                break

            gradient_dir /= np.linalg.norm(gradient_dir)
            current_pos += step_size * gradient_dir
            path.append(current_pos.copy())

        return np.array(path) if len(path) > 1 else None