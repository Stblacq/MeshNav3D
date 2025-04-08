import os
import time
from typing import Optional
import numpy as np
import psutil
import pyvista as pv
import potpourri3d as pp3d
from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput

class HeatMethodPlanner(Planner):
    def plan(self, input_data: PlannerInput, plotter: Optional[pv.Plotter] = None, **_) -> PlannerOutput:
        """
        Compute a geodesic path between start and goal points on a mesh using the heat method.

        Args:
            input_data (PlannerInput): Object containing start_point, goal_point, mesh, color, etc.
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization

        Returns:
            PlannerOutput: Object containing path information
        """
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss

        start_point = np.asarray(input_data.start_point).reshape(3)
        goal_point = np.asarray(input_data.goal_point).reshape(3)
        mesh = input_data.mesh
        max_iterations = input_data.max_iterations

        vertices = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))

        solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces, t_coef=1.0, use_robust=True)
        distances = solver.compute_distance(goal_idx)
        path_points = self._trace_geodesic_path(vertices, faces, distances, start_idx, goal_idx, max_iterations)

        output = PlannerOutput(
            start_point=start_point,
            goal_point=goal_point,
            path_points=path_points,
            start_idx=start_idx,
            goal_idx=goal_idx,
            execution_time=time.time() - start_time,
            memory_used_mb=(process.memory_info().rss - start_memory) / 1024 / 1024,
        )


        if plotter is not None:
            plotter.add_points(start_point, color='red', point_size=10)
            plotter.add_points(goal_point, color='green', point_size=10)
            if output.success and output.path_points is not None:
                plotter.add_points(output.path_points, color=input_data.color, point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return output

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