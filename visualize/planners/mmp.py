from typing import Optional

import numpy as np
import pyvista as pv
from pygeodesic import geodesic

from visualize.planners.planner import Planner


class MMPPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a geodesic path between start and goal points on a mesh using the
        Mitchell-Mount-Papadimitriou (MMP) algorithm.

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

        geo_alg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))

        try:
            distance, path = geo_alg.geodesicDistance(start_idx, goal_idx)
        except Exception as e:
            print(f"Geodesic computation failed: {e}")
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

        path_points = np.array(path)
        path_length = float(distance)
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
