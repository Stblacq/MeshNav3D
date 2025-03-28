from typing import Optional
import numpy as np
import pyvista as pv
import potpourri3d as pp3d

from visualize.planners.planner import Planner


class FlipOutPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Compute a path between start and goal points on a mesh using EdgeFlipGeodesicSolver.

        Args:
            start_point (np.ndarray): Starting point coordinates [x, y, z]
            goal_point (np.ndarray): Goal point coordinates [x, y, z]
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
            mesh (pv.DataSet): Input mesh to plan on
            time_horizon (float): Maximum time horizon for planning (default: 10.0)
            max_iterations (int): Maximum number of iterations for edge flipping (default: 1000)

        Returns:
            Optional[dict]: Dictionary containing path information or None if path not found
        """
        start_point = np.asarray(start_point).reshape(3)
        goal_point = np.asarray(goal_point).reshape(3)

        V = np.asarray(mesh.points)
        F = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:])

        start_idx = np.argmin(np.linalg.norm(V - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(V - goal_point, axis=1))

        try:
            path_solver = pp3d.EdgeFlipGeodesicSolver(V, F)
        except Exception as e:
            result = {
                'path_points': None,
                'path_length': 0.0,
                'travel_time': 0.0,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': False,
                'error': str(e)
            }
            if plotter is not None:
                plotter.add_mesh(mesh, opacity=0.5)
                plotter.add_points(start_point, color='red', point_size=10)
                plotter.add_points(goal_point, color='green', point_size=10)
                plotter.show_axes()
                plotter.show_bounds()
            return result

        try:
            path_pts = path_solver.find_geodesic_path(
                v_start=start_idx,
                v_end=goal_idx,
                max_iterations=max_iterations
            )
        except Exception as e:
            result = {
                'path_points': None,
                'path_length': 0.0,
                'travel_time': 0.0,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': False,
                'error': str(e)
            }
            if plotter is not None:
                plotter.add_mesh(mesh, opacity=0.5)
                plotter.add_points(start_point, color='red', point_size=10)
                plotter.add_points(goal_point, color='green', point_size=10)
                plotter.show_axes()
                plotter.show_bounds()
            return result
        if path_pts.size == 0:
            result = {
                'path_points': None,
                'path_length': 0.0,
                'travel_time': 0.0,
                'start_idx': start_idx,
                'goal_idx': goal_idx,
                'success': False
            }
        else:
            path_diffs = np.diff(path_pts, axis=0)
            path_length = np.sum(np.linalg.norm(path_diffs, axis=1))
            travel_time = path_length / time_horizon if time_horizon > 0 else 0.0

            result = {
                'path_points': path_pts,
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
                plotter.add_points(path_pts, color='blue', point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return result
