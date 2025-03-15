import numpy as np
import pyvista as pv
from typing import Optional


def dijkstra_planner(start_point: np.ndarray,
                     goal_point: np.ndarray,
                     plotter: Optional[pv.Plotter],
                     mesh: pv.DataSet,
                     time_horizon: float = 10.0) -> Optional[dict]:
    """
    Compute a geodesic path between start and goal points on a mesh using PyVista's geodesic functionality.

    Args:
        start_point (np.ndarray): Starting point coordinates [x, y, z]
        goal_point (np.ndarray): Goal point coordinates [x, y, z]
        plotter (Optional[pv.Plotter]): PyVista plotter for visualization
        mesh (pv.DataSet): Input mesh to plan on
        time_horizon (float): Maximum time horizon for planning (default: 10.0)

    Returns:
        Optional[dict]: Dictionary containing path information or None if path not found
    """

    start_point = np.asarray(start_point).reshape(3)
    goal_point = np.asarray(goal_point).reshape(3)

    start_idx = mesh.find_closest_point(start_point)
    goal_idx = mesh.find_closest_point(goal_point)

    path = mesh.geodesic(start_idx, goal_idx)

    path_points = path.points

    path_length = path.length


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
        plotter.add_mesh(path, color='blue', line_width=3)
        plotter.show_axes()
        plotter.show_bounds()

    return result

