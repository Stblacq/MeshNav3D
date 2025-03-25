from typing import Optional

import numpy as np
import pyvista as pv
from pyvista.plotting import Plotter

from visualize.planners.planner import Planner


def check_mesh_size(mesh: pv.DataSet):
    num_points = mesh.n_points
    num_cells = mesh.n_cells
    print(f"Number of points in the mesh: {num_points}")
    print(f"Number of cells in the mesh: {num_cells}")


def get_sorted_boundary_points(mesh: pv.DataSet,
                               start_point: np.ndarray,
                               goal_point: np.ndarray,
                               radius: float,
                               num_samples: int = 100):
    distances = np.linalg.norm(mesh.points - start_point, axis=1)

    tolerance = 0.1
    mask = np.abs(distances - radius) < tolerance

    boundary_points = mesh.points[mask]

    if len(boundary_points) > num_samples:
        sampled_indices = np.random.choice(len(boundary_points), num_samples, replace=False)
        boundary_points = boundary_points[sampled_indices]

    boundary_points = sorted(boundary_points, key=lambda point: np.linalg.norm(point - goal_point))

    return boundary_points


def dummy_planner(start_point: np.ndarray,
                  goal_point: np.ndarray,
                  mesh: pv.DataSet) -> np.ndarray | None:
    return [start_point, goal_point]


def geodesic_planner(start_point: np.ndarray, goal_point: np.ndarray, mesh: pv.PolyData, plotter: Plotter):
    start_id = mesh.find_closest_point(start_point)
    goal_id = mesh.find_closest_point(goal_point)

    path = mesh.geodesic(start_id, goal_id)
    plotter.add_mesh(path, color='white', line_width=5)




def is_point_in_mesh(mesh: pv.DataSet, point: np.ndarray, tolerance: float = 1e-6) -> bool:
    closest_point_index = mesh.find_closest_point(point)
    closest_point = mesh.points[closest_point_index]
    distance_to_closest_point = np.linalg.norm(closest_point - point)
    return distance_to_closest_point < tolerance


def extract_sub_mesh(mesh, radius, start_point):
    distances = np.linalg.norm(mesh.points - start_point, axis=1)
    mask = distances <= radius
    extracted_mesh = mesh.extract_points(mask)
    return extracted_mesh


class SSPPlanner(Planner):
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:

        radius = 0.3
        path = []
        current_start = start_point
        iteration = 0

        while iteration < max_iterations:
            extracted_mesh = extract_sub_mesh(mesh, radius, current_start)
            plotter.add_mesh(extracted_mesh, color='white')
            boundary_points = get_sorted_boundary_points(extracted_mesh, current_start, goal_point, radius)

            if is_point_in_mesh(extracted_mesh, goal_point):
                path.extend([current_start, goal_point])
                break
            else:
                progress_made = False
                for closest_point in boundary_points:
                    path.extend([current_start, closest_point])
                    current_start = closest_point
                    progress_made = True
                    break

                if not progress_made:
                    print("No progress made, stopping the planner.")
                    break
            iteration += 1

        if iteration == max_iterations:
            print("Reached maximum iterations, stopping the planner.")

        return path
