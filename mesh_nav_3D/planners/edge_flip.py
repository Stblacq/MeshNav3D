from typing import Optional
import numpy as np
import pyvista as pv
import potpourri3d as pp3d

from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput

class FlipOutPlanner(Planner):
    def plan(self,
             input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **_) -> PlannerOutput:
        """
        Compute a path between start and goal points on a mesh using EdgeFlipGeodesicSolver.

        Args:
            input_data (PlannerInput): Input data containing start_point, goal_point, mesh, etc.
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization

        Returns:
            PlannerOutput: Object containing path information
        """
        # Extract parameters from input_data
        start_point = np.asarray(input_data.start_point).reshape(3)
        goal_point = np.asarray(input_data.goal_point).reshape(3)
        mesh = input_data.mesh
        time_horizon = getattr(input_data, 'time_horizon', 10.0)
        max_iterations = getattr(input_data, 'max_iterations', 1000)

        V = np.asarray(mesh.points)
        F = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:])

        start_idx = np.argmin(np.linalg.norm(V - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(V - goal_point, axis=1))

        try:
            path_solver = pp3d.EdgeFlipGeodesicSolver(V, F)
            path_pts = path_solver.find_geodesic_path(
                v_start=start_idx,
                v_end=goal_idx,
                max_iterations=max_iterations
            )
            if path_pts.size == 0:
                raise ValueError("Path not found")

            path_diffs = np.diff(path_pts, axis=0)
            path_length = np.sum(np.linalg.norm(path_diffs, axis=1))

            output = PlannerOutput(
                start_point=start_point,
                goal_point=goal_point,
                path_points=path_pts,
                path_length=path_length,
                start_idx=start_idx,
                goal_idx=goal_idx,
                success=True
            )
        except Exception:
            output = PlannerOutput(
                start_point=start_point,
                goal_point=goal_point,
                path_points=None,
                path_length=0.0,
                start_idx=start_idx,
                goal_idx=goal_idx,
                success=False
            )

        if plotter is not None:
            plotter.add_mesh(mesh, opacity=0.5)
            plotter.add_points(start_point, color='red', point_size=10)
            plotter.add_points(goal_point, color='green', point_size=10)
            if output.success and output.path_points is not None:
                plotter.add_points(output.path_points, color=input_data.color, point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return output