import os
import time
from typing import Optional
import numpy as np
import psutil
import pyvista as pv
from pygeodesic import geodesic
from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput

class MMPPlanner(Planner):
    def plan(self, input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None,
             **_) -> PlannerOutput:
        """
        Compute a geodesic path between start and goal points on a mesh using the
        Mitchell-Mount-Papadimitriou (MMP) algorithm.

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

        vertices = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        geo_alg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))

        distance, path = geo_alg.geodesicDistance(start_idx, goal_idx)
        path_points = np.array(path)

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
            plotter.add_mesh(mesh, opacity=0.5)
            plotter.add_points(start_point, color='red', point_size=10)
            plotter.add_points(goal_point, color='green', point_size=10)
            if output.success and output.path_points is not None:
                plotter.add_points(output.path_points, color=input_data.color, point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return output