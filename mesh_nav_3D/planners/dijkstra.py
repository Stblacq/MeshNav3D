import os
import time
from typing import Optional

import psutil
import pyvista as pv

from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput


class DijkstraPlanner(Planner):
    def plan(self,
             input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **_) -> PlannerOutput:
        """
        Compute a geodesic path using Dijkstra's algorithm with PlannerInput.
        """
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_idx = input_data.mesh.find_closest_point(input_data.start_point)
        goal_idx = input_data.mesh.find_closest_point(input_data.goal_point)
        path = input_data.mesh.geodesic(start_idx, goal_idx)

        output = PlannerOutput(
            start_point=input_data.start_point,
            goal_point=input_data.goal_point,
            path_points=path.points,
            start_idx=start_idx,
            goal_idx=goal_idx,
            execution_time= time.time() - start_time,
            memory_used_mb= (process.memory_info().rss - start_memory) / 1024 / 1024,
        )

        if plotter is not None:
            plotter.add_points(input_data.start_point, color='red', point_size=10)
            plotter.add_points(input_data.goal_point, color='green', point_size=10)
            plotter.add_mesh(path, color=input_data.color, line_width=3)
            plotter.show_axes()
            plotter.show_bounds()

        return output