from abc import ABC, abstractmethod
import numpy as np
import pyvista as pv
import os
from datetime import datetime
from planners.planner import Planner, PlannerInput, PlannerConfig


class BaseVisualizer(ABC):
    def __init__(self, planner_config: PlannerConfig):
        self.mesh = planner_config.mesh
        self.output_dir = planner_config.output_dir
        self.transform = self._get_transform(planner_config.up)
        self.planner_config = planner_config
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _get_transform(up: str) -> np.ndarray:
        if up.lower() not in 'xyz':
            raise ValueError("up must be 'x', 'y', or 'z'")
        i = 'xyz'.index(up.lower())
        t = np.eye(4)
        t[[2, i]] = t[[i, 2]]
        return t

    def setup_plotter(self):
        mesh = self.mesh.transform(self.transform)
        mesh["Elevation"] = mesh.points[:, 2]
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars="Elevation", cmap="terrain", color='lightblue', show_edges=True)
        plotter.add_axes(interactive=True)
        self.mesh = mesh
        return plotter

    @abstractmethod
    def visualize(self): pass


class SinglePlannerVisualizer(BaseVisualizer):
    def __init__(self, planner: Planner, config: PlannerConfig):
        self.planner = planner
        super().__init__(config)

    def visualize(self):
        plotter = self.setup_plotter()
        points = []

        def callback(p, _):
            points.append(p)
            plotter.add_mesh(pv.PolyData(p), color='red', point_size=2)
            if len(points) < 2:
                return
            start, goal = points[:2]
            points.clear()

            plan_input = PlannerInput(start_point=start, goal_point=goal, **self.planner_config.model_dump())
            result = self.planner.plan(plan_input, plotter)
            path = os.path.join(self.output_dir, f"plan_{self.planner.__class__.__name__}_{datetime.now():%Y%m%d_%H%M%S}")
            result.save_to_file(path)
            print(f"Plan saved to: {path}")

            plotter.add_points(start, color='red', point_size=5)
            plotter.add_points(goal, color='green', point_size=5)
            plotter.show()

        plotter.enable_point_picking(callback=callback, use_picker=True)
        plotter.show()


class MultiPlannerVisualizer(BaseVisualizer):
    def __init__(self, planners: list[Planner], config: PlannerConfig):
        self.planners = planners
        super().__init__(config)

    def visualize(self):
        plotter = self.setup_plotter()
        points = []
        colors = ["blue", "yellow", "purple", "orange", "cyan", "magenta"]

        def callback(p, _):
            points.append(p)
            plotter.add_mesh(pv.PolyData(p), color='red', point_size=2)
            if len(points) < 2:
                return
            start, goal = points[:2]
            points.clear()

            for i, planner in enumerate(self.planners):
                color = colors[i % len(colors)]
                self.planner_config.color = color
                plan_input = PlannerInput(start_point=start,
                                          goal_point=goal,
                                          **self.planner_config.model_dump())
                result = planner.plan(plan_input, plotter)
                path = os.path.join(self.output_dir, f"plan_{planner.__class__.__name__}_{datetime.now():%Y%m%d_%H%M%S}")
                result.save_to_file(path)
                print(f"Plan saved to: {path}")

            plotter.add_points(start, color='red', point_size=5)
            plotter.add_points(goal, color='green', point_size=5)
            plotter.show()

        plotter.enable_point_picking(callback=callback, use_picker=True)
        plotter.show()
