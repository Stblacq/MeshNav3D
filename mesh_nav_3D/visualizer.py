from abc import ABC, abstractmethod
import numpy as np
import pyvista as pv
import os
from datetime import datetime
from planners.planner import Planner

class BaseVisualizer(ABC):
    def __init__(self, mesh_file_path: str,
                 up: str = "z",
                 output_dir: str = os.path.join(os.getcwd(), "outputs")):
        self.clicked_points = []
        self.mesh_file_path = mesh_file_path
        self.output_dir = output_dir
        self.transformation_matrix = self.get_up_axis_transform(up)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def get_up_axis_transform(up: str) -> np.ndarray:
        up = up.lower()
        if up not in 'xyz':
            raise ValueError("up must be 'x', 'y', or 'z'")
        transformation_matrix = np.eye(4)
        i = 'xyz'.index(up)
        transformation_matrix[[2, i]] = transformation_matrix[[i, 2]]
        return transformation_matrix

    def set_up_mesh(self, pv_mesh):
        pv_mesh = pv_mesh.transform(self.transformation_matrix)
        pv_mesh["Elevation"] = pv_mesh.points[:, 2]
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh,
                         scalars="Elevation",
                         cmap="terrain",
                         color='lightblue',
                         show_edges=True)
        plotter.add_axes(interactive=True)
        return plotter

    def save_to_npz(self, data: dict, planner_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plan_{planner_name}_{timestamp}.npz"
        filepath = os.path.join(self.output_dir, filename)
        save_dict = {}
        for key, value in data.items():
            if hasattr(value, 'points'):
                save_dict[key] = np.asarray(value.points)
            else:
                save_dict[key] = np.asarray(value) if not isinstance(value, (int, float, str, bool)) else value
        np.savez(filepath, **save_dict)
        return filepath

    @abstractmethod
    def visualize(self):
        pass

class SinglePlannerVisualizer(BaseVisualizer):
    def __init__(self, mesh_file_path: str,
                 planner: Planner,
                 up: str = "z",
                 output_dir: str = os.path.join(os.getcwd(), "outputs")
                 ):
        super().__init__(mesh_file_path, up, output_dir)
        self.planner = planner
        self.visualize()

    def visualize(self):
        pv_mesh = pv.read(self.mesh_file_path)
        plotter = self.set_up_mesh(pv_mesh)

        def callback(point, _):
            self.clicked_points.append(point)
            plotter.add_mesh(pv.PolyData(point), color='red', point_size=2)
            if len(self.clicked_points) == 2:
                start, goal = self.clicked_points
                self.clicked_points.clear()
                plan_result = self.planner.plan(start, goal, plotter, pv_mesh)
                # Save the planning result
                filepath = self.save_to_npz(plan_result, self.planner.__class__.__name__)
                print(f"Plan saved to: {filepath}")
                plotter.add_points(start, color='red', point_size=5)
                plotter.add_points(goal, color='green', point_size=5)
                plotter.show()

        plotter.enable_point_picking(callback=callback, use_picker=True)
        plotter.show()

class MultiPlannerVisualizer(BaseVisualizer):
    def __init__(self, mesh_file_path: str,
                 planners: list[Planner],
                 up: str = "z",
                 output_dir: str = os.path.join(os.getcwd(), "outputs")
                 ):
        super().__init__(mesh_file_path, up, output_dir)
        self.planners = planners
        self.visualize()

    def visualize(self):
        pv_mesh = pv.read(self.mesh_file_path)
        plotter = self.set_up_mesh(pv_mesh)
        colors = ["blue", "yellow", "purple", "orange", "cyan", "magenta"]

        def callback(point, _):
            self.clicked_points.append(point)
            plotter.add_mesh(pv.PolyData(point), color='red', point_size=2)
            if len(self.clicked_points) == 2:
                start, goal = self.clicked_points
                self.clicked_points.clear()
                for i, planner in enumerate(self.planners):
                    color = colors[i % len(colors)]
                    plan_result = planner.plan(start, goal, plotter, pv_mesh, color=color)
                    filepath = self.save_to_npz(plan_result, planner.__class__.__name__)
                    print(f"Plan saved to: {filepath}")
                plotter.add_points(start, color='red', point_size=5)
                plotter.add_points(goal, color='green', point_size=5)
                plotter.show()

        plotter.enable_point_picking(callback=callback, use_picker=True)
        plotter.show()