import pyvista as pv

from visualize.planners.dijkstra import dijkstra_planner
from visualize.planners.log_mppi import log_mppi_planner
from visualize.planners.mppi import mppi_planner
from visualize.planners.ssp import sequential_submesh_planner
import os
from typing import Callable



class Visualizer:
    def __init__(self, mesh_file_path: str, planner: Callable):
        self.clicked_points = []
        self.mesh_file_path = mesh_file_path
        self.planner = planner
        self.visualize()

    def visualize(self):
        if not os.path.isfile(self.mesh_file_path):
            print(f"The specified file was not found: {self.mesh_file_path}")
            return
        pv_mesh = pv.read(self.mesh_file_path)
        # transformation_matrix = np.array([[1, 0, 0, 0],
        #                                   [0, 0, 1, 0],
        #                                   [0, 1, 0, 0],
        #                                   [0, 0, 0, 1]])
        # pv_mesh = pv_mesh.transform(transformation_matrix)
        pv_mesh.compute_normals(inplace=True,progress_bar=True,flip_normals=True,
    consistent_normals=True,
    auto_orient_normals=True,)
        pv_mesh["Elevation"] = pv_mesh.points[:, 2]
        plotter = pv.Plotter()

        plotter.add_mesh(pv_mesh, scalars="Elevation", cmap="terrain", color='lightblue', show_edges=True)
        plotter.add_axes(interactive=True)

        def callback(point, _):
            self.clicked_points.append(point)
            plotter.add_mesh(pv.PolyData(point), color='red', point_size=2)

            if len(self.clicked_points) == 2:
                start, goal = self.clicked_points
                print(f"Start:{start}{type(start)}-Goal:{goal}{type(goal)}")
                self.clicked_points.clear()
                planning_response =  self.planner(start, goal, plotter, pv_mesh)

                plotter.add_points(start, color='red', point_size=5, label='Start Point')
                plotter.add_points(goal, color='green', point_size=5, label='Goal Point')

                plotter.add_legend()
                plotter.show()

        plotter.enable_point_picking(callback=callback, use_picker=True)
        plotter.show()


def run(example: str):
    if example == 'ssp':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", 'terrain_mesh.obj')
        visualizer = Visualizer(file_path, sequential_submesh_planner)
        visualizer.visualize()
    elif example == 'mppi':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", 'terrain_mesh.obj')
        visualizer = Visualizer(file_path, mppi_planner)
        visualizer.visualize()
    elif example == 'log_mppi':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", 'terrain_mesh.obj')
        visualizer = Visualizer(file_path, log_mppi_planner)
        visualizer.visualize()
    elif example == 'dijkstra_planner':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", 'terrain_mesh.obj')
        visualizer = Visualizer(file_path, dijkstra_planner)
        visualizer.visualize()
    else:
        print("Invalid example specified.")
        return


if __name__ == "__main__":
    run('dijkstra_planner')
