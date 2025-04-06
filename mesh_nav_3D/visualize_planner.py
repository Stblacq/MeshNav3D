import os
from typing import Union, Callable, Optional, List

from mesh_nav_3D.planners.a_star import AStarPlanner
from mesh_nav_3D.planners.dijkstra import DijkstraPlanner
from mesh_nav_3D.planners.edge_flip import FlipOutPlanner
from mesh_nav_3D.planners.fmm import FastMarchingPlanner
from mesh_nav_3D.planners.greedy_bfs import GreedyBFSPlanner
from mesh_nav_3D.planners.heat_method import HeatMethodPlanner
from mesh_nav_3D.planners.log_mppi import LogMPPIPlanner
from mesh_nav_3D.planners.mmp import MMPPlanner
from mesh_nav_3D.planners.mppi import MPPIPlanner
from mesh_nav_3D.planners.planner import Planner
from mesh_nav_3D.planners.theta_star import ThetaStarPlanner
from mesh_nav_3D.visualizer import SinglePlannerVisualizer, MultiPlannerVisualizer


def snake_to_pascal(snake_str: str) -> str:
    return ''.join(word.capitalize() for word in snake_str.split('_'))



def instantiate_planner(planner: Union[str, Callable]) -> Optional[Planner]:
    try:
        class_name = snake_to_pascal(planner) if isinstance(planner, str) and '_' in planner else planner
        return globals()[class_name]() if isinstance(planner, str) else planner()
    except Exception as e:
        print(f"Error instantiating planner {planner}: {str(e)}")
        return None



def visualize_multiple_planners(planners: List[Union[str, Callable]], up: str = "z") -> None:
    planner_instances = []
    for planner in planners:
        planner_instance = instantiate_planner(planner)
        if planner_instance is None:
            continue
        planner_instances.append(planner_instance)

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "terrain_mesh.obj")
    visualizer = MultiPlannerVisualizer(file_path, planner_instances, up)
    visualizer.visualize()


def visualize_single_planner(planner: Union[str, Callable], up: str = "z") -> None:
    planner_instance = instantiate_planner(planner)
    if planner_instance is None:
        return
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "terrain_mesh.obj")
    visualizer = SinglePlannerVisualizer(file_path, planner_instance, up)
    visualizer.visualize()