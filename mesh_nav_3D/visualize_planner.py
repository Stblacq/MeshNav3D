import os
from typing import Union, Callable, List

from mesh_nav_3D.planners.planner import PlannerConfig
from mesh_nav_3D.planners import  instantiate_planner

from mesh_nav_3D.visualizer import SinglePlannerVisualizer, MultiPlannerVisualizer


def get_mesh_path(mesh_file_path: str) -> str:
    if os.path.isabs(mesh_file_path): final_mesh_path = mesh_file_path
    else: final_mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", f"{mesh_file_path}.obj")
    if not os.path.isfile(final_mesh_path): raise FileNotFoundError(f"Mesh file not found: {final_mesh_path}")
    return final_mesh_path


def visualize_single_planner(planner: Union[str, Callable], config: PlannerConfig):
    instance = instantiate_planner(planner)
    if instance:
        SinglePlannerVisualizer(instance, config).visualize()


def visualize_multiple_planners(planners: List[Union[str, Callable]], config: PlannerConfig):
    instances = [instantiate_planner(planner) for planner in planners if instantiate_planner(planner)]
    if not instances:
        raise ValueError("No valid planners to visualize.")
    MultiPlannerVisualizer(instances, config).visualize()
