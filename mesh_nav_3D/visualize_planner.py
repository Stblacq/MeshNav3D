import os
from typing import Union, Callable, Optional, List

from mesh_nav_3D.planners.planner import PlannerConfig
from planners import (Planner,
                      AStarPlanner,
                      DijkstraPlanner,
                      FlipOutPlanner,
                      FastMarchingPlanner,
                      GreedyBFSPlanner,
                      HeatMethodPlanner,
                      MMPPlanner,
                      MPPIPlanner,
                      ThetaStarPlanner)

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
