import os
from typing import Union, Callable, Optional, List

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


def visualize_multiple_planners(planners: List[Union[str, Callable]],
                                mesh_file_path: str,
                                up: str = "z",
                                output_dir: str = os.path.join(os.getcwd(), "outputs")
                                ) -> None:
    planner_instances = []
    for planner in planners:
        planner_instance = instantiate_planner(planner)
        if planner_instance is None: continue
        planner_instances.append(planner_instance)
    if not planner_instances:raise ValueError("No valid planners to visualize.")
    final_mesh_path = get_mesh_path(mesh_file_path)
    visualizer = MultiPlannerVisualizer(final_mesh_path, planner_instances, up, output_dir)
    visualizer.visualize()



def visualize_single_planner(planner: Union[str, Callable],
                             mesh_file_path: str,
                             up: str = "z",
                             output_dir: str = os.path.join(os.getcwd(), "outputs")
                             ) -> None:
    planner_instance = instantiate_planner(planner)
    if planner_instance is None:
        return
    final_mesh_path = get_mesh_path(mesh_file_path)
    visualizer = SinglePlannerVisualizer(final_mesh_path, planner_instance, up,output_dir)
    visualizer.visualize()