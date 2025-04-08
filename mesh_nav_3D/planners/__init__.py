from typing import Union, Callable, Optional

from mesh_nav_3D.planners.a_star import AStarPlanner
from mesh_nav_3D.planners.dijkstra import DijkstraPlanner
from mesh_nav_3D.planners.edge_flip import FlipOutPlanner
from mesh_nav_3D.planners.fmm import FastMarchingPlanner
from mesh_nav_3D.planners.greedy_bfs import GreedyBFSPlanner
from mesh_nav_3D.planners.heat_method import HeatMethodPlanner
from mesh_nav_3D.planners.mmp import MMPPlanner
from mesh_nav_3D.planners.mppi import MPPIPlanner
from mesh_nav_3D.planners.planner import Planner
from mesh_nav_3D.planners.theta_star import ThetaStarPlanner

def snake_to_pascal(snake_str: str) -> str:
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def instantiate_planner(planner_: Union[str, Callable]) -> Optional[Planner]:
    try:
        class_name = snake_to_pascal(planner_) if isinstance(planner_, str) and '_' in planner_ else planner_
        return globals()[class_name]() if isinstance(planner_, str) else planner_()
    except Exception as e:
        print(f"Error instantiating planner {planner_}: {str(e)}")
        return None
