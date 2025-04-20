from typing import List, Dict, Optional
import numpy as np

from mesh_nav_3D.planners import instantiate_planner
from mesh_nav_3D.planners.planner import PlannerConfig, PlannerInput, PlannerOutput

def compare_planners(
    mesh_file_path: str,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    planners: List[str],
    output_dir: str = "metrics_outputs",
    save_results: bool = True
) -> Dict[str, Optional[PlannerOutput]]:
    """
    Compare multiple planners on a given mesh with specified start and goal points.

    Args:
        mesh_file_path (str): Path to the mesh file (e.g., 'terrain_mesh.obj').
        start_point (np.ndarray): 3D coordinates of the start point.
        goal_point (np.ndarray): 3D coordinates of the goal point.
        planners (List[str]): List of planner names to compare (e.g., ['AStarPlanner', 'DijkstraPlanner']).
        output_dir (str): Directory to save planner outputs (default: 'metrics_outputs').
        save_results (bool): Whether to save results to files (default: True).

    Returns:
        Dict[str, Optional[PlannerOutput]]: Dictionary mapping planner names to their results or None if instantiation fails.
    """
    config = PlannerConfig(mesh_file_path=mesh_file_path, output_dir=output_dir)
    input_data = PlannerInput(start_point=start_point, goal_point=goal_point, **config.model_dump())

    results = {}

    for planner_name in planners:
        planner = instantiate_planner(planner_name)
        if planner:
            result = planner.plan(input_data)
            print(
                f"{planner_name}: Path Length={result.path_length:.2e}, "
                f"Efficiency={result.path_efficiency:.2e}, "
                f"Time={result.execution_time:.2e}s,"
                f"Memory Usage= {result.memory_used_mb}"
            )
            if save_results:
                result.save_to_file(f"{output_dir}/{planner_name}")
            results[planner_name] = result
        else:
            print(f"Failed to instantiate planner: {planner_name}")
            results[planner_name] = None

    return results
