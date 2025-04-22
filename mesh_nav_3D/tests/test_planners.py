import numpy as np

from mesh_nav_3D.planners import (AStarPlanner,
                                  DijkstraPlanner,
                                  FlipOutPlanner,
                                  FastMarchingPlanner,
                                  ThetaStarPlanner,
                                  MMPPlanner,
                                  HeatMethodPlanner,
                                  GreedyBFSPlanner)
from mesh_nav_3D.planners.planner import PlannerInput


def test_planners():
    scenarios = [
        (1, np.array([5.00, 9.55, 0.95]), np.array([6.82, 1.36, 0.10]), 8.42),
        (2, np.array([9.09, 9.55, -0.33]), np.array([7.73, 1.36, 0.20]), 8.31),
        (3, np.array([5.00, 9.55, 0.95]), np.array([0.91, 0.91, 0.48]), 9.57),
        (4, np.array([9.09, 10.00, -0.27]), np.array([2.73, 0.91, 0.25]), 11.11),
        (5, np.array([0.45, 9.55, -0.44]), np.array([8.18, 1.36, 0.19]), 11.27),
    ]
    planners = [AStarPlanner,DijkstraPlanner, FlipOutPlanner, FastMarchingPlanner,
                GreedyBFSPlanner, HeatMethodPlanner, MMPPlanner, ThetaStarPlanner]

    for _, start,goal,_  in scenarios:

        planner_input = PlannerInput(mesh_file_path="terrain_mesh",
                                     start_point=start,
                                     goal_point=goal)
        for planner in planners:
            output = planner().plan(planner_input)

            assert output.success
            assert output.path_points is not None
            assert output.path_points.shape[1] == 3
            assert np.allclose(output.start_point, start)
            assert np.allclose(output.goal_point, goal)
            assert output.path_length is not None
            assert output.execution_time is not None
            assert output.memory_used_mb is not None
