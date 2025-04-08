import numpy as np

from mesh_nav_3D.compare_planners import compare_planners
from mesh_nav_3D.visualize_planner import visualize_single_planner, visualize_multiple_planners, PlannerConfig

# if __name__ == "__main__":
#     # visualize_single_planner('dijkstra_planner',PlannerConfig(mesh_file_path="terrain_mesh"))
#     # visualize_multiple_planners(['dijkstra_planner', 'FlipOutPlanner'],
#     #                             PlannerConfig(mesh_file_path="terrain_mesh"))
#


if __name__ == "__main__":
    # scenarios = [
    #     (1, np.array([5.00, 9.55, 0.95]), np.array([6.82, 1.36, 0.10]), 8.42),
    #     (2, np.array([9.09, 9.55, -0.33]), np.array([7.73, 1.36, 0.20]), 8.31),
    #     (3, np.array([5.00, 9.55, 0.95]), np.array([0.91, 0.91, 0.48]), 9.57),
    #     (4, np.array([9.09, 10.00, -0.27]), np.array([2.73, 0.91, 0.25]), 11.11),
    #     (5, np.array([0.45, 9.55, -0.44]), np.array([8.18, 1.36, 0.19]), 11.27)
    # ]
    scenarios = [
        (1, np.array([0.56, 0.89, -0.93]), np.array([0.78, 0.22, 0.49]), 1.58),
        (2, np.array([0.22, 0.89, -0.60]), np.array([0.67, 0.22, 0.66]), 1.50),
        (3, np.array([1.00, 0.33, 0.00]), np.array([0.11, 0.78, -0.26]), 1.03),
        (4, np.array([0.33, 1.00, -0.87]), np.array([0.89, 0.11, 0.32]), 1.58),
        (5, np.array([0.11, 1.00, -0.34]), np.array([0.78, 0.33, 0.32]), 1.15)
    ]
    for scenario in scenarios:
        print(f">>>>>>>>>>>>>>>>>>>>> Scenario {scenario[0]}")
        start_point = scenario[1]
        goal_point = scenario[2]
        planners = ["AStarPlanner",
                    "DijkstraPlanner",
                    "FlipOutPlanner",
                    "FastMarchingPlanner",
                    "GreedyBFSPlanner",
                    "HeatMethodPlanner" ,
                    "MMPPlanner",
                    "ThetaStarPlanner"]
        results = compare_planners(
            mesh_file_path="simple_terrain",
            start_point=start_point,
            goal_point=goal_point,
            planners=planners,
            output_dir="metrics_outputs",
            save_results=True
        )
    # print(results)