from mesh_nav_3D.visualize_planner import visualize_single_planner, visualize_multiple_planners, PlannerConfig

if __name__ == "__main__":
    # visualize_single_planner('dijkstra_planner')
    visualize_multiple_planners(['dijkstra_planner', 'FlipOutPlanner'],
                                PlannerConfig(mesh_file_path="terrain_mesh"))

