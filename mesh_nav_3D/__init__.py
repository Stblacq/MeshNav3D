from mesh_nav_3D.visualize_planner import visualize_single_planner, visualize_multiple_planners

if __name__ == "__main__":
    # visualize_single_planner('dijkstra_planner')
    visualize_multiple_planners(['dijkstra_planner',"FlipOutPlanner"])
