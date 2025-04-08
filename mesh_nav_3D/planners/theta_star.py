import os
import time
from typing import Optional
import numpy as np
import psutil
import pyvista as pv
from heapq import heappush, heappop

from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput

class ThetaStarPlanner(Planner):
    def plan(self, input_data: PlannerInput, plotter: Optional[pv.Plotter] = None, **_) -> PlannerOutput:
        """
        Compute a path between start and goal points on a mesh using Theta* algorithm.

        Args:
            input_data (PlannerInput): Input data containing start_point, goal_point, mesh, etc.
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization

        Returns:
            PlannerOutput: Object containing path information
        """
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss

        start_point = np.asarray(input_data.start_point).reshape(3)
        goal_point = np.asarray(input_data.goal_point).reshape(3)
        mesh = input_data.mesh
        color = getattr(input_data, 'color', 'blue')
        max_iterations = getattr(input_data, 'max_iterations', 1000)

        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()

        edges_polydata = mesh.extract_all_edges()
        edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]
        vertices = edges_polydata.points

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))
        start_vertex = vertices[start_idx]
        goal_vertex = vertices[goal_idx]

        adj_list = {i: [] for i in range(len(vertices))}
        for p1_idx, p2_idx in edge_cells:
            dist = np.linalg.norm(vertices[p1_idx] - vertices[p2_idx])
            adj_list[p1_idx].append((p2_idx, dist))
            adj_list[p2_idx].append((p1_idx, dist))

        def heuristic(vertex_idx):
            return np.linalg.norm(vertices[vertex_idx] - goal_vertex)

        def line_of_sight(p1_idx: int, p2_idx: int) -> bool:
            """
            Check if there's a clear line of sight between two vertices using ray tracing.

            Args:
                p1_idx (int): Index of first vertex
                p2_idx (int): Index of second vertex

            Returns:
                bool: True if line of sight exists, False otherwise
            """
            p1 = vertices[p1_idx]
            p2 = vertices[p2_idx]
            direction = p2 - p1
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                return True
            direction = direction / distance
            start = p1 + direction * 1e-4
            points, _ = mesh.ray_trace(start, p2)
            for point in points:
                dist_to_intersection = np.linalg.norm(point - start)
                if dist_to_intersection < distance - 1e-4:
                    return False
            return True

        open_set = [(heuristic(start_idx), start_idx, 0, [start_vertex])]  # (f_score, vertex_idx, g_score, path)
        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: heuristic(start_idx)}
        visited = set()
        iteration = 0

        while open_set and iteration < max_iterations:
            f, current_idx, g, path = heappop(open_set)
            iteration += 1

            if current_idx in visited:
                continue
            visited.add(current_idx)

            if current_idx == goal_idx:
                path_points = np.array(path)
                output = PlannerOutput(
                    start_point=start_point,
                    goal_point=goal_point,
                    path_points=path_points,
                    start_idx=start_idx,
                    goal_idx=goal_idx,
                    execution_time=time.time() - start_time,
                    memory_used_mb=(process.memory_info().rss - start_memory) / 1024 / 1024,
                )
                break

            for next_idx, dist in adj_list[current_idx]:
                if next_idx in visited:
                    continue

                if current_idx in came_from:
                    parent_idx = came_from[current_idx]
                    if line_of_sight(parent_idx, next_idx):
                        tentative_g = g_score[parent_idx] + np.linalg.norm(
                            vertices[parent_idx] - vertices[next_idx]
                        )
                        parent_path = path[:-1]
                    else:
                        tentative_g = g + dist
                        parent_path = path
                else:
                    tentative_g = g + dist
                    parent_path = path

                if next_idx not in g_score or tentative_g < g_score[next_idx]:
                    came_from[next_idx] = current_idx
                    g_score[next_idx] = tentative_g
                    f_score[next_idx] = tentative_g + heuristic(next_idx)
                    new_path = parent_path + [vertices[next_idx]]
                    heappush(open_set, (f_score[next_idx], next_idx, tentative_g, new_path))
        else:
            output = PlannerOutput(
                start_point=start_point,
                goal_point=goal_point,
                start_idx=start_idx,
                goal_idx=goal_idx,
                execution_time=time.time() - start_time,
                memory_used_mb=(process.memory_info().rss - start_memory) / 1024 / 1024,
            )

        if plotter is not None:
            plotter.add_points(start_point, color='red', point_size=10)
            plotter.add_points(goal_point, color='green', point_size=10)
            if output.success and output.path_points is not None:
                plotter.add_points(output.path_points, color=color, point_size=5)
                plotter.add_lines(np.vstack((output.path_points[:-1], output.path_points[1:])).T.reshape(-1, 3),
                                  color=color)
            plotter.show_axes()
            plotter.show_bounds()

        return output