from typing import Optional
import numpy as np
import pyvista as pv
from heapq import heappush, heappop

from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput

class FastMarchingPlanner(Planner):
    def plan(self,
             input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **_) -> PlannerOutput:
        """
        Compute a path between start and goal points on a mesh using Fast-Marching Method.
        Args:
            input_data (PlannerInput): Input data containing start_point, goal_point, mesh, etc.
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization
        Returns:
            PlannerOutput: Object containing path information
        """
        start_point = np.asarray(input_data.start_point).reshape(3)
        goal_point = np.asarray(input_data.goal_point).reshape(3)
        mesh = input_data.mesh
        max_iterations = input_data.max_iterations

        edges_polydata = mesh.extract_all_edges()
        edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]
        vertices = edges_polydata.points

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))
        goal_vertex = vertices[goal_idx]

        adj_list = {i: [] for i in range(len(vertices))}
        for p1_idx, p2_idx in edge_cells:
            dist = np.linalg.norm(vertices[p1_idx] - vertices[p2_idx])
            adj_list[p1_idx].append((p2_idx, dist))
            adj_list[p2_idx].append((p1_idx, dist))

        distances = np.full(len(vertices), np.inf)
        distances[start_idx] = 0.0
        heap = [(0.0, start_idx)]
        visited = set()

        while heap:
            dist, current_idx = heappop(heap)
            if current_idx in visited:
                continue
            visited.add(current_idx)
            for next_idx, edge_dist in adj_list[current_idx]:
                if next_idx in visited:
                    continue
                new_dist = dist + edge_dist
                if new_dist < distances[next_idx]:
                    distances[next_idx] = new_dist
                    heappush(heap, (new_dist, next_idx))

        if np.isinf(distances[goal_idx]):
            output = PlannerOutput(
                start_point=start_point,
                goal_point=goal_point,
                path_points=None,
                path_length=0.0,
                start_idx=start_idx,
                goal_idx=goal_idx,
                success=False
            )
        else:
            path = [goal_vertex]
            current_point = goal_vertex
            current_idx = goal_idx
            path_length = 0.0
            iteration = 0

            while current_idx != start_idx and iteration < max_iterations:
                min_dist = np.inf
                next_idx = None
                for neighbor_idx, edge_dist in adj_list[current_idx]:
                    if distances[neighbor_idx] < min_dist:
                        min_dist = distances[neighbor_idx]
                        next_idx = neighbor_idx
                if next_idx is None or min_dist >= distances[current_idx]:
                    break
                path_length += np.linalg.norm(vertices[next_idx] - current_point)
                current_point = vertices[next_idx]
                path.append(current_point)
                current_idx = next_idx
                iteration += 1

            if current_idx != start_idx:
                output = PlannerOutput(
                    start_point=start_point,
                    goal_point=goal_point,
                    path_points=None,
                    path_length=0.0,
                    start_idx=start_idx,
                    goal_idx=goal_idx,
                    success=False
                )
            else:
                path_points = np.array(path[::-1])
                output = PlannerOutput(
                    start_point=start_point,
                    goal_point=goal_point,
                    path_points=path_points,
                    path_length=path_length,
                    start_idx=start_idx,
                    goal_idx=goal_idx,
                    success=True
                )

        if plotter is not None:
            plotter.add_mesh(mesh, opacity=0.5)
            plotter.add_points(start_point, color='red', point_size=10)
            plotter.add_points(goal_point, color='green', point_size=10)
            if output.success and output.path_points is not None:
                plotter.add_points(output.path_points, color=input_data.color, point_size=5)
            plotter.show_axes()
            plotter.show_bounds()

        return output