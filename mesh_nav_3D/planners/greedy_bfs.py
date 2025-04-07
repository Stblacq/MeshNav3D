import os
import time
from typing import Optional, Dict, Set
import heapq
import numpy as np
import psutil
import pyvista as pv

from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput

class GreedyBFSPlanner(Planner):
    def plan(self,
             input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **_) -> PlannerOutput:
        """
        Compute a geodesic path between start and goal points on a mesh using Greedy Best-First Search.
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
        max_iterations = input_data.max_iterations

        vertices = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]

        start_idx = np.argmin(np.linalg.norm(vertices - start_point, axis=1))
        goal_idx = np.argmin(np.linalg.norm(vertices - goal_point, axis=1))

        adjacency = self._build_adjacency_list(vertices, faces)
        path_indices = self._greedy_bfs(
            start_idx, goal_idx, vertices, adjacency, max_iterations
        )
        if path_indices is None or len(path_indices) == 0:
            raise ValueError("Path not found")

        path_points = vertices[path_indices]

        output = PlannerOutput(
            start_point=start_point,
            goal_point=goal_point,
            path_points=path_points,
            start_idx=start_idx,
            goal_idx=goal_idx,
            execution_time=time.time() - start_time,
            memory_used_mb=(process.memory_info().rss - start_memory) / 1024 / 1024,
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

    def _build_adjacency_list(self, vertices: np.ndarray, faces: np.ndarray) -> Dict[int, Set[int]]:
        """Build adjacency list from mesh faces."""
        adjacency = {i: set() for i in range(len(vertices))}
        for face in faces:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                adjacency[face[i]].add(face[j])
                adjacency[face[j]].add(face[i])
        return adjacency

    def _greedy_bfs(self, start_idx: int, goal_idx: int, vertices: np.ndarray,
                    adjacency: Dict[int, Set[int]], max_iterations: int) -> Optional[list]:
        """Perform Greedy Best-First Search."""
        goal_pos = vertices[goal_idx]
        frontier = [(np.linalg.norm(vertices[start_idx] - goal_pos), 0, start_idx)]
        came_from = {start_idx: None}
        visited = set()
        iteration = 0

        while frontier and iteration < max_iterations:
            _, _, current_idx = heapq.heappop(frontier)
            if current_idx == goal_idx:
                path = []
                while current_idx is not None:
                    path.append(current_idx)
                    current_idx = came_from[current_idx]
                return path[::-1]
            if current_idx in visited:
                continue
            visited.add(current_idx)
            for neighbor_idx in adjacency[current_idx]:
                if neighbor_idx not in visited:
                    heuristic = np.linalg.norm(vertices[neighbor_idx] - goal_pos)
                    heapq.heappush(frontier, (heuristic, iteration + 1, neighbor_idx))
                    if neighbor_idx not in came_from:
                        came_from[neighbor_idx] = current_idx
            iteration += 1
        return None