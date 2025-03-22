import numpy as np
import pyvista as pv
from queue import PriorityQueue
from typing import Optional

from visualize.planners.planner import Planner


class JPSPlanner(Planner):
    def __init__(self):
        self.edges = {}

    def _build_graph(self, mesh: pv.PolyData):
        """Constructs the adjacency graph from the 3D mesh."""
        self.edges = {}
        
        points = mesh.points
        connectivity = mesh.extract_geometry().faces.reshape(-1, 4)[:, 1:]
        
        for triangle in connectivity:
            for i, u in enumerate(triangle):
                for j, v in enumerate(triangle):
                    if i != j:
                        self.edges.setdefault(u, set()).add(v)
                        self.edges.setdefault(v, set()).add(u)

    def _jump(self, current, direction, goal):
        """Recursive jump function to move in a direction until an obstacle or forced neighbor is found."""
        if current not in self.edges:
            return None
        
        next_node = current + direction
        if next_node not in self.edges:
            return None
        
        if next_node == goal or any(neigh not in self.edges[next_node] for neigh in self.edges[current]):
            return next_node
        
        return self._jump(next_node, direction, goal)

    def _find_neighbors(self, node):
        """Returns valid neighbors of a node."""
        return list(self.edges.get(node, []))

    def plan(self, start_point: np.ndarray, goal_point: np.ndarray, plotter: Optional[pv.Plotter], mesh: pv.DataSet) -> Optional[dict]:
        """Performs Jump Point Search (JPS) on the 3D mesh."""
        start_point = np.asarray(start_point).reshape(3)
        goal_point = np.asarray(goal_point).reshape(3)

        start_idx = mesh.find_closest_point(start_point)
        goal_idx = mesh.find_closest_point(goal_point)

        self._build_graph(mesh)
        
        if start_idx not in self.edges or goal_idx not in self.edges:
            print("JPS could not find a valid path.")
            return None

        pq = PriorityQueue()
        pq.put((0, start_idx))
        came_from = {}
        g_score = {start_idx: 0}

        while not pq.empty():
            _, current = pq.get()
            
            if current == goal_idx:
                break
            
            for neighbor in self._find_neighbors(current):
                direction = neighbor - current
                jump_point = self._jump(current, direction, goal_idx)
                
                if jump_point and jump_point not in g_score:
                    g_score[jump_point] = g_score[current] + np.linalg.norm(mesh.points[jump_point] - mesh.points[current])
                    pq.put((g_score[jump_point], jump_point))
                    came_from[jump_point] = current

        if goal_idx not in came_from:
            print("JPS could not find a valid path.")
            return None

        path = []
        current = goal_idx
        while current in came_from:
            path.append(mesh.points[current])
            current = came_from[current]
        path.append(mesh.points[start_idx])
        path.reverse()

        result = {'path_points': np.array(path), 'success': True}
        
        if plotter:
            plotter.add_mesh(mesh, opacity=0.5)
            plotter.add_points(start_point, color='red', point_size=10)
            plotter.add_points(goal_point, color='green', point_size=10)
            plotter.add_mesh(pv.PolyData(path), color='yellow', line_width=3)
            plotter.show_axes()
            plotter.show_bounds()
            # plotter.show()
        
        return result
