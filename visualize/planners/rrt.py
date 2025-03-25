import random
import time
from typing import Optional, List

import numpy as np
import pyvista as pv

from visualize import Planner

pv.global_theme.allow_empty_mesh = True

def _visualize_path(plotter: pv.Plotter, path: List[np.ndarray]) -> None:
    """Visualize the final path"""
    # Create a polyline for the path
    path_points = np.array(path)
    path_poly = pv.PolyData()
    path_poly.points = path_points

    # Create lines connecting points
    cells = []
    for i in range(len(path) - 1):
        cells.extend([2, i, i + 1])
    path_poly.lines = cells

    # Add to plotter
    plotter.add_mesh(path_poly, color='yellow', line_width=4)
    plotter.update()


def _check_collision(p1: np.ndarray, p2: np.ndarray, mesh: pv.DataSet) -> bool:
    """Check if the line segment from p1 to p2 collides with the mesh"""
    # Create a line segment
    line = pv.Line(p1, p2)

    # Check for intersection with the mesh
    _, num = line.collision(mesh)

    # Check if there are intersection points
    return num > 0


class RRTPlanner(Planner):
    def __init__(self, step_size=0.5, goal_sample_rate=0.1, max_extend_length=0.5):
        """
        Initialize the RRT planner with parameters

        Args:
            step_size: Distance to extend tree in each iteration
            goal_sample_rate: Probability of sampling the goal point
            max_extend_length: Maximum length to extend tree in each iteration
        """
        self.lines_actor = None
        self.points_actor = None
        self.tree_lines = None
        self.tree_points = None
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_extend_length = max_extend_length
        self.tree = None
        self.path = None

    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """
        Plan a path from start_point to goal_point using RRT

        Args:
            start_point: Starting position as numpy array [x, y, z]
            goal_point: Goal position as numpy array [x, y, z]
            plotter: PyVista plotter object for visualization
            mesh: PyVista mesh for collision checking
            time_horizon: Maximum planning time in seconds
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary containing path and planning information or None if no path found
        """
        # Start timing
        start_time = time.time()

        # Initialize tree with start node
        self.tree = {'vertices': [start_point],
                     'edges': [],
                     'parents': [-1]}  # -1 indicates no parent

        # Initialize visualization if plotter is provided
        if plotter is not None:
            # Add start and goal points to the plot
            plotter.add_mesh(pv.Sphere(radius=0.2, center=start_point), color='green')
            plotter.add_mesh(pv.Sphere(radius=0.2, center=goal_point), color='red')

            # Add tree visualization
            self.tree_points = pv.PolyData(np.array([start_point]))
            self.tree_lines = pv.PolyData()
            self.points_actor = plotter.add_mesh(self.tree_points, color='white',
                                                 point_size=5, render_points_as_spheres=True)
            self.lines_actor = plotter.add_mesh(self.tree_lines, color='cyan', line_width=2)

        # Get bounds of the environment from the mesh
        bounds = mesh.bounds
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        # Main RRT loop
        i = 0
        goal_reached = False
        goal_idx = -1

        while i < max_iterations and (time.time() - start_time) < time_horizon:
            # Sample random point (with bias toward goal)
            if random.random() < self.goal_sample_rate:
                random_point = goal_point
            else:
                random_point = np.array([
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max),
                    random.uniform(z_min, z_max)
                ])

            # Find nearest vertex in the tree
            nearest_idx = self._find_nearest(random_point)
            nearest_point = self.tree['vertices'][nearest_idx]

            # Create new point by moving from nearest point toward random point
            new_point = self._steer(nearest_point, random_point)

            # Check if the path to new_point is collision-free
            if not _check_collision(nearest_point, new_point, mesh):
                # Add vertex and edge to tree
                self.tree['vertices'].append(new_point)
                new_idx = len(self.tree['vertices']) - 1
                self.tree['edges'].append((nearest_idx, new_idx))
                self.tree['parents'].append(nearest_idx)

                # Update visualization
                if plotter is not None:
                    self._update_visualization(plotter, nearest_point, new_point)
                    plotter.update()

                # Check if goal is reached
                distance_to_goal = np.linalg.norm(new_point - goal_point)
                if distance_to_goal < self.step_size:
                    # Check if there's a direct path to the goal
                    if not _check_collision(new_point, goal_point, mesh):
                        # Add goal to tree
                        self.tree['vertices'].append(goal_point)
                        goal_idx = len(self.tree['vertices']) - 1
                        self.tree['edges'].append((new_idx, goal_idx))
                        self.tree['parents'].append(new_idx)

                        # Update visualization with the final connection
                        if plotter is not None:
                            self._update_visualization(plotter, new_point, goal_point)
                            plotter.update()

                        goal_reached = True
                        break

            i += 1

        # Extract path if goal was reached
        if goal_reached:
            path = self._extract_path(goal_idx)

            # Visualize the final path
            if plotter is not None:
                _visualize_path(plotter, path)

            # Calculate path length
            path_length = 0.0
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(path[i+1] - path[i])

            # Return results
            return {
                'path': path,
                'iterations': i,
                'planning_time': time.time() - start_time,
                'path_length': path_length,
                'success': True
            }
        else:
            print(f"Failed to find path within constraints: {i} iterations, {time.time() - start_time:.2f} seconds")
            return None

    def _find_nearest(self, point: np.ndarray) -> int:
        """Find the nearest vertex index in the tree to the given point"""
        distances = [np.linalg.norm(p - point) for p in self.tree['vertices']]
        return distances.index(min(distances))

    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Create a new point by moving from from_point toward to_point"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_point
        else:
            return from_point + (direction / distance) * self.step_size

    def _extract_path(self, goal_idx: int) -> List[np.ndarray]:
        """Extract the path from start to goal using parent pointers"""
        path = []
        current_idx = goal_idx

        # Traverse from goal to start
        while current_idx != -1:
            path.append(self.tree['vertices'][current_idx])
            current_idx = self.tree['parents'][current_idx]

        # Reverse to get path from start to goal
        return path[::-1]

    def _update_visualization(self, plotter: pv.Plotter, p1: np.ndarray, p2: np.ndarray) -> None:
        """Update the visualization with a new point and edge"""
        # Add the new point
        points = np.vstack([self.tree_points.points, p2])
    
        # Update the points visualization
        plotter.remove_actor(self.points_actor)
        self.tree_points = pv.PolyData(points)
        self.points_actor = plotter.add_mesh(self.tree_points, color='white',
                                             point_size=5, render_points_as_spheres=True)
    
        # Add the new edge
        line = pv.Line(p1, p2)
        if self.tree_lines.n_points == 0:
            self.tree_lines = line
        else:
            self.tree_lines = self.tree_lines.merge(line)
    
        # Update the lines visualization
        plotter.remove_actor(self.lines_actor)
        self.lines_actor = plotter.add_mesh(self.tree_lines, color='cyan', line_width=2)
