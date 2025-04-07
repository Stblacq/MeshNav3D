import time
from typing import Optional

import numpy as np
import pyvista as pv

from mesh_nav_3D.planners.planner import Planner, PlannerInput, PlannerOutput


def cost_function(current_state, next_state, control, target, mesh):
    distance_cost = np.linalg.norm(next_state[:3] - target[:3])

    a_to_b = next_state[:3] - current_state[:3]
    b_to_target = target[:3] - next_state[:3]
    norm_a_to_b = np.linalg.norm(a_to_b)
    norm_b_to_target = np.linalg.norm(b_to_target)

    if norm_a_to_b > 1e-6 and norm_b_to_target > 1e-6:  # Avoid division by zero
        cos_theta = np.clip(np.dot(a_to_b / norm_a_to_b, b_to_target / norm_b_to_target), -1.0, 1.0)
    else:
        cos_theta = 1.0  # Default to 1 (zero angle) if there's no meaningful direction

    theta = np.arccos(cos_theta)
    theta_max = np.radians(60)
    a_max = 0.5
    steering_rate_max = np.radians(60)

    curvature_cost = (max(0, theta - theta_max) / theta_max) ** 2

    acceleration_cost = (max(0, abs(control[0]) - a_max) / a_max) ** 2
    steering_rate_cost = (max(0, abs(control[1]) - steering_rate_max) / steering_rate_max) ** 2

    return distance_cost + curvature_cost + acceleration_cost + steering_rate_cost


def dynamics(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """ x = [x, y, z, yaw, v], u = [acceleration, steering angle rate] """

    acceleration, steering_rate = u
    x_pos, y_pos, z_pos, yaw, v = x

    # Update velocity
    v_new = v + acceleration * dt
    v_new = max(0, v_new)  # Ensure non-negative velocity

    # Update yaw
    yaw_new = yaw + steering_rate * dt

    # Update position
    x_new = x_pos + v_new * np.cos(yaw) * dt
    y_new = y_pos + v_new * np.sin(yaw) * dt
    z_new = z_pos  # Assuming flat ground

    return np.array([x_new, y_new, z_new, yaw_new, v_new])


def mppi(x, target, dt, mesh, num_samples=100, time_horizon=10, lambda_=1.0):
    control_samples = np.random.uniform(-1, 1,
                                        (num_samples, time_horizon, 2))  # Only 2 controls now (accel, steering_rate)

    trajectory_costs = get_trajectories_costs(control_samples, dt, target, x, mesh)

    weights = np.exp(-trajectory_costs / lambda_)

    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / num_samples
    else:
        weights /= np.sum(weights)

    optimal_control = np.dot(weights, control_samples[:, 0])

    return optimal_control


def get_trajectories_costs(control_samples, dt, target, x, mesh):
    num_samples, _, _ = control_samples.shape
    trajectory_costs = np.zeros(num_samples)
    for i in range(num_samples):
        control_sequence = control_samples[i]
        cost = get_trajectory_cost(dt, control_sequence, target, x, mesh)
        trajectory_costs[i] = cost

    min_cost = np.min(trajectory_costs)
    trajectory_costs -= min_cost
    return trajectory_costs


def get_trajectory_cost(dt, control_sequence, target, current_state, mesh):
    x_current = current_state
    cost = 0
    for t in range(control_sequence.shape[0]):
        u = control_sequence[t]
        x_next = dynamics(x_current, u, dt)
        cost += cost_function(x_current, x_next, u, target, mesh)
        x_current = x_next
    return cost


def project_to_closest_face(x, mesh):
    surface = mesh.extract_surface()
    closest_point_id = surface.find_closest_point(x[:3])

    closest_point = surface.points[closest_point_id]
    closest_normal = surface.point_normals[closest_point_id]

    A, B, C = closest_normal
    D = -np.dot(closest_normal, closest_point)

    distance_to_plane = (A * x[0] + B * x[1] + C * x[2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)

    projected_point = x[:3] - distance_to_plane * closest_normal

    x[:3] = projected_point
    return x


class MPPIPlanner(Planner):
    def plan(self, input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **kwargs) -> PlannerOutput:
        """
        Compute a path between start and goal points on a mesh using Model Predictive Path Integral (MPPI) control.

        Args:
            input_data (PlannerInput): Input data containing start_point, goal_point, mesh, etc.
            plotter (Optional[pv.Plotter]): PyVista plotter for visualization.

        Returns:
            PlannerOutput: Object containing path information such as trajectory, controls, and execution time.
        """
        start_point = np.asarray(input_data.start_point).reshape(3)
        goal_point = np.asarray(input_data.goal_point).reshape(3)
        mesh = input_data.mesh
        color = input_data.color
        time_horizon = input_data.time_horizon
        max_iterations = input_data.max_iterations

        dt = kwargs.get('dt',0.5)
        trajectory = []
        control_efforts = []
        x_current = np.concatenate([start_point, [0, 0]])  # [x, y, z, yaw, v]

        trajectory.append(x_current.copy())
        start_time = time.time()

        for step in range(max_iterations):
            optimal_control = mppi(x_current, goal_point, dt,
                                   mesh, num_samples=100,
                                   time_horizon=time_horizon,
                                   lambda_=1.0)
            x_current = dynamics(x_current, optimal_control, dt)
            x_proj = project_to_closest_face(x_current, mesh)
            x_current = x_proj
            trajectory.append(x_current.copy())
            control_efforts.append(optimal_control)

            if plotter is not None:
                plotter.add_mesh(pv.Sphere(radius=0.3, center=x_current[:3]), color=color)

            if np.linalg.norm(x_current[:3] - goal_point) < 0.1:
                if plotter is not None:
                    plotter.add_mesh(pv.Sphere(radius=0.3, center=goal_point), color=color)
                trajectory.append(np.concatenate([goal_point, [0, 0]]))
                execution_time = time.time() - start_time

                return PlannerOutput(
                    start_point=start_point,
                    goal_point=goal_point,
                    path_points=np.array(trajectory),
                    path_length=0,
                    start_idx=0,
                    goal_idx=0,
                    success=True
                )
        print("Max steps reached without reaching the goal.")
        execution_time = time.time() - start_time
        return PlannerOutput(
                    start_point=start_point,
                    goal_point=goal_point,
                    path_points=np.array(trajectory),
                    path_length=0,
                    start_idx=0,
                    goal_idx=0,
                    success=True
                )