import time

import numpy as np
import pyvista as pv



theta_max = np.radians(90)
a_max = 0.5
steering_rate_max = np.radians(90)
v_max =0.9

def cost_function(current_state, next_state, control, target):
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

def project_to_closest_face(x, mesh):
    surface = mesh.extract_surface()
    closest_point_id = surface.find_closest_point(x[:3])

    closest_point = surface.points[closest_point_id]
    closest_normal = surface.point_normals[closest_point_id]

    # Plane equation: Ax + By + Cz + D = 0
    A, B, C = closest_normal
    D = -np.dot(closest_normal, closest_point)

    # Distance from point to plane
    distance_to_plane = (A * x[0] + B * x[1] + C * x[2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)

    # Projected point
    projected_point = x[:3] - distance_to_plane * closest_normal

    x[:3] = projected_point
    return x

def log_mppi(x, target, dt,  mesh, num_samples=100, time_horizon=10, lambda_=1.0):
    p = 0.2
    # normal_samples = np.random.normal(loc=0, scale=1, size=(num_samples, time_horizon, 2))
    # lognormal_samples = np.random.lognormal(mean=0, sigma=1, size=(num_samples, time_horizon, 2))
    #
    # mix_mask = np.random.rand(num_samples, time_horizon, 2) < p
    # control_samples = np.where(mix_mask, normal_samples, lognormal_samples)

    control_samples = np.random.normal(loc=0, scale=1, size=(num_samples, time_horizon, 2))


    trajectory_costs = get_trajectories_costs(control_samples, dt, target, x,  mesh)

    weights = np.exp(-trajectory_costs / lambda_)

    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / num_samples
    else:
        weights /= np.sum(weights)

    optimal_control = np.dot(weights, control_samples[:, 0])

    return optimal_control


def get_trajectories_costs(control_samples, dt, target, x,  mesh):
    num_samples, _, _ = control_samples.shape
    trajectory_costs = np.zeros(num_samples)
    for i in range(num_samples):
        control_sequence = control_samples[i]
        cost = get_trajectory_cost(dt, control_sequence, target, x,  mesh)
        trajectory_costs[i] = cost

    min_cost = np.min(trajectory_costs)
    trajectory_costs -= min_cost
    return trajectory_costs


def get_trajectory_cost(dt, control_sequence, target, current_state,  mesh):
    x_current = current_state
    cost = 0
    for t in range(control_sequence.shape[0]):
        u = control_sequence[t]
        x_next = dynamics(x_current, u, dt)
        cost += cost_function(x_current, x_next, u, target)
        x_current = x_next
    return cost


def log_mppi_planner(start_point: np.ndarray,
                 goal_point: np.ndarray,
                 plotter: pv.Plotter | None,
                 mesh: pv.DataSet,
                 time_horizon=10) -> dict :
    dt = 0.3
    max_steps = 100000
    trajectory = []
    control_efforts = []
    x_current = np.concatenate([start_point, [0, 0]])  # start_point was 3D, now adding yaw=0, v=0

    trajectory.append(x_current.copy())
    start_time = time.time()

    for step in range(max_steps):
        optimal_control = log_mppi(x_current, goal_point, dt,
                               mesh, num_samples=100, time_horizon=time_horizon, lambda_=1.0)
        x_current = dynamics(x_current, optimal_control, dt)
        x_proj = project_to_closest_face(x_current, mesh)
        x_current = x_proj
        trajectory.append(x_current.copy())
        control_efforts.append(optimal_control)
        plotter.add_mesh(pv.Sphere(radius=0.05, center=x_current[:3]), color='pink')


        if np.linalg.norm(x_current[:3] - goal_point) < 0.1:
            plotter.add_mesh(pv.Sphere(radius=0.05, center=goal_point), color='green')
            trajectory.append(np.concatenate([goal_point, [0, 0]]))
            execution_time = time.time() - start_time

            return {
                "trajectory": trajectory,
                "controls": control_efforts,
                "execution_time": execution_time,
            }

    print("Max steps reached without reaching the goal.")
    return None
