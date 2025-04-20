import os

import pybullet as p
import numpy as np
from time import sleep
import pyvista as pv

# Constants
PHYSICS_STEP = 1.0 / 200.0
CAMERA_CONFIG = {
    'distance': 15.0,
    'yaw': 0.0,
    'pitch': -30.0,
    'target': [0, 0, 0],
    'yaw_increment': 5.0,
    'pitch_increment': 5.0,
    'pitch_min': -89.0,
    'pitch_max': 89.0
}
MESH_CONFIG = {
    'file': os.path.join(os.path.dirname(os.path.abspath(__file__)),"meshes", "terrain_mesh.obj"),
    'scale': [3, 3, 3]
}
BALL_CONFIG = {
    'radius': 0.1,
    'mass': 1.0,
    'grid_size': 10,
    'z_offset': 0.5
}


def initialize_pybullet():
    """Initialize PyBullet simulation environment."""
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -10)
    p.setPhysicsEngineParameter(fixedTimeStep=PHYSICS_STEP)


def load_mesh_data():
    """Load and process mesh data using PyVista."""
    try:
        mesh = pv.read(MESH_CONFIG['file'])
        bounds = mesh.bounds
        center = np.array([(bounds[i] + bounds[i + 1]) / 2 for i in range(0, 6, 2)])
        extents = np.array([bounds[i + 1] - bounds[i] for i in range(0, 6, 2)])
        max_height = bounds[5]
    except Exception as e:
        print(f"Error loading mesh: {e}. Using default values.")
        center = np.array([0, 0, 0])
        extents = np.array([10, 10, 1])
        max_height = 1.0
    return center, extents, max_height


def create_terrain(center, extents):
    """Create terrain with collision and visual shapes."""
    position = -center * MESH_CONFIG['scale']
    position[2] = 0.0

    collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=MESH_CONFIG['file'],
        meshScale=MESH_CONFIG['scale'],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=MESH_CONFIG['file'],
        meshScale=MESH_CONFIG['scale']
    )

    terrain = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=[0, 0, 0, 1]
    )
    print(f"Terrain placed at: {position}")
    return terrain, position


def create_balls(extents, max_height):
    """Create a grid of balls."""
    balls = []
    init_positions = []

    width = extents[0] * MESH_CONFIG['scale'][0]
    length = extents[1] * MESH_CONFIG['scale'][1]
    z_pos = max_height * MESH_CONFIG['scale'][2] + BALL_CONFIG['radius'] + BALL_CONFIG['z_offset']

    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_CONFIG['radius'])

    for i in range(BALL_CONFIG['grid_size']):
        for j in range(BALL_CONFIG['grid_size']):
            x = (i - 4.5) * width / 9
            y = (j - 4.5) * length / 9
            ball = p.createMultiBody(
                baseMass=BALL_CONFIG['mass'],
                baseCollisionShapeIndex=collision,
                basePosition=[x, y, z_pos]
            )
            balls.append(ball)
            init_positions.append([x, y, z_pos])

    return balls, init_positions


def update_camera():
    """Update camera position based on current configuration."""
    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_CONFIG['distance'],
        cameraYaw=CAMERA_CONFIG['yaw'],
        cameraPitch=CAMERA_CONFIG['pitch'],
        cameraTargetPosition=CAMERA_CONFIG['target']
    )


def handle_input(balls, init_positions):
    """Handle keyboard input for camera control and ball reset."""
    keys = p.getKeyboardEvents()
    for key, state in keys.items():
        if not (state & p.KEY_WAS_TRIGGERED):
            continue

        if key == p.B3G_F5:
            print("Resetting balls to initial positions.")
            for ball, pos in zip(balls, init_positions):
                p.resetBasePositionAndOrientation(ball, pos, [0, 0, 0, 1])

        elif key == p.B3G_LEFT_ARROW:
            CAMERA_CONFIG['yaw'] -= CAMERA_CONFIG['yaw_increment']
        elif key == p.B3G_RIGHT_ARROW:
            CAMERA_CONFIG['yaw'] += CAMERA_CONFIG['yaw_increment']
        elif key == p.B3G_UP_ARROW:
            CAMERA_CONFIG['pitch'] = max(
                CAMERA_CONFIG['pitch_min'],
                CAMERA_CONFIG['pitch'] - CAMERA_CONFIG['pitch_increment']
            )
        elif key == p.B3G_DOWN_ARROW:
            CAMERA_CONFIG['pitch'] = min(
                CAMERA_CONFIG['pitch_max'],
                CAMERA_CONFIG['pitch'] + CAMERA_CONFIG['pitch_increment']
            )

        if key in (p.B3G_LEFT_ARROW, p.B3G_RIGHT_ARROW, p.B3G_UP_ARROW, p.B3G_DOWN_ARROW):
            update_camera()
            print(f"Camera yaw: {CAMERA_CONFIG['yaw']}, pitch: {CAMERA_CONFIG['pitch']}")


def main():
    """Main simulation loop."""
    initialize_pybullet()
    center, extents, max_height = load_mesh_data()
    terrain, terrain_pos = create_terrain(center, extents)
    balls, init_positions = create_balls(extents, max_height)
    update_camera()

    while True:
        handle_input(balls, init_positions)
        p.stepSimulation()
        sleep(PHYSICS_STEP)


if __name__ == "__main__":
    main()