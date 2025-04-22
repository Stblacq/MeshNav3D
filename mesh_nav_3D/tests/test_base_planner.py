from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mesh_nav_3D.planners.planner import PlannerInput, PlannerOutput


@pytest.fixture
def dummy_mesh():
    return MagicMock(name="DummyMesh")

@pytest.fixture
def valid_points():
    return np.array([1.0, 2.0, 3.0])

def test_planner_input_validates_points(dummy_mesh, valid_points):
    with patch("mesh_nav_3D.planners.planner.mesh_loader", return_value=dummy_mesh):
        data = PlannerInput(
            mesh_file_path="dummy",
            start_point=valid_points,
            goal_point=valid_points
        )
        assert isinstance(data.start_point, np.ndarray)
        assert data.start_point.shape == (3,)


def test_planner_output_path_length_computation(valid_points):
    path_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])
    out = PlannerOutput(
        start_point=valid_points,
        goal_point=valid_points,
        path_points=path_points
    )
    expected_length = 1.0 + 1.0
    assert np.isclose(out.path_length, expected_length)

def test_planner_output_success_flag(valid_points):
    out = PlannerOutput(
        start_point=valid_points,
        goal_point=valid_points,
        path_points=np.array([[0, 0, 0], [1, 0, 0]])
    )
    assert out.success is True

    out_empty = PlannerOutput(
        start_point=valid_points,
        goal_point=valid_points,
        path_points=np.array([[0, 0, 0]])
    )
    assert out_empty.success is False

def test_planner_output_efficiency(valid_points):
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    out = PlannerOutput(
        start_point=np.array([0, 0, 0]),
        goal_point=np.array([1, 1, 0]),
        path_points=path
    )
    total_dist = 1 + 1
    direct_dist = np.linalg.norm([1, 1, 0])
    expected_efficiency = total_dist - direct_dist
    assert np.isclose(out.path_efficiency, expected_efficiency)

def test_planner_output_file_saving(tmp_path, valid_points):
    path_points = np.array([[0, 0, 0], [1, 1, 1]])
    output = PlannerOutput(
        start_point=valid_points,
        goal_point=valid_points,
        path_points=path_points,
        execution_time=0.1,
        memory_used_mb=42.0
    )
    out_path = tmp_path / "test_output"
    output.save_to_file(str(out_path))

    assert (tmp_path / "test_output_points.npy").exists()

    loaded = np.load(tmp_path / "test_output_points.npy")
    np.testing.assert_array_equal(loaded, path_points)
