import os
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pyvista as pv
from pydantic import BaseModel, field_validator, computed_field, Field
import json
from pathlib import Path


def mesh_loader(mesh_file_path):
    if os.path.isabs(mesh_file_path): final_path = mesh_file_path
    else: final_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "meshes", f"{mesh_file_path}.obj")
    if not os.path.isfile(final_path): raise FileNotFoundError(f"Mesh file not found: {final_path}")
    return pv.read(final_path)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        return super().default(obj)


class PlannerConfig(BaseModel):
    """Base configuration shared across planner models."""
    mesh_file_path: str
    mesh: pv.DataSet = Field(init=False, default=None)
    color: str = "blue"
    time_horizon: float = 10.0
    max_iterations: int = 1000
    up: str = "z"
    output_dir: str = "outputs"

    model_config = dict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        mesh = mesh_loader(self.mesh_file_path)
        self.mesh = mesh


class PlannerInput(PlannerConfig):
    """Pydantic model for planner input parameters."""
    start_point: np.ndarray
    goal_point: np.ndarray

    @classmethod
    @field_validator('start_point', 'goal_point')
    def validate_points(cls, value):
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError("Points must be 3D coordinates [x, y, z]")
        return value

class PlannerOutput(BaseModel):
    """Pydantic model for planner output results."""
    start_point: np.ndarray
    goal_point: np.ndarray
    path_points: Optional[np.ndarray] = None
    start_idx: Optional[int] = None
    goal_idx: Optional[int] = None
    execution_time: Optional[float] = None
    memory_used_mb: Optional[float] = None

    @field_validator('start_point', 'goal_point')
    def validate_points(cls, value):
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError("Points must be 3D coordinates [x, y, z]")
        return value

    @computed_field
    @property
    def path_length(self) -> Optional[float]:
        """Compute total path length if path_points are available."""
        if self.path_points is None or len(self.path_points) < 2:
            return None
        segments = np.diff(self.path_points, axis=0)
        return float(np.sum(np.linalg.norm(segments, axis=1)))

    @computed_field
    @property
    def success(self) -> bool:
        """Success is True if a valid path exists."""
        return self.path_points is not None and len(self.path_points) >= 2

    @computed_field
    @property
    def path_efficiency(self) -> Optional[float]:
        """Compute path efficiency as redundant distance (extra vs direct)."""
        if self.path_points is None or len(self.path_points) < 2:
            return None
        total_distance = np.sum(np.linalg.norm(np.diff(self.path_points, axis=0), axis=1))
        direct_distance = np.linalg.norm(self.goal_point - self.start_point)
        return float(total_distance - direct_distance)

    model_config = dict(arbitrary_types_allowed=True,computed_fields = True)

    def save_to_file(self, filepath: str) -> None:
        """
        Save the planner output to files.
        - Metadata as JSON at {filepath}.json
        - path_points as .npy at {filepath}_points.npy if present
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data_dict = self.model_dump(exclude={'path_points'})

        json_content = json.dumps(data_dict, indent=4, cls=NumpyEncoder)
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as file:
            file.write(json_content)

        if self.path_points is not None:
            npy_path = path.with_name(f"{path.name}_points").with_suffix('.npy')
            np.save(npy_path, self.path_points)

class Planner(ABC):
    @abstractmethod
    def plan(self,
             input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **kwargs) -> PlannerOutput:
        """Method to plan the best path using PlannerInput."""