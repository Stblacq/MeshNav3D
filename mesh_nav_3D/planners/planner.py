from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pyvista as pv
from pydantic import BaseModel, field_validator
import json
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PlannerInput(BaseModel):
    """Pydantic model for planner input parameters."""
    start_point: np.ndarray
    goal_point: np.ndarray
    mesh: pv.DataSet
    color: str = "blue"
    time_horizon: int = 10.0
    max_iterations: int = 1000

    @classmethod
    @field_validator('start_point', 'goal_point')
    def validate_points(cls, value):
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError("Points must be 3D coordinates [x, y, z]")
        return value

    model_config = dict(
        arbitrary_types_allowed=True
    )


class PlannerOutput(BaseModel):
    """Pydantic model for planner output results."""
    start_point: np.ndarray
    goal_point: np.ndarray
    path_points: Optional[np.ndarray] = None
    path_length: float = 0.0
    start_idx: Optional[int] = None
    goal_idx: Optional[int] = None
    success: bool = False

    @field_validator('start_point', 'goal_point')
    def validate_points(cls, value):
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError("Points must be 3D coordinates [x, y, z]")
        return value

    model_config = dict(
        arbitrary_types_allowed=True
    )

    def save_to_file(self, filepath: str) -> None:
        """
        Save the planner output to files.
        - Metadata as JSON at {filepath}.json
        - path_points as .npy at {filepath}_points.npy if present
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get the data dictionary, excluding path_points
        data_dict = self.model_dump(exclude={'path_points'})

        # Serialize with the custom encoder
        json_content = json.dumps(data_dict, indent=4, cls=NumpyEncoder)

        # Write to JSON file
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as file:
            file.write(json_content)

        # Save path_points separately if present
        if self.path_points is not None:
            npy_path = path.with_name(f"{path.name}_points").with_suffix('.npy')
            np.save(npy_path, self.path_points)


class Planner(ABC):
    @abstractmethod
    def plan(self,
             input_data: PlannerInput,
             plotter: Optional[pv.Plotter] = None, **kwargs) -> PlannerOutput:
        """Method to plan the best path using PlannerInput."""