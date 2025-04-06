from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pyvista as pv


class Planner(ABC):
    @abstractmethod
    def plan(self, start_point: np.ndarray,
             goal_point: np.ndarray,
             plotter: Optional[pv.Plotter],
             mesh: pv.DataSet,
             color="blue",
             time_horizon: float = 10.0,
             max_iterations: int = 1000) -> Optional[dict]:
        """Method to plan best path"""
        pass
