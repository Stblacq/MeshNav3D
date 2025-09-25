# mesh_nav_3D
This package provides a 3D mesh-based planning framework with multiple path planning algorithms and visualization tools.

## 📦 Installation

### Prerequisites
You need to install Python 3.12+ and the corresponding pip package manager. The installation process may vary depending on your operating system:

**For Ubuntu/Debian Linux:**
```bash
sudo apt update
sudo apt install python3-venv python3-pip
```

**For macOS:**
- Install Python 3.12+ from [python.org](https://python.org) or using Homebrew:
```bash
brew install python@3.12
```

**For Windows:**
- Download and install Python 3.12+ from [python.org](https://python.org)

### Install the Package
Ensure you have `pip >= 25.2`, then install the package:

```bash
pip install mesh_nav_3D
```

---

## 🚀 Quick Start

### Visualizing Planners
To visualize a single or multiple planners on a mesh, use the following code:

```python
from mesh_nav_3D import visualize_single_planner, visualize_multiple_planners, PlannerConfig

if __name__ == "__main__":
    # To visualize a single planner
    # visualize_single_planner('dijkstra_planner', PlannerConfig(mesh_file_path="terrain_mesh"))
    
    # To visualize multiple planners
    visualize_multiple_planners(
        ['AStarPlanner', 'HeatMethodPlanner'],
        PlannerConfig(mesh_file_path="terrain_mesh")
    )
```

Use the inbuilt `"terrain_mesh"` or a path to a valid mesh file.

---

### Benchmarking Multiple Planners
You can compare multiple planners across several scenarios using the following code:

```python
from mesh_nav_3D import compare_planners, PlannerConfig
import numpy as np

if __name__ == "__main__":
    scenarios = [
        (1, np.array([5.00, 9.55, 0.95]), np.array([6.82, 1.36, 0.10]), 8.42),
        (2, np.array([9.09, 9.55, -0.33]), np.array([7.73, 1.36, 0.20]), 8.31),
        (3, np.array([5.00, 9.55, 0.95]), np.array([0.91, 0.91, 0.48]), 9.57),
        (4, np.array([9.09, 10.00, -0.27]), np.array([2.73, 0.91, 0.25]), 11.11),
        (5, np.array([0.45, 9.55, -0.44]), np.array([8.18, 1.36, 0.19]), 11.27),
    ]
    
    for scenario in scenarios:
        print(f">>>>>>>>>>>>>>>>>>>>> Scenario {scenario[0]}")
        start_point = scenario[1]
        goal_point = scenario[2]
        planners = [
            "AStarPlanner",
            "DijkstraPlanner",
            "FlipOutPlanner",
            "FastMarchingPlanner",
            "GreedyBFSPlanner",
            "HeatMethodPlanner",
            "MMPPlanner",
            "ThetaStarPlanner"
        ]
        results = compare_planners(
            mesh_file_path="terrain_mesh",
            start_point=start_point,
            goal_point=goal_point,
            planners=planners,
            output_dir="metrics_outputs",
            save_results=True
        )
```

Results will be saved to the `metrics_outputs/` folder.

---

## 📝 Notes

- Replace mesh file paths with actual paths to your `.obj`, `.ply`, or supported mesh files.
- Ensure all dependencies are installed. If not bundled in the wheel, manually install them:

```bash
pip install pyvista>=0.44.2 tqdm>=4.67.1 pygeodesic>=0.1.11 potpourri3d>=1.2.1 pydantic>=2.11.2 psutil>=7.0.0
```

- **Planner Naming**: Planner names can be provided in the following formats:
  - **snake_case**: e.g., `"dijkstra_planner"`
  - **PascalCase**: e.g., `"AStarPlanner"`
  - **Class instances**: e.g., `AStarPlanner()` (direct class instance)
  
  All these formats will work and the system will handle the conversion automatically.

---

## 🤝 License
MIT.

## Citation
```
@article{Jerome-2025,
 author = {Jerome, Otobong and Bassey, Christian and Okeme, Christian and Maloletov, Alexander and Kulathunga,Geesara },
 doi = {10.5334/jors.573},
 journal = {Journal of Open Research Software},
 keyword = {en},
 month = {Sep},
 title = {MeshNav3D: Software for Visualizing and Benchmarking Uneven Terrain Planning Algorithms},
 year = {2025}
}```
