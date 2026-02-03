# Ray Tracing Shapely

A Python ray tracing library using Shapely for computational geometry operations.

The project has started as a translation of the original [Ray Optics Simulation](https://github.com/ricktu288/ray-optics) JavaScript project. Original work copyright 2024 The Ray Optics Simulation authors 
and contributors.

If you use this please give credits to the original authors ( see section Acknowledgments)

## Overview

This library provides a Python implementation for 2D ray tracing simulations, including:

- **Core simulation engine** - Ray tracing through optical elements (lenses, prisms, mirrors)
- **Optical elements** - Glass objects, ideal lenses, spherical lenses, blockers, detectors
- **Light sources** - Point sources, beams, single rays, angle sources
- **Analysis utilities** - Geometric analysis of glass interfaces and boundaries (Python-specific)
- **Visualization** - SVG rendering of ray diagrams (Python-specific)

## Installation

Library is not in its complete version (part of the original work is not translated). Contributions are welcomed!

still in test.pypi.org (version 0.1.0). Install via pip from TestPyPI:
```bash
pip install -i https://test.pypi.org/simple/ ray-tracing-shapely==0.1.0
```

Or install from source:
We are currently in early development stage, so it is recommended to install from source:
current version is 0.2.0
```bash
git clone https://github.com/micheleben/ray_tracing_shapely.git
cd ray_tracing_shapely
pip install -e .
```

## Quick Start

```python
from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs import Glass, SingleRay
from ray_tracing_shapely.core.simulator import Simulator

# Create a scene
scene = Scene()
scene.color_mode = 'linear'

# Add a glass prism
prism = Glass(scene)
prism.path = [
    {'x': 100, 'y': 50, 'arc': False},
    {'x': 200, 'y': 50, 'arc': False},
    {'x': 150, 'y': 150, 'arc': False}
]
prism.refIndex = 1.5
scene.add_object(prism)

# Add a light ray
ray = SingleRay(scene)
ray.p1 = {'x': 50, 'y': 100}
ray.p2 = {'x': 100, 'y': 100}
scene.add_object(ray)

# Run simulation
simulator = Simulator(scene, max_rays=1000)
segments = simulator.run()

print(f"Traced {len(segments)} ray segments")
```

## Analysis Utilities (Python-Specific)

The `analysis` module provides utilities not available in the JavaScript version.

### Geometry analysis

```python
from ray_tracing_shapely.analysis import analyze_scene_geometry, describe_edges

# Analyze glass interfaces and boundaries
analysis = analyze_scene_geometry(scene)
for interface in analysis.interfaces:
    print(f"Interface: n1={interface.n1}, n2={interface.n2}, length={interface.length}")

# Describe edges of a glass object (text or XML)
print(describe_edges(prism, format='text'))
```

### Ray-geometry queries

```python
from ray_tracing_shapely.analysis import (
    find_rays_inside_glass, find_rays_crossing_edge,
    find_rays_by_angle_to_edge, find_rays_by_polarization,
)

# Which rays are inside the prism? Which cross the exit face?
inside = find_rays_inside_glass(segments, prism)
crossing = find_rays_crossing_edge(segments, prism, 'SE')

# Filter by incidence angle or polarization state
grazing = find_rays_by_angle_to_edge(segments, prism, 'SE', min_angle=80, max_angle=90)
polarized = find_rays_by_polarization(segments, min_dop=0.5)
```

### Ray lineage tracking

Every ray segment carries a UUID and a parent link, forming a tree that
records how rays split at optical surfaces. After simulation, the lineage
tree supports post-hoc path analysis:

```python
result = simulator.run_with_result()
lineage = result.lineage

from ray_tracing_shapely.analysis.lineage_analysis import (
    rank_paths_by_energy, check_energy_conservation, detect_tir_traps,
)

# Rank optical paths by terminal brightness
ranked = rank_paths_by_energy(lineage)
for r in ranked[:3]:
    print(f"Energy={r['energy']:.4f}  path={' -> '.join(r['path_types'])}")

# Verify energy conservation at every branch point
check = check_energy_conservation(lineage)
print(f"Valid: {check['is_valid']}, checks: {check['total_checks']}")

# Detect rays trapped by repeated TIR
traps = detect_tir_traps(lineage, min_tir_count=2)
```

### Standalone Fresnel utilities

```python
from ray_tracing_shapely.analysis import fresnel_transmittances, critical_angle, brewster_angle

r = fresnel_transmittances(n1=1.0, n2=1.5, theta_i_deg=45)
print(f"T_s={r['T_s']:.4f}  T_p={r['T_p']:.4f}  theta_t={r['theta_t_deg']:.1f}°")

print(f"Critical angle (glass->air): {critical_angle(1.5, 1.0):.1f}°")
print(f"Brewster angle (air->glass): {brewster_angle(1.0, 1.5):.1f}°")
```

### Agentic tools (LLM tool-use APIs)

For use with Claude API, claudette, langchain, or similar frameworks where
tool inputs/outputs must be JSON-serializable:

```python
from ray_tracing_shapely.analysis import (
    set_context_from_result, get_agentic_tools,
)

# Register simulation context once
set_context_from_result(scene=scene, result=result)

# Get tool wrappers that accept strings and return XML
tools = get_agentic_tools()
# tools = [{'name': 'find_rays_inside_glass_xml', 'function': ..., 'description': ...}, ...]
```

### Tool discovery

```python
from ray_tracing_shapely.analysis import list_available_tools
print(list_available_tools())  # lists all analysis functions with signatures
```

## Project Structure

```
ray_tracing_shapely/
├── src-python/
│   └── ray_tracing_shapely/
│       ├── __init__.py
│       ├── analysis/              # Python-specific analysis utilities
│       │   ├── __init__.py
│       │   ├── glass_geometry.py  # Edge descriptions, interfaces, boundaries
│       │   ├── saving.py          # CSV/XML export, ray filtering, statistics
│       │   ├── simulation_result.py  # SimulationResult container
│       │   ├── ray_geometry_queries.py  # Spatial queries (rays vs. geometry)
│       │   ├── lineage_analysis.py    # Post-hoc ray tree analysis
│       │   ├── fresnel_utils.py       # Standalone Fresnel equation utilities
│       │   ├── agentic_tools.py       # JSON-serializable wrappers for LLM APIs
│       │   └── tool_registry.py       # Tool discovery registry
│       ├── core/                  # Core simulation engine
│       │   ├── scene.py
│       │   ├── simulator.py
│       │   ├── ray.py             # Ray with TIR, grazing, lineage tracking
│       │   ├── ray_lineage.py     # Parent-child tree tracker
│       │   ├── geometry/
│       │   └── scene_objs/
│       ├── developer_tests/       # Development tests
│       └── examples/              # Example simulations
├── doc/
│   └── roadmap/                   # Design documents and implementation notes
├── pyproject.toml
├── README.md
└── .gitignore
```

## Dependencies

- Python >= 3.8
- shapely >= 2.0.0
- sympy >= 1.12
- numpy >= 1.20.0
- svgwrite >= 1.4.0

## Python-Specific Features

This library includes several features not available in the original JavaScript implementation:

1. **TIR Tracking** - Each ray segment tracks whether it was produced by Total Internal Reflection, whether its endpoint caused TIR, and the cumulative TIR count in its lineage
2. **Grazing Incidence Tracking** - Three independent criteria detect near-critical-angle refraction: angle threshold, polarization ratio, and total transmission
3. **Source Tracking** - Rays carry the UUID and label of the light source that emitted them, enabling multi-source and multi-wavelength analysis
4. **Ray Lineage Tracking** - Every segment has a UUID and parent link. The `RayLineage` class reconstructs the full ray tree after simulation for path analysis, branching statistics, and energy conservation checks
5. **Geometry Analysis** - Shapely-based analysis of glass interfaces, boundaries, edge descriptions, and ray-geometry spatial queries
6. **Fresnel Utilities** - Standalone Fresnel equation solver for transmittances, reflectances, critical angle, and Brewster's angle
7. **Agentic Tools** - JSON-serializable wrappers for LLM tool-use APIs (Claude, claudette, langchain)
8. **Simulation Results** - `SimulationResult` dataclass captures segments, scene snapshot, lineage, and metadata for post-hoc analysis
9. **Export** - CSV and XML export of ray data with filtering by TIR, grazing, wavelength, and source

## License

Apache License 2.0

## Acknowledgments

Based on the [Ray Optics Simulation](https://github.com/ricktu288/ray-optics) JavaScript project.
