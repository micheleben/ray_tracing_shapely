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

```bash
pip install -i https://test.pypi.org/simple/ ray-tracing-shapely==0.1.0
```

Or install from source:

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

The `analysis` module provides utilities for geometric analysis not available in the JavaScript version:

```python
from ray_tracing_shapely.analysis import analyze_scene_geometry

# Analyze glass interfaces
analysis = analyze_scene_geometry(scene)

for interface in analysis.interfaces:
    print(f"Interface: n1={interface.n1}, n2={interface.n2}")
    print(f"  Length: {interface.length}")
    print(f"  Center: {interface.center}")
    print(f"  Normal: {interface.normal_at(0.5)}")

for boundary in analysis.boundaries:
    print(f"Glass: n={boundary.n}")
    print(f"  Area: {boundary.area}")
    print(f"  Centroid: {boundary.centroid}")
```

## Project Structure

```
ray_tracing_shapely/
├── src-python/
│   └── ray_tracing_shapely/
│       ├── __init__.py
│       ├── analysis/          # Python-specific analysis utilities
│       │   ├── __init__.py
│       │   └── glass_geometry.py
│       ├── core/              # Core simulation engine
│       │   ├── __init__.py
│       │   ├── scene.py
│       │   ├── simulator.py
│       │   ├── ray.py
│       │   ├── geometry/
│       │   └── scene_objs/
│       ├── developer_tests/   # Development tests
│       └── examples/          # Example simulations
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

1. **TIR Tracking** - Track Total Internal Reflection events in ray segments
2. **Brightness Threshold Control** - Explicit control over minimum brightness threshold
3. **Geometry Analysis** - Shapely-based analysis of glass interfaces and boundaries

## License

Apache License 2.0

## Acknowledgments

Based on the [Ray Optics Simulation](https://github.com/ricktu288/ray-optics) JavaScript project.
