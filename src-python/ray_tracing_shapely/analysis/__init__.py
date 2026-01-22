"""
Original work Copyright 2024 The Ray Optics Simulation authors and contributors
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

===============================================================================
PYTHON-SPECIFIC MODULE: Analysis Utilities
===============================================================================
This module is a Python-specific addition and does NOT exist in the original
JavaScript Ray Optics Simulation codebase. It provides analysis utilities for
extracting geometric information from scenes, such as:

- Glass boundary properties (area, centroid, perimeter)
- Interface detection between adjacent glass objects
- Interface properties (length, center, normal vectors)

These utilities leverage the Shapely library for computational geometry
operations that are not available in the JavaScript version.
===============================================================================
"""


from .glass_geometry import (
    GlassInterface,
    GlassBoundary,
    SceneGeometryAnalysis,
    analyze_scene_geometry,
    glass_to_polygon,
    # Edge description (Python-specific)
    EdgeType,
    EdgeDescription,
    get_edge_descriptions,
    describe_edges,
)
from .saving import (
    save_rays_csv,
    filter_tir_rays,
    get_ray_statistics,
)
from .simulation_result import (
    SceneSnapshot,
    SimulationResult,
    describe_simulation_result,
)

__all__ = [
    # Glass geometry analysis
    'GlassInterface',
    'GlassBoundary',
    'SceneGeometryAnalysis',
    'analyze_scene_geometry',
    'glass_to_polygon',
    # Edge description (Python-specific)
    'EdgeType',
    'EdgeDescription',
    'get_edge_descriptions',
    'describe_edges',
    # Ray data export and statistics
    'save_rays_csv',
    'filter_tir_rays',
    'get_ray_statistics',
    # Simulation result container (Python-specific)
    'SceneSnapshot',
    'SimulationResult',
    'describe_simulation_result',
]

