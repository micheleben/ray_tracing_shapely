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
PYTHON-SPECIFIC MODULE: Ray Data Export Utilities
===============================================================================
This module is a Python-specific addition and does NOT exist in the original
JavaScript Ray Optics Simulation codebase. It provides utilities for exporting
ray tracing results to various file formats:

- CSV: Tabular data with all ray properties
- JSON: Structured data for programmatic access (planned)

The export includes Python-specific TIR (Total Internal Reflection) tracking
data that is not available in the JavaScript version.
===============================================================================
"""

import csv
import math
import os
from pathlib import Path
from typing import List, Optional, Union

from ..core.ray import Ray


def save_rays_csv(
    ray_segments: List[Ray],
    output_path: Union[str, Path],
    filename: str = "rays.csv",
    precision_coords: int = 4,
    precision_brightness: int = 6,
) -> Path:
    """
    Export ray segment data to a CSV file.

    Exports all ray properties including geometry, brightness (both polarizations),
    wavelength, and TIR (Total Internal Reflection) tracking information.

    Args:
        ray_segments: List of Ray objects to export.
        output_path: Directory path where the CSV file will be saved.
            Can be a string or Path object.
        filename: Name of the output CSV file (default: "rays.csv").
        precision_coords: Decimal places for coordinate values (default: 4).
        precision_brightness: Decimal places for brightness values (default: 6).

    Returns:
        Path: Full path to the created CSV file.

    Raises:
        OSError: If the output directory cannot be created or file cannot be written.

    Example:
        >>> from ray_tracing_shapely.analysis import save_rays_csv
        >>> # After running a simulation that produces ray_segments
        >>> output_file = save_rays_csv(ray_segments, "./output")
        >>> print(f"Saved to: {output_file}")
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / filename

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header row with all ray properties
        writer.writerow([
            'ray_index',
            'p1_x',
            'p1_y',
            'p2_x',
            'p2_y',
            'brightness_s',
            'brightness_p',
            'brightness_total',
            'wavelength',
            'gap',
            'length',
            'is_tir_result',
            'caused_tir',
            'tir_count',
            # Grazing incidence flags (Python-specific)
            'is_grazing_result__angle',
            'caused_grazing__angle',
            'is_grazing_result__polar',
            'caused_grazing__polar',
            'is_grazing_result__transm',
            'caused_grazing__transm',
        ])

        coord_fmt = f"{{:.{precision_coords}f}}"
        brightness_fmt = f"{{:.{precision_brightness}f}}"

        for i, ray in enumerate(ray_segments):
            dx = ray.p2['x'] - ray.p1['x']
            dy = ray.p2['y'] - ray.p1['y']
            length = math.sqrt(dx * dx + dy * dy)

            writer.writerow([
                i,
                coord_fmt.format(ray.p1['x']),
                coord_fmt.format(ray.p1['y']),
                coord_fmt.format(ray.p2['x']),
                coord_fmt.format(ray.p2['y']),
                brightness_fmt.format(ray.brightness_s),
                brightness_fmt.format(ray.brightness_p),
                brightness_fmt.format(ray.total_brightness),
                ray.wavelength if ray.wavelength is not None else '',
                ray.gap,
                coord_fmt.format(length),
                ray.is_tir_result,
                ray.caused_tir,
                ray.tir_count,
                # Grazing incidence flags (Python-specific)
                getattr(ray, 'is_grazing_result__angle', False),
                getattr(ray, 'caused_grazing__angle', False),
                getattr(ray, 'is_grazing_result__polar', False),
                getattr(ray, 'caused_grazing__polar', False),
                getattr(ray, 'is_grazing_result__transm', False),
                getattr(ray, 'caused_grazing__transm', False),
            ])

    return csv_file


def filter_tir_rays(ray_segments: List[Ray], tir_only: bool = True) -> List[Ray]:
    """
    Filter rays based on TIR (Total Internal Reflection) status.

    Args:
        ray_segments: List of Ray objects to filter.
        tir_only: If True, return only rays that experienced TIR.
            If False, return rays that did NOT experience TIR.

    Returns:
        List[Ray]: Filtered list of rays.

    Example:
        >>> tir_rays = filter_tir_rays(ray_segments, tir_only=True)
        >>> non_tir_rays = filter_tir_rays(ray_segments, tir_only=False)
    """
    if tir_only:
        return [ray for ray in ray_segments if ray.is_tir_result or ray.tir_count > 0]
    else:
        return [ray for ray in ray_segments if not ray.is_tir_result and ray.tir_count == 0]


def filter_grazing_rays(
    ray_segments: List[Ray],
    grazing_only: bool = True,
    criterion: Optional[str] = None
) -> List[Ray]:
    """
    Filter rays based on grazing incidence status.

    Grazing incidence occurs at angles near the critical angle, where
    polarization effects become extreme. Three independent criteria are tracked:
    - 'angle': Incidence angle above threshold (e.g., 85°)
    - 'polar': Polarization ratio (p/s) above threshold
    - 'transm': Total transmission below threshold

    Args:
        ray_segments: List of Ray objects to filter.
        grazing_only: If True, return only rays that experienced grazing.
            If False, return rays that did NOT experience grazing.
        criterion: Optional specific criterion to filter by:
            - 'angle': Only angle criterion
            - 'polar': Only polarization criterion
            - 'transm': Only transmission criterion
            - None: Any criterion (default)

    Returns:
        List[Ray]: Filtered list of rays.

    Example:
        >>> # Get all grazing rays (any criterion)
        >>> grazing_rays = filter_grazing_rays(ray_segments)
        >>> # Get only rays that triggered the angle criterion
        >>> angle_grazing = filter_grazing_rays(ray_segments, criterion='angle')
        >>> # Get rays that did NOT experience grazing
        >>> non_grazing = filter_grazing_rays(ray_segments, grazing_only=False)
    """
    def is_grazing(ray: Ray) -> bool:
        """Check if ray experienced grazing based on criterion."""
        if criterion == 'angle':
            return getattr(ray, 'is_grazing_result__angle', False)
        elif criterion == 'polar':
            return getattr(ray, 'is_grazing_result__polar', False)
        elif criterion == 'transm':
            return getattr(ray, 'is_grazing_result__transm', False)
        else:
            # Any criterion
            return (
                getattr(ray, 'is_grazing_result__angle', False) or
                getattr(ray, 'is_grazing_result__polar', False) or
                getattr(ray, 'is_grazing_result__transm', False)
            )

    if grazing_only:
        return [ray for ray in ray_segments if is_grazing(ray)]
    else:
        return [ray for ray in ray_segments if not is_grazing(ray)]


def get_ray_statistics(ray_segments: List[Ray]) -> dict:
    """
    Compute statistics about a collection of ray segments.

    Args:
        ray_segments: List of Ray objects to analyze.

    Returns:
        dict: Dictionary containing:
            - total_rays: Total number of ray segments
            - tir_rays: Number of rays that experienced TIR
            - gap_rays: Number of gap (non-drawn) segments
            - total_brightness: Sum of all ray brightnesses
            - avg_brightness: Average brightness per ray
            - max_tir_count: Maximum TIR count in any ray lineage
            - wavelengths: Set of unique wavelengths (None for white light)
            - total_length: Sum of all ray segment lengths
            - grazing_rays_angle: Number of rays with grazing angle criterion
            - grazing_rays_polar: Number of rays with grazing polarization criterion
            - grazing_rays_transm: Number of rays with grazing transmission criterion
            - grazing_rays_any: Number of rays with any grazing criterion

    Example:
        >>> stats = get_ray_statistics(ray_segments)
        >>> print(f"Total rays: {stats['total_rays']}")
        >>> print(f"TIR events: {stats['tir_rays']}")
        >>> print(f"Grazing events: {stats['grazing_rays_any']}")
    """
    if not ray_segments:
        return {
            'total_rays': 0,
            'tir_rays': 0,
            'gap_rays': 0,
            'total_brightness': 0.0,
            'avg_brightness': 0.0,
            'max_tir_count': 0,
            'wavelengths': set(),
            'total_length': 0.0,
            'grazing_rays_angle': 0,
            'grazing_rays_polar': 0,
            'grazing_rays_transm': 0,
            'grazing_rays_any': 0,
        }

    tir_count = sum(1 for ray in ray_segments if ray.is_tir_result or ray.tir_count > 0)
    gap_count = sum(1 for ray in ray_segments if ray.gap)
    total_brightness = sum(ray.total_brightness for ray in ray_segments)
    max_tir = max(ray.tir_count for ray in ray_segments)
    wavelengths = {ray.wavelength for ray in ray_segments}

    # Grazing incidence statistics
    grazing_angle = sum(1 for ray in ray_segments if getattr(ray, 'is_grazing_result__angle', False))
    grazing_polar = sum(1 for ray in ray_segments if getattr(ray, 'is_grazing_result__polar', False))
    grazing_transm = sum(1 for ray in ray_segments if getattr(ray, 'is_grazing_result__transm', False))
    grazing_any = sum(1 for ray in ray_segments if (
        getattr(ray, 'is_grazing_result__angle', False) or
        getattr(ray, 'is_grazing_result__polar', False) or
        getattr(ray, 'is_grazing_result__transm', False)
    ))

    total_length = 0.0
    for ray in ray_segments:
        dx = ray.p2['x'] - ray.p1['x']
        dy = ray.p2['y'] - ray.p1['y']
        total_length += math.sqrt(dx * dx + dy * dy)

    return {
        'total_rays': len(ray_segments),
        'tir_rays': tir_count,
        'gap_rays': gap_count,
        'total_brightness': total_brightness,
        'avg_brightness': total_brightness / len(ray_segments),
        'max_tir_count': max_tir,
        'wavelengths': wavelengths,
        'total_length': total_length,
        'grazing_rays_angle': grazing_angle,
        'grazing_rays_polar': grazing_polar,
        'grazing_rays_transm': grazing_transm,
        'grazing_rays_any': grazing_any,
    }
