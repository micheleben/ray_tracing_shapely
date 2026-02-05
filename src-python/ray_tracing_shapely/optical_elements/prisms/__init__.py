"""
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
PRISMS SUB-MODULE
===============================================================================
Prism types with physics-driven constructors and dual-labeling architecture.

Classes:
- BasePrism: Base class for all prisms (dual labeling, geometry from parameters)
- EquilateralPrism: 60-60-60 dispersing prism
- RightAnglePrism: 45-90-45 TIR prism
- RefractometerPrism: Symmetric trapezoid for critical-angle measurements

Factory Functions:
- equilateral_prism(): Create an equilateral prism
- right_angle_prism(): Create a right-angle prism
- refractometer_prism(): Create a refractometer prism

Utility Modules:
- prism_utils: Deviation, dispersion, and geometry calculations
- tir_utils: Critical angle and refractometer design utilities
===============================================================================
"""

from typing import Tuple, TYPE_CHECKING

# Import classes
from .base_prism import BasePrism
from .equilateral import EquilateralPrism
from .right_angle import RightAnglePrism
from .refractometer import RefractometerPrism

# Import utility modules
from . import prism_utils
from . import tir_utils

if TYPE_CHECKING:
    from ....core.scene import Scene


# ============================================================================
# Factory Functions
# ============================================================================

def equilateral_prism(
    scene: 'Scene',
    side_length: float,
    position: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
    n: float = 1.5,
    anchor: str = 'centroid'
) -> EquilateralPrism:
    """
    Create an equilateral (60-60-60) dispersing prism.

    Args:
        scene: The scene this prism belongs to.
        side_length: Length of each side.
        position: Reference point coordinates.
        rotation: Rotation angle in degrees.
        n: Refractive index.
        anchor: Position reference ('centroid' or 'apex').

    Returns:
        A new EquilateralPrism instance.
    """
    return EquilateralPrism(
        scene=scene,
        side_length=side_length,
        position=position,
        rotation=rotation,
        n=n,
        anchor=anchor
    )


def right_angle_prism(
    scene: 'Scene',
    leg_length: float,
    position: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
    n: float = 1.5,
    anchor: str = 'centroid'
) -> RightAnglePrism:
    """
    Create a 45-90-45 right-angle prism for TIR applications.

    Args:
        scene: The scene this prism belongs to.
        leg_length: Length of each leg (the two equal sides).
        position: Reference point coordinates.
        rotation: Rotation angle in degrees.
        n: Refractive index.
        anchor: Position reference ('centroid' or 'apex').

    Returns:
        A new RightAnglePrism instance.
    """
    return RightAnglePrism(
        scene=scene,
        leg_length=leg_length,
        position=position,
        rotation=rotation,
        n=n,
        anchor=anchor
    )


def refractometer_prism(
    scene: 'Scene',
    n_prism: float,
    n_sample_range: Tuple[float, float],
    measuring_surface_length: float,
    system_type: str = 'angular',
    position: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0
) -> RefractometerPrism:
    """
    Create a refractometer prism from physics parameters.

    The face angle is computed automatically from the sample refractive
    index range to ensure normal exit at mid-range.

    Args:
        scene: The scene this prism belongs to.
        n_prism: Refractive index of the prism material.
        n_sample_range: Tuple of (n_min, n_max) for the sample.
        measuring_surface_length: Length of the measuring surface.
        system_type: 'angular' (Abbe-like) or 'geometric' (lensless).
        position: Reference point coordinates.
        rotation: Rotation angle in degrees.

    Returns:
        A new RefractometerPrism instance.
    """
    return RefractometerPrism(
        scene=scene,
        n_prism=n_prism,
        n_sample_range=n_sample_range,
        measuring_surface_length=measuring_surface_length,
        system_type=system_type,
        position=position,
        rotation=rotation
    )


__all__ = [
    # Base class
    'BasePrism',
    # Prism classes
    'EquilateralPrism',
    'RightAnglePrism',
    'RefractometerPrism',
    # Factory functions
    'equilateral_prism',
    'right_angle_prism',
    'refractometer_prism',
    # Utility modules
    'prism_utils',
    'tir_utils',
]
