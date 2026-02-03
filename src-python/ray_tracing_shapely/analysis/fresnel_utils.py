"""
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
PYTHON-SPECIFIC MODULE: Fresnel Equation Utilities
===============================================================================
Standalone functions that solve the Fresnel equations to compute expected
transmittances, reflectances, polarization ratios, and critical angles.

The physics is already implemented inside BaseGlass.refract() but is not
callable standalone.  These utilities let a designer or agent ask
"what should I expect?" without running a simulation.

All functions return values; no print() side effects.
===============================================================================
"""

from __future__ import annotations
import math
from typing import Dict


def fresnel_transmittances(
    n1: float, n2: float, theta_i_deg: float
) -> Dict[str, float]:
    """
    Compute Fresnel power transmittances and reflectances at an interface.

    Uses the standard Fresnel equations for a planar interface between
    two dielectric media.

    Args:
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmitting medium.
        theta_i_deg: Angle of incidence in degrees (from normal).

    Returns:
        Dict with keys:
        - 'T_s': s-polarization power transmittance
        - 'T_p': p-polarization power transmittance
        - 'R_s': s-polarization power reflectance
        - 'R_p': p-polarization power reflectance
        - 'ratio_Tp_Ts': T_p / T_s (or float('inf') if T_s ~ 0)
        - 'theta_t_deg': refraction angle in degrees

    Raises:
        ValueError: If the angle exceeds the critical angle (TIR).
    """
    theta_i = math.radians(theta_i_deg)
    cos_i = math.cos(theta_i)
    sin_i = math.sin(theta_i)

    # Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t)
    sin_t = (n1 / n2) * sin_i
    if abs(sin_t) > 1.0:
        raise ValueError(
            f"Total internal reflection: angle {theta_i_deg:.2f}° exceeds "
            f"critical angle {critical_angle(n1, n2):.2f}° for n1={n1}, n2={n2}."
        )

    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    theta_t_deg = math.degrees(math.asin(sin_t))

    # Fresnel reflection coefficients (intensity)
    R_s = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    R_p = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2

    T_s = 1.0 - R_s
    T_p = 1.0 - R_p

    if T_s > 1e-10:
        ratio = T_p / T_s
    else:
        ratio = float('inf')

    return {
        'T_s': T_s,
        'T_p': T_p,
        'R_s': R_s,
        'R_p': R_p,
        'ratio_Tp_Ts': ratio,
        'theta_t_deg': theta_t_deg,
    }


def critical_angle(n1: float, n2: float) -> float:
    """
    Compute the critical angle for total internal reflection.

    TIR occurs when light travels from a denser to a rarer medium
    (n1 > n2) at an angle exceeding this value.

    Args:
        n1: Refractive index of the denser medium (must be > n2).
        n2: Refractive index of the rarer medium.

    Returns:
        Critical angle in degrees.

    Raises:
        ValueError: If n1 <= n2 (no TIR possible).
    """
    if n1 <= n2:
        raise ValueError(
            f"No TIR possible: n1={n1} must be greater than n2={n2}."
        )
    return math.degrees(math.asin(n2 / n1))


def brewster_angle(n1: float, n2: float) -> float:
    """
    Compute Brewster's angle (where R_p = 0).

    At Brewster's angle, reflected light is fully s-polarized.
    This angle exists for any pair of dielectric media.

    Args:
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmitting medium.

    Returns:
        Brewster's angle in degrees.
    """
    return math.degrees(math.atan(n2 / n1))
