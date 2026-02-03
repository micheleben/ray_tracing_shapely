"""
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
PYTHON-SPECIFIC MODULE: Agentic Tool Wrappers (JSON-Serializable)
===============================================================================
String-in / string-out wrappers around the analysis tools, designed for use
with LLM tool-use APIs (Claude API, claudette, lisette, langchain, etc.)
where inputs and outputs must be JSON-serializable.

Each wrapper:
- Accepts only JSON-serializable parameters (strings, numbers)
- Resolves live Python objects from a pre-registered module-level context
- Delegates to the original analysis function
- Returns results as XML strings via rays_to_xml()

Usage:
    # After running a simulation:
    from ray_tracing_shapely.analysis.agentic_tools import (
        set_context_from_result,
        find_rays_inside_glass_xml,
    )
    set_context_from_result(scene=scene, result=result)

    # Pass tools to an LLM framework:
    chat = Chat(model, tools=[find_rays_inside_glass_xml])
===============================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from .ray_geometry_queries import (
    get_object_by_name,
    find_rays_inside_glass,
    find_rays_crossing_edge,
    find_rays_by_angle_to_edge,
    find_rays_by_polarization,
)
from .saving import rays_to_xml

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.ray import Ray
    from ..core.ray_lineage import RayLineage
    from .simulation_result import SimulationResult


# =============================================================================
# Context management
# =============================================================================

_CONTEXT: Dict[str, Any] = {}

_NO_CONTEXT_MSG = (
    "No context set. Call set_context() or set_context_from_result() "
    "before using agentic tools."
)


def set_context(
    scene: 'Scene',
    segments: List['Ray'],
    lineage: Optional['RayLineage'] = None,
) -> None:
    """
    Register simulation objects for use by agentic tool wrappers.

    Must be called before any agentic tool is invoked.

    Args:
        scene: The Scene object (needed to resolve glass names via
            get_object_by_name).
        segments: The ray segments from the simulation.
        lineage: Optional RayLineage for lineage-based tools.
    """
    _CONTEXT['scene'] = scene
    _CONTEXT['segments'] = segments
    _CONTEXT['lineage'] = lineage


def set_context_from_result(
    scene: 'Scene',
    result: 'SimulationResult',
) -> None:
    """
    Convenience: populate context from a SimulationResult.

    Note: SimulationResult stores a SceneSnapshot (flat summary), not
    the live Scene.  The caller must pass the original Scene separately.

    Args:
        scene: The live Scene object used in the simulation.
        result: The SimulationResult returned by run_with_result().
    """
    set_context(
        scene=scene,
        segments=result.segments,
        lineage=result.lineage,
    )


def get_context() -> Dict[str, Any]:
    """Return a copy of the current context dict (for inspection/debugging)."""
    return dict(_CONTEXT)


def clear_context() -> None:
    """Clear the context (e.g. between simulation runs)."""
    _CONTEXT.clear()


def _require_context() -> tuple:
    """
    Return (scene, segments) from context, or raise RuntimeError.

    Returns:
        (scene, segments) tuple.

    Raises:
        RuntimeError: If set_context() has not been called.
    """
    if 'scene' not in _CONTEXT or 'segments' not in _CONTEXT:
        raise RuntimeError(_NO_CONTEXT_MSG)
    return _CONTEXT['scene'], _CONTEXT['segments']


# =============================================================================
# String-based tool wrappers
# =============================================================================

def find_rays_inside_glass_xml(glass_name: str) -> str:
    """
    Find rays whose midpoint is inside a named glass object.

    Args:
        glass_name: Name of the glass object in the scene.

    Returns:
        XML string describing the matching rays.

    Raises:
        RuntimeError: If set_context() has not been called.
        ValueError: If glass_name doesn't match any object.
    """
    scene, segments = _require_context()
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_inside_glass(segments, glass)
    return rays_to_xml(rays)


def find_rays_crossing_edge_xml(glass_name: str, edge_label: str) -> str:
    """
    Find rays that cross a specific edge of a named glass object.

    Args:
        glass_name: Name of the glass object in the scene.
        edge_label: Label of the edge (short_label, long_label, or index).

    Returns:
        XML string describing the matching rays.

    Raises:
        RuntimeError: If set_context() has not been called.
        ValueError: If glass_name or edge_label doesn't match.
    """
    scene, segments = _require_context()
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_crossing_edge(segments, glass, edge_label)
    return rays_to_xml(rays)


def find_rays_by_angle_to_edge_xml(
    glass_name: str,
    edge_label: str,
    min_angle: float = 0.0,
    max_angle: float = 90.0,
) -> str:
    """
    Find rays within an angle range relative to a named glass edge.

    Args:
        glass_name: Name of the glass object in the scene.
        edge_label: Label of the reference edge.
        min_angle: Minimum angle from edge normal in degrees (default: 0).
        max_angle: Maximum angle from edge normal in degrees (default: 90).

    Returns:
        XML string describing the matching rays.

    Raises:
        RuntimeError: If set_context() has not been called.
        ValueError: If glass_name or edge_label doesn't match.
    """
    scene, segments = _require_context()
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_by_angle_to_edge(segments, glass, edge_label,
                                       min_angle=min_angle,
                                       max_angle=max_angle)
    return rays_to_xml(rays)


def find_rays_by_polarization_xml(
    min_dop: float = 0.0,
    max_dop: float = 1.0,
) -> str:
    """
    Filter rays by degree of polarization.

    Args:
        min_dop: Minimum degree of polarization (0-1, default: 0).
        max_dop: Maximum degree of polarization (0-1, default: 1).

    Returns:
        XML string describing the matching rays.

    Raises:
        RuntimeError: If set_context() has not been called.
    """
    _, segments = _require_context()
    rays = find_rays_by_polarization(segments, min_dop=min_dop, max_dop=max_dop)
    return rays_to_xml(rays)
