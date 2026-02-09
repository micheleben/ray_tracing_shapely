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

import sqlite3
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

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
# Structured error response helpers
# =============================================================================

def _ok(data: Any) -> Dict[str, Any]:
    """Wrap a successful result in the standard agentic response envelope."""
    return {"status": "ok", "data": data}


def _error(message: str) -> Dict[str, str]:
    """Wrap an error message in the standard agentic response envelope."""
    return {"status": "error", "message": message}


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

    Must be called before any agentic tool is invoked. Also creates an
    in-memory SQLite database with precomputed spatial relationships
    for use by query_rays() and describe_schema().

    Args:
        scene: The Scene object (needed to resolve glass names via
            get_object_by_name).
        segments: The ray segments from the simulation.
        lineage: Optional RayLineage for lineage-based tools.
    """
    from .agentic_db import create_database

    _CONTEXT['scene'] = scene
    _CONTEXT['segments'] = segments
    _CONTEXT['lineage'] = lineage
    _CONTEXT['db'] = create_database(scene, segments)


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
    db = _CONTEXT.get('db')
    if db is not None:
        db.close()
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


def _require_db() -> sqlite3.Connection:
    """
    Return the SQLite connection from context, or raise RuntimeError.

    Returns:
        The sqlite3.Connection.

    Raises:
        RuntimeError: If set_context() has not been called or DB is missing.
    """
    if 'db' not in _CONTEXT:
        raise RuntimeError(_NO_CONTEXT_MSG)
    return _CONTEXT['db']


# =============================================================================
# SQL query tools (Phase 1)
# =============================================================================

def query_rays(sql: str) -> Dict[str, Any]:
    """
    Run a read-only SQL query against the simulation database.

    The database contains five tables:
    - rays: One row per ray segment with all properties
    - glass_objects: One row per glass object in the scene
    - edges: One row per edge of each glass object
    - ray_glass_membership: Join table (ray_uuid, glass_uuid) for rays inside glass
    - ray_edge_crossing: Join table for rays crossing glass edges

    Use describe_schema() to see full table definitions.

    Only SELECT statements are allowed. Results are limited to 200 rows
    by default -- use LIMIT or more specific WHERE conditions for large results.

    Args:
        sql: A SQL SELECT query string.

    Returns:
        Structured dict with status and query results, or error message.
        On success, data contains 'columns', 'rows', and 'row_count'.

    Example queries:
        "SELECT COUNT(*) as n, AVG(brightness_total) as avg_b FROM rays"
        "SELECT r.uuid, r.brightness_total FROM rays r
         JOIN ray_glass_membership m ON r.uuid = m.ray_uuid
         WHERE m.glass_name = 'Main Prism'"
        "SELECT DISTINCT glass_name FROM ray_glass_membership"
    """
    try:
        from .agentic_db import execute_query
        db = _require_db()
        result = execute_query(db, sql)
        return _ok(result)
    except ValueError as e:
        return _error(str(e))
    except sqlite3.OperationalError as e:
        return _error(f"SQL error: {e}")
    except RuntimeError as e:
        return _error(str(e))


def describe_schema() -> Dict[str, Any]:
    """
    Return the schema of the simulation database.

    Lists all tables, their columns (with types and descriptions),
    and current row counts. Use this to discover what data is available
    before writing SQL queries with query_rays().

    Returns:
        Structured dict with table definitions and row counts.
    """
    try:
        from .agentic_db import get_schema_description
        db = _require_db()
        schema = get_schema_description(db)
        return _ok({"tables": schema})
    except RuntimeError as e:
        return _error(str(e))


# =============================================================================
# Legacy XML tool wrappers (superseded by query_rays, kept for backward compat)
# =============================================================================

def find_rays_inside_glass_xml(glass_name: str) -> Dict[str, Any]:
    """
    Find rays whose midpoint is inside a named glass object.

    Args:
        glass_name: Name of the glass object in the scene.

    Returns:
        Structured dict with status and XML data, or error message.
    """
    try:
        scene, segments = _require_context()
        glass = get_object_by_name(scene, glass_name)
        rays = find_rays_inside_glass(segments, glass)
        return _ok(rays_to_xml(rays))
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def find_rays_crossing_edge_xml(glass_name: str, edge_label: str) -> Dict[str, Any]:
    """
    Find rays that cross a specific edge of a named glass object.

    Args:
        glass_name: Name of the glass object in the scene.
        edge_label: Label of the edge (short_label, long_label, or index).

    Returns:
        Structured dict with status and XML data, or error message.
    """
    try:
        scene, segments = _require_context()
        glass = get_object_by_name(scene, glass_name)
        rays = find_rays_crossing_edge(segments, glass, edge_label)
        return _ok(rays_to_xml(rays))
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def find_rays_by_angle_to_edge_xml(
    glass_name: str,
    edge_label: str,
    min_angle: float = 0.0,
    max_angle: float = 90.0,
) -> Dict[str, Any]:
    """
    Find rays within an angle range relative to a named glass edge.

    Args:
        glass_name: Name of the glass object in the scene.
        edge_label: Label of the reference edge.
        min_angle: Minimum angle from edge normal in degrees (default: 0).
        max_angle: Maximum angle from edge normal in degrees (default: 90).

    Returns:
        Structured dict with status and XML data, or error message.
    """
    try:
        scene, segments = _require_context()
        glass = get_object_by_name(scene, glass_name)
        rays = find_rays_by_angle_to_edge(segments, glass, edge_label,
                                           min_angle=min_angle,
                                           max_angle=max_angle)
        return _ok(rays_to_xml(rays))
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def find_rays_by_polarization_xml(
    min_dop: float = 0.0,
    max_dop: float = 1.0,
) -> Dict[str, Any]:
    """
    Filter rays by degree of polarization.

    Args:
        min_dop: Minimum degree of polarization (0-1, default: 0).
        max_dop: Maximum degree of polarization (0-1, default: 1).

    Returns:
        Structured dict with status and XML data, or error message.
    """
    try:
        _, segments = _require_context()
        rays = find_rays_by_polarization(segments, min_dop=min_dop, max_dop=max_dop)
        return _ok(rays_to_xml(rays))
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


# =============================================================================
# Viewbox helpers
# =============================================================================

_VIEWBOX_PADDING_FACTOR = 0.05  # 5% padding on each side


def _compute_scene_bounds(scene: 'Scene') -> Tuple[float, float, float, float]:
    """
    Compute a bounding box (min_x, min_y, width, height) from all scene objects.

    Iterates over scene.objs and collects coordinates from paths, p1/p2
    endpoints, and point-source x/y attributes.

    Returns:
        (min_x, min_y, width, height) in Y-up coordinates with 5% padding.
    """
    xs: List[float] = []
    ys: List[float] = []

    for obj in scene.objs:
        if hasattr(obj, 'path') and hasattr(obj.path, '__len__'):
            for pt in obj.path:
                xs.append(pt['x'])
                ys.append(pt['y'])
        if hasattr(obj, 'p1'):
            xs.append(obj.p1['x'])
            ys.append(obj.p1['y'])
        if hasattr(obj, 'p2'):
            xs.append(obj.p2['x'])
            ys.append(obj.p2['y'])
        if hasattr(obj, 'x') and hasattr(obj, 'y'):
            xs.append(obj.x)
            ys.append(obj.y)

    if not xs:
        return (0, 0, 800, 600)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max_x - min_x
    h = max_y - min_y
    # Avoid zero-size viewbox
    if w < 1e-6:
        w = 100
    if h < 1e-6:
        h = 100

    pad_x = w * _VIEWBOX_PADDING_FACTOR
    pad_y = h * _VIEWBOX_PADDING_FACTOR
    return (min_x - pad_x, min_y - pad_y, w + 2 * pad_x, h + 2 * pad_y)


def _parse_viewbox(
    viewbox_str: str,
    scene: 'Scene',
) -> Tuple[float, float, float, float]:
    """
    Parse a viewbox string into a (min_x, min_y, width, height) tuple.

    Accepts either ``"auto"`` (computes bounds from the scene) or a
    comma-separated string like ``"0,0,400,300"``.

    Args:
        viewbox_str: ``"auto"`` or ``"min_x,min_y,width,height"``.
        scene: The Scene object (used when viewbox_str is ``"auto"``).

    Returns:
        Tuple of (min_x, min_y, width, height).
    """
    if viewbox_str == 'auto':
        return _compute_scene_bounds(scene)
    parts = [float(v.strip()) for v in viewbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError(
            f"viewbox must be 'auto' or 'min_x,min_y,w,h', got: {viewbox_str!r}"
        )
    return tuple(parts)


# =============================================================================
# SVG rendering tools (Phase 7)
# =============================================================================

def render_scene_svg(
    width: int = 800,
    height: int = 600,
    viewbox: str = 'auto',
) -> Dict[str, Any]:
    """
    Render the full scene and all rays as an SVG string.

    This is the baseline rendering — no highlights, just the scene as-is.
    Useful as a "show me the current state" tool.

    Args:
        width: SVG width in pixels.
        height: SVG height in pixels.
        viewbox: Viewbox as "min_x,min_y,width,height" or "auto" to compute
            from scene bounds.

    Returns:
        Structured dict with status and SVG data, or error message.
    """
    try:
        from ..core.svg_renderer import SVGRenderer

        scene, segments = _require_context()
        vb = _parse_viewbox(viewbox, scene)
        renderer = SVGRenderer(width=width, height=height, viewbox=vb)
        renderer.draw_scene(scene, segments)
        return _ok(renderer.to_string())
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def highlight_rays_inside_glass_svg(
    glass_name: str,
    highlight_color: str = 'yellow',
    width: int = 800,
    height: int = 600,
    viewbox: str = 'auto',
) -> Dict[str, Any]:
    """
    Render the scene with rays inside a glass object highlighted.

    Combines find_rays_inside_glass() with highlight rendering.

    Args:
        glass_name: Name of the glass object.
        highlight_color: CSS color for highlighted rays.
        width: SVG width in pixels.
        height: SVG height in pixels.
        viewbox: Viewbox as "min_x,min_y,width,height" or "auto".

    Returns:
        Structured dict with status and SVG data, or error message.
    """
    try:
        from ..core.svg_renderer import SVGRenderer

        scene, segments = _require_context()
        glass = get_object_by_name(scene, glass_name)
        rays = find_rays_inside_glass(segments, glass)
        ray_uuids = {r.uuid for r in rays}

        vb = _parse_viewbox(viewbox, scene)
        renderer = SVGRenderer(width=width, height=height, viewbox=vb)
        renderer.draw_scene_with_highlights(
            scene, segments,
            highlight_ray_uuids=ray_uuids,
            highlight_glass_names=[glass_name],
            highlight_color=highlight_color,
        )
        return _ok(renderer.to_string())
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def highlight_rays_crossing_edge_svg(
    glass_name: str,
    edge_label: str,
    highlight_color: str = 'yellow',
    width: int = 800,
    height: int = 600,
    viewbox: str = 'auto',
) -> Dict[str, Any]:
    """
    Render the scene with rays crossing a specific edge highlighted.

    Combines find_rays_crossing_edge() with highlight rendering.
    Also highlights the edge itself.

    Args:
        glass_name: Name of the glass object.
        edge_label: Label of the edge (short_label, long_label, or index).
        highlight_color: CSS color for highlighted rays.
        width: SVG width in pixels.
        height: SVG height in pixels.
        viewbox: Viewbox as "min_x,min_y,width,height" or "auto".

    Returns:
        Structured dict with status and SVG data, or error message.
    """
    try:
        from ..core.svg_renderer import SVGRenderer

        scene, segments = _require_context()
        glass = get_object_by_name(scene, glass_name)
        rays = find_rays_crossing_edge(segments, glass, edge_label)
        ray_uuids = {r.uuid for r in rays}

        vb = _parse_viewbox(viewbox, scene)
        renderer = SVGRenderer(width=width, height=height, viewbox=vb)
        renderer.draw_scene_with_highlights(
            scene, segments,
            highlight_ray_uuids=ray_uuids,
            highlight_edge_specs=[(glass_name, edge_label)],
            highlight_color=highlight_color,
        )
        return _ok(renderer.to_string())
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def highlight_rays_by_polarization_svg(
    min_dop: float = 0.0,
    max_dop: float = 1.0,
    highlight_color: str = 'magenta',
    width: int = 800,
    height: int = 600,
    viewbox: str = 'auto',
) -> Dict[str, Any]:
    """
    Render the scene with rays filtered by degree of polarization highlighted.

    Args:
        min_dop: Minimum degree of polarization (0-1).
        max_dop: Maximum degree of polarization (0-1).
        highlight_color: CSS color for highlighted rays.
        width: SVG width in pixels.
        height: SVG height in pixels.
        viewbox: Viewbox as "min_x,min_y,width,height" or "auto".

    Returns:
        Structured dict with status and SVG data, or error message.
    """
    try:
        from ..core.svg_renderer import SVGRenderer

        scene, segments = _require_context()
        rays = find_rays_by_polarization(segments, min_dop=min_dop, max_dop=max_dop)
        ray_uuids = {r.uuid for r in rays}

        vb = _parse_viewbox(viewbox, scene)
        renderer = SVGRenderer(width=width, height=height, viewbox=vb)
        renderer.draw_scene_with_highlights(
            scene, segments,
            highlight_ray_uuids=ray_uuids,
            highlight_color=highlight_color,
        )
        return _ok(renderer.to_string())
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))


def highlight_custom_rays_svg(
    ray_uuids_csv: str,
    highlight_color: str = 'yellow',
    width: int = 800,
    height: int = 600,
    viewbox: str = 'auto',
) -> Dict[str, Any]:
    """
    Render the scene with a specific set of rays highlighted by uuid.

    This is the escape hatch: the agent can compose arbitrary queries,
    collect uuids, and pass them here for visualization.

    Args:
        ray_uuids_csv: Comma-separated ray uuids to highlight.
        highlight_color: CSS color for highlighted rays.
        width: SVG width in pixels.
        height: SVG height in pixels.
        viewbox: Viewbox as "min_x,min_y,width,height" or "auto".

    Returns:
        Structured dict with status and SVG data, or error message.
    """
    try:
        from ..core.svg_renderer import SVGRenderer

        scene, segments = _require_context()
        ray_uuids = set(u.strip() for u in ray_uuids_csv.split(',') if u.strip())

        vb = _parse_viewbox(viewbox, scene)
        renderer = SVGRenderer(width=width, height=height, viewbox=vb)
        renderer.draw_scene_with_highlights(
            scene, segments,
            highlight_ray_uuids=ray_uuids,
            highlight_color=highlight_color,
        )
        return _ok(renderer.to_string())
    except (ValueError, KeyError) as e:
        return _error(str(e))
    except RuntimeError as e:
        return _error(str(e))
