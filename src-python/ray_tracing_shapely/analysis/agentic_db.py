"""
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
PYTHON-SPECIFIC MODULE: Agentic SQLite Database (Phase 1)
===============================================================================
In-memory SQLite database for agentic ray-tracing queries.

Creates and populates five tables from simulation data at set_context() time:
- rays:                 One row per ray segment with all properties
- glass_objects:        One row per glass object in the scene
- edges:                One row per edge of each glass object
- ray_glass_membership: Precomputed join (which rays are inside which glass)
- ray_edge_crossing:    Precomputed join (which rays cross which edge)

The LLM agent queries this database via SQL through the query_rays() tool,
which enforces SELECT-only access.
===============================================================================
"""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from shapely.geometry import Point, LineString

from .glass_geometry import glass_to_polygon, get_edge_descriptions
from .ray_geometry_queries import _ray_endpoints

if TYPE_CHECKING:
    from ..core.ray import Ray
    from ..core.scene import Scene


# =============================================================================
# Constants
# =============================================================================

MAX_QUERY_ROWS = 200

_FORBIDDEN_KEYWORDS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|'
    r'PRAGMA|REPLACE|VACUUM|REINDEX)\b',
    re.IGNORECASE,
)


# =============================================================================
# Schema DDL
# =============================================================================

_SCHEMA_RAYS = """\
CREATE TABLE rays (
    uuid            TEXT PRIMARY KEY,
    parent_uuid     TEXT,
    interaction_type TEXT NOT NULL,
    p1_x            REAL NOT NULL,
    p1_y            REAL NOT NULL,
    p2_x            REAL NOT NULL,
    p2_y            REAL NOT NULL,
    midpoint_x      REAL NOT NULL,
    midpoint_y      REAL NOT NULL,
    length          REAL NOT NULL,
    brightness_s    REAL NOT NULL,
    brightness_p    REAL NOT NULL,
    brightness_total REAL NOT NULL,
    degree_of_polarization REAL NOT NULL,
    polarization_ratio REAL,
    wavelength      REAL,
    gap             INTEGER NOT NULL,
    is_new          INTEGER NOT NULL,
    is_tir_result   INTEGER NOT NULL,
    caused_tir      INTEGER NOT NULL,
    tir_count       INTEGER NOT NULL,
    is_grazing_result__angle  INTEGER NOT NULL,
    caused_grazing__angle     INTEGER NOT NULL,
    is_grazing_result__polar  INTEGER NOT NULL,
    caused_grazing__polar     INTEGER NOT NULL,
    is_grazing_result__transm INTEGER NOT NULL,
    caused_grazing__transm    INTEGER NOT NULL,
    source_uuid     TEXT,
    source_label    TEXT
);
"""

_SCHEMA_GLASS_OBJECTS = """\
CREATE TABLE glass_objects (
    uuid            TEXT PRIMARY KEY,
    name            TEXT,
    display_name    TEXT NOT NULL,
    glass_type      TEXT NOT NULL,
    ref_index       REAL NOT NULL,
    edge_count      INTEGER NOT NULL,
    centroid_x      REAL,
    centroid_y      REAL,
    area            REAL,
    perimeter       REAL
);
"""

_SCHEMA_EDGES = """\
CREATE TABLE edges (
    glass_uuid      TEXT NOT NULL,
    glass_name      TEXT,
    edge_index      INTEGER NOT NULL,
    short_label     TEXT NOT NULL,
    long_label      TEXT NOT NULL,
    edge_type       TEXT NOT NULL,
    p1_x            REAL NOT NULL,
    p1_y            REAL NOT NULL,
    p2_x            REAL NOT NULL,
    p2_y            REAL NOT NULL,
    midpoint_x      REAL NOT NULL,
    midpoint_y      REAL NOT NULL,
    length          REAL NOT NULL,
    PRIMARY KEY (glass_uuid, edge_index)
);
"""

_SCHEMA_RAY_GLASS_MEMBERSHIP = """\
CREATE TABLE ray_glass_membership (
    ray_uuid        TEXT NOT NULL,
    glass_uuid      TEXT NOT NULL,
    glass_name      TEXT,
    PRIMARY KEY (ray_uuid, glass_uuid)
);
"""

_SCHEMA_RAY_EDGE_CROSSING = """\
CREATE TABLE ray_edge_crossing (
    ray_uuid        TEXT NOT NULL,
    glass_uuid      TEXT NOT NULL,
    glass_name      TEXT,
    edge_index      INTEGER NOT NULL,
    edge_short_label TEXT NOT NULL,
    PRIMARY KEY (ray_uuid, glass_uuid, edge_index)
);
"""


# =============================================================================
# Column metadata (for describe_schema)
# =============================================================================

_COLUMN_DESCRIPTIONS: Dict[str, List[Dict[str, str]]] = {
    "rays": [
        {"name": "uuid", "type": "TEXT", "description": "Unique ray segment identifier"},
        {"name": "parent_uuid", "type": "TEXT", "description": "UUID of parent ray (NULL for source rays)"},
        {"name": "interaction_type", "type": "TEXT", "description": "How ray was created: 'source', 'reflect', 'refract', 'tir'"},
        {"name": "p1_x", "type": "REAL", "description": "Start point X coordinate"},
        {"name": "p1_y", "type": "REAL", "description": "Start point Y coordinate"},
        {"name": "p2_x", "type": "REAL", "description": "End point X coordinate"},
        {"name": "p2_y", "type": "REAL", "description": "End point Y coordinate"},
        {"name": "midpoint_x", "type": "REAL", "description": "Midpoint X coordinate"},
        {"name": "midpoint_y", "type": "REAL", "description": "Midpoint Y coordinate"},
        {"name": "length", "type": "REAL", "description": "Segment length in scene units"},
        {"name": "brightness_s", "type": "REAL", "description": "S-polarization brightness (0 to 1)"},
        {"name": "brightness_p", "type": "REAL", "description": "P-polarization brightness (0 to 1)"},
        {"name": "brightness_total", "type": "REAL", "description": "Total brightness (s + p)"},
        {"name": "degree_of_polarization", "type": "REAL", "description": "Degree of polarization (0=unpolarized, 1=fully polarized)"},
        {"name": "polarization_ratio", "type": "REAL", "description": "Ratio brightness_p / brightness_s (NULL if s near zero)"},
        {"name": "wavelength", "type": "REAL", "description": "Wavelength in nm (NULL for white light)"},
        {"name": "gap", "type": "INTEGER", "description": "1 if gap segment (not drawn), 0 otherwise"},
        {"name": "is_new", "type": "INTEGER", "description": "1 if not yet processed, 0 otherwise"},
        {"name": "is_tir_result", "type": "INTEGER", "description": "1 if produced by total internal reflection"},
        {"name": "caused_tir", "type": "INTEGER", "description": "1 if endpoint caused TIR"},
        {"name": "tir_count", "type": "INTEGER", "description": "Cumulative TIR count in lineage"},
        {"name": "is_grazing_result__angle", "type": "INTEGER", "description": "1 if produced by grazing incidence (angle criterion)"},
        {"name": "caused_grazing__angle", "type": "INTEGER", "description": "1 if endpoint triggered angle grazing criterion"},
        {"name": "is_grazing_result__polar", "type": "INTEGER", "description": "1 if produced by grazing incidence (polarization criterion)"},
        {"name": "caused_grazing__polar", "type": "INTEGER", "description": "1 if endpoint triggered polarization grazing criterion"},
        {"name": "is_grazing_result__transm", "type": "INTEGER", "description": "1 if produced by grazing incidence (transmission criterion)"},
        {"name": "caused_grazing__transm", "type": "INTEGER", "description": "1 if endpoint triggered transmission grazing criterion"},
        {"name": "source_uuid", "type": "TEXT", "description": "UUID of the light source that emitted this ray"},
        {"name": "source_label", "type": "TEXT", "description": "Human-readable source label (e.g. 'red', 'chief')"},
    ],
    "glass_objects": [
        {"name": "uuid", "type": "TEXT", "description": "Unique glass object identifier"},
        {"name": "name", "type": "TEXT", "description": "User-defined name (may be NULL)"},
        {"name": "display_name", "type": "TEXT", "description": "Display name (name or type + short UUID fallback)"},
        {"name": "glass_type", "type": "TEXT", "description": "Object type (e.g. 'Glass', 'SphericalLens')"},
        {"name": "ref_index", "type": "REAL", "description": "Refractive index"},
        {"name": "edge_count", "type": "INTEGER", "description": "Number of edges"},
        {"name": "centroid_x", "type": "REAL", "description": "Centroid X coordinate"},
        {"name": "centroid_y", "type": "REAL", "description": "Centroid Y coordinate"},
        {"name": "area", "type": "REAL", "description": "Area in scene units squared"},
        {"name": "perimeter", "type": "REAL", "description": "Perimeter in scene units"},
    ],
    "edges": [
        {"name": "glass_uuid", "type": "TEXT", "description": "UUID of the owning glass object"},
        {"name": "glass_name", "type": "TEXT", "description": "Name of the glass (denormalized)"},
        {"name": "edge_index", "type": "INTEGER", "description": "Edge index (0-based)"},
        {"name": "short_label", "type": "TEXT", "description": "Short edge label (e.g. 'S', 'NE', '0')"},
        {"name": "long_label", "type": "TEXT", "description": "Long edge label (e.g. 'South Edge')"},
        {"name": "edge_type", "type": "TEXT", "description": "Geometry type: 'line', 'circular', 'equation'"},
        {"name": "p1_x", "type": "REAL", "description": "Edge start point X"},
        {"name": "p1_y", "type": "REAL", "description": "Edge start point Y"},
        {"name": "p2_x", "type": "REAL", "description": "Edge end point X"},
        {"name": "p2_y", "type": "REAL", "description": "Edge end point Y"},
        {"name": "midpoint_x", "type": "REAL", "description": "Edge midpoint X"},
        {"name": "midpoint_y", "type": "REAL", "description": "Edge midpoint Y"},
        {"name": "length", "type": "REAL", "description": "Edge length in scene units"},
    ],
    "ray_glass_membership": [
        {"name": "ray_uuid", "type": "TEXT", "description": "Ray segment UUID"},
        {"name": "glass_uuid", "type": "TEXT", "description": "Glass object UUID"},
        {"name": "glass_name", "type": "TEXT", "description": "Glass name (denormalized)"},
    ],
    "ray_edge_crossing": [
        {"name": "ray_uuid", "type": "TEXT", "description": "Ray segment UUID"},
        {"name": "glass_uuid", "type": "TEXT", "description": "Glass object UUID"},
        {"name": "glass_name", "type": "TEXT", "description": "Glass name (denormalized)"},
        {"name": "edge_index", "type": "INTEGER", "description": "Index of the crossed edge"},
        {"name": "edge_short_label", "type": "TEXT", "description": "Short label of the crossed edge"},
    ],
}


# =============================================================================
# Database creation and population
# =============================================================================

def create_database(
    scene: 'Scene',
    segments: List['Ray'],
) -> sqlite3.Connection:
    """
    Create and populate an in-memory SQLite database from simulation data.

    Called at set_context() time. Creates 5 tables and precomputes spatial
    join tables (ray-glass containment, ray-edge crossing) using Shapely.

    Args:
        scene: The live Scene object.
        segments: List of Ray segments from the simulation.

    Returns:
        An in-memory sqlite3.Connection with all tables populated.
    """
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')

    conn.executescript(_SCHEMA_RAYS)
    conn.executescript(_SCHEMA_GLASS_OBJECTS)
    conn.executescript(_SCHEMA_EDGES)
    conn.executescript(_SCHEMA_RAY_GLASS_MEMBERSHIP)
    conn.executescript(_SCHEMA_RAY_EDGE_CROSSING)

    _populate_rays(conn, segments)
    _populate_glass_and_spatial(conn, scene, segments)

    conn.commit()
    return conn


def _populate_rays(conn: sqlite3.Connection, segments: List['Ray']) -> None:
    """Insert one row per ray segment into the rays table."""
    rows = []
    for ray in segments:
        p1x, p1y, p2x, p2y = _ray_endpoints(ray)
        mx = (p1x + p2x) / 2.0
        my = (p1y + p2y) / 2.0
        dx = p2x - p1x
        dy = p2y - p1y
        length = math.sqrt(dx * dx + dy * dy)

        polar_ratio: Optional[float] = None
        if ray.brightness_s > 1e-10:
            polar_ratio = ray.brightness_p / ray.brightness_s

        rows.append((
            ray.uuid,
            ray.parent_uuid,
            ray.interaction_type,
            p1x, p1y, p2x, p2y,
            mx, my,
            length,
            ray.brightness_s,
            ray.brightness_p,
            ray.total_brightness,
            ray.degree_of_polarization,
            polar_ratio,
            ray.wavelength,
            int(ray.gap),
            int(ray.is_new),
            int(ray.is_tir_result),
            int(ray.caused_tir),
            ray.tir_count,
            int(getattr(ray, 'is_grazing_result__angle', False)),
            int(getattr(ray, 'caused_grazing__angle', False)),
            int(getattr(ray, 'is_grazing_result__polar', False)),
            int(getattr(ray, 'caused_grazing__polar', False)),
            int(getattr(ray, 'is_grazing_result__transm', False)),
            int(getattr(ray, 'caused_grazing__transm', False)),
            ray.source_uuid,
            ray.source_label,
        ))

    conn.executemany(
        'INSERT INTO rays VALUES (' + ','.join(['?'] * 29) + ')',
        rows,
    )


def _populate_glass_and_spatial(
    conn: sqlite3.Connection,
    scene: 'Scene',
    segments: List['Ray'],
) -> None:
    """
    Populate glass_objects, edges, and precompute spatial join tables.

    Iterates over all BaseGlass instances in the scene. For each glass:
    creates a Shapely polygon, tests ray midpoint containment, and tests
    ray-edge intersection.
    """
    from ..core.scene_objs.base_glass import BaseGlass

    glasses = [obj for obj in scene.objs if isinstance(obj, BaseGlass)]

    # Precompute ray geometries once
    ray_mids: List[tuple] = []
    ray_lines: List[tuple] = []
    for ray in segments:
        p1x, p1y, p2x, p2y = _ray_endpoints(ray)
        mx = (p1x + p2x) / 2.0
        my = (p1y + p2y) / 2.0
        ray_mids.append((ray.uuid, Point(mx, my)))
        ray_lines.append((ray.uuid, LineString([(p1x, p1y), (p2x, p2y)])))

    membership_rows: List[tuple] = []
    crossing_rows: List[tuple] = []

    for glass in glasses:
        glass_uuid = glass.uuid
        glass_name = getattr(glass, 'name', None)
        glass_type = getattr(glass.__class__, 'type', None) or glass.__class__.__name__
        ref_index = getattr(glass, 'refIndex', 1.5)
        display_name = glass.get_display_name()

        poly = glass_to_polygon(glass)
        centroid = poly.centroid
        area = poly.area
        perimeter = poly.length

        edge_descs = get_edge_descriptions(glass)

        # Insert glass_objects row
        conn.execute(
            'INSERT INTO glass_objects VALUES (?,?,?,?,?,?,?,?,?,?)',
            (glass_uuid, glass_name, display_name, glass_type,
             ref_index, len(edge_descs),
             centroid.x, centroid.y, area, perimeter),
        )

        # Insert edges rows
        for ed in edge_descs:
            conn.execute(
                'INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (glass_uuid, glass_name, ed.index,
                 ed.short_label, ed.long_label, ed.edge_type.value,
                 ed.p1.x, ed.p1.y, ed.p2.x, ed.p2.y,
                 ed.midpoint.x, ed.midpoint.y, ed.length),
            )

        # Precompute ray-glass membership (containment)
        for ray_uuid, mid in ray_mids:
            if poly.contains(mid):
                membership_rows.append((ray_uuid, glass_uuid, glass_name))

        # Precompute ray-edge crossings
        for ed in edge_descs:
            edge_line = LineString([(ed.p1.x, ed.p1.y), (ed.p2.x, ed.p2.y)])
            for ray_uuid, ray_line in ray_lines:
                if ray_line.intersects(edge_line):
                    crossing_rows.append(
                        (ray_uuid, glass_uuid, glass_name,
                         ed.index, ed.short_label)
                    )

    conn.executemany(
        'INSERT INTO ray_glass_membership VALUES (?,?,?)',
        membership_rows,
    )
    conn.executemany(
        'INSERT INTO ray_edge_crossing VALUES (?,?,?,?,?)',
        crossing_rows,
    )


# =============================================================================
# SQL validation and query execution
# =============================================================================

def _validate_sql(sql: str) -> Optional[str]:
    """
    Validate that sql is a read-only SELECT statement.

    Returns None if valid, or an error message string if invalid.
    """
    stripped = sql.strip()
    if not stripped:
        return "Empty SQL statement."
    if not stripped.upper().startswith('SELECT'):
        return "Only SELECT statements are allowed. Your query must start with SELECT."
    if _FORBIDDEN_KEYWORDS.search(stripped):
        return (
            "Statement contains forbidden keywords (INSERT, UPDATE, DELETE, DROP, etc.). "
            "Only SELECT queries are allowed."
        )
    without_trailing = stripped.rstrip('; \t\n')
    if ';' in without_trailing:
        return "Multiple statements are not allowed. Please send a single SELECT query."
    return None


def _convert_value(value: Any) -> Any:
    """Convert SQLite values for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def execute_query(
    conn: sqlite3.Connection,
    sql: str,
    max_rows: int = MAX_QUERY_ROWS,
) -> Dict[str, Any]:
    """
    Execute a validated SELECT query and return columnar results.

    Args:
        conn: The SQLite connection.
        sql: The SQL query string (must be a SELECT).
        max_rows: Maximum number of rows to return.

    Returns:
        Dict with 'columns', 'rows', 'row_count', and optionally
        'truncated'/'note' if results were capped.

    Raises:
        ValueError: If the SQL is not a valid SELECT statement.
        sqlite3.OperationalError: If the SQL has syntax errors or
            references unknown tables/columns.
    """
    validation_error = _validate_sql(sql)
    if validation_error:
        raise ValueError(validation_error)

    cursor = conn.execute(sql)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []

    rows = []
    truncated = False
    for row in cursor:
        if len(rows) >= max_rows:
            truncated = True
            break
        rows.append([_convert_value(v) for v in row])

    result: Dict[str, Any] = {
        'columns': columns,
        'rows': rows,
        'row_count': len(rows),
    }
    if truncated:
        result['truncated'] = True
        result['max_rows'] = max_rows
        result['note'] = (
            f"Results truncated to {max_rows} rows. "
            f"Add LIMIT to your query or use more specific WHERE conditions."
        )

    return result


# =============================================================================
# Schema description
# =============================================================================

def get_schema_description(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Return a structured description of all tables in the database.

    Includes column names, types, descriptions, and current row counts.

    Args:
        conn: The SQLite connection.

    Returns:
        Dict mapping table names to their schema info and row counts.
    """
    tables: Dict[str, Any] = {}

    for table_name, col_descs in _COLUMN_DESCRIPTIONS.items():
        cursor = conn.execute(f'SELECT COUNT(*) FROM {table_name}')
        row_count = cursor.fetchone()[0]

        tables[table_name] = {
            'columns': col_descs,
            'row_count': row_count,
        }

    return tables
