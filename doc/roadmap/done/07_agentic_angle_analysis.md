## Status of Issue #7: Ambiguous angle semantics
In the analysis in doc roadmap 06_refactor_analysis.md, issue #7 was identified as a significant gap in the agentic system. This is how the issue was worded:

Ambiguous angle semantics in `find_rays_by_angle_to_edge_xml`

The docstring says "Minimum angle from edge normal in degrees" but doesn't
clarify:

* *0 = perpendicular to edge* (head-on) vs. *0 = parallel to edge* (grazing)
  — the underlying code uses 0 = perpendicular (measuring from normal), but
  this is counter-intuitive for optics, where "angle of incidence" is measured
  from the normal too. An LLM might interpret it either way.
* The `proximity` parameter from the underlying function is silently dropped
  in the wrapper. This means the default proximity (2x edge length) is always
  used, which may not be appropriate for all scenes.

## The current status of the code:

### 1. Ambiguous docstring semantics — Resolved in the underlying function
The underlying function `ray_geometry_queries.py`:299-370 now has a clear, unambiguous docstring (lines 315-317):
```python
"""
Angles are in degrees, measured from the edge normal:
- 0 = perpendicular to edge (head-on incidence)
- 90 = parallel to edge (grazing incidence)
"""
```
This was improved from the vague wording that the roadmap flagged. The semantics are explicitly documented.

### 2. proximity silently dropped in `_xml` wrapper — Still present, but moot
The wrapper `agentic_tools.py`:410-438 still does not expose the proximity parameter — it calls `find_rays_by_angle_to_edge()` with only `min_angle` and `max_angle`, letting proximity default to 2x edge length.

However, the new architecture (`agentic_db.py`) makes this largely moot. The refactored system replaced the old `_xml` wrappers with an SQL-based approach:

The `ray_edge_crossing` table (`agentic_db.py`:138-147) precomputes which rays cross which edges using geometric intersection — no proximity filter needed since it checks actual intersection, not proximity.
The rays table has coordinate columns `(p1_x, p1_y, p2_x, p2_y, midpoint_x, midpoint_y)` that let the LLM compute distances via SQL if desired.

### 3. `Angle-to-edge` not precomputed in DB — Not implemented
The roadmap (line 292) proposed precomputing `angle_to_edge_X` columns in the database during setup, but this was not done. The `ray_edge_crossing`table only stores `(ray_uuid, glass_uuid, glass_name, edge_index, edge_short_label)` — pure crossing facts, no angle data.

This means an LLM agent using the new SQL system cannot query "find rays hitting edge X at angles between 10-30 degrees" — the geometric angle computation from `find_rays_by_angle_to_edge()` has no SQL equivalent. The agent would need to:

Use raw coordinate data from the rays table and edges table to compute angles in SQL (mathematically possible but complex), or
Fall back to the old `find_rays_by_angle_to_edge_xml` wrapper.

### Summary
(Sub-issues ->  Status)

* Ambiguous docstring	-> Fixed in underlying function
* proximity dropped in wrapper->Still present but moot (old `_xml` wrappers are superseded)
* Angle precomputation in DB ->	Not implemented 

The importan point:
*angle-based queries are a gap in the new SQL layer*

The main remaining gap is that `angle-to-normal` data is not available in the SQL database, so this class of query can't be done through the new `query_rays()` tool.


## Proposed Changes needed to add angle_to_normal to ray_edge_crossing
There are 4 touch-points, all in `agentic_db.py`.

### 1st Touch-point: Schema DDL — add column to ray_edge_crossing (line ~138)
Add an `angle_to_normal` REAL column to `_SCHEMA_RAY_EDGE_CROSSING`:

```sql
CREATE TABLE ray_edge_crossing (
    ray_uuid         TEXT NOT NULL,
    glass_uuid       TEXT NOT NULL,
    glass_name       TEXT,
    edge_index       INTEGER NOT NULL,
    edge_short_label TEXT NOT NULL,
    angle_to_normal  REAL NOT NULL,   -- NEW: degrees, 0=head-on, 90=grazing
    PRIMARY KEY (ray_uuid, glass_uuid, edge_index)
);
```
Semantics match the existing convention in `ray_geometry_queries.py`:315-317: 0° = perpendicular to edge (head-on), 90° = parallel (grazing).

### 2nd Touch-point: Column metadata — update `_COLUMN_DESCRIPTIONS` (line ~218)
Add the description entry to the "ray_edge_crossing" list so that `describe_schema()` exposes the new column to the LLM:

```python
{"name": "angle_to_normal", "type": "REAL",
 "description": "Angle between ray direction and edge outward normal in degrees "
                "(0 = head-on / perpendicular to edge, 90 = grazing / parallel to edge)"},
```

### 3rd Touch-point: Precomputation loop — compute angle during crossing detection (lines ~380-389)
This is the core change. The current crossing loop at `agentic_db.py`:381-389 only checks intersection:

```python
# Current code

for ed in edge_descs:
    edge_line = LineString([(ed.p1.x, ed.p1.y), (ed.p2.x, ed.p2.y)])
    for ray_uuid, ray_line in ray_lines:
        if ray_line.intersects(edge_line):
            crossing_rows.append(
                (ray_uuid, glass_uuid, glass_name,
                 ed.index, ed.short_label)
            )
```            
It needs to also compute the outward normal and angle for each crossing. The logic to reuse is the same as `ray_geometry_queries.py`:196-219 (outward normal) and `ray_geometry_queries.py`:352-367 (angle from dot product). 

Concretely:

Before the inner edge loop, compute the centroid once (already available as centroid on line 352).
For each edge, compute the outward normal using the centroid disambiguation — the same algorithm as `_edge_outward_normal`: compute both candidate normals `(-dy/L, dx/L)` and `(dy/L, -dx/L)`, pick the one whose dot product with (`edge_midpoint` → centroid) is negative.

For each crossing ray, compute the angle: extract the ray direction unit vector from `ray_line` coordinates, take `abs(dot(ray_dir, normal))`, clamp to `[0, 1]`, `math.degrees(math.acos(...))`.
The simplest approach is to import and call `_edge_outward_normal` directly from `ray_geometry_queries`. However, that function takes an `EdgeDescription` and a `BaseGlass` object, while the `agentic_db.py` loop already has the `EdgeDescription` (ed) and the `centroid` (centroid). 

Two options:

Option A: Import `_edge_outward_normal` from `ray_geometry_queries` — but it calls `glass_to_polygon(glass)` internally to get the centroid, which is redundant since `agentic_db.py` already computed it. It also requires passing the `glass` object, which is available in the loop.

Option B (cleaner): Write a small local helper in `agentic_db.py` that takes `(ed, centroid)` and returns `(nx, ny)`. This avoids importing a private function and avoids recomputing the polygon/centroid. The math is 10 lines.

The ray direction also needs precomputing. The current `ray_lines` list stores `(uuid, LineString)`. You'd need to also precompute the unit direction vector for each ray. This could be done by extending the precomputation loop at lines 331-339 to also store `(rdx, rdy)` per ray, or by extracting it from the `LineString` coordinates inside the crossing loop.

### 4th Touch-point: INSERT statement — add the 6th placeholder (line ~396)
Update the insert from 5 to 6 values:

```python
conn.executemany(
    'INSERT INTO ray_edge_crossing VALUES (?,?,?,?,?,?)',  # was 5 ?'s
    crossing_rows,
)
```
And each crossing_rows.append(...) adds the angle as the 6th element.

### What does NOT need to change
`ray_geometry_queries.py`: The underlying `find_rays_by_angle_to_edge()` function stays as-is. It's still useful for non-SQL callers.
`agentic_tools.py`: The old _xml wrapper is superseded and doesn't need updating.
`tool_registry.py`: The registry entries don't change — the SQL tool's schema is generic (`query_rays(sql: str)`).

### What the LLM agent gains
After this change, the agent can write:

```sql
-- Rays hitting the South edge at near-normal incidence (< 15°)
SELECT r.uuid, r.brightness_total, c.angle_to_normal
FROM ray_edge_crossing c
JOIN rays r ON r.uuid = c.ray_uuid
WHERE c.edge_short_label = 'S' AND c.angle_to_normal < 15.0
```
```sql
-- Compare incidence angles across edges
SELECT c.edge_short_label, AVG(c.angle_to_normal), COUNT(*)
FROM ray_edge_crossing c
GROUP BY c.edge_short_label
```

This completely subsumes `find_rays_by_angle_to_edge_xml` with full composability, no proximity heuristic needed (it uses actual intersection), and unambiguous semantics (column description in `describe_schema()` spells out the convention).


## Implementation Notes

Implementation completed. Option B was chosen for the normal helper. All four touch-points were applied to `agentic_db.py`. No other files were modified.

### Touch-point 1: Schema DDL (line ~138)
Added `angle_to_normal REAL NOT NULL` column to `_SCHEMA_RAY_EDGE_CROSSING`, positioned after `edge_short_label` and before the PRIMARY KEY constraint. Column alignment was adjusted to match the existing style.

### Touch-point 2: Column metadata (line ~219)
Added the `angle_to_normal` entry to the `"ray_edge_crossing"` list in `_COLUMN_DESCRIPTIONS` with description:
`"Angle between ray direction and edge outward normal in degrees (0 = head-on / perpendicular to edge, 90 = grazing / parallel to edge)"`

### Touch-point 3: Geometry helpers and crossing loop
**Option B was implemented** -- two local helper functions were added to `agentic_db.py` in a new section `"Geometry helpers for angle computation"`, placed between the column metadata and the database creation section:

- `_outward_normal(edge_desc, centroid_x, centroid_y)` -- Same algorithm as `ray_geometry_queries._edge_outward_normal` but takes centroid coordinates directly instead of a `BaseGlass` object. Avoids importing a private function and avoids the redundant `glass_to_polygon()` call since `_populate_glass_and_spatial` already has the centroid.

- `_angle_between_ray_and_normal(ray_dx, ray_dy, normal)` -- Computes `math.degrees(math.acos(abs(dot)))` with clamping. Factored out as a named function for clarity, matching the logic at `ray_geometry_queries.py`:361-367.

**Ray direction precomputation:** The `ray_lines` list was extended from `(uuid, LineString)` to `(uuid, LineString, unit_dx, unit_dy)`. The unit direction vector is computed once per ray during the existing precomputation loop, avoiding redundant work inside the O(rays x edges) crossing loop.

**Crossing loop changes:** Before the inner edge loop, `cx, cy = centroid.x, centroid.y` is extracted once per glass. For each edge, `_outward_normal(ed, cx, cy)` is called once. For each crossing, `_angle_between_ray_and_normal(rdx, rdy, normal)` computes the angle, which is appended as the 6th tuple element.

### Touch-point 4: INSERT statement (line ~469)
Updated from `VALUES (?,?,?,?,?)` to `VALUES (?,?,?,?,?,?)` -- 6 placeholders matching the 6-element tuples in `crossing_rows`.

### Note: multiple ray segments can cross the same edge
The crossing table records every (ray_segment, edge) geometric intersection. In a typical prism scenario the ray tracer produces three segments: the incoming source ray, the refracted interior ray, and the exit ray. The interior ray geometrically intersects *both* the entry edge and the exit edge, so the table will contain two rows for it — each with a different `angle_to_normal` because the two edges have different outward normals.

Concretely, for a ray entering edge A and exiting edge B:

| ray segment      | edge crossed | angle_to_normal meaning               |
|------------------|--------------|----------------------------------------|
| source ray       | edge A       | angle of incidence at A (exterior)     |
| interior ray     | edge A       | angle of refraction at A (interior)    |
| interior ray     | edge B       | angle of incidence at B (interior)     |
| exit ray         | edge B       | angle of refraction at B (exterior)    |

The data is unambiguous — the composite key `(ray_uuid, glass_uuid, edge_index)` identifies each crossing uniquely — but an LLM agent querying the table needs to join on the `rays` table and filter by `interaction_type` to isolate the specific segment of interest. For example, to get only the *incoming* angle of incidence at an edge:

```sql
SELECT c.angle_to_normal
FROM ray_edge_crossing c
JOIN rays r ON r.uuid = c.ray_uuid
WHERE c.edge_short_label = 'S' AND r.interaction_type = 'source'
```

### What was NOT changed
- `ray_geometry_queries.py` -- untouched, `_edge_outward_normal` and `find_rays_by_angle_to_edge` remain as-is for non-SQL callers.
- `agentic_tools.py` -- untouched, the old `_xml` wrappers are superseded.
- `tool_registry.py` -- untouched, the `query_rays(sql: str)` tool is schema-generic.
- No new imports were added (the existing `import math` already covered the needs).