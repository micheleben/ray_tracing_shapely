# Ray Lineage Tracking & Bidirectional Tracing Roadmap

## Purpose

Enable parent-child relationship tracking for all ray segments in a simulation,
building toward post-hoc path analysis, importance sampling, and bidirectional
ray tracing.

## Long-term aim

1. **Post-hoc path analysis** -- after a simulation completes, query which optical
   paths carry the most energy to a detector, identify TIR traps, and understand
   branching patterns across the optical system.
2. **Importance sampling** -- use path analysis from a coarse forward run to build
   angular importance distributions, then allocate more rays to high-contribution
   directions in a refined run.
3. **Bidirectional tracing** -- trace rays forward from sources *and* backward from
   detectors, then connect the two trees at shared surfaces to dramatically
   improve efficiency for systems where only a small fraction of source rays
   reach the detector.

---

## Current Architecture

### Project layout (key files)

```
src-python/ray_tracing_shapely/
  core/
    ray.py                              # Ray class
    simulator.py                        # Main tracing engine
    scene.py                            # Scene container & settings
    geometry.py                         # geometry.point(), geometry.line()
    equation.py                         # Math utilities
    constants.py                        # Physical constants
    scene_objs/
      base_scene_obj.py                 # Base class for all objects (has _uuid)
      base_glass.py                     # Snell's law, Fresnel equations, TIR
      base_filter.py                    # Wavelength filtering
      base_custom_surface.py            # Custom surface support
      base_grin_glass.py                # GRIN glass support
      glass/glass.py                    # Arbitrary polygon glass
      glass/spherical_lens.py           # Spherical lens
      glass/ideal_lens.py              # Ideal thin lens
      mirror/mirror.py                  # Flat mirror
      blocker/blocker.py                # Light blocker
      light_source/point_source.py      # Point source, beam, etc.
      other/detector.py                 # Detector
  analysis/
    saving.py                           # CSV/XML export
    simulation_result.py                # Result wrapper
```

### The Ray class (`core/ray.py`)

A plain Python class (not a dataclass). Key fields:

| Field | Type | Purpose |
|-------|------|---------|
| `p1`, `p2` | `dict[str, float]` | Segment endpoints `{'x': float, 'y': float}` |
| `brightness_s`, `brightness_p` | `float` | S- and P-polarization brightness |
| `wavelength` | `float \| None` | Wavelength in nm (None = white) |
| `gap` | `bool` | If True, segment is not drawn |
| `is_new` | `bool` | If True, ray has not been processed |
| `body_merging_obj` | `Any \| None` | For surface merging |
| `is_tir_result` | `bool` | Produced by a TIR event |
| `caused_tir` | `bool` | This segment's endpoint caused TIR |
| `tir_count` | `int` | Cumulative TIR count in lineage |
| `is_grazing_result__angle` | `bool` | Grazing incidence (angle criterion) |
| `caused_grazing__angle` | `bool` | Endpoint triggered angle criterion |
| `is_grazing_result__polar` | `bool` | Grazing incidence (polarization criterion) |
| `caused_grazing__polar` | `bool` | Endpoint triggered polar criterion |
| `is_grazing_result__transm` | `bool` | Grazing incidence (transmission criterion) |
| `caused_grazing__transm` | `bool` | Endpoint triggered transm criterion |
| `source_uuid` | `str \| None` | UUID of emitting light source |
| `source_label` | `str \| None` | Human-readable label |

**Not currently present:** `uuid`, `parent_uuid`, `interaction_type`.

The `copy()` method deep-copies `p1`/`p2` dicts and propagates all tracking
fields. Position-specific flags (`caused_tir`, `caused_grazing__*`) are reset
to False in copies.

### The simulator loop (`core/simulator.py`)

#### `Simulator.run()` (line 82)

1. Reset state (`processed_ray_count = 0`, `ray_segments = []`)
2. Call `on_simulation_start()` on each optical object -- light sources emit
   initial rays into `pending_rays`
3. Call `_process_rays()`
4. Return `ray_segments` (flat list of all completed segments)

#### `Simulator._process_rays()` (line 190)

FIFO queue loop:

```
while pending_rays and processed_ray_count < max_rays:
    ray = pending_rays.pop(0)
    ray.is_new = False

    # Extend ray to large distance for intersection testing
    # (source rays have short p2, just a direction)

    intersection_info = _find_nearest_intersection(ray)

    if no intersection:
        ray_segments.append(ray)          # <-- TERMINAL SEGMENT
    else:
        ray.p2 = intersection_point       # truncate at hit
        ray_segments.append(ray)          # <-- COMPLETED SEGMENT

        # Build OutputRayGeom with original direction + propagated flags
        result = obj.on_ray_incident(output_ray_geom, ...)

        if result is not None:
            # BRANCH A: dict with 'newRays' (from refract)
            #   -> convert each via _dict_to_ray, enqueue
            # BRANCH B: list of rays (beam splitter)
            #   -> convert each via _dict_to_ray, enqueue
            # BRANCH C: single ray
            #   -> convert via _dict_to_ray, enqueue
        else:
            # BRANCH D: in-place modification (TIR, mirror)
            #   -> convert output_ray_geom via _dict_to_ray, enqueue

    processed_ray_count += 1
```

#### `ray_segments` vs `pending_rays`

- **`ray_segments`**: completed segments with final geometry (p1 truncated to
  intersection or extended to infinity). This is the simulation output.
- **`pending_rays`**: rays awaiting processing. Their p2 is a direction, not a
  final endpoint. They become segments after the next iteration.

This distinction matters for lineage: a ray's uuid should be assigned at
creation time, but its segment geometry is only finalized when it is appended
to `ray_segments`.

#### `_dict_to_ray()` (line 562)

Converts geometry objects / dicts / OutputRayGeom back to `Ray` instances.
Already has a pattern for propagating Python-specific fields via
`hasattr`/`getattr`. New fields (`uuid`, `parent_uuid`, `interaction_type`)
follow the same pattern.

### Where rays are spawned

There are **4 distinct creation paths**:

#### 1. Light source emission (no parent)

- `PointSource.on_simulation_start()` at `light_source/point_source.py:212`
- Creates rays uniformly in all directions
- Sets `source_uuid = self.uuid`, `source_label = self.name`
- These are root nodes in the lineage graph (`parent_uuid = None`)

#### 2. Refraction + Fresnel reflection at glass

- `BaseGlass.refract()` at `base_glass.py:413`
- When `sq1 >= 0` (no TIR), creates:
  - **`ray2`** (reflected): `geometry.line()` at line 655, brightness scaled by
    Fresnel R_s, R_p
  - **`ray3`** (refracted): `geometry.line()` at line 735, brightness scaled by
    T_s, T_p
- Returns `{'newRays': [ray2], 'isAbsorbed': False}` and modifies the input
  ray to become ray3
- Back in the simulator, `newRays` items go through BRANCH A (line 323) and
  the modified input goes through BRANCH D path implicitly (it's the main
  continuation)

**Important subtlety**: `refract()` actually returns the *reflected* ray in
`newRays` and modifies the input ray in-place to become the *refracted* ray.
So the refracted ray is the "continuation" and the reflected ray is the
"spawn". Both should get the incident segment's uuid as `parent_uuid`.

#### 3. Total Internal Reflection (TIR) at glass

- `BaseGlass.refract()` at `base_glass.py:566`
- When `sq1 < 0`, modifies ray in-place (reflected direction)
- Sets `ray.is_tir_result = True`, increments `ray.tir_count`
- Returns `None`
- Simulator handles via BRANCH D (line 346)
- Single child, `interaction_type = 'tir'`

#### 4. Mirror reflection

- `Mirror.on_ray_incident()` at `mirror/mirror.py:166`
- Modifies ray in-place with reflected direction (law of reflection)
- Returns `None`
- Simulator handles via BRANCH D (line 346)
- Single child, `interaction_type = 'reflect'`

### The geometry object layer

Physics code (`refract()`, `on_ray_incident()`) does **not** work with `Ray`
objects directly. Instead it works with:

- `geometry.line()` objects (lightweight, have `p1`, `p2` as geometry points)
- `OutputRayGeom` (ad-hoc object built in `_process_rays` at line 267)

The conversion back to `Ray` happens in `_dict_to_ray()`. This means
`uuid`/`parent_uuid` either need to flow through geometry objects (adding
attributes) or be assigned at the simulator level after conversion.

**Recommendation**: assign at the simulator level. This avoids touching physics
code for uuid propagation.

---

## Phase 1: Ray Lineage Fields

### Changes to `Ray.__init__()` in `ray.py`

Add three fields:

```python
import uuid as _uuid_mod

# in __init__:
self.uuid: str = str(_uuid_mod.uuid4())
self.parent_uuid: Optional[str] = None
self.interaction_type: str = 'source'  # 'source', 'reflect', 'refract', 'tir'
```

### Changes to `Ray.copy()`

```python
# uuid gets a NEW value (copy is a new segment)
new_ray.uuid = str(_uuid_mod.uuid4())
# parent_uuid is copied (same parent as original, caller may override)
new_ray.parent_uuid = self.parent_uuid
new_ray.interaction_type = self.interaction_type
```

### Changes to `_dict_to_ray()` in `simulator.py`

Add `hasattr`/`getattr` propagation for `uuid`, `parent_uuid`,
`interaction_type` in both the object-attribute path and the dict path, same
pattern as existing TIR/grazing fields.

### Changes to `_process_rays()` in `simulator.py`

After each `_dict_to_ray()` conversion, set `parent_uuid` and
`interaction_type` on the new ray:

- **BRANCH A** (newRays from refract, line 323): these are Fresnel reflections.
  Set `new_ray.parent_uuid = ray.uuid`, `new_ray.interaction_type = 'reflect'`.
- **BRANCH B** (list, line 331): same as BRANCH A (context-dependent).
- **BRANCH C** (single ray, line 339): set based on object type.
- **BRANCH D** (in-place / None return, line 346):
  - If `is_tir_result`: `interaction_type = 'tir'`
  - If from mirror: `interaction_type = 'reflect'`
  - Otherwise: `interaction_type = 'refract'`
  - Set `new_ray.parent_uuid = ray.uuid`

For the refracted continuation (BRANCH D when refract returns non-None but
modifies the input), the modified input ray goes through BRANCH D. Set
`parent_uuid = ray.uuid`.

### Changes to light sources

In `on_simulation_start()`, each created ray already gets a new `Ray()` which
will auto-generate a uuid. Set `interaction_type = 'source'` (the default).
`parent_uuid` stays `None`.

---

## Phase 2: RayLineage Tracker

### New file: `core/ray_lineage.py`

A lightweight tree structure. NetworkX is optional -- a plain dict
implementation covers all needed operations without adding a heavy dependency.
NetworkX can be used as an optional backend for visualization and graph export.

```python
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RayLineage:
    """Tracks parent-child relationships for all ray segments."""

    _parents: dict[str, str | None] = field(default_factory=dict)
    _children: dict[str, list[str]] = field(default_factory=dict)
    _segments: dict[str, 'Ray'] = field(default_factory=dict)
    _interaction_types: dict[str, str] = field(default_factory=dict)

    def register(self, segment: 'Ray') -> None:
        self._segments[segment.uuid] = segment
        self._parents[segment.uuid] = segment.parent_uuid
        self._interaction_types[segment.uuid] = segment.interaction_type
        if segment.uuid not in self._children:
            self._children[segment.uuid] = []
        if segment.parent_uuid:
            self._children.setdefault(segment.parent_uuid, []).append(segment.uuid)

    def get_ancestors(self, uuid: str) -> list['Ray']:
        """Walk up the tree to the source."""
        result = []
        current = self._parents.get(uuid)
        while current is not None:
            result.append(self._segments[current])
            current = self._parents.get(current)
        result.reverse()
        return result

    def get_full_path(self, uuid: str) -> list['Ray']:
        """Complete chain from source to this segment (inclusive)."""
        return self.get_ancestors(uuid) + [self._segments[uuid]]

    def get_children(self, uuid: str) -> list['Ray']:
        """Direct children of this segment."""
        return [self._segments[c] for c in self._children.get(uuid, [])]

    def get_descendants(self, uuid: str) -> list['Ray']:
        """All segments spawned from this one (BFS)."""
        result = []
        queue = list(self._children.get(uuid, []))
        while queue:
            child = queue.pop(0)
            result.append(self._segments[child])
            queue.extend(self._children.get(child, []))
        return result

    def get_siblings(self, uuid: str) -> list['Ray']:
        """Other segments from same parent (e.g., reflected + refracted pair)."""
        parent = self._parents.get(uuid)
        if parent is None:
            return []
        return [self._segments[c] for c in self._children[parent] if c != uuid]

    def get_roots(self) -> list['Ray']:
        """All source segments (no parent)."""
        return [self._segments[u] for u, p in self._parents.items() if p is None]

    def get_subtree_uuids(self, uuid: str) -> set[str]:
        """All uuids in the subtree rooted at uuid (inclusive)."""
        result = {uuid}
        queue = list(self._children.get(uuid, []))
        while queue:
            child = queue.pop(0)
            result.add(child)
            queue.extend(self._children.get(child, []))
        return result

    def to_networkx(self) -> 'nx.DiGraph':
        """Export to NetworkX DiGraph (requires networkx installed)."""
        import networkx as nx
        G = nx.DiGraph()
        for uuid, segment in self._segments.items():
            G.add_node(uuid, interaction=self._interaction_types.get(uuid, 'unknown'))
        for uuid, parent in self._parents.items():
            if parent is not None:
                G.add_edge(parent, uuid)
        return G
```

### Integration with `Simulator`

Add a `lineage: RayLineage` attribute to the Simulator. Call
`self.lineage.register(ray)` at each `ray_segments.append(ray)` and at each
`pending_rays.append(new_ray)`.

Expose `lineage` on the simulation result so downstream analysis code can
query it.

---

## Phase 3: Post-hoc Path Analysis

With lineage in place, implement analysis utilities:

### Energy path ranking

For each terminal segment (no children, or hits detector), compute cumulative
brightness along its full path. Rank paths by energy delivered to detector.

```python
def rank_detector_paths(lineage: RayLineage, detector_uuids: set[str]) -> list:
    """Rank paths by energy delivered to detector."""
    paths = []
    for uuid in detector_uuids:
        path = lineage.get_full_path(uuid)
        terminal = path[-1]
        energy = terminal.brightness_s + terminal.brightness_p
        paths.append((energy, path))
    paths.sort(key=lambda x: x[0], reverse=True)
    return paths
```

### Branching statistics per surface

For each optical object, count how many rays split (refract + reflect) vs
single-path (TIR, mirror). Identify objects that cause the most ray
proliferation -- useful for ray budget management.

### TIR trap detection

Find subtrees where `tir_count` grows above a threshold. These indicate
geometries where rays bounce internally many times. Graph depth per subtree
identifies light traps.

### Angular importance distribution

For rays that reach a detector, trace back to source and record the source
emission angle. Build a histogram. This becomes the importance distribution
for a refined run.

---

## Phase 4: Bidirectional Tracing

This is the most significant architectural change. The goal is to trace rays
from both sources and detectors, then connect the forward and backward trees
at shared optical surfaces.

### Concept

```
FORWARD:  Source ---> Surface A ---> Surface B ---> (stops)
BACKWARD: Detector ---> Surface B ---> Surface A ---> (stops)

CONNECTION: At Surface B, find forward ray F and backward ray B that
            hit the same point. The full path is:
            Source -> ... -> F -> connection -> B -> ... -> Detector
```

### Backward tracing requirements

1. **Backward detector source**: A new light source type that emits rays from
   the detector surface, uniformly into the acceptance cone. Equivalent to
   running the detector as a source.

2. **Reversed Snell's law**: Refraction is reversible -- a backward ray hitting
   a glass surface from the outside refracts the same way. The physics code
   works unchanged as long as the refractive index ratio is inverted. Since
   `get_incident_type()` already determines inside/outside based on geometry,
   backward rays will refract correctly without changes to `refract()`.

3. **Reversed Fresnel coefficients**: The transmission/reflection split is the
   same in both directions by reciprocity (Stokes relations). The existing
   Fresnel code is already direction-independent.

4. **Separate lineage trees**: Forward and backward runs produce independent
   lineage graphs. They must be stored separately.

### Connecting forward and backward trees

At each optical surface, both forward and backward rays produce intersection
records. The connection algorithm:

1. For each surface, collect all forward intersection points and all backward
   intersection points.
2. For each forward point F, find the nearest backward point B on the same
   surface within a spatial tolerance.
3. Verify geometric compatibility: the forward ray's outgoing direction at F
   must be compatible with the backward ray's incoming direction at B (within
   angular tolerance after accounting for refraction).
4. If compatible, create a virtual connection edge in the combined graph.
5. The full path energy is the product of forward path transmittance and
   backward path transmittance (adjusted for the connection geometry).

### Surface hit records

To enable connection, each ray segment that hits a surface should store:

```python
# Additional fields for bidirectional connection
self.hit_surface_uuid: Optional[str] = None   # which surface was hit
self.hit_point: Optional[dict] = None          # exact intersection point
self.hit_normal: Optional[dict] = None         # surface normal at hit
self.incident_angle: Optional[float] = None    # angle of incidence
```

These are already computed inside `refract()` and `on_ray_incident()` but
currently discarded. Preserving them on the ray segment enables spatial
matching between forward and backward hits.

### NetworkX role in bidirectional tracing

This is where NetworkX becomes genuinely useful:

- **Bipartite matching**: Finding optimal pairings between forward and backward
  hits at each surface. `nx.bipartite` or weighted matching algorithms.
- **Shortest path with weights**: Once the combined graph is built (forward
  tree + connection edges + backward tree), finding the minimum-loss path
  from source to detector using edge weights = -log(transmittance).
- **Subgraph extraction**: Extracting all paths that successfully connect
  source to detector through the combined graph.

### Simulator changes for bidirectional mode

```python
class Simulator:
    def run_bidirectional(self, detector_obj):
        # Forward pass
        forward_segments = self.run()
        forward_lineage = self.lineage

        # Backward pass: emit from detector
        backward_source = BackwardDetectorSource(detector_obj)
        self.scene.add_object(backward_source)
        # ... run backward, collect backward_lineage

        # Connect trees
        connections = connect_trees(
            forward_lineage, backward_lineage,
            spatial_tolerance=1e-4,
            angular_tolerance_deg=1.0
        )

        # Build combined graph and find connected paths
        combined = build_combined_graph(
            forward_lineage, backward_lineage, connections
        )
        return combined
```

---

## Phase 5: Importance Sampling via Lineage Analysis

### Two-pass approach

1. **Coarse run**: Run forward simulation with uniform source emission at low
   `ray_density`. Build lineage. Identify which source angles lead to detector
   hits using `rank_detector_paths()`.

2. **Build importance distribution**: From the coarse run, create an angular
   PDF for the source. Angles that produced detector hits get higher weight.
   Smooth the distribution (kernel density estimation) to avoid missing nearby
   angles.

3. **Refined run**: Emit rays from the source according to the importance
   distribution instead of uniformly. Each ray carries a weight
   `w = p_uniform(angle) / p_importance(angle)` to correct for the biased
   sampling.

4. **Weighted analysis**: All downstream analysis (detector irradiance, etc.)
   uses the weight `w` to produce unbiased estimates with lower variance.

### Source modifications

Light sources need an optional `angular_pdf` parameter. When set, rays are
emitted according to this distribution rather than uniformly. The `Ray` class
needs a `weight: float = 1.0` field for importance weighting.

### Combining with bidirectional tracing

The bidirectional approach provides even better importance information:
backward rays from the detector directly identify which directions and surface
points are relevant. The backward tree's angular distribution at each surface
is the ideal importance function for the forward pass through that surface.

---

## Implementation Order and Dependencies

```
Phase 1: Ray lineage fields
   |   (uuid, parent_uuid, interaction_type on Ray)
   |   (propagation in _dict_to_ray, _process_rays)
   |
Phase 2: RayLineage tracker
   |   (register/query API, Simulator integration)
   |   (optional NetworkX export via to_networkx())
   |
Phase 3: Post-hoc path analysis
   |   (energy ranking, branching stats, TIR traps)
   |   (angular importance extraction)
   |
   +---> Phase 5: Importance sampling (depends on Phase 3)
   |     (two-pass approach, weighted rays)
   |
Phase 4: Bidirectional tracing
         (backward sources, surface hit records, tree connection)
         (NetworkX becomes a real dependency here)
```

Phases 1-3 can use the lightweight dict-based tree. Phase 4 is where NetworkX
earns its place as a real dependency.

---

## Key Design Decisions

1. **uuid generation strategy**: Use `uuid4()`. Produces long strings but is
   globally unique across runs, which is important for cross-run analysis.
   **Decision: uuid4.** :white_check_mark:

2. **Where to store lineage**: On the `SimulationResult` (persisted with
   results), not just the `Simulator` instance. This enables post-hoc analysis
   after the simulation object is discarded.
   **Decision: SimulationResult.** :white_check_mark:

3. **Segment vs ray identity**: The uuid is assigned at ray creation time (in
   `Ray.__init__`), so `parent_uuid` can be set immediately when spawning
   child rays. The same uuid follows the ray as it becomes a completed segment
   in `ray_segments`.
   **Decision: assigned at creation.** :white_check_mark:

4. **Memory overhead**: Each uuid4 string is ~36 bytes. For 10,000 segments
   with uuid + parent_uuid + interaction_type, that's ~1MB. Negligible.

5. **Backward compatibility**: All new fields have defaults (`uuid` auto-
   generated, `parent_uuid = None`, `interaction_type = 'source'`). Existing
   code that doesn't use lineage tracking works unchanged.

6. **Surface hit record storage** (Phase 4): Adding `hit_point`, `hit_normal`,
   `incident_angle` to every ray segment increases memory. Consider storing
   these only when bidirectional mode is enabled.
