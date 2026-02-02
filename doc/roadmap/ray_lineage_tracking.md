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

## Phase 1: Ray Lineage Fields -- IMPLEMENTED

> **Status**: Complete. All changes verified with prism + mirror and TIR test
> scenes. Zero orphans (every `parent_uuid` resolves to a segment in the
> output).

### Files modified

| File | What changed |
|------|-------------|
| `core/ray.py` | Added `import uuid`; 3 new fields in `__init__`; updated `copy()` and `__repr__()` |
| `core/scene_objs/base_glass.py` | Tagged `ray2` with `interaction_type='reflect'`, `ray3` with `interaction_type='refract'` |
| `core/simulator.py` | Propagation in `_dict_to_ray()`; parent/type assignment in all 4 branches of `_process_rays()` |

### Changes to `Ray.__init__()` in `ray.py`

Three new fields added after the source tracking block (lines 130-139):

```python
import uuid as _uuid_mod

# in __init__:
self.uuid: str = str(_uuid_mod.uuid4())   # Unique ID for this ray segment
self.parent_uuid: Optional[str] = None     # UUID of the parent ray (None for source rays)
self.interaction_type: str = 'source'      # 'source', 'reflect', 'refract', 'tir'
```

### Changes to `Ray.copy()`

`copy()` generates a **new uuid** (via `Ray.__init__`) since the copy is a
distinct segment. `parent_uuid` and `interaction_type` are copied from the
original. The caller (i.e. the simulator) is responsible for overriding
`parent_uuid` if the copy represents a child ray.

```python
new_ray.parent_uuid = self.parent_uuid
new_ray.interaction_type = self.interaction_type
```

### Changes to `Ray.__repr__()`

Added a `lineage_str` block that shows:
- Truncated uuid (first 8 chars) -- always shown
- Truncated parent_uuid -- shown only if not None
- `interaction_type` -- shown only if not `'source'` (to reduce noise)

Example output:
```
Ray(..., uuid=7e7befa6..., parent=47a32355..., reflect)
```

### Changes to `BaseGlass.refract()` in `base_glass.py`

The physics code creates `ray2` (Fresnel reflection) and `ray3` (refracted
ray) as `geometry.line()` objects. These lightweight objects accept arbitrary
attributes, so we tag them directly:

- `ray2.interaction_type = 'reflect'` (after line 668)
- `ray3.interaction_type = 'refract'` (after line 745)

This avoids needing heuristics in the simulator to distinguish reflected from
refracted rays in the `newRays` list. The tag flows through `_dict_to_ray()`
automatically.

Note: `parent_uuid` is **not** set here because `geometry.line()` objects
don't carry uuids. Parent assignment happens at the simulator level.

### Changes to `_dict_to_ray()` in `simulator.py`

Added `hasattr`/`getattr` propagation for `uuid`, `parent_uuid`,
`interaction_type` in both code paths (object-attribute and dict), following
the same pattern used for TIR and grazing flags. When a geometry object
carries `interaction_type` (set by `refract()`), it flows through to the new
`Ray` instance.

### Changes to `_process_rays()` in `simulator.py`

After each `_dict_to_ray()` conversion, `parent_uuid` is set to the parent
ray's uuid. The `interaction_type` assignment depends on the branch:

- **BRANCH A** (`result` is dict with `newRays`, from `refract()`):
  `interaction_type` is already set on the geometry objects by `refract()`
  (`'reflect'` for `ray2`, `'refract'` for `ray3`). The simulator sets
  `new_ray.parent_uuid = ray.uuid`.

- **BRANCH B** (`result` is a list): same as BRANCH A. Sets
  `new_ray.parent_uuid = ray.uuid`.

- **BRANCH C** (`result` is a single ray): sets
  `new_ray.parent_uuid = ray.uuid`.

- **BRANCH D** (`result` is None, in-place modification): the simulator
  determines `interaction_type` from context:
  - If `output_ray_geom.is_tir_result` is True: `interaction_type = 'tir'`
  - Otherwise (mirror reflection): `interaction_type = 'reflect'`
  - Sets `new_ray.parent_uuid = ray.uuid`

### Light sources -- no changes needed

Source rays are created as `geometry.line()` objects in `on_simulation_start()`
and converted to `Ray` objects by `_dict_to_ray()` in `Simulator.run()`.
The `Ray.__init__()` constructor auto-generates a uuid and defaults to
`parent_uuid=None`, `interaction_type='source'` -- exactly correct for root
rays. No light source code was modified.

### Verification results

**Prism + mirror test** (6 segments):
```
uuid=7e7befa6... ROOT        type=source   brightness=1.000000
uuid=79cfd24f... parent=7e.. type=reflect  brightness=0.042479  (Fresnel at prism entry)
uuid=47a32355... parent=7e.. type=refract  brightness=0.957521  (into prism)
uuid=231211e5... parent=47.. type=reflect  brightness=0.148086  (Fresnel at prism exit)
uuid=d9c6c279... parent=47.. type=refract  brightness=0.809435  (out of prism)
uuid=44201116... parent=23.. type=refract  brightness=0.139403  (Fresnel reflection re-exits)
```

**TIR test** (7 segments):
```
uuid=52b11409... ROOT        type=source   tir_count=0
uuid=dd4d4010... parent=52.. type=reflect  tir_count=0  (Fresnel at entry)
uuid=06276200... parent=52.. type=refract  tir_count=0  (into glass)
uuid=0c40b21f... parent=06.. type=tir      tir_count=1  (TIR at steep surface)
uuid=9a3009a6... parent=0c.. type=reflect  tir_count=1  (Fresnel at next surface)
uuid=15938107... parent=0c.. type=refract  tir_count=1  (exits glass)
uuid=a0d0807f... parent=9a.. type=refract  tir_count=1  (Fresnel reflection re-exits)
```

Zero orphans in both tests: every `parent_uuid` resolves to a segment uuid in
the output list.

---

## Phase 2: RayLineage Tracker -- IMPLEMENTED

> **Status**: Complete. RayLineage class created, integrated into Simulator
> and SimulationResult. Verified with prism + mirror and TIR test scenes.

### Files modified / created

| File | What changed |
|------|-------------|
| `core/ray_lineage.py` | **New file.** `RayLineage` class with dict-based tree and optional NetworkX export |
| `core/simulator.py` | Added `from .ray_lineage import RayLineage`; `self.lineage` attribute; `register()` calls; passed to `run_with_result()` |
| `analysis/simulation_result.py` | Added `lineage: Optional[RayLineage]` field; wired through `create()` factory |

### `RayLineage` class (`core/ray_lineage.py`)

A plain Python class (not a dataclass) with four internal dicts:

- `_parents: dict[str, str | None]` -- uuid to parent_uuid
- `_children: dict[str, list[str]]` -- uuid to list of child uuids
- `_segments: dict[str, Ray]` -- uuid to Ray object
- `_interaction_types: dict[str, str]` -- uuid to interaction type

**Registration:** `register(segment)` is called once per completed segment.
It populates all four dicts from the segment's `uuid`, `parent_uuid`, and
`interaction_type` fields.

**Query API:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_ancestors(uuid)` | `list[Ray]` | Chain back to source (root-first), excludes self |
| `get_full_path(uuid)` | `list[Ray]` | Root to self (inclusive) |
| `get_children(uuid)` | `list[Ray]` | Direct children |
| `get_descendants(uuid)` | `list[Ray]` | All descendants (BFS order) |
| `get_siblings(uuid)` | `list[Ray]` | Other children of same parent |
| `get_roots()` | `list[Ray]` | All source segments |
| `get_leaves()` | `list[Ray]` | All terminal segments (no children) |
| `get_subtree_uuids(uuid)` | `set[str]` | All uuids in subtree (inclusive) |
| `get_tree_depth(uuid)` | `int` | Depth (0 for roots) |
| `get_segments_by_type(type)` | `list[Ray]` | All segments of given interaction type |
| `get_segment(uuid)` | `Ray \| None` | Lookup by uuid |
| `get_lineage_statistics()` | `dict` | Summary stats (counts, max depth, branching) |
| `to_networkx()` | `nx.DiGraph` | Export (optional, requires networkx installed) |

**Design decision: register on `ray_segments.append` only.** We register
when a segment is appended to `ray_segments` (final geometry), not when it
enters `pending_rays`. This ensures the lineage tracker only contains segments
with finalized p1/p2 coordinates. Child rays in `pending_rays` will be
registered when they are processed and become segments themselves. Parent-child
linkage still works because `parent_uuid` is set at enqueue time and the
parent is always registered before its children (FIFO processing order).

### Simulator integration

- `self.lineage = RayLineage()` initialized in `__init__()` and reset in
  `run()`.
- `self.lineage.register(ray)` called immediately after each
  `self.ray_segments.append(ray)` (two call sites: no-intersection path at
  line 241 and intersection path at line 263).
- `self.lineage` passed to `SimulationResult.create()` in `run_with_result()`.

### SimulationResult integration

- Added `lineage: Optional[RayLineage] = None` field to the dataclass.
- Added `lineage` parameter to `create()` factory method.
- Accessible as `result.lineage` after `run_with_result()`.

### Verification results

**Prism + mirror test:**
```
RayLineage(segments=6, roots=1, leaves=3, max_depth=3)
Interaction counts: {'source': 1, 'reflect': 2, 'refract': 3}
Branching factor avg: 1.67

Paths to leaves:
  source -> reflect                          (depth=1)
  source -> refract -> refract               (depth=2)
  source -> refract -> reflect -> refract    (depth=3)
```

**TIR test:**
```
RayLineage(segments=7, roots=1, leaves=3, max_depth=4)
Interaction counts: {'source': 1, 'reflect': 2, 'refract': 3, 'tir': 1}

Path to TIR: source -> refract -> tir
Descendants of TIR segment: 3 (reflect, refract, refract)
```

`result.lineage.segment_count == result.segment_count` confirmed in both
tests.

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
