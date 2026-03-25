# Tools to analyse rays and geometries
We want some tools for querying optical simulation results in an efficient way. In this scope we want tools that describe rays in relationship with geometries.
The geometry-ray tools belong in a new file (analysis/ray_geometry_queries.py) and its functions takes typically in a `List[Ray]` + scene objects or an object of the class `Scene` as a container and strings representing the name of the scene objects.

### Relationship to lineage tools

The lineage analysis module (`analysis/lineage_analysis.py`) operates on the **ray tree** (parent-child graph) and answers questions about **paths**: which paths carry energy, where do they branch, what angular distribution do they have.

The geometry-ray query tools proposed here operate on **segments vs. scene geometry** and answer a fundamentally different class of question: which segments are inside a glass, which cross a specific edge, etc. They don't need the lineage tree -- they work on a flat `List[Ray]`.

The two tool families compose naturally via uuid-based filtering. The lineage functions already accept `leaf_uuids` parameters:

```python
# "Which paths through the prism's exit face carry the most energy?"
exit_rays = find_rays_crossing_edge(segments, prism, 'east')
exit_leaf_uuids = [r.uuid for r in exit_rays if not lineage.get_children(r.uuid)]
ranked = rank_paths_by_energy(lineage, leaf_uuids=exit_leaf_uuids)
```

### Existing infrastructure to build on

- `glass_to_polygon(glass)` in `analysis/glass_geometry.py` -- converts a glass object's path to a Shapely `Polygon`
- `get_edge_descriptions(glass)` in `analysis/glass_geometry.py` -- returns `List[EdgeDescription]` with p1, p2, midpoint, label, edge_type for each edge
- `BaseGlass.edge_labels`, `get_edge_label()`, `label_edge()` -- edge labeling on glass objects
- `BaseSceneObj.name`, `BaseSceneObj.uuid`, `BaseSceneObj.get_display_name()` -- object identification
- `Ray.degree_of_polarization`, `Ray.polarization_ratio` -- computed polarization metrics
- `Ray.uuid`, `Ray.interaction_type` -- lineage identity per segment

---

## Phase 0: Scene object lookup utilities

`Scene` currently has no methods to find objects by name, type, or uuid. The query functions need a way to resolve string names to objects, so callers can write `find_rays_inside_glass(segments, scene, 'Main Prism')` instead of passing the object directly.

```python
def get_object_by_name(scene: Scene, name: str) -> BaseSceneObj:
    """
    Find a scene object by its user-defined name.

    Raises ValueError if no object has that name, or if multiple objects
    share the same name (names are not enforced unique).
    """

def get_object_by_uuid(scene: Scene, uuid: str) -> BaseSceneObj:
    """
    Find a scene object by its UUID (exact match or prefix match).

    Supports partial uuids (e.g. first 8 chars) for convenience.
    Raises ValueError if not found or if prefix is ambiguous.
    """

def get_objects_by_type(scene: Scene, type_name: str) -> List[BaseSceneObj]:
    """
    Find all scene objects of a given type (e.g. 'Glass', 'Mirror', 'PointSource').

    Matches against the class-level `type` attribute.
    Returns empty list if none found.
    """
```

These should live as standalone functions in the new `ray_geometry_queries.py` module (not as methods on Scene) to avoid modifying core code and keep the analysis module self-contained.

> **Implementation status: DONE**
> Implemented in `analysis/ray_geometry_queries.py` and exported via `analysis/__init__.py`.
> All three functions follow the signatures above. `get_object_by_uuid` supports both exact and prefix matching. `get_objects_by_type` falls back to `__class__.__name__` if no class-level `type` attribute is present.

---

## Phase 1: Custom query functions (recommended for agents)
A series of tools that an agent can call with clear semantics, rather than crafting XPath queries.

These four functions operate on segments vs. scene geometry. They answer a fundamentally different class of question:

| Proposed function | Question it answers |
|---|---|
| `find_rays_inside_glass` | Which segments are geometrically inside a glass body? |
| `find_rays_by_angle_to_edge` | What's the incidence angle at a specific edge? |
| `find_rays_by_polarization` | Which segments have extreme polarization? |
| `find_rays_crossing_edge` | Which segments intersect a specific glass edge? |

These are geometry-centric: they need Shapely `contains`, `intersects`, edge label lookups, and normal-vector calculations. They don't need the lineage tree at all -- they operate on a flat `List[Ray]`.

```python
def find_rays_inside_glass(segments: List[Ray], glass_obj) -> List[Ray]:
    """
    Return rays whose midpoint is inside the given glass object.

    Uses glass_to_polygon() to get the Shapely polygon, then tests
    whether the midpoint of each segment (p1+p2)/2 is contained.

    Note: a ray crossing a glass boundary will have its midpoint inside
    only if the segment is mostly interior. For boundary-crossing rays,
    use find_rays_crossing_edge() instead.
    """

def find_rays_by_angle_to_edge(
    segments: List[Ray], glass_obj, edge_label: str,
    min_angle: float = 0.0, max_angle: float = 90.0
) -> List[Ray]:
    """
    Return rays within angle range relative to a specified edge.

    Computes the angle between each ray's direction vector and the
    edge's outward normal. Only rays whose midpoint is within a
    reasonable distance of the edge are considered (to avoid matching
    rays on the far side of the glass).

    Angles are in degrees, measured from the edge normal:
    - 0 = perpendicular to edge (head-on)
    - 90 = parallel to edge (grazing)

    Uses get_edge_descriptions() to resolve the edge_label to geometry.
    """

def find_rays_by_polarization(
    segments: List[Ray],
    min_dop: float = 0.0,
    max_dop: float = 1.0
) -> List[Ray]:
    """
    Filter rays by degree of polarization.

    Uses Ray.degree_of_polarization property. 0 = unpolarized, 1 = fully polarized.

    Note: this function is not geometry-dependent -- it's a pure filter
    on ray properties, similar to filter_tir_rays and filter_grazing_rays
    in saving.py. Placed here to keep the query toolkit self-contained
    for agents.
    """

def find_rays_crossing_edge(
    segments: List[Ray], glass_obj, edge_label: str
) -> List[Ray]:
    """
    Return rays that cross the specified edge.

    Uses Shapely LineString intersection between each ray segment
    (p1->p2) and the edge geometry (from get_edge_descriptions()).

    Optionally could return crossing direction (entering vs. exiting)
    by comparing the ray direction against the edge outward normal.
    This can also be inferred from interaction_type: 'refract' segments
    crossing an edge are entering or exiting the glass body.
    """
```

### Design notes for phase 1

- **Edge resolution**: `find_rays_by_angle_to_edge` and `find_rays_crossing_edge` need to resolve an edge label string to actual geometry. They should use `get_edge_descriptions(glass_obj)` and find the `EdgeDescription` whose `short_label` or `long_label` matches the given `edge_label`. Raise `ValueError` if the label doesn't match any edge.

- **Return full Ray objects** (not uuids). This keeps them directly composable with lineage functions -- just extract `.uuid` when bridging:
  ```python
  crossing = find_rays_crossing_edge(segments, prism, 'east')
  uuids = [r.uuid for r in crossing]
  ```

- **Consider adding crossing direction to `find_rays_crossing_edge`**: returning a list of `(Ray, str)` tuples where the string is `'entering'` or `'exiting'` would be useful. This can be computed by checking whether the dot product of the ray direction and the edge outward normal is positive (entering) or negative (exiting). Alternatively, keep the simple `List[Ray]` signature and offer a separate `classify_crossing_direction()` function.

> **Implementation status: DONE**
> Implemented in `analysis/ray_geometry_queries.py` and exported via `analysis/__init__.py`.
>
> **Internal helpers added** (not exported, prefixed with `_`):
> - `_resolve_edge(glass, edge_label)` — multi-strategy matching: short_label → long_label → numeric index string. Raises `ValueError` with list of available labels on miss.
> - `_edge_to_linestring(edge)` — converts `EdgeDescription` to Shapely `LineString`.
> - `_edge_outward_normal(edge, glass)` — computes unit normal pointing away from glass centroid. Uses dot product between candidate normal and midpoint-to-centroid vector to pick the outward direction.
> - `_ray_endpoints(ray)` — handles the dual `dict` / Shapely `Point` representation of `ray.p1` and `ray.p2`.
>
> **Design decisions**:
> - `find_rays_inside_glass` uses `Polygon.contains(midpoint)`, not intersection. This means segments that straddle a boundary are only included if their midpoint is interior.
> - `find_rays_crossing_edge` uses `LineString.intersects()` between ray segment and edge. Returns `List[Ray]` (not tuples with direction). Crossing direction classification deferred to a future helper.
> - `find_rays_by_angle_to_edge` uses absolute dot product between ray direction and edge outward normal, so the angle is always in [0°, 90°] regardless of ray travel direction. Default proximity filter is 2× edge length.
> - `find_rays_by_polarization` is a pure filter using `Ray.degree_of_polarization`.

---

## Phase 2: Low level and flexible XPath for everything left out

This option leverages the `saving.rays_to_xml()` function and just parses its output. Here we list a series of examples that can be used to work on the XML with queries on its values. This is the escape hatch for ad-hoc queries that don't justify a dedicated function.

**Note**: `lxml` would be an optional dependency (like `networkx` for lineage export). The standard library `xml.etree.ElementTree` can handle basic XPath but lacks the full XPath 1.0 support (no numeric comparisons in predicates). For full XPath, `lxml` is needed.

```python
from lxml import etree

xml_string = rays_to_xml(segments)
root = etree.fromstring(xml_string.encode('utf-8'))

# Find all rays with high polarization (dop > 0.3)
polarized = root.xpath("//ray[degree_of_polarization > 0.3]")

# Find all rays with a specific wavelength
red_rays = root.xpath("//ray[wavelength = 650]")

# Find rays that experienced TIR
tir_rays = root.xpath("//ray[tir/is_tir_result = 'true']")

# Find rays from a specific source label
labeled = root.xpath("//ray[source/label = 'Red Ray']")

# Find rays with brightness above a threshold
bright = root.xpath("//ray[brightness_total > 0.5]")

# Find rays with grazing incidence (any criterion)
grazing = root.xpath("//ray[grazing]")

# Combine criteria: bright TIR rays with high polarization
combined = root.xpath(
    "//ray[brightness_total > 0.1 and tir and degree_of_polarization > 0.5]"
)

# Count rays by wavelength (returns text nodes, not rays)
wavelengths = root.xpath("//ray/wavelength/text()")
from collections import Counter
wl_counts = Counter(wavelengths)
```

### Bridging XPath results back to Ray objects

The `<ray index="...">` attribute maps directly to the position in the `List[Ray]` that was passed to `rays_to_xml()`. This enables the round-trip:

```python
# XPath query -> indices -> Ray objects
xml_string = rays_to_xml(segments)
root = etree.fromstring(xml_string.encode('utf-8'))
hits = root.xpath("//ray[degree_of_polarization > 0.8]")
ray_objects = [segments[int(h.get('index'))] for h in hits]

# Now compose with lineage
leaf_uuids = [r.uuid for r in ray_objects if not lineage.get_children(r.uuid)]
ranked = rank_paths_by_energy(lineage, leaf_uuids=leaf_uuids)
```
