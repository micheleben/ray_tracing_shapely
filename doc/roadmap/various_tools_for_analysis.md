# Various tools for analysis
This file is a collection of various tools that will go into a dedicated module in `analysis/` and are supposed to be useful for the designer or for an LLM agent to interpret the context.

---

## Section 1: Fresnel equation utilities

Standalone functions that solve the Fresnel equations to compute expected transmittances, reflectances, polarization ratios, and critical angles. The physics is already implemented inside `BaseGlass.refract()` but is not callable standalone -- these utilities let a designer or agent ask "what should I expect?" without running a simulation.

Target file: `analysis/fresnel_utils.py` (new module).

### Proposed functions

```python
def fresnel_transmittances(
    n1: float, n2: float, theta_i_deg: float
) -> Dict[str, float]:
    """
    Compute Fresnel power transmittances and reflectances at an interface.

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


def critical_angle(n1: float, n2: float) -> float:
    """
    Compute the critical angle for total internal reflection.

    Args:
        n1: Refractive index of the denser medium (must be > n2).
        n2: Refractive index of the rarer medium.

    Returns:
        Critical angle in degrees.

    Raises:
        ValueError: If n1 <= n2 (no TIR possible).
    """


def brewster_angle(n1: float, n2: float) -> float:
    """
    Compute Brewster's angle (where R_p = 0).

    At Brewster's angle, reflected light is fully s-polarized.

    Args:
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmitting medium.

    Returns:
        Brewster's angle in degrees.
    """
```

### Design notes

- All functions **return** values; no `print()` side effects. Callers format output as needed.
- `fresnel_transmittances` raises `ValueError` on TIR instead of returning `None` -- this avoids mixed return types. Callers who need to check for TIR should call `critical_angle()` first or catch the exception.
- `brewster_angle` is not in the original proposal but naturally belongs here -- it's the angle where `R_p = 0` and is useful for understanding polarization effects.
- The inline demo/sweep code from the original proposal (hardcoded `n1, n2`, brute-force angle search) belongs in a test or notebook example, not in the function bodies.

### Original notebook-style code (reference only)

The following code was the starting point for the proposal. The sweep logic should be moved to docstring examples or tests.

```python
# Example: sweep angles and print T_p/T_s table
n1, n2 = 1.5, 1.785
for angle in [80, 85, 87, 88, 89, 89.5, 89.9]:
    result = fresnel_transmittances(n1, n2, angle)
    print(f"{angle:8.3f}  T_p/T_s={result['ratio_Tp_Ts']:6.4f}  "
          f"T_s={result['T_s']:.6f}  T_p={result['T_p']:.6f}")
```

---

## Section 2: Geometry convenience tools

General tools for locating points on edges and describing scene geometry. Useful for an LLM agent answering questions like "give me the coordinates of the point at 3/4 of the south edge of the prism".

These belong in `analysis/ray_geometry_queries.py` alongside the existing Phase 0/1 tools.

### Proposed functions

```python
def interpolate_along_edge(
    glass_obj: 'BaseGlass',
    edge_label: str,
    fraction: float = 0.5
) -> Tuple[float, float]:
    """
    Get the (x, y) coordinates at a fractional position along a glass edge.

    Uses get_edge_descriptions() to resolve the edge, then Shapely's
    LineString.interpolate() for the actual computation.

    Args:
        glass_obj: A BaseGlass object with labeled edges.
        edge_label: Label of the edge (short_label, long_label, or index string).
        fraction: Position along the edge, 0.0 = start (p1), 1.0 = end (p2).

    Returns:
        (x, y) tuple of the interpolated point coordinates.

    Raises:
        ValueError: If edge_label doesn't match any edge.

    Example:
        >>> # Point at 3/4 along the south edge
        >>> x, y = interpolate_along_edge(prism, 'S', 3/4)
    """


def describe_all_glass_edges(
    scene: 'Scene',
    format: str = 'text'
) -> str:
    """
    Describe all edges of all glass objects in a scene.

    Iterates over all BaseGlass subclass objects in the scene and
    concatenates their edge descriptions.

    Args:
        scene: The Scene to describe.
        format: Output format ('text' or 'xml'). Default: 'text'.

    Returns:
        Concatenated edge descriptions for all glass objects.
    """
```

### Design notes

- `interpolate_along_edge` replaces the original `calc_rationale_position_in_between_two_points`. Key changes:
  - Takes a glass object + edge label instead of raw coordinates, so it composes with the existing edge resolution infrastructure (`_resolve_edge`).
  - Uses a single `fraction: float` in [0, 1] instead of `(t, n)` integers. The caller can write `3/4` directly.
  - Returns `(x, y)` tuple; no `print()` side effects.
  - If the caller has raw coordinates instead of a glass object, Shapely's `LineString([(x1,y1),(x2,y2)]).interpolate(fraction, normalized=True)` is a one-liner -- no dedicated function needed.

- `describe_all_glass_edges` replaces the original version. Key changes:
  - Uses `isinstance(obj, BaseGlass)` instead of `isinstance(obj, Glass)` to cover all glass subclasses.
  - Supports both `'text'` and `'xml'` format via the existing `describe_edges()` function.
  - Drops the lxml round-trip (parse then re-serialize) that was only needed to strip the XML declaration -- just concatenates the output of `describe_edges()` directly.

---

## Section 3: Tool discovery (`list_available_tools`)

An agent or user needs a way to discover what analysis tools are available without reading source files. This is a static registry that returns a structured list of all public functions and classes in the `analysis` module, organized by sub-module.

Target file: `analysis/tool_registry.py` (new module), exported via `analysis/__init__.py`.

### Proposed API

```python
def list_available_tools(format: str = 'text') -> str:
    """
    List all public analysis tools available in the analysis module.

    Args:
        format: 'text' for human-readable table, 'dict' for structured data.

    Returns:
        Formatted string listing all tools with one-line descriptions.
    """
```

### Static registry (current state of the analysis module)

#### Module: `analysis.glass_geometry` -- Glass boundary and edge analysis

| Name | Kind | Signature | Description |
|---|---|---|---|
| `EdgeType` | Enum | `LINE, CIRCULAR, EQUATION` | Type of edge geometry |
| `EdgeDescription` | dataclass | `index, edge_type, p1, p2, midpoint, length, short_label, long_label` | Describes a single edge of a glass object |
| `GlassInterface` | dataclass | `geometry, glass1, glass2, n1, n2` | Shared edge between two adjacent glass objects |
| `GlassBoundary` | dataclass | `geometry, glass, n` | Full boundary polygon of a glass object |
| `SceneGeometryAnalysis` | dataclass | `boundaries, interfaces, exterior_edges` | Complete geometric analysis of a scene's glass objects |
| `get_edge_descriptions` | function | `(glass) -> List[EdgeDescription]` | Get detailed descriptions of all edges in a glass object |
| `describe_edges` | function | `(glass, format='text', show_coordinates=True) -> str` | Generate a formatted description of all edges of a glass object |
| `glass_to_polygon` | function | `(glass) -> Polygon` | Convert a Glass object's path to a Shapely Polygon |
| `analyze_scene_geometry` | function | `(scene) -> SceneGeometryAnalysis` | Analyze all glass objects in a scene and extract geometric relationships |

#### Module: `analysis.saving` -- Ray data export, filtering, and statistics

| Name | Kind | Signature | Description |
|---|---|---|---|
| `save_rays_csv` | function | `(ray_segments, output_path, filename='rays.csv', ...) -> Path` | Export ray segment data to a CSV file |
| `rays_to_xml` | function | `(ray_segments, precision_coords=4, precision_brightness=6) -> str` | Export ray segment data to an XML string |
| `filter_tir_rays` | function | `(ray_segments, tir_only=True) -> List[Ray]` | Filter rays based on TIR status |
| `filter_grazing_rays` | function | `(ray_segments, grazing_only=True, criterion=None) -> List[Ray]` | Filter rays based on grazing incidence status |
| `get_ray_statistics` | function | `(ray_segments) -> dict` | Compute statistics about a collection of ray segments |

#### Module: `analysis.simulation_result` -- Simulation result container

| Name | Kind | Signature | Description |
|---|---|---|---|
| `SceneSnapshot` | dataclass | `uuid, name, object_count, ...` | Captures the state of a scene at a point in time |
| `SimulationResult` | dataclass | `uuid, name, timestamp, segments, lineage, ...` | Container for simulation results with full context |
| `describe_simulation_result` | function | `(result, format='xml', include_segments=False, max_segments=100) -> str` | Generate a formatted description of a simulation result |

Key `SimulationResult` methods:
- `create(scene, segments, max_rays, processed_ray_count, ...) -> SimulationResult` -- factory method
- `get_wavelength_groups() -> Dict[Optional[float], List[Ray]]` -- group segments by wavelength
- `get_source_groups() -> Dict[Optional[str], List[Ray]]` -- group segments by source UUID
- `get_label_groups() -> Dict[Optional[str], List[Ray]]` -- group segments by source label
- `get_rays_by_wavelength(wavelength, tolerance=1.0) -> List[Ray]` -- filter by wavelength
- `get_rays_by_label(label) -> List[Ray]` -- filter by source label
- `get_rays_by_source(source_uuid) -> List[Ray]` -- filter by source UUID
- `get_wavelength_statistics() -> Dict` -- wavelength distribution stats
- `get_source_statistics() -> Dict` -- source distribution stats
- `get_grazing_statistics() -> Dict` -- grazing incidence stats
- `get_tir_statistics() -> Dict` -- TIR stats

#### Module: `analysis.ray_geometry_queries` -- Ray-geometry spatial queries

| Name | Kind | Signature | Description |
|---|---|---|---|
| `get_object_by_name` | function | `(scene, name) -> BaseSceneObj` | Find a scene object by its user-defined name |
| `get_object_by_uuid` | function | `(scene, uuid) -> BaseSceneObj` | Find a scene object by UUID (exact or prefix match) |
| `get_objects_by_type` | function | `(scene, type_name) -> List[BaseSceneObj]` | Find all scene objects of a given type |
| `find_rays_inside_glass` | function | `(segments, glass_obj) -> List[Ray]` | Return rays whose midpoint is inside a glass object |
| `find_rays_crossing_edge` | function | `(segments, glass_obj, edge_label) -> List[Ray]` | Return rays that intersect a specific glass edge |
| `find_rays_by_angle_to_edge` | function | `(segments, glass_obj, edge_label, min_angle=0, max_angle=90, proximity=None) -> List[Ray]` | Return rays within an angle range relative to an edge |
| `find_rays_by_polarization` | function | `(segments, min_dop=0, max_dop=1) -> List[Ray]` | Filter rays by degree of polarization |

#### Module: `analysis.lineage_analysis` -- Post-hoc ray tree analysis

| Name | Kind | Signature | Description |
|---|---|---|---|
| `rank_paths_by_energy` | function | `(lineage, leaf_uuids=None) -> List[Dict]` | Rank optical paths by terminal segment brightness |
| `get_branching_statistics` | function | `(lineage) -> Dict` | Analyze ray tree branching patterns |
| `detect_tir_traps` | function | `(lineage, min_tir_count=2) -> List[Dict]` | Find ray subtrees trapped by repeated TIR |
| `extract_angular_distribution` | function | `(lineage, leaf_uuids=None) -> List[Dict]` | Compute emission angle for each leaf ray |
| `build_angular_histogram` | function | `(angular_data, n_bins=36, weight_by_energy=True) -> Dict` | Build histogram of emission angles |
| `check_energy_conservation` | function | `(lineage) -> Dict` | Verify energy conservation at each branching point |

### Implementation approach

Start with a **static registry**: a function that returns the tables above as a formatted string or structured data. The registry is a hardcoded list of `(module, name, kind, signature, description)` tuples maintained alongside the code.

A future improvement could use introspection (walk `analysis.__all__`, extract first docstring lines) to stay in sync automatically, but the static approach is simpler to get right and easier for an agent to parse.
