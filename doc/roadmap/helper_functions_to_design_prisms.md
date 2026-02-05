# A New Module for Prisms

## Introduction

We want to build a new top-level module `optical_elements` (alongside existing `core` and `analysis`) with `prisms` as its first sub-module. The module provides **convenience constructors** that compute vertex geometry from physical parameters (apex angle + characteristic size) so users never have to specify raw vertex coordinates for standard prism types.

### Design philosophy: LLM-agent-in-the-loop

A core design goal is that **LLM agents will participate in the optical design workflow**. The geometric representation of the design space is communicated through language — when a user prompts *"the edge of the prism that is facing the lens on its left"*, the agent must resolve that natural-language reference to a specific edge.

This drives the **dual-labeling architecture** described in the labeling section below: every edge carries both a *functional* label (what it does optically) and a *cardinal* label (where it points in the scene). The agent uses cardinal labels for spatial reasoning and functional labels for optical reasoning.

### Overview of prisms

Here is an overview of common prism types and their characteristics:

1. **Right-Angle Prism (90-45-45)**
   Simplest prism design with 45°-90°-45° angles.
   Uses total internal reflection (TIR) at the hypotenuse.
   Deflects light by 90° or 180° (depending on orientation).
   Often used as a mirror substitute.

2. **Dove Prism**
   Truncated right-angle prism (long, trapezoidal shape).
   Light enters/exits through angled end faces.
   Inverts image vertically without deviation.
   Rotating the prism rotates the image at 2× the prism rotation rate.
   Used in beam rotators and image manipulation.

3. **Amici (Roof) Prism**
   Right-angle prism with a "roof" edge (90° ridge) on the hypotenuse.
   Inverts image in both axes (180° rotation).
   Common in binoculars and telescopes for image erecting.
   Requires very precise roof angle (tolerance ~2 arcseconds).
   **Note:** Inherently 3D — in 2D the roof ridge cannot be represented. A 2D implementation would approximate it as a standard right-angle prism with an explanatory note. **Deferred to a later phase.**

4. **Retro-Reflector (Corner Cube)**
   3 mutually perpendicular surfaces.
   Returns light exactly parallel to incident direction.
   Used in surveying, laser ranging, bicycle reflectors.
   Works over wide acceptance angle.
   **Note:** Inherently 3D geometry. In 2D it reduces to two mirrors at 90°, which is a compound element, not a single prism. **Deferred / out of scope for initial release.**

5. **Coupling Prism**
   Used to couple light into/out of waveguides or thin films.
   Exploits evanescent wave coupling or frustrated TIR.
   Common types: Kretschmann and Otto configurations for SPR.

6. **Ophthalmic Prism**
   Low-angle wedge prisms for vision correction.
   Measured in prism diopters (1Δ = 1 cm deviation at 1 m).
   Used to correct strabismus, diplopia, or convergence issues.
   Often incorporated into spectacle lenses.

7. **Dispersing Prisms**
   - Equilateral (60-60-60): classic dispersion/spectrum demonstration
   - Littrow: 30-60-90, used at minimum deviation for one wavelength
   - Pellin-Broca: constant 90° deviation, wavelength-selectable output (4-sided: 90-75-135-60)

8. **Anamorphic Prism Pairs**
   Two prisms that expand/compress beam in one axis.
   Used to circularize elliptical laser diode beams.
   Magnification depends on angles and orientation.

9. **Wedge Prism**
   Small apex angle, deviates beam by small fixed angle.
   Used for beam steering, riflescope adjustments.
   Rotating wedge pairs create variable deviation (Risley prisms).

### Fixed-angle vs. variable-angle prisms

Most standard prisms have **fixed angles by definition** — the only free geometric parameter is a single characteristic length:

| Prism type         | Angles (fixed)      | Free parameter    |
|--------------------|---------------------|--------------------|
| Right-Angle        | 45-90-45            | `leg_length`       |
| Equilateral        | 60-60-60            | `side_length`      |
| Littrow            | 30-60-90            | `side_length`      |
| Pellin-Broca       | 90-75-135-60        | `char_length`      |

Variable-angle prisms require explicit angle parameters:

| Prism type         | Variable angles      | Free parameters              |
|--------------------|----------------------|------------------------------|
| Wedge              | `apex_angle` (small) | `apex_angle`, `size`         |
| Refractometer      | computed from physics | `n_prism`, `n_sample_range`, `size` |
| Dove               | end face angle       | `length`, `height`, `end_angle` |

The `BasePrism` API should reflect this distinction: fixed-angle subclasses need only a `from_size()` constructor, while variable-angle subclasses expose angle parameters.

---

### Coding style conventions

All new and refactored code in this roadmap must follow these style rules:

- **Full type annotations** on every function parameter and return type. Avoid `Any` — use specific types, `Union`, `Optional`, or protocol classes instead. If a parameter can accept multiple types, spell them out explicitly (e.g. `Union[str, int]` not `Any`).
- **Methods that perform actions should return `bool`**: class methods that execute an operation (drawing, saving, modifying state) should return `True` on success. This gives callers a way to check completion without relying on exception-only signaling. Example:
  ```python
  def draw_ray_segment(self, ray: 'Ray', color: str = 'red', ...) -> bool:
      # ... drawing logic ...
      self.layer_rays.add(line)
      return True
  ```
- **Pure query methods** (getters, computations) return their natural type — no `bool` wrapping needed.
- Use `TYPE_CHECKING` blocks for imports that are only needed for annotations, to avoid circular imports at runtime.

---

## Dual-Labeling Architecture

### Motivation

Edge labels in this project serve two distinct audiences performing different kinds of reasoning:

1. **Optical reasoning** — *"apply AR coating to the entrance face"*, *"which edge acts as the TIR surface?"*
   Requires labels that describe what an edge **does** in the optical path. These are **functional labels**, determined by the prism type and invariant under rotation.

2. **Spatial / scene-layout reasoning** — *"the edge facing the lens on the left"*, *"rotate the prism so that the beam enters from the west"*
   Requires labels that describe **where** an edge points in the scene coordinate system. These are **cardinal labels**, computed from the edge midpoint's angle relative to the centroid (the existing `auto_label_cardinal()` system in `BaseGlass`).

An LLM agent resolving the prompt *"point the entrance face toward the lens on the left"* needs **both** systems simultaneously:
1. "the lens on the left" → spatial reasoning → the lens is to the West
2. "entrance face" → functional reasoning → edge index 0 of this prism type
3. Compute the rotation that makes edge 0's cardinal label become "W"

### Label layers on `BasePrism`

Every `BasePrism` edge carries **two label pairs** (short + long for each layer):

| Layer       | Set when          | Changes on rotation? | Example (short / long)        |
|-------------|-------------------|-----------------------|-------------------------------|
| Functional  | At construction   | No                    | `"E"` / `"Entrance Face"`    |
| Cardinal    | After placement   | Yes (must recompute)  | `"W"` / `"West Edge"`        |

**Important:** The functional short labels (`E`, `H`, `X`, `M`, `B`, …) are a separate namespace from the cardinal short labels (`N`, `S`, `E`, `W`, …). The overlap of `"E"` (Entrance) with `"E"` (East) is intentional — they live in different layers and are disambiguated by context (functional vs. cardinal query).

### Functional labels by prism type

Each `BasePrism` subclass defines a class-level `_edge_roles` list that maps edge indices (in path order) to functional labels. Path vertices are always ordered **counterclockwise** (Shapely exterior ring convention), and edge `i` connects vertex `i` to vertex `(i+1) % n`.

#### RightAnglePrism (3 edges)
```
Vertex layout (before rotation):

    V2 (right angle, 90°)
    |\
    | \
    |  \   ← Edge 1: Hypotenuse (H)
    |   \
    |    \
    V0----V1
      Edge 0: Entrance Face (E)

Edge 2 (V2→V0): Exit Face (X)
```

| Edge | Short | Long             | Notes                          |
|------|-------|------------------|--------------------------------|
| 0    | `E`   | Entrance Face    | One of the two leg faces       |
| 1    | `H`   | Hypotenuse       | TIR surface (when n ≥ √2)     |
| 2    | `X`   | Exit Face        | The other leg face             |

| Vertex | Short | Long          |
|--------|-------|---------------|
| 0      | `A1`  | Acute Vertex 1 |
| 1      | `A2`  | Acute Vertex 2 |
| 2      | `R`   | Right-Angle Vertex |

#### EquilateralPrism (3 edges)
```
Vertex layout (before rotation):

        V2 (apex, top)
       / \
      /   \
     /     \
    V0-----V1
      Base (B)
```

| Edge | Short | Long             | Notes                              |
|------|-------|------------------|------------------------------------|
| 0    | `B`   | Base             | Bottom edge, opposite the apex     |
| 1    | `X`   | Exit Face        | Right refracting surface           |
| 2    | `E`   | Entrance Face    | Left refracting surface            |

| Vertex | Short | Long         |
|--------|-------|--------------|
| 0      | `BL`  | Base Left    |
| 1      | `BR`  | Base Right   |
| 2      | `A`   | Apex         |

**Convention note:** In an equilateral prism, "Entrance" and "Exit" are symmetric — light can enter from either refracting face. The labels above assume the canonical orientation (light enters from the left). Calling `prism.swap_entrance_exit()` could flip the E/X assignment if needed.

#### DovePrism (4 edges)
```
Vertex layout (before rotation):

    V3---------V2
     \          /
      \        /    ← Edge 2 (V2→V3 is NOT an edge;
       \      /        edges go 0→1→2→3→0)
    V0----------V1

Edges in CCW order:
  Edge 0 (V0→V1): Base (TIR surface)
  Edge 1 (V1→V2): Exit Face
  Edge 2 (V2→V3): Top
  Edge 3 (V3→V0): Entrance Face
```

| Edge | Short | Long             | Notes                          |
|------|-------|------------------|--------------------------------|
| 0    | `B`   | Base (TIR)       | Long bottom face, TIR surface  |
| 1    | `X`   | Exit Face        | Angled end face                |
| 2    | `T`   | Top              | Short top face (truncation)    |
| 3    | `E`   | Entrance Face    | Angled end face                |

#### WedgePrism (3 or 4 edges, depending on representation)

For a triangular wedge (simplest):

| Edge | Short | Long             | Notes                          |
|------|-------|------------------|--------------------------------|
| 0    | `B`   | Base             | Thick bottom edge              |
| 1    | `X`   | Exit Face        | Refracting surface             |
| 2    | `E`   | Entrance Face    | Refracting surface             |

For a truncated wedge (trapezoid):

| Edge | Short | Long             | Notes                          |
|------|-------|------------------|--------------------------------|
| 0    | `B`   | Base             | Thick bottom edge              |
| 1    | `X`   | Exit Face        | Refracting surface             |
| 2    | `A`   | Apex Edge        | Thin top edge (truncated apex) |
| 3    | `E`   | Entrance Face    | Refracting surface             |

#### RefractometerPrism (4 edges, symmetric trapezoid)

| Edge | Short | Long               | Notes                              |
|------|-------|--------------------|------------------------------------|
| 0    | `B`   | Base               | Bottom edge                        |
| 1    | `X`   | Exit Face          | Refracting surface                 |
| 2    | `M`   | Measuring Surface  | Sample contact / TIR boundary      |
| 3    | `E`   | Entrance Face      | Refracting surface                 |

#### PellinBrocaPrism (4 edges)

| Edge | Short | Long                 | Notes                          |
|------|-------|----------------------|--------------------------------|
| 0    | `E`   | Entrance Face        | 90° angle corner               |
| 1    | `R1`  | First Reflecting     | Internal reflecting surface    |
| 2    | `X`   | Exit Face            | 90° deviation output           |
| 3    | `R2`  | Second Reflecting    | Internal reflecting surface    |

### Implementation

```python
class BasePrism(Glass):
    """Base class for all prisms with convenient constructors."""

    # Subclasses override these class variables
    _edge_roles: ClassVar[List[Tuple[str, str]]] = []     # [(short, long), ...]
    _vertex_roles: ClassVar[List[Tuple[str, str]]] = []   # [(short, long), ...]

    def _apply_functional_labels(self) -> None:
        """
        Apply optical-function labels to edges.
        Stored in a separate attribute from the cardinal labels in BaseGlass.
        """
        self._functional_labels: Dict[int, Tuple[str, str]] = {}
        for i, (short, long) in enumerate(self._edge_roles):
            self._functional_labels[i] = (short, long)

    def _apply_vertex_labels(self) -> None:
        """Apply optical-function labels to vertices."""
        self._vertex_labels: Dict[int, Tuple[str, str]] = {}
        for i, (short, long) in enumerate(self._vertex_roles):
            self._vertex_labels[i] = (short, long)

    def get_functional_label(self, edge_index: int) -> Optional[Tuple[str, str]]:
        """Return (short, long) functional label for an edge."""
        return self._functional_labels.get(edge_index)

    def find_edge_by_functional_label(self, short_label: str) -> Optional[int]:
        """Return edge index for a functional short label (e.g., 'E', 'H')."""
        for i, (s, _) in self._functional_labels.items():
            if s == short_label:
                return i
        return None

    def get_vertex_label(self, vertex_index: int) -> Optional[Tuple[str, str]]:
        """Return (short, long) label for a vertex."""
        return self._vertex_labels.get(vertex_index)

    def find_vertex_by_label(self, short_label: str) -> Optional[int]:
        """Return vertex index for a short label (e.g., 'A' for Apex)."""
        for i, (s, _) in self._vertex_labels.items():
            if s == short_label:
                return i
        return None

    def label_summary(self) -> str:
        """
        Return a text summary combining functional and cardinal labels.
        Useful for LLM agents to understand the full edge semantics.

        Example output:
          Edge 0: Entrance Face (E) | facing West (W)
          Edge 1: Hypotenuse (H)    | facing North-East (NE)
          Edge 2: Exit Face (X)     | facing South (S)
        """
        lines: List[str] = []
        for i in range(self._get_edge_count()):
            func = self._functional_labels.get(i, (str(i), str(i)))
            card = self.get_edge_label(i)  # cardinal, from BaseGlass
            card_short = card[0] if card else "?"
            card_long = card[1] if card else "?"
            lines.append(
                f"Edge {i}: {func[1]} ({func[0]}) | facing {card_long} ({card_short})"
            )
        return "\n".join(lines)
```

### Cardinal labels: when to compute

Cardinal labels depend on the prism's **orientation in the scene**. They should be computed (or recomputed) in these situations:
- After initial placement (`__init__` or `set_position`)
- After rotation changes
- On demand via `auto_label_cardinal()` (already exists in `BaseGlass`)

`BasePrism.__init__` should call both `_apply_functional_labels()` and `auto_label_cardinal()` at the end of construction, so that both layers are populated from the start.

### Agent-facing API: `describe_prism()`

For LLM agents, a single function should return all label information in a structured format (extending the existing `describe_edges()` in `analysis/glass_geometry.py`):

```python
def describe_prism(prism: BasePrism, format: str = 'text') -> str:
    """
    Return a description of the prism that includes:
    - Prism type and key parameters (apex angle, size, refractive index)
    - For each edge: functional label, cardinal label, length, endpoints
    - For each vertex: label, coordinates
    """
    ...
```

This gives an LLM agent a complete semantic map of the prism in one call.

---

## Foundation Structure

### Directory layout

```
ray_tracing_shapely/
├── core/                    # existing
├── analysis/                # existing
├── optical_elements/        # NEW TOP MODULE
│   ├── __init__.py
│   ├── base_optical_element.py   # optional: shared utils for optical elements
│   ├── prisms/              # SUB-MODULE
│   │   ├── __init__.py      # factory functions + re-exports
│   │   ├── base_prism.py    # BasePrism class (dual labeling, _compute_path)
│   │   ├── prism_utils.py   # deviation, dispersion, geometry utilities
│   │   ├── tir_utils.py     # critical angle, refractometer design utilities
│   │   ├── right_angle.py   # 45-90-45 prism
│   │   ├── equilateral.py   # 60-60-60 prism
│   │   ├── refractometer.py # Refractometer / measuring prism
│   │   ├── dove.py          # Dove prism
│   │   ├── wedge.py         # Wedge / Risley prisms
│   │   ├── pellin_broca.py  # Pellin-Broca prism
│   │   ├── coupling.py      # Coupling prism
│   │   └── ophthalmic.py    # Ophthalmic wedge prism
│   └── lenses/              # future sub-module
```

### `BasePrism` class

```python
class BasePrism(Glass):
    """Base class for all prisms with convenient constructors."""

    # --- Class variables (overridden by subclasses) ---
    _edge_roles: ClassVar[List[Tuple[str, str]]] = []
    _vertex_roles: ClassVar[List[Tuple[str, str]]] = []

    # --- Core attributes ---
    apex_angle: float                    # Primary apex angle (degrees)
    size: float                          # Characteristic size (e.g., side length)
    _position: Tuple[float, float]       # Reference point
    _rotation: float                     # Rotation angle (degrees)
    _anchor: str                         # What _position refers to: 'centroid' or 'apex'

    def __init__(
        self,
        scene: 'Scene',
        size: float,
        position: Tuple[float, float] = (0, 0),
        rotation: float = 0,
        n: float = 1.5,
        anchor: str = 'centroid',
        **kwargs: Any
    ) -> None:
        super().__init__(scene)
        self.size = size
        self._position = position
        self._rotation = rotation
        self._anchor = anchor
        self.refIndex = n
        # Build geometry
        self.path = self._compute_path()
        # Apply both label layers
        self._apply_functional_labels()
        self._apply_vertex_labels()
        self.auto_label_cardinal()

    @abstractmethod
    def _compute_path(self) -> List[Dict[str, Union[float, bool]]]:
        """
        Compute vertex path from parameters.

        Contract:
        - Return vertices in counterclockwise order (Shapely convention).
        - Each vertex is {'x': float, 'y': float, 'arc': False}.
        - Edge i connects vertex i to vertex (i+1) % n.
        - The ordering must match _edge_roles and _vertex_roles indices.
        """
        ...

    @classmethod
    def from_vertices(
        cls,
        scene: 'Scene',
        vertices: List[Tuple[float, float]],
        n: float = 1.5
    ) -> 'BasePrism':
        """Create from explicit vertex coordinates (escape hatch)."""
        ...
```

### Anchor point convention

The `position` parameter needs a defined reference point. Different use cases favor different anchors:
- `anchor='centroid'` (default): `position` is the geometric centroid of the polygon. Intuitive for scene layout.
- `anchor='apex'`: `position` is the apex vertex. Useful when positioning a prism tip at a known point.

Each subclass documents which vertex is the "apex" for `anchor='apex'` mode.

---

## Phase 1: Foundation

**Goal:** Directory structure, `BasePrism` base class, dual-labeling infrastructure.

Tasks:
- Create `optical_elements/` directory structure and all `__init__.py` files
- Implement `BasePrism(Glass)` with:
  - `_compute_path()` abstract method
  - `_edge_roles` / `_vertex_roles` class variables
  - `_apply_functional_labels()` / `_apply_vertex_labels()` methods
  - Functional label query methods (`get_functional_label`, `find_edge_by_functional_label`)
  - Vertex label query methods (`get_vertex_label`, `find_vertex_by_label`)
  - `label_summary()` method for combined functional + cardinal output
  - `from_vertices()` escape-hatch constructor
  - `anchor` parameter support (`'centroid'` / `'apex'`)

---

## Phase 2: Core Utilities

**Goal:** Prism-specific calculation functions, separate from the prism geometry classes.

### `prism_utils.py` — deviation and dispersion

```python
def minimum_deviation(apex_angle_deg: float, n: float) -> float:
    """
    Minimum deviation angle (degrees) for a prism.
    D_min = 2 * arcsin(n * sin(A/2)) - A
    """
    ...

def refractive_index_from_deviation(apex_angle_deg: float, d_min_deg: float) -> float:
    """
    Compute n from measured minimum deviation.
    n = sin((D_min + A) / 2) / sin(A / 2)
    """
    ...

def deviation_at_incidence(
    apex_angle_deg: float, n: float, theta_i_deg: float
) -> float:
    """
    Total deviation for arbitrary incidence angle.
    Applies Snell's law at both surfaces sequentially.
    Returns NaN if TIR occurs inside the prism.
    """
    ...

def angular_dispersion(
    apex_angle_deg: float, n: float, dn_dlambda: float
) -> float:
    """
    Angular dispersion dD/dλ at minimum deviation.
    dD/dλ = (2 sin(A/2)) / sqrt(1 - n² sin²(A/2)) · dn/dλ
    """
    ...

def resolving_power(base_length: float, dn_dlambda: float) -> float:
    """
    Spectral resolving power R = b · |dn/dλ|
    where b is the prism base length.
    """
    ...
```

### `tir_utils.py` — refractometer design utilities

(Carried over from original roadmap, unchanged)

```python
def critical_angle(n_sample: float, n_prism: float) -> float:
    """Calculate critical angle in radians."""
    if n_sample >= n_prism:
        raise ValueError("n_prism must be > n_sample for TIR")
    return math.asin(n_sample / n_prism)

def prism_angle_for_refractometer(
    n_sample_range: Tuple[float, float],
    n_prism: float,
    margin_deg: float = 5.0
) -> float:
    """Calculate optimal prism angle for a given measurement range."""
    theta_c_max = critical_angle(n_sample_range[1], n_prism)
    return 90 - math.degrees(theta_c_max) + margin_deg

def measurement_range_from_prism(
    prism_angle: float,
    n_prism: float,
    field_angle_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Calculate measurable n_sample range for a given prism."""
    ...
```

### Relationship to existing `analysis/fresnel_utils.py`

The existing `fresnel_utils.py` already provides `critical_angle(n1, n2)` and `brewster_angle(n1, n2)`. The new `tir_utils.py` focuses on **design** utilities (computing prism angles from measurement requirements), while `fresnel_utils.py` provides **analysis** utilities (computing reflection/transmission for given angles). They are complementary. The new module should import and re-use the existing functions where applicable rather than duplicating them.

---

## Phase 3: Prism Implementations (Priority Order)

Each implementation consists of:
1. `_edge_roles` and `_vertex_roles` class variables (functional labels)
2. `_compute_path()` method (vertex geometry from parameters)
3. Any type-specific properties or methods

### Priority 1: `RightAnglePrism`

```python
class RightAnglePrism(BasePrism):
    """45-90-45 right-angle prism."""

    _edge_roles = [
        ("E", "Entrance Face"),
        ("H", "Hypotenuse"),
        ("X", "Exit Face"),
    ]
    _vertex_roles = [
        ("A1", "Acute Vertex 1"),
        ("A2", "Acute Vertex 2"),
        ("R", "Right-Angle Vertex"),
    ]

    def __init__(self, scene: 'Scene', leg_length: float, **kwargs: Any) -> None:
        self.leg_length = leg_length
        super().__init__(scene, size=leg_length, **kwargs)

    def _compute_path(self) -> List[Dict[str, Union[float, bool]]]:
        """
        Vertices (CCW, before rotation/translation):
          V0 = (0, 0)              Acute 1
          V1 = (leg, 0)            Acute 2
          V2 = (0, leg)            Right angle

        Edge 0 (V0→V1): Entrance face (bottom leg)
        Edge 1 (V1→V2): Hypotenuse
        Edge 2 (V2→V0): Exit face (left leg)
        """
        ...
```

**Key parameters:** `leg_length`, `n` (refractive index), `position`, `rotation`, `anchor`.

**Minimum n for TIR:** n ≥ √2 ≈ 1.414 (so that incidence angle of 45° on hypotenuse exceeds the critical angle).

### Priority 2: `EquilateralPrism`

```python
class EquilateralPrism(BasePrism):
    """60-60-60 equilateral dispersing prism."""

    _edge_roles = [
        ("B", "Base"),
        ("X", "Exit Face"),
        ("E", "Entrance Face"),
    ]
    _vertex_roles = [
        ("BL", "Base Left"),
        ("BR", "Base Right"),
        ("A", "Apex"),
    ]
```

**Key parameters:** `side_length`, `n`, `position`, `rotation`, `anchor`.

**Note on E/X symmetry:** In an equilateral prism, entrance and exit faces are interchangeable. The labels assume canonical orientation (light enters from the left). A `swap_entrance_exit()` method could flip the assignments.

### Priority 3: `WedgePrism`

```python
class WedgePrism(BasePrism):
    """Small-angle wedge prism for beam steering."""

    _edge_roles = [
        ("B", "Base"),
        ("X", "Exit Face"),
        ("E", "Entrance Face"),
    ]
```

**Key parameters:** `apex_angle` (small, typically 1°–10°), `size` (base width), `n`, `position`, `rotation`.

This is the first **variable-angle** prism. The constructor takes `apex_angle` explicitly.

### Priority 4: `RefractometerPrism`

```python
class RefractometerPrism(BasePrism):
    """
    Symmetric trapezoidal prism for refractometer applications.
    Prism angles determined by target Brewster angle / sample n range.
    """

    _edge_roles = [
        ("B", "Base"),
        ("X", "Exit Face"),
        ("M", "Measuring Surface"),
        ("E", "Entrance Face"),
    ]
```

This gives users a physics-driven API: specify what you want to measure, and the prism geometry is computed automatically.

#### Core physics

The critical angle $\theta_c$ at the prism–sample interface is given by Snell's law:

$$n_s = n_{prism} \cdot \sin(\theta_c)$$

The prism face angle is cut so that light reflected at the **center** of the expected measurement range exits **perpendicular** (normal) to the exit face. This avoids additional refraction and ghost reflections at the exit surface.

The center critical angle is:

$$\theta_{c,mid} = \arcsin\!\left(\frac{n_{mid}}{n_{prism}}\right), \quad n_{mid} = \frac{n_{min} + n_{max}}{2}$$

The face angle (angle between the measuring surface and each entrance/exit face) is then:

$$\alpha = 90° - \theta_{c,mid}$$

Example: measuring sugar (Brix 0–85%), critical angles range from ~61° to ~70°. The face is cut at ~65° so the center-range beam exits straight into the sensor.

#### Two system architectures: `system_type`

The prism geometry depends on which optical system it sits in. The `system_type` parameter (`'angular'` or `'geometric'`) controls this because the two architectures impose **different constraints on the measuring surface length**.

##### The Focused "Angular" System (Abbe-like)

Standard architecture for high-precision digital refractometers. Components: distributed light source (LED + diffuser) → prism → lens → sensor.

1. **Light source:** A diffuser ensures each point on the measuring surface is illuminated from all angles simultaneously (just like the classical Abbe refractometer).
2. **Lens as Fourier transformer:** Takes all rays leaving the prism at the same angle $\theta$ (regardless of where on the prism they originated) and focuses them to a single row on the sensor: $x = f \cdot \tan(\theta)$.
3. **Shadow line:** The sensor sees a cutoff in the angular domain — bright above the critical angle (TIR), dark below (refraction). The cutoff position encodes $n_{sample}$.

**Key constraint:** Resolution depends on the focal length $f$ (longer $f$ → more pixels per degree). The measuring surface length is a **free parameter** — it just needs to be large enough to collect sufficient light. The prism size is chosen for mechanical convenience, not dictated by the measurement range.

##### The Lensless "Geometric" System

Compact architecture with no lens. Components: point source → prism → sensor (direct imaging).

1. **Light source:** A point source (laser or small LED) without collimating optics. Because light originates from a single point, each position on the measuring surface is hit at a **unique angle** determined by the geometry.
2. **Spatial angle mapping:** The incidence angle varies along the measuring surface:
   - Ray A hits the left side at 60°
   - Ray B hits the center at 65°
   - Ray C hits the right side at 70°
3. **Physical TIR boundary:** If the critical angle is 68°, one side of the measuring surface reflects (TIR), the other transmits. The transition is a physical spatial boundary on the prism face.
4. **Detection:** The sensor directly images the prism face. The light/dark transition position maps back to a specific angle, hence a specific $n_{sample}$.

**Key constraint:** The position on the sensor is $x = L \cdot \tan(\theta)$ where $L$ is the path length from prism to sensor. There is no lens to magnify or compress — the geometry is 1:1 (modified by distance). This means:
- **Resolution** requires large $L$ or small pixel size (to distinguish $n = 1.330$ from $n = 1.331$).
- **Measurement range** requires a physically long measuring surface. To capture angles from $\theta_{c,min}$ to $\theta_{c,max}$, the source must illuminate the full angular span across the measuring surface. If the prism face is too short, the range is clipped.
- **The measuring surface length is NOT a free parameter** — it is dictated by the source distance, the angular range, and the sensor size.

#### Constructor strategy

The two system types produce the same prism shape (symmetric trapezoid) but differ in how the measuring surface length is determined. This calls for **three constructors**:

##### Constructor 1: `__init__` (primary, physics-driven)

```python
def __init__(
    self,
    scene: 'Scene',
    n_prism: float,
    n_sample_range: Tuple[float, float],  # (n_min, n_max)
    measuring_surface_length: float,
    system_type: str = 'angular',         # 'angular' or 'geometric'
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0
) -> None:
```

Works for both system types. The face angle $\alpha$ is computed from `n_sample_range` center (normal-exit condition). The user provides `measuring_surface_length` explicitly:
- For `system_type='angular'`: this is a free choice (mechanical convenience).
- For `system_type='geometric'`: the user is responsible for choosing a length consistent with their source/sensor geometry. A validation method `validate_geometric_system()` can warn if the length appears too short for the requested range.

Stored attributes:
- `self.n_prism`, `self.n_sample_range`, `self.system_type`
- `self.face_angle` (computed $\alpha$)
- `self.theta_c_range` (computed critical angle range)

##### Constructor 2: `from_apex_angle` (explicit geometry, escape hatch)

```python
@classmethod
def from_apex_angle(
    cls,
    scene: 'Scene',
    face_angle_deg: float,
    n_prism: float,
    measuring_surface_length: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0
) -> 'RefractometerPrism':
    """
    Create from an explicit face angle, bypassing the physics derivation.

    The face_angle is the angle between the measuring surface (M) and each
    entrance/exit face (E, X). The full prism geometry is determined:
      - Measuring surface length is given
      - Entrance and exit faces are symmetric, angled at face_angle from M
      - Base length is computed from measuring_surface_length and face_angle

    Use case: the user already knows the exact prism geometry (e.g., from
    a manufacturer datasheet or a prior design iteration).
    """
    # Back-compute the n_sample_range this geometry corresponds to:
    #   theta_c_mid = 90° - face_angle
    #   n_mid = n_prism * sin(theta_c_mid)
    # Store as informational metadata (not used for geometry).
    ...
```

##### Constructor 3: `from_geometric_constraints` (lensless-system-specific)

```python
@classmethod
def from_geometric_constraints(
    cls,
    scene: 'Scene',
    n_prism: float,
    n_sample_range: Tuple[float, float],
    source_distance: float,            # distance from point source to measuring surface center
    sensor_length: float,              # physical length of the linear sensor array
    path_length: float,                # distance from prism exit face to sensor
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0
) -> 'RefractometerPrism':
    """
    Compute both face angle AND measuring surface length from physical
    system constraints. Specific to the lensless "geometric" architecture.

    The constructor:
    1. Computes the face angle from n_sample_range center (normal-exit condition),
       same as the primary constructor.
    2. Computes the required measuring surface length from the angular range
       and source distance:
         - theta_c_min = arcsin(n_min / n_prism)
         - theta_c_max = arcsin(n_max / n_prism)
         - The source must illuminate the measuring surface across the full
           [theta_c_min, theta_c_max] span. The required length depends on the
           source distance and the angular subtense.
    3. Validates sensor coverage: checks that sensor_length and path_length
       are sufficient to capture the full reflected angular range:
         - required_sensor = path_length * (tan(theta_out_max) - tan(theta_out_min))
         - Warns if sensor_length < required_sensor (range will be clipped)

    Stored attributes (in addition to base):
      - self.source_distance
      - self.sensor_length, self.path_length
      - self.computed_measuring_length  (the derived measuring surface length)

    Raises:
        ValueError: if n_prism <= n_max (TIR impossible)
        ValueError: if source_distance <= 0 or path_length <= 0
    """
    ...
```

This constructor is the "tell me your hardware, I'll design the prism" API. The user specifies their point source distance, sensor size, and path length, and the prism geometry is fully determined — both angles and dimensions.

##### Query methods on `RefractometerPrism`

Regardless of which constructor was used, the instance exposes these query methods:

```python
def critical_angle_for(self, n_sample: float) -> float:
    """Return critical angle (degrees) for a given sample index."""
    ...

def exit_angle_for(self, n_sample: float) -> float:
    """
    Return the angle (degrees from normal) at which light exits the
    exit face for a given sample's critical angle reflection.
    Returns 0.0 when n_sample = n_mid (normal exit by design).
    """
    ...

def sensor_position_for(self, n_sample: float) -> float:
    """
    (Geometric system only) Return the expected sensor position (in length
    units) for a given sample index, based on stored path_length.
    Raises ValueError if system_type != 'geometric' or path_length not set.
    """
    ...

def validate_geometric_system(self) -> List[str]:
    """
    Return a list of warning messages if the current geometry has issues:
    - Measuring surface too short for the angular range
    - Sensor too small to capture full range
    - Exit angles too far from normal (ghost reflection risk)
    Returns empty list if everything looks good.
    """
    ...
```

### Priority 5: `DovePrism`

```python
class DovePrism(BasePrism):
    """Truncated right-angle prism (trapezoidal cross-section)."""

    _edge_roles = [
        ("B", "Base (TIR)"),
        ("X", "Exit Face"),
        ("T", "Top"),
        ("E", "Entrance Face"),
    ]
```

**Key parameters:** `length`, `height`, `end_angle` (angle of the entrance/exit faces), `n`, `position`, `rotation`.

### Priority 6: `PellinBrocaPrism`

```python
class PellinBrocaPrism(BasePrism):
    """4-sided prism with 90-75-135-60 angles. Constant 90° deviation."""

    _edge_roles = [
        ("E", "Entrance Face"),
        ("R1", "First Reflecting"),
        ("X", "Exit Face"),
        ("R2", "Second Reflecting"),
    ]
```

**Key parameters:** `char_length` (characteristic aperture), `n`, `position`, `rotation`.

### Deferred

- **AmiciPrism** — requires 3D roof geometry; 2D approximation is just a `RightAnglePrism`. Defer.
- **RetroReflector** — inherently 3D (corner cube). In 2D it's two mirrors at 90°, which is a compound scene setup, not a single prism class. Defer / out of scope.
- **CouplingPrism** — depends on waveguide/thin-film coupling physics not yet in the simulator. Defer.
- **AnamorphicPrismPair** — compound element (two prisms). Could be a factory function that returns a pair of `WedgePrism` objects. Defer to after single-prism types are stable.

---

## Phase 4: Factory Functions and Convenience API

```python
# In optical_elements/prisms/__init__.py

def right_angle_prism(
    scene: 'Scene',
    leg_length: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0,
    n: float = 1.5
) -> RightAnglePrism:
    ...

def equilateral_prism(
    scene: 'Scene',
    side_length: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0,
    n: float = 1.5
) -> EquilateralPrism:
    ...

def wedge_prism(
    scene: 'Scene',
    apex_angle: float,
    size: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0,
    n: float = 1.5
) -> WedgePrism:
    ...

def refractometer_prism(
    scene: 'Scene',
    n_prism: float,
    n_sample_range: Tuple[float, float],
    size: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0
) -> RefractometerPrism:
    ...

def dove_prism(
    scene: 'Scene',
    length: float,
    height: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0,
    n: float = 1.5
) -> DovePrism:
    ...

def pellin_broca_prism(
    scene: 'Scene',
    char_length: float,
    position: Tuple[float, float] = (0, 0),
    rotation: float = 0,
    n: float = 1.5
) -> PellinBrocaPrism:
    ...
```

### Integration with `describe_prism()`

Add to `analysis/glass_geometry.py` (or a new `analysis/prism_geometry.py`):

```python
def describe_prism(prism: 'BasePrism', format: str = 'text') -> str:
    """
    Structured description of a prism for LLM agents.
    Combines functional labels, cardinal labels, geometry, and physics.
    """
    ...
```

---

## Phase 5: Testing

### Geometry verification
- For each prism type: verify vertex positions, internal angles, edge lengths, and polygon area against analytical formulas.
- Verify counterclockwise vertex ordering (positive signed area).

### Label correctness
- For each prism type: verify `_edge_roles` count matches vertex count.
- Verify `get_functional_label(i)` returns expected values for all edges.
- Verify `auto_label_cardinal()` produces sensible directions for default orientation.
- Verify that after 90° rotation, cardinal labels shift accordingly but functional labels remain unchanged.

### Integration tests
- Create prism → add to scene → run simulation → verify:
  - TIR occurs on expected face (e.g., hypotenuse of right-angle prism)
  - Refraction occurs on entrance/exit faces
  - Deviation angle matches `minimum_deviation()` prediction for equilateral prism at symmetric incidence

### Utility function tests
- `minimum_deviation()`: compare against known values (e.g., equilateral prism with n=1.5 → D_min ≈ 37.2°)
- `refractive_index_from_deviation()`: round-trip test with `minimum_deviation()`
- `deviation_at_incidence()`: verify symmetry at minimum deviation, verify TIR detection
- `critical_angle()` and `prism_angle_for_refractometer()`: verify against hand calculations

---

## Phase 6: Examples

### Example 1: Classic dispersion (equilateral prism + white light)
White beam → equilateral prism → rainbow fan of exit rays. Demonstrates Cauchy dispersion and functional labels.

### Example 2: TIR demo (right-angle prism)
Beam entering through entrance face → TIR at hypotenuse → exit through exit face at 90°. Shows functional labels corresponding to optical behavior.

### Example 3: Refractometer simulation
`RefractometerPrism` created from `n_sample_range`. Rays at various angles demonstrate TIR boundary on the measuring surface.

### Example 4: Minimum deviation measurement
Equilateral prism with a beam at the minimum deviation angle. Compare simulated deviation against `minimum_deviation()` prediction.

### Example 5: Dual-labeling demo
Create a prism, rotate it to several orientations. Print `label_summary()` at each orientation to show functional labels staying fixed while cardinal labels update.

---

## Future Considerations (Not in Scope)

- **Sellmeier dispersion model:** The Cauchy model (`n = A + B/λ²`) loses accuracy outside the visible range. An optional Sellmeier model in `BaseGlass` would improve broadband simulations. This is a `core` enhancement, not prism-specific.
- **3D prism types:** Amici roof, corner cube, pentaprism with coated faces — require a 3D extension or explicit documentation of 2D limitations.
- **Compound elements:** Anamorphic prism pairs, Wollaston/Rochon polarizers (require birefringence support), Amici direct-vision prisms (cemented multi-element). These depend on features not yet in the simulator (birefringence, cemented interfaces).
- **Material database integration:** Connecting to refractiveindex.info or Schott glass catalog for real material data (e.g., via PyOptik library).
