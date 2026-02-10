# New methods to build up a RefractometerPrism

In this document we will describe the new methods that we will add to the `RefractometerPrism` class to allow users to create custom prisms.

## `from_critical_ray_path` — Derive prism geometry from TIR angle and ray path length

Vertex layout (before rotation):

            M
    V3---------V2
     \          /
   E  \        / X   ← Edge 2 (V2→V3 is NOT an edge;
       \      /        edges go 0→1→2→3→0)
       V0----V1
          B

    Edge 0: Base (B) | facing South Edge (S)
    Edge 1: Exit Face (X) | facing East Edge (E)
    Edge 2: Measuring Surface (M) | facing North Edge (N)
    Edge 3: Entrance Face (E) | facing West Edge (W)

### Description

The prism is a symmetric trapezoidal prism with the base (B) facing south, the exit face (X) facing east, the measuring surface (M) facing north and the entrance face (E) facing west. The prism is symmetric with respect to the vertical plane that bisects the base and the measuring surface. This means that the angle between the entrance face and the base equals the angle between the exit face and the base, and that the entrance and exit faces have equal length.

The key idea of this construction method is that **all prism dimensions are fully determined by just two inputs**: the critical angle at the measuring surface (dictated by `n_prism` and `n_target`) and the desired total ray path length `L` inside the prism.

We design the prism so that a ray entering at the midpoint of the entrance face (E), perpendicular to it, travels through the prism, hits the midpoint of the measuring surface (M) at exactly the critical angle for TIR, reflects, and exits at the midpoint of the exit face (X), perpendicular to it. Because the ray enters and exits perpendicular to the entrance and exit faces, it undergoes no refraction at those surfaces. The incidence at the critical angle at the measuring surface ensures best resolution around the TIR transition for the target sample.

By symmetry the ray path has two equal legs of length `L/2`: one from E midpoint to M midpoint, and one from M midpoint to X midpoint. The midpoint of E sits at height `h/2` (since E spans from height 0 to height h), and the midpoint of M sits at height `h`. Therefore the vertical component of each leg is `h/2`.

**Validity constraint:** The base length `B` is positive only when `theta_c > 45°`. For `theta_c = 45°` the shape degenerates into a triangle (`B = 0`), and for `theta_c < 45°` the geometry is invalid. In typical refractometer applications (e.g. glass prism `n = 1.72` measuring water `n = 1.33`, giving `theta_c ≈ 50.7°`), this constraint is naturally satisfied.

### Derivation

If we call `n_prism` the refractive index of the prism and `n_target` the refractive index of the target sample, the critical angle is:

    theta_c = arcsin(n_target / n_prism)

Given `L` (total ray path length from E midpoint → M midpoint → X midpoint), each leg has length `L/2`. The angle of incidence at M (measured from the surface normal, which is vertical) equals `theta_c`. From the right triangle formed by one leg:

**Height of the prism:**

    h/2 = (L/2) * cos(theta_c)
    h = L * cos(theta_c)

**Length of the entrance face (= exit face by symmetry):**

The entrance face E has its endpoints at heights 0 and h, with the ray direction perpendicular to E. From the relationship `|E| = h / sin(theta_c)`:

    E = L * cos(theta_c) / sin(theta_c)

**Length of the measuring surface:**

The three quantities M, L, and E satisfy a Pythagorean relation (`M` is the hypotenuse):

    M^2 = L^2 + E^2
    M = L / sin(theta_c)

**Horizontal overhang of each slanted edge:**

Each slanted edge (E or X) forms a right triangle with the height h as one leg:

    b = sqrt(E^2 - h^2)

**Length of the base:**

    B = M - 2*b

### Numerical example

For `n_prism = 1.72`, `n_target = 1.33` (water), `L = 20`:

    theta_c = arcsin(1.33 / 1.72) = 50.66°
    h   = 20 * cos(50.66°)              = 12.69
    E   = 20 * cos(50.66°) / sin(50.66°) = 16.42
    M   = 20 / sin(50.66°)              = 25.88
    b   = sqrt(16.42^2 - 12.69^2)       = 10.42
    B   = 25.88 - 2 * 10.42             =  5.04

### Implementation instructions

#### 1. Add a helper function to `tir_utils.py`

Add a function that computes the prism dimensions from `n_prism`, `n_target`, and `L`:

```python
def trapezoid_from_critical_ray_path(
    n_target: float,
    n_prism: float,
    ray_path_length: float
) -> dict:
    """
    Compute symmetric trapezoid dimensions from TIR critical angle
    and total internal ray path length.

    Args:
        n_target: Refractive index of the target sample.
        n_prism: Refractive index of the prism (must be > n_target).
        ray_path_length: Total ray path length (L) inside the prism.

    Returns:
        Dict with keys: 'theta_c_deg', 'h', 'E', 'M', 'B', 'b',
                         'face_angle_deg'.

    Raises:
        ValueError: If n_prism <= n_target (TIR impossible).
        ValueError: If theta_c <= 45° (base length would be non-positive).
    """
```

The function computes `theta_c`, `h`, `E`, `M`, `b`, `B` using the formulas above, and also returns `face_angle_deg = 90 - theta_c` for compatibility with the existing constructor. It should validate that `theta_c > 45°` and raise `ValueError` otherwise.

#### 2. Add the `from_critical_ray_path` class method to `RefractometerPrism`

Follow the same pattern as `from_apex_angle` and `from_geometric_constraints`:

```python
@classmethod
def from_critical_ray_path(
    cls,
    scene: 'Scene',
    n_prism: float,
    n_target: float,
    ray_path_length: float,
    position: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0
) -> 'RefractometerPrism':
    """
    Create a prism whose geometry is fully determined by the critical
    angle for a target sample and the desired internal ray path length.

    The ray enters perpendicular to the entrance face at its midpoint,
    hits the measuring surface at the critical angle at its midpoint,
    reflects via TIR, and exits perpendicular to the exit face at its
    midpoint.

    Args:
        scene: The scene this prism belongs to.
        n_prism: Refractive index of the prism material.
        n_target: Refractive index of the target sample.
        ray_path_length: Total ray path length (L) inside the prism
                         (from entrance face midpoint to exit face midpoint
                         via measuring surface midpoint).
        position: Reference point coordinates.
        rotation: Rotation angle in degrees.

    Returns:
        A new RefractometerPrism instance.

    Raises:
        ValueError: If n_prism <= n_target (TIR impossible).
        ValueError: If theta_c <= 45° (geometry invalid).
    """
```

Implementation steps:

1. Call `tir_utils.trapezoid_from_critical_ray_path(n_target, n_prism, ray_path_length)` to get the dimensions dict.
2. Estimate an `n_sample_range` centered on `n_target` (e.g. `n_target ± 0.10`, clamped to `[1.0, n_prism - 0.01]`).
3. Create the instance via `cls(scene, n_prism, n_sample_range, measuring_surface_length=dims['M'], ...)`.
4. Override `instance.face_angle` with `dims['face_angle_deg']`.
5. Rebuild the vertex path using the computed dimensions (`M`, `B`, `h`) — **not** the default `_compute_path` which uses an arbitrary height `H = M * 0.5`. Build the vertices directly:

```python
half_B = dims['B'] / 2
half_M = dims['M'] / 2
H = dims['h']
vertices = [
    (-half_B, 0.0),   # V0
    ( half_B, 0.0),   # V1
    ( half_M, H),     # V2
    (-half_M, H),     # V3
]
instance.path = instance._apply_rotation_and_translation(vertices)
```

6. Re-apply cardinal labels: `instance.auto_label_cardinal()`.
7. Store extra attributes on the instance for later inspection:
   - `instance.n_target = n_target`
   - `instance.ray_path_length = ray_path_length`

#### 3. Add a factory function in `__init__.py`

Expose `from_critical_ray_path` through the prisms package `__init__.py`, following the same pattern as the existing `refractometer_prism` factory function.

#### 4. Add tests in `developer_tests/test_prisms.py`

- **Dimension test:** Given known `n_prism`, `n_target`, `L`, verify that `M`, `h`, `E`, `B` match expected values (use the numerical example above).
- **Symmetry test:** Verify that the entrance and exit faces have equal length.
- **Perpendicularity test:** Create a ray perpendicular to E at its midpoint, trace it, and verify it hits M at its midpoint at angle `theta_c` and exits perpendicular to X.
- **Degenerate case test:** With `theta_c = 45°` verify `ValueError` is raised.
- **Invalid case test:** With `n_target >= n_prism` verify `ValueError` is raised.
