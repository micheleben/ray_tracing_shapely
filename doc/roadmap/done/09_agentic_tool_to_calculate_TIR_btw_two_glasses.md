# Agentic Tool to calculate TIR and TIR physics between two glasses
## Code Style
When implementing this roadmap please follow the code style as described in CONTRIBUTING.md

## Aim
We want a tool an agent can call to calculate the TIR angle between two glasses of the scene.

The tool should also be a general tool we can use to see how the relative phase shift between polarizations varies approaching the TIR angle and over the TIR angle.

### the way I envisioned the tool:

It should have as input parameters:
* the two names of the glasses, we ask the user to put the first glass as the one with higher index of refraction.
* optionally the angle of incidence between the glass with higher index and the glass with lower index.
* optionally a delta_angle to describe angles that are close to the TIR or close to BREWSTER angles. default 1 degree.

The code should do the following behaviour:

* It should check which one has the greater index of refraction and return an agentic friendly error if the first prism is not the one with the highest index

* It should return the following physical quantities:
    + some flags to understand which output the agent is seeing (covering the different options we see below and errors as we described above)
    + the TIR angle
    + if the angle is provided:
        - if it produces refraction calculate all the refraction quantities ('T_s', 'T_p', 'ratio_Tp_Ts','R_s','R_p'). Calculate the phase difference between s and p even if it is trivial. Flag this with "angle < TIR" or similar
        - if it does not produce refraction return "not applicable" or similar for the refraction quantities and calculate the phase shifts. Flag this with "angle > TIR"
        - special flag for angle within +/- delta_angle (default +/- 1 degree) to the TIR angle
        - special flag for angle within +/- delta_angle (default +/- 1 degree) to the BREWSTER angle
    + if the angle is not provided
        - flag this as "default calcs at + delta_angle degree from TIR" and return calculations for an angle that is + delta_angle from the TIR

## Physics Background
### 1. Phase Shifts Before TIR

(\(\theta _{i}<\theta _{c}\))

In this regime, the reflection coefficients are real numbers. A real value means the phase shift
(\(\delta \))
can only be:
* \(\delta =0\): The reflected wave is in phase with the incident wave (positive coefficient).

* \(\delta =\pi \) (\(180^{\circ }\)): The reflected wave is perfectly out of phase (negative coefficient).

For internal reflection (moving from a denser to a rarer medium), the phase behavior is as follows:

 * s-polarization: Typically has no phase shift

 (\(\delta _{s}=0\))

 for all angles below the critical angle.

 p-polarization: Has a \(\pi \) phase shift at low angles, but this switches to \(0\) once you surpass the Brewster angle.

#### Fresnel amplitude coefficients (sub-critical regime)

To compute the phase shifts below TIR we need the Fresnel **amplitude** reflection coefficients (not the power/intensity coefficients already in `fresnel_transmittances`):

 \(r_{s}=\frac{n_{1}\cos \theta _{i}-n_{2}\cos \theta _{t}}{n_{1}\cos \theta _{i}+n_{2}\cos \theta _{t}}\)

 \(r_{p}=\frac{n_{2}\cos \theta _{i}-n_{1}\cos \theta _{t}}{n_{2}\cos \theta _{i}+n_{1}\cos \theta _{t}}\)

where \(\theta _{t}\) comes from Snell's law: \(n_{1}\sin \theta _{i}=n_{2}\sin \theta _{t}\).

These are real numbers below the critical angle.  The phase is:
 * \(\delta = 0\) if the coefficient is positive (\(r > 0\))
 * \(\delta = \pi\) if the coefficient is negative (\(r < 0\))

Note: the power reflectances already in the codebase are \(R_{s}=r_{s}^{2}\) and \(R_{p}=r_{p}^{2}\), so they lose the sign information.  The new function must compute the amplitude coefficients to recover the phase.

 ### 2. Phase Shifts During TIR
 (\(\theta _{i}>\theta _{c}\))

 Once you surpass the critical angle, the reflection coefficients become complex numbers.
 This results in continuous, non-trivial phase shifts

 (\(\delta _{s},\delta _{p}\))

 that depend on the specific angle of incidence. The Formulas for Phase Shifts in TIR The phase advances for the two components are given by these Fresnel-derived formulas:

  \(\tan \left(\frac{\delta _{s}}{2}\right)=\frac{\sqrt{n^{2}\sin ^{2}\theta _{i}-1}}{n\cos \theta _{i}}\)\(\tan \left(\frac{\delta _{p}}{2}\right)=\frac{n\sqrt{n^{2}\sin ^{2}\theta _{i}-1}}{\cos \theta _{i}}=n^{2}\tan \left(\frac{\delta _{s}}{2}\right)\)

 \(n=n_{incident}/n_{transmitted}\): The relative refractive index (e.g., \(1.5\) for glass-to-air).
 \(\theta _{i}\): The angle of incidence.
 \(\delta _{s},\delta _{p}\): The absolute phase shifts for s- and p-polarized light.

 #### 3. The Relative Phase Shift
 (\(\Delta \))
 The most important value for polarization is the differential phase shift
 (\(\Delta =\delta _{p}-\delta _{s}\)).

 This difference is what turns linear polarization into elliptical polarization:

 \(\tan \left(\frac{\Delta }{2}\right)=\frac{\cos \theta _{i}\sqrt{n^{2}\sin ^{2}\theta _{i}-1}}{n\sin ^{2}\theta _{i}}\)

 ### Summary of Differences Phase Characteristic

 | Region | Angles | Phase shift Nature | Coefficients | Results | Slope |
 |:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
 | Below Critical Angle | (\(\theta _{i}<\theta _{c}\)) | Trivial: Discrete values (\(0\) or \(\pi \)) | Real Coefficients | Reflected light stays linear | Constant between transitions |
 | Above Critical Angle |  (\(\theta _{i}>\theta _{c}\)) | Non-Trivial: Continuous variation | Complex numbers | Reflected light becomes elliptical | Infinite slope at the critical angle |

## Status of the codebase

we have in analysis.fresnel_utils

```python
def fresnel_transmittances(
    n1:float, n2:float, theta_i_deg:float
)->Dict[str, float]:
```
```python
Compute Fresnel power transmittances and reflectances at an interface.

Uses the standard Fresnel equations for a planar interface between
two dielectric media.

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

```
It is a great one to study refraction , here we want to study reflectance.

### Glass name to refractive index resolution

The agentic tool takes glass names (strings), but the physics functions work with raw `n1`, `n2` floats.  Resolution path:

1. The agentic wrapper calls `_require_context()` to get the `scene` object.
2. It calls `get_object_by_name(scene, glass_name)` (from `ray_geometry_queries`) to get the glass object.
3. It reads `glass.refIndex` to get the refractive index.

This means:
- The **pure-physics function** in `fresnel_utils.py` takes `n1: float, n2: float` (no scene dependency).
- The **agentic wrapper** in `agentic_tools.py` resolves names → `n` values, calls the physics function, then decorates the result with flags.
- The agentic wrapper **requires context** (unlike `fresnel_transmittances_tool` which is standalone). If the user wants to use raw n values without a scene, they call the physics function directly.

## Proposed changes
Few touch_points:
1. add a pure-physics function in `analysis.fresnel_utils` that takes `n1, n2, theta_i_deg` floats and implements the physics we discussed above (both sub-critical and TIR regimes). This function has no scene dependency. I understand part of the physics overlap with `fresnel_transmittances` but we will likely keep both.
2. add the agentic wrapper in `agentic_tools.py` that resolves glass names to `n` values, calls the physics function, and adds the flag logic.
3. JSON schema additions in `tool_registry.py` (`get_agentic_tools` return list).
4. update `_REGISTRY` list in `tool_registry.py`.

### Touch point 1: pure-physics function in `fresnel_utils.py`

New function signature:

```python
def tir_analysis(
    n1: float, n2: float, theta_i_deg: Optional[float] = None,
    delta_angle_deg: float = 1.0,
) -> Dict[str, Any]:
```

This function:
- Validates `n1 > n2` (raises `ValueError` if not).
- Computes TIR critical angle and Brewster angle.
- If `theta_i_deg` is None, uses `critical_angle + delta_angle_deg` as the default angle.
- Determines the regime (sub-critical or TIR).
- Computes phase shifts using the amplitude coefficients (sub-critical) or the TIR formulas (super-critical).
- Computes Fresnel power quantities (T_s, T_p, R_s, R_p) when in refraction regime.
- Returns a flat dict with all results (see output schema below).

### Touch point 2: agentic wrapper in `agentic_tools.py`

New function signature:

```python
def tir_analysis_tool(
    glass_name_high_n: str,
    glass_name_low_n: str,
    theta_i_deg: Optional[float] = None,
    delta_angle_deg: float = 1.0,
) -> Dict[str, Any]:
```

This wrapper:
- Calls `_require_context()` to get the scene.
- Resolves both glass names via `get_object_by_name(scene, name)`.
- Reads `glass.refIndex` from each.
- Validates that the first glass has the higher index; if not, returns a structured error with the actual n values so the agent can self-correct.
- Delegates to `tir_analysis(n1, n2, theta_i_deg, delta_angle_deg)` from `fresnel_utils`.
- Wraps the result in `_ok(data)` / `_error(message)`.

### Touch point 3 & 4: JSON schema and registry in `tool_registry.py`

Add to `_REGISTRY`:

```python
{
    'module': 'analysis.fresnel_utils',
    'name': 'tir_analysis',
    'kind': 'function',
    'signature': '(n1, n2, theta_i_deg=None, delta_angle_deg=1.0) -> Dict[str, Any]',
    'description': 'Compute TIR angle, phase shifts, and Fresnel quantities for two media',
},
```

```python
{
    'module': 'analysis.agentic_tools',
    'name': 'tir_analysis_tool',
    'kind': 'function',
    'signature': '(glass_name_high_n, glass_name_low_n, theta_i_deg=None, delta_angle_deg=1.0) -> Dict[str, Any]',
    'description': 'Compute TIR angle, phase shifts, and Fresnel quantities between two named glasses',
},
```

Add to `get_agentic_tools()` return list:

```python
{
    'name': 'tir_analysis',
    'function': tir_analysis_tool,
    'description': (
        'Compute TIR critical angle, Brewster angle, Fresnel reflectances, '
        'and phase shifts between two named glasses. The first glass must have '
        'the higher refractive index. Optionally provide an angle of incidence; '
        'if omitted, defaults to critical_angle + delta_angle_deg.'
    ),
    'input_schema': {
        'type': 'object',
        'properties': {
            'glass_name_high_n': {
                'type': 'string',
                'description': 'Name of the glass with the higher refractive index.',
            },
            'glass_name_low_n': {
                'type': 'string',
                'description': 'Name of the glass with the lower refractive index.',
            },
            'theta_i_deg': {
                'type': 'number',
                'description': (
                    'Angle of incidence in degrees (from normal). '
                    'If omitted, defaults to critical_angle + delta_angle_deg.'
                ),
            },
            'delta_angle_deg': {
                'type': 'number',
                'description': (
                    'Proximity threshold in degrees for near-TIR and near-Brewster flags. '
                    'Also used as the offset from TIR when no angle is provided.'
                ),
                'default': 1.0,
            },
        },
        'required': ['glass_name_high_n', 'glass_name_low_n'],
    },
},
```

## Output schema

### Success — sub-critical regime (angle < TIR, refraction occurs)

```json
{
    "status": "ok",
    "data": {
        "regime": "refraction",
        "near_tir": false,
        "near_brewster": true,
        "angle_provided": true,
        "glass_high_n": "Flint Glass",
        "glass_low_n": "Crown Glass",
        "n1": 1.72,
        "n2": 1.52,
        "theta_i_deg": 41.5,
        "tir_angle_deg": 62.03,
        "brewster_angle_deg": 41.48,
        "delta_angle_deg": 1.0,

        "T_s": 0.9812,
        "T_p": 0.9998,
        "R_s": 0.0188,
        "R_p": 0.0002,
        "ratio_Tp_Ts": 1.019,
        "theta_t_deg": 48.12,

        "delta_s_deg": 0.0,
        "delta_p_deg": 0.0,
        "delta_relative_deg": 0.0
    }
}
```

### Success — TIR regime (angle > TIR, total internal reflection)

```json
{
    "status": "ok",
    "data": {
        "regime": "tir",
        "near_tir": false,
        "near_brewster": false,
        "angle_provided": true,
        "glass_high_n": "Flint Glass",
        "glass_low_n": "Crown Glass",
        "n1": 1.72,
        "n2": 1.52,
        "theta_i_deg": 75.0,
        "tir_angle_deg": 62.03,
        "brewster_angle_deg": 41.48,
        "delta_angle_deg": 1.0,

        "T_s": null,
        "T_p": null,
        "R_s": 1.0,
        "R_p": 1.0,
        "ratio_Tp_Ts": null,
        "theta_t_deg": null,

        "delta_s_deg": 98.42,
        "delta_p_deg": 143.17,
        "delta_relative_deg": 44.75
    }
}
```

### Success — no angle provided (defaults to TIR + delta)

```json
{
    "status": "ok",
    "data": {
        "regime": "tir",
        "near_tir": true,
        "near_brewster": false,
        "angle_provided": false,
        "glass_high_n": "Flint Glass",
        "glass_low_n": "Crown Glass",
        "n1": 1.72,
        "n2": 1.52,
        "theta_i_deg": 63.03,
        "tir_angle_deg": 62.03,
        "brewster_angle_deg": 41.48,
        "delta_angle_deg": 1.0,

        "T_s": null,
        "T_p": null,
        "R_s": 1.0,
        "R_p": 1.0,
        "ratio_Tp_Ts": null,
        "theta_t_deg": null,

        "delta_s_deg": 14.21,
        "delta_p_deg": 22.87,
        "delta_relative_deg": 8.66
    }
}
```

### Error — wrong glass order

```json
{
    "status": "error",
    "message": "First glass must have higher refractive index. 'Crown Glass' has n=1.52 but 'Flint Glass' has n=1.72. Swap the arguments: tir_analysis(glass_name_high_n='Flint Glass', glass_name_low_n='Crown Glass', ...)"
}
```

### Error — glass not found

```json
{
    "status": "error",
    "message": "No object named 'Prism1'. Available glass objects: ['Flint Glass', 'Crown Glass']"
}
```

### Error — no context

```json
{
    "status": "error",
    "message": "No context set. Call set_context() or set_context_from_result() before using agentic tools."
}
```

### Output schema field reference

| Field | Type | Present | Description |
|:------|:-----|:--------|:------------|
| `regime` | `str` | always | `"refraction"` or `"tir"` |
| `near_tir` | `bool` | always | `true` if angle is within `+/- delta_angle_deg` of the critical angle |
| `near_brewster` | `bool` | always | `true` if angle is within `+/- delta_angle_deg` of Brewster's angle |
| `angle_provided` | `bool` | always | `false` if the tool used the default angle (TIR + delta) |
| `glass_high_n` | `str` | always | name of the glass with higher n |
| `glass_low_n` | `str` | always | name of the glass with lower n |
| `n1` | `float` | always | refractive index of the denser glass |
| `n2` | `float` | always | refractive index of the rarer glass |
| `theta_i_deg` | `float` | always | angle of incidence used for the calculation (degrees) |
| `tir_angle_deg` | `float` | always | critical angle for TIR (degrees) |
| `brewster_angle_deg` | `float` | always | Brewster's angle (degrees) |
| `delta_angle_deg` | `float` | always | proximity threshold used (degrees) |
| `T_s` | `float \| null` | always | s-pol power transmittance (`null` in TIR regime) |
| `T_p` | `float \| null` | always | p-pol power transmittance (`null` in TIR regime) |
| `R_s` | `float` | always | s-pol power reflectance (1.0 in TIR regime) |
| `R_p` | `float` | always | p-pol power reflectance (1.0 in TIR regime) |
| `ratio_Tp_Ts` | `float \| null` | always | T_p / T_s (`null` in TIR regime) |
| `theta_t_deg` | `float \| null` | always | refraction angle (`null` in TIR regime) |
| `delta_s_deg` | `float` | always | s-pol phase shift on reflection (degrees) |
| `delta_p_deg` | `float` | always | p-pol phase shift on reflection (degrees) |
| `delta_relative_deg` | `float` | always | relative phase shift delta_p - delta_s (degrees) |

## Implementation notes

**Status: IMPLEMENTED** — all touch points completed and verified.

### Files changed

| File | Change |
|:-----|:-------|
| `analysis/fresnel_utils.py` | Added `tir_analysis()` (~100 lines). Import of `Any`, `Optional` added. |
| `analysis/agentic_tools.py` | Added `tir_analysis_tool()` wrapper. Added imports for `get_objects_by_type` and `Optional`. |
| `analysis/tool_registry.py` | Two `_REGISTRY` entries (fresnel_utils + agentic_tools). One `get_agentic_tools()` entry with full `input_schema`. Import of `tir_analysis_tool`. |
| `analysis/__init__.py` | Exported `tir_analysis` (from fresnel_utils) and `tir_analysis_tool` (from agentic_tools). Added to `__all__`. |
| `developer_tests/test_phase2_schemas_and_lineage.py` | Updated expected tool count from 14 to 15. |

### Files created

| File | Description |
|:-----|:------------|
| `developer_tests/test_tir_analysis.py` | 15 tests: 9 pure-physics + 6 agentic wrapper. |

### Test results

- `test_tir_analysis.py`: **15/15 passed**
- `test_phase2_schemas_and_lineage.py`: **12/12 passed** (including updated tool count)

### Deviations from the roadmap

1. **`glass.refIndex` not `glass.n`** — the Glass object stores the refractive index as the `refIndex` attribute (inherited from `serializable_defaults` in `BaseSceneObj`). The roadmap originally said `glass.n`. Corrected in the roadmap during implementation.

2. **Defensive `refIndex` check** — the agentic wrapper uses `getattr(glass, 'refIndex', None)` instead of direct attribute access. If a non-Glass scene object is passed by name, it returns a structured error ("have no refIndex attribute. Are they Glass objects?") rather than crashing with `AttributeError`. This was not in the original roadmap but follows the CONTRIBUTING.md guideline of discoverable error messages.

3. **`glass_high_n` / `glass_low_n` added by the wrapper, not the physics function** — the pure-physics `tir_analysis()` returns only `n1` and `n2` (floats). The agentic wrapper injects the `glass_high_n` and `glass_low_n` string fields into the result dict before wrapping with `_ok()`. This keeps the physics function scene-free as intended.

4. **Sub-critical phase shifts as 0.0 or 180.0** — the roadmap described these qualitatively. The implementation computes the amplitude reflection coefficients `r_s` and `r_p` and maps their sign to `0.0` or `180.0` degrees. This gives the agent a uniform `delta_s_deg` / `delta_p_deg` / `delta_relative_deg` interface across both regimes.

### Physics verification highlights

- Brewster phase flip confirmed: `delta_p_deg` = 180 below Brewster, 0 above (for n1 > n2 internal reflection).
- `delta_s_deg` = 0 everywhere below TIR (for n1 > n2), consistent with theory.
- Phase shifts approach 0 continuously just above TIR (test measured 2.02 deg at TIR + 0.01 deg).
- Phase shifts increase monotonically above TIR (153 deg at 80 deg for n=1.5/1.0).
- Energy conservation T + R = 1 verified for refraction regime.
