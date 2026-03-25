# Agentic Tools: JSON-Serializable Wrappers

## Problem statement

LLM tool-use APIs (Claude API, claudette, lisette, langchain, etc.) require
that tool inputs and outputs are JSON-serializable: dicts, lists, strings,
numbers.  Our analysis tools take and return live Python objects (`Glass`
instances, `List[Ray]`, `RayLineage`) that don't survive JSON serialization.

We need **string-in / string-out wrapper functions** that:
- Accept only JSON-serializable parameters (object names as strings, numeric thresholds)
- Resolve live objects from a pre-registered context
- Delegate to the existing analysis functions
- Return results as XML strings (via `rays_to_xml()`)

Target file: `analysis/agentic_tools.py` (new module).

---

## Context management

The wrappers need access to live Python objects (the `Scene` for resolving
glass names, the segment list, the lineage tree).  These cannot be passed as
tool parameters, so they are stored in a module-level context dict.

### Why not `globals()`?

`globals()` called from inside a library module returns the **library's**
namespace, not the caller's.  If `find_rays_inside_glass_xml()` lives in
`analysis/agentic_tools.py`, then `globals()` inside that function sees
`agentic_tools`'s module globals — not the IPython kernel's variables or the
caller's script namespace.  The solveit scenario would need to pass its
`globals()` dict explicitly, which is equivalent to an explicit context
anyway.

### Unified approach: explicit `_CONTEXT` dict

A single `_CONTEXT` dict serves both scenarios.  The caller populates it
once after running a simulation; the tool functions read from it.

```python
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.ray import Ray
    from ..core.ray_lineage import RayLineage
    from .simulation_result import SimulationResult

# Module-level context: holds live objects for agentic tools
_CONTEXT: Dict[str, Any] = {}


def set_context(
    scene: 'Scene',
    segments: List['Ray'],
    lineage: Optional['RayLineage'] = None,
) -> None:
    """
    Register simulation objects for use by agentic tool wrappers.

    Must be called before any agentic tool is invoked.

    Args:
        scene: The Scene object (needed to resolve glass names via
            get_object_by_name).
        segments: The ray segments from the simulation.
        lineage: Optional RayLineage for lineage-based tools.
    """
    _CONTEXT['scene'] = scene
    _CONTEXT['segments'] = segments
    _CONTEXT['lineage'] = lineage


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
    """Return the current context dict (for inspection/debugging)."""
    return dict(_CONTEXT)


def clear_context() -> None:
    """Clear the context (e.g. between simulation runs)."""
    _CONTEXT.clear()
```

### Usage in solveit (persistent IPython kernel)

```python
# In a code cell:
result = simulator.run_with_result(name="Grazing test")

# Register context once — tools can then be called by the LLM
from ray_tracing_shapely.analysis.agentic_tools import set_context_from_result
set_context_from_result(scene=scene, result=result)
```

### Usage in standalone scripts (claudette/lisette)

```python
from ray_tracing_shapely.analysis.agentic_tools import (
    set_context_from_result,
    find_rays_inside_glass_xml,
    find_rays_crossing_edge_xml,
)

# Run simulation
result = simulator.run_with_result()
set_context_from_result(scene=scene, result=result)

# Pass tools to LLM framework
from claudette import Chat
chat = Chat(model, tools=[find_rays_inside_glass_xml, find_rays_crossing_edge_xml])
```

Both scenarios use the same code — the only difference is who calls
`set_context_from_result()`.

---

## Phase 1: String-based tool wrappers -- IMPLEMENTED

> **Status**: Complete. Four XML wrapper functions and context management
> implemented in `analysis/agentic_tools.py`. `get_agentic_tools()` added to
> `analysis/tool_registry.py`. All exported via `analysis/__init__.py`.

Each wrapper is a 1:1 match to an existing analysis function.  It takes only
JSON-serializable parameters, resolves objects from `_CONTEXT`, delegates to
the original function, and serializes the result with `rays_to_xml()`.

### Wrapped tools

```python
def find_rays_inside_glass_xml(glass_name: str) -> str:
    """
    Find rays whose midpoint is inside a named glass object.

    Args:
        glass_name: Name of the glass object in the scene.

    Returns:
        XML string describing the matching rays.

    Raises:
        RuntimeError: If set_context() has not been called.
        ValueError: If glass_name doesn't match any object.
    """
    scene = _CONTEXT['scene']
    segments = _CONTEXT['segments']
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_inside_glass(segments, glass)
    return rays_to_xml(rays)


def find_rays_crossing_edge_xml(glass_name: str, edge_label: str) -> str:
    """
    Find rays that cross a specific edge of a named glass object.

    Args:
        glass_name: Name of the glass object in the scene.
        edge_label: Label of the edge (short_label, long_label, or index).

    Returns:
        XML string describing the matching rays.
    """


def find_rays_by_angle_to_edge_xml(
    glass_name: str,
    edge_label: str,
    min_angle: float = 0.0,
    max_angle: float = 90.0,
) -> str:
    """
    Find rays within an angle range relative to a named glass edge.

    Args:
        glass_name: Name of the glass object in the scene.
        edge_label: Label of the reference edge.
        min_angle: Minimum angle from edge normal in degrees.
        max_angle: Maximum angle from edge normal in degrees.

    Returns:
        XML string describing the matching rays.
    """


def find_rays_by_polarization_xml(
    min_dop: float = 0.0,
    max_dop: float = 1.0,
) -> str:
    """
    Filter rays by degree of polarization.

    Args:
        min_dop: Minimum degree of polarization (0-1).
        max_dop: Maximum degree of polarization (0-1).

    Returns:
        XML string describing the matching rays.
    """
```

### Implementation pattern

Each wrapper follows the same 4-step pattern:

1. Read `_CONTEXT['scene']` and `_CONTEXT['segments']` (raise `RuntimeError`
   with a clear message if context is empty)
2. Resolve string names to objects via `get_object_by_name()`
3. Call the original analysis function
4. Serialize with `rays_to_xml()` and return the string

### Error handling

- Missing context → `RuntimeError("No context set. Call set_context() or
  set_context_from_result() before using agentic tools.")`
- Bad glass name → `ValueError` from `get_object_by_name()` (propagated as-is,
  includes the list of available names)
- Bad edge label → `ValueError` from the underlying tool (includes available
  labels)

### Why 1:1 wrappers, not composite pipelines

The original draft proposed `find_grazing_rays_in_named_glass()` which chains
`find_rays_inside_glass` + `filter_grazing_rays` + `rays_to_xml` in a single
function.  This is a composite pipeline, not a wrapper.

We prefer 1:1 wrappers because:
- The LLM agent can compose them freely (find inside glass, then filter by
  polarization, etc.)
- Composite pipelines bake in assumptions about what the user wants
- If we later add more analysis functions, each gets one wrapper — no
  combinatorial explosion of pipeline variants

Composite convenience functions can be added later if common patterns emerge,
but the foundation should be generic wrappers.

---

## Phase 2: Update tool_registry.py -- IMPLEMENTED

> **Status**: Complete. `get_agentic_tools()` added to `tool_registry.py`.
> Static registry updated with 8 new entries (6 agentic_tools + 2 registry).

Add a function that returns the agentic tools as a list of dicts with
metadata that any LLM framework can consume.

```python
def get_agentic_tools() -> List[Dict[str, Any]]:
    """
    Return the string-based agentic tool wrappers with metadata.

    Each entry contains:
    - 'name': function name (str)
    - 'function': the callable
    - 'description': one-line description (str)
    - 'parameters': dict of parameter names to types/descriptions

    These are framework-agnostic.  To use with a specific framework:

        # claudette
        from claudette import Chat
        tools = [t['function'] for t in get_agentic_tools()]
        chat = Chat(model, tools=tools)

        # langchain
        from langchain.tools import StructuredTool
        tools = [StructuredTool.from_function(t['function']) for t in get_agentic_tools()]
    """
```

### Why not a `Tool` class?

Different LLM frameworks define their own tool types (claudette uses decorated
functions, langchain uses `StructuredTool`, Claude API uses JSON schemas).
Defining our own `Tool` class would just add another layer to unwrap.

Plain dicts with `name`, `function`, `description` are the lowest common
denominator that every framework can consume.  The docstrings on the wrapper
functions themselves carry the parameter descriptions that frameworks like
claudette extract automatically via introspection.

### Files created / modified

| File | Action |
|------|--------|
| `analysis/agentic_tools.py` | **Created.** Context management (`set_context`, `set_context_from_result`, `get_context`, `clear_context`) + 4 XML wrapper functions + `_require_context()` helper |
| `analysis/tool_registry.py` | **Modified.** Added `get_agentic_tools()` function + 8 new entries to static registry |
| `analysis/__init__.py` | **Modified.** Added imports and `__all__` entries for all new public symbols |

### Verification results

Tested with a point source + prism scene (100 segments):

```
Context management:
  set_context_from_result() correctly populates scene, segments, lineage
  clear_context() correctly resets — tools raise RuntimeError after clear

Tool wrappers:
  find_rays_inside_glass_xml('Main Prism')     -> 25 rays (XML, 10685 chars)
  find_rays_crossing_edge_xml('Main Prism','N') -> 18 rays
  find_rays_by_angle_to_edge_xml('Main Prism','N', 0, 30) -> 12 rays
  find_rays_by_polarization_xml(0.0, 0.1)      -> 67 rays

Error handling:
  No context  -> RuntimeError with clear message
  Bad name    -> ValueError("No object named 'Nonexistent'. Named objects: ['Main Prism']")

get_agentic_tools():
  Returns 4 dicts with 'name', 'function', 'description'
  All functions are callable
```

---

## Future extensions

Once the Phase 1 wrappers are proven, the same pattern extends to:

- Lineage tools: `rank_paths_by_energy_xml()`, `check_energy_conservation_xml()`
- Fresnel tools: already JSON-serializable (float in, dict out) — may just
  need `format='xml'` variants
- Scene description tools: `describe_all_glass_edges()` already returns strings
- Composite pipelines: e.g. `find_bright_paths_through_edge_xml(glass_name,
  edge_label, min_energy)` if common agent workflows emerge
