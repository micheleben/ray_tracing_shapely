# Refactor JSON-Serializable Agentic Tools

We are refactoring and re-engineering the tools initially conceived in
`tools_with_JSON_serializable_data.md`. That roadmap says the tools are for
"LLM tool-use APIs (Claude API, claudette, lisette, langchain, etc.)".
An analysis of the current code shows that they fall short of the purpose.

---

## Critical Analysis of agentic_tools.py

### 1. No JSON Schema definitions — the biggest gap

The Claude API tool-use spec requires explicit JSON Schema for every parameter.
Example:

```json
{"name": "find_rays_crossing_edge_xml", "input_schema": {
  "type": "object",
  "properties": {
    "glass_name": {"type": "string", "description": "..."},
    "edge_label": {"type": "string", "description": "..."}
  },
  "required": ["glass_name", "edge_label"]
}}
```

Currently, the tools have no schema at all. The `get_agentic_tools()` registry
returns only `name`, `function`, `description` — the `parameters` field that
the roadmap originally specified was dropped during implementation. This works
with claudette (which introspects Python signatures), but fails for:

- Raw Claude API calls
- LangChain `StructuredTool` (needs explicit schemas for non-trivial params)
- Any framework that doesn't do Python introspection

### 2. The LLM agent has no way to discover valid parameter values

This is the most practically confusing problem. Consider
`find_rays_crossing_edge_xml(glass_name, edge_label)`:

1. *What glass names exist?* The agent must guess. There's no
   `list_scene_objects()` tool.
2. *What edge labels are valid?* The docstring says "short_label, long_label,
   or index" but gives no examples. Are they `"N"`, `"North"`,
   `"entry_face"`, `"0"`? The agent has no discovery mechanism.

The underlying functions `describe_all_glass_edges()` and
`get_object_by_name()` already exist and produce exactly this information —
but they have no agentic wrappers. The agent is blind.

### 3. XML output is a context-window bomb with no summary

`find_rays_inside_glass_xml('Main Prism')` returns 25 rays x ~40 lines each =
~1000 lines of XML dumped into the LLM's context. Every ray includes TIR
tracking, grazing flags, polarization metrics, source tracking — even when the
query was just "which rays are inside this glass?"

Problems:

1. **No summary preamble:** The `<rays count="25">` attribute exists but it's
   buried. An agent-friendly response should lead with a summary ("25 rays
   found, total brightness = 0.83, 3 with TIR").
2. **No field filtering:** Every property is always serialized. A polarization
   query doesn't need geometry data; a geometry query doesn't need grazing
   flags.
3. **Token cost:** For 100+ rays this can exceed the useful context window of
   the agent's next reasoning step.

### 4. Tools are not composable despite the 1:1 design philosophy

The roadmap says "The LLM agent can compose them freely (find inside glass,
then filter by polarization)." But it can't. Consider:

1. Agent calls `find_rays_inside_glass_xml("Main Prism")` -> gets XML string
2. Agent wants to filter those rays by polarization -> calls
   `find_rays_by_polarization_xml(0.8, 1.0)` -> this filters **all** segments,
   not the subset from step 1

There's no way to chain queries. The agent would need to:

* Parse UUIDs from the XML output
* Pass them to... what? There's no `filter_rays_by_uuid_xml()` that accepts a
  previous result

The `highlight_custom_rays_svg()` tool accepts `ray_uuids_csv` so it
recognizes this need for visualization, but there's no equivalent for the
query side. The intersection of two query results is impossible within the
current tool surface.

### 5. Missing agentic wrappers for critical existing functions

Functions that exist in the analysis module but have no agentic wrapper:

| Function | Why it matters for an agent |
|---|---|
| `describe_all_glass_edges(scene)` | Discovery — the agent needs this to know what `edge_label` values to use |
| `get_ray_statistics(segments)` | Orientation — "how many rays, what's the total brightness?" before diving into queries |
| `filter_tir_rays(segments)` | TIR analysis — common question in optics |
| `filter_grazing_rays(segments)` | Grazing analysis — already in saving.py |
| `rank_paths_by_energy(lineage)` | Lineage — "which optical path carries the most light?" |
| `check_energy_conservation(lineage)` | Validation — "is the simulation physically consistent?" |
| `fresnel_transmittances(n1, n2, theta)` | Already JSON-serializable — pure float->dict, trivial to wrap |
| `describe_simulation_result(result)` | Context — lets the agent understand what it's working with |

The first two (`describe_all_glass_edges` and `get_ray_statistics`) are
arguably prerequisites — without them, the agent is using the other tools
blind.

### 6. SVG tools are useless to most LLM agents

The 5 SVG tools (`render_scene_svg`, `highlight_rays_inside_glass_svg`, etc.)
return raw SVG strings. But:

* Text-only LLMs (most API usage) cannot interpret SVG. The string is opaque.
* Multimodal LLMs need a rendered image (PNG/JPEG), not SVG source.
* In a notebook environment (solve.it), displaying SVG requires a separate
  `display()` call that the agent can't make via these tools.
* These tools serve an unclear audience. If the goal is agent-generated
  visualization, they'd need to either save to files and return file paths, or
  be wrapped in a display mechanism for the specific environment.

### 7. Ambiguous angle semantics in `find_rays_by_angle_to_edge_xml`

The docstring says "Minimum angle from edge normal in degrees" but doesn't
clarify:

* *0 = perpendicular to edge* (head-on) vs. *0 = parallel to edge* (grazing)
  — the underlying code uses 0 = perpendicular (measuring from normal), but
  this is counter-intuitive for optics, where "angle of incidence" is measured
  from the normal too. An LLM might interpret it either way.
* The `proximity` parameter from the underlying function is silently dropped
  in the wrapper. This means the default proximity (2x edge length) is always
  used, which may not be appropriate for all scenes.

### 8. Viewbox as a comma-separated string is fragile

The pattern `viewbox: str = 'auto'` accepting either `"auto"` or
`"0,0,400,300"` is an odd hybrid. An LLM will struggle with the
comma-separated format because:

* It's a mini-DSL inside a string parameter
* There's no validation hint in the schema about the two possible formats
* If the agent generates `"0, 0, 400, 300"` (with spaces after commas) it
  works only because `.strip()` is called on each part

For a JSON Schema, this should be either separate `min_x`, `min_y`, `width`,
`height` numeric params with an `auto_viewbox: bool` flag, or the `viewbox`
param should have an `enum` plus a structured alternative.

### 9. Error handling is exception-based, not agent-friendly

When a tool fails (bad glass name, missing context), it raises Python
exceptions. Different frameworks handle this differently:

* claudette: catches and returns the traceback to the LLM
* Raw API usage: crashes the application
* LangChain: depends on the error handler

Agent-friendly tools typically return structured error responses:

```python
{"status": "error", "message": "No object named 'Prism1'. Available: ['Main Prism']"}
```

---

## Implementation Plan

> **Note on existing XML tools.** The old XML-based wrappers
> (`find_rays_inside_glass_xml`, etc.) are not removed. Since LLM agents only
> see tools that are explicitly declared during the interaction, we simply omit
> the superseded tools from the tool list passed to the agent. The old tools
> remain available for backward compatibility or manual use.

### Phase 0: Structured error convention

**Cross-cutting style rule — established before any new code is written.**

All agentic tool functions (new and refactored) must return structured error
dicts instead of raising exceptions. The underlying analysis functions continue
to raise as before; the agentic layer catches and transforms:

```python
# Agentic tool pattern:
def some_tool(param: str) -> dict:
    try:
        result = underlying_function(param)
        return {"status": "ok", "data": result}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
```

This convention applies to every tool written in Phases 1-3.

---

### Phase 1: SQLite infrastructure for tabular data

This is the foundation. It solves problems 2 (discovery), 3 (context window),
and 4 (composability) in one move.

#### Motivation

The agentic tools problem is tabular, not hierarchical. A ray segment is a
flat row of properties: p1, p2, brightness_s, brightness_p, wavelength,
is_tir, degree_of_polarization, etc. The queries agents want to run are
exactly what SQL is built for:

```sql
-- "How many rays are inside the prism, and what's the average brightness?"
SELECT COUNT(*), AVG(brightness_total) FROM rays WHERE glass_name = 'Main Prism'

-- "Which of THOSE are highly polarized?"  (composability for free)
SELECT uuid, brightness_total, degree_of_polarization FROM rays
WHERE glass_name = 'Main Prism' AND degree_of_polarization > 0.8

-- Discovery: "What objects exist?"
SELECT DISTINCT glass_name FROM ray_glass_membership
```

#### Advantages over the current XML approach

| Problem | Current (XML) | With SQLite |
|---------|---------------|-------------|
| Context window | 1000+ lines per query | `{"count": 25, "avg_brightness": 0.83}` — the agent controls what it `SELECT`s |
| Composability | Impossible (XML string in, XML string out) | Native (`WHERE ... AND ...`, subqueries) |
| Discovery | None — agent guesses glass names | `SELECT DISTINCT glass_name FROM edges` |
| Field filtering | Always dumps everything | Agent picks columns: `SELECT uuid, brightness_total` |
| New query types | Requires new Python wrapper | Agent just writes different SQL — zero code changes |
| Tool count | 9+ wrappers and growing | 2-3 tools total |

That last point is significant: every time you add a new analysis function
today, you need a new Python wrapper + registry entry + schema. With SQL, new
query patterns cost nothing.

#### Architecture

SQLite in-memory is zero-infrastructure — Python's `sqlite3` is built-in, no
dependencies, no server:

```python
# After simulation:
set_context_from_result(scene, result)
# This would also do:
#   CREATE TABLE rays (uuid, p1_x, p1_y, ..., brightness_s, brightness_p, ...)
#   CREATE TABLE glass_objects (name, uuid, edge_count, ...)
#   CREATE TABLE edges (glass_name, short_label, long_label, p1_x, ...)
#   CREATE TABLE ray_glass_membership (ray_uuid, glass_name)
#   CREATE TABLE ray_edge_crossing (ray_uuid, glass_name, edge_label)
#   INSERT INTO rays ... (one row per segment)
```

Then the agentic tool surface shrinks dramatically. Instead of 9+ specialized
tools, you could have:

1. **`query_rays(sql: str) -> str`** — runs a `SELECT` on the rays DB,
   returns compact JSON results
2. **`describe_schema() -> str`** — returns table definitions (auto-discovery)
3. Render tools (see Phase 3)

The LLM writes SQL, which it's very good at. The composability problem
vanishes because SQL has `WHERE`, `AND`, `JOIN`, subqueries natively.

#### Precomputation scope

The spatial relationships (which rays are inside which glass, which rays cross
which edge) must be precomputed at `set_context()` time using the existing
Shapely-based functions, then stored as join tables.

For a scene with G glass objects and E edges, the cost is:
- G containment checks per ray (one `polygon.contains(midpoint)` per glass)
- E intersection checks per ray (one `LineString.intersects()` per edge)

For typical scenes (2-5 glass objects, 6-15 edges, 50-200 rays) this is
trivially fast (< 100ms). For larger scenes, consider adding a spatial index
threshold or logging a warning if precomputation exceeds a time budget.

#### Limitations

1. **Spatial/geometric queries.** "Rays inside glass" uses Shapely
   `polygon.contains(midpoint)` — a geometric operation, not a simple column
   filter. Solved by precomputing spatial memberships into join tables
   (`ray_glass_membership`, `ray_edge_crossing`) during `set_context()`. This
   is a one-time upfront cost, not a fundamental obstacle.

2. **Edge-crossing and angle-to-normal queries.** These are geometric too. The
   angle computation (`find_rays_by_angle_to_edge`) depends on Shapely
   normals. Options:
   * Precompute `angle_to_edge_X` columns for each edge during setup
     (feasible for scenes with few glass objects)
   * Keep a small number of geometry tools alongside the SQL tool for queries
     that genuinely need spatial computation

3. **SQL injection.** In theory the agent writes arbitrary SQL. In practice:
   the DB is ephemeral, in-memory, read-only (enforce `SELECT`-only), and
   contains only simulation data. The blast radius is zero. Further restrict
   with `conn.execute()` instead of `executescript()`.

4. **Lineage still doesn't fit.** Tree-walking queries
   (`rank_paths_by_energy`, `detect_tir_traps`) are still better served by the
   Python lineage API. The database approach applies to the flat segment data,
   not the tree. The two coexist (see Phase 2).

#### A pragmatic hybrid

Keep both layers:

1. **SQL layer** for all flat ray/segment queries, statistics, filtering, and
   discovery. This replaces the XML wrapper tools and solves composability,
   context window, and discovery in one move.
2. **Python tools** for the things SQL can't do: lineage tree walks, SVG
   rendering, Fresnel calculations, and any geometric query too expensive to
   precompute.

The `set_context_from_result()` call does the precomputation — populates the
SQLite tables, computes spatial memberships (which rays are inside which glass,
which rays cross which edge), and stores them as join tables. After that, the
agent queries freely.

---

### Phase 2: JSON Schema definitions + lineage wrappers

Now that the tool surface is stable (`query_rays`, `describe_schema`, lineage
tools, render tools), write the JSON Schemas once, for the final tool set.

#### JSON Schema definitions

Add a proper `input_schema` dict to each tool returned by
`get_agentic_tools()`:

```python
{
    'name': 'query_rays',
    'function': query_rays,
    'description': 'Run a read-only SQL query against the simulation database.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'sql': {
                'type': 'string',
                'description': 'A SELECT query against tables: rays, '
                               'glass_objects, edges, ray_glass_membership, '
                               'ray_edge_crossing. Only SELECT is allowed.'
            }
        },
        'required': ['sql']
    }
}
```

#### Lineage tool wrappers

These are the Python-side tools that coexist with the SQL layer. Lineage is
tree-structured and does not fit relational tables, so these remain as
conventional wrappers:

* **`rank_paths_by_energy()`** — returns JSON list of paths sorted by terminal
  brightness
* **`check_energy_conservation()`** — returns JSON report of energy
  conservation at each branching point
* **`fresnel_transmittances(n1, n2, theta_i_deg)`** — already
  JSON-serializable (float in, dict out), just needs the schema definition

All follow the Phase 0 structured error convention.

---

### Phase 3: RenderResult layer (SVG/PNG)

Refactor the SVG tools to return file paths and descriptors instead of raw SVG
strings.

`SVGRenderer` stays untouched — it's a rendering engine. We add a thin
`RenderResult` layer on top:

```
+---------------------------------------------+
|  SVGRenderer  (unchanged)                   |
|  Produces SVG string / saves .svg file      |
+----------------------+----------------------+
                       |
+----------------------v----------------------+
|  RenderResult (new, thin layer)             |
|  Saves .svg + converts .png                 |
|  Returns JSON-serializable descriptor       |
+----------+-----------------+----------------+
           |                 |
+----------v-----+   +------v--------+
| Human GUI      |   | LLM Tool      |
| IPython        |   | Loads PNG     |
| display()      |   | as base64     |
| of SVG/HTML    |   | or file ref   |
+----------------+   +---------------+
```

#### The return contract

The render tools return a dict (JSON-serializable) instead of a raw SVG
string:

```python
{
    "svg_path": "/tmp/ray_sim_a3f2/scene_001.svg",
    "png_path": "/tmp/ray_sim_a3f2/scene_001.png",
    "width": 800,
    "height": 600,
    "description": "Scene with 3 objects, 47 ray segments. "
                   "Highlighted: 12 rays inside 'Main Prism'.",
    "highlight_summary": {
        "highlighted_rays": 12,
        "total_rays": 47,
        "glass_highlighted": ["Main Prism"]
    }
}
```

This solves multiple problems at once:

* The LLM gets a text description even without loading the image — it knows
  what the render shows
* The multimodal LLM path just reads `png_path` and sends it to the vision API
* The human notebook path reads `svg_path` and calls `IPython.display.SVG()`
* The file paths are stable — the agent can reference earlier renders by path

#### Cairo: optional SVG-to-PNG conversion

The `cairosvg` dependency is pluggable and optional:

```python
def _svg_to_png(svg_string: str, png_path: str, width: int, height: int) -> bool:
    """Convert SVG to PNG. Returns False if no backend is available."""
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_string.encode('utf-8'),
                         write_to=png_path,
                         output_width=width, output_height=height)
        return True
    except ImportError:
        pass
    # Could add fallback backends here
    return False
```

When cairosvg is not installed, the render result degrades gracefully:

```python
{
    "svg_path": "/tmp/.../scene.svg",
    "png_path": null,
    "png_available": false,
    "png_backend": "none (install cairosvg for PNG support)"
}
```

#### The two consumer wrappers

**1. For notebooks (human GUI):**

```python
def display_render(render_result: dict) -> None:
    """Display a render result in a Jupyter/solve.it notebook."""
    from IPython.display import display, SVG
    display(SVG(filename=render_result['svg_path']))
```

**2. For multimodal LLM tools:**

```python
def render_scene_for_llm(...) -> dict:
    """Agentic tool: render scene, return descriptor + PNG path."""
    # 1. Render SVG via SVGRenderer
    # 2. Convert to PNG
    # 3. Return the descriptor dict
    ...
```

The LLM framework then does its thing with the PNG — Claude API takes base64
image content blocks, GPT-4V takes URLs or base64, etc. That
framework-specific plumbing lives outside this library.

#### Where files go

A session-scoped temp directory is created at `set_context()` time:

```python
import tempfile

def set_context_from_result(scene, result):
    ...
    _CONTEXT['render_dir'] = tempfile.mkdtemp(prefix='ray_sim_')
```

Each render call auto-generates a filename with an incrementing counter and
renderer uuid (`<renderer_uuid>_scene_001.svg`,
`<renderer_uuid>_highlight_002.svg`, etc.). This way:

* Files don't collide across calls
* The agent can reference earlier renders by path
* Cleanup is trivial (`shutil.rmtree` on the temp dir, or let the OS handle
  it)

#### What this means for the agentic tool surface

The current 5 SVG tools (`render_scene_svg`, `highlight_rays_inside_glass_svg`,
etc.) would each return the descriptor dict instead of raw SVG strings. The
registry would expose:

1. `render_scene(...)` -> `{"svg_path": ..., "png_path": ..., "description": ...}`
2. `highlight_rays(...)` -> same contract
3. `display_render(render_result)` -> human-only, not an LLM tool

The `description` field is the quiet win here — the LLM gets a
natural-language summary of what the image shows, even if it never loads the
PNG. For a text-only agent that's already enough ("the render shows 12
highlighted rays inside Main Prism, 8 of which experienced TIR"). For a
multimodal agent, it's a useful caption alongside the image.

---

## Implementation Notes

### Phase 0 — DONE

All 9 existing agentic tools in `agentic_tools.py` now return structured
`{"status": "ok", "data": ...}` or `{"status": "error", "message": ...}`
dicts instead of raising exceptions. Added `_ok()` and `_error()` helper
functions. Updated return types from `str` to `Dict[str, Any]`.

Fixed `test_phase7_agentic_svg_tools.py` — 5 SVG test functions updated to
unwrap `result["data"]` since tools now return dicts instead of raw strings.

Created `CONTRIBUTING.md` documenting the Phase 0 structured error convention
alongside general project conventions.

### Phase 1 — DONE

Created `analysis/agentic_db.py` with in-memory SQLite infrastructure:

- **5 tables**: `rays` (29 cols), `glass_objects` (10 cols), `edges` (13
  cols), `ray_glass_membership` (precomputed containment), `ray_edge_crossing`
  (precomputed intersection)
- **SQL validation**: SELECT-only enforcement, forbidden keyword regex,
  single-statement only
- **Columnar results**: `{columns, rows, row_count}` format for token
  efficiency, truncation at 200 rows

Added `query_rays(sql)` and `describe_schema()` agentic tools. Database is
created at `set_context()` time and closed at `clear_context()`. Spatial joins
are precomputed using existing `glass_to_polygon()` and Shapely containment /
intersection checks.

Test: `test_phase1_agentic_db.py` — 17 tests covering schema, population,
computed columns, spatial join correctness (validated against
`find_rays_inside_glass` and `find_rays_crossing_edge`), SQL injection
rejection, and registry integration.

### Phase 2 — DONE

Two deliverables:

**1. JSON Schema definitions on all tools.** Every tool returned by
`get_agentic_tools()` now includes an `input_schema` dict (proper JSON
Schema with `type`, `properties`, `required`). This enables raw Claude API
tool-use, LangChain `StructuredTool`, and any framework that needs explicit
schemas. Shared SVG property fragments (`width`, `height`, `viewbox`) are
defined once and spread into each SVG tool schema.

**2. Three new agentic wrappers:**

- **`rank_paths_by_energy(top_n=10)`** — delegates to
  `lineage_analysis.rank_paths_by_energy()`, converts the `'path'` field
  (containing Ray objects) to `'path_uuids'` (list of uuid strings for use
  with `highlight_custom_rays_svg`). Returns top N results.
- **`check_energy_conservation()`** — delegates to
  `lineage_analysis.check_energy_conservation()`. Return value is already
  JSON-serializable.
- **`fresnel_transmittances_tool(n1, n2, theta_i_deg)`** — delegates to
  `fresnel_utils.fresnel_transmittances()`. Standalone (no context needed).
  Named `_tool` to avoid collision with the raw function in `__init__.py`.

Added `_require_lineage()` helper that returns `_CONTEXT['lineage']` or a
structured error if lineage was not passed to `set_context()`.

Total agentic tools: 14 (2 SQL + 3 lineage/Fresnel + 4 legacy XML + 5 SVG).

Test: `test_phase2_schemas_and_lineage.py` — 12 tests covering schema
presence/validity, lineage tool results, JSON-serializability (no Ray objects
in output), top_n slicing, error handling for missing lineage, Fresnel
correctness including TIR error case, and registry integration.
