# Refactoring of SVG Renderer: Simulation-Aligned Metadata

## Goal

Two complementary goals:

1. **Simulation-aligned SVG**: Make every SVG element traceable back to the simulation object it represents. A user opening the SVG in Inkscape can find rays by uuid, glass objects by name, and edges by label.

2. **Agent-driven visual feedback**: Let an LLM agent produce SVGs that **highlight query results** -- so when a user asks "show me the rays crossing the W edge of the measuring prism", the agent can render an SVG with those 54 rays drawn in a highlight color on top of the full scene. The SVG becomes the communication channel between agent and human.

### The interaction we're building toward

```
User:  "Highlight all rays crossing the W edge of the measuring prism"

Agent: 1. Calls find_rays_crossing_edge_xml(glass_name='measuring prism', edge_label='W')
       2. Calls render_scene_with_highlights_svg(
              highlight_rays=crossing_ray_uuids,
              highlight_edges=[('measuring prism', 'W')],
              highlight_color='yellow',
              dim_others=True
          )
       3. Returns SVG to the GUI

User:  Sees the full scene with the 54 crossing rays in yellow, everything else dimmed.
```

Phases 0-5 build the metadata foundation. Phases 6-7 build the agentic rendering tools on top.

---

## Current state

The SVG renderer (`core/svg_renderer.py`) produces valid SVG with four layer groups (`objects`, `graphic_symb`, `rays`, `labels`), but the element identifiers are **not aligned with the simulation**:

| Element | Current `id` | Problem |
|---|---|---|
| Ray segment | `ray-b0.847-w650` | Not unique (two rays with same brightness+wavelength collide). No uuid. No link to lineage. |
| Glass path | *(none)* | No `id` at all. Cannot be found in Inkscape's object list. |
| Glass edge | *(part of glass path)* | Edges are not individual elements. Cannot be selected or labeled independently. |
| Lens | *(none)* | No `id`. |
| Point source | *(none)* | No `id`. |
| Layer groups | `objects`, `rays`, `labels` | Bare ids only. Not recognized as Inkscape layers. |

### Rendering pipeline

There is **no centralized render method** on `Scene`. The pipeline is manual:

```python
renderer = SVGRenderer(width=800, height=600, viewbox=(...))
# User manually draws each object and ray
for obj in scene.objs:
    renderer.draw_glass_path(obj.path, ...)
for seg in segments:
    renderer.draw_ray_segment(seg, ...)
renderer.save('output.svg')
```

This design is intentional (gives users full control), and we preserve it. The refactoring only changes **what metadata is attached** to each SVG element, not the drawing pipeline.

### svgwrite profile

The renderer currently uses `profile='tiny'` which limits custom attributes. The comment at line 351 notes: *"data-\* attributes are not supported in svgwrite tiny profile"*. Switching to `profile='full'` enables `data-*` attributes and the Inkscape extension (`svgwrite.extensions.Inkscape`), which provides proper layer support and `inkscape:label`.

---

## SVG attributes we can use

| SVG attribute | Purpose | Inkscape support |
|---|---|---|
| `id` | Unique identifier (uuid) | Shown in Object Properties, used for navigation |
| `inkscape:label` | Human-readable name | Shown in Layers & Objects panel -- this is what users see |
| `class` | Category for CSS/scripting | Preserved, useful for batch selection |
| `data-*` | Arbitrary simulation metadata | Preserved in file, ignored by Inkscape UI |

The `inkscape:label` is the key attribute -- it's what appears in the Layers & Objects panel tree. The `id` provides the machine-readable link back to simulation uuids.

### Inkscape namespace

To use `inkscape:label`, the SVG root element needs the Inkscape namespace:
```xml
<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ...>
```

`svgwrite` provides this via `svgwrite.extensions.Inkscape`, or it can be added manually by registering the namespace on the drawing.

---

## Phase 0: Profile switch and Inkscape layer support

Switch from `profile='tiny'` to `profile='full'` and make the four layer groups proper Inkscape layers.

### Changes

**In `__init__`:**
```python
# Before
self.dwg = svgwrite.Drawing(size=(...), profile='tiny')
self.layer_objects = self.dwg.add(self.dwg.g(id='objects', transform='scale(1, -1)'))

# After
self.dwg = svgwrite.Drawing(size=(...), profile='full')
# Register Inkscape namespace
self.dwg['xmlns:inkscape'] = 'http://www.inkscape.org/namespaces/inkscape'

self.layer_objects = self.dwg.add(self.dwg.g(
    id='layer-objects',
    transform='scale(1, -1)',
    **{'inkscape:groupmode': 'layer', 'inkscape:label': 'Objects'}
))
# Same for layer_graphic_symb, layer_rays, layer_labels
```

### Risks

- `profile='full'` enables stricter SVG validation for some features. Need to verify that existing drawings still save without errors.
- Existing code that checks `'id="objects"'` in SVG strings (e.g. the `__main__` test block at line 1240) needs updating to `'id="layer-objects"'`.

### Verification

- Generate an SVG with the current examples (e.g. `tir_demo.py`), open in Inkscape, confirm the four layers appear in the Layers panel with readable names.

---

## Phase 1: Ray segment metadata

Add simulation-aligned identifiers to ray segments. This is the highest-value change.

### Changes to `draw_ray_segment`

```python
def draw_ray_segment(self, ray, color='red', opacity=1.0, stroke_width=1.5,
                     extend_to_edge=False, draw_gap_rays=False, show_arrow=False,
                     arrow_size=None, arrow_position=0.67):

    # Build id from ray uuid (guaranteed unique per segment)
    ray_uuid = getattr(ray, 'uuid', None)
    ray_id = f'ray-{ray_uuid}' if ray_uuid else f'ray-b{ray.total_brightness:.3f}'

    # Build human-readable label for Inkscape
    label_parts = []
    if ray.wavelength is not None:
        label_parts.append(f'{ray.wavelength:.0f}nm')
    label_parts.append(f'b={ray.total_brightness:.3f}')
    interaction = getattr(ray, 'interaction_type', None)
    if interaction:
        label_parts.append(interaction)
    ray_label = ' '.join(label_parts)

    # ... existing drawing code ...

    line = self.dwg.line(
        start=(p1['x'], p1['y']),
        end=(p2['x'], p2['y']),
        stroke=color,
        stroke_width=stroke_width,
        stroke_opacity=opacity,
        id=ray_id,
    )
    line['class'] = 'ray'
    line['inkscape:label'] = ray_label

    # Optional: data attributes for scripting
    if ray_uuid:
        line['data-uuid'] = ray_uuid
    if ray.wavelength is not None:
        line['data-wavelength'] = str(ray.wavelength)
    line['data-brightness-s'] = f'{ray.brightness_s:.6f}'
    line['data-brightness-p'] = f'{ray.brightness_p:.6f}'
    parent_uuid = getattr(ray, 'parent_uuid', None)
    if parent_uuid:
        line['data-parent-uuid'] = parent_uuid

    self.layer_rays.add(line)
```

### Same pattern for `_draw_ray_with_arrow`

The arrow case creates a `<g>` group. The group gets the `id`, `inkscape:label`, `class`, and `data-*` attributes; the child line/polygon elements inside it don't need ids.

### Impact on call sites

**None.** The `ray` parameter already carries uuid, interaction_type, wavelength, brightness. The method signature doesn't change. Existing example code continues to work -- it just gets richer SVG output.

### What you see in Inkscape

In the Layers & Objects panel, expanding the "Rays" layer shows:
```
▼ Rays
   650nm b=0.847 refract
   650nm b=0.423 reflect
   546nm b=1.000 source
   ...
```

Each item is clickable and highlights the corresponding ray on the canvas.

---

## Phase 2: Scene object metadata

Add identifiers to glass paths, lenses, points, and line segments.

### `draw_glass_path`

```python
def draw_glass_path(self, path, fill='cyan', fill_opacity=0.3, stroke='navy',
                    stroke_width=2, label=None, glass_obj=None):
    # ... existing path building code ...

    glass_path = self.dwg.path(d=path_string, fill=fill, ...)

    # Add metadata if glass_obj is provided
    if glass_obj is not None:
        obj_uuid = getattr(glass_obj, 'uuid', None)
        obj_name = getattr(glass_obj, 'name', None) or getattr(glass_obj, 'get_display_name', lambda: None)()
        if obj_uuid:
            glass_path['id'] = f'glass-{obj_uuid}'
            glass_path['data-uuid'] = obj_uuid
        if obj_name:
            glass_path['inkscape:label'] = obj_name
        glass_path['class'] = 'glass'

    self.layer_objects.add(glass_path)
```

**Backward compatibility:** The new `glass_obj=None` parameter is optional. Existing calls without it produce the same output as before. New calls pass the object for richer metadata.

### `draw_line_segment`, `draw_lens`, `draw_point`

Same pattern: add optional `scene_obj=None` parameter. When provided, attach `id`, `inkscape:label`, `class`, and `data-uuid`.

### Impact on call sites

Call sites that want metadata pass the scene object:
```python
# Before
renderer.draw_glass_path(prism.path, fill='cyan', label='Prism')

# After (backward compatible -- old calls still work)
renderer.draw_glass_path(prism.path, fill='cyan', label='Prism', glass_obj=prism)
```

---

## Phase 3: Glass edge tagging

Individual glass edges are currently drawn as part of a single `<path>` element. To make them individually addressable, we add edge overlay elements.

### Approach: Edge overlays (not path splitting)

Splitting the glass `<path>` into individual edge segments would break arc rendering and fill behavior. Instead, draw invisible (or very thin) overlay `<line>` elements on top of each edge, inside a sub-group:

```python
def draw_glass_edge_overlays(self, glass_obj, stroke='none', stroke_width=0):
    """
    Add invisible overlay elements for each labeled edge of a glass object.

    These elements carry edge metadata (id, inkscape:label, data-* attributes)
    and can be selected in Inkscape to find specific edges. They are drawn
    with no visible stroke by default (purely for metadata), but can be made
    visible for debugging.
    """
    edges = get_edge_descriptions(glass_obj)
    obj_uuid = getattr(glass_obj, 'uuid', '')

    edge_group = self.dwg.g(
        id=f'edges-{obj_uuid}',
    )
    edge_group['inkscape:label'] = f'Edges: {getattr(glass_obj, "name", obj_uuid[:12])}'
    edge_group['class'] = 'glass-edges'

    for edge in edges:
        line = self.dwg.line(
            start=(edge.p1.x, edge.p1.y),
            end=(edge.p2.x, edge.p2.y),
            stroke=stroke,
            stroke_width=stroke_width,
        )
        line['id'] = f'{obj_uuid}-edge-{edge.index}'
        line['inkscape:label'] = edge.short_label or f'edge-{edge.index}'
        line['class'] = 'glass-edge'
        line['data-edge-index'] = str(edge.index)
        if edge.long_label:
            line['data-long-label'] = edge.long_label
        edge_group.add(line)

    self.layer_objects.add(edge_group)
```

### What you see in Inkscape

```
▼ Objects
   ▼ Prism
      (glass fill+stroke path)
   ▼ Edges: Prism
      S
      NE
      NW
```

Each edge label is selectable. When selected, it highlights the edge on the canvas (if given a visible stroke) or shows its position in the XML editor.

### Note on arc edges

`get_edge_descriptions` returns `p1` and `p2` for each edge. For line edges, a `<line>` overlay is exact. For arc edges, the overlay would be a straight-line approximation of the arc. If precise arc overlays are needed, this can be extended later using the same arc calculation from `draw_glass_path`. For metadata/navigation purposes, the straight-line approximation is sufficient.

---

## Phase 4: Optional `data-*` richness control

Adding full `data-*` attributes to every ray increases SVG file size. For a 10,000-segment simulation, each ray gains ~200 bytes of metadata, adding ~2MB. This is acceptable for most cases, but a control flag is useful.

### `__init__` parameter

```python
def __init__(self, width=800, height=600, viewbox=None, metadata_level='standard'):
    """
    Args:
        metadata_level: Controls how much simulation metadata to embed.
            - 'none': No simulation metadata (current behavior, smallest files)
            - 'standard': id + inkscape:label + class (good for Inkscape navigation)
            - 'full': All of 'standard' plus data-* attributes (good for scripting)
    """
    self.metadata_level = metadata_level
```

| Level | `id` | `inkscape:label` | `class` | `data-*` | Typical overhead per ray |
|---|---|---|---|---|---|
| `'none'` | generic | no | no | no | ~30 bytes (current) |
| `'standard'` | uuid-based | yes | yes | no | ~100 bytes |
| `'full'` | uuid-based | yes | yes | yes | ~250 bytes |

---

## Phase 5: Convenience method `draw_scene`

Once all element types carry proper metadata, add an optional convenience method that draws an entire scene in one call. This does **not** replace the manual pipeline -- it's a shortcut for the common case.

```python
def draw_scene(self, scene, segments=None, draw_rays=True, draw_objects=True,
               draw_edge_labels=True, draw_edge_overlays=True, **ray_kwargs):
    """
    Draw all objects and rays from a scene with full metadata.

    This is a convenience method. For fine-grained control over appearance,
    use the individual draw_* methods directly.
    """
    if draw_objects:
        for obj in scene.objs:
            if hasattr(obj, 'path') and len(obj.path) >= 3:
                self.draw_glass_path(obj.path, label=obj.get_display_name(), glass_obj=obj)
                if draw_edge_overlays:
                    self.draw_glass_edge_overlays(obj)
                if draw_edge_labels:
                    self.draw_glass_edge_labels(obj)
            elif hasattr(obj, 'focal_length'):
                self.draw_lens(obj.p1, obj.p2, obj.focal_length,
                              label=obj.get_display_name(), scene_obj=obj)
            # ... other object types ...

    if draw_rays and segments:
        for seg in segments:
            self.draw_ray_with_scene_settings(seg, scene, **ray_kwargs)
```

### Design notes

- Object type detection uses duck typing (`hasattr(obj, 'path')`, `hasattr(obj, 'focal_length')`) rather than isinstance checks, to avoid importing all scene object types into the renderer.
- This method is optional and additive. It doesn't change the existing manual pipeline.

---

## Implementation order and dependencies

```
Phase 0: Profile switch + Inkscape layers
   │     (no API changes, foundation for all subsequent phases)
   │
   ├── Phase 1: Ray metadata        (highest value, standalone)
   ├── Phase 2: Object metadata      (standalone, parallel with Phase 1)
   └── Phase 3: Edge overlays        (depends on Phase 0 for namespace)
         │
         Phase 4: Metadata level flag (refines Phases 1-3)
         │
         Phase 5: draw_scene          (depends on Phases 1-3)
```

Phases 1 and 2 can be implemented in parallel. Phase 3 depends on Phase 0 for namespace support. Phase 4 is a refinement pass. Phase 5 is optional convenience.

---

## Existing call sites (for reference)

Files that use SVGRenderer and will benefit from the refactoring (no breaking changes -- all new parameters are optional):

| File | Methods used |
|---|---|
| `examples/tir_demo/tir_demo.py` | `draw_ray_segment`, `draw_line_segment`, `draw_glass_edge_labels` |
| `examples/collimation_demo/demo.py` | `draw_ray_segment`, `draw_lens`, `draw_point`, `draw_line_segment` |
| `examples/coupled_prisms/uncoupled_copled_prisms.py` | `draw_ray_segment`, `draw_ray_with_scene_settings`, `draw_line_segment`, `draw_lens` |
| `examples/bulk_lens_demo/bulk_lens_collimation_demo.py` | `draw_ray_segment`, `draw_glass_path`, `draw_point`, `draw_line_segment` |
| `examples/bulk_lens_demo/lens_validator.py` | `draw_line_segment` |

---

## Phase 6: Highlight rendering on SVGRenderer

The core rendering capability: draw a full scene with specific elements visually highlighted. This is a method on `SVGRenderer`, not an agentic tool -- it's the building block that agentic tools will call.

### Key concept: two-pass rendering

The highlight workflow renders the scene in two passes:

1. **Base pass**: Draw everything (objects + all rays) in a dimmed/muted style
2. **Highlight pass**: Re-draw only the selected elements in a vivid highlight style

This produces a scene where the context is visible but the highlighted elements pop out visually.

### Proposed method

```python
def draw_scene_with_highlights(
    self,
    scene: 'Scene',
    segments: List['Ray'],
    highlight_ray_uuids: Optional[Set[str]] = None,
    highlight_edge_specs: Optional[List[Tuple[str, str]]] = None,
    highlight_glass_names: Optional[List[str]] = None,
    highlight_color: str = 'yellow',
    highlight_stroke_width: float = 3.0,
    dim_opacity: float = 0.15,
    base_ray_color: str = 'gray',
    edge_highlight_color: str = 'lime',
    edge_highlight_width: float = 4.0,
):
    """
    Render a full scene with specific elements highlighted.

    Args:
        scene: The Scene object.
        segments: All ray segments from the simulation.
        highlight_ray_uuids: Set of ray uuids to highlight. If None, no
            rays are highlighted (all drawn normally).
        highlight_edge_specs: List of (glass_name, edge_label) tuples.
            Matched edges are drawn with a vivid overlay.
        highlight_glass_names: List of glass object names to highlight
            (drawn with a stronger fill/stroke).
        highlight_color: CSS color for highlighted rays (default: 'yellow').
        highlight_stroke_width: Stroke width for highlighted rays.
        dim_opacity: Opacity for non-highlighted rays (default: 0.15).
        base_ray_color: Color for non-highlighted rays (default: 'gray').
        edge_highlight_color: Color for highlighted edges (default: 'lime').
        edge_highlight_width: Stroke width for highlighted edges.
    """
    # --- Pass 1: Draw all objects normally ---
    for obj in scene.objs:
        if hasattr(obj, 'path') and len(obj.path) >= 3:
            name = getattr(obj, 'name', None)
            is_highlighted = (highlight_glass_names and name in highlight_glass_names)
            fill_opacity = 0.5 if is_highlighted else 0.2
            self.draw_glass_path(obj.path, fill_opacity=fill_opacity, glass_obj=obj)
            self.draw_glass_edge_labels(obj)
        # ... other object types ...

    # --- Pass 2: Draw all rays dimmed ---
    highlight_set = highlight_ray_uuids or set()
    for seg in segments:
        seg_uuid = getattr(seg, 'uuid', None)
        if seg_uuid and seg_uuid in highlight_set:
            continue  # Skip highlighted rays in the dim pass
        self.draw_ray_segment(seg, color=base_ray_color, opacity=dim_opacity,
                              stroke_width=1.0)

    # --- Pass 3: Draw highlighted rays on top ---
    for seg in segments:
        seg_uuid = getattr(seg, 'uuid', None)
        if seg_uuid and seg_uuid in highlight_set:
            self.draw_ray_segment(seg, color=highlight_color,
                                  opacity=1.0, stroke_width=highlight_stroke_width)

    # --- Pass 4: Draw highlighted edges on top ---
    if highlight_edge_specs:
        for glass_name, edge_label in highlight_edge_specs:
            glass = get_object_by_name(scene, glass_name)
            edge = _resolve_edge(glass, edge_label)
            line = self.dwg.line(
                start=(edge.p1.x, edge.p1.y),
                end=(edge.p2.x, edge.p2.y),
                stroke=edge_highlight_color,
                stroke_width=edge_highlight_width,
                stroke_opacity=0.9,
            )
            line['inkscape:label'] = f'highlight: {glass_name} {edge_label}'
            line['class'] = 'highlight-edge'
            self.layer_objects.add(line)
```

### Why highlight rays are drawn last

SVG renders in document order (later elements on top). Drawing highlighted rays after the dimmed rays ensures they are visually on top, even if they overlap with other rays. The highlighted edges are also drawn last, on top of the glass paths.

### Highlight layers

For better Inkscape navigation, the highlighted elements could go into a dedicated layer:

```python
self.layer_highlights = self.dwg.add(self.dwg.g(
    id='layer-highlights',
    transform='scale(1, -1)',
    **{'inkscape:groupmode': 'layer', 'inkscape:label': 'Highlights'}
))
```

This lets the user toggle highlights on/off in Inkscape by hiding the layer.

---

## Phase 7: Agentic highlight tools

Wrap Phase 6 as string-in/string-out functions in `agentic_tools.py`, following the existing pattern (context-based, JSON-serializable parameters, returns a string).

### How the existing agentic tools work

The current tools in `agentic_tools.py` follow this pattern:
```python
# Module-level context (set once after simulation)
_CONTEXT = {}  # {'scene': Scene, 'segments': List[Ray], 'lineage': RayLineage}

def set_context_from_result(scene, result):
    _CONTEXT['scene'] = scene
    _CONTEXT['segments'] = result.segments
    _CONTEXT['lineage'] = result.lineage

# Each tool: string params -> resolve from context -> call analysis function -> return string
def find_rays_crossing_edge_xml(glass_name: str, edge_label: str) -> str:
    scene, segments = _require_context()
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_crossing_edge(segments, glass, edge_label)
    return rays_to_xml(rays)
```

### Proposed agentic SVG tools

These tools compose the **query** step (find the rays) with the **render** step (produce the SVG) in a single call. The agent doesn't need to manage intermediate state.

```python
def render_scene_svg(
    width: int = 800,
    height: int = 600,
    viewbox: str = "auto"
) -> str:
    """
    Render the full scene and all rays as an SVG string.

    This is the baseline rendering -- no highlights, just the scene as-is.
    Useful as a "show me the current state" tool.

    Args:
        width: SVG width in pixels.
        height: SVG height in pixels.
        viewbox: Viewbox as "min_x,min_y,width,height" or "auto" to compute
            from scene bounds.

    Returns:
        SVG string.
    """
    scene, segments = _require_context()
    renderer = SVGRenderer(width=width, height=height, viewbox=_parse_viewbox(viewbox, scene))
    renderer.draw_scene(scene, segments)
    return renderer.to_string()


def highlight_rays_inside_glass_svg(
    glass_name: str,
    highlight_color: str = 'yellow',
    width: int = 800,
    height: int = 600,
    viewbox: str = "auto"
) -> str:
    """
    Render the scene with rays inside a glass object highlighted.

    Combines find_rays_inside_glass() with highlight rendering.

    Args:
        glass_name: Name of the glass object.
        highlight_color: CSS color for highlighted rays.

    Returns:
        SVG string with highlighted rays.
    """
    scene, segments = _require_context()
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_inside_glass(segments, glass)
    ray_uuids = {r.uuid for r in rays}

    renderer = SVGRenderer(width=width, height=height, viewbox=_parse_viewbox(viewbox, scene))
    renderer.draw_scene_with_highlights(
        scene, segments,
        highlight_ray_uuids=ray_uuids,
        highlight_glass_names=[glass_name],
        highlight_color=highlight_color,
    )
    return renderer.to_string()


def highlight_rays_crossing_edge_svg(
    glass_name: str,
    edge_label: str,
    highlight_color: str = 'yellow',
    width: int = 800,
    height: int = 600,
    viewbox: str = "auto"
) -> str:
    """
    Render the scene with rays crossing a specific edge highlighted.

    Combines find_rays_crossing_edge() with highlight rendering.
    Also highlights the edge itself.

    Args:
        glass_name: Name of the glass object.
        edge_label: Label of the edge.
        highlight_color: CSS color for highlighted rays.

    Returns:
        SVG string with highlighted rays and edge.
    """
    scene, segments = _require_context()
    glass = get_object_by_name(scene, glass_name)
    rays = find_rays_crossing_edge(segments, glass, edge_label)
    ray_uuids = {r.uuid for r in rays}

    renderer = SVGRenderer(width=width, height=height, viewbox=_parse_viewbox(viewbox, scene))
    renderer.draw_scene_with_highlights(
        scene, segments,
        highlight_ray_uuids=ray_uuids,
        highlight_edge_specs=[(glass_name, edge_label)],
        highlight_color=highlight_color,
    )
    return renderer.to_string()


def highlight_rays_by_polarization_svg(
    min_dop: float = 0.0,
    max_dop: float = 1.0,
    highlight_color: str = 'magenta',
    width: int = 800,
    height: int = 600,
    viewbox: str = "auto"
) -> str:
    """
    Render the scene with rays filtered by degree of polarization highlighted.

    Args:
        min_dop: Minimum degree of polarization.
        max_dop: Maximum degree of polarization.
        highlight_color: CSS color for highlighted rays.

    Returns:
        SVG string with highlighted rays.
    """
    scene, segments = _require_context()
    rays = find_rays_by_polarization(segments, min_dop, max_dop)
    ray_uuids = {r.uuid for r in rays}

    renderer = SVGRenderer(width=width, height=height, viewbox=_parse_viewbox(viewbox, scene))
    renderer.draw_scene_with_highlights(
        scene, segments,
        highlight_ray_uuids=ray_uuids,
        highlight_color=highlight_color,
    )
    return renderer.to_string()


def highlight_custom_rays_svg(
    ray_uuids_csv: str,
    highlight_color: str = 'yellow',
    width: int = 800,
    height: int = 600,
    viewbox: str = "auto"
) -> str:
    """
    Render the scene with a specific set of rays highlighted by uuid.

    This is the escape hatch: the agent can compose arbitrary queries,
    collect uuids, and pass them here for visualization.

    Args:
        ray_uuids_csv: Comma-separated ray uuids to highlight.
        highlight_color: CSS color for highlighted rays.

    Returns:
        SVG string.
    """
    scene, segments = _require_context()
    ray_uuids = set(u.strip() for u in ray_uuids_csv.split(',') if u.strip())

    renderer = SVGRenderer(width=width, height=height, viewbox=_parse_viewbox(viewbox, scene))
    renderer.draw_scene_with_highlights(
        scene, segments,
        highlight_ray_uuids=ray_uuids,
        highlight_color=highlight_color,
    )
    return renderer.to_string()
```

### Design notes

**Why dedicated functions instead of one generic highlight tool?**

The LLM agent needs to decide which tool to call based on the user's natural language request. Dedicated functions like `highlight_rays_crossing_edge_svg(glass_name, edge_label)` are easier for the LLM to map to than a generic `render_highlight_svg(query_type, param1, param2, ...)`. Each tool name describes exactly what it does.

The `highlight_custom_rays_svg(ray_uuids_csv)` is the generic fallback for cases where the agent composes a multi-step query (e.g. "rays inside glass A that also cross edge B of glass C") and wants to visualize the intersection.

**String-only parameters:**

Following the existing agentic_tools pattern, all parameters are JSON-serializable strings/numbers. The `ray_uuids_csv` parameter uses comma-separated strings rather than a list, because tool-calling frameworks typically handle flat strings better than arrays.

**Viewbox auto-computation:**

The `viewbox="auto"` mode would compute a bounding box from all scene objects plus some padding. This avoids requiring the agent to know viewport coordinates. A helper `_parse_viewbox(viewbox_str, scene)` would handle both explicit `"0,0,400,300"` and `"auto"`.

**SVG as return value, not file:**

The tools return SVG strings (via `renderer.to_string()`), not files. The GUI framework (solve.it, claudette, etc.) is responsible for displaying the SVG. This keeps the tools stateless and framework-agnostic.

### How the GUI displays it

The exact mechanism depends on the framework:

- **solve.it notebooks**: SVG string can be rendered inline via `IPython.display.SVG(svg_string)` or embedded in an HTML cell
- **claudette/lisette**: The tool result (SVG string) would need a UI component that renders SVG. This might require a convention like wrapping the SVG in a special tag that the UI recognizes
- **Web-based UIs**: The SVG string can be injected directly into the DOM

The agentic tools don't need to know about the display mechanism -- they just produce the SVG string.

---

## Updated implementation order

```
Phase 0: Profile switch + Inkscape layers
   │     (foundation)
   │
   ├── Phase 1: Ray metadata              (highest value)
   ├── Phase 2: Object metadata            (parallel with Phase 1)
   └── Phase 3: Edge overlays              (depends on Phase 0)
         │
         Phase 4: Metadata level flag      (refinement)
         │
         Phase 5: draw_scene               (convenience, depends on 1-3)
         │
         Phase 6: draw_scene_with_highlights  (core highlight capability)
         │
         Phase 7: Agentic SVG tools          (string-in/SVG-out wrappers)
```

Phases 0-5 are the metadata foundation. Phase 6 is the highlight renderer. Phase 7 is the agentic API that ties everything together.

---

## Relationship to other roadmaps

- **`rays_and_geometry_analysis.md`**: Phase 7 tools compose the query functions from that roadmap (find_rays_inside_glass, find_rays_crossing_edge, etc.) with SVG rendering.
- **`various_tools_for_analysis.md`**: The tool registry (`list_available_tools`) should include the Phase 7 agentic SVG tools once implemented.
- **`ray_lineage_tracking.md`**: Lineage-based queries can feed into `highlight_custom_rays_svg` by extracting uuids from path analysis results.
