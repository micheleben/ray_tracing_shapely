"""
Original work Copyright 2024 The Ray Optics Simulation authors and contributors
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

===============================================================================
PYTHON-SPECIFIC MODULE: Simulation Result Container
===============================================================================
This module provides a container for simulation results that captures:
- The ray segments produced by the simulation
- The scene configuration at simulation time
- Metadata for correlating results across multiple runs

This enables LLM-friendly conversations about multiple simulations and their
relationships to scene configurations.
===============================================================================
"""

import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.ray import Ray
    from ..core.scene import Scene


@dataclass
class SceneSnapshot:
    """
    Captures the state of a scene at a specific point in time.

    This is a lightweight snapshot that records identifying information
    and key settings, not a full serialization of the scene.

    Attributes:
        uuid: The scene's UUID at snapshot time
        name: The scene's display name
        object_count: Total number of objects in the scene
        optical_object_count: Number of optical objects
        object_uuids: List of UUIDs for all objects in the scene
        object_summary: Brief description of objects (e.g., "2 Glass, 1 PointSource")
        settings: Dictionary of key scene settings
    """
    uuid: str
    name: str
    object_count: int
    optical_object_count: int
    object_uuids: List[str]
    object_summary: str
    settings: Dict[str, Any]

    @classmethod
    def from_scene(cls, scene: 'Scene') -> 'SceneSnapshot':
        """
        Create a snapshot from a Scene object.

        Args:
            scene: The Scene to snapshot

        Returns:
            A SceneSnapshot capturing the scene's current state
        """
        # Collect object UUIDs
        object_uuids = []
        type_counts: Dict[str, int] = {}

        for obj in scene.objs:
            if hasattr(obj, 'uuid'):
                object_uuids.append(obj.uuid)

            # Count by type
            type_name = getattr(obj, 'type', obj.__class__.__name__)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Build object summary
        summary_parts = [f"{count} {name}" for name, count in sorted(type_counts.items())]
        object_summary = ", ".join(summary_parts) if summary_parts else "empty"

        # Capture key settings
        settings = {
            'mode': scene.mode,
            'color_mode': scene.color_mode,
            'simulate_colors': scene.simulate_colors,
            'ray_density': scene.ray_density,
            'length_scale': scene.length_scale,
            'min_brightness_exp': scene.min_brightness_exp,
        }

        return cls(
            uuid=scene.uuid,
            name=scene.get_display_name(),
            object_count=len(scene.objs),
            optical_object_count=len(scene.optical_objs),
            object_uuids=object_uuids,
            object_summary=object_summary,
            settings=settings,
        )


@dataclass
class SimulationResult:
    """
    Container for simulation results with full context.

    This class wraps the ray segments returned by a simulation run along with
    metadata that allows correlating results with scene configurations. It
    enables conversations like:
    - "In simulation sim_abc123, which rays hit the prism?"
    - "Compare results between sim_001 and sim_002"
    - "What scene settings were used for sim_abc123?"

    Attributes:
        uuid: Unique identifier for this simulation run
        name: Optional human-readable name for this run
        timestamp: ISO format timestamp when simulation completed
        scene_snapshot: Snapshot of scene state at simulation time
        segments: The ray segments produced by the simulation
        max_rays: Maximum rays limit used for the simulation
        processed_ray_count: Number of rays actually processed
        total_truncation: Sum of truncated ray brightness
        undefined_behavior_count: Count of undefined behavior incidents
        warnings: List of warning messages from the simulation
        error: Error message if simulation failed, None otherwise
    """
    # Identification
    uuid: str
    name: Optional[str]
    timestamp: str

    # Scene context
    scene_snapshot: SceneSnapshot

    # Results
    segments: List['Ray']

    # Simulation parameters
    max_rays: int

    # Statistics
    processed_ray_count: int
    total_truncation: float = 0.0
    undefined_behavior_count: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def create(
        cls,
        scene: 'Scene',
        segments: List['Ray'],
        max_rays: int,
        processed_ray_count: int,
        total_truncation: float = 0.0,
        undefined_behavior_count: int = 0,
        name: Optional[str] = None
    ) -> 'SimulationResult':
        """
        Create a SimulationResult from simulation outputs.

        This is the primary factory method for creating SimulationResult objects.

        Args:
            scene: The Scene that was simulated
            segments: The ray segments produced
            max_rays: Maximum rays limit used
            processed_ray_count: Number of rays processed
            total_truncation: Sum of truncated brightness (default: 0.0)
            undefined_behavior_count: Count of undefined behaviors (default: 0)
            name: Optional name for this simulation run

        Returns:
            A new SimulationResult instance
        """
        # Collect warnings
        warnings = []
        if scene.warning:
            warnings.append(scene.warning)

        return cls(
            uuid=str(uuid_module.uuid4()),
            name=name,
            timestamp=datetime.now().isoformat(),
            scene_snapshot=SceneSnapshot.from_scene(scene),
            segments=segments,
            max_rays=max_rays,
            processed_ray_count=processed_ray_count,
            total_truncation=total_truncation,
            undefined_behavior_count=undefined_behavior_count,
            warnings=warnings,
            error=scene.error,
        )

    def get_display_name(self) -> str:
        """
        Get a display name for this simulation result.

        Returns the user-defined name if set, otherwise returns a combination
        of "Simulation" and a short UUID suffix.

        Returns:
            A string suitable for display (e.g., "TIR Test Run 1" or "Simulation_a1b2c3d4").
        """
        if self.name:
            return self.name
        short_uuid = self.uuid[:8]
        return f"Simulation_{short_uuid}"

    @property
    def segment_count(self) -> int:
        """Get the number of ray segments in the result."""
        return len(self.segments)

    @property
    def success(self) -> bool:
        """Check if the simulation completed successfully (no error)."""
        return self.error is None

    @property
    def hit_ray_limit(self) -> bool:
        """Check if the simulation hit the maximum ray limit."""
        return self.processed_ray_count >= self.max_rays

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "OK" if self.success else "ERROR"
        return (f"SimulationResult('{self.get_display_name()}', "
                f"segments={self.segment_count}, "
                f"rays={self.processed_ray_count}/{self.max_rays}, "
                f"status={status})")


def _escape_xml(text: str) -> str:
    """Escape special characters for XML."""
    if text is None:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def describe_simulation_result(
    result: SimulationResult,
    format: str = 'xml',
    include_segments: bool = False,
    max_segments: int = 100
) -> str:
    """
    Generate a formatted description of a simulation result.

    Args:
        result: The SimulationResult to describe
        format: Output format - 'xml' for XML, 'text' for human-readable
        include_segments: If True, include individual segment data
        max_segments: Maximum number of segments to include (default: 100)

    Returns:
        Formatted string describing the simulation result

    Example (XML format):
        >>> print(describe_simulation_result(result))
        <?xml version="1.0" encoding="UTF-8"?>
        <simulation_result>
          <identification>
            <uuid>abc123...</uuid>
            <name>TIR Test</name>
            <timestamp>2026-01-21T10:30:00</timestamp>
          </identification>
          <scene_snapshot>
            <uuid>def456...</uuid>
            <name>Scene_def456</name>
            <object_count>3</object_count>
            <object_summary>1 Glass, 1 PointSource, 1 Blocker</object_summary>
            <settings>
              <mode>rays</mode>
              <ray_density>0.1</ray_density>
              ...
            </settings>
          </scene_snapshot>
          <results>
            <segment_count>150</segment_count>
            <processed_ray_count>150</processed_ray_count>
            <max_rays>10000</max_rays>
            <hit_ray_limit>false</hit_ray_limit>
          </results>
          <status>
            <success>true</success>
            <warnings/>
          </status>
        </simulation_result>
    """
    if format == 'xml':
        return _describe_result_xml(result, include_segments, max_segments)
    else:
        return _describe_result_text(result, include_segments, max_segments)


def _describe_result_xml(
    result: SimulationResult,
    include_segments: bool,
    max_segments: int
) -> str:
    """Generate XML format for simulation result description."""
    lines = []

    # XML header
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<simulation_result>')

    # Identification section
    lines.append('  <identification>')
    lines.append(f'    <uuid>{_escape_xml(result.uuid)}</uuid>')
    if result.name:
        lines.append(f'    <name>{_escape_xml(result.name)}</name>')
    lines.append(f'    <display_name>{_escape_xml(result.get_display_name())}</display_name>')
    lines.append(f'    <timestamp>{_escape_xml(result.timestamp)}</timestamp>')
    lines.append('  </identification>')

    # Scene snapshot section
    snap = result.scene_snapshot
    lines.append('  <scene_snapshot>')
    lines.append(f'    <uuid>{_escape_xml(snap.uuid)}</uuid>')
    lines.append(f'    <name>{_escape_xml(snap.name)}</name>')
    lines.append(f'    <object_count>{snap.object_count}</object_count>')
    lines.append(f'    <optical_object_count>{snap.optical_object_count}</optical_object_count>')
    lines.append(f'    <object_summary>{_escape_xml(snap.object_summary)}</object_summary>')

    # Scene settings
    lines.append('    <settings>')
    for key, value in snap.settings.items():
        lines.append(f'      <{key}>{_escape_xml(str(value))}</{key}>')
    lines.append('    </settings>')

    # Object UUIDs (abbreviated if many)
    if snap.object_uuids:
        lines.append('    <object_uuids>')
        for obj_uuid in snap.object_uuids[:20]:  # Limit to first 20
            lines.append(f'      <object_uuid>{_escape_xml(obj_uuid)}</object_uuid>')
        if len(snap.object_uuids) > 20:
            lines.append(f'      <!-- ... and {len(snap.object_uuids) - 20} more -->')
        lines.append('    </object_uuids>')

    lines.append('  </scene_snapshot>')

    # Results section
    lines.append('  <results>')
    lines.append(f'    <segment_count>{result.segment_count}</segment_count>')
    lines.append(f'    <processed_ray_count>{result.processed_ray_count}</processed_ray_count>')
    lines.append(f'    <max_rays>{result.max_rays}</max_rays>')
    lines.append(f'    <hit_ray_limit>{str(result.hit_ray_limit).lower()}</hit_ray_limit>')
    lines.append(f'    <total_truncation>{result.total_truncation:.6f}</total_truncation>')
    lines.append(f'    <undefined_behavior_count>{result.undefined_behavior_count}</undefined_behavior_count>')
    lines.append('  </results>')

    # Status section
    lines.append('  <status>')
    lines.append(f'    <success>{str(result.success).lower()}</success>')
    if result.error:
        lines.append(f'    <error>{_escape_xml(result.error)}</error>')
    if result.warnings:
        lines.append('    <warnings>')
        for warning in result.warnings:
            lines.append(f'      <warning>{_escape_xml(warning)}</warning>')
        lines.append('    </warnings>')
    else:
        lines.append('    <warnings/>')
    lines.append('  </status>')

    # Optional segments section
    if include_segments and result.segments:
        lines.append('  <segments>')
        for i, seg in enumerate(result.segments[:max_segments]):
            brightness = seg.brightness_s + seg.brightness_p
            tir_attrs = ""
            if getattr(seg, 'is_tir_result', False):
                tir_attrs += ' is_tir_result="true"'
            if getattr(seg, 'caused_tir', False):
                tir_attrs += ' caused_tir="true"'

            lines.append(f'    <segment index="{i}" brightness="{brightness:.4f}"{tir_attrs}>')
            lines.append(f'      <p1 x="{seg.p1["x"]:.4f}" y="{seg.p1["y"]:.4f}"/>')
            lines.append(f'      <p2 x="{seg.p2["x"]:.4f}" y="{seg.p2["y"]:.4f}"/>')
            if seg.wavelength is not None:
                lines.append(f'      <wavelength>{seg.wavelength}</wavelength>')
            lines.append('    </segment>')

        if len(result.segments) > max_segments:
            lines.append(f'    <!-- ... and {len(result.segments) - max_segments} more segments -->')
        lines.append('  </segments>')

    lines.append('</simulation_result>')

    return "\n".join(lines)


def _describe_result_text(
    result: SimulationResult,
    include_segments: bool,
    max_segments: int
) -> str:
    """Generate human-readable text format for simulation result description."""
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Simulation Result: {result.get_display_name()}")
    lines.append("=" * 70)

    # Identification
    lines.append(f"\nUUID: {result.uuid}")
    lines.append(f"Timestamp: {result.timestamp}")

    # Scene info
    snap = result.scene_snapshot
    lines.append(f"\nScene: {snap.name} (uuid: {snap.uuid[:8]}...)")
    lines.append(f"Objects: {snap.object_summary}")
    lines.append(f"Total objects: {snap.object_count} ({snap.optical_object_count} optical)")

    # Settings
    lines.append("\nScene Settings:")
    for key, value in snap.settings.items():
        lines.append(f"  {key}: {value}")

    # Results
    lines.append("\nResults:")
    lines.append(f"  Segments: {result.segment_count}")
    lines.append(f"  Processed rays: {result.processed_ray_count} / {result.max_rays}")
    if result.hit_ray_limit:
        lines.append("  *** HIT RAY LIMIT ***")
    lines.append(f"  Total truncation: {result.total_truncation:.6f}")
    lines.append(f"  Undefined behaviors: {result.undefined_behavior_count}")

    # Status
    lines.append("\nStatus:")
    lines.append(f"  Success: {result.success}")
    if result.error:
        lines.append(f"  Error: {result.error}")
    if result.warnings:
        lines.append("  Warnings:")
        for warning in result.warnings:
            lines.append(f"    - {warning}")

    # Optional segments
    if include_segments and result.segments:
        lines.append(f"\nSegments (first {min(max_segments, len(result.segments))} of {len(result.segments)}):")
        lines.append("-" * 70)
        lines.append(f"{'Index':>6} | {'Brightness':>10} | {'P1':>20} | {'P2':>20} | TIR")
        lines.append("-" * 70)

        for i, seg in enumerate(result.segments[:max_segments]):
            brightness = seg.brightness_s + seg.brightness_p
            p1_str = f"({seg.p1['x']:.2f}, {seg.p1['y']:.2f})"
            p2_str = f"({seg.p2['x']:.2f}, {seg.p2['y']:.2f})"
            tir_str = ""
            if getattr(seg, 'is_tir_result', False):
                tir_str = "TIR"
            if getattr(seg, 'caused_tir', False):
                tir_str += "->TIR"
            lines.append(f"{i:>6} | {brightness:>10.4f} | {p1_str:>20} | {p2_str:>20} | {tir_str}")

        if len(result.segments) > max_segments:
            lines.append(f"  ... and {len(result.segments) - max_segments} more segments")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    print("Testing SimulationResult class...\n")

    # Create mock objects for testing
    class MockScene:
        def __init__(self):
            self._uuid = str(uuid_module.uuid4())
            self.name = "Test Scene"
            self.objs = []
            self.optical_objs = []
            self.mode = 'rays'
            self.color_mode = 'default'
            self.simulate_colors = False
            self.ray_density = 0.1
            self.length_scale = 1.0
            self.min_brightness_exp = None
            self.warning = None
            self.error = None

        @property
        def uuid(self):
            return self._uuid

        def get_display_name(self):
            return self.name if self.name else f"Scene_{self._uuid[:8]}"

    class MockObject:
        def __init__(self, obj_type):
            self._uuid = str(uuid_module.uuid4())
            self.type = obj_type

        @property
        def uuid(self):
            return self._uuid

    class MockRay:
        def __init__(self, p1, p2, brightness=1.0):
            self.p1 = p1
            self.p2 = p2
            self.brightness_s = brightness / 2
            self.brightness_p = brightness / 2
            self.wavelength = None
            self.is_tir_result = False
            self.caused_tir = False

    # Test 1: Create SceneSnapshot
    print("Test 1: SceneSnapshot from Scene")
    scene = MockScene()
    scene.objs = [MockObject('Glass'), MockObject('Glass'), MockObject('PointSource')]
    scene.optical_objs = scene.objs.copy()

    snapshot = SceneSnapshot.from_scene(scene)
    print(f"  Scene UUID: {snapshot.uuid[:8]}...")
    print(f"  Scene name: {snapshot.name}")
    print(f"  Object count: {snapshot.object_count}")
    print(f"  Object summary: {snapshot.object_summary}")
    print(f"  Settings: {snapshot.settings}")

    # Test 2: Create SimulationResult
    print("\nTest 2: SimulationResult.create()")
    rays = [
        MockRay({'x': 0, 'y': 0}, {'x': 100, 'y': 0}, 1.0),
        MockRay({'x': 100, 'y': 0}, {'x': 150, 'y': 50}, 0.8),
        MockRay({'x': 150, 'y': 50}, {'x': 200, 'y': 50}, 0.6),
    ]
    rays[1].is_tir_result = True  # Mark as TIR result

    result = SimulationResult.create(
        scene=scene,
        segments=rays,
        max_rays=10000,
        processed_ray_count=3,
        total_truncation=0.001,
        name="Test Run 1"
    )

    print(f"  {result}")
    print(f"  Display name: {result.get_display_name()}")
    print(f"  Success: {result.success}")
    print(f"  Hit ray limit: {result.hit_ray_limit}")

    # Test 3: XML output
    print("\nTest 3: describe_simulation_result() - XML format")
    xml_output = describe_simulation_result(result, format='xml')
    print(xml_output[:1500] + "...\n")

    # Test 4: Text output
    print("\nTest 4: describe_simulation_result() - Text format")
    text_output = describe_simulation_result(result, format='text')
    print(text_output)

    # Test 5: With segments
    print("\nTest 5: describe_simulation_result() with segments")
    xml_with_segments = describe_simulation_result(result, format='xml', include_segments=True)
    print(xml_with_segments)

    # Test 6: Error case
    print("\nTest 6: Simulation with error")
    scene_error = MockScene()
    scene_error.error = "Ray limit exceeded"
    scene_error.warning = "Some rays were truncated"

    result_error = SimulationResult.create(
        scene=scene_error,
        segments=[],
        max_rays=100,
        processed_ray_count=100,
        name="Error Test"
    )

    print(f"  {result_error}")
    print(f"  Success: {result_error.success}")
    print(f"  Error: {result_error.error}")
    print(f"  Warnings: {result_error.warnings}")

    print("\nSimulationResult test completed successfully!")
