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

Glass Geometry Analysis Utility

This module provides utilities for analyzing the geometric relationships
between glass objects in a scene, including:
- Finding shared interfaces between adjacent glasses
- Computing boundary properties (area, centroid, perimeter)
- Computing interface properties (length, center, normal vectors)
- Describing individual edges of glass objects (Python-specific feature)

This is useful for:
- Understanding optical system geometry
- Computing interface areas for Fresnel calculations
- Visualizing glass arrangements
- Identifying specific edges by index, label, length, or position
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, TYPE_CHECKING

from shapely.geometry import Polygon, LineString, Point, MultiLineString, GeometryCollection
from shapely.ops import unary_union

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.scene_objs.base_glass import BaseGlass


# =============================================================================
# PYTHON-SPECIFIC FEATURE: Edge Description
# =============================================================================
# Provides detailed information about individual edges of glass objects,
# including geometry type, coordinates, length, and labels.
# =============================================================================

class EdgeType(Enum):
    """
    Type of edge geometry.

    Used to distinguish between different edge types in glass objects,
    supporting future extension to parametric curves.
    """
    LINE = "line"           # Straight line segment
    CIRCULAR = "circular"   # Circular arc (defined by 'arc' flag in path)
    EQUATION = "equation"   # Parametric equation (from ParamCurveObjMixin)


@dataclass
class EdgeDescription:
    """
    Describes a single edge of a glass object.

    This dataclass provides comprehensive information about an edge,
    useful for identifying specific edges (e.g., shortest, longest,
    or by cardinal direction) and for debugging/visualization.

    Attributes:
        index: Edge index (0-based, corresponding to path segment)
        edge_type: Type of edge geometry (line, circular, equation)
        p1: Start point as Shapely Point
        p2: End point as Shapely Point
        midpoint: Midpoint of the edge as Shapely Point
        length: Length of the edge in scene units
        short_label: Short label for the edge (e.g., "N", "E", or "0")
        long_label: Long descriptive name (e.g., "North Edge", or "0")
    """
    index: int
    edge_type: EdgeType
    p1: Point
    p2: Point
    midpoint: Point
    length: float
    short_label: str
    long_label: str

    def __repr__(self) -> str:
        return (f"EdgeDescription(index={self.index}, type={self.edge_type.value}, "
                f"label='{self.short_label}', length={self.length:.4f}, "
                f"midpoint=({self.midpoint.x:.2f}, {self.midpoint.y:.2f}))")


def get_edge_descriptions(glass: 'BaseGlass') -> List[EdgeDescription]:
    """
    Get detailed descriptions of all edges in a glass object.

    Iterates through the glass path and returns structured information
    about each edge, including geometry, coordinates, length, and labels.

    Args:
        glass: A BaseGlass object (Glass, SphericalLens, etc.) with a 'path'
               attribute containing a list of points.

    Returns:
        List of EdgeDescription objects, one for each edge. The list is
        ordered by edge index (0 to n-1 for n edges).

    Example:
        >>> from ray_tracing_shapely.analysis import get_edge_descriptions
        >>> edges = get_edge_descriptions(prism)
        >>> for edge in edges:
        ...     print(f"Edge {edge.index} ({edge.short_label}): {edge.length:.2f} units")
        Edge 0 (S): 100.00 units
        Edge 1 (NE): 93.30 units
        Edge 2 (NW): 93.30 units

        >>> # Find shortest edge
        >>> shortest = min(edges, key=lambda e: e.length)
        >>> print(f"Shortest: {shortest.short_label} at {shortest.length:.2f}")
    """
    if not hasattr(glass, 'path') or not glass.path:
        return []

    descriptions = []
    path = glass.path
    n = len(path)

    for i in range(n):
        # Get start and end points (wrapping around for closed polygon)
        p1_dict = path[i]
        p2_dict = path[(i + 1) % n]

        p1 = Point(p1_dict['x'], p1_dict['y'])
        p2 = Point(p2_dict['x'], p2_dict['y'])

        # Calculate midpoint
        mid_x = (p1_dict['x'] + p2_dict['x']) / 2
        mid_y = (p1_dict['y'] + p2_dict['y']) / 2
        midpoint = Point(mid_x, mid_y)

        # Calculate length
        dx = p2_dict['x'] - p1_dict['x']
        dy = p2_dict['y'] - p1_dict['y']
        length = math.sqrt(dx * dx + dy * dy)

        # Determine edge type
        # Check if the next point (p2) has arc=True, which means this segment
        # is part of a circular arc
        next_point = path[(i + 1) % n]
        if next_point.get('arc', False):
            edge_type = EdgeType.CIRCULAR
        elif hasattr(glass, 'pieces'):
            # Has parametric equation pieces (from ParamCurveObjMixin)
            edge_type = EdgeType.EQUATION
        else:
            edge_type = EdgeType.LINE

        # Get labels from the glass object
        label = glass.get_edge_label(i)
        if label:
            short_label, long_label = label
        else:
            short_label = str(i)
            long_label = str(i)

        descriptions.append(EdgeDescription(
            index=i,
            edge_type=edge_type,
            p1=p1,
            p2=p2,
            midpoint=midpoint,
            length=length,
            short_label=short_label,
            long_label=long_label
        ))

    return descriptions


def describe_edges(glass: 'BaseGlass', show_coordinates: bool = True) -> None:
    """
    Print a formatted table describing all edges of a glass object.

    Displays edge information including index, label, type, length,
    and optionally coordinates (endpoints and midpoint).

    Args:
        glass: A BaseGlass object to describe.
        show_coordinates: If True, include endpoint and midpoint coordinates.
            Defaults to True.

    Example:
        >>> from ray_tracing_shapely.analysis import describe_edges
        >>> describe_edges(prism)

        Edge Descriptions for Glass (3 edges)
        ══════════════════════════════════════════════════════════════════════
        Index │ Label │ Type     │ Length  │ P1           │ P2           │ Midpoint
        ──────┼───────┼──────────┼─────────┼──────────────┼──────────────┼─────────────
            0 │ S     │ line     │  100.00 │ (100.0, 200.0) │ (200.0, 200.0) │ (150.0, 200.0)
            1 │ NE    │ line     │   93.30 │ (200.0, 200.0) │ (150.0, 113.4) │ (175.0, 156.7)
            2 │ NW    │ line     │   93.30 │ (150.0, 113.4) │ (100.0, 200.0) │ (125.0, 156.7)
        ══════════════════════════════════════════════════════════════════════
        Total perimeter: 286.60 units
        Shortest edge: NE (index 1) at 93.30 units
        Longest edge: S (index 0) at 100.00 units
    """
    edges = get_edge_descriptions(glass)

    if not edges:
        print("No edges found (empty path)")
        return

    # Get glass type name
    glass_type = getattr(glass, 'type', glass.__class__.__name__)

    print(f"\nEdge Descriptions for {glass_type} ({len(edges)} edges)")
    print("=" * 78)

    if show_coordinates:
        # Full table with coordinates
        header = "Index | Label | Type     | Length  | P1             | P2             | Midpoint"
        print(header)
        print("------+-------+----------+---------+----------------+----------------+----------------")

        for edge in edges:
            p1_str = f"({edge.p1.x:.1f}, {edge.p1.y:.1f})"
            p2_str = f"({edge.p2.x:.1f}, {edge.p2.y:.1f})"
            mid_str = f"({edge.midpoint.x:.1f}, {edge.midpoint.y:.1f})"
            print(f"{edge.index:5d} | {edge.short_label:<5s} | {edge.edge_type.value:<8s} | {edge.length:7.2f} | {p1_str:<14s} | {p2_str:<14s} | {mid_str}")
    else:
        # Compact table without coordinates
        header = "Index | Label | Long Name            | Type     | Length"
        print(header)
        print("------+-------+----------------------+----------+---------")

        for edge in edges:
            long_name = edge.long_label[:20] if len(edge.long_label) > 20 else edge.long_label
            print(f"{edge.index:5d} | {edge.short_label:<5s} | {long_name:<20s} | {edge.edge_type.value:<8s} | {edge.length:7.2f}")

    print("=" * 78)

    # Summary statistics
    total_perimeter = sum(e.length for e in edges)
    shortest = min(edges, key=lambda e: e.length)
    longest = max(edges, key=lambda e: e.length)

    print(f"Total perimeter: {total_perimeter:.2f} units")
    print(f"Shortest edge: {shortest.short_label} (index {shortest.index}) at {shortest.length:.2f} units")
    print(f"Longest edge: {longest.short_label} (index {longest.index}) at {longest.length:.2f} units")


def glass_to_polygon(glass: 'BaseGlass') -> Polygon:
    """
    Convert a Glass object's path to a Shapely Polygon.

    Args:
        glass: A BaseGlass object with a 'path' attribute containing
               a list of points with 'x' and 'y' keys.

    Returns:
        A Shapely Polygon representing the glass boundary.

    Note:
        This currently ignores 'arc' flags in the path points.
        Arc segments would need special handling for curved glass surfaces.
    """
    coords = [(p['x'], p['y']) for p in glass.path]
    return Polygon(coords)


@dataclass
class GlassInterface:
    """
    Represents a shared edge between two adjacent glass objects.

    Attributes:
        geometry: The shared edge as a Shapely LineString
        glass1: First glass object
        glass2: Second glass object
        n1: Refractive index of glass1
        n2: Refractive index of glass2
    """
    geometry: LineString
    glass1: 'BaseGlass'
    glass2: 'BaseGlass'
    n1: float
    n2: float

    @property
    def center(self) -> Point:
        """
        Get the midpoint of the interface.

        Returns:
            Shapely Point at the center of the interface line.
        """
        return self.geometry.interpolate(0.5, normalized=True)

    @property
    def length(self) -> float:
        """
        Get the length of the interface.

        Returns:
            Length of the interface in scene units.
        """
        return self.geometry.length

    def normal_at(self, position: float = 0.5) -> Tuple[float, float]:
        """
        Compute the unit normal vector at a position along the interface.

        The normal is computed perpendicular to the local tangent direction
        and is oriented to point from glass1 toward glass2.

        Args:
            position: Normalized position along the interface (0.0 = start, 1.0 = end).
                     Default is 0.5 (midpoint).

        Returns:
            Tuple (nx, ny) representing the unit normal vector pointing
            from glass1 toward glass2.
        """
        # Get tangent by sampling two nearby points
        delta = 0.01
        pos1 = max(0.0, position - delta)
        pos2 = min(1.0, position + delta)

        p1 = self.geometry.interpolate(pos1, normalized=True)
        p2 = self.geometry.interpolate(pos2, normalized=True)

        # Tangent vector
        tx = p2.x - p1.x
        ty = p2.y - p1.y
        t_length = math.sqrt(tx * tx + ty * ty)

        if t_length < 1e-10:
            # Degenerate case - return arbitrary normal
            return (1.0, 0.0)

        tx /= t_length
        ty /= t_length

        # Normal is perpendicular to tangent (rotate 90 degrees)
        nx, ny = -ty, tx

        # Ensure normal points from glass1 toward glass2
        # Check by seeing which side glass2's centroid is on
        glass2_centroid = glass_to_polygon(self.glass2).centroid
        interface_center = self.center

        # Vector from interface center to glass2 centroid
        to_glass2_x = glass2_centroid.x - interface_center.x
        to_glass2_y = glass2_centroid.y - interface_center.y

        # If normal points away from glass2, flip it
        dot_product = nx * to_glass2_x + ny * to_glass2_y
        if dot_product < 0:
            nx, ny = -nx, -ny

        return (nx, ny)

    def point_at(self, position: float = 0.5) -> Point:
        """
        Get a point at a normalized position along the interface.

        Args:
            position: Normalized position (0.0 = start, 1.0 = end).

        Returns:
            Shapely Point at the specified position.
        """
        return self.geometry.interpolate(position, normalized=True)

    @property
    def start_point(self) -> Point:
        """Get the start point of the interface."""
        return Point(self.geometry.coords[0])

    @property
    def end_point(self) -> Point:
        """Get the end point of the interface."""
        return Point(self.geometry.coords[-1])

    def __repr__(self) -> str:
        return (f"GlassInterface(n1={self.n1:.3f}, n2={self.n2:.3f}, "
                f"length={self.length:.4f}, center=({self.center.x:.2f}, {self.center.y:.2f}))")


@dataclass
class GlassBoundary:
    """
    Represents the full boundary of a glass object.

    Attributes:
        geometry: The glass shape as a Shapely Polygon
        glass: The glass object
        n: Refractive index
    """
    geometry: Polygon
    glass: 'BaseGlass'
    n: float

    @property
    def centroid(self) -> Point:
        """
        Get the geometric center (centroid) of the glass.

        Returns:
            Shapely Point at the centroid.
        """
        return self.geometry.centroid

    @property
    def area(self) -> float:
        """
        Get the area of the glass.

        Returns:
            Area in scene units squared.
        """
        return self.geometry.area

    @property
    def perimeter(self) -> float:
        """
        Get the total perimeter length of the glass.

        Returns:
            Perimeter length in scene units.
        """
        return self.geometry.length

    @property
    def exterior(self) -> LineString:
        """
        Get the outer boundary ring as a LineString.

        Returns:
            Shapely LineString representing the outer boundary.
        """
        return LineString(self.geometry.exterior.coords)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the glass.

        Returns:
            Tuple (minx, miny, maxx, maxy).
        """
        return self.geometry.bounds

    def __repr__(self) -> str:
        cx, cy = self.centroid.x, self.centroid.y
        return (f"GlassBoundary(n={self.n:.3f}, area={self.area:.4f}, "
                f"perimeter={self.perimeter:.4f}, centroid=({cx:.2f}, {cy:.2f}))")


@dataclass
class SceneGeometryAnalysis:
    """
    Complete geometric analysis of a scene's glass objects.

    Attributes:
        boundaries: List of all glass boundaries
        interfaces: List of all shared interfaces between adjacent glasses
        exterior_edges: List of edges not shared with other glasses (exposed to air)
    """
    boundaries: List[GlassBoundary] = field(default_factory=list)
    interfaces: List[GlassInterface] = field(default_factory=list)
    exterior_edges: List[LineString] = field(default_factory=list)

    def get_interface_between(
        self,
        glass1: 'BaseGlass',
        glass2: 'BaseGlass'
    ) -> Optional[GlassInterface]:
        """
        Find the interface between two specific glass objects.

        Args:
            glass1: First glass object
            glass2: Second glass object

        Returns:
            The GlassInterface between the two glasses, or None if they
            don't share an interface.
        """
        for interface in self.interfaces:
            if {interface.glass1, interface.glass2} == {glass1, glass2}:
                return interface
        return None

    def get_boundary_for(self, glass: 'BaseGlass') -> Optional[GlassBoundary]:
        """
        Find the boundary for a specific glass object.

        Args:
            glass: The glass object

        Returns:
            The GlassBoundary for the glass, or None if not found.
        """
        for boundary in self.boundaries:
            if boundary.glass is glass:
                return boundary
        return None

    def get_interfaces_for(self, glass: 'BaseGlass') -> List[GlassInterface]:
        """
        Find all interfaces involving a specific glass object.

        Args:
            glass: The glass object

        Returns:
            List of GlassInterface objects where the glass is either glass1 or glass2.
        """
        return [
            interface for interface in self.interfaces
            if interface.glass1 is glass or interface.glass2 is glass
        ]

    @property
    def total_interface_length(self) -> float:
        """Get the total length of all interfaces."""
        return sum(i.length for i in self.interfaces)

    @property
    def total_exterior_length(self) -> float:
        """Get the total length of all exterior edges."""
        return sum(e.length for e in self.exterior_edges)

    def __repr__(self) -> str:
        return (f"SceneGeometryAnalysis("
                f"boundaries={len(self.boundaries)}, "
                f"interfaces={len(self.interfaces)}, "
                f"exterior_edges={len(self.exterior_edges)})")


def _find_adjacent_pairs(
    glasses: List['BaseGlass']
) -> List[Tuple['BaseGlass', 'BaseGlass', LineString]]:
    """
    Find all pairs of glasses that share an edge.

    Args:
        glasses: List of BaseGlass objects to analyze

    Returns:
        List of tuples (glass1, glass2, shared_edge) for each adjacent pair.
        The shared_edge is a LineString representing the interface.
    """
    pairs = []

    for i, g1 in enumerate(glasses):
        poly1 = glass_to_polygon(g1)

        for g2 in glasses[i + 1:]:
            poly2 = glass_to_polygon(g2)

            # Check if boundaries intersect
            try:
                intersection = poly1.boundary.intersection(poly2.boundary)
            except Exception:
                # Handle any shapely errors gracefully
                continue

            # Process different intersection types
            if intersection.is_empty:
                continue
            elif isinstance(intersection, LineString):
                if intersection.length > 1e-6:
                    pairs.append((g1, g2, intersection))
            elif isinstance(intersection, MultiLineString):
                # Multiple shared edges - add each one
                for line in intersection.geoms:
                    if isinstance(line, LineString) and line.length > 1e-6:
                        pairs.append((g1, g2, line))
            elif isinstance(intersection, GeometryCollection):
                # Mixed geometry - extract LineStrings
                for geom in intersection.geoms:
                    if isinstance(geom, LineString) and geom.length > 1e-6:
                        pairs.append((g1, g2, geom))
            # Point intersections (touching at a single point) are ignored

    return pairs


def _find_exterior_edges(
    glass: 'BaseGlass',
    interfaces: List[GlassInterface]
) -> List[LineString]:
    """
    Find edges of a glass that are NOT shared with other glasses.

    These are the "free" boundaries exposed to air/vacuum.

    Args:
        glass: The glass object to analyze
        interfaces: List of all interfaces in the scene

    Returns:
        List of LineStrings representing exterior edges.
    """
    poly = glass_to_polygon(glass)
    boundary = LineString(poly.exterior.coords)

    # Collect all interface geometries for this glass
    shared_parts = []
    for interface in interfaces:
        if interface.glass1 is glass or interface.glass2 is glass:
            shared_parts.append(interface.geometry)

    if not shared_parts:
        # No shared interfaces - entire boundary is exterior
        return [boundary]

    # Subtract shared parts from boundary
    try:
        shared_union = unary_union(shared_parts)
        exterior = boundary.difference(shared_union)

        # Convert result to list of LineStrings
        if exterior.is_empty:
            return []
        elif isinstance(exterior, LineString):
            return [exterior] if exterior.length > 1e-6 else []
        elif isinstance(exterior, MultiLineString):
            return [line for line in exterior.geoms
                    if isinstance(line, LineString) and line.length > 1e-6]
        elif isinstance(exterior, GeometryCollection):
            return [geom for geom in exterior.geoms
                    if isinstance(geom, LineString) and geom.length > 1e-6]
    except Exception:
        # On error, return the full boundary
        return [boundary]

    return []


def analyze_scene_geometry(scene: 'Scene') -> SceneGeometryAnalysis:
    """
    Analyze all glass objects in a scene and extract geometric relationships.

    This function finds:
    - All glass boundaries with their geometric properties
    - All shared interfaces between adjacent glasses
    - All exterior edges (boundaries exposed to air)

    Args:
        scene: The Scene object containing optical objects

    Returns:
        SceneGeometryAnalysis object containing:
        - boundaries: List of GlassBoundary for each glass
        - interfaces: List of GlassInterface for each shared edge
        - exterior_edges: List of LineString for edges not shared

    Example:
        >>> analysis = analyze_scene_geometry(scene)
        >>> for interface in analysis.interfaces:
        ...     print(f"Interface n1={interface.n1}, n2={interface.n2}")
        ...     print(f"  Length: {interface.length:.2f}")
        ...     print(f"  Center: {interface.center}")
    """
    # Import here to avoid circular imports
    from ..core.scene_objs.base_glass import BaseGlass

    # Filter to only glass objects
    glasses = [obj for obj in scene.optical_objs if isinstance(obj, BaseGlass)]

    if not glasses:
        return SceneGeometryAnalysis()

    # Create boundaries for all glasses
    boundaries = []
    for glass in glasses:
        try:
            poly = glass_to_polygon(glass)
            boundaries.append(GlassBoundary(
                geometry=poly,
                glass=glass,
                n=getattr(glass, 'refIndex', 1.5)  # Default to 1.5 if not set
            ))
        except Exception:
            # Skip glasses that can't be converted to polygons
            continue

    # Find all adjacent pairs and create interfaces
    interfaces = []
    for g1, g2, shared_edge in _find_adjacent_pairs(glasses):
        interfaces.append(GlassInterface(
            geometry=shared_edge,
            glass1=g1,
            glass2=g2,
            n1=getattr(g1, 'refIndex', 1.5),
            n2=getattr(g2, 'refIndex', 1.5)
        ))

    # Find exterior edges for all glasses
    exterior_edges = []
    for glass in glasses:
        try:
            edges = _find_exterior_edges(glass, interfaces)
            exterior_edges.extend(edges)
        except Exception:
            continue

    return SceneGeometryAnalysis(
        boundaries=boundaries,
        interfaces=interfaces,
        exterior_edges=exterior_edges
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Glass Geometry Analysis...\n")

    # We'll create a mock test since we need the full module structure
    # In practice, you would use this with actual Scene and Glass objects

    # Test 1: Basic Shapely operations
    print("Test 1: Basic Shapely Polygon operations")
    poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

    intersection = poly1.boundary.intersection(poly2.boundary)
    print(f"  Polygon 1 area: {poly1.area}")
    print(f"  Polygon 2 area: {poly2.area}")
    print(f"  Boundary intersection type: {type(intersection).__name__}")
    print(f"  Intersection length: {intersection.length if hasattr(intersection, 'length') else 'N/A'}")

    # Test 2: Normal vector calculation
    print("\nTest 2: Normal vector from LineString")
    line = LineString([(0, 0), (10, 0)])

    # Get tangent
    p1 = line.interpolate(0.4, normalized=True)
    p2 = line.interpolate(0.6, normalized=True)
    tx, ty = p2.x - p1.x, p2.y - p1.y
    length = math.sqrt(tx*tx + ty*ty)
    tx, ty = tx/length, ty/length
    nx, ny = -ty, tx

    print(f"  Line: {list(line.coords)}")
    print(f"  Tangent: ({tx:.4f}, {ty:.4f})")
    print(f"  Normal: ({nx:.4f}, {ny:.4f})")

    # Test 3: Interface center
    print("\nTest 3: LineString interpolation")
    line2 = LineString([(0, 0), (10, 5)])
    center = line2.interpolate(0.5, normalized=True)
    print(f"  Line: {list(line2.coords)}")
    print(f"  Center: ({center.x:.2f}, {center.y:.2f})")
    print(f"  Expected: (5.0, 2.5)")

    # Test 4: Boundary difference
    print("\nTest 4: Boundary difference (exterior edges)")
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    shared = LineString([(10, 0), (10, 10)])

    boundary = LineString(square.exterior.coords)
    exterior = boundary.difference(shared)

    print(f"  Square boundary length: {boundary.length}")
    print(f"  Shared edge length: {shared.length}")
    print(f"  Exterior type: {type(exterior).__name__}")
    if hasattr(exterior, 'length'):
        print(f"  Exterior length: {exterior.length}")

    print("\nGlass Geometry Analysis tests completed!")
    print("\nTo test with actual Scene objects, use:")
    print("  from ray_tracing_shapely.core.analysis import analyze_scene_geometry")
    print("  analysis = analyze_scene_geometry(scene)")

    # =========================================================================
    # Test 5: Edge Description (Python-specific feature)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 5: Edge Description (Python-specific feature)")
    print("=" * 60)

    # Create a mock Glass class for testing
    class MockGlass:
        def __init__(self):
            self.type = 'Glass'
            self.path = [
                {'x': 100, 'y': 200, 'arc': False},
                {'x': 200, 'y': 200, 'arc': False},
                {'x': 150, 'y': 113.4, 'arc': False}
            ]
            self._edge_labels = None

        def _get_edge_count(self):
            return len(self.path) if self.path else 0

        def _initialize_edge_labels(self):
            self._edge_labels = {i: (str(i), str(i)) for i in range(self._get_edge_count())}

        @property
        def edge_labels(self):
            if self._edge_labels is None:
                self._initialize_edge_labels()
            return self._edge_labels

        def get_edge_label(self, edge_index):
            return self.edge_labels.get(edge_index)

        def label_edge(self, edge_index, short_label, long_name):
            self.edge_labels[edge_index] = (short_label, long_name)

    # Test with default numeric labels
    print("\n--- Test 5a: Default numeric labels ---")
    mock_prism = MockGlass()
    edges = get_edge_descriptions(mock_prism)

    print(f"Number of edges: {len(edges)}")
    for edge in edges:
        print(f"  {edge}")

    # Test with custom labels
    print("\n--- Test 5b: Custom labels ---")
    mock_prism.label_edge(0, "S", "South Edge (Base)")
    mock_prism.label_edge(1, "NE", "North East Edge")
    mock_prism.label_edge(2, "NW", "North West Edge")

    edges = get_edge_descriptions(mock_prism)
    for edge in edges:
        print(f"  Edge {edge.index}: {edge.short_label} ({edge.long_label}) - {edge.length:.2f} units")

    # Test describe_edges() function
    print("\n--- Test 5c: describe_edges() with coordinates ---")
    describe_edges(mock_prism, show_coordinates=True)

    print("\n--- Test 5d: describe_edges() without coordinates ---")
    describe_edges(mock_prism, show_coordinates=False)

    # Test finding shortest/longest edges
    print("\n--- Test 5e: Finding shortest/longest edges ---")
    shortest = min(edges, key=lambda e: e.length)
    longest = max(edges, key=lambda e: e.length)
    print(f"Shortest edge: {shortest.short_label} (index {shortest.index}) at {shortest.length:.2f} units")
    print(f"Longest edge: {longest.short_label} (index {longest.index}) at {longest.length:.2f} units")

    # Test with glass that has arc edges
    print("\n--- Test 5f: Glass with arc edge ---")
    mock_lens = MockGlass()
    mock_lens.type = 'SphericalLens'
    mock_lens.path = [
        {'x': 0, 'y': 0, 'arc': False},
        {'x': 50, 'y': 25, 'arc': True},  # This point marks an arc
        {'x': 100, 'y': 0, 'arc': False},
        {'x': 50, 'y': -25, 'arc': True},
    ]
    mock_lens._edge_labels = None

    edges = get_edge_descriptions(mock_lens)
    for edge in edges:
        print(f"  Edge {edge.index}: type={edge.edge_type.value}, length={edge.length:.2f}")

    # Test empty glass
    print("\n--- Test 5g: Empty glass ---")
    empty_glass = MockGlass()
    empty_glass.path = []
    edges = get_edge_descriptions(empty_glass)
    print(f"Edges for empty glass: {len(edges)}")

    print("\n" + "=" * 60)
    print("Edge Description tests completed!")
    print("=" * 60)
