"""
Copyright 2024 The Ray Optics Simulation authors and contributors

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

This is useful for:
- Understanding optical system geometry
- Computing interface areas for Fresnel calculations
- Visualizing glass arrangements
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING

from shapely.geometry import Polygon, LineString, Point, MultiLineString, GeometryCollection
from shapely.ops import unary_union

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..core.scene_objs.base_glass import BaseGlass


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
    print("  from ray_optics_shapely.core.analysis import analyze_scene_geometry")
    print("  analysis = analyze_scene_geometry(scene)")
