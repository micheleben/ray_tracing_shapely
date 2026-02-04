"""
Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
PYTHON-SPECIFIC MODULE: Tool Discovery Registry
===============================================================================
Static registry of all public analysis tools.  Provides a single entry point
for agents and users to discover available functions, classes, and their
signatures without reading source files.

The registry is maintained as a hardcoded list alongside the code.
===============================================================================
"""

from __future__ import annotations
from typing import Dict, List, Any, Union


# =============================================================================
# Registry data
# =============================================================================

_REGISTRY: List[Dict[str, str]] = [
    # -------------------------------------------------------------------------
    # analysis.glass_geometry -- Glass boundary and edge analysis
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.glass_geometry',
        'name': 'EdgeType',
        'kind': 'Enum',
        'signature': 'LINE, CIRCULAR, EQUATION',
        'description': 'Type of edge geometry',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'EdgeDescription',
        'kind': 'dataclass',
        'signature': 'index, edge_type, p1, p2, midpoint, length, short_label, long_label',
        'description': 'Describes a single edge of a glass object',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'GlassInterface',
        'kind': 'dataclass',
        'signature': 'geometry, glass1, glass2, n1, n2',
        'description': 'Shared edge between two adjacent glass objects',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'GlassBoundary',
        'kind': 'dataclass',
        'signature': 'geometry, glass, n',
        'description': 'Full boundary polygon of a glass object',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'SceneGeometryAnalysis',
        'kind': 'dataclass',
        'signature': 'boundaries, interfaces, exterior_edges',
        'description': 'Complete geometric analysis of a scene\'s glass objects',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'get_edge_descriptions',
        'kind': 'function',
        'signature': '(glass) -> List[EdgeDescription]',
        'description': 'Get detailed descriptions of all edges in a glass object',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'describe_edges',
        'kind': 'function',
        'signature': "(glass, format='text', show_coordinates=True) -> str",
        'description': 'Generate a formatted description of all edges of a glass object',
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'glass_to_polygon',
        'kind': 'function',
        'signature': '(glass) -> Polygon',
        'description': "Convert a Glass object's path to a Shapely Polygon",
    },
    {
        'module': 'analysis.glass_geometry',
        'name': 'analyze_scene_geometry',
        'kind': 'function',
        'signature': '(scene) -> SceneGeometryAnalysis',
        'description': 'Analyze all glass objects in a scene and extract geometric relationships',
    },
    # -------------------------------------------------------------------------
    # analysis.saving -- Ray data export, filtering, and statistics
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.saving',
        'name': 'save_rays_csv',
        'kind': 'function',
        'signature': "(ray_segments, output_path, filename='rays.csv', ...) -> Path",
        'description': 'Export ray segment data to a CSV file',
    },
    {
        'module': 'analysis.saving',
        'name': 'rays_to_xml',
        'kind': 'function',
        'signature': '(ray_segments, precision_coords=4, precision_brightness=6) -> str',
        'description': 'Export ray segment data to an XML string',
    },
    {
        'module': 'analysis.saving',
        'name': 'filter_tir_rays',
        'kind': 'function',
        'signature': '(ray_segments, tir_only=True) -> List[Ray]',
        'description': 'Filter rays based on TIR status',
    },
    {
        'module': 'analysis.saving',
        'name': 'filter_grazing_rays',
        'kind': 'function',
        'signature': '(ray_segments, grazing_only=True, criterion=None) -> List[Ray]',
        'description': 'Filter rays based on grazing incidence status',
    },
    {
        'module': 'analysis.saving',
        'name': 'get_ray_statistics',
        'kind': 'function',
        'signature': '(ray_segments) -> dict',
        'description': 'Compute statistics about a collection of ray segments',
    },
    # -------------------------------------------------------------------------
    # analysis.simulation_result -- Simulation result container
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.simulation_result',
        'name': 'SceneSnapshot',
        'kind': 'dataclass',
        'signature': 'uuid, name, object_count, ...',
        'description': 'Captures the state of a scene at a point in time',
    },
    {
        'module': 'analysis.simulation_result',
        'name': 'SimulationResult',
        'kind': 'dataclass',
        'signature': 'uuid, name, timestamp, segments, lineage, ...',
        'description': 'Container for simulation results with full context',
    },
    {
        'module': 'analysis.simulation_result',
        'name': 'describe_simulation_result',
        'kind': 'function',
        'signature': "(result, format='xml', include_segments=False, max_segments=100) -> str",
        'description': 'Generate a formatted description of a simulation result',
    },
    # -------------------------------------------------------------------------
    # analysis.ray_geometry_queries -- Ray-geometry spatial queries
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'get_object_by_name',
        'kind': 'function',
        'signature': '(scene, name) -> BaseSceneObj',
        'description': 'Find a scene object by its user-defined name',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'get_object_by_uuid',
        'kind': 'function',
        'signature': '(scene, uuid) -> BaseSceneObj',
        'description': 'Find a scene object by UUID (exact or prefix match)',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'get_objects_by_type',
        'kind': 'function',
        'signature': '(scene, type_name) -> List[BaseSceneObj]',
        'description': 'Find all scene objects of a given type',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'find_rays_inside_glass',
        'kind': 'function',
        'signature': '(segments, glass_obj) -> List[Ray]',
        'description': 'Return rays whose midpoint is inside a glass object',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'find_rays_crossing_edge',
        'kind': 'function',
        'signature': '(segments, glass_obj, edge_label) -> List[Ray]',
        'description': 'Return rays that intersect a specific glass edge',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'find_rays_by_angle_to_edge',
        'kind': 'function',
        'signature': '(segments, glass_obj, edge_label, min_angle=0, max_angle=90, proximity=None) -> List[Ray]',
        'description': 'Return rays within an angle range relative to an edge',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'find_rays_by_polarization',
        'kind': 'function',
        'signature': '(segments, min_dop=0, max_dop=1) -> List[Ray]',
        'description': 'Filter rays by degree of polarization',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'interpolate_along_edge',
        'kind': 'function',
        'signature': "(glass_obj, edge_label, fraction=0.5) -> Tuple[float, float]",
        'description': 'Get (x, y) coordinates at a fractional position along a glass edge',
    },
    {
        'module': 'analysis.ray_geometry_queries',
        'name': 'describe_all_glass_edges',
        'kind': 'function',
        'signature': "(scene, format='text') -> str",
        'description': 'Describe all edges of all glass objects in a scene',
    },
    # -------------------------------------------------------------------------
    # analysis.lineage_analysis -- Post-hoc ray tree analysis
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.lineage_analysis',
        'name': 'rank_paths_by_energy',
        'kind': 'function',
        'signature': '(lineage, leaf_uuids=None) -> List[Dict]',
        'description': 'Rank optical paths by terminal segment brightness',
    },
    {
        'module': 'analysis.lineage_analysis',
        'name': 'get_branching_statistics',
        'kind': 'function',
        'signature': '(lineage) -> Dict',
        'description': 'Analyze ray tree branching patterns',
    },
    {
        'module': 'analysis.lineage_analysis',
        'name': 'detect_tir_traps',
        'kind': 'function',
        'signature': '(lineage, min_tir_count=2) -> List[Dict]',
        'description': 'Find ray subtrees trapped by repeated TIR',
    },
    {
        'module': 'analysis.lineage_analysis',
        'name': 'extract_angular_distribution',
        'kind': 'function',
        'signature': '(lineage, leaf_uuids=None) -> List[Dict]',
        'description': 'Compute emission angle for each leaf ray',
    },
    {
        'module': 'analysis.lineage_analysis',
        'name': 'build_angular_histogram',
        'kind': 'function',
        'signature': '(angular_data, n_bins=36, weight_by_energy=True) -> Dict',
        'description': 'Build histogram of emission angles',
    },
    {
        'module': 'analysis.lineage_analysis',
        'name': 'check_energy_conservation',
        'kind': 'function',
        'signature': '(lineage) -> Dict',
        'description': 'Verify energy conservation at each branching point',
    },
    # -------------------------------------------------------------------------
    # analysis.fresnel_utils -- Fresnel equation utilities
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.fresnel_utils',
        'name': 'fresnel_transmittances',
        'kind': 'function',
        'signature': '(n1, n2, theta_i_deg) -> Dict[str, float]',
        'description': 'Compute Fresnel power transmittances and reflectances at an interface',
    },
    {
        'module': 'analysis.fresnel_utils',
        'name': 'critical_angle',
        'kind': 'function',
        'signature': '(n1, n2) -> float',
        'description': 'Compute the critical angle for total internal reflection',
    },
    {
        'module': 'analysis.fresnel_utils',
        'name': 'brewster_angle',
        'kind': 'function',
        'signature': '(n1, n2) -> float',
        'description': "Compute Brewster's angle (where R_p = 0)",
    },
    # -------------------------------------------------------------------------
    # analysis.agentic_tools -- JSON-serializable wrappers for LLM tool-use
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.agentic_tools',
        'name': 'set_context',
        'kind': 'function',
        'signature': '(scene, segments, lineage=None) -> None',
        'description': 'Register simulation objects for agentic tool wrappers',
    },
    {
        'module': 'analysis.agentic_tools',
        'name': 'set_context_from_result',
        'kind': 'function',
        'signature': '(scene, result) -> None',
        'description': 'Populate agentic tool context from a SimulationResult',
    },
    {
        'module': 'analysis.agentic_tools',
        'name': 'find_rays_inside_glass_xml',
        'kind': 'function',
        'signature': '(glass_name) -> str',
        'description': 'Find rays inside a named glass, return XML string',
    },
    {
        'module': 'analysis.agentic_tools',
        'name': 'find_rays_crossing_edge_xml',
        'kind': 'function',
        'signature': '(glass_name, edge_label) -> str',
        'description': 'Find rays crossing a named glass edge, return XML string',
    },
    {
        'module': 'analysis.agentic_tools',
        'name': 'find_rays_by_angle_to_edge_xml',
        'kind': 'function',
        'signature': '(glass_name, edge_label, min_angle=0, max_angle=90) -> str',
        'description': 'Find rays by angle to a named glass edge, return XML string',
    },
    {
        'module': 'analysis.agentic_tools',
        'name': 'find_rays_by_polarization_xml',
        'kind': 'function',
        'signature': '(min_dop=0, max_dop=1) -> str',
        'description': 'Filter rays by degree of polarization, return XML string',
    },
    # -------------------------------------------------------------------------
    # analysis.tool_registry -- Tool discovery
    # -------------------------------------------------------------------------
    {
        'module': 'analysis.tool_registry',
        'name': 'list_available_tools',
        'kind': 'function',
        'signature': "(format='text') -> str | Dict",
        'description': 'List all public analysis tools available in the analysis module',
    },
    {
        'module': 'analysis.tool_registry',
        'name': 'get_agentic_tools',
        'kind': 'function',
        'signature': '() -> List[Dict[str, Any]]',
        'description': 'Return agentic tool wrappers with metadata for LLM frameworks',
    },
]


# =============================================================================
# Public API
# =============================================================================

def list_available_tools(format: str = 'text') -> Union[str, Dict[str, List[Dict[str, str]]]]:
    """
    List all public analysis tools available in the analysis module.

    Args:
        format: 'text' for a human-readable table, 'dict' for structured
            data organized by module.

    Returns:
        Formatted string (if format='text') or dict mapping module names
        to lists of tool entries (if format='dict').
    """
    if format == 'dict':
        result: Dict[str, List[Dict[str, str]]] = {}
        for entry in _REGISTRY:
            module = entry['module']
            if module not in result:
                result[module] = []
            result[module].append({
                'name': entry['name'],
                'kind': entry['kind'],
                'signature': entry['signature'],
                'description': entry['description'],
            })
        return result

    # Text format: grouped by module
    lines = []
    current_module = None
    for entry in _REGISTRY:
        if entry['module'] != current_module:
            current_module = entry['module']
            if lines:
                lines.append('')
            lines.append(f'--- {current_module} ---')

        kind_tag = f"[{entry['kind']}]"
        lines.append(f"  {entry['name']:40s} {kind_tag:12s} {entry['description']}")

    return '\n'.join(lines)


def get_agentic_tools() -> List[Dict[str, Any]]:
    """
    Return the string-based agentic tool wrappers with metadata.

    Each entry contains:
    - 'name': function name (str)
    - 'function': the callable
    - 'description': one-line description (str)

    These are framework-agnostic.  To use with a specific framework::

        # claudette
        from claudette import Chat
        tools = [t['function'] for t in get_agentic_tools()]
        chat = Chat(model, tools=tools)

        # langchain
        from langchain.tools import StructuredTool
        tools = [StructuredTool.from_function(t['function']) for t in get_agentic_tools()]

    Returns:
        List of dicts with tool metadata and callables.
    """
    from .agentic_tools import (
        find_rays_inside_glass_xml,
        find_rays_crossing_edge_xml,
        find_rays_by_angle_to_edge_xml,
        find_rays_by_polarization_xml,
    )

    return [
        {
            'name': 'find_rays_inside_glass_xml',
            'function': find_rays_inside_glass_xml,
            'description': 'Find rays inside a named glass, return XML string',
        },
        {
            'name': 'find_rays_crossing_edge_xml',
            'function': find_rays_crossing_edge_xml,
            'description': 'Find rays crossing a named glass edge, return XML string',
        },
        {
            'name': 'find_rays_by_angle_to_edge_xml',
            'function': find_rays_by_angle_to_edge_xml,
            'description': 'Find rays by angle to a named glass edge, return XML string',
        },
        {
            'name': 'find_rays_by_polarization_xml',
            'function': find_rays_by_polarization_xml,
            'description': 'Filter rays by degree of polarization, return XML string',
        },
    ]

def generate_tool_note_for_solveit_notebook(tool_list:List[Dict[str, Any]]= None):
    """
    Generate a markdown text which is supposed to be copied in a note in a solve.it notebook.
    This is a little helper function for solve.it.com env, which is a jupyter notebook style 
    environment for agent assisted exploratory programming.
       
    Returns:
        A string with the formatting that is used by solve.it.com. 
    """
    if tool_list is None:
        tool_list = get_agentic_tools()
    names = [t['name'] for t in tool_list]
    return f"### optical tools:\ntools from ray_tracing_shapely that you can use: &`{', '.join(names)}`"
