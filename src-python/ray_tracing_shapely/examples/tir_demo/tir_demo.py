import sys
import os
from typing import List

# Add parent directories to path to from ray_tracing_shapely import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs.light_source.point_source import PointSource
from ray_tracing_shapely.core.scene_objs.glass.ideal_lens import IdealLens
from ray_tracing_shapely.core.scene_objs.blocker.blocker import Blocker
from ray_tracing_shapely.core.simulator import Simulator
from ray_tracing_shapely.core.svg_renderer import SVGRenderer
from ray_tracing_shapely.core.scene_objs.glass.glass import Glass
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay

def tir_demo():
    """Total Internal Reflection demonstration.
    
    Problem Statement
    Demonstrate Total Internal Reflection (TIR) in a glass prism when a light ray 
    hits an internal surface at an angle greater than the critical angle.

    Physics Background
    Critical angle: θ_c = arcsin(n₂/n₁) = arcsin(1/1.5) ≈ 41.8°
    For TIR to occur: incident angle (from normal) must be > 41.8°
    When TIR occurs: ray reflects instead of refracting
    
    Expected Behavior
    Ray enters prism through left face (refracts inward due to n=1.5)
    Ray travels to opposite face
    If angle > 41.8° from normal: TIR occurs → ray reflects internally
    Ray exits through another face
    
    My Plan
    Prism: Equilateral triangle (60° angles)
    Ray entry: Aim at left face at ~45° to create steep internal angle
    Check: Ray should reflect off right face instead of exiting"""

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)     
    print("Setting up Total Internal Reflection in a Prism...\n")

    # Create scene
    scene = Scene()
    scene.ray_density = 0.1

    # Create equilateral triangle prism
    # Vertices at (100, 200), (200, 200), (150, 113.4) for 60° angles
    prism = Glass(scene)
    prism.path = [
        {'x': 100, 'y': 200, 'arc': False},
        {'x': 200, 'y': 200, 'arc': False},
        {'x': 150, 'y': 113.4, 'arc': False}
    ]
    prism.not_done = False
    prism.refIndex = 1.5

    ray_list: List[SingleRay] = []

    # Create 25 rays at different angles to demonstrate TIR
    for i in range(10):
        new_ray_source = SingleRay(scene)
        new_ray_source.p1 = {'x': 90, 'y': 140}
        new_ray_source.p2 = {'x': 120, 'y': 200 - i*3}  # Sweep downward
        new_ray_source.brightness = 1.0
        ray_list.append(new_ray_source)

    
#    # Create point source at origin
#     point_source = PointSource(scene)
#     point_source.x = 50
#     point_source.y = 150
#     point_source.brightness = 1.0


    # Add to scene
    scene.add_object(prism)
    for ray_source in ray_list:
        scene.add_object(ray_source)

    # Run simulation
    # verbose levels: 0=silent (default), 1=verbose (ray processing), 2=debug (detailed refraction)
    simulator = Simulator(scene, max_rays=1000, verbose=0)
    segments = simulator.run()

    print(f"Rays traced: {simulator.processed_ray_count}")
    print(f"Segments: {len(segments)}")

    # Render
    renderer = SVGRenderer(width=800, height=600, viewbox=(0, 50, 300, 200))
    renderer.draw_line_segment(prism.path[0], prism.path[1], color='blue', stroke_width=2, label='Prism')
    renderer.draw_line_segment(prism.path[1], prism.path[2], color='blue', stroke_width=2)
    renderer.draw_line_segment(prism.path[2], prism.path[0], color='blue', stroke_width=2)

    for seg in segments:
        renderer.draw_ray_segment(seg, color='red', opacity=0.8, stroke_width=1.5, draw_gap_rays=True)

    output_file = output_dir + '/prism_tir.svg'
    renderer.save(output_file)
    print(f"\nSaved to: {output_file}")

if __name__ == '__main__':
    tir_demo()