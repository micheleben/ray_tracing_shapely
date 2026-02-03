# solve the serialization problem in JSON API
## problem statement
in some LLM API we can only pass JSON-serializable data (dicts, lists, strings, numbers) . Objects like a Glass instances or a list of Ray objects have methods and attibutes (like .path,brightness_p ) that don't survive JSON serialization..

we have a list of tools that are intrinsically for direct Python use and not for API-mediated calls with complex objects...
we would like to have tools that could substitute the following ones but with serializable data:

find_rays_inside_glass, 
find_rays_crossing_edge, 
find_rays_by_angle_to_edge, 
find_rays_by_polarization

The first tool can be approached like this: 
```python
def find_grazing_rays_in_named_glass(
    glass_name: str,
    sim_result_name: str = "my_res_grazing"
) -> str:
    """Fully string-based tool that returns XML."""
    sim = globals()[sim_result_name]
    glass = get_object_by_name(scene=sim.scene, name=glass_name)
    rays_inside = find_rays_inside_glass(sim.sim_results.segments, glass)
    grazing = filter_grazing_rays(rays_inside)
    return rays_to_xml(grazing)  # Return string, not objects
```


