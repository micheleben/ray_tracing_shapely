# Migration Instructions for ray_tracing_shapely

This document contains instructions for an agent to update the Python files after migration from `ray_optics_shapely` to `ray_tracing_shapely`.

## Overview

The package has been renamed from `ray_optics_shapely` to `ray_tracing_shapely`. All internal imports need to be updated to reflect this change.

## Task 1: Update All Import Statements

### Files to Update

Search for all occurrences of `ray_optics_shapely` and replace with `ray_tracing_shapely` in all `.py` files under:
- `src-python/ray_tracing_shapely/`

### Specific Patterns to Find and Replace

```
# Pattern 1: Standard imports
ray_optics_shapely -> ray_tracing_shapely

# This will catch:
from ray_optics_shapely.core.scene import Scene
from ray_optics_shapely.analysis import analyze_scene_geometry
import ray_optics_shapely.core.geometry as geometry
```

### Files That Likely Need Updates

1. **analysis/glass_geometry.py** - Has TYPE_CHECKING imports
2. **analysis/__init__.py** - May have package references in docstrings
3. **core/__init__.py** - May have package references
4. **core/simulator.py** - Has conditional imports for `__main__`
5. **core/scene_objs/base_scene_obj.py** - May have imports
6. **core/scene_objs/other/detector.py** - Has conditional imports for `__main__`
7. **examples/**/*.py** - All example files have imports
8. **developer_tests/**/*.py** - All test files have imports

## Task 2: Update __init__.py Files

### src-python/ray_tracing_shapely/core/__init__.py

Ensure it properly exports the main classes:

```python
"""
Core simulation module for ray_tracing_shapely.

Contains the main simulation engine, scene management, and optical objects.
"""

from .scene import Scene
from .simulator import Simulator
from .ray import Ray

__all__ = ['Scene', 'Simulator', 'Ray']
```

### src-python/ray_tracing_shapely/analysis/__init__.py

The docstring should reference `ray_tracing_shapely` not `ray_optics_shapely`.

## Task 3: Verify Conditional Imports

Several files use `if __name__ == "__main__":` blocks with different import paths. These need to be updated:

### Pattern in simulator.py, detector.py, etc:

```python
# OLD:
if __name__ == "__main__":
    from ray_optics_shapely.core.scene_objs.base_glass import BaseGlass
else:
    from .scene_objs.base_glass import BaseGlass

# NEW:
if __name__ == "__main__":
    from ray_tracing_shapely.core.scene_objs.base_glass import BaseGlass
else:
    from .scene_objs.base_glass import BaseGlass
```

## Task 4: Update sys.path Insertions in Examples

Example files often add the parent directory to sys.path. Check and update these paths:

```python
# Look for patterns like:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
```

These should still work but verify the import statements that follow use `ray_tracing_shapely`.

## Commands to Find All Occurrences

```bash
# Find all occurrences of the old package name
grep -r "ray_optics_shapely" src-python/

# Count occurrences
grep -r "ray_optics_shapely" src-python/ | wc -l
```

## Validation

After making all changes, run:

```bash
# Install in development mode
pip install -e .

# Test basic import
python -c "from ray_tracing_shapely import Scene, Simulator, Ray; print('OK')"

# Test analysis import
python -c "from ray_tracing_shapely.analysis import analyze_scene_geometry; print('OK')"

# Run an example (if dependencies are installed)
cd src-python/ray_tracing_shapely/examples/coupled_prisms
python test_glass_geometry.py
```

## Summary Checklist

- [ ] Replace all `ray_optics_shapely` with `ray_tracing_shapely` in imports
- [ ] Update docstrings that reference the old package name
- [ ] Verify conditional imports in files with `if __name__ == "__main__":`
- [ ] Test that the package can be imported correctly
- [ ] Run example scripts to verify functionality
