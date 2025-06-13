# Sprint 001 Tasks

## Task 1: Project Setup
**Status**: 🔴 Not Started  
**Assignee**: Developer  
**Priority**: High  
**Estimated Time**: 1 hour

### Subtasks:
- [ ] Create package directory structure
- [ ] Write setup.py with project metadata
- [ ] Create requirements.txt with dependencies
- [ ] Set up .gitignore for Python projects
- [ ] Create __init__.py files

### Definition of Done:
- Package can be installed with `pip install -e .`
- All dependencies are listed
- Project follows Python packaging best practices

---

## Task 2: Test Data Generation Utilities
**Status**: 🔴 Not Started  
**Assignee**: Developer  
**Priority**: High  
**Estimated Time**: 3 hours

### Subtasks:
- [ ] Create `tests/data_generator.py`
- [ ] Implement shape generation functions (rectangle, circle, triangle)
- [ ] Create COCO annotation builder class
- [ ] Generate test dataset (5 images, 20+ objects)
- [ ] Save test data to `tests/data/`

### Definition of Done:
- Can generate synthetic images with various shapes
- Produces valid COCO format JSON
- Test data is reproducible (fixed seed)
- Images and annotations are saved to disk

---

## Task 3: Basic COCO Loader
**Status**: 🔴 Not Started  
**Assignee**: Developer  
**Priority**: High  
**Estimated Time**: 4 hours

### Subtasks:
- [ ] Create `fastscp/coco_loader.py`
- [ ] Implement COCOLoader class
- [ ] Add annotation parsing method
- [ ] Implement object extraction (crop from image)
- [ ] Add basic caching dictionary

### Definition of Done:
- Can load COCO annotations from file
- Extracts object crops correctly
- Caches loaded objects in memory
- Handles missing files gracefully

---

## Task 4: SimpleCopyPaste Transform (MVP)
**Status**: 🔴 Not Started  
**Assignee**: Developer  
**Priority**: High  
**Estimated Time**: 4 hours

### Subtasks:
- [ ] Create `fastscp/transforms.py`
- [ ] Implement SimpleCopyPaste class (Albumentations compatible)
- [ ] Add initialization with COCO file and object counts
- [ ] Implement apply() method
- [ ] Add basic overlay blending

### Definition of Done:
- Transform works with Albumentations pipeline
- Can paste objects based on count dictionary
- Produces valid augmented images
- No crashes on edge cases

---

## Task 5: Core Test Suite
**Status**: 🔴 Not Started  
**Assignee**: Developer  
**Priority**: Medium  
**Estimated Time**: 3 hours

### Subtasks:
- [ ] Set up pytest configuration
- [ ] Write tests for data generator
- [ ] Write tests for COCO loader
- [ ] Write integration test for SimpleCopyPaste
- [ ] Add basic performance benchmark

### Definition of Done:
- All tests pass
- Code coverage > 80%
- Performance baseline established
- Tests are reproducible

---

## Notes
- Complete tasks in order (dependencies)
- Commit after each task completion
- Update status as you progress
- Ask for help if blocked