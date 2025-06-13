# FastSCP Project Manifest

## Project Information
- **Name**: FastSCP
- **Type**: Python Library
- **Domain**: Computer Vision / Data Augmentation
- **Status**: Initial Development
- **Created**: 2025-01-13

## Project Structure

### Documentation
- `/01_PROJECT_DOCS/`
  - `ARCHITECTURE.md` - Technical design and structure
  - `PROJECT_OVERVIEW.md` - Goals and vision

### Requirements
- `/02_REQUIREMENTS/`
  - `/milestone_001_mvp_implementation/` - Current milestone
    - `MILESTONE_OVERVIEW.md` - Milestone scope and goals
    - `requirements.md` - Detailed requirements (R001-R006)

### Sprints
- `/03_ACTIVE_SPRINT/` - (Empty - ready for first sprint)
- `/04_COMPLETED_SPRINTS/` - (Empty - no completed sprints yet)

## Current State

### Active Milestone
**Milestone 001: MVP Implementation**
- Building core SimpleCopyPaste transformation
- Implementing COCO object loader with caching
- Creating test data generation utilities
- Establishing performance baselines

### Key Decisions Made
1. **Architecture**: Modular design with separate transform, loader, and cache components
2. **Performance**: Focus on NumPy operations and intelligent caching
3. **Testing**: Synthetic data generation for reproducible tests
4. **API**: Simple dictionary-based configuration

### Next Actions
1. Create first sprint from milestone requirements
2. Set up Python project structure
3. Implement core components
4. Build test suite

## Quick Commands
- View current milestone: `cat .simone/02_REQUIREMENTS/milestone_001_mvp_implementation/MILESTONE_OVERVIEW.md`
- Check requirements: `cat .simone/02_REQUIREMENTS/milestone_001_mvp_implementation/requirements.md`
- Start new sprint: Use Simone sprint creation command

## Development Workflow
1. Pick requirements from current milestone
2. Create sprint with selected tasks
3. Implement and test
4. Move completed work to completed sprints
5. Update manifest with progress
