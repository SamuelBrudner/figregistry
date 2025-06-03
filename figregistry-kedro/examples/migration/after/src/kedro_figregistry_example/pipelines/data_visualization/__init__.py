"""Data Visualization Pipeline Package - Migration Example (After FigRegistry-Kedro Integration).

This package demonstrates the 'after' state of migrating an existing Kedro project to use
figregistry-kedro integration. The pipeline package structure remains unchanged from the
original project, showcasing the minimal migration impact while gaining automated figure
styling, condition-based visualization, and zero-touch figure management capabilities.

Migration Transformation Demonstrated:
    - Package structure: UNCHANGED (maintaining existing organization patterns)
    - Import patterns: UNCHANGED (standard Kedro pipeline discovery still works)
    - Node functions: SIMPLIFIED (removed manual plt.savefig() calls)
    - Catalog configuration: ENHANCED (added FigRegistry-specific parameters)
    - Output quality: IMPROVED (consistent styling across all figures)

Pipeline Functions:
    create_pipeline(): Main pipeline with converted figure management
    create_exploratory_pipeline(): Exploratory analysis with automated styling
    create_presentation_pipeline(): Stakeholder visualizations with condition-based styling

Integration Benefits Gained Through Migration:
    - F-005: Automated figure styling via FigureDataSet catalog integration
    - F-002: Condition-based styling replacing manual style specification
    - F-005-RQ-001: Zero-touch figure processing (eliminated plt.savefig() calls)
    - F-005-RQ-002: Integrated versioning with Kedro's catalog system
    - F-005-RQ-004: Context injection for dynamic styling based on pipeline parameters
    - F-004: Multiple output purposes with automatic path management

Usage (Unchanged from Original Project):
    The pipeline functions remain discoverable by Kedro's framework through the same
    import patterns used before migration, demonstrating minimal code changes:
    
    ```python
    from kedro_figregistry_example.pipelines.data_visualization import (
        create_pipeline as create_data_visualization_pipeline
    )
    
    def register_pipelines() -> Dict[str, Pipeline]:
        return {
            "data_visualization": create_data_visualization_pipeline(),
            "__default__": create_data_visualization_pipeline(),
        }
    ```

Requirements Fulfilled:
    - F-008: Valid Python package initialization maintaining Kedro pipeline discovery
    - Section 0.2.1: Unchanged import patterns showing minimal migration impact
    - F-005: Complete Kedro framework integration through existing discovery mechanisms
    - Section 0.1.3: Unchanged package structure patterns demonstrating architecture preservation
    - Standard Python package conventions for consistent namespace management
"""

from .pipeline import (
    create_pipeline,
    create_exploratory_pipeline,
    create_presentation_pipeline
)

__all__ = [
    "create_pipeline", 
    "create_exploratory_pipeline",
    "create_presentation_pipeline"
]

# Package metadata highlighting migration transformation
__package_info__ = {
    "name": "data_visualization",
    "description": "Migration example: Kedro pipeline enhanced with figregistry-kedro integration",
    "migration_status": "after",
    "version": "1.1.0",  # Version bump reflecting figregistry-kedro integration
    "original_version": "1.0.0",  # Pre-migration baseline
    "structure_changes": "none",  # Package structure unchanged
    "integration_gains": [
        "F-005: Automated figure styling via FigureDataSet",
        "F-002: Condition-based styling replacing manual styling code",
        "F-004: Automated output management with versioning",
        "F-005-RQ-001: Eliminated manual plt.savefig() calls",
        "F-005-RQ-002: Integrated Kedro catalog versioning",
        "F-005-RQ-004: Pipeline parameter-driven styling"
    ],
    "migration_impact": {
        "package_imports": "unchanged",
        "pipeline_discovery": "unchanged", 
        "namespace_management": "unchanged",
        "kedro_compatibility": "unchanged (>=0.18.0,<0.20.0)",
        "added_dependencies": ["figregistry>=0.3.0", "figregistry-kedro>=0.1.0"]
    },
    "demonstration_focus": "Minimal migration impact with maximum automation gains"
}