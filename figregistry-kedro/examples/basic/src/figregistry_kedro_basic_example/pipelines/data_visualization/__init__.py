"""Data Visualization Pipeline Package - FigRegistry-Kedro Integration Demo.

This package provides the complete data visualization pipeline demonstrating automated
figure styling, condition-based visualization, and zero-touch figure management within
Kedro's catalog-based data processing framework.

Pipeline Functions:
    create_pipeline(): Main pipeline demonstrating complete figregistry-kedro integration
    create_exploratory_pipeline(): Focused exploratory analysis workflow
    create_presentation_pipeline(): Stakeholder-ready visualizations
    create_publication_pipeline(): Academic publication quality figures

Integration Features Demonstrated:
    - F-005: Automated figure styling via FigureDataSet integration
    - F-002: Condition-based styling through pipeline parameter resolution  
    - F-005-RQ-001: Zero-touch figure processing (no manual plt.savefig() calls)
    - F-005-RQ-002: Kedro versioning integration for figure outputs
    - F-005-RQ-004: Context injection for conditional styling
    - F-004: Multiple output purposes (exploratory, presentation, publication)

Usage:
    The pipeline functions are automatically discoverable by Kedro's framework
    and can be registered in pipeline_registry.py:
    
    ```python
    from figregistry_kedro_basic_example.pipelines.data_visualization import (
        create_pipeline as create_data_visualization_pipeline
    )
    
    def register_pipelines() -> Dict[str, Pipeline]:
        return {
            "data_visualization": create_data_visualization_pipeline(),
            "__default__": create_data_visualization_pipeline(),
        }
    ```

Requirements Fulfilled:
    - F-008: Valid Python package initialization for Kedro pipeline discovery
    - Section 0.2.1: Proper import patterns for pipeline registry access
    - F-005: Kedro framework integration requirements
    - Standard Python package conventions for namespace management
"""

from .pipeline import (
    create_pipeline,
    create_exploratory_pipeline, 
    create_presentation_pipeline,
    create_publication_pipeline
)

__all__ = [
    "create_pipeline",
    "create_exploratory_pipeline",
    "create_presentation_pipeline", 
    "create_publication_pipeline"
]

# Package metadata for integration verification
__package_info__ = {
    "name": "data_visualization",
    "description": "FigRegistry-Kedro integration demonstration pipeline",
    "version": "0.1.0",
    "integration_features": [
        "F-005: FigureDataSet integration",
        "F-002: Condition-based styling",
        "F-004: Automated output management",
        "F-005-RQ-001: Zero-touch figure processing",
        "F-005-RQ-002: Kedro versioning integration"
    ],
    "kedro_compatibility": ">=0.18.0,<0.20.0",
    "figregistry_compatibility": ">=0.3.0"
}