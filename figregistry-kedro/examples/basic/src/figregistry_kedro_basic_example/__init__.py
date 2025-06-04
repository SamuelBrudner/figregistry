"""Basic FigRegistry-Kedro Integration Example Project.

This package demonstrates essential figregistry-kedro plugin capabilities through
a minimal Kedro project showcasing automated figure styling and versioning within
data science pipelines. The example eliminates manual plt.savefig() calls by
integrating FigRegistry's condition-based styling system with Kedro's catalog-based
data management.

Key Features Demonstrated:
- Automated figure styling through FigureDataSet integration (F-005)
- Lifecycle hooks for seamless FigRegistry initialization (F-006)
- Configuration bridge between Kedro and FigRegistry systems (F-007)
- Zero-touch figure management eliminating manual save operations
- Condition-based styling with simple experimental parameter resolution
- Standard Kedro project structure with plugin enhancement

Technical Integration:
This example project follows Kedro conventions while showcasing the figregistry-kedro
plugin's ability to automatically apply styling based on experimental conditions,
manage figure versioning through Kedro's catalog system, and provide seamless
integration between both frameworks without modifying existing pipeline logic.

Project Structure:
- nodes.py: Pipeline node functions that create matplotlib figures
- pipeline_registry.py: Kedro pipeline registration and discovery
- settings.py: FigRegistryHooks registration for lifecycle integration
- pipelines/: Modular pipeline definitions demonstrating automated styling

Usage:
This package serves as both a functional Kedro project and an educational
reference for implementing figregistry-kedro integration in real-world
data science workflows.
"""

__version__ = "0.1.0"

# Package metadata for Kedro project discovery and plugin integration
__author__ = "FigRegistry Contributors"
__email__ = "contributors@figregistry.org"

# Public API for the basic example project
# Following standard Python package conventions for namespace management
__all__ = [
    "__version__",
]

# Project identification constants for Kedro framework integration
PROJECT_NAME = "figregistry-kedro-basic-example"
PROJECT_VERSION = __version__
PACKAGE_NAME = "figregistry_kedro_basic_example"

# Technical metadata supporting F-008 requirements for plugin packaging
KEDRO_VERSION_REQUIREMENT = ">=0.18.0,<0.20.0"
FIGREGISTRY_VERSION_REQUIREMENT = ">=0.3.0"
PYTHON_VERSION_REQUIREMENT = ">=3.10"

# Integration metadata for figregistry-kedro plugin demonstration
INTEGRATION_FEATURES = [
    "F-005: FigureDataSet Integration",
    "F-006: Lifecycle Hooks", 
    "F-007: Configuration Bridge",
    "Automated Figure Styling",
    "Zero-Touch Figure Management",
    "Condition-Based Styling",
]

# Educational classification for example project categorization
EXAMPLE_METADATA = {
    "complexity_level": "basic",
    "target_audience": "new_figregistry_kedro_users",
    "use_case_category": "introductory_demonstration",
    "integration_scope": "essential_plugin_features",
    "demonstrated_workflows": [
        "automated_figure_styling",
        "catalog_based_figure_management", 
        "condition_based_visualization",
        "pipeline_lifecycle_integration",
    ]
}