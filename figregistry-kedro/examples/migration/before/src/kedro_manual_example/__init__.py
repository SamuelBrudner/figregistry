"""Traditional Kedro Project Example - Manual Figure Management.

This package demonstrates traditional Kedro project structure and workflows before
figregistry-kedro integration. The project showcases manual matplotlib figure 
management patterns including scattered plt.savefig() calls, hardcoded styling,
manual file path management, and code duplication issues that automated figure
styling solutions address.

Traditional Workflow Characteristics:
- Manual plt.savefig() calls scattered throughout pipeline nodes
- Hardcoded styling parameters duplicated across multiple functions
- Inconsistent file naming and path management approaches
- Manual configuration of colors, markers, and layout parameters
- Repetitive figure styling code without systematic management
- No automated versioning or condition-based styling capabilities

Problem Areas Demonstrated:
This example highlights the maintenance overhead and code quality issues
associated with manual figure management in data science pipelines,
providing clear motivation for automated styling and versioning solutions.

Project Structure:
- nodes.py: Pipeline functions with manual figure management
- pipeline_registry.py: Standard Kedro pipeline registration
- settings.py: Traditional project configuration without lifecycle hooks

Educational Purpose:
This package serves as the baseline state in migration documentation,
demonstrating the problematic patterns that automated figure management
systems like figregistry-kedro are designed to eliminate.
"""

__version__ = "0.1.0"

# Standard package metadata for traditional Kedro project
__author__ = "Example Project Contributors"
__email__ = "example@kedro.org"

# Public API following standard Python package conventions
__all__ = [
    "__version__",
]

# Project identification constants for standard Kedro framework usage
PROJECT_NAME = "kedro-manual-example"
PROJECT_VERSION = __version__
PACKAGE_NAME = "kedro_manual_example"

# Standard Kedro project requirements
KEDRO_VERSION_REQUIREMENT = ">=0.18.0,<0.20.0"
PYTHON_VERSION_REQUIREMENT = ">=3.10"

# Traditional project metadata highlighting manual approach characteristics
PROJECT_CHARACTERISTICS = [
    "Manual figure management with plt.savefig() calls",
    "Hardcoded styling parameters throughout codebase",
    "Inconsistent file naming and organization",
    "Code duplication in figure styling logic",
    "No automated versioning capabilities",
    "Manual configuration of visualization parameters",
]

# Classification metadata for migration example documentation
EXAMPLE_METADATA = {
    "purpose": "migration_baseline_demonstration",
    "workflow_type": "traditional_manual_management", 
    "complexity_level": "basic",
    "target_audience": "figregistry_kedro_evaluation",
    "problem_areas": [
        "scattered_figure_saving_logic",
        "hardcoded_styling_parameters",
        "manual_file_management",
        "code_duplication_patterns",
        "inconsistent_naming_conventions",
        "maintenance_overhead",
    ]
}

# Traditional Kedro project structure demonstration
TRADITIONAL_COMPONENTS = {
    "pipeline_organization": "standard_kedro_patterns",
    "figure_management": "manual_matplotlib_operations",
    "configuration": "scattered_hardcoded_parameters",
    "versioning": "manual_timestamp_approaches",
    "styling": "duplicated_styling_code_blocks",
}