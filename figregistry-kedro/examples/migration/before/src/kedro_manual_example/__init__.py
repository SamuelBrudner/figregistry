"""Traditional Kedro Project Example - Manual Figure Management.

This package demonstrates a conventional Kedro machine learning pipeline project
that relies on manual matplotlib figure management approaches. This example serves
as the "before" state in migration documentation, showing the scattered plt.savefig()
calls, hardcoded styling, and manual file management that figregistry-kedro eliminates.

Traditional Manual Patterns Demonstrated:
    - Manual plt.savefig() calls scattered throughout pipeline nodes
    - Hardcoded styling parameters within individual functions
    - Manual file path construction and naming conventions
    - Code duplication for styling across different visualization functions
    - Inconsistent experimental condition handling without systematic management
    - Manual configuration management without automated initialization

Project Structure (Traditional):
    - pipeline_registry.py: Standard pipeline definitions without lifecycle hooks
    - nodes.py: Pipeline node functions with manual figure saving and styling
    - settings.py: Basic Kedro project settings without FigRegistryHooks

Pain Points Highlighted:
    - Maintenance overhead from scattered styling code
    - Configuration drift across different visualizations
    - Manual version management and file organization
    - Inconsistent styling across experimental conditions
    - Code duplication and reduced maintainability

This traditional approach requires developers to manually manage every aspect
of figure creation, styling, and persistence within their pipeline node functions,
leading to the maintenance challenges that automated solutions address.

Usage Example (Traditional Manual Approach):
    # Execute pipeline with manual figure management
    kedro run --pipeline=data_visualization
    
    # Each node function contains manual plt.savefig() calls
    # Styling must be manually configured in each function
    # File paths and naming require manual management

For comparison with the automated approach, see the figregistry-kedro examples.
"""

# Basic package metadata for traditional Kedro project
__version__ = "1.0.0"
__author__ = "Data Science Team"
__description__ = "Traditional Kedro project with manual figure management"

# Import core components following standard Kedro patterns
# No figregistry-kedro integration - purely traditional approach

try:
    # Standard Kedro pipeline registry import
    from .pipeline_registry import create_pipeline
    
    # Traditional node functions with manual figure management
    from .nodes import (
        create_exploratory_plot,
        generate_summary_visualization,
        create_analysis_chart
    )
    
    # Mark successful imports for basic validation
    _IMPORTS_SUCCESS = True
    
except ImportError as e:
    # Basic error handling for missing components
    import warnings
    warnings.warn(
        f"Could not import pipeline components: {e}. "
        "Check that all files are present in the package.",
        ImportWarning
    )
    _IMPORTS_SUCCESS = False


# Standard exports for Kedro project discovery
# Traditional approach - no automation or advanced integration
__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__description__",
    
    # Core pipeline function (required for Kedro)
    "create_pipeline",
    
    # Node functions for manual figure management
    "create_exploratory_plot",
    "generate_summary_visualization", 
    "create_analysis_chart",
    
    # Status indicator
    "_IMPORTS_SUCCESS"
]


def get_project_info() -> dict:
    """Get basic project information for traditional Kedro project.
    
    Returns:
        Dictionary containing standard project metadata and status,
        without advanced integration features or automation capabilities.
        
    Example:
        >>> from kedro_manual_example import get_project_info
        >>> info = get_project_info()
        >>> print(f"Traditional Project: {info['name']} v{info['version']}")
    """
    return {
        "name": "kedro-manual-example",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "imports_success": _IMPORTS_SUCCESS,
        "kedro_compatible": True,
        "automation_features": False,
        "manual_management_required": True,
        "pain_points": [
            "Manual plt.savefig() calls in every node",
            "Hardcoded styling parameters throughout code",
            "Manual file path and naming management",
            "Code duplication across visualization functions",
            "Inconsistent experimental condition handling",
            "No systematic configuration management"
        ]
    }


def check_dependencies() -> bool:
    """Check if basic dependencies are available for traditional workflow.
    
    Validates that standard scientific computing packages are installed
    for manual figure management workflow. No automated styling or
    configuration management dependencies required.
    
    Returns:
        True if basic dependencies are available, False otherwise.
        
    Example:
        >>> from kedro_manual_example import check_dependencies
        >>> if check_dependencies():
        ...     print("Ready for manual figure management workflow")
        ... else:
        ...     print("Missing basic dependencies")
    """
    try:
        # Check standard scientific computing dependencies
        import kedro
        import matplotlib
        import pandas
        import numpy
        
        # Validate project structure is importable
        if not _IMPORTS_SUCCESS:
            return False
            
        return True
        
    except ImportError:
        return False


# Project configuration for traditional manual approach
TRADITIONAL_CONFIG = {
    "approach": "manual",
    "automation_level": "none",
    "required_kedro_version": ">=0.18.0",
    "figure_management": "manual_plt_savefig",
    "styling_approach": "hardcoded_in_functions",
    "configuration_management": "manual",
    "maintenance_requirements": [
        "Update styling code in multiple functions",
        "Manually manage file paths and naming",
        "Coordinate experimental condition handling",
        "Maintain consistent styling across pipeline",
        "Handle version management manually"
    ],
    "migration_benefits": [
        "Eliminate scattered plt.savefig() calls",
        "Centralize styling configuration",
        "Automate file management",
        "Reduce code duplication",
        "Enable systematic condition handling"
    ]
}