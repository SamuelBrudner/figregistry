"""FigRegistry Kedro Basic Example Project.

This package provides a complete demonstration of figregistry-kedro integration
within a Kedro machine learning pipeline project. The example showcases zero-touch
figure management, automated styling based on experimental conditions, and seamless
integration between Kedro's catalog system and FigRegistry's configuration-driven
visualization management.

Key Integration Features Demonstrated:
    - F-005: Automated figure management through FigureDataSet integration
    - F-006: Lifecycle hooks for configuration initialization and context management  
    - F-007: Configuration bridge between Kedro and FigRegistry systems
    - F-002: Condition-based styling through pipeline parameter resolution
    - F-004: Purpose-driven output quality (exploratory/presentation/publication)

Project Structure:
    - pipeline_registry.py: Complete pipeline definitions demonstrating integration
    - nodes.py: Pipeline node functions creating matplotlib figures without manual styling
    - settings.py: Kedro project settings with FigRegistryHooks registration
    - conf/: Configuration files including catalog.yml and figregistry.yml

The project eliminates manual plt.savefig() calls throughout pipeline nodes,
instead relying on FigureDataSet to automatically apply condition-based styling
during Kedro catalog save operations. This demonstrates the core value proposition
of zero-touch figure management within data science workflows.

Usage Example:
    # Execute complete workflow with automated figure styling
    kedro run --pipeline=data_visualization
    
    # Run specific visualization stages
    kedro run --pipeline=exploratory_analysis
    kedro run --pipeline=publication_quality
    
    # Execute with custom experimental conditions
    kedro run --params experiment_condition:treatment_group_a

For detailed setup and execution instructions, see the project README.md.
"""

# Package metadata for the basic example project
__version__ = "0.1.0"
__author__ = "FigRegistry Kedro Integration Team"
__description__ = "Basic example demonstrating figregistry-kedro integration"

# Import key components for easier access from the package
# These imports support proper Kedro project structure and pipeline discovery

try:
    # Import pipeline creation function for Kedro framework discovery
    from .pipeline_registry import create_pipeline, create_pipelines
    
    # Import node functions for potential direct access and testing
    from .nodes import (
        generate_synthetic_data,
        create_exploratory_data_plot,
        create_manuscript_figure_1,
        create_simple_example
    )
    
    # Mark successful imports for debugging and validation
    _IMPORTS_AVAILABLE = True
    
except ImportError as e:
    # Graceful fallback for development environments or incomplete installations
    import warnings
    warnings.warn(
        f"Some components could not be imported: {e}. "
        "This may indicate missing dependencies or incomplete installation.",
        ImportWarning
    )
    _IMPORTS_AVAILABLE = False


# Export key functions for Kedro project discovery and external access
# This supports F-008 requirements for proper plugin packaging and distribution
__all__ = [
    # Package metadata
    "__version__",
    "__author__", 
    "__description__",
    
    # Pipeline registry functions (required for Kedro project discovery)
    "create_pipeline",
    "create_pipelines",
    
    # Selected node functions for testing and validation
    "generate_synthetic_data",
    "create_exploratory_data_plot", 
    "create_manuscript_figure_1",
    "create_simple_example",
    
    # Status indicator
    "_IMPORTS_AVAILABLE"
]


def get_project_info() -> dict:
    """Get project metadata and status information.
    
    Returns:
        Dictionary containing project metadata, version information,
        and import status for validation and debugging purposes.
        
    Example:
        >>> from figregistry_kedro_basic_example import get_project_info
        >>> info = get_project_info()
        >>> print(f"Project: {info['name']} v{info['version']}")
    """
    return {
        "name": "figregistry-kedro-basic-example",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "imports_available": _IMPORTS_AVAILABLE,
        "kedro_compatible": True,
        "figregistry_integration": True,
        "features_demonstrated": [
            "F-005: FigureDataSet Integration",
            "F-006: Lifecycle Hooks", 
            "F-007: Configuration Bridge",
            "F-002: Condition-Based Styling",
            "F-004: Purpose-Driven Output Quality"
        ]
    }


def validate_installation() -> bool:
    """Validate that all required dependencies are properly installed.
    
    Performs a quick check to ensure the basic example can run successfully
    by validating imports and key dependency availability.
    
    Returns:
        True if installation is complete and functional, False otherwise.
        
    Example:
        >>> from figregistry_kedro_basic_example import validate_installation
        >>> if validate_installation():
        ...     print("Ready to run kedro pipelines!")
        ... else:
        ...     print("Installation issues detected")
    """
    try:
        # Check core dependencies
        import kedro
        import figregistry
        import matplotlib
        import pandas
        import numpy
        
        # Check figregistry-kedro plugin availability
        import figregistry_kedro
        from figregistry_kedro.datasets import FigureDataSet
        from figregistry_kedro.hooks import FigRegistryHooks
        
        # Validate project components are importable
        if not _IMPORTS_AVAILABLE:
            return False
            
        return True
        
    except ImportError:
        return False


# Project-level configuration and utilities
PROJECT_CONFIG = {
    "pipeline_default": "data_visualization",
    "supported_kedro_versions": ">=0.18.0,<0.20.0",
    "required_figregistry_version": ">=0.3.0",
    "example_complexity": "basic",
    "educational_focus": [
        "Zero-touch figure management",
        "Automated styling application", 
        "Kedro catalog integration",
        "Configuration bridge usage",
        "Lifecycle hook registration"
    ]
}