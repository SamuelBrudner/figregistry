"""
FigRegistry-Kedro Integration Plugin

This package extends FigRegistry's scientific visualization management capabilities
into the Kedro machine learning pipeline framework through a dedicated plugin that
enables automated figure styling, versioning, and management within Kedro data
science workflows while maintaining FigRegistry's core principle of zero external
dependencies.

The plugin provides three core integration components:
- FigureDataSet: Custom Kedro AbstractDataSet for automated figure styling
- FigRegistryHooks: Lifecycle hooks for configuration initialization
- FigRegistryConfigBridge: Configuration translation between frameworks

Key Features:
- Automated condition-based styling through FigRegistry integration (F-005)
- Non-invasive lifecycle integration preserving Kedro's execution model (F-006)
- Configuration merging with clear precedence rules (F-007)
- Plugin packaging and distribution following modern Python standards (F-008)
- Thread-safe operation for parallel pipeline execution
- <5% performance overhead compared to manual matplotlib operations
- Full compatibility with Kedro versioning and experiment tracking

Usage:
    # Register hooks in Kedro project settings.py
    from figregistry_kedro.hooks import FigRegistryHooks
    HOOKS = (FigRegistryHooks(),)
    
    # Configure datasets in Kedro catalog.yml
    my_figure:
        type: figregistry_kedro.FigureDataSet
        filepath: data/08_reporting/my_figure.png
        purpose: presentation
        condition_param: experiment_condition

Version Compatibility:
- figregistry>=0.3.0: Core visualization functionality
- kedro>=0.18.0,<0.20.0: Pipeline framework integration
- Python>=3.10: Advanced type annotation support
"""

# Package metadata following semantic versioning (Section 3.2.3.1)
__version__ = "0.1.0"
__author__ = "FigRegistry Development Team"
__email__ = "figregistry@example.com"
__description__ = "FigRegistry integration plugin for Kedro machine learning pipelines"
__url__ = "https://github.com/figregistry/figregistry-kedro"

# Import core integration components for public API (Section 0.1.2)
from .datasets import (
    FigureDataSet,
    FigureDatasetError,
    validate_figure_dataset_config,
    get_available_purposes,
    get_performance_summary
)

from .hooks import (
    FigRegistryHooks,
    HookInitializationError,
    HookExecutionError,
    get_global_hook_state,
    clear_global_hook_state
)

from .config import (
    FigRegistryConfigBridge,
    FigRegistryConfigSchema,
    ConfigMergeError,
    ConfigValidationError,
    init_config,
    get_merged_config
)

# Define public API for convenient access to plugin functionality
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__description__",
    
    # Core dataset component (F-005: Kedro FigureDataSet Integration)
    "FigureDataSet",
    "FigureDatasetError",
    "validate_figure_dataset_config",
    "get_available_purposes",
    "get_performance_summary",
    
    # Lifecycle hooks component (F-006: Kedro Lifecycle Hooks)
    "FigRegistryHooks", 
    "HookInitializationError",
    "HookExecutionError",
    "get_global_hook_state",
    "clear_global_hook_state",
    
    # Configuration bridge component (F-007: FigRegistry-Kedro Config Bridge)
    "FigRegistryConfigBridge",
    "FigRegistryConfigSchema",
    "ConfigMergeError", 
    "ConfigValidationError",
    "init_config",
    "get_merged_config"
]

# Plugin discovery metadata for Kedro (F-008: Plugin Packaging and Distribution)
def get_plugin_info():
    """
    Return plugin information for Kedro plugin discovery system.
    
    This function provides metadata required for Kedro's plugin ecosystem,
    enabling automatic discovery and registration of the figregistry-kedro
    plugin when installed in a Python environment alongside Kedro.
    
    Returns:
        Dictionary containing plugin metadata for Kedro registration
    """
    return {
        "name": "figregistry-kedro",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "url": __url__,
        "components": {
            "datasets": ["FigureDataSet"],
            "hooks": ["FigRegistryHooks"],
            "config": ["FigRegistryConfigBridge"]
        },
        "requirements": {
            "figregistry": ">=0.3.0",
            "kedro": ">=0.18.0,<0.20.0",
            "python": ">=3.10"
        },
        "features": [
            "automated_figure_styling",
            "condition_based_visualization", 
            "kedro_lifecycle_integration",
            "configuration_merging",
            "versioned_figure_management"
        ]
    }

# Package initialization logging for development and debugging
import logging
logger = logging.getLogger(__name__)

def _log_initialization_info():
    """Log package initialization information for debugging purposes."""
    try:
        # Check for required dependencies
        dependencies_status = {}
        
        try:
            import figregistry
            dependencies_status["figregistry"] = getattr(figregistry, "__version__", "unknown")
        except ImportError:
            dependencies_status["figregistry"] = "not_available"
        
        try:
            import kedro
            dependencies_status["kedro"] = getattr(kedro, "__version__", "unknown")
        except ImportError:
            dependencies_status["kedro"] = "not_available"
        
        try:
            import matplotlib
            dependencies_status["matplotlib"] = getattr(matplotlib, "__version__", "unknown")
        except ImportError:
            dependencies_status["matplotlib"] = "not_available"
        
        logger.debug(
            f"figregistry-kedro v{__version__} initialized with dependencies: "
            f"figregistry={dependencies_status['figregistry']}, "
            f"kedro={dependencies_status['kedro']}, " 
            f"matplotlib={dependencies_status['matplotlib']}"
        )
        
        # Log any dependency warnings
        if dependencies_status["figregistry"] == "not_available":
            logger.warning("FigRegistry not available - styling features will be disabled")
        
        if dependencies_status["kedro"] == "not_available":
            logger.warning("Kedro not available - plugin integration will be disabled")
            
    except Exception as e:
        logger.debug(f"Package initialization logging failed: {e}")

# Initialize package logging
_log_initialization_info()

# Compatibility validation for version constraints (Section 3.2.3.1)
def validate_version_compatibility():
    """
    Validate that installed dependency versions meet plugin requirements.
    
    This function checks installed versions of figregistry and kedro against
    the plugin's compatibility requirements, providing early warning of
    potential compatibility issues during development and deployment.
    
    Returns:
        Boolean indicating if all version requirements are satisfied
    
    Raises:
        ImportError: When required dependencies are missing
        ValueError: When installed versions don't meet requirements
    """
    import pkg_resources
    
    requirements = {
        "figregistry": ">=0.3.0",
        "kedro": ">=0.18.0,<0.20.0"
    }
    
    for package, version_spec in requirements.items():
        try:
            pkg_resources.require(f"{package}{version_spec}")
            logger.debug(f"Version compatibility validated for {package}{version_spec}")
        except pkg_resources.DistributionNotFound:
            raise ImportError(f"Required package not found: {package}")
        except pkg_resources.VersionConflict as e:
            raise ValueError(f"Version compatibility issue: {e}")
    
    return True

# Performance monitoring initialization for enterprise environments
_performance_metrics = {
    "plugin_load_time": None,
    "import_start_time": None
}

def get_plugin_performance_metrics():
    """
    Get plugin performance metrics for monitoring and optimization.
    
    Returns:
        Dictionary containing plugin-level performance statistics
    """
    return {
        "version": __version__,
        "performance_metrics": _performance_metrics.copy(),
        "component_metrics": {
            "datasets": get_performance_summary(),
            "hooks": get_global_hook_state(),
            "config": {}  # ConfigBridge metrics available through instance
        }
    }

# Record plugin initialization completion
import time
_performance_metrics["import_start_time"] = time.time()

# Export convenience functions for common plugin operations
def create_figure_dataset(**kwargs):
    """
    Convenience function for creating FigureDataSet instances.
    
    Args:
        **kwargs: Parameters passed to FigureDataSet constructor
    
    Returns:
        Configured FigureDataSet instance
    
    Usage:
        dataset = figregistry_kedro.create_figure_dataset(
            filepath="data/08_reporting/analysis.png",
            purpose="publication",
            condition_param="experiment_type"
        )
    """
    return FigureDataSet(**kwargs)

def create_hooks(**kwargs):
    """
    Convenience function for creating FigRegistryHooks instances.
    
    Args:
        **kwargs: Parameters passed to FigRegistryHooks constructor
    
    Returns:
        Configured FigRegistryHooks instance
    
    Usage:
        hooks = figregistry_kedro.create_hooks(
            enable_performance_monitoring=True,
            strict_validation=True
        )
    """
    return FigRegistryHooks(**kwargs)

def create_config_bridge(**kwargs):
    """
    Convenience function for creating FigRegistryConfigBridge instances.
    
    Args:
        **kwargs: Parameters passed to FigRegistryConfigBridge constructor
    
    Returns:
        Configured FigRegistryConfigBridge instance
    
    Usage:
        bridge = figregistry_kedro.create_config_bridge(
            cache_enabled=True,
            validation_strict=True
        )
    """
    return FigRegistryConfigBridge(**kwargs)

# Add convenience functions to public API
__all__.extend([
    "get_plugin_info",
    "validate_version_compatibility", 
    "get_plugin_performance_metrics",
    "create_figure_dataset",
    "create_hooks",
    "create_config_bridge"
])

# Complete plugin initialization
_performance_metrics["plugin_load_time"] = (time.time() - _performance_metrics["import_start_time"]) * 1000

logger.debug(f"figregistry-kedro plugin initialization completed in {_performance_metrics['plugin_load_time']:.2f}ms")