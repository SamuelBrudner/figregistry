"""FigRegistry-Kedro Plugin.

A bridge between FigRegistry's scientific visualization management capabilities 
and Kedro's machine learning pipeline framework. This plugin enables automated 
figure styling, versioning, and management within Kedro data science workflows 
while maintaining FigRegistry's core principle of zero external dependencies.

The plugin provides three core integration components:

- **FigureDataSet**: Custom Kedro AbstractDataSet for matplotlib figures with 
  automated FigRegistry styling and versioning
- **FigRegistryHooks**: Lifecycle hooks for non-invasive FigRegistry initialization 
  and context management throughout Kedro pipeline execution
- **FigRegistryConfigBridge**: Configuration translation layer that merges Kedro's 
  ConfigLoader output with FigRegistry's YAML-based configuration system

Key Features:
- Zero-touch figure management in Kedro workflows
- Automatic condition-based styling without manual intervention
- Seamless integration with Kedro's catalog versioning system
- Thread-safe operation for parallel pipeline execution
- <5% performance overhead compared to manual figure operations
- Environment-specific configuration support

Installation:
    ```bash
    pip install figregistry-kedro
    ```

Quick Start:
    1. Add FigRegistry hooks to your Kedro project's settings.py:
    
    ```python
    from figregistry_kedro import FigRegistryHooks
    
    HOOKS = (FigRegistryHooks(),)
    ```
    
    2. Configure FigureDataSet in your catalog.yml:
    
    ```yaml
    experiment_plots:
      type: figregistry_kedro.FigureDataSet
      filepath: data/08_reporting/experiment_results.png
      purpose: presentation
      condition_param: experiment_condition
    ```
    
    3. Return matplotlib figures from your pipeline nodes:
    
    ```python
    def create_visualization(data):
        fig, ax = plt.subplots()
        # ... create your plot ...
        return fig  # FigRegistry styling applied automatically
    ```

Dependencies:
- figregistry>=0.3.0: Core visualization management system
- kedro>=0.18.0,<0.20.0: Machine learning pipeline framework
- matplotlib>=3.9.0: Plotting backend (inherited from figregistry)
- pydantic>=2.9.0: Configuration validation (inherited from figregistry)

Compatibility:
- Python 3.10, 3.11, 3.12, 3.13
- Cross-platform support (macOS, Linux, Windows)
- Thread-safe for parallel Kedro runners

Examples:
    For detailed examples and migration guides, see the `examples/` directory 
    in the source repository or visit the documentation.
"""

import warnings
from typing import Optional

# Package metadata
__version__ = "0.1.0"
__author__ = "FigRegistry Team"
__email__ = "support@figregistry.io"
__description__ = "Kedro plugin for FigRegistry scientific visualization management"
__url__ = "https://github.com/figregistry/figregistry-kedro"

# Compatibility requirements
__requires_python__ = ">=3.10"
__requires_figregistry__ = ">=0.3.0"
__requires_kedro__ = ">=0.18.0,<0.20.0"

# Import core components with graceful fallback for missing dependencies
try:
    from .datasets import FigureDataSet, FigureDataSetError
    _datasets_available = True
except ImportError as e:
    warnings.warn(
        f"FigureDataSet not available: {e}. "
        f"Please ensure figregistry>={__requires_figregistry__} and "
        f"kedro{__requires_kedro__} are installed.",
        ImportWarning
    )
    FigureDataSet = None
    FigureDataSetError = None
    _datasets_available = False

try:
    from .hooks import FigRegistryHooks, HookExecutionError
    _hooks_available = True
except ImportError as e:
    warnings.warn(
        f"FigRegistryHooks not available: {e}. "
        f"Please ensure kedro{__requires_kedro__} is installed.",
        ImportWarning
    )
    FigRegistryHooks = None
    HookExecutionError = None
    _hooks_available = False

try:
    from .config import FigRegistryConfigBridge, ConfigurationMergeError
    _config_available = True
except ImportError as e:
    warnings.warn(
        f"FigRegistryConfigBridge not available: {e}. "
        f"Please ensure figregistry>={__requires_figregistry__} and "
        f"kedro{__requires_kedro__} are installed.",
        ImportWarning
    )
    FigRegistryConfigBridge = None
    ConfigurationMergeError = None
    _config_available = False

# Convenience imports for common configuration functions
try:
    from .config import init_config, get_config_bridge
    _config_functions_available = True
except ImportError:
    init_config = None
    get_config_bridge = None
    _config_functions_available = False

# Convenience function for creating hooks with configuration
try:
    from .hooks import create_hooks
    _hook_factory_available = True
except ImportError:
    create_hooks = None
    _hook_factory_available = False

# Convenience function for creating datasets
try:
    from .datasets import create_figure_dataset, validate_figure_dataset_config
    _dataset_utilities_available = True
except ImportError:
    create_figure_dataset = None
    validate_figure_dataset_config = None
    _dataset_utilities_available = False


def get_plugin_info() -> dict:
    """Get comprehensive plugin information and availability status.
    
    Returns:
        Dictionary containing plugin metadata, version information, 
        dependency status, and component availability.
        
    Example:
        ```python
        from figregistry_kedro import get_plugin_info
        
        info = get_plugin_info()
        print(f"Plugin version: {info['version']}")
        print(f"All components available: {info['fully_functional']}")
        ```
    """
    return {
        "name": "figregistry-kedro",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "url": __url__,
        "requires_python": __requires_python__,
        "requires_figregistry": __requires_figregistry__,
        "requires_kedro": __requires_kedro__,
        "components": {
            "FigureDataSet": _datasets_available,
            "FigRegistryHooks": _hooks_available,
            "FigRegistryConfigBridge": _config_available,
            "config_functions": _config_functions_available,
            "hook_factory": _hook_factory_available,
            "dataset_utilities": _dataset_utilities_available
        },
        "fully_functional": all([
            _datasets_available,
            _hooks_available,
            _config_available,
            _config_functions_available,
            _hook_factory_available,
            _dataset_utilities_available
        ])
    }


def check_dependencies() -> bool:
    """Check if all required dependencies are available.
    
    Returns:
        True if all dependencies are available and plugin is fully functional,
        False otherwise.
        
    Example:
        ```python
        from figregistry_kedro import check_dependencies
        
        if check_dependencies():
            print("All dependencies available - plugin ready to use")
        else:
            print("Missing dependencies - check installation")
        ```
    """
    info = get_plugin_info()
    return info["fully_functional"]


def get_version() -> str:
    """Get the plugin version string.
    
    Returns:
        Semantic version string (e.g., "0.1.0")
        
    Example:
        ```python
        from figregistry_kedro import get_version
        
        print(f"Using figregistry-kedro version {get_version()}")
        ```
    """
    return __version__


# Public API exports
# Only export components that are successfully imported
__all__ = [
    # Package metadata
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "__url__",
    "__requires_python__",
    "__requires_figregistry__",
    "__requires_kedro__",
    
    # Utility functions (always available)
    "get_plugin_info",
    "check_dependencies", 
    "get_version"
]

# Add core components to exports if available
if _datasets_available:
    __all__.extend([
        "FigureDataSet",
        "FigureDataSetError"
    ])

if _hooks_available:
    __all__.extend([
        "FigRegistryHooks", 
        "HookExecutionError"
    ])

if _config_available:
    __all__.extend([
        "FigRegistryConfigBridge",
        "ConfigurationMergeError"
    ])

# Add convenience functions to exports if available
if _config_functions_available:
    __all__.extend([
        "init_config",
        "get_config_bridge"
    ])

if _hook_factory_available:
    __all__.append("create_hooks")

if _dataset_utilities_available:
    __all__.extend([
        "create_figure_dataset",
        "validate_figure_dataset_config"
    ])

# Verify plugin initialization
if not check_dependencies():
    warnings.warn(
        "figregistry-kedro plugin is not fully functional due to missing dependencies. "
        f"Please ensure figregistry>={__requires_figregistry__} and "
        f"kedro{__requires_kedro__} are installed. "
        "Run `pip install figregistry kedro` to install missing dependencies.",
        ImportWarning
    )


# Plugin compatibility validation
def _validate_plugin_compatibility():
    """Validate plugin compatibility with installed versions."""
    try:
        import figregistry
        import kedro
        from packaging import version
        
        # Check FigRegistry version compatibility
        if hasattr(figregistry, '__version__'):
            figregistry_version = figregistry.__version__
            if version.parse(figregistry_version) < version.parse(__requires_figregistry__.replace('>=', '')):
                warnings.warn(
                    f"FigRegistry version {figregistry_version} may not be fully compatible. "
                    f"Recommended: {__requires_figregistry__}",
                    UserWarning
                )
        
        # Check Kedro version compatibility  
        if hasattr(kedro, '__version__'):
            kedro_version = kedro.__version__
            min_kedro = __requires_kedro__.split(',')[0].replace('>=', '')
            max_kedro = __requires_kedro__.split(',')[1].replace('<', '')
            
            if not (version.parse(min_kedro) <= version.parse(kedro_version) < version.parse(max_kedro)):
                warnings.warn(
                    f"Kedro version {kedro_version} may not be fully compatible. "
                    f"Recommended: {__requires_kedro__}",
                    UserWarning
                )
                
    except ImportError:
        # Dependencies not available - warnings already issued above
        pass
    except Exception as e:
        # Ignore version checking errors in production
        pass

# Run compatibility validation on import
_validate_plugin_compatibility()