# FigRegistryHooks API Reference

Complete API reference for the `FigRegistryHooks` class providing non-invasive lifecycle integration between FigRegistry and Kedro frameworks. This documentation covers hook specifications, registration methods, initialization sequences, and lifecycle management for automated FigRegistry configuration setup.

## Overview

The `FigRegistryHooks` class implements Kedro's hook specifications to provide seamless integration of FigRegistry's configuration-driven visualization system within Kedro's pipeline execution framework. The hooks ensure proper initialization of FigRegistry configuration context, maintain state throughout pipeline execution, and manage cleanup operations while preserving both systems' architectural principles.

**Key Features:**
- Non-invasive integration preserving Kedro's execution model
- Thread-safe operation for parallel pipeline execution
- <5ms hook execution overhead for minimal performance impact
- Automatic configuration bridge initialization
- Support for environment-specific configurations
- Comprehensive error handling and graceful degradation

## Class Definition

```python
class FigRegistryHooks:
    """Kedro lifecycle hooks for FigRegistry integration.
    
    This class provides lifecycle management for FigRegistry initialization within 
    Kedro's execution framework through non-invasive hook registration. The hooks 
    ensure proper FigRegistry context initialization, configuration management, and 
    dataset registration throughout Kedro pipeline execution while maintaining 
    separation of concerns between the frameworks.
    """
```

## Constructor

### `__init__(self, auto_initialize=True, enable_performance_monitoring=False, fallback_on_error=True, max_initialization_time=0.005)`

Initialize FigRegistry hooks with configurable behavior options.

**Parameters:**
- `auto_initialize` (bool, default=True): Automatically initialize FigRegistry configuration during hooks
- `enable_performance_monitoring` (bool, default=False): Enable detailed performance timing logs
- `fallback_on_error` (bool, default=True): Continue execution if FigRegistry initialization fails
- `max_initialization_time` (float, default=0.005): Maximum allowed initialization time in seconds (5ms)

**Returns:**
- FigRegistryHooks instance configured with specified behavior

**Example:**
```python
from figregistry_kedro.hooks import FigRegistryHooks

# Basic initialization with defaults
hooks = FigRegistryHooks()

# Custom configuration
hooks = FigRegistryHooks(
    auto_initialize=True,
    enable_performance_monitoring=True,
    fallback_on_error=False,
    max_initialization_time=0.003  # 3ms timeout
)
```

## Hook Methods

### `after_config_loaded(self, context, config_loader, conf_source=None)`

**Hook Specification:** Kedro `after_config_loaded` hook  
**Execution Phase:** After Kedro has loaded its configuration system but before pipeline execution  
**Performance Target:** <5ms execution time per F-006-RQ-001 requirements

Initialize FigRegistry configuration after Kedro config is loaded. This hook executes the FigRegistryConfigBridge to merge Kedro and FigRegistry configurations, establishing the configuration state required for automated figure styling.

**Parameters:**
- `context` (KedroContext, optional): Kedro context instance containing project configuration
- `config_loader` (ConfigLoader, optional): Kedro ConfigLoader instance for accessing configurations
- `conf_source` (str, optional): Configuration source path

**Raises:**
- `HookExecutionError`: If configuration initialization fails and `fallback_on_error=False`
- `ConfigurationMergeError`: If configuration merging between Kedro and FigRegistry fails

**Implementation Details:**
- Creates `FigRegistryConfigBridge` instance with Kedro ConfigLoader integration
- Determines environment from Kedro context (`context.env` or `config_loader.env`)
- Initializes FigRegistry with merged configuration using `figregistry.init_config()`
- Stores configuration bridge instance for access by downstream components
- Marks FigRegistry as initialized for the current environment

**Thread Safety:** Thread-safe through configuration bridge state management and atomic initialization

**Example Usage:**
```python
# Automatic execution during Kedro pipeline startup
# No manual invocation required - registered via settings.py

# Hook registration in settings.py enables automatic execution:
# HOOKS = (FigRegistryHooks(),)
```

### `before_pipeline_run(self, run_params, pipeline, catalog)`

**Hook Specification:** Kedro `before_pipeline_run` hook  
**Execution Phase:** Before each pipeline run execution  
**Performance Target:** <5ms execution time per Section 5.2.7 requirements

Initialize pipeline-specific FigRegistry context before pipeline execution. This hook ensures FigRegistry context is properly configured for the specific pipeline and manages pipeline registration for concurrent execution tracking.

**Parameters:**
- `run_params` (Dict[str, Any]): Pipeline run parameters including `session_id` and `pipeline_name`
- `pipeline` (Pipeline, optional): Kedro pipeline instance being executed
- `catalog` (DataCatalog, optional): Data catalog instance for the pipeline

**Implementation Details:**
- Extracts pipeline name from `run_params.get('pipeline_name', 'default')`
- Registers pipeline as active for concurrent execution tracking
- Validates that configuration bridge is available from `after_config_loaded`
- Performs late initialization if required and `auto_initialize=True`
- Logs pipeline statistics for monitoring concurrent execution

**Concurrent Execution Support:**
- Thread-safe pipeline registration and state management
- Tracking of active pipeline count for resource management
- Support for Kedro's parallel runners without synchronization overhead

**Example Scenario:**
```python
# During 'kedro run --pipeline=data_processing'
run_params = {
    'pipeline_name': 'data_processing',
    'session_id': 'session_2024-01-15T10:30:45.123Z'
}
# Hook automatically registers pipeline and ensures FigRegistry context availability
```

### `after_pipeline_run(self, run_params, pipeline, catalog)`

**Hook Specification:** Kedro `after_pipeline_run` hook  
**Execution Phase:** After each pipeline run completion  
**Purpose:** Resource cleanup and state management per F-006-RQ-003 requirements

Clean up pipeline-specific context after pipeline execution. This hook performs cleanup and maintains proper state management for concurrent execution scenarios.

**Parameters:**
- `run_params` (Dict[str, Any]): Pipeline run parameters including `session_id` and `pipeline_name`
- `pipeline` (Pipeline, optional): Kedro pipeline instance that was executed
- `catalog` (DataCatalog, optional): Data catalog instance for the pipeline

**Implementation Details:**
- Unregisters completed pipeline from active tracking
- Performs cache cleanup when no more active pipelines remain
- Maintains concurrent execution statistics
- Optional memory optimization through cache clearing

**Resource Management:**
- Automatic cleanup when `active_pipeline_count` reaches zero
- Thread-safe pipeline unregistration
- Memory management through configuration bridge cache control

### `on_node_error(self, error, node_name, catalog, inputs, is_async=False)`

**Hook Specification:** Kedro `on_node_error` hook  
**Execution Phase:** When node execution errors occur  
**Purpose:** Error handling for FigRegistry-related node failures

Handle node execution errors that may affect FigRegistry state, ensuring proper cleanup and error reporting.

**Parameters:**
- `error` (Exception): Exception that occurred during node execution
- `node_name` (str): Name of the node that failed
- `catalog` (DataCatalog, optional): Data catalog instance
- `inputs` (Dict[str, Any]): Node input data
- `is_async` (bool, default=False): Whether the node was executed asynchronously

**Error Handling Scope:**
- `ConfigurationMergeError`: FigRegistry configuration merge failures
- `HookExecutionError`: Hook-specific execution errors
- General exceptions: Logged for awareness without interference

## State Management

### `get_state(self) -> Dict[str, Any]`

Get current hook state for debugging and monitoring purposes.

**Returns:**
- Dict containing current hook state information:
  - `initialized` (bool): Whether FigRegistry has been initialized
  - `active_pipelines` (int): Number of currently active pipelines
  - `bridge_available` (bool): Whether configuration bridge is available
  - `auto_initialize` (bool): Auto-initialization setting
  - `performance_monitoring` (bool): Performance monitoring status
  - `fallback_on_error` (bool): Error fallback behavior

**Example:**
```python
hooks = FigRegistryHooks()
state = hooks.get_state()
print(f"FigRegistry initialized: {state['initialized']}")
print(f"Active pipelines: {state['active_pipelines']}")
```

### `reset_state(self)`

Reset hook state for testing or cleanup purposes.

**Warning:** This method should only be used in testing scenarios or for explicit cleanup. It clears all hook state including configuration bridge and pipeline tracking.

## Configuration Context Management

### FigRegistryConfigBridge Integration

The hooks integrate with `FigRegistryConfigBridge` to provide unified configuration management between Kedro and FigRegistry systems per F-006-RQ-002 requirements.

**Configuration Precedence:**
1. Kedro environment-specific configurations (`conf/local/figregistry.yml`)
2. Kedro base configurations (`conf/base/figregistry.yml`)
3. Standalone FigRegistry configurations (`figregistry.yaml`)

**Bridge Access:**
```python
# Configuration bridge available to downstream components
from figregistry_kedro.config import get_bridge_instance

bridge = get_bridge_instance()
if bridge:
    merged_config = bridge.get_merged_config()
```

## Registration and Activation

### Settings.py Registration

Register FigRegistryHooks in your Kedro project's `settings.py` file per F-006-RQ-004 selective registration requirements.

**Basic Registration:**
```python
# src/your_project/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
```

**Custom Configuration:**
```python
# src/your_project/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (
    FigRegistryHooks(
        auto_initialize=True,
        enable_performance_monitoring=True,
        fallback_on_error=False,
        max_initialization_time=0.003
    ),
)
```

**Selective Registration:**
```python
# Conditional registration based on environment
import os
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = []

# Only register in development and production environments
if os.getenv('KEDRO_ENV') in ['local', 'production']:
    HOOKS.append(FigRegistryHooks())
```

### Plugin Discovery

The hooks are automatically discoverable through Kedro's plugin system when `figregistry-kedro` is installed:

```bash
pip install figregistry-kedro
kedro info  # Verify plugin discovery
```

## Performance Specifications

### Execution Overhead

**Performance Targets (per Section 5.2.8):**
- `after_config_loaded`: <5ms per execution
- `before_pipeline_run`: <5ms per execution  
- `after_pipeline_run`: <2ms per execution
- Total hook overhead: <20ms per pipeline run

**Performance Monitoring:**
```python
hooks = FigRegistryHooks(enable_performance_monitoring=True)
# Detailed timing logs in debug output when enabled
```

**Memory Usage:**
- Hook state management: <1MB additional memory
- Configuration bridge: <2MB for complex configurations
- Total plugin overhead: <5MB per Section 6.6.3.6 requirements

### Thread Safety Specifications

**Concurrent Execution Support:**
- Thread-safe configuration bridge access through read-only operations
- Atomic pipeline registration and state management
- No synchronization overhead for parallel Kedro runners
- Independent hook state per pipeline execution context

**Parallel Runner Compatibility:**
- `SequentialRunner`: Full compatibility
- `ParallelRunner`: Thread-safe operation with shared configuration
- `ThreadRunner`: Concurrent execution support with state isolation

## Error Handling and Recovery

### Error Types

**ConfigurationMergeError:**
```python
# Raised when Kedro/FigRegistry configuration merging fails
try:
    # Hook execution automatically handles this
    pass
except ConfigurationMergeError as e:
    print(f"Configuration merge failed: {e}")
    for error in e.errors:
        print(f"  - {error.get('field', 'unknown')}: {error.get('message', 'unknown error')}")
```

**HookExecutionError:**
```python
# Raised when hook operations fail (if fallback_on_error=False)
try:
    # Hook execution
    pass
except HookExecutionError as e:
    print(f"Hook '{e.hook_name}' failed: {e}")
    if e.original_error:
        print(f"Original error: {e.original_error}")
```

### Graceful Degradation

**Fallback Behavior (when `fallback_on_error=True`):**
- Configuration initialization failures: Continue with warnings
- Bridge creation errors: Proceed without FigRegistry integration
- Environment detection failures: Use 'base' environment as default
- Performance threshold breaches: Log warnings but continue execution

**Recovery Mechanisms:**
- Late initialization attempts in `before_pipeline_run`
- Automatic retry for transient configuration errors
- Graceful handling of missing Kedro dependencies
- Detailed error logging for debugging

## Utility Functions

### `create_hooks(**config) -> FigRegistryHooks`

Create FigRegistryHooks instance with configuration override.

**Parameters:**
- `**config`: Configuration overrides for hook behavior

**Returns:**
- Configured FigRegistryHooks instance

**Example:**
```python
from figregistry_kedro.hooks import create_hooks

# In settings.py
HOOKS = (
    create_hooks(
        enable_performance_monitoring=True,
        fallback_on_error=False
    ),
)
```

### `get_hook_state() -> Optional[Dict[str, Any]]`

Get the state of globally registered FigRegistry hooks.

**Returns:**
- Hook state dictionary if hooks are active, None otherwise

**Note:** This function provides inspection capabilities for debugging and monitoring purposes.

## Integration Examples

### Basic Integration

```python
# settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)

# conf/base/figregistry.yml
figregistry_version: "0.3.0"
style:
  exploratory:
    color: "#1f77b4"
    marker: "o"
  presentation:
    color: "#ff7f0e"
    marker: "s"

# conf/base/catalog.yml
exploratory_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/exploratory_analysis.png
  condition_param: "exploration"
  purpose: "exploratory"
```

### Advanced Multi-Environment Setup

```python
# settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (
    FigRegistryHooks(
        enable_performance_monitoring=True,
        max_initialization_time=0.010  # 10ms for complex configs
    ),
)

# conf/base/figregistry.yml (base configuration)
figregistry_version: "0.3.0"
style:
  default:
    color: "#cccccc"
    marker: "."

# conf/local/figregistry.yml (development overrides)
style:
  default:
    color: "#ff0000"  # Red for development
  debug:
    color: "#00ff00"
    marker: "x"
```

### Pipeline Node Integration

```python
# nodes.py
import matplotlib.pyplot as plt

def create_analysis_plot(data):
    """Create analysis plot - no manual save required."""
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    ax.set_title('Analysis Results')
    return fig  # FigureDataSet handles styling and saving automatically

# pipeline.py
from kedro.pipeline import Pipeline, node

def create_pipeline():
    return Pipeline([
        node(
            func=create_analysis_plot,
            inputs="processed_data",
            outputs="analysis_plot",  # Configured as FigureDataSet in catalog
            name="create_analysis_node"
        )
    ])
```

## Troubleshooting

### Common Issues

**Hook Not Executing:**
- Verify registration in `settings.py`
- Check Kedro version compatibility (>=0.18.0,<0.20.0)
- Ensure `figregistry-kedro` is properly installed

**Configuration Merge Failures:**
- Validate YAML syntax in `conf/base/figregistry.yml`
- Check Pydantic schema compliance
- Verify environment-specific override syntax

**Performance Issues:**
- Enable performance monitoring to identify bottlenecks
- Check for complex configuration patterns
- Verify adequate system resources

### Debug Information

**Enable Debug Logging:**
```python
import logging
logging.getLogger('figregistry_kedro.hooks').setLevel(logging.DEBUG)
```

**State Inspection:**
```python
# During pipeline execution
from figregistry_kedro.hooks import get_hook_state

state = get_hook_state()
if state:
    print(f"Hook status: {state}")
```

## Version Compatibility

**Supported Versions:**
- Python: 3.10, 3.11, 3.12
- Kedro: 0.18.0 - 0.19.x
- FigRegistry: >=0.3.0

**Compatibility Matrix:**
| Python | Kedro 0.18.x | Kedro 0.19.x | Status |
|--------|--------------|--------------|--------|
| 3.10.x | ✅ Full support | ✅ Full support | Baseline compatibility |
| 3.11.x | ✅ Full support | ✅ Full support | Recommended |
| 3.12.x | ✅ Full support | ✅ Full support | Latest features |

## See Also

- [FigureDataSet API Reference](datasets.md) - Dataset implementation for figure persistence
- [Configuration Bridge API Reference](config.md) - Configuration merging and management
- [Installation Guide](../installation.md) - Setup and installation instructions
- [Configuration Guide](../configuration.md) - Detailed configuration options
- [Integration Examples](../../examples/) - Complete working examples

---

*This documentation corresponds to figregistry-kedro v0.1.0+. For the latest API updates, consult the [GitHub repository](https://github.com/your-org/figregistry-kedro).*