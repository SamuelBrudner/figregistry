# FigRegistryHooks API Reference

Complete API reference for the FigRegistryHooks class providing non-invasive lifecycle integration between FigRegistry and Kedro frameworks.

## Overview

The `FigRegistryHooks` component provides lifecycle management for FigRegistry initialization within Kedro's execution framework through the hook specification system. This component ensures proper FigRegistry context initialization, configuration management, and resource cleanup throughout Kedro pipeline execution while maintaining separation of concerns between frameworks.

**Key Features:**
- Non-invasive lifecycle integration preserving Kedro's execution model (F-006)
- Thread-safe operation for parallel pipeline execution
- <5ms hook execution overhead for minimal performance impact
- Automatic configuration bridge initialization and management
- Comprehensive error handling with graceful degradation
- Performance monitoring and metrics collection

## Class Definition

```python
class FigRegistryHooks:
    """
    Kedro lifecycle hooks for FigRegistry integration and context management.
    
    Provides non-invasive lifecycle integration between FigRegistry and Kedro 
    through the framework's hook specification system. Manages FigRegistry 
    initialization during pipeline startup, maintains configuration context 
    throughout execution, and ensures proper cleanup after pipeline completion.
    """
```

## Constructor

### `__init__()`

```python
def __init__(
    self,
    enable_performance_monitoring: bool = True,
    initialization_timeout_ms: float = 5000.0,
    config_cache_enabled: bool = True,
    strict_validation: bool = True,
    fallback_on_errors: bool = True
) -> None
```

Initialize FigRegistry lifecycle hooks with configuration options.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_performance_monitoring` | `bool` | `True` | Enable detailed performance tracking and metrics collection |
| `initialization_timeout_ms` | `float` | `5000.0` | Maximum time allowed for hook initialization (milliseconds) |
| `config_cache_enabled` | `bool` | `True` | Enable configuration caching for improved performance |
| `strict_validation` | `bool` | `True` | Enable strict validation for merged configurations |
| `fallback_on_errors` | `bool` | `True` | Enable graceful fallback when initialization fails |

**Example:**

```python
from figregistry_kedro.hooks import FigRegistryHooks

# Basic initialization
hooks = FigRegistryHooks()

# Custom configuration
hooks = FigRegistryHooks(
    enable_performance_monitoring=True,
    initialization_timeout_ms=3000.0,
    strict_validation=False,
    fallback_on_errors=True
)
```

## Hook Methods

### `after_config_loaded()` 

```python
@hook_impl
def after_config_loaded(
    self,
    context: KedroContext,
    config_loader: ConfigLoader,
    conf_source: str,
) -> None
```

Initialize FigRegistry configuration bridge after Kedro config loading.

This hook executes after Kedro's configuration system loads all project configurations, providing the optimal point for FigRegistry configuration bridge initialization. The hook merges Kedro's environment-specific configurations with FigRegistry's YAML settings while maintaining validation and type safety.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `context` | `KedroContext` | Kedro project context with loaded configurations |
| `config_loader` | `ConfigLoader` | Kedro ConfigLoader instance for accessing configurations |
| `conf_source` | `str` | Configuration source path for environment resolution |

**Behavior:**
- Initializes `FigRegistryConfigBridge` for configuration merging
- Merges Kedro environment-specific configurations with FigRegistry settings
- Validates merged configuration against Pydantic schema
- Caches merged configuration for pipeline execution
- Handles graceful fallback on configuration errors (if enabled)

**Performance Requirements:**
- Execution time target: <5ms per invocation
- Configuration merge time: <10ms for complex configurations
- Memory overhead: <2MB for enterprise configurations

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `HookInitializationError` | When configuration bridge initialization fails |
| `ConfigMergeError` | When configuration merging encounters validation errors |
| `ConfigValidationError` | When merged configuration fails Pydantic validation |

**Example Usage:**

The hook executes automatically during Kedro startup when registered in `settings.py`. Manual execution not recommended.

```python
# Hook executed automatically by Kedro framework
# User code: Register in settings.py
HOOKS = (FigRegistryHooks(),)
```

### `before_pipeline_run()`

```python
@hook_impl
def before_pipeline_run(
    self,
    run_params: Dict[str, Any],
    pipeline: Pipeline,
    catalog: DataCatalog
) -> None
```

Setup FigRegistry context before pipeline execution begins.

This hook executes immediately before Kedro pipeline execution, ensuring that FigRegistry configuration is properly initialized and available for FigureDataSet instances throughout the pipeline. The hook establishes the configuration context required for automated figure styling.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_params` | `Dict[str, Any]` | Pipeline execution parameters including run_id |
| `pipeline` | `Pipeline` | Kedro pipeline instance to be executed |
| `catalog` | `DataCatalog` | Data catalog with dataset configurations |

**Behavior:**
- Validates hook initialization state from `after_config_loaded`
- Initializes FigRegistry with merged configuration context
- Establishes styling context for automated figure processing
- Validates catalog integration for FigureDataSet entries
- Tracks active pipeline contexts for cleanup management

**Performance Requirements:**
- Execution time target: <5ms per pipeline execution
- Context setup overhead: <1% of total pipeline execution time
- Memory usage: Minimal additional allocation beyond configuration cache

**Thread Safety:**
- Fully thread-safe for concurrent pipeline execution
- Independent styling contexts for parallel pipeline nodes
- Atomic configuration operations preventing race conditions

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `HookExecutionError` | When context setup fails without fallback enabled |
| `ConfigValidationError` | When configuration state is invalid for pipeline execution |

**Example Usage:**

```python
# Automatic execution during pipeline run
kedro run --pipeline data_visualization

# Context established automatically for downstream FigureDataSet usage
```

### `after_pipeline_run()`

```python
@hook_impl
def after_pipeline_run(
    self,
    run_params: Dict[str, Any],
    pipeline: Pipeline,
    catalog: DataCatalog
) -> None
```

Cleanup FigRegistry context and resources after pipeline execution.

This hook executes after Kedro pipeline completion, ensuring proper cleanup of FigRegistry configuration state and resources. The hook maintains system stability and prevents resource leaks in long-running or repeatedly executed pipeline environments.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_params` | `Dict[str, Any]` | Pipeline execution parameters including run_id |
| `pipeline` | `Pipeline` | Kedro pipeline instance that was executed |
| `catalog` | `DataCatalog` | Data catalog used during execution |

**Behavior:**
- Removes active pipeline context from tracking
- Logs performance metrics if monitoring enabled
- Clears FigRegistry state when no active contexts remain
- Updates global hook state for resource management
- Performs cleanup without disrupting pipeline completion

**Performance Requirements:**
- Execution time target: <5ms per pipeline completion
- Resource cleanup: Complete memory release for inactive contexts
- No impact on pipeline completion status or timing

**Error Handling:**
- Non-blocking cleanup prevents pipeline completion disruption
- Errors logged but not propagated to avoid masking pipeline issues
- Graceful degradation for partial cleanup scenarios

**Example Usage:**

```python
# Automatic execution after pipeline completion
# No user intervention required - cleanup handled transparently
```

## State Management Methods

### `is_initialized()`

```python
def is_initialized(self) -> bool
```

Check if hooks are properly initialized and ready for use.

**Returns:**
- `bool`: `True` if hooks are initialized and configuration bridge is ready, `False` otherwise

**Example:**

```python
hooks = FigRegistryHooks()
print(f"Hooks ready: {hooks.is_initialized()}")  # False before config load

# After Kedro calls after_config_loaded hook
print(f"Hooks ready: {hooks.is_initialized()}")  # True if successful
```

### `get_current_config()`

```python
def get_current_config(self) -> Optional[Dict[str, Any]]
```

Get the current FigRegistry configuration established by hooks.

**Returns:**
- `Optional[Dict[str, Any]]`: Current merged configuration dictionary or `None` if not initialized

**Example:**

```python
hooks = FigRegistryHooks()
config = hooks.get_current_config()

if config:
    print(f"Available styles: {list(config.get('styles', {}).keys())}")
    print(f"Output settings: {config.get('outputs', {})}")
```

### `clear_state()`

```python
def clear_state(self) -> None
```

Clear all hook state and force re-initialization on next use.

This method is useful for testing scenarios or when configuration changes require complete hook re-initialization.

**Behavior:**
- Resets initialization flag and configuration bridge
- Clears active contexts and performance metrics
- Updates global hook state for coordination
- Forces complete re-initialization on next hook execution

**Example:**

```python
hooks = FigRegistryHooks()
# ... hook usage ...

# Reset for testing or configuration changes
hooks.clear_state()
assert not hooks.is_initialized()
```

## Performance and Monitoring

### `get_performance_metrics()`

```python
def get_performance_metrics(self) -> Dict[str, Any]
```

Get comprehensive performance metrics for monitoring and optimization.

**Returns:**
- `Dict[str, Any]`: Dictionary containing detailed performance statistics

**Metrics Structure:**

```python
{
    "hook_invocations": int,           # Total hook method calls
    "initialization_times": List[float],  # Hook initialization times (ms)
    "cleanup_times": List[float],      # Cleanup operation times (ms)
    "errors": List[Dict],              # Error information with timestamps
    "warnings": List[Dict],            # Performance warnings
    "current_state": {
        "initialized": bool,           # Current initialization status
        "active_contexts": int,        # Number of active pipeline contexts
        "has_config": bool,           # Configuration availability
        "config_keys": List[str]      # Available configuration sections
    },
    "config_bridge": Dict            # Configuration bridge performance metrics
}
```

**Example:**

```python
hooks = FigRegistryHooks()
# ... after pipeline execution ...

metrics = hooks.get_performance_metrics()
print(f"Average initialization time: {sum(metrics['initialization_times']) / len(metrics['initialization_times'])}ms")
print(f"Total hook invocations: {metrics['hook_invocations']}")
print(f"Active contexts: {metrics['current_state']['active_contexts']}")
```

## Registration and Setup

### Settings Configuration

Register hooks in your Kedro project's `settings.py` file:

```python
# src/project_name/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

# Basic registration
HOOKS = (FigRegistryHooks(),)

# Custom configuration
HOOKS = (
    FigRegistryHooks(
        enable_performance_monitoring=True,
        initialization_timeout_ms=3000.0,
        strict_validation=False,
        fallback_on_errors=True
    ),
)

# Multiple hooks with different configurations
HOOKS = (
    FigRegistryHooks(enable_performance_monitoring=True),
    # Other hooks...
)
```

### Project Structure Requirements

Ensure your Kedro project has the following configuration structure:

```
conf/
├── base/
│   ├── catalog.yml
│   └── figregistry.yml          # Optional FigRegistry configuration
├── local/
│   └── figregistry.yml          # Environment-specific overrides
└── production/
    └── figregistry.yml          # Production configuration
```

### Configuration Integration

The hooks automatically merge Kedro and FigRegistry configurations:

```yaml
# conf/base/figregistry.yml
metadata:
  config_version: "1.0.0"
  created_by: "kedro-figregistry-integration"

styles:
  experiment_a:
    color: "#2E86AB"
    marker: "o"
    linestyle: "-"
  
outputs:
  base_path: "data/08_reporting"
  naming:
    template: "{name}_{condition}_{ts}"

kedro:
  hooks:
    performance_monitoring: true
    initialization_timeout_ms: 5000
```

## Error Handling

### Exception Types

The hooks provide specific exception types for different error scenarios:

#### `HookInitializationError`

```python
class HookInitializationError(Exception):
    """Raised when hook initialization fails with detailed error information."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause
```

**Common Causes:**
- Missing Kedro framework dependencies
- FigRegistry package not available
- Configuration bridge initialization failure
- Timeout exceeded during initialization

#### `HookExecutionError`

```python
class HookExecutionError(Exception):
    """Raised when hook execution encounters errors during pipeline lifecycle."""
    
    def __init__(self, message: str, hook_method: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.hook_method = hook_method
        self.cause = cause
```

**Common Causes:**
- Context setup failure during `before_pipeline_run`
- Configuration validation errors
- Resource allocation failures
- Thread safety violations

### Error Recovery

The hooks provide several error recovery mechanisms:

#### Graceful Fallback

When `fallback_on_errors=True` (default):

```python
# Hooks continue operation with minimal functionality
# Degraded mode provides basic configuration without styling
# Pipeline execution proceeds without FigRegistry features
```

#### Strict Validation Mode

When `strict_validation=True` (default):

```python
# Configuration errors halt pipeline execution
# Clear error messages identify configuration issues
# Prevents silent failures in production environments
```

#### Lazy Initialization

Automatic fallback initialization when normal hook lifecycle fails:

```python
def _attempt_lazy_initialization(self) -> None:
    """Attempt lazy initialization when hooks weren't properly initialized."""
    # Creates minimal configuration bridge with default settings
    # Maintains basic functionality for degraded operation
    # Logs warnings for troubleshooting configuration issues
```

## Thread Safety

### Concurrent Pipeline Execution

The hooks are designed for thread-safe operation supporting Kedro's parallel execution models:

- **Independent Contexts**: Each pipeline execution maintains separate styling contexts
- **Atomic Operations**: Configuration access uses atomic dictionary operations
- **Thread-Local Storage**: Pipeline-specific state isolated per thread
- **Lock-Free Design**: Minimizes synchronization overhead for performance

### Performance Guidelines

**Recommended Concurrency Limits:**
- Up to 4 concurrent figure operations per CPU core
- Maximum 16 parallel pipeline executions for optimal performance
- Configuration caching shared across threads for efficiency

**Memory Management:**
- Thread-safe configuration cache with automatic cleanup
- Independent memory allocation per pipeline context
- Garbage collection operates normally without manual intervention

## Integration Examples

### Basic Integration

Minimal setup for FigRegistry-Kedro integration:

```python
# settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
```

```yaml
# conf/base/catalog.yml
styled_analysis_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/analysis_plot.png
  purpose: exploratory
  condition_param: experiment_type
```

### Advanced Configuration

Enterprise setup with comprehensive monitoring:

```python
# settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (
    FigRegistryHooks(
        enable_performance_monitoring=True,
        initialization_timeout_ms=3000.0,
        config_cache_enabled=True,
        strict_validation=True,
        fallback_on_errors=False  # Fail fast in production
    ),
)
```

```yaml
# conf/base/figregistry.yml
metadata:
  config_version: "1.0.0"
  description: "Production FigRegistry configuration"

styles:
  control_group:
    color: "#95A5A6"
    marker: "o"
    label: "Control"
  treatment_a:
    color: "#3498DB"
    marker: "s"
    label: "Treatment A"
  treatment_b:
    color: "#E74C3C"
    marker: "^"
    label: "Treatment B"

outputs:
  base_path: "data/08_reporting"
  naming:
    template: "{pipeline}_{name}_{condition}_{ts}"
    timestamp_format: "%Y%m%d_%H%M%S"

kedro:
  hooks:
    performance_monitoring: true
    initialization_timeout_ms: 3000
    cache_enabled: true
    strict_validation: true
```

### Testing Integration

Test-specific hook configuration:

```python
# tests/conftest.py
import pytest
from figregistry_kedro.hooks import FigRegistryHooks, clear_global_hook_state

@pytest.fixture(autouse=True)
def reset_hooks():
    """Reset hook state between tests."""
    clear_global_hook_state()
    yield
    clear_global_hook_state()

@pytest.fixture
def test_hooks():
    """Provide test-configured hooks."""
    return FigRegistryHooks(
        enable_performance_monitoring=False,
        initialization_timeout_ms=1000.0,
        strict_validation=False,
        fallback_on_errors=True
    )
```

## Global State Management

### Global Functions

#### `get_global_hook_state()`

```python
def get_global_hook_state() -> Dict[str, Any]
```

Get current global hook state for monitoring and debugging.

**Returns:**
- `Dict[str, Any]`: Global state information including initialization status, active instances, and performance metrics

#### `clear_global_hook_state()`

```python
def clear_global_hook_state() -> None
```

Clear global hook state for testing and cleanup purposes.

This function is primarily used for testing scenarios where complete state reset is required between test cases.

### Context Manager

#### `hook_context()`

```python
@contextmanager
def hook_context():
    """Context manager for temporary hook state management."""
```

Provides controlled access to hook state for testing and debugging purposes with automatic cleanup.

**Example:**

```python
from figregistry_kedro.hooks import hook_context

with hook_context() as hook_state:
    # Hook state is isolated within this context
    hook_state["initialized"] = True
    # Modifications are automatically reverted on exit
```

## Performance Targets

### Execution Time Requirements

| Operation | Target Time | Measurement |
|-----------|-------------|-------------|
| Hook initialization | <5ms | Per `after_config_loaded` invocation |
| Pipeline context setup | <5ms | Per `before_pipeline_run` invocation |
| Configuration merge | <10ms | Complex enterprise configurations |
| Context cleanup | <5ms | Per `after_pipeline_run` invocation |
| State queries | <1ms | Per `is_initialized()` or `get_current_config()` call |

### Memory Usage Guidelines

| Component | Target Usage | Scaling |
|-----------|-------------|---------|
| Configuration cache | <2MB | Linear with total configuration size |
| Active contexts | <1MB | Per concurrent pipeline execution |
| Performance metrics | <100KB | Per hook instance |
| Global state | <500KB | Constant regardless of usage |

### Concurrency Limits

| Scenario | Recommended Limit | Performance Impact |
|----------|------------------|-------------------|
| Concurrent pipelines | 16 parallel executions | Optimal resource utilization |
| Figure operations | 4 per CPU core | Aligns with matplotlib constraints |
| Configuration access | Unlimited | Lock-free design supports high concurrency |
| Hook instances | 1 per Kedro project | Shared global state for coordination |

## See Also

- [FigRegistryConfigBridge API](config.md) - Configuration merging and validation
- [FigureDataSet API](datasets.md) - Automated figure styling in Kedro catalogs
- [Configuration Guide](../configuration.md) - Complete configuration setup instructions
- [Installation Guide](../installation.md) - Package installation and setup