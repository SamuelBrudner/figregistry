# FigRegistryConfigBridge API Documentation

## Overview

The `FigRegistryConfigBridge` class serves as the configuration translation layer between Kedro's ConfigLoader system and FigRegistry's YAML-based configuration management. This bridge enables seamless integration of environment-specific Kedro parameters with FigRegistry's condition-based styling system while maintaining configuration validation and type safety across both frameworks.

## Features

- **Configuration Merging**: Seamlessly merges Kedro ConfigLoader with FigRegistry YAML configurations (F-007)
- **Environment Support**: Environment-specific configuration support for multi-stage deployments (base, local, staging, production)
- **High Performance**: <10ms configuration merging overhead target (Section 5.2.8)
- **Thread Safety**: Supports concurrent access for parallel Kedro execution
- **Validation**: Comprehensive Pydantic validation for merged configurations
- **Error Handling**: Detailed error aggregation for configuration validation failures

---

## Classes

### FigRegistryConfigBridge

Configuration translation layer for Kedro-FigRegistry integration.

```python
class FigRegistryConfigBridge:
    """
    Configuration translation layer between Kedro's ConfigLoader and FigRegistry's
    YAML-based configuration management system.
    """
```

#### Constructor

```python
def __init__(
    self,
    cache_enabled: bool = True,
    validation_strict: bool = True,
    performance_target_ms: float = 10.0,
    max_cache_size: int = 1000
) -> None
```

Initialize the FigRegistry configuration bridge.

**Parameters:**
- `cache_enabled` (`bool`, optional): Enable configuration caching for performance. Default: `True`
- `validation_strict` (`bool`, optional): Enable strict validation for merged configurations. Default: `True`  
- `performance_target_ms` (`float`, optional): Target merge time in milliseconds. Default: `10.0`
- `max_cache_size` (`int`, optional): Maximum cache entries for merged configurations. Default: `1000`

**Example:**
```python
from figregistry_kedro.config import FigRegistryConfigBridge

# Default configuration
bridge = FigRegistryConfigBridge()

# Custom configuration
bridge = FigRegistryConfigBridge(
    cache_enabled=True,
    validation_strict=True,
    performance_target_ms=5.0,
    max_cache_size=500
)
```

---

#### merge_configurations

```python
def merge_configurations(
    self,
    config_loader: Optional[Any] = None,
    environment: str = "base",
    project_path: Optional[Path] = None,
    figregistry_config_name: str = "figregistry",
    **override_params
) -> Dict[str, Any]
```

Merge Kedro ConfigLoader configurations with FigRegistry YAML settings.

This method implements the core configuration bridge functionality (F-007-RQ-002) by loading configurations from both Kedro's ConfigLoader system and traditional figregistry.yaml files, then merging them with clear precedence rules while maintaining validation and type safety.

**Parameters:**
- `config_loader` (`Optional[Any]`): Kedro ConfigLoader instance for loading project configurations
- `environment` (`str`, optional): Environment name for configuration resolution. Default: `"base"`
- `project_path` (`Optional[Path]`): Project root path for configuration file discovery
- `figregistry_config_name` (`str`, optional): Name of FigRegistry config file (without .yml extension). Default: `"figregistry"`
- `**override_params`: Additional parameters to override in merged configuration

**Returns:**
- `Dict[str, Any]`: Merged and validated configuration ready for FigRegistry initialization

**Raises:**
- `ConfigMergeError`: When configuration merging fails
- `ConfigValidationError`: When validation of merged configuration fails (F-007-RQ-003)

**Configuration Precedence Rules:**
1. Override parameters passed directly to method (highest priority)
2. Kedro configuration values (environment-specific)
3. FigRegistry configuration values
4. Default values (lowest priority)

**Environment-Specific Configuration Support:**
The bridge supports multiple configuration environments following Kedro conventions:

- `base`: Base configuration from `conf/base/figregistry.yml`
- `local`: Local development overrides from `conf/local/figregistry.yml`
- `staging`: Staging environment from `conf/staging/figregistry.yml`
- `production`: Production environment from `conf/production/figregistry.yml`

**Performance Specifications:**
- Target merge time: <10ms (configurable via `performance_target_ms`)
- Memory overhead: <2MB for enterprise-scale configurations
- Cache hit ratio: >90% in typical usage scenarios

**Example:**
```python
from kedro.config import ConfigLoader
from pathlib import Path

# Basic usage
bridge = FigRegistryConfigBridge()
config = bridge.merge_configurations(
    config_loader=config_loader,
    environment="local"
)

# Advanced usage with overrides
config = bridge.merge_configurations(
    config_loader=config_loader,
    environment="production",
    project_path=Path("/path/to/project"),
    figregistry_config_name="custom_figregistry",
    experiment_condition="treatment_A",
    output_base_path="/custom/outputs"
)
```

---

#### get_performance_metrics

```python
def get_performance_metrics(self) -> Dict[str, Any]
```

Get performance metrics for configuration bridge operations.

**Returns:**
- `Dict[str, Any]`: Dictionary containing performance statistics

**Metrics Included:**
- `merge_times`: Statistics on configuration merge operations
  - `count`: Number of merge operations performed
  - `average_ms`: Average merge time in milliseconds
  - `max_ms`: Maximum merge time observed
  - `min_ms`: Minimum merge time observed
  - `target_ms`: Configured performance target
- `cache_stats`: Cache performance statistics
  - `hits`: Number of cache hits
  - `misses`: Number of cache misses
- `cache_size`: Current number of cached configurations
- `max_cache_size`: Maximum cache size limit

**Example:**
```python
bridge = FigRegistryConfigBridge()
# ... perform operations ...
metrics = bridge.get_performance_metrics()

print(f"Average merge time: {metrics['merge_times']['average_ms']:.2f}ms")
print(f"Cache hit ratio: {metrics['cache_stats']['hits'] / (metrics['cache_stats']['hits'] + metrics['cache_stats']['misses']):.2%}")
```

---

#### clear_cache

```python
def clear_cache(self) -> None
```

Clear configuration cache and reset cache statistics.

This method is useful for testing, debugging, or when configuration files have been modified and cached configurations should be invalidated.

**Example:**
```python
bridge = FigRegistryConfigBridge()
# ... perform cached operations ...
bridge.clear_cache()  # Force reload of configurations
```

---

### FigRegistryConfigSchema

Pydantic schema for validating merged FigRegistry configurations.

```python
class FigRegistryConfigSchema(BaseModel):
    """
    Pydantic schema for validating merged FigRegistry configurations.
    
    This schema ensures type safety and validation for configurations that merge
    Kedro ConfigLoader settings with traditional FigRegistry YAML configurations.
    """
```

**Core Configuration Sections:**

#### Required Fields

- `styles` (`Dict[str, Dict[str, Any]]`): Condition-based style mappings for experimental visualizations
- `defaults` (`Dict[str, Any]`): Default styling parameters and fallback configurations
- `outputs` (`Dict[str, Any]`): Output management configuration for automated file handling

#### Optional Fields

- `figregistry_version` (`Optional[str]`): FigRegistry version constraint for compatibility validation
- `metadata` (`Optional[Dict[str, Any]]`): Configuration metadata for tracking and validation
- `palettes` (`Optional[Dict[str, Union[List[str], Dict[str, str]]]]`): Color palettes and fallback styling
- `kedro` (`Optional[Dict[str, Any]]`): Kedro-specific configuration extensions and integration settings
- `style_inheritance` (`Optional[Dict[str, Any]]`): Style inheritance and composition rules
- `conditional_rules` (`Optional[Dict[str, Any]]`): Conditional styling rules for complex scenarios
- `performance` (`Optional[Dict[str, Any]]`): Performance monitoring and optimization settings
- `validation` (`Optional[Dict[str, Any]]`): Configuration validation schema and rules
- `examples` (`Optional[Dict[str, Any]]`): Usage examples and documentation

**Validation Rules:**

1. **Style Validation**: Each style must contain at minimum a color specification
2. **Output Validation**: Output configuration must include 'base_path' field
3. **Version Validation**: Version constraints must follow PEP 440 format
4. **Extensibility**: Additional fields allowed for future extensibility

**Example:**
```python
from figregistry_kedro.config import FigRegistryConfigSchema

# Validate configuration
config_dict = {
    "styles": {
        "exploratory": {"color": "#A8E6CF", "marker": "o"},
        "presentation": {"color": "#FFB6C1", "linewidth": 2.0}
    },
    "defaults": {
        "figure": {"figsize": [10, 8], "dpi": 150}
    },
    "outputs": {
        "base_path": "data/08_reporting"
    }
}

# This will validate and potentially raise ValidationError
validated_config = FigRegistryConfigSchema(**config_dict)
```

---

## Exception Classes

### ConfigMergeError

Raised when configuration merging fails with validation or processing errors.

```python
class ConfigMergeError(Exception):
    """Raised when configuration merging fails with validation or processing errors."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []
```

**Attributes:**
- `errors` (`List[str]`): List of specific error messages that contributed to the failure

### ConfigValidationError

Raised when configuration validation fails with detailed error information.

```python
class ConfigValidationError(Exception):
    """Raised when configuration validation fails with detailed error information."""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or []
```

**Attributes:**
- `validation_errors` (`List[str]`): List of Pydantic validation error messages

---

## Module Functions

### init_config

```python
def init_config(
    config_loader: Optional[Any] = None,
    environment: str = "base",
    project_path: Optional[Path] = None,
    **kwargs
) -> Optional[Any]
```

Initialize FigRegistry configuration through the configuration bridge.

This function provides the primary interface for initializing FigRegistry with merged Kedro and FigRegistry configurations during pipeline startup. It implements F-007-RQ-001 (figregistry.yaml loading via Kedro ConfigLoader).

**Parameters:**
- `config_loader` (`Optional[Any]`): Kedro ConfigLoader instance for loading project configurations
- `environment` (`str`, optional): Environment name for configuration resolution. Default: `"base"`
- `project_path` (`Optional[Path]`): Project root path for configuration file discovery
- `**kwargs`: Additional parameters to pass to configuration bridge

**Returns:**
- `Optional[Any]`: FigRegistry configuration object if successful, None otherwise

**Raises:**
- `ConfigMergeError`: When configuration merging fails
- `ConfigValidationError`: When validation of merged configuration fails

**Usage in Kedro Hooks:**
```python
from figregistry_kedro.config import init_config

class MyHooks:
    def before_pipeline_run(self, run_params, pipeline, catalog):
        # Initialize FigRegistry with merged configuration
        config = init_config(
            config_loader=run_params.get('config_loader'),
            environment=run_params.get('env', 'base')
        )
        return config
```

**Advanced Usage:**
```python
from pathlib import Path

# Production environment with custom settings
config = init_config(
    config_loader=config_loader,
    environment="production",
    project_path=Path("/path/to/project"),
    validation_strict=True,
    cache_enabled=True
)
```

---

### get_merged_config

```python
def get_merged_config(
    config_loader: Optional[Any] = None,
    environment: str = "base",
    project_path: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]
```

Get merged configuration without initializing FigRegistry.

This function provides access to the merged configuration dictionary for inspection or manual initialization purposes. It performs the same configuration merging as `init_config()` but returns the raw dictionary instead of initializing FigRegistry.

**Parameters:**
- `config_loader` (`Optional[Any]`): Kedro ConfigLoader instance
- `environment` (`str`, optional): Environment name for configuration resolution. Default: `"base"`
- `project_path` (`Optional[Path]`): Project root path for configuration file discovery
- `**kwargs`: Additional parameters for configuration bridge

**Returns:**
- `Dict[str, Any]`: Merged configuration dictionary

**Example:**
```python
from figregistry_kedro.config import get_merged_config

# Get merged configuration for inspection
config = get_merged_config(config_loader, environment="staging")
print(f"Available styles: {list(config['styles'].keys())}")
print(f"Output base path: {config['outputs']['base_path']}")

# Inspect Kedro-specific parameters
if '_kedro_context' in config:
    kedro_params = config['_kedro_context']['parameters']
    print(f"Experiment condition: {kedro_params.get('experiment_condition')}")
```

---

## Configuration Merging Details

### Environment-Specific Configuration Paths

The configuration bridge searches for FigRegistry configuration files in the following order:

1. **Project Root**: `{project_path}/figregistry.yaml`, `{project_path}/figregistry.yml`
2. **Kedro Base**: `{project_path}/conf/base/figregistry.yml`, `{project_path}/conf/base/figregistry.yaml`
3. **Environment-Specific**: `{project_path}/conf/{environment}/figregistry.yml`, `{project_path}/conf/{environment}/figregistry.yaml`

### Parameter Integration

The bridge automatically extracts and integrates Kedro parameters into FigRegistry configuration:

**Experimental Condition Parameters:**
- `experiment_condition`: Primary condition for style resolution
- `experiment_phase`: Secondary condition parameter
- `analysis_stage`: Analysis stage identifier
- `model_type`: Model type for conditional styling

**Visualization Settings:**
- `plot_settings.figure_size`: Mapped to `defaults.figure.figsize`
- `plot_settings.dpi`: Mapped to `defaults.figure.dpi`

**Output Configuration:**
- `execution_config.output_base_path`: Mapped to `outputs.base_path`
- `execution_config.figure_formats`: Mapped to `outputs.formats.defaults.exploratory`

### Kedro Context Integration

The merged configuration includes a special `_kedro_context` section containing:

```python
{
    "_kedro_context": {
        "parameters": {
            # All Kedro parameters from config_loader.get("parameters")
        }
    },
    "condition_parameters": {
        # Extracted condition parameters for style resolution
        "experiment_condition": "treatment_A",
        "experiment_phase": "phase_1"
    }
}
```

This enables FigureDataSet instances to resolve experimental conditions dynamically from pipeline run context.

---

## Thread Safety and Concurrency

The FigRegistryConfigBridge is designed for thread-safe operation in concurrent Kedro environments:

### Thread Safety Features

1. **Immutable Configurations**: Once merged, configurations are treated as immutable
2. **Thread-Safe Caching**: Configuration cache uses threading.Lock for synchronization
3. **Independent Instances**: Each bridge instance maintains separate state
4. **Atomic Operations**: File system operations are atomic where possible

### Concurrency Guidelines

- **Parallel Pipelines**: Multiple Kedro pipelines can safely use the same bridge instance
- **Concurrent Datasets**: FigureDataSet instances can safely access merged configurations concurrently
- **Cache Contention**: Cache operations are serialized but optimized for minimal lock contention
- **Memory Safety**: Deep copying ensures no shared mutable state between threads

**Example Multi-threaded Usage:**
```python
import threading
from figregistry_kedro.config import FigRegistryConfigBridge

# Shared bridge instance (thread-safe)
bridge = FigRegistryConfigBridge()

def pipeline_worker(environment):
    """Worker function for parallel pipeline execution"""
    config = bridge.merge_configurations(
        config_loader=config_loader,
        environment=environment
    )
    # Use config safely in this thread
    return config

# Run multiple environments concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(pipeline_worker, env) 
        for env in ["local", "staging", "production"]
    ]
    configs = [future.result() for future in futures]
```

---

## Performance Optimization

### Caching Strategy

The configuration bridge implements an intelligent caching system:

1. **Cache Key Generation**: Based on config_loader, environment, project_path, and override parameters
2. **LRU Eviction**: Simple FIFO eviction when cache size limit reached
3. **Cache Invalidation**: Manual cache clearing available for testing/debugging
4. **Memory Management**: Deep copying prevents cache corruption

### Performance Monitoring

Built-in performance tracking provides visibility into operation efficiency:

```python
bridge = FigRegistryConfigBridge(performance_target_ms=5.0)

# Perform operations...
metrics = bridge.get_performance_metrics()

# Check if performance targets are met
avg_time = metrics['merge_times']['average_ms']
if avg_time > bridge.performance_target_ms:
    print(f"Performance degradation detected: {avg_time}ms > {bridge.performance_target_ms}ms")

# Monitor cache efficiency
cache_hit_ratio = metrics['cache_stats']['hits'] / (
    metrics['cache_stats']['hits'] + metrics['cache_stats']['misses']
)
print(f"Cache hit ratio: {cache_hit_ratio:.2%}")
```

---

## Error Handling and Debugging

### Configuration Validation Errors

When configuration validation fails, detailed error information is provided:

```python
try:
    config = bridge.merge_configurations(config_loader, environment="production")
except ConfigValidationError as e:
    print(f"Configuration validation failed: {e}")
    for error in e.validation_errors:
        print(f"  - {error}")
```

### Merge Errors

Configuration merge failures provide comprehensive error context:

```python
try:
    config = bridge.merge_configurations(config_loader, environment="invalid")
except ConfigMergeError as e:
    print(f"Configuration merge failed: {e}")
    for error in e.errors:
        print(f"  - {error}")
```

### Debug Logging

Enable debug logging to trace configuration operations:

```python
import logging

# Enable debug logging for configuration bridge
logging.getLogger('figregistry_kedro.config').setLevel(logging.DEBUG)

# Operations will now log detailed information
bridge = FigRegistryConfigBridge()
config = bridge.merge_configurations(config_loader, environment="local")
```

---

## Integration Examples

### Basic Kedro Project Integration

```python
# In your Kedro project's hooks.py
from figregistry_kedro.config import init_config
from kedro.framework.hooks import hook_impl

class FigRegistryProjectHooks:
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """Initialize FigRegistry before pipeline execution"""
        config_loader = run_params.get('config_loader')
        environment = run_params.get('env', 'base')
        
        # Initialize FigRegistry with merged configuration
        figregistry_config = init_config(
            config_loader=config_loader,
            environment=environment
        )
        
        # Store in run context for datasets
        run_params['figregistry_config'] = figregistry_config
```

### Custom Configuration Override

```python
# Advanced configuration with custom overrides
from figregistry_kedro.config import FigRegistryConfigBridge
from pathlib import Path

def setup_custom_figregistry(project_path: str, experiment_id: str):
    """Setup FigRegistry with custom experimental configuration"""
    bridge = FigRegistryConfigBridge(
        validation_strict=True,
        performance_target_ms=5.0
    )
    
    config = bridge.merge_configurations(
        environment="production",
        project_path=Path(project_path),
        # Custom experimental parameters
        experiment_condition=f"experiment_{experiment_id}",
        experiment_phase="treatment",
        output_base_path=f"/experiments/{experiment_id}/outputs",
        # Performance overrides
        figure_dpi=300,
        figure_formats=["png", "pdf", "svg"]
    )
    
    return config
```

---

## API Reference Summary

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| `FigRegistryConfigBridge` | Main configuration bridge class | `merge_configurations()`, `get_performance_metrics()`, `clear_cache()` |
| `FigRegistryConfigSchema` | Pydantic validation schema | Automatic validation via constructor |
| `init_config()` | Primary initialization function | Initializes FigRegistry with merged config |
| `get_merged_config()` | Configuration inspection function | Returns raw merged configuration |
| `ConfigMergeError` | Configuration merge exception | Exception with error details |
| `ConfigValidationError` | Configuration validation exception | Exception with validation details |

## Requirements Compliance

This API documentation addresses the following requirements:

- **F-007**: FigRegistry-Kedro Config Bridge complete functionality
- **F-007-RQ-001**: figregistry.yaml loading via Kedro ConfigLoader (implemented in `init_config()`)
- **F-007-RQ-002**: Project-specific and default configuration merging (detailed in precedence rules)
- **F-007-RQ-003**: Merged configuration Pydantic schema validation (implemented in `FigRegistryConfigSchema`)
- **Section 5.2.5**: FigRegistryConfigBridge component architecture and patterns
- **Performance Requirements**: <10ms configuration merging overhead with monitoring
- **Thread Safety**: Concurrent access patterns for parallel Kedro execution