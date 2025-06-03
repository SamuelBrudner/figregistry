# FigRegistryConfigBridge API Reference

The `FigRegistryConfigBridge` serves as the configuration translation layer between Kedro's ConfigLoader system and FigRegistry's YAML-based configuration management. This bridge enables seamless merging of environment-specific Kedro parameters with traditional `figregistry.yaml` settings while maintaining validation and type safety across both frameworks.

## Overview

The configuration bridge solves the challenge of unifying two different configuration systems by:

- **Seamless Configuration Merging**: Combines standalone `figregistry.yaml` files with Kedro's environment-specific configuration management
- **Type Safety**: Validates merged configurations using Pydantic schemas to ensure data integrity
- **Performance Optimization**: Implements caching and lazy evaluation to achieve <10ms configuration merging overhead
- **Thread Safety**: Supports concurrent access patterns required for parallel Kedro runner execution
- **Environment Awareness**: Handles multiple deployment environments (base, local, staging, production)

## Core Components

### FigRegistryConfigBridge Class

The main bridge class that orchestrates configuration merging and validation.

```python
from figregistry_kedro.config import FigRegistryConfigBridge
from kedro.config import ConfigLoader

# Initialize with Kedro ConfigLoader
config_loader = ConfigLoader("conf")
bridge = FigRegistryConfigBridge(
    config_loader=config_loader,
    environment="production",
    enable_caching=True
)

# Get merged configuration
merged_config = bridge.get_merged_config()
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_loader` | `Optional[ConfigLoader]` | `None` | Kedro ConfigLoader instance for accessing project configuration |
| `environment` | `Optional[str]` | `"base"` | Target environment for configuration resolution |
| `enable_caching` | `bool` | `True` | Enable configuration caching for performance optimization |

#### Key Methods

##### `get_merged_config() -> FigRegistryKedroConfig`

The primary method for obtaining merged and validated configuration.

**Returns**: Validated `FigRegistryKedroConfig` instance with merged settings

**Raises**: 
- `ConfigurationMergeError`: If configuration merging or validation fails
- `ValidationError`: If Pydantic validation fails

**Performance**: Guarantees <10ms execution time for configuration merging

```python
try:
    config = bridge.get_merged_config()
    print(f"Loaded configuration for environment: {config.environment}")
    print(f"Available styles: {list(config.styles.keys())}")
except ConfigurationMergeError as e:
    print(f"Configuration error: {e}")
    print(f"Error details: {e.errors}")
```

##### `clear_cache()`

Clears the instance-level configuration cache for forced reload.

```python
# Force configuration reload
bridge.clear_cache()
config = bridge.get_merged_config()  # Will reload from files
```

##### `clear_global_cache()` (Class Method)

Clears the global configuration cache shared across all bridge instances.

```python
# Clear global cache (affects all instances)
FigRegistryConfigBridge.clear_global_cache()
```

### FigRegistryKedroConfig Model

Pydantic model that ensures type safety across both configuration systems and provides comprehensive validation for merged configuration structures.

```python
from figregistry_kedro.config import FigRegistryKedroConfig

# Access configuration sections
config = bridge.get_merged_config()

# Core FigRegistry sections
styles = config.styles  # Dict[str, Dict[str, Any]]
palettes = config.palettes  # Dict[str, Any]
outputs = config.outputs  # Dict[str, Any]
defaults = config.defaults  # Dict[str, Any]

# Kedro-specific integration
kedro_settings = config.kedro  # Dict[str, Any]
environment = config.environment  # str

# Configuration flags
concurrent_access = config.enable_concurrent_access  # bool
validation_enabled = config.validation_enabled  # bool
```

#### Configuration Sections

| Section | Type | Description |
|---------|------|-------------|
| `styles` | `Dict[str, Dict[str, Any]]` | Condition-based style mappings for experimental visualizations |
| `palettes` | `Dict[str, Any]` | Color palette definitions and defaults |
| `outputs` | `Dict[str, Any]` | Output path configurations and naming conventions |
| `defaults` | `Dict[str, Any]` | Default styling parameters and fallback values |
| `kedro` | `Dict[str, Any]` | Kedro-specific configuration overrides and extensions |
| `environment` | `str` | Current environment for configuration resolution |
| `enable_concurrent_access` | `bool` | Enable thread-safe configuration access |
| `validation_enabled` | `bool` | Enable comprehensive configuration validation |

### Primary Entry Point Function

#### `init_config()` Function

The main entry point for FigRegistry initialization during Kedro project startup.

```python
from figregistry_kedro.config import init_config
from kedro.config import ConfigLoader

# Initialize FigRegistry with Kedro integration
config_loader = ConfigLoader("conf")
figregistry_config = init_config(
    config_loader=config_loader,
    environment="production"
)

if figregistry_config:
    print("FigRegistry initialized successfully!")
else:
    print("FigRegistry not available")
```

**Parameters**:
- `config_loader`: Kedro ConfigLoader instance for accessing project configuration
- `environment`: Target environment for configuration resolution
- `**kwargs`: Additional configuration parameters

**Returns**: FigRegistry Config instance if successful, None if FigRegistry not available

**Raises**: 
- `ConfigurationMergeError`: If configuration merging fails
- `ImportError`: If required dependencies are not available

## Configuration Patterns

### Environment-Specific Configuration

The bridge supports Kedro's standard environment-specific configuration patterns:

```yaml
# conf/base/figregistry.yml - Base configuration
styles:
  experiment_control:
    color: "#1f77b4"
    marker: "o"
  experiment_treatment:
    color: "#ff7f0e" 
    marker: "s"

outputs:
  base_path: "data/08_reporting"
  timestamp_format: "{name}_{ts}.png"

# conf/production/figregistry.yml - Production overrides
outputs:
  base_path: "/shared/production/figures"
  timestamp_format: "{name}_prod_{ts}.pdf"
  
kedro:
  performance_mode: true
  cache_figures: true
```

### Configuration Precedence Rules

The bridge applies the following precedence rules during merging:

1. **Environment-specific Kedro configurations** override base configurations
2. **Kedro configurations** override standalone `figregistry.yaml` settings  
3. **Deep merging** for nested dictionaries
4. **Complete replacement** for lists (no merging)

```python
# Example of precedence in action
# figregistry.yaml (standalone)
styles = {
    "default": {"color": "blue", "marker": "o"}
}

# conf/base/figregistry.yml (Kedro base)
styles = {
    "default": {"color": "red"}  # Overrides color, keeps marker
}

# Result after merging
styles = {
    "default": {"color": "red", "marker": "o"}
}
```

### Kedro-Specific Extensions

The bridge supports Kedro-specific configuration extensions in the `kedro` section:

```yaml
# conf/base/figregistry.yml
kedro:
  # Performance optimization
  cache_configurations: true
  lazy_loading: true
  
  # Integration settings
  auto_register_datasets: true
  hook_priority: 100
  
  # Environment mappings
  data_layer_purposes:
    "01_raw": "exploratory"
    "02_intermediate": "exploratory" 
    "03_primary": "analysis"
    "08_reporting": "presentation"
```

## Error Handling

### ConfigurationMergeError

Raised when configuration merging or validation fails, providing detailed error information:

```python
from figregistry_kedro.config import ConfigurationMergeError

try:
    config = bridge.get_merged_config()
except ConfigurationMergeError as e:
    print(f"Configuration merge failed: {e}")
    
    # Access detailed error information
    for error in e.errors:
        print(f"Field: {error['field']}")
        print(f"Message: {error['message']}")
        print(f"Type: {error['type']}")
```

### Validation Error Handling

Pydantic validation errors are captured and converted to `ConfigurationMergeError` with structured error details:

```python
# Example validation error scenario
invalid_config = {
    "styles": "not_a_dict",  # Should be Dict[str, Dict[str, Any]]
    "outputs": {
        "base_path": 123  # Should be string or Path
    }
}

# This will raise ConfigurationMergeError with detailed field-level errors
```

## Performance Considerations

### Caching Strategy

The bridge implements intelligent caching to meet the <10ms configuration merging overhead requirement:

```python
# Configuration caching levels
class FigRegistryConfigBridge:
    # Class-level cache shared across instances
    _config_cache: Dict[str, FigRegistryKedroConfig] = {}
    
    def __init__(self, enable_caching: bool = True):
        # Instance-level cache for quick access
        self._local_cache: Optional[FigRegistryKedroConfig] = None
```

### Performance Monitoring

The bridge includes built-in performance monitoring:

```python
# Performance logging is automatic
# Configuration merging completed in 8.5ms (INFO level)
# Configuration merging took 12.3ms, exceeding 10ms target (WARNING level)
```

### Cache Key Generation

Cache keys are generated based on configuration content and environment:

```python
def _generate_cache_key(self, config_data: Dict[str, Any]) -> str:
    """Generate deterministic cache key from configuration and environment."""
    cache_content = {
        'environment': self.environment,
        'config_hash': hashlib.md5(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()
    }
    return json.dumps(cache_content, sort_keys=True)
```

## Thread Safety

### Concurrent Access Support

The bridge provides thread-safe configuration access for parallel Kedro runners:

```python
import threading
from concurrent.futures import ThreadPoolExecutor

# Thread-safe configuration access
class FigRegistryConfigBridge:
    _cache_lock = threading.RLock()  # Reentrant lock for thread safety
    
    def get_merged_config(self):
        # All cache operations are protected by locks
        with self._cache_lock:
            # Safe concurrent access
            pass
```

### Module-Level Instance Management

Thread-safe module-level instance for shared access:

```python
from figregistry_kedro.config import get_bridge_instance, set_bridge_instance

# Thread-safe module-level access
bridge = get_bridge_instance()
if bridge is None:
    # Initialize and set bridge instance
    new_bridge = FigRegistryConfigBridge(config_loader=config_loader)
    set_bridge_instance(new_bridge)
```

## Integration Examples

### Basic Integration

```python
from figregistry_kedro.config import init_config
from kedro.config import ConfigLoader

# Simple initialization
config_loader = ConfigLoader("conf")
init_config(config_loader=config_loader, environment="base")
```

### Advanced Integration with Error Handling

```python
from figregistry_kedro.config import (
    FigRegistryConfigBridge, 
    ConfigurationMergeError,
    init_config
)
from kedro.config import ConfigLoader
import logging

logger = logging.getLogger(__name__)

def setup_figregistry_integration(environment: str = "base"):
    """Set up FigRegistry integration with comprehensive error handling."""
    try:
        # Initialize ConfigLoader
        config_loader = ConfigLoader("conf")
        
        # Create bridge with caching enabled
        bridge = FigRegistryConfigBridge(
            config_loader=config_loader,
            environment=environment,
            enable_caching=True
        )
        
        # Get merged configuration
        merged_config = bridge.get_merged_config()
        logger.info(f"Configuration loaded for environment: {environment}")
        logger.info(f"Available styles: {len(merged_config.styles)}")
        
        # Initialize FigRegistry
        figregistry_config = init_config(
            config_loader=config_loader,
            environment=environment
        )
        
        return figregistry_config
        
    except ConfigurationMergeError as e:
        logger.error(f"Configuration merge failed: {e}")
        
        # Log detailed errors
        for error in e.errors:
            logger.error(f"  {error['field']}: {error['message']}")
        
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during FigRegistry setup: {e}")
        raise

# Usage in Kedro hooks
class FigRegistryHooks:
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """Initialize FigRegistry before pipeline execution."""
        environment = run_params.get('env', 'base')
        setup_figregistry_integration(environment)
```

### Configuration Validation Example

```python
from figregistry_kedro.config import FigRegistryKedroConfig
from pydantic import ValidationError

def validate_figregistry_config(config_dict: dict) -> bool:
    """Validate configuration dictionary against Pydantic schema."""
    try:
        validated_config = FigRegistryKedroConfig(**config_dict)
        print("Configuration validation successful!")
        
        # Access validated sections
        print(f"Styles defined: {list(validated_config.styles.keys())}")
        print(f"Environment: {validated_config.environment}")
        print(f"Concurrent access: {validated_config.enable_concurrent_access}")
        
        return True
        
    except ValidationError as e:
        print("Configuration validation failed:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  {field}: {error['msg']}")
        
        return False

# Example usage
config_data = {
    "styles": {
        "control": {"color": "#1f77b4", "marker": "o"},
        "treatment": {"color": "#ff7f0e", "marker": "s"}
    },
    "outputs": {
        "base_path": "data/08_reporting",
        "timestamp_format": "{name}_{ts}.png"
    },
    "environment": "production"
}

is_valid = validate_figregistry_config(config_data)
```

## Best Practices

### 1. Environment-Specific Configuration

Organize configurations using Kedro's environment structure:

```
conf/
├── base/
│   ├── figregistry.yml          # Base configuration
│   └── catalog.yml
├── local/
│   ├── figregistry.yml          # Local development overrides
│   └── catalog.yml
└── production/
    ├── figregistry.yml          # Production-specific settings
    └── catalog.yml
```

### 2. Performance Optimization

Enable caching for production environments:

```python
# Production setup
bridge = FigRegistryConfigBridge(
    config_loader=config_loader,
    environment="production",
    enable_caching=True  # Important for performance
)
```

### 3. Error Handling

Always handle configuration errors gracefully:

```python
try:
    config = bridge.get_merged_config()
except ConfigurationMergeError as e:
    # Log errors and provide fallback
    logger.error(f"Configuration failed: {e}")
    # Use minimal safe defaults
    config = fallback_configuration()
```

### 4. Thread Safety

For parallel execution, ensure proper bridge instance management:

```python
# Thread-safe singleton pattern
_bridge_instance = None
_instance_lock = threading.Lock()

def get_configured_bridge(config_loader, environment):
    global _bridge_instance
    
    with _instance_lock:
        if _bridge_instance is None:
            _bridge_instance = FigRegistryConfigBridge(
                config_loader=config_loader,
                environment=environment
            )
    
    return _bridge_instance
```

### 5. Configuration Validation

Validate configurations during development:

```python
# Development validation
if environment == "local":
    config = bridge.get_merged_config()
    
    # Validate required sections
    assert config.styles, "Styles section is required"
    assert config.outputs, "Outputs section is required"
    
    # Validate performance settings
    assert config.enable_concurrent_access, "Concurrent access should be enabled"
```

## Migration from Standalone FigRegistry

When migrating from standalone FigRegistry usage to the Kedro integration:

### 1. Move Configuration Files

```bash
# Before: standalone figregistry.yaml
figregistry.yaml

# After: Kedro structure
conf/base/figregistry.yml
```

### 2. Update Configuration Access

```python
# Before: Direct FigRegistry usage
import figregistry
figregistry.init_config()

# After: Kedro integration
from figregistry_kedro.config import init_config
from kedro.config import ConfigLoader

config_loader = ConfigLoader("conf")
init_config(config_loader=config_loader)
```

### 3. Environment-Specific Settings

```yaml
# Before: Single figregistry.yaml
outputs:
  base_path: "figures"

# After: Environment-specific
# conf/base/figregistry.yml
outputs:
  base_path: "data/08_reporting"

# conf/production/figregistry.yml  
outputs:
  base_path: "/shared/production/figures"
```

This comprehensive API reference provides everything needed to effectively use the FigRegistryConfigBridge for seamless configuration management between Kedro and FigRegistry systems.