"""
FigRegistry-Kedro Configuration Bridge

This module implements the FigRegistryConfigBridge that serves as the configuration
translation layer between Kedro's ConfigLoader system and FigRegistry's YAML-based
configuration management. The bridge enables seamless merging of environment-specific
Kedro parameters with traditional figregistry.yaml settings while maintaining
validation and type safety across both frameworks.

Key Features:
- Configuration merging with clear precedence rules (F-007)
- Pydantic validation for type safety across configuration systems
- Performance-optimized operation (<10ms merge time target)
- Thread-safe concurrent access for parallel Kedro runners
- Comprehensive error aggregation for configuration validation failures
- Environment-specific configuration support for multi-stage deployments

The component operates as a read-only translator that respects configuration
hierarchies from both systems while providing clear precedence rules for
conflict resolution.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import copy

from pydantic import BaseModel, Field, ValidationError, validator
import yaml

# Optional Kedro imports with graceful fallback
try:
    from kedro.config import ConfigLoader
    from kedro.framework.context import KedroContext
    HAS_KEDRO = True
except ImportError:
    HAS_KEDRO = False
    ConfigLoader = None
    KedroContext = None

# FigRegistry imports with graceful fallback  
try:
    import figregistry
    from figregistry import Config as FigRegistryConfig
    HAS_FIGREGISTRY = True
except ImportError:
    HAS_FIGREGISTRY = False
    figregistry = None
    FigRegistryConfig = None

# Configure module logger
logger = logging.getLogger(__name__)

# Module-level configuration cache with thread safety
_config_cache: Dict[str, Any] = {}
_cache_lock = Lock()

# Performance tracking for monitoring configuration operation times
_performance_metrics = {
    "merge_times": [],
    "cache_hits": 0,
    "cache_misses": 0,
    "validation_failures": 0
}


class ConfigMergeError(Exception):
    """Raised when configuration merging fails with validation or processing errors."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class ConfigValidationError(Exception):
    """Raised when configuration validation fails with detailed error information."""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or []


class FigRegistryConfigSchema(BaseModel):
    """
    Pydantic schema for validating merged FigRegistry configurations.
    
    This schema ensures type safety and validation for configurations that merge
    Kedro ConfigLoader settings with traditional FigRegistry YAML configurations.
    It provides validation for all required and optional configuration sections
    while maintaining compatibility with both systems' configuration structures.
    """
    
    # Core FigRegistry configuration sections
    figregistry_version: Optional[str] = Field(
        default=">=0.3.0",
        description="FigRegistry version constraint for compatibility validation"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration metadata for tracking and validation"
    )
    
    styles: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Condition-based style mappings for experimental visualizations"
    )
    
    palettes: Optional[Dict[str, Union[List[str], Dict[str, str]]]] = Field(
        default_factory=dict,
        description="Color palettes and fallback styling for undefined conditions"
    )
    
    defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default styling parameters and fallback configurations"
    )
    
    outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output management configuration for automated file handling"
    )
    
    # Kedro-specific integration settings
    kedro: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Kedro-specific configuration extensions and integration settings"
    )
    
    # Advanced configuration sections
    style_inheritance: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Style inheritance and composition rules"
    )
    
    conditional_rules: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Conditional styling rules for complex scenarios"
    )
    
    performance: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance monitoring and optimization settings"
    )
    
    validation: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration validation schema and rules"
    )
    
    examples: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Usage examples and documentation"
    )
    
    @validator('styles')
    def validate_styles(cls, v):
        """Validate that each style contains at minimum a color specification."""
        for style_name, style_config in v.items():
            if not isinstance(style_config, dict):
                raise ValueError(f"Style '{style_name}' must be a dictionary")
            if 'color' not in style_config:
                logger.warning(f"Style '{style_name}' missing required 'color' field")
        return v
    
    @validator('outputs')
    def validate_outputs(cls, v):
        """Validate output configuration has required base_path."""
        if v and 'base_path' not in v:
            raise ValueError("Output configuration must include 'base_path' field")
        return v
    
    @validator('figregistry_version')
    def validate_version_constraint(cls, v):
        """Validate version constraint format."""
        if v and not any(op in v for op in ['>=', '>', '<=', '<', '==']):
            raise ValueError(f"Invalid version constraint format: {v}")
        return v
    
    class Config:
        """Pydantic model configuration for enhanced validation and performance."""
        extra = "allow"  # Allow additional fields for extensibility
        validate_assignment = True  # Validate on field assignment
        arbitrary_types_allowed = True  # Allow complex types
        use_enum_values = True  # Use enum values for serialization


class FigRegistryConfigBridge:
    """
    Configuration translation layer between Kedro's ConfigLoader and FigRegistry's
    YAML-based configuration management system.
    
    This bridge enables seamless integration of environment-specific Kedro parameters
    with FigRegistry's condition-based styling system while maintaining configuration
    validation and type safety across both frameworks. The component operates as a
    read-only translator that respects configuration hierarchies from both systems
    while providing clear precedence rules for conflict resolution.
    
    Key Features:
    - Merges Kedro ConfigLoader with FigRegistry YAML configurations (F-007)
    - Provides environment-specific configuration support (F-007.2)
    - Maintains <10ms configuration merging overhead (Section 5.2.8)
    - Supports concurrent access for parallel Kedro execution
    - Comprehensive error aggregation for validation failures
    - Pydantic validation for type safety across both systems
    
    Usage:
        bridge = FigRegistryConfigBridge()
        config = bridge.merge_configurations(config_loader, environment="local")
        figregistry.init_config(config)
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        validation_strict: bool = True,
        performance_target_ms: float = 10.0,
        max_cache_size: int = 1000
    ):
        """
        Initialize the FigRegistry configuration bridge.
        
        Args:
            cache_enabled: Enable configuration caching for performance
            validation_strict: Enable strict validation for merged configurations
            performance_target_ms: Target merge time in milliseconds (<10ms default)
            max_cache_size: Maximum cache entries for merged configurations
        """
        self.cache_enabled = cache_enabled
        self.validation_strict = validation_strict
        self.performance_target_ms = performance_target_ms
        self.max_cache_size = max_cache_size
        
        # Thread safety for concurrent access
        self._lock = Lock()
        
        # Performance monitoring
        self._merge_times: List[float] = []
        self._cache_stats = {"hits": 0, "misses": 0}
        
        logger.info(
            f"Initialized FigRegistryConfigBridge with cache={cache_enabled}, "
            f"strict_validation={validation_strict}, "
            f"target_time={performance_target_ms}ms"
        )
    
    def merge_configurations(
        self,
        config_loader: Optional[Any] = None,
        environment: str = "base",
        project_path: Optional[Path] = None,
        figregistry_config_name: str = "figregistry",
        **override_params
    ) -> Dict[str, Any]:
        """
        Merge Kedro ConfigLoader configurations with FigRegistry YAML settings.
        
        This method implements the core configuration bridge functionality by loading
        configurations from both Kedro's ConfigLoader system and traditional
        figregistry.yaml files, then merging them with clear precedence rules
        while maintaining validation and type safety.
        
        Args:
            config_loader: Kedro ConfigLoader instance for loading project configurations
            environment: Environment name for configuration resolution (local, staging, production)
            project_path: Project root path for configuration file discovery
            figregistry_config_name: Name of FigRegistry config file (without .yml extension)
            **override_params: Additional parameters to override in merged configuration
        
        Returns:
            Dict containing merged and validated configuration ready for FigRegistry initialization
        
        Raises:
            ConfigMergeError: When configuration merging fails
            ConfigValidationError: When validation of merged configuration fails
        """
        start_time = time.time()
        
        try:
            # Generate cache key for configuration lookup
            cache_key = self._generate_cache_key(
                config_loader, environment, project_path, figregistry_config_name, **override_params
            )
            
            # Check cache first for performance optimization
            if self.cache_enabled:
                cached_config = self._get_cached_config(cache_key)
                if cached_config is not None:
                    self._cache_stats["hits"] += 1
                    logger.debug(f"Configuration cache hit for key: {cache_key}")
                    return cached_config
                self._cache_stats["misses"] += 1
            
            logger.info(f"Merging configurations for environment: {environment}")
            
            # Load configurations from both systems
            kedro_config = self._load_kedro_config(config_loader, environment)
            figregistry_config = self._load_figregistry_config(
                project_path, figregistry_config_name, environment
            )
            
            # Merge configurations with precedence rules
            merged_config = self._merge_config_sections(
                kedro_config, figregistry_config, **override_params
            )
            
            # Validate merged configuration
            if self.validation_strict:
                validated_config = self._validate_configuration(merged_config)
            else:
                validated_config = merged_config
            
            # Cache merged configuration for future use
            if self.cache_enabled:
                self._cache_config(cache_key, validated_config)
            
            # Track performance metrics
            merge_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self._merge_times.append(merge_time)
            
            if merge_time > self.performance_target_ms:
                logger.warning(
                    f"Configuration merge time {merge_time:.2f}ms exceeds target "
                    f"{self.performance_target_ms}ms"
                )
            else:
                logger.debug(f"Configuration merge completed in {merge_time:.2f}ms")
            
            return validated_config
            
        except Exception as e:
            merge_time = (time.time() - start_time) * 1000
            logger.error(f"Configuration merge failed after {merge_time:.2f}ms: {e}")
            raise ConfigMergeError(f"Failed to merge configurations: {e}") from e
    
    def _load_kedro_config(
        self, 
        config_loader: Optional[Any], 
        environment: str
    ) -> Dict[str, Any]:
        """
        Load configuration from Kedro's ConfigLoader system.
        
        Args:
            config_loader: Kedro ConfigLoader instance
            environment: Environment name for configuration resolution
        
        Returns:
            Dictionary containing Kedro configuration sections
        """
        kedro_config = {}
        
        if config_loader is None:
            logger.debug("No Kedro ConfigLoader provided, using empty configuration")
            return kedro_config
        
        if not HAS_KEDRO:
            logger.warning("Kedro not available, skipping Kedro configuration loading")
            return kedro_config
        
        try:
            # Load all configuration sections from Kedro
            config_sections = ["parameters", "figregistry", "catalog", "logging"]
            
            for section in config_sections:
                try:
                    section_config = config_loader.get(section, environment)
                    if section_config:
                        kedro_config[section] = section_config
                        logger.debug(f"Loaded Kedro config section '{section}' for environment '{environment}'")
                except Exception as e:
                    logger.debug(f"Kedro config section '{section}' not found or failed to load: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(kedro_config)} Kedro configuration sections")
            
        except Exception as e:
            logger.error(f"Failed to load Kedro configuration: {e}")
            # Continue with empty configuration rather than failing completely
        
        return kedro_config
    
    def _load_figregistry_config(
        self,
        project_path: Optional[Path],
        config_name: str,
        environment: str
    ) -> Dict[str, Any]:
        """
        Load configuration from FigRegistry YAML files.
        
        Args:
            project_path: Project root path for configuration file discovery
            config_name: Name of FigRegistry config file (without .yml extension)
            environment: Environment name for configuration resolution
        
        Returns:
            Dictionary containing FigRegistry configuration
        """
        figregistry_config = {}
        
        # Define potential configuration file paths
        config_paths = self._get_figregistry_config_paths(project_path, config_name, environment)
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f) or {}
                    
                    # Merge with existing configuration (later files override earlier)
                    figregistry_config = self._deep_merge_dicts(figregistry_config, file_config)
                    logger.debug(f"Loaded FigRegistry config from: {config_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load FigRegistry config from {config_path}: {e}")
                    continue
        
        if not figregistry_config:
            logger.warning("No FigRegistry configuration files found, using defaults")
            figregistry_config = self._get_default_figregistry_config()
        
        return figregistry_config
    
    def _get_figregistry_config_paths(
        self,
        project_path: Optional[Path],
        config_name: str,
        environment: str
    ) -> List[Path]:
        """
        Generate list of potential FigRegistry configuration file paths.
        
        Args:
            project_path: Project root path
            config_name: Configuration file name (without extension)
            environment: Environment name
        
        Returns:
            List of Path objects for configuration file locations
        """
        paths = []
        
        # Set default project path if not provided
        if project_path is None:
            project_path = Path.cwd()
        
        # Traditional FigRegistry config in project root
        paths.append(project_path / f"{config_name}.yaml")
        paths.append(project_path / f"{config_name}.yml")
        
        # Kedro-style configuration paths
        conf_base = project_path / "conf" / "base"
        if conf_base.exists():
            paths.append(conf_base / f"{config_name}.yml")
            paths.append(conf_base / f"{config_name}.yaml")
        
        # Environment-specific configuration paths
        if environment != "base":
            conf_env = project_path / "conf" / environment
            if conf_env.exists():
                paths.append(conf_env / f"{config_name}.yml")
                paths.append(conf_env / f"{config_name}.yaml")
        
        return paths
    
    def _merge_config_sections(
        self,
        kedro_config: Dict[str, Any],
        figregistry_config: Dict[str, Any],
        **override_params
    ) -> Dict[str, Any]:
        """
        Merge configuration sections with precedence rules.
        
        Precedence order (highest to lowest):
        1. Override parameters passed directly to method
        2. Kedro configuration values (environment-specific)
        3. FigRegistry configuration values
        4. Default values
        
        Args:
            kedro_config: Configuration from Kedro ConfigLoader
            figregistry_config: Configuration from FigRegistry YAML files
            **override_params: Direct override parameters
        
        Returns:
            Merged configuration dictionary
        """
        # Start with FigRegistry configuration as base
        merged_config = copy.deepcopy(figregistry_config)
        
        # Merge Kedro parameters into styles and other relevant sections
        if "parameters" in kedro_config:
            parameters = kedro_config["parameters"]
            
            # Add experimental condition parameters to configuration context
            merged_config.setdefault("_kedro_context", {})
            merged_config["_kedro_context"]["parameters"] = parameters
            
            # Merge parameters into appropriate configuration sections
            self._merge_parameters_into_config(merged_config, parameters)
        
        # Merge Kedro figregistry configuration if present
        if "figregistry" in kedro_config:
            kedro_figregistry = kedro_config["figregistry"]
            merged_config = self._deep_merge_dicts(merged_config, kedro_figregistry)
        
        # Apply direct override parameters
        if override_params:
            merged_config = self._deep_merge_dicts(merged_config, override_params)
        
        # Ensure required sections exist with defaults
        self._ensure_required_sections(merged_config)
        
        return merged_config
    
    def _merge_parameters_into_config(
        self,
        config: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> None:
        """
        Merge Kedro parameters into appropriate configuration sections.
        
        Args:
            config: Configuration dictionary to modify in-place
            parameters: Kedro parameters to merge
        """
        # Extract experimental condition parameters for style resolution
        condition_params = {
            "experiment_condition": parameters.get("experiment_condition"),
            "experiment_phase": parameters.get("experiment_phase"),
            "analysis_stage": parameters.get("analysis_stage"),
            "model_type": parameters.get("model_type")
        }
        
        # Remove None values
        condition_params = {k: v for k, v in condition_params.items() if v is not None}
        
        if condition_params:
            config.setdefault("condition_parameters", {}).update(condition_params)
        
        # Merge visualization parameters if present
        if "plot_settings" in parameters:
            plot_settings = parameters["plot_settings"]
            config.setdefault("defaults", {}).setdefault("figure", {}).update({
                "figsize": plot_settings.get("figure_size", [10, 8]),
                "dpi": plot_settings.get("dpi", 150)
            })
        
        # Merge output configuration from execution parameters
        if "execution_config" in parameters:
            exec_config = parameters["execution_config"]
            if "output_base_path" in exec_config:
                config.setdefault("outputs", {})["base_path"] = exec_config["output_base_path"]
            if "figure_formats" in exec_config:
                config.setdefault("outputs", {}).setdefault("formats", {})["defaults"] = {
                    "exploratory": exec_config["figure_formats"]
                }
    
    def _deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries with dict2 values taking precedence.
        
        Args:
            dict1: Base dictionary
            dict2: Override dictionary (takes precedence)
        
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _ensure_required_sections(self, config: Dict[str, Any]) -> None:
        """
        Ensure required configuration sections exist with appropriate defaults.
        
        Args:
            config: Configuration dictionary to modify in-place
        """
        # Required sections with defaults
        required_sections = {
            "styles": {},
            "defaults": {
                "figure": {"figsize": [10, 8], "dpi": 150},
                "line": {"color": "#2E86AB", "linewidth": 2.0},
                "fallback_style": {
                    "color": "#95A5A6",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Unknown Condition"
                }
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "naming": {"template": "{name}_{condition}_{ts}"}
            },
            "kedro": {
                "config_bridge": {"enabled": True, "merge_strategy": "override"},
                "datasets": {"default_purpose": "exploratory"}
            }
        }
        
        for section, default_values in required_sections.items():
            if section not in config:
                config[section] = copy.deepcopy(default_values)
            elif isinstance(default_values, dict):
                for key, value in default_values.items():
                    if key not in config[section]:
                        config[section][key] = copy.deepcopy(value)
    
    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate merged configuration using Pydantic schema.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Validated configuration dictionary
        
        Raises:
            ConfigValidationError: When validation fails
        """
        try:
            # Validate using Pydantic schema
            validated = FigRegistryConfigSchema(**config)
            
            # Convert back to dictionary for FigRegistry compatibility
            validated_dict = validated.dict()
            
            logger.debug("Configuration validation passed")
            return validated_dict
            
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                message = error["msg"]
                error_messages.append(f"{field}: {message}")
            
            logger.error(f"Configuration validation failed with {len(error_messages)} errors")
            raise ConfigValidationError(
                "Configuration validation failed",
                validation_errors=error_messages
            ) from e
    
    def _get_default_figregistry_config(self) -> Dict[str, Any]:
        """
        Generate default FigRegistry configuration when no files are found.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "metadata": {
                "config_version": "1.0.0",
                "created_by": "figregistry-kedro config bridge",
                "description": "Default configuration for Kedro integration"
            },
            "styles": {
                "exploratory": {
                    "color": "#A8E6CF",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Exploratory"
                },
                "presentation": {
                    "color": "#FFB6C1",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 2.0,
                    "alpha": 0.8,
                    "label": "Presentation"
                },
                "publication": {
                    "color": "#1A1A1A",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 2.5,
                    "alpha": 1.0,
                    "label": "Publication"
                }
            },
            "defaults": {
                "figure": {"figsize": [10, 8], "dpi": 150},
                "line": {"color": "#2E86AB", "linewidth": 2.0},
                "fallback_style": {
                    "color": "#95A5A6",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Unknown Condition"
                }
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "naming": {"template": "{name}_{condition}_{ts}"}
            }
        }
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key for configuration lookup.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            String cache key
        """
        # Create hashable representation of arguments
        key_parts = []
        
        for arg in args:
            if hasattr(arg, '__class__'):
                key_parts.append(f"{arg.__class__.__name__}")
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return "_".join(key_parts)
    
    def _get_cached_config(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve configuration from cache if available.
        
        Args:
            cache_key: Cache key for lookup
        
        Returns:
            Cached configuration or None if not found
        """
        with self._lock:
            return _config_cache.get(cache_key)
    
    def _cache_config(self, cache_key: str, config: Dict[str, Any]) -> None:
        """
        Cache configuration for future retrieval.
        
        Args:
            cache_key: Key for cache storage
            config: Configuration to cache
        """
        with self._lock:
            # Implement cache size limit
            if len(_config_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO eviction)
                oldest_key = next(iter(_config_cache))
                del _config_cache[oldest_key]
            
            _config_cache[cache_key] = copy.deepcopy(config)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for configuration bridge operations.
        
        Returns:
            Dictionary containing performance statistics
        """
        with self._lock:
            return {
                "merge_times": {
                    "count": len(self._merge_times),
                    "average_ms": sum(self._merge_times) / len(self._merge_times) if self._merge_times else 0,
                    "max_ms": max(self._merge_times) if self._merge_times else 0,
                    "min_ms": min(self._merge_times) if self._merge_times else 0,
                    "target_ms": self.performance_target_ms
                },
                "cache_stats": self._cache_stats.copy(),
                "cache_size": len(_config_cache),
                "max_cache_size": self.max_cache_size
            }
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        with self._lock:
            _config_cache.clear()
            self._cache_stats = {"hits": 0, "misses": 0}
        logger.info("Configuration cache cleared")


def init_config(
    config_loader: Optional[Any] = None,
    environment: str = "base",
    project_path: Optional[Path] = None,
    **kwargs
) -> Optional[Any]:
    """
    Initialize FigRegistry configuration through the configuration bridge.
    
    This function provides the primary interface for initializing FigRegistry
    with merged Kedro and FigRegistry configurations during pipeline startup.
    It creates a FigRegistryConfigBridge instance, merges configurations from
    both systems, and initializes FigRegistry with the resulting configuration.
    
    Args:
        config_loader: Kedro ConfigLoader instance for loading project configurations
        environment: Environment name for configuration resolution (default: "base")
        project_path: Project root path for configuration file discovery
        **kwargs: Additional parameters to pass to configuration bridge
    
    Returns:
        FigRegistry configuration object if successful, None otherwise
    
    Raises:
        ConfigMergeError: When configuration merging fails
        ConfigValidationError: When validation of merged configuration fails
    
    Usage:
        # Basic usage in Kedro hooks
        from figregistry_kedro.config import init_config
        
        config = init_config(config_loader, environment="local")
        
        # Advanced usage with overrides
        config = init_config(
            config_loader,
            environment="production",
            project_path=Path("/path/to/project"),
            validation_strict=True
        )
    """
    if not HAS_FIGREGISTRY:
        logger.error("FigRegistry not available - cannot initialize configuration")
        return None
    
    try:
        # Create configuration bridge
        bridge = FigRegistryConfigBridge(**kwargs)
        
        # Merge configurations from both systems
        merged_config = bridge.merge_configurations(
            config_loader=config_loader,
            environment=environment,
            project_path=project_path
        )
        
        logger.info(f"Initializing FigRegistry with merged configuration for environment: {environment}")
        
        # Initialize FigRegistry with merged configuration
        # Note: This assumes FigRegistry has an init_config function that accepts a dict
        if hasattr(figregistry, 'init_config'):
            return figregistry.init_config(merged_config)
        elif hasattr(figregistry, 'Config'):
            # Fallback to Config class initialization
            return figregistry.Config(**merged_config)
        else:
            logger.warning("FigRegistry init_config method not found - returning raw configuration")
            return merged_config
    
    except Exception as e:
        logger.error(f"Failed to initialize FigRegistry configuration: {e}")
        # Re-raise for proper error handling by calling code
        raise


def get_merged_config(
    config_loader: Optional[Any] = None,
    environment: str = "base",
    project_path: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get merged configuration without initializing FigRegistry.
    
    This function provides access to the merged configuration dictionary
    for inspection or manual initialization purposes. It performs the same
    configuration merging as init_config() but returns the raw dictionary
    instead of initializing FigRegistry.
    
    Args:
        config_loader: Kedro ConfigLoader instance
        environment: Environment name for configuration resolution
        project_path: Project root path for configuration file discovery
        **kwargs: Additional parameters for configuration bridge
    
    Returns:
        Merged configuration dictionary
    
    Usage:
        # Get merged configuration for inspection
        config = get_merged_config(config_loader, environment="staging")
        print(f"Available styles: {list(config['styles'].keys())}")
    """
    bridge = FigRegistryConfigBridge(**kwargs)
    return bridge.merge_configurations(
        config_loader=config_loader,
        environment=environment,
        project_path=project_path
    )


# Export public API
__all__ = [
    "FigRegistryConfigBridge",
    "FigRegistryConfigSchema", 
    "ConfigMergeError",
    "ConfigValidationError",
    "init_config",
    "get_merged_config"
]