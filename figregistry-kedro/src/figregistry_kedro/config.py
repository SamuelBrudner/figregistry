"""FigRegistry-Kedro Configuration Bridge.

This module provides the FigRegistryConfigBridge that serves as the configuration 
translation layer between Kedro's ConfigLoader system and FigRegistry's YAML-based 
configuration management. The bridge enables seamless merging of environment-specific 
Kedro parameters with traditional figregistry.yaml settings while maintaining 
validation and type safety across both frameworks.

The component operates as a read-only translator that respects configuration 
hierarchies from both systems while providing clear precedence rules for 
conflict resolution.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

try:
    import figregistry
    from figregistry.core.config import Config as FigRegistryConfig, load_config
except ImportError:
    warnings.warn(
        "FigRegistry not found. Please ensure figregistry>=0.3.0 is installed.",
        ImportWarning
    )
    figregistry = None
    FigRegistryConfig = None
    load_config = None

try:
    from kedro.config import ConfigLoader
except ImportError:
    warnings.warn(
        "Kedro not found. Please ensure kedro>=0.18.0,<0.20.0 is installed.",
        ImportWarning
    )
    ConfigLoader = None

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic.types import StrictBool, StrictStr

logger = logging.getLogger(__name__)

class FigRegistryKedroConfig(BaseModel):
    """Pydantic model for merged FigRegistry-Kedro configuration.
    
    This model ensures type safety across both configuration systems and provides
    comprehensive validation for merged configuration structures.
    """
    
    # Core FigRegistry configuration sections
    styles: Optional[Dict[str, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Condition-based style mappings for experimental visualizations"
    )
    
    palettes: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Color palette definitions and defaults"
    )
    
    outputs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Output path configurations and naming conventions"
    )
    
    defaults: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Default styling parameters and fallback values"
    )
    
    # Kedro-specific integration settings
    kedro: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Kedro-specific configuration overrides and extensions"
    )
    
    # Environment and precedence settings
    environment: Optional[StrictStr] = Field(
        default="base",
        description="Current environment for configuration resolution"
    )
    
    enable_concurrent_access: Optional[StrictBool] = Field(
        default=True,
        description="Enable thread-safe configuration access for parallel runners"
    )
    
    validation_enabled: Optional[StrictBool] = Field(
        default=True,
        description="Enable comprehensive configuration validation"
    )
    
    @validator('styles')
    def validate_styles(cls, v):
        """Validate style mappings structure."""
        if not isinstance(v, dict):
            raise ValueError("Styles must be a dictionary")
        
        for condition, style_dict in v.items():
            if not isinstance(style_dict, dict):
                raise ValueError(f"Style definition for '{condition}' must be a dictionary")
        
        return v
    
    @validator('outputs')
    def validate_outputs(cls, v):
        """Validate output configuration structure."""
        if not isinstance(v, dict):
            raise ValueError("Outputs must be a dictionary")
            
        # Validate common output configuration keys if present
        if 'base_path' in v and not isinstance(v['base_path'], (str, Path)):
            raise ValueError("Output base_path must be a string or Path")
            
        return v
    
    class Config:
        """Pydantic configuration settings."""
        extra = "allow"  # Allow additional fields for flexibility
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Use enum values in validation


class ConfigurationMergeError(Exception):
    """Exception raised when configuration merging fails."""
    
    def __init__(self, message: str, errors: Optional[list] = None):
        super().__init__(message)
        self.errors = errors or []


class FigRegistryConfigBridge:
    """Configuration bridge between Kedro ConfigLoader and FigRegistry.
    
    This bridge enables seamless integration of environment-specific Kedro 
    parameters with FigRegistry's condition-based styling system while 
    maintaining configuration validation and type safety across both frameworks.
    
    Features:
    - Seamless merging of Kedro and FigRegistry configurations
    - Environment-specific configuration overrides
    - Pydantic validation for type safety
    - Concurrent access support for parallel execution
    - <10ms configuration merging overhead
    - Comprehensive error aggregation and reporting
    """
    
    # Class-level configuration cache for performance
    _config_cache: Dict[str, FigRegistryKedroConfig] = {}
    _cache_lock = threading.RLock()
    
    def __init__(self, 
                 config_loader: Optional['ConfigLoader'] = None,
                 environment: Optional[str] = None,
                 enable_caching: bool = True):
        """Initialize the configuration bridge.
        
        Args:
            config_loader: Kedro ConfigLoader instance for accessing project configuration
            environment: Target environment for configuration resolution (e.g., 'local', 'production')
            enable_caching: Enable configuration caching for performance optimization
        """
        self.config_loader = config_loader
        self.environment = environment or "base"
        self.enable_caching = enable_caching
        self._local_cache: Optional[FigRegistryKedroConfig] = None
        self._cache_key: Optional[str] = None
        
        logger.debug(f"Initialized FigRegistryConfigBridge for environment: {self.environment}")
    
    def _generate_cache_key(self, config_data: Dict[str, Any]) -> str:
        """Generate a cache key based on configuration content and environment."""
        import hashlib
        import json
        
        # Create a deterministic representation of the configuration
        cache_content = {
            'environment': self.environment,
            'config_hash': hashlib.md5(
                json.dumps(config_data, sort_keys=True).encode()
            ).hexdigest()
        }
        
        return json.dumps(cache_content, sort_keys=True)
    
    def _load_figregistry_config(self) -> Dict[str, Any]:
        """Load traditional FigRegistry configuration from figregistry.yaml."""
        config_data = {}
        
        # Try to load standalone figregistry.yaml first
        figregistry_path = Path("figregistry.yaml")
        if figregistry_path.exists():
            try:
                with open(figregistry_path, 'r') as f:
                    standalone_config = yaml.safe_load(f) or {}
                config_data.update(standalone_config)
                logger.debug(f"Loaded standalone figregistry.yaml with {len(standalone_config)} sections")
            except Exception as e:
                logger.warning(f"Failed to load standalone figregistry.yaml: {e}")
        
        return config_data
    
    def _load_kedro_figregistry_config(self) -> Dict[str, Any]:
        """Load FigRegistry configuration from Kedro's configuration system."""
        if self.config_loader is None:
            return {}
            
        config_data = {}
        
        try:
            # Try to load figregistry configuration through Kedro's ConfigLoader
            # Support multiple naming patterns for flexibility
            patterns = [
                "figregistry",
                "figregistry*",
                "**/figregistry.yml",
                "**/figregistry.yaml"
            ]
            
            for pattern in patterns:
                try:
                    kedro_config = self.config_loader.get(pattern)
                    if kedro_config:
                        config_data.update(kedro_config)
                        logger.debug(f"Loaded Kedro FigRegistry config with pattern '{pattern}': {len(kedro_config)} sections")
                        break
                except Exception as e:
                    logger.debug(f"Pattern '{pattern}' not found or failed: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to load FigRegistry config from Kedro: {e}")
        
        return config_data
    
    def _merge_configurations(self, 
                            figregistry_config: Dict[str, Any], 
                            kedro_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge FigRegistry and Kedro configurations with proper precedence rules.
        
        Precedence Rules:
        1. Environment-specific Kedro configurations override base configurations
        2. Kedro configurations override standalone figregistry.yaml settings
        3. Deep merging for nested dictionaries
        4. Lists are replaced entirely (no merging)
        
        Args:
            figregistry_config: Configuration from figregistry.yaml
            kedro_config: Configuration from Kedro ConfigLoader
            
        Returns:
            Merged configuration dictionary
        """
        start_time = time.perf_counter()
        
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Deep merge two dictionaries with override precedence."""
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    result[key] = deep_merge(result[key], value)
                else:
                    # Override or add new key
                    result[key] = value
            
            return result
        
        # Start with FigRegistry base configuration
        merged_config = figregistry_config.copy()
        
        # Merge Kedro configuration with precedence
        merged_config = deep_merge(merged_config, kedro_config)
        
        # Add environment metadata
        merged_config['environment'] = self.environment
        
        # Add Kedro-specific settings if not present
        if 'kedro' not in merged_config:
            merged_config['kedro'] = {}
            
        merge_time = time.perf_counter() - start_time
        
        # Ensure performance requirement of <10ms
        if merge_time > 0.01:  # 10ms
            logger.warning(f"Configuration merging took {merge_time*1000:.2f}ms, exceeding 10ms target")
        else:
            logger.debug(f"Configuration merging completed in {merge_time*1000:.2f}ms")
        
        return merged_config
    
    def _validate_configuration(self, config_data: Dict[str, Any]) -> FigRegistryKedroConfig:
        """Validate merged configuration using Pydantic model.
        
        Args:
            config_data: Merged configuration dictionary
            
        Returns:
            Validated FigRegistryKedroConfig instance
            
        Raises:
            ConfigurationMergeError: If validation fails
        """
        try:
            validated_config = FigRegistryKedroConfig(**config_data)
            logger.debug("Configuration validation successful")
            return validated_config
            
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append({
                    'field': '.'.join(str(loc) for loc in error['loc']),
                    'message': error['msg'],
                    'type': error['type']
                })
            
            error_message = f"Configuration validation failed with {len(error_details)} errors"
            logger.error(f"{error_message}: {error_details}")
            
            raise ConfigurationMergeError(error_message, error_details)
    
    def get_merged_config(self) -> FigRegistryKedroConfig:
        """Get merged and validated FigRegistry-Kedro configuration.
        
        This method implements the core configuration bridge functionality,
        merging traditional figregistry.yaml settings with Kedro's environment-specific
        configurations while maintaining validation and caching for performance.
        
        Returns:
            Validated FigRegistryKedroConfig instance
            
        Raises:
            ConfigurationMergeError: If configuration merging or validation fails
        """
        start_time = time.perf_counter()
        
        try:
            # Load configurations from both sources
            figregistry_config = self._load_figregistry_config()
            kedro_config = self._load_kedro_figregistry_config()
            
            # Merge configurations with precedence rules
            merged_config = self._merge_configurations(figregistry_config, kedro_config)
            
            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(merged_config)
                
                with self._cache_lock:
                    if cache_key in self._config_cache:
                        logger.debug("Using cached configuration")
                        self._local_cache = self._config_cache[cache_key]
                        self._cache_key = cache_key
                        return self._local_cache
            
            # Validate merged configuration
            validated_config = self._validate_configuration(merged_config)
            
            # Cache validated configuration if enabled
            if self.enable_caching:
                with self._cache_lock:
                    self._config_cache[cache_key] = validated_config
                    self._local_cache = validated_config
                    self._cache_key = cache_key
            
            total_time = time.perf_counter() - start_time
            logger.debug(f"Configuration bridge completed in {total_time*1000:.2f}ms")
            
            return validated_config
            
        except Exception as e:
            if isinstance(e, ConfigurationMergeError):
                raise
            
            error_message = f"Configuration bridge failed: {str(e)}"
            logger.error(error_message)
            raise ConfigurationMergeError(error_message)
    
    def clear_cache(self):
        """Clear configuration cache for forced reload."""
        with self._cache_lock:
            self._config_cache.clear()
            self._local_cache = None
            self._cache_key = None
        logger.debug("Configuration cache cleared")
    
    @classmethod
    def clear_global_cache(cls):
        """Clear global configuration cache."""
        with cls._cache_lock:
            cls._config_cache.clear()
        logger.debug("Global configuration cache cleared")


def init_config(config_loader: Optional['ConfigLoader'] = None,
                environment: Optional[str] = None,
                **kwargs) -> Optional[Any]:
    """Initialize FigRegistry configuration through Kedro integration.
    
    This function serves as the primary entry point for FigRegistry initialization
    during Kedro project startup. It creates a configuration bridge, merges
    Kedro and FigRegistry configurations, and initializes FigRegistry with the
    unified configuration.
    
    Args:
        config_loader: Kedro ConfigLoader instance for accessing project configuration
        environment: Target environment for configuration resolution
        **kwargs: Additional configuration parameters
        
    Returns:
        FigRegistry Config instance if successful, None if FigRegistry not available
        
    Raises:
        ConfigurationMergeError: If configuration merging fails
        ImportError: If required dependencies are not available
    """
    if figregistry is None or load_config is None:
        logger.warning("FigRegistry not available - skipping configuration initialization")
        return None
    
    if ConfigLoader is not None and config_loader is None:
        logger.info("No Kedro ConfigLoader provided - using standalone FigRegistry configuration")
    
    try:
        # Create configuration bridge
        bridge = FigRegistryConfigBridge(
            config_loader=config_loader,
            environment=environment,
            **kwargs
        )
        
        # Get merged configuration
        merged_config = bridge.get_merged_config()
        
        # Convert to FigRegistry-compatible format
        config_dict = merged_config.dict(exclude={'kedro', 'environment', 
                                                 'enable_concurrent_access', 
                                                 'validation_enabled'})
        
        # Initialize FigRegistry with merged configuration
        if hasattr(figregistry, 'init_config'):
            # Use FigRegistry's init_config if available
            figregistry_config = figregistry.init_config(config_dict)
        else:
            # Fallback to creating config directly
            figregistry_config = FigRegistryConfig(**config_dict) if FigRegistryConfig else None
        
        logger.info(f"FigRegistry initialized successfully for environment: {environment or 'base'}")
        return figregistry_config
        
    except Exception as e:
        error_message = f"FigRegistry initialization failed: {str(e)}"
        logger.error(error_message)
        
        if isinstance(e, (ConfigurationMergeError, ValidationError)):
            raise
        
        raise ConfigurationMergeError(error_message)


def get_config_bridge(config_loader: Optional['ConfigLoader'] = None,
                     environment: Optional[str] = None,
                     **kwargs) -> FigRegistryConfigBridge:
    """Get a FigRegistryConfigBridge instance.
    
    Convenience function for creating configuration bridge instances with
    consistent parameter handling.
    
    Args:
        config_loader: Kedro ConfigLoader instance
        environment: Target environment
        **kwargs: Additional bridge configuration
        
    Returns:
        FigRegistryConfigBridge instance
    """
    return FigRegistryConfigBridge(
        config_loader=config_loader,
        environment=environment,
        **kwargs
    )


# Thread-safe configuration bridge instance for module-level access
_bridge_instance: Optional[FigRegistryConfigBridge] = None
_bridge_lock = threading.RLock()


def get_bridge_instance() -> Optional[FigRegistryConfigBridge]:
    """Get the module-level configuration bridge instance."""
    with _bridge_lock:
        return _bridge_instance


def set_bridge_instance(bridge: FigRegistryConfigBridge):
    """Set the module-level configuration bridge instance."""
    global _bridge_instance
    with _bridge_lock:
        _bridge_instance = bridge