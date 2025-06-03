"""Kedro Project Settings for FigRegistry Basic Example.

This file demonstrates the essential configuration for enabling figregistry-kedro 
integration within a Kedro project. The settings showcase how to register 
FigRegistryHooks for automatic lifecycle management, eliminating the need for 
manual FigRegistry configuration in pipeline nodes while maintaining full 
compatibility with Kedro's project structure and execution model.

The configuration enables F-006 Kedro Lifecycle Hooks functionality, providing:
- Automated FigRegistry initialization before pipeline execution
- Configuration bridge between Kedro and FigRegistry systems  
- Context management for FigureDataSet instances throughout pipeline lifecycle
- Non-invasive integration preserving Kedro's standard execution patterns

For production deployments, the hook configuration can be customized to enable
performance monitoring, adjust error handling behavior, and optimize for 
specific execution environments.

References:
    - Kedro Settings Documentation: https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html
    - FigRegistry Kedro Integration: https://github.com/blitzy-public-samples/figregistry-kedro
    - Hook Implementation: figregistry_kedro.hooks.FigRegistryHooks
"""

from typing import Any, Dict, List, Tuple

# Import FigRegistryHooks for lifecycle integration per F-006 requirements
try:
    from figregistry_kedro.hooks import FigRegistryHooks
except ImportError:
    # Graceful fallback for development environments where figregistry-kedro is not installed
    import warnings
    warnings.warn(
        "figregistry-kedro not found. Please install with: pip install figregistry-kedro",
        ImportWarning
    )
    FigRegistryHooks = None


# Kedro Project Settings
# ======================

# Session Store Configuration
# Configure the session store for pipeline execution data
# For basic examples, use in-memory store to avoid external dependencies
SESSION_STORE_CLASS = "kedro.framework.session.store.BaseSessionStore"

# Hook Registration - CRITICAL for FigRegistry Integration
# ========================================================

# Register FigRegistryHooks to enable lifecycle integration per F-006 requirements.
# These hooks provide:
# - F-006-RQ-001: Automated FigRegistry initialization before pipeline execution
# - F-006-RQ-002: Context management for downstream FigureDataSet instances
# - F-006-RQ-003: State cleanup after pipeline completion
# - F-006-RQ-004: Selective hook registration via settings configuration

if FigRegistryHooks is not None:
    # Standard hook registration for basic example demonstrating minimal configuration
    # This enables automatic FigRegistry initialization with default settings optimized
    # for development and demonstration scenarios
    HOOKS: Tuple[Any, ...] = (
        FigRegistryHooks(
            # Enable automatic FigRegistry configuration initialization during Kedro startup
            # This eliminates the need for manual figregistry.init_config() calls in pipeline nodes
            auto_initialize=True,
            
            # Disable performance monitoring for basic example to minimize log verbosity
            # In production environments, enable this for performance tracking and optimization
            enable_performance_monitoring=False,
            
            # Enable graceful fallback on errors to prevent pipeline failures during development
            # This allows pipelines to continue execution even if FigRegistry initialization fails
            fallback_on_error=True,
            
            # Set maximum initialization time to 10ms for basic example (more lenient than production)
            # This accounts for potential slower file system access in development environments
            max_initialization_time=0.010  # 10ms for development environments
        ),
    )
else:
    # Fallback configuration when figregistry-kedro is not available
    # This allows the Kedro project to function without the plugin for development convenience
    HOOKS: Tuple[Any, ...] = ()

# Context Class Configuration
# ===========================

# Use Kedro's standard context class for the basic example
# This provides all standard Kedro functionality without additional customization
CONTEXT_CLASS = "kedro.framework.context.KedroContext"

# Configuration Loader Settings
# ==============================

# Configure Kedro's ConfigLoader for environment-specific configuration management
# The OmegaConfigLoader provides robust YAML processing compatible with FigRegistry configurations
CONFIG_LOADER_CLASS = "kedro.config.OmegaConfigLoader"

# Configuration loader parameters for enhanced FigRegistry integration
CONFIG_LOADER_ARGS: Dict[str, Any] = {
    # Environment variable to control configuration environment (development, staging, production)
    "base_env": "base",
    
    # Default environment if KEDRO_ENV is not set
    "default_run_env": "local",
    
    # Enable configuration merging for FigRegistry YAML files
    # This allows environment-specific overrides of FigRegistry configurations
    "config_patterns": {
        # Standard Kedro configuration patterns
        "catalog": ["catalog*.yml", "catalog*.yaml"],
        "parameters": ["parameters*.yml", "parameters*.yaml"],
        "credentials": ["credentials*.yml", "credentials*.yaml"],
        
        # FigRegistry configuration pattern for Kedro integration
        # Supports both traditional figregistry.yaml and Kedro-style figregistry*.yml patterns
        "figregistry": ["figregistry*.yml", "figregistry*.yaml"]
    }
}

# Data Catalog Configuration
# ===========================

# Use Kedro's standard data catalog for the basic example
# The DataCatalog will automatically discover and use FigureDataSet instances
# based on catalog configuration entries
DATA_CATALOG_CLASS = "kedro.io.DataCatalog"

# Runner Configuration
# ====================

# Use sequential runner for basic example to simplify debugging and demonstration
# For production environments, consider using ParallelRunner or ThreadRunner
# FigRegistryHooks support concurrent execution with proper thread safety
RUNNER_CLASS = "kedro.runner.SequentialRunner"

# Logging Configuration
# =====================

# Configure logging for FigRegistry integration debugging
# The basic example uses INFO level to show integration progress without overwhelming detail
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
    },
    
    "loggers": {
        # Enable INFO level logging for FigRegistry components to show integration progress
        "figregistry": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        
        # Enable INFO level logging for figregistry-kedro plugin components
        "figregistry_kedro": {
            "level": "INFO", 
            "handlers": ["console"],
            "propagate": False
        },
        
        # Standard Kedro logging configuration
        "kedro": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    },
    
    "root": {
        "level": "WARNING",
        "handlers": ["console"]
    }
}

# Plugin Discovery Configuration
# ===============================

# Kedro will automatically discover the figregistry-kedro plugin when installed
# No explicit plugin registration required due to standard Python package entry points

# Environment Variable Defaults
# ==============================

# Set default environment variables for the basic example
# These can be overridden by actual environment variables during execution
ENVIRONMENT_DEFAULTS = {
    # Default to local environment for development convenience
    "KEDRO_ENV": "local",
    
    # Configure FigRegistry to use relative paths suitable for the example structure
    "FIGREGISTRY_OUTPUT_BASE": "data/08_reporting",
    
    # Enable FigRegistry logging for integration demonstration
    "FIGREGISTRY_LOG_LEVEL": "INFO"
}

# Advanced Hook Configuration Examples (Commented for Basic Example)
# ===================================================================

# For production environments or advanced usage scenarios, FigRegistryHooks
# can be configured with additional parameters:

# HOOKS = (
#     FigRegistryHooks(
#         # Enable detailed performance monitoring for production optimization
#         enable_performance_monitoring=True,
#         
#         # Disable error fallback for strict production environments
#         fallback_on_error=False,
#         
#         # Strict timing requirements for high-performance pipelines
#         max_initialization_time=0.005,  # 5ms maximum
#     ),
# )

# Alternatively, use the convenience function for advanced configuration:

# from figregistry_kedro.hooks import create_hooks
# 
# HOOKS = (
#     create_hooks(
#         enable_performance_monitoring=True,
#         fallback_on_error=False
#     ),
# )

# Integration Validation
# ======================

# Validate that hook registration is successful during module import
if FigRegistryHooks is not None and len(HOOKS) > 0:
    # Verify hook configuration for development debugging
    hook_instance = HOOKS[0]
    
    # Log hook configuration for development transparency
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"FigRegistryHooks registered with configuration: {hook_instance.get_state()}")
else:
    # Warn about missing figregistry-kedro integration
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "FigRegistryHooks not registered. Install figregistry-kedro to enable automated figure management."
    )

# Module Metadata
# ===============

__version__ = "0.1.0"
__author__ = "FigRegistry Kedro Integration Team"
__description__ = "Basic example settings demonstrating figregistry-kedro integration"

# Export configuration elements for external access
__all__ = [
    "HOOKS",
    "CONTEXT_CLASS", 
    "CONFIG_LOADER_CLASS",
    "CONFIG_LOADER_ARGS",
    "DATA_CATALOG_CLASS",
    "RUNNER_CLASS",
    "LOGGING_CONFIG",
    "ENVIRONMENT_DEFAULTS"
]