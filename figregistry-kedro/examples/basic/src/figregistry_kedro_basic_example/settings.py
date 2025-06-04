"""
Kedro Project Settings for FigRegistry-Kedro Basic Example

This module configures the Kedro project settings for the basic figregistry-kedro
integration example, demonstrating how to enable automated figure styling and
versioning throughout Kedro pipeline execution. The primary responsibility is
registering FigRegistryHooks to provide non-invasive lifecycle integration
between FigRegistry's configuration-driven visualization system and Kedro's
catalog-based data pipeline architecture.

Key Features Demonstrated:
- F-006: Lifecycle integration through FigRegistryHooks registration
- F-006-RQ-001: Automated FigRegistry initialization before pipeline execution
- F-006-RQ-002: Context management for FigureDataSet instances
- Section 0.1.1: Zero-touch figure management in data science workflows
- Section 0.2.1: Plugin activation patterns for broader ecosystem adoption

The settings configuration enables:
- Automatic FigRegistry configuration merging during project startup
- Thread-safe context management for parallel pipeline execution
- Performance monitoring and error handling for enterprise environments
- Seamless integration with existing Kedro workflows without code changes

Usage:
    This file is automatically loaded by Kedro during project initialization.
    No manual configuration required - the hooks activate automatically when
    the project starts, providing transparent figure management capabilities
    throughout all pipeline executions.

Example:
    # Run Kedro pipeline with automated figure styling
    kedro run --pipeline=data_visualization
    
    # FigRegistryHooks automatically:
    # 1. Initialize configuration during project startup
    # 2. Establish styling context before pipeline execution
    # 3. Enable FigureDataSet automated styling throughout pipeline
    # 4. Clean up resources after pipeline completion
"""

from typing import Any, Dict, Iterable, Optional

# Import FigRegistryHooks for lifecycle integration
from figregistry_kedro.hooks import FigRegistryHooks

# Kedro framework version requirement for compatibility
# This example requires Kedro >= 0.18.0 for AbstractDataSet interface stability
# and hook specification compatibility as defined in Section 5.3.4.1
KEDRO_VERSION_REQUIRED = ">=0.18.0,<0.20.0"

# Project metadata for identification and debugging
PROJECT_NAME = "figregistry_kedro_basic_example"
PROJECT_VERSION = "0.1.0"

# FigRegistryHooks Configuration
# 
# The hooks are configured with enterprise-grade settings that balance
# performance, reliability, and observability requirements. These settings
# demonstrate production-ready configuration patterns while maintaining
# optimal performance for the basic example use case.
FIGREGISTRY_HOOKS_CONFIG = {
    # Enable comprehensive performance monitoring for demonstration
    # In production environments, this provides visibility into hook execution
    # times, configuration merging performance, and resource usage patterns
    "enable_performance_monitoring": True,
    
    # Set initialization timeout to 5 seconds (5000ms) to accommodate
    # complex configuration merging scenarios while maintaining the <5ms
    # target execution time per Section 5.2.8 performance requirements
    "initialization_timeout_ms": 5000.0,
    
    # Enable configuration caching for optimal performance during pipeline
    # execution. This ensures merged Kedro-FigRegistry configurations are
    # cached throughout the session for consistent styling behavior
    "config_cache_enabled": True,
    
    # Enable strict validation for merged configurations to ensure type
    # safety and schema compliance across both Kedro and FigRegistry
    # configuration systems per Section 5.3.4.2 configuration strategy
    "strict_validation": True,
    
    # Enable graceful fallback behavior when configuration initialization
    # fails, allowing pipelines to continue execution with default styling
    # rather than failing completely per F-006 non-invasive integration
    "fallback_on_errors": True
}

# Kedro Hooks Registration
#
# This is the core configuration that enables F-006 lifecycle integration.
# FigRegistryHooks is registered as a tuple entry in the HOOKS configuration,
# following Kedro's standard plugin registration patterns. The hooks will
# automatically activate during pipeline execution to provide:
#
# 1. Configuration Bridge Initialization (after_config_loaded)
#    - Merges Kedro environment-specific configurations with FigRegistry YAML
#    - Validates merged configurations for type safety and schema compliance
#    - Establishes configuration context for FigureDataSet instances
#
# 2. Pipeline Context Setup (before_pipeline_run)
#    - Initializes FigRegistry with merged configuration parameters
#    - Establishes styling context for automated figure management
#    - Validates catalog integration for FigureDataSet entries
#
# 3. Resource Cleanup (after_pipeline_run)
#    - Cleans up FigRegistry context and configuration state
#    - Logs performance metrics for monitoring and optimization
#    - Ensures proper resource management in long-running environments
HOOKS: Iterable[Any] = (
    # Register FigRegistryHooks with enterprise-grade configuration
    # This enables automated figure styling and versioning throughout
    # all pipeline executions per F-006 feature requirements
    FigRegistryHooks(**FIGREGISTRY_HOOKS_CONFIG),
)

# Session Store Configuration
#
# Configure session store for optimal performance with figregistry-kedro
# integration. The memory session store provides fastest access patterns
# for configuration and context data required by the hooks and datasets.
SESSION_STORE_CLASS = "kedro.io.MemoryDataSet"

# Session Store Arguments
SESSION_STORE_ARGS: Dict[str, Any] = {
    # Configure session store for efficient configuration caching
    # This supports the FigRegistryHooks configuration management
    # and FigureDataSet context access patterns
}

# Data Catalog Configuration
#
# Optional catalog configuration for enhanced figregistry-kedro integration.
# These settings optimize catalog behavior for FigureDataSet operations
# and automated figure versioning within the basic example workflows.
CATALOG_CONFIG: Dict[str, Any] = {
    # Enable catalog versioning to support FigureDataSet versioning
    # integration with Kedro's built-in experiment tracking capabilities
    "versioned": True,
    
    # Configure catalog for optimal performance with figure datasets
    # This reduces overhead for FigureDataSet save/load operations
    "enable_cache": True,
}

# Configuration Loader Settings
#
# Enhanced configuration loader settings that support the FigRegistryConfigBridge
# integration for seamless merging of Kedro and FigRegistry configurations.
# These settings ensure proper configuration hierarchy and environment support.
CONFIG_LOADER_CLASS = "kedro.config.ConfigLoader"
CONFIG_LOADER_ARGS: Dict[str, Any] = {
    # Define configuration sources that include both standard Kedro
    # configuration patterns and figregistry.yml support
    "config_patterns": {
        # Standard Kedro configuration patterns
        "catalog": ["catalog*.yml", "catalog*.yaml"],
        "parameters": ["parameters*.yml", "parameters*.yaml"],
        "credentials": ["credentials*.yml", "credentials*.yaml"],
        
        # FigRegistry configuration integration
        # This enables conf/base/figregistry.yml and environment-specific
        # figregistry configurations per Section 5.3.4.2 merging strategy
        "figregistry": ["figregistry*.yml", "figregistry*.yaml"],
    },
    
    # Enable environment-specific configuration overrides
    # This supports multi-environment deployments with different
    # FigRegistry styling configurations for dev/staging/production
    "base_env": "base",
    "default_run_env": "local",
}

# Pipeline Discovery Configuration
#
# Configure pipeline discovery to work optimally with figregistry-kedro
# integration, ensuring proper initialization order and context management.
PIPELINES_MODULE = f"{PROJECT_NAME}.pipeline_registry"

# Advanced Hook Configuration Options
#
# These optional configurations demonstrate advanced hook customization
# patterns for enterprise environments with specific requirements for
# monitoring, error handling, and performance optimization.

# Optional: Custom hook initialization parameters for specialized environments
ADVANCED_HOOK_CONFIG: Optional[Dict[str, Any]] = {
    # Custom configuration for enterprise logging integration
    "logging_config": {
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "enable_structured_logging": False,  # Set to True for JSON logging
    },
    
    # Performance monitoring configuration for enterprise environments
    "monitoring_config": {
        "enable_metrics_collection": True,
        "metrics_export_interval": 300,  # 5 minutes
        "performance_alerting_threshold_ms": 100,  # Alert if >100ms
    },
    
    # Error handling configuration for production resilience
    "error_handling_config": {
        "max_retry_attempts": 3,
        "retry_delay_ms": 1000,
        "escalation_threshold": 5,  # Escalate after 5 failures
    },
}

# Example: Environment-Specific Hook Configuration
#
# This demonstrates how to customize hook behavior based on deployment
# environment, enabling different settings for development, testing,
# and production environments while maintaining consistent functionality.
def get_environment_specific_hooks(environment: str = "local") -> Iterable[Any]:
    """
    Get environment-specific FigRegistryHooks configuration.
    
    This function demonstrates how to customize hook configuration based on
    the deployment environment, enabling different performance characteristics
    and error handling behaviors for development, testing, and production.
    
    Args:
        environment: Deployment environment name (local, dev, staging, prod)
        
    Returns:
        Tuple of configured FigRegistryHooks instances
        
    Example:
        # Use environment-specific configuration
        HOOKS = get_environment_specific_hooks("production")
    """
    # Base configuration applies to all environments
    base_config = FIGREGISTRY_HOOKS_CONFIG.copy()
    
    # Environment-specific configuration overrides
    if environment == "production":
        # Production optimized settings
        base_config.update({
            "enable_performance_monitoring": True,
            "initialization_timeout_ms": 2000.0,  # Stricter timeout
            "strict_validation": True,
            "fallback_on_errors": False,  # Fail fast in production
        })
    elif environment == "development":
        # Development optimized settings
        base_config.update({
            "enable_performance_monitoring": True,
            "initialization_timeout_ms": 10000.0,  # Relaxed timeout
            "strict_validation": False,  # Allow experimentation
            "fallback_on_errors": True,  # Continue on errors
        })
    elif environment == "testing":
        # Testing optimized settings
        base_config.update({
            "enable_performance_monitoring": False,  # Reduce overhead
            "initialization_timeout_ms": 1000.0,  # Fast timeout
            "strict_validation": True,
            "fallback_on_errors": False,  # Detect issues early
        })
    
    return (FigRegistryHooks(**base_config),)

# Documentation for Developers
#
# This section provides comprehensive information for developers working
# with the figregistry-kedro integration, including common patterns,
# troubleshooting guidance, and extension points for custom workflows.

# Common Integration Patterns:
#
# 1. Basic Hook Registration (Current Configuration):
#    HOOKS = (FigRegistryHooks(),)
#    
#    Enables automated figure styling with default settings. Suitable for
#    most development and testing scenarios.
#
# 2. Custom Hook Configuration:
#    HOOKS = (FigRegistryHooks(
#        enable_performance_monitoring=True,
#        initialization_timeout_ms=3000.0,
#        strict_validation=True
#    ),)
#    
#    Provides fine-grained control over hook behavior for specific
#    requirements or enterprise environments.
#
# 3. Environment-Specific Configuration:
#    HOOKS = get_environment_specific_hooks(os.getenv("KEDRO_ENV", "local"))
#    
#    Automatically adjusts hook configuration based on deployment
#    environment for optimal performance and reliability.

# Troubleshooting Guide:
#
# Common Issues and Solutions:
#
# 1. Hook Initialization Failures:
#    - Verify figregistry-kedro package is installed
#    - Check Kedro version compatibility (>= 0.18.0)
#    - Ensure figregistry.yml exists or fallback_on_errors=True
#
# 2. Configuration Merging Errors:
#    - Validate YAML syntax in both Kedro and FigRegistry configurations
#    - Check configuration schema compatibility
#    - Enable strict_validation=False for debugging
#
# 3. Performance Issues:
#    - Reduce initialization_timeout_ms for faster startup
#    - Disable performance monitoring in high-volume scenarios
#    - Enable config_cache for improved runtime performance
#
# 4. Integration Conflicts:
#    - Ensure no duplicate hook registrations
#    - Verify FigureDataSet catalog entries are properly configured
#    - Check for conflicting matplotlib rcParams settings

# Version Compatibility Matrix:
#
# This configuration is tested and supported with:
# - Python: >= 3.10, < 4.0
# - Kedro: >= 0.18.0, < 0.20.0
# - FigRegistry: >= 0.3.0
# - Matplotlib: >= 3.9.0
# - Pydantic: >= 2.9.0
#
# For other version combinations, refer to the figregistry-kedro
# documentation and compatibility testing results.

# Export Configuration Summary
#
# Summary of key configuration decisions for this basic example:
#
# ✓ FigRegistryHooks registered with production-ready settings
# ✓ Performance monitoring enabled for visibility
# ✓ Graceful error handling with fallback behavior
# ✓ Configuration caching enabled for optimal performance
# ✓ Strict validation for type safety and schema compliance
# ✓ Environment-specific configuration support available
# ✓ Comprehensive documentation for developers and operators
#
# This configuration demonstrates the full capabilities of figregistry-kedro
# integration while maintaining simplicity for the basic example use case.
# The settings can be customized further based on specific project requirements
# and deployment environments.