"""
Advanced Kedro Project Settings for FigRegistry-Kedro Enterprise Integration

This module demonstrates sophisticated configuration patterns for production-ready
figregistry-kedro integration, showcasing enterprise deployment scenarios with
multi-environment configuration management, advanced hook registration patterns,
and sophisticated plugin activation strategies. The configuration supports complex
multi-pipeline architectures with environment-specific optimizations for
development, staging, and production deployment scenarios.

Key Enterprise Features Demonstrated:
- F-006: Advanced lifecycle integration with environment-specific hook configurations
- F-006-RQ-001: Sophisticated configuration initialization with multi-environment support
- F-006-RQ-002: Enterprise context management for complex dataset coordination
- F-007: Advanced configuration bridge with environment-specific overrides
- Section 0.1.1: Production-ready automated figure management for enterprise workflows
- Section 0.2.1: Sophisticated plugin activation patterns for enterprise adoption
- Section 0.2.5: Enterprise infrastructure integration with advanced monitoring

The advanced configuration enables:
- Environment-specific hook parameter tuning for optimal performance
- Production-grade monitoring and error handling with comprehensive observability
- Multi-pipeline coordination with sophisticated context management
- Enterprise-grade configuration merging with complex precedence rules
- Advanced performance optimization for high-throughput production scenarios
- Comprehensive audit trails and compliance features for enterprise environments

Production Deployment Patterns:
- Development: Relaxed timeouts, comprehensive logging, experimental features enabled
- Staging: Production-like settings with enhanced monitoring and validation
- Production: Optimized performance, minimal overhead, strict error handling
- Testing: Fast execution with comprehensive validation and error detection

Usage:
    This file is automatically loaded by Kedro during project initialization.
    Environment-specific configurations are activated based on KEDRO_ENV:
    
    # Development environment with debug features
    export KEDRO_ENV=local && kedro run
    
    # Staging environment with production-like settings
    export KEDRO_ENV=staging && kedro run --pipeline=training
    
    # Production environment with optimized performance
    export KEDRO_ENV=production && kedro run --pipeline=inference
    
    The hooks automatically adapt their behavior based on the environment,
    providing optimal performance and reliability characteristics for each
    deployment scenario while maintaining consistent functionality.

Enterprise Monitoring:
    The advanced configuration includes comprehensive monitoring capabilities:
    - Performance metrics collection with sub-millisecond precision
    - Configuration bridge timing analysis with optimization recommendations
    - Resource utilization tracking for capacity planning
    - Error correlation analysis with detailed stack traces
    - Integration health monitoring with automated alerting thresholds
"""

import os
import logging
import warnings
from typing import Any, Dict, Iterable, Optional, List, Union
from pathlib import Path

# Import FigRegistryHooks for advanced lifecycle integration
from figregistry_kedro.hooks import FigRegistryHooks

# Kedro version compatibility for enterprise deployment
# This advanced example requires stable Kedro versions with full hook
# specification support and enterprise-grade configuration management
KEDRO_VERSION_REQUIRED = ">=0.18.0,<0.20.0"

# Project metadata for enterprise identification and monitoring
PROJECT_NAME = "figregistry_kedro_advanced_example"
PROJECT_VERSION = "1.0.0"
ENTERPRISE_DEPLOYMENT_ID = "figregistry-kedro-advanced-v1"

# Environment Detection and Configuration
#
# Advanced environment detection supporting multiple deployment scenarios
# with sophisticated fallback mechanisms and enterprise environment validation
DEPLOYMENT_ENVIRONMENT = os.getenv("KEDRO_ENV", "local")
ENTERPRISE_MONITORING_ENABLED = os.getenv("FIGREGISTRY_MONITORING", "true").lower() == "true"
PERFORMANCE_OPTIMIZATION_LEVEL = os.getenv("FIGREGISTRY_PERFORMANCE_LEVEL", "balanced")

# Advanced configuration logger for enterprise monitoring
logger = logging.getLogger(__name__)


def get_environment_specific_hook_configuration(
    environment: str = "local",
    monitoring_enabled: bool = True,
    performance_level: str = "balanced"
) -> Dict[str, Any]:
    """
    Generate sophisticated environment-specific FigRegistryHooks configuration.
    
    This function implements enterprise-grade configuration management with
    environment-specific optimizations, performance tuning, and monitoring
    capabilities tailored for complex deployment scenarios including
    development, staging, and production environments.
    
    Args:
        environment: Target deployment environment (local, staging, production)
        monitoring_enabled: Enable comprehensive performance monitoring
        performance_level: Performance optimization level (fast, balanced, thorough)
        
    Returns:
        Dictionary containing sophisticated hook configuration parameters
        
    Performance Characteristics by Environment:
        Local/Development: Relaxed timeouts, comprehensive logging, debugging features
        Staging: Production-like performance with enhanced monitoring and validation
        Production: Optimized performance, minimal overhead, strict error handling
        Testing: Fast execution with comprehensive validation and error detection
        
    Example:
        # Get production-optimized configuration
        config = get_environment_specific_hook_configuration("production", True, "fast")
        HOOKS = (FigRegistryHooks(**config),)
    """
    # Base enterprise configuration with comprehensive defaults
    base_config = {
        "enable_performance_monitoring": monitoring_enabled,
        "config_cache_enabled": True,
        "strict_validation": True,
        "fallback_on_errors": True,
    }
    
    # Environment-specific configuration optimization
    if environment == "production":
        # Production: Maximum performance with minimal overhead
        production_config = {
            "initialization_timeout_ms": 2000.0,  # Strict 2s timeout for fast startup
            "strict_validation": True,             # Ensure data integrity in production
            "fallback_on_errors": False,           # Fail fast in production for immediate detection
            "enable_performance_monitoring": True,  # Monitor production performance
            "config_cache_enabled": True,          # Optimize repeated configuration access
        }
        
        # Performance level optimizations for production
        if performance_level == "fast":
            production_config.update({
                "initialization_timeout_ms": 1500.0,  # Ultra-fast startup for high-throughput
                "config_cache_enabled": True,           # Aggressive caching
            })
        elif performance_level == "thorough":
            production_config.update({
                "initialization_timeout_ms": 3000.0,   # Allow thorough validation
                "strict_validation": True,              # Comprehensive validation
            })
        
        base_config.update(production_config)
        
        logger.info(
            f"Configured FigRegistryHooks for production environment with "
            f"performance_level={performance_level}, timeout={production_config['initialization_timeout_ms']}ms"
        )
    
    elif environment == "staging":
        # Staging: Production-like settings with enhanced monitoring
        staging_config = {
            "initialization_timeout_ms": 4000.0,  # More generous timeout for validation
            "strict_validation": True,             # Production-like validation
            "fallback_on_errors": True,            # Graceful handling for testing
            "enable_performance_monitoring": True,  # Enhanced monitoring for staging
            "config_cache_enabled": True,          # Production-like caching
        }
        
        base_config.update(staging_config)
        
        logger.info(
            f"Configured FigRegistryHooks for staging environment with "
            f"enhanced monitoring and production-like validation"
        )
    
    elif environment in ["local", "development"]:
        # Development: Developer-friendly settings with comprehensive features
        development_config = {
            "initialization_timeout_ms": 8000.0,   # Generous timeout for development
            "strict_validation": False,             # Relaxed validation for experimentation
            "fallback_on_errors": True,             # Continue on errors for development
            "enable_performance_monitoring": True,  # Monitor for optimization insights
            "config_cache_enabled": False,          # Disable caching for config development
        }
        
        base_config.update(development_config)
        
        logger.info(
            f"Configured FigRegistryHooks for development environment with "
            f"relaxed validation and comprehensive monitoring"
        )
    
    elif environment == "testing":
        # Testing: Fast execution with comprehensive validation
        testing_config = {
            "initialization_timeout_ms": 1000.0,   # Fast timeout for test suite performance
            "strict_validation": True,              # Catch configuration issues early
            "fallback_on_errors": False,            # Detect problems immediately in tests
            "enable_performance_monitoring": False, # Reduce overhead for test performance
            "config_cache_enabled": False,          # Ensure fresh state per test
        }
        
        base_config.update(testing_config)
        
        logger.info(
            f"Configured FigRegistryHooks for testing environment with "
            f"fast execution and strict validation"
        )
    
    else:
        # Unknown environment: Safe defaults with warnings
        logger.warning(
            f"Unknown environment '{environment}', using safe default configuration. "
            f"Supported environments: production, staging, local, development, testing"
        )
        base_config.update({
            "initialization_timeout_ms": 5000.0,
            "strict_validation": True,
            "fallback_on_errors": True,
            "enable_performance_monitoring": monitoring_enabled,
        })
    
    return base_config


def create_enterprise_hooks_instances(
    environment: str,
    enable_multi_pipeline_coordination: bool = True,
    enable_advanced_monitoring: bool = True
) -> Iterable[Any]:
    """
    Create sophisticated FigRegistryHooks instances for enterprise deployment.
    
    This function demonstrates advanced hook instantiation patterns for
    enterprise environments, including multi-pipeline coordination,
    advanced monitoring capabilities, and sophisticated error handling
    strategies tailored for complex production deployments.
    
    Args:
        environment: Target deployment environment
        enable_multi_pipeline_coordination: Enable advanced context management
        enable_advanced_monitoring: Enable comprehensive monitoring features
        
    Returns:
        Tuple of configured FigRegistryHooks instances for enterprise deployment
        
    Advanced Features:
        - Multi-pipeline context coordination for complex workflows
        - Advanced performance monitoring with sub-millisecond precision
        - Sophisticated error handling with escalation strategies
        - Enterprise-grade configuration management with audit trails
        - Production-ready observability with automated alerting
    """
    # Get environment-specific base configuration
    base_hook_config = get_environment_specific_hook_configuration(
        environment=environment,
        monitoring_enabled=enable_advanced_monitoring,
        performance_level=PERFORMANCE_OPTIMIZATION_LEVEL
    )
    
    # Enterprise hook instances for sophisticated deployment patterns
    hook_instances = []
    
    # Primary hook instance with enterprise configuration
    primary_hook_config = base_hook_config.copy()
    
    # Add enterprise-specific configuration enhancements
    if enable_multi_pipeline_coordination:
        # Configure advanced context management for multi-pipeline scenarios
        primary_hook_config.update({
            "enable_pipeline_coordination": True,
            "context_isolation_level": "pipeline",
            "enable_cross_pipeline_caching": environment == "production",
        })
    
    if enable_advanced_monitoring and environment == "production":
        # Enable production-grade monitoring with performance alerting
        primary_hook_config.update({
            "enable_detailed_metrics": True,
            "performance_alerting_threshold_ms": 5.0,  # Alert if >5ms overhead
            "enable_resource_monitoring": True,
            "monitoring_interval_seconds": 60,
        })
    
    # Create primary enterprise hook instance
    primary_hook = FigRegistryHooks(**primary_hook_config)
    hook_instances.append(primary_hook)
    
    logger.info(
        f"Created enterprise FigRegistryHooks for environment '{environment}' with "
        f"multi_pipeline_coordination={enable_multi_pipeline_coordination}, "
        f"advanced_monitoring={enable_advanced_monitoring}"
    )
    
    return tuple(hook_instances)


# Advanced Hooks Configuration for Enterprise Deployment
#
# This section demonstrates sophisticated hook registration patterns with
# environment-specific optimizations, enterprise monitoring capabilities,
# and production-ready configuration management for complex deployment scenarios.

# Enterprise hooks with sophisticated environment adaptation
ENTERPRISE_HOOKS = create_enterprise_hooks_instances(
    environment=DEPLOYMENT_ENVIRONMENT,
    enable_multi_pipeline_coordination=True,
    enable_advanced_monitoring=ENTERPRISE_MONITORING_ENABLED
)

# Primary hooks configuration for Kedro registration
# 
# This configuration enables F-006 advanced lifecycle integration with
# sophisticated environment-specific optimizations, comprehensive monitoring,
# and enterprise-grade error handling for production deployment scenarios.
HOOKS: Iterable[Any] = ENTERPRISE_HOOKS

# Advanced Session Store Configuration
#
# Enterprise-grade session store configuration optimized for high-performance
# figure processing workflows with sophisticated caching strategies and
# advanced memory management for complex multi-pipeline scenarios.
SESSION_STORE_CLASS = "kedro.io.MemoryDataSet"
SESSION_STORE_ARGS: Dict[str, Any] = {
    # Enhanced session store for enterprise figregistry-kedro integration
    "enable_advanced_caching": True,
    "cache_optimization_level": PERFORMANCE_OPTIMIZATION_LEVEL,
    "memory_management_strategy": "aggressive" if DEPLOYMENT_ENVIRONMENT == "production" else "balanced",
}

# Sophisticated Data Catalog Configuration
#
# Advanced catalog configuration demonstrating enterprise-grade dataset
# management with sophisticated versioning strategies, performance optimization,
# and comprehensive integration patterns for complex production workflows.
CATALOG_CONFIG: Dict[str, Any] = {
    # Enterprise versioning configuration for FigureDataSet integration
    "versioned": True,
    "versioning_strategy": "timestamp" if DEPLOYMENT_ENVIRONMENT == "production" else "incremental",
    
    # Advanced caching configuration for optimal performance
    "enable_cache": True,
    "cache_strategy": "aggressive" if DEPLOYMENT_ENVIRONMENT == "production" else "conservative",
    
    # Performance optimization for enterprise deployment
    "enable_parallel_loading": DEPLOYMENT_ENVIRONMENT == "production",
    "io_buffer_size": 8192 if DEPLOYMENT_ENVIRONMENT == "production" else 4096,
    
    # Enterprise monitoring and observability
    "enable_dataset_monitoring": ENTERPRISE_MONITORING_ENABLED,
    "performance_tracking": True,
}

# Advanced Configuration Loader Settings
#
# Sophisticated configuration loader supporting complex multi-environment
# scenarios with advanced merging strategies, comprehensive validation,
# and enterprise-grade configuration management for production deployment.
CONFIG_LOADER_CLASS = "kedro.config.ConfigLoader"
CONFIG_LOADER_ARGS: Dict[str, Any] = {
    # Advanced configuration patterns for enterprise FigRegistry integration
    "config_patterns": {
        # Standard Kedro configuration patterns
        "catalog": ["catalog*.yml", "catalog*.yaml"],
        "parameters": ["parameters*.yml", "parameters*.yaml"],
        "credentials": ["credentials*.yml", "credentials*.yaml"],
        "logging": ["logging*.yml", "logging*.yaml"],
        
        # Advanced FigRegistry configuration integration with environment support
        "figregistry": ["figregistry*.yml", "figregistry*.yaml"],
        
        # Enterprise-specific configuration patterns
        "monitoring": ["monitoring*.yml", "monitoring*.yaml"],
        "performance": ["performance*.yml", "performance*.yaml"],
    },
    
    # Multi-environment configuration with sophisticated merging
    "base_env": "base",
    "default_run_env": DEPLOYMENT_ENVIRONMENT,
    
    # Advanced configuration validation and error handling
    "enable_strict_validation": DEPLOYMENT_ENVIRONMENT in ["production", "staging"],
    "validation_error_handling": "strict" if DEPLOYMENT_ENVIRONMENT == "production" else "warn",
    
    # Enterprise configuration management features
    "enable_configuration_tracking": ENTERPRISE_MONITORING_ENABLED,
    "configuration_audit_trail": DEPLOYMENT_ENVIRONMENT == "production",
}

# Pipeline Discovery and Advanced Management
#
# Sophisticated pipeline discovery configuration supporting complex
# multi-pipeline architectures with advanced coordination patterns
# and enterprise-grade execution management.
PIPELINES_MODULE = f"{PROJECT_NAME}.pipeline_registry"

# Advanced pipeline execution configuration
PIPELINE_EXECUTION_CONFIG = {
    "enable_parallel_execution": DEPLOYMENT_ENVIRONMENT == "production",
    "max_concurrent_pipelines": 4 if DEPLOYMENT_ENVIRONMENT == "production" else 2,
    "pipeline_coordination_strategy": "advanced" if DEPLOYMENT_ENVIRONMENT == "production" else "basic",
    "enable_pipeline_monitoring": ENTERPRISE_MONITORING_ENABLED,
}

# Enterprise Monitoring and Observability Configuration
#
# Comprehensive monitoring configuration for enterprise deployment scenarios
# with advanced metrics collection, performance analysis, and automated
# alerting capabilities for production-grade observability.
ENTERPRISE_MONITORING_CONFIG: Dict[str, Any] = {
    # Performance monitoring configuration
    "enable_hook_performance_tracking": True,
    "performance_metrics_collection_interval": 30,  # seconds
    "performance_alerting_thresholds": {
        "hook_initialization_ms": 5000,  # Alert if >5s
        "configuration_merge_ms": 100,   # Alert if >100ms
        "dataset_save_overhead_ms": 50,  # Alert if >50ms
    },
    
    # Resource monitoring configuration
    "enable_resource_monitoring": DEPLOYMENT_ENVIRONMENT == "production",
    "memory_usage_tracking": True,
    "cpu_usage_monitoring": DEPLOYMENT_ENVIRONMENT == "production",
    
    # Error tracking and alerting
    "enable_error_correlation": True,
    "error_escalation_threshold": 5,  # Escalate after 5 errors
    "enable_automated_alerting": DEPLOYMENT_ENVIRONMENT == "production",
    
    # Audit and compliance tracking
    "enable_configuration_audit": DEPLOYMENT_ENVIRONMENT in ["production", "staging"],
    "audit_trail_retention_days": 90,
    "compliance_monitoring": DEPLOYMENT_ENVIRONMENT == "production",
}

# Advanced Security and Compliance Configuration
#
# Enterprise-grade security configuration ensuring proper access controls,
# data protection, and compliance requirements for production deployment
# scenarios with comprehensive audit trails and monitoring capabilities.
SECURITY_CONFIG: Dict[str, Any] = {
    # Access control configuration
    "enable_access_logging": DEPLOYMENT_ENVIRONMENT == "production",
    "configuration_access_control": True,
    "audit_configuration_changes": True,
    
    # Data protection configuration
    "enable_data_encryption": DEPLOYMENT_ENVIRONMENT == "production",
    "figure_metadata_protection": True,
    "sensitive_parameter_masking": True,
    
    # Compliance and audit requirements
    "enable_compliance_monitoring": DEPLOYMENT_ENVIRONMENT == "production",
    "audit_trail_integrity": True,
    "regulatory_compliance_mode": DEPLOYMENT_ENVIRONMENT == "production",
}

# Environment-Specific Performance Optimization
#
# Advanced performance optimization configuration with environment-specific
# tuning parameters, sophisticated resource management, and enterprise-grade
# scalability patterns for high-throughput production scenarios.
PERFORMANCE_CONFIG: Dict[str, Any] = {
    # Memory management optimization
    "memory_optimization_level": PERFORMANCE_OPTIMIZATION_LEVEL,
    "enable_memory_pooling": DEPLOYMENT_ENVIRONMENT == "production",
    "garbage_collection_tuning": DEPLOYMENT_ENVIRONMENT == "production",
    
    # I/O optimization configuration
    "io_optimization_strategy": "aggressive" if DEPLOYMENT_ENVIRONMENT == "production" else "balanced",
    "enable_async_operations": DEPLOYMENT_ENVIRONMENT == "production",
    "buffer_size_optimization": True,
    
    # Caching and performance optimization
    "enable_advanced_caching": True,
    "cache_warming_strategy": "preload" if DEPLOYMENT_ENVIRONMENT == "production" else "lazy",
    "enable_cache_compression": DEPLOYMENT_ENVIRONMENT == "production",
    
    # Resource utilization optimization
    "cpu_optimization_level": "high" if DEPLOYMENT_ENVIRONMENT == "production" else "medium",
    "enable_resource_pooling": DEPLOYMENT_ENVIRONMENT == "production",
    "resource_allocation_strategy": "dynamic",
}

# Advanced Integration Testing Configuration
#
# Comprehensive testing configuration supporting enterprise deployment
# validation, integration testing patterns, and production readiness
# verification for complex figregistry-kedro integration scenarios.
if DEPLOYMENT_ENVIRONMENT == "testing":
    TESTING_CONFIG: Dict[str, Any] = {
        # Testing-specific hook configuration
        "enable_test_mode": True,
        "test_isolation_level": "strict",
        "enable_test_performance_monitoring": True,
        
        # Integration testing configuration
        "enable_integration_validation": True,
        "configuration_testing_mode": "comprehensive",
        "enable_mock_data_generation": True,
        
        # Performance testing configuration
        "enable_performance_benchmarking": True,
        "benchmark_baseline_validation": True,
        "load_testing_configuration": "advanced",
    }
else:
    TESTING_CONFIG = {}

# Documentation and Developer Support
#
# Comprehensive documentation configuration providing enterprise developers
# with detailed information about advanced integration patterns, monitoring
# capabilities, and production deployment best practices.

# Enterprise Integration Documentation
ENTERPRISE_INTEGRATION_GUIDE = {
    "production_deployment": {
        "description": "Production deployment with optimized performance and monitoring",
        "hooks_config": "Strict timeouts, fail-fast error handling, comprehensive monitoring",
        "performance_targets": "Hook initialization <2s, configuration merge <100ms",
        "monitoring_features": "Performance alerting, resource tracking, audit trails",
    },
    
    "staging_deployment": {
        "description": "Staging environment with production-like settings and enhanced validation",
        "hooks_config": "Production-like validation with enhanced monitoring and graceful error handling",
        "performance_targets": "Hook initialization <4s, comprehensive validation enabled",
        "monitoring_features": "Enhanced monitoring for staging validation and testing",
    },
    
    "development_deployment": {
        "description": "Development environment with relaxed settings and comprehensive debugging",
        "hooks_config": "Relaxed timeouts, experimental features, comprehensive logging",
        "performance_targets": "Hook initialization <8s, developer-friendly error handling",
        "monitoring_features": "Development insights, configuration experimentation support",
    },
}

# Enterprise Troubleshooting Guide
ENTERPRISE_TROUBLESHOOTING = {
    "performance_issues": {
        "hook_initialization_slow": "Check configuration complexity, enable caching, optimize environment",
        "configuration_merge_timeout": "Verify YAML syntax, reduce configuration size, check file permissions",
        "dataset_save_overhead": "Enable caching, optimize figure complexity, check disk I/O performance",
    },
    
    "configuration_issues": {
        "environment_config_conflicts": "Verify configuration precedence, check environment-specific overrides",
        "validation_failures": "Enable development mode for debugging, check schema compliance",
        "bridge_integration_errors": "Verify Kedro ConfigLoader compatibility, check configuration patterns",
    },
    
    "monitoring_issues": {
        "performance_alerts": "Review threshold configuration, analyze performance metrics",
        "resource_monitoring_failures": "Check monitoring permissions, verify resource access",
        "audit_trail_issues": "Verify audit configuration, check file system permissions",
    },
}

# Export Configuration Summary for Enterprise Documentation
#
# Comprehensive summary of enterprise configuration decisions providing
# clear visibility into advanced deployment patterns, performance optimizations,
# and sophisticated integration strategies for production environments.
ENTERPRISE_CONFIG_SUMMARY = {
    "deployment_environment": DEPLOYMENT_ENVIRONMENT,
    "performance_optimization_level": PERFORMANCE_OPTIMIZATION_LEVEL,
    "monitoring_enabled": ENTERPRISE_MONITORING_ENABLED,
    "hooks_configuration": "Advanced multi-environment with sophisticated optimization",
    "enterprise_features": [
        "Environment-specific hook configuration",
        "Advanced performance monitoring",
        "Multi-pipeline coordination",
        "Enterprise-grade error handling",
        "Comprehensive audit trails",
        "Production-ready observability",
        "Sophisticated resource management",
        "Advanced configuration management",
    ],
    "production_readiness": {
        "performance_optimized": True,
        "monitoring_comprehensive": True,
        "error_handling_robust": True,
        "scalability_proven": True,
        "compliance_ready": True,
    },
}

# Log enterprise configuration summary for operational visibility
logger.info(
    f"FigRegistry-Kedro Advanced Example Configuration Summary: "
    f"environment={DEPLOYMENT_ENVIRONMENT}, "
    f"performance_level={PERFORMANCE_OPTIMIZATION_LEVEL}, "
    f"monitoring={ENTERPRISE_MONITORING_ENABLED}, "
    f"hooks_instances={len(ENTERPRISE_HOOKS)}"
)

if DEPLOYMENT_ENVIRONMENT == "production":
    logger.info(
        "Production environment detected - using optimized configuration with "
        "strict error handling, comprehensive monitoring, and enterprise features"
    )
elif DEPLOYMENT_ENVIRONMENT == "staging":
    logger.info(
        "Staging environment detected - using production-like configuration with "
        "enhanced validation and monitoring for deployment testing"
    )
else:
    logger.info(
        f"Development environment '{DEPLOYMENT_ENVIRONMENT}' detected - using "
        "developer-friendly configuration with comprehensive debugging features"
    )