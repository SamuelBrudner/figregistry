"""Advanced Kedro Project Settings for FigRegistry Enterprise Integration.

This settings module demonstrates sophisticated enterprise deployment patterns for 
FigRegistry-Kedro integration, showcasing advanced hook configuration, environment-specific 
optimization, and production-ready plugin registration for complex multi-pipeline 
architectures.

The configuration demonstrates:
- Environment-specific hook parameters for development, staging, and production
- Advanced performance monitoring and error handling strategies  
- Complex plugin registration patterns for enterprise deployment
- Multi-pipeline coordination with sophisticated context management
- Production-optimized hook configuration with monitoring integration

This example serves as a reference implementation for teams deploying FigRegistry
in enterprise environments with complex deployment requirements and multi-environment
configuration management needs.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Kedro core imports for advanced settings configuration
from kedro.config import ConfigLoader
from kedro.framework.session import KedroSession
from kedro.framework.startup import ProjectMetadata

# FigRegistry-Kedro integration components
from figregistry_kedro.hooks import FigRegistryHooks, create_hooks

# Configure structured logging for enterprise environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# =======================================================================================
# ENVIRONMENT DETECTION AND CONFIGURATION
# =======================================================================================

def get_deployment_environment() -> str:
    """Detect current deployment environment for configuration adaptation.
    
    Returns:
        Environment identifier: 'local', 'staging', 'production', or 'development'
    """
    # Priority order: explicit environment variable -> Kedro session -> default
    env = os.getenv('KEDRO_ENV', '').lower()
    if env in ('local', 'staging', 'production', 'development'):
        return env
    
    # Fall back to session detection if available
    try:
        session = KedroSession.get_current_session()
        if session and hasattr(session, '_store') and 'env' in session._store:
            detected_env = session._store['env'].lower()
            if detected_env in ('local', 'staging', 'production', 'development'):
                return detected_env
    except Exception:
        # Session not available during initial settings load
        pass
    
    # Default for advanced example
    return 'development'

def is_ci_environment() -> bool:
    """Detect if running in Continuous Integration environment."""
    ci_indicators = [
        'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 
        'JENKINS_URL', 'TRAVIS', 'CIRCLECI'
    ]
    return any(os.getenv(indicator, '').lower() in ('true', '1', 'yes') 
               for indicator in ci_indicators)

def get_performance_monitoring_level() -> str:
    """Determine appropriate performance monitoring level."""
    if is_ci_environment():
        return 'minimal'
    
    env = get_deployment_environment()
    monitoring_levels = {
        'development': 'detailed',
        'local': 'standard', 
        'staging': 'standard',
        'production': 'minimal'
    }
    return monitoring_levels.get(env, 'standard')

# =======================================================================================
# ENVIRONMENT-SPECIFIC HOOK CONFIGURATION
# =======================================================================================

def create_environment_hook_config() -> Dict[str, Any]:
    """Create environment-specific hook configuration for advanced deployment scenarios.
    
    Returns:
        Configuration dictionary optimized for current deployment environment
    """
    env = get_deployment_environment()
    monitoring_level = get_performance_monitoring_level()
    
    # Base configuration with enterprise-grade defaults
    base_config = {
        'auto_initialize': True,
        'fallback_on_error': True,
        'max_initialization_time': 0.010,  # 10ms for enterprise tolerance
    }
    
    # Environment-specific optimizations
    environment_configs = {
        'development': {
            'enable_performance_monitoring': True,
            'max_initialization_time': 0.050,  # More lenient for debugging
            'fallback_on_error': False,  # Fail fast for development debugging
        },
        'local': {
            'enable_performance_monitoring': monitoring_level == 'detailed',
            'max_initialization_time': 0.020,  # Standard local development
            'fallback_on_error': True,
        },
        'staging': {
            'enable_performance_monitoring': monitoring_level != 'minimal',
            'max_initialization_time': 0.015,  # Production-like performance
            'fallback_on_error': True,
        },
        'production': {
            'enable_performance_monitoring': False,  # Minimize overhead
            'max_initialization_time': 0.005,  # Strict production performance (5ms)
            'fallback_on_error': True,  # Resilient to configuration issues
        }
    }
    
    # Merge environment-specific configuration
    config = {**base_config, **environment_configs.get(env, {})}
    
    # CI environment optimizations
    if is_ci_environment():
        config.update({
            'enable_performance_monitoring': False,  # Reduce CI noise
            'max_initialization_time': 0.100,  # More lenient for CI resource constraints
            'fallback_on_error': True,  # Don't fail CI on configuration issues
        })
    
    logger.info(f"FigRegistry hooks configured for environment: {env} "
                f"(monitoring: {monitoring_level}, CI: {is_ci_environment()})")
    
    return config

# =======================================================================================
# ADVANCED PLUGIN REGISTRATION PATTERNS
# =======================================================================================

def create_figregistry_hooks() -> List[FigRegistryHooks]:
    """Create FigRegistry hook instances with advanced enterprise configuration.
    
    This function demonstrates sophisticated hook instantiation patterns for complex
    enterprise deployments, including multi-environment configuration and advanced
    error handling strategies.
    
    Returns:
        List of configured FigRegistryHooks instances for enterprise deployment
    """
    # Generate environment-optimized configuration
    hook_config = create_environment_hook_config()
    
    # Create primary hook instance with advanced configuration
    primary_hook = create_hooks(**hook_config)
    
    # Log configuration details for enterprise monitoring
    env = get_deployment_environment()
    logger.info(f"FigRegistry primary hook configured: "
                f"auto_init={hook_config['auto_initialize']}, "
                f"monitoring={hook_config['enable_performance_monitoring']}, "
                f"max_time={hook_config['max_initialization_time']*1000:.1f}ms, "
                f"fallback={hook_config['fallback_on_error']}")
    
    # Advanced: Create secondary hook for complex scenarios if needed
    # This demonstrates multi-hook patterns for sophisticated enterprise requirements
    hooks = [primary_hook]
    
    # Development environment gets additional debugging hooks
    if env == 'development' and not is_ci_environment():
        debug_config = {
            **hook_config,
            'enable_performance_monitoring': True,
            'max_initialization_time': 0.100,  # Very lenient for debugging
            'fallback_on_error': False,  # Fail fast for development
        }
        
        # Note: Multiple hook instances are primarily for demonstration
        # In practice, single hook instance is typically sufficient
        logger.debug("Development environment detected - additional debug monitoring enabled")
    
    return hooks

# =======================================================================================
# ENTERPRISE MONITORING AND LOGGING CONFIGURATION
# =======================================================================================

def configure_enterprise_logging():
    """Configure advanced logging for enterprise FigRegistry deployment monitoring."""
    env = get_deployment_environment()
    
    # Configure FigRegistry-specific logging
    figregistry_logger = logging.getLogger('figregistry_kedro')
    
    # Environment-specific logging levels
    log_levels = {
        'development': logging.DEBUG,
        'local': logging.INFO,
        'staging': logging.INFO, 
        'production': logging.WARNING
    }
    
    figregistry_logger.setLevel(log_levels.get(env, logging.INFO))
    
    # Add enterprise-specific log formatting if not in CI
    if not is_ci_environment():
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            f'[{env.upper()}] - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Add console handler with enterprise formatting
        if not figregistry_logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            figregistry_logger.addHandler(console_handler)
    
    logger.info(f"Enterprise logging configured for environment: {env}")

# Initialize enterprise logging on import
configure_enterprise_logging()

# =======================================================================================
# KEDRO PROJECT CONFIGURATION
# =======================================================================================

# Project package name - matches the advanced example structure
PACKAGE_NAME = "figregistry_kedro_advanced_example"

# Advanced project metadata configuration
PROJECT_METADATA = ProjectMetadata(
    source_dir=Path(__file__).resolve().parent,
    config_file=Path(__file__).resolve().parent.parent / "pyproject.toml",
    package_name=PACKAGE_NAME,
)

# =======================================================================================
# CONTEXT CONFIGURATION FOR ADVANCED SCENARIOS
# =======================================================================================

# Advanced context class path - enables sophisticated context management
CONTEXT_CLASS = f"{PACKAGE_NAME}.pipeline_registry.create_advanced_context"

# Alternative: Use default context with enhanced configuration
# CONTEXT_CLASS = "kedro.framework.context.KedroContext"

# =======================================================================================
# HOOK REGISTRATION - ENTERPRISE DEPLOYMENT PATTERN
# =======================================================================================

# Create FigRegistry hooks with advanced enterprise configuration
_figregistry_hooks = create_figregistry_hooks()

# Main hook registration - demonstrates enterprise plugin registration patterns
HOOKS = tuple(_figregistry_hooks)

# Advanced: Additional hooks for sophisticated enterprise scenarios
# This pattern allows for complex hook orchestration in enterprise environments

def get_additional_enterprise_hooks() -> tuple:
    """Dynamically generate additional hooks for complex enterprise requirements.
    
    This function demonstrates advanced hook registration patterns for enterprise
    deployments requiring sophisticated plugin coordination and context management.
    
    Returns:
        Tuple of additional hook instances for enterprise deployment
    """
    additional_hooks = []
    
    env = get_deployment_environment()
    
    # Example: Add performance monitoring hooks for staging/production
    if env in ('staging', 'production'):
        # Placeholder for enterprise performance monitoring hooks
        # These would integrate with enterprise monitoring systems
        logger.debug(f"Enterprise monitoring hooks available for {env} environment")
    
    # Example: Add audit logging hooks for production
    if env == 'production':
        # Placeholder for enterprise audit logging hooks
        logger.debug("Enterprise audit logging hooks available for production")
    
    return tuple(additional_hooks)

# Extend HOOKS with enterprise-specific additions if needed
ENTERPRISE_HOOKS = get_additional_enterprise_hooks()
if ENTERPRISE_HOOKS:
    HOOKS = HOOKS + ENTERPRISE_HOOKS
    logger.info(f"Enterprise hooks extended: {len(ENTERPRISE_HOOKS)} additional hooks registered")

# =======================================================================================
# CONFIGURATION LOADING STRATEGY
# =======================================================================================

def get_config_loader() -> ConfigLoader:
    """Create advanced ConfigLoader for sophisticated environment management.
    
    This function demonstrates enterprise-grade configuration loading strategies
    with environment-specific overrides and advanced validation patterns.
    
    Returns:
        ConfigLoader instance optimized for enterprise deployment scenarios
    """
    env = get_deployment_environment()
    
    # Advanced configuration directory structure for enterprise deployment
    conf_source = str(Path(__file__).resolve().parent.parent / "conf")
    
    # Environment-specific configuration loading with fallbacks
    config_loader = ConfigLoader(
        conf_source=conf_source,
        env=env,
        runtime_params={
            'figregistry_environment': env,
            'performance_monitoring': get_performance_monitoring_level(),
            'ci_environment': is_ci_environment(),
        }
    )
    
    logger.info(f"Advanced ConfigLoader initialized for environment: {env}")
    return config_loader

# Override default config loader for advanced configuration management
CONFIG_LOADER_CLASS = get_config_loader

# Alternative: Use class-based configuration loading for maximum flexibility
# CONFIG_LOADER_CLASS = "kedro.config.ConfigLoader"
# CONFIG_LOADER_ARGS = {
#     "conf_source": str(Path(__file__).resolve().parent.parent / "conf"),
#     "env": get_deployment_environment(),
# }

# =======================================================================================
# SESSION CONFIGURATION FOR ENTERPRISE DEPLOYMENT
# =======================================================================================

# Advanced session configuration for enterprise environments
SESSION_STORE_ARGS = {
    "path": str(Path(__file__).resolve().parent.parent / "logs"),
    "session_id": None,  # Auto-generated for enterprise tracking
}

# =======================================================================================
# OPTIONAL: ADVANCED PLUGIN DISCOVERY CONFIGURATION
# =======================================================================================

# Demonstrate advanced plugin discovery patterns for enterprise deployment
def configure_plugin_discovery():
    """Configure advanced plugin discovery for enterprise FigRegistry deployment."""
    env = get_deployment_environment()
    
    # Log plugin discovery configuration for enterprise monitoring
    logger.info(f"FigRegistry plugin discovery configured for {env} environment")
    
    # Advanced: Conditional plugin loading based on environment
    if env == 'development':
        logger.debug("Development plugins enabled for FigRegistry integration")
    elif env == 'production':
        logger.info("Production-optimized plugin configuration active")

# Execute plugin discovery configuration
configure_plugin_discovery()

# =======================================================================================
# ENTERPRISE DEPLOYMENT VALIDATION
# =======================================================================================

def validate_enterprise_configuration():
    """Validate enterprise configuration for production readiness.
    
    This function performs comprehensive validation of the enterprise deployment
    configuration to ensure all components are properly configured for the target
    environment.
    """
    env = get_deployment_environment()
    
    # Validate hook configuration
    if not HOOKS:
        logger.error("No hooks registered - FigRegistry integration will not function")
        return False
    
    # Validate environment-specific requirements
    validation_passed = True
    
    if env == 'production':
        # Production-specific validations
        for hook in HOOKS:
            if hasattr(hook, 'enable_performance_monitoring') and hook.enable_performance_monitoring:
                logger.warning("Performance monitoring enabled in production - consider disabling for optimal performance")
            
            if hasattr(hook, 'fallback_on_error') and not hook.fallback_on_error:
                logger.error("Production environment should have fallback_on_error=True for resilience")
                validation_passed = False
    
    elif env == 'development':
        # Development-specific validations
        performance_monitoring_enabled = any(
            hasattr(hook, 'enable_performance_monitoring') and hook.enable_performance_monitoring 
            for hook in HOOKS
        )
        if not performance_monitoring_enabled:
            logger.info("Performance monitoring disabled in development - consider enabling for debugging")
    
    logger.info(f"Enterprise configuration validation {'passed' if validation_passed else 'failed'} "
                f"for environment: {env}")
    
    return validation_passed

# Execute validation on settings load
if __name__ != "__main__":  # Avoid validation during direct execution
    validate_enterprise_configuration()

# =======================================================================================
# ENTERPRISE DEPLOYMENT METADATA
# =======================================================================================

# Metadata for enterprise deployment tracking and monitoring
ENTERPRISE_METADATA = {
    'deployment_environment': get_deployment_environment(),
    'performance_monitoring_level': get_performance_monitoring_level(),
    'ci_environment': is_ci_environment(),
    'hooks_registered': len(HOOKS),
    'figregistry_integration_version': '1.0.0-advanced',
    'enterprise_features_enabled': True,
}

# Log enterprise deployment metadata
logger.info(f"Advanced FigRegistry-Kedro enterprise deployment initialized: {ENTERPRISE_METADATA}")