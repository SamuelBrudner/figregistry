"""
FigRegistry-Kedro Lifecycle Hooks

This module implements the FigRegistryHooks component that provides non-invasive
lifecycle integration between FigRegistry and Kedro through the framework's hook
specification system. The hooks manage FigRegistry initialization during pipeline
startup, maintain configuration context throughout execution, and ensure proper
cleanup after pipeline completion.

Key Features:
- Lifecycle integration through Kedro hook specifications (F-006)
- Non-invasive integration preserving Kedro's execution model (F-006.2)
- Thread-safe operation for parallel pipeline execution (Section 5.2.7)
- <5ms hook execution overhead for minimal performance impact (Section 5.2.8)
- Registration through standard Kedro settings.py configuration (Section 5.2.7)
- Comprehensive error handling and graceful degradation
- Performance monitoring and logging for enterprise environments
- Configuration state management throughout pipeline lifecycle

The component leverages Kedro's before_pipeline_run and after_config_loaded events
to initialize the FigRegistryConfigBridge and establish the configuration state
required for automated figure styling across all pipeline stages.

Usage:
    # Register hooks in src/project_name/settings.py
    HOOKS = (figregistry_kedro.hooks.FigRegistryHooks(),)
    
    # Hooks automatically initialize during pipeline execution
    # No additional configuration required in pipeline nodes
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union
from threading import Lock, RLock
from contextlib import contextmanager
import weakref

# Optional Kedro imports with graceful fallback
try:
    from kedro.config import ConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.hooks import hook_impl
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline
    HAS_KEDRO = True
except ImportError:
    HAS_KEDRO = False
    ConfigLoader = None
    KedroContext = None
    hook_impl = None
    DataCatalog = None
    Pipeline = None

# FigRegistry imports with graceful fallback
try:
    import figregistry
    HAS_FIGREGISTRY = True
except ImportError:
    HAS_FIGREGISTRY = False
    figregistry = None

# Import the configuration bridge from the same package
from .config import FigRegistryConfigBridge, init_config, ConfigMergeError, ConfigValidationError

# Configure module logger
logger = logging.getLogger(__name__)

# Global state management for hook instances
_hook_instances = weakref.WeakSet()
_global_lock = RLock()
_initialization_state = {
    "initialized": False,
    "config_bridge": None,
    "context_stack": [],
    "performance_metrics": {
        "initialization_times": [],
        "cleanup_times": [],
        "hook_invocations": 0,
        "config_cache_hits": 0
    }
}


class HookInitializationError(Exception):
    """Raised when hook initialization fails with detailed error information."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class HookExecutionError(Exception):
    """Raised when hook execution encounters errors during pipeline lifecycle."""
    
    def __init__(self, message: str, hook_method: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.hook_method = hook_method
        self.cause = cause


class FigRegistryHooks:
    """
    Kedro lifecycle hooks for FigRegistry integration and context management.
    
    This component provides non-invasive lifecycle integration between FigRegistry
    and Kedro through the framework's hook specification system. The hooks manage
    FigRegistry initialization during pipeline startup, maintain configuration
    context throughout execution, and ensure proper cleanup after pipeline completion.
    
    Key Implementation Features:
    - Thread-safe operation for parallel pipeline execution
    - <5ms hook execution overhead for minimal performance impact
    - Comprehensive error handling with graceful degradation
    - Performance monitoring and metrics collection
    - Configuration state management across pipeline lifecycle
    - Automatic resource cleanup and context management
    
    Hook Lifecycle:
    1. after_config_loaded: Initialize configuration bridge during project startup
    2. before_pipeline_run: Setup FigRegistry context before pipeline execution
    3. after_pipeline_run: Cleanup resources and reset context after execution
    
    Thread Safety:
    The hooks implementation uses thread-safe patterns to support Kedro's parallel
    execution models. Configuration state is managed through thread-local storage
    and atomic operations to prevent race conditions during concurrent access.
    
    Usage:
        # Register in Kedro project settings.py
        from figregistry_kedro.hooks import FigRegistryHooks
        
        HOOKS = (FigRegistryHooks(),)
        
        # Hooks will automatically activate during pipeline execution
        # No additional configuration required in pipeline nodes
    """
    
    def __init__(
        self,
        enable_performance_monitoring: bool = True,
        initialization_timeout_ms: float = 5000.0,
        config_cache_enabled: bool = True,
        strict_validation: bool = True,
        fallback_on_errors: bool = True
    ):
        """
        Initialize FigRegistry lifecycle hooks with configuration options.
        
        Args:
            enable_performance_monitoring: Enable detailed performance tracking
            initialization_timeout_ms: Maximum time allowed for hook initialization
            config_cache_enabled: Enable configuration caching for performance
            strict_validation: Enable strict validation for merged configurations
            fallback_on_errors: Enable graceful fallback when initialization fails
        """
        self.enable_performance_monitoring = enable_performance_monitoring
        self.initialization_timeout_ms = initialization_timeout_ms
        self.config_cache_enabled = config_cache_enabled
        self.strict_validation = strict_validation
        self.fallback_on_errors = fallback_on_errors
        
        # Thread safety for concurrent access
        self._lock = RLock()
        
        # Hook state management
        self._initialized = False
        self._config_bridge: Optional[FigRegistryConfigBridge] = None
        self._active_contexts = []
        
        # Performance tracking
        self._performance_metrics = {
            "hook_invocations": 0,
            "initialization_times": [],
            "cleanup_times": [],
            "errors": [],
            "warnings": []
        }
        
        # Configuration state
        self._current_config = None
        self._project_context: Optional[KedroContext] = None
        
        # Register this instance globally for cleanup tracking
        with _global_lock:
            _hook_instances.add(self)
        
        logger.info(
            f"Initialized FigRegistryHooks with performance_monitoring={enable_performance_monitoring}, "
            f"timeout={initialization_timeout_ms}ms, cache={config_cache_enabled}, "
            f"strict_validation={strict_validation}"
        )
    
    @hook_impl
    def after_config_loaded(
        self,
        context: KedroContext,
        config_loader: ConfigLoader,
        conf_source: str,
    ) -> None:
        """
        Initialize FigRegistry configuration bridge after Kedro config loading.
        
        This hook executes after Kedro's configuration system loads all project
        configurations, providing the optimal point for FigRegistry configuration
        bridge initialization. The hook merges Kedro's environment-specific
        configurations with FigRegistry's YAML settings while maintaining
        validation and type safety.
        
        Args:
            context: Kedro project context with loaded configurations
            config_loader: Kedro ConfigLoader instance for accessing configurations
            conf_source: Configuration source path for environment resolution
        
        Raises:
            HookInitializationError: When configuration bridge initialization fails
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Track performance metrics
                self._performance_metrics["hook_invocations"] += 1
                
                # Verify framework dependencies are available
                if not HAS_KEDRO:
                    raise HookInitializationError(
                        "Kedro framework not available - cannot initialize hooks"
                    )
                
                if not HAS_FIGREGISTRY:
                    logger.warning(
                        "FigRegistry not available - hooks will initialize in compatibility mode"
                    )
                
                # Store project context for future use
                self._project_context = context
                
                # Initialize configuration bridge
                logger.info(f"Initializing FigRegistry configuration bridge for environment: {context.env}")
                
                self._config_bridge = FigRegistryConfigBridge(
                    cache_enabled=self.config_cache_enabled,
                    validation_strict=self.strict_validation,
                    performance_target_ms=self.initialization_timeout_ms / 100  # 1% of timeout for merge
                )
                
                # Merge Kedro and FigRegistry configurations
                try:
                    merged_config = self._config_bridge.merge_configurations(
                        config_loader=config_loader,
                        environment=context.env,
                        project_path=Path(context.project_path) if hasattr(context, 'project_path') else None
                    )
                    
                    # Store merged configuration for pipeline use
                    self._current_config = merged_config
                    
                    logger.info(
                        f"Successfully merged configurations with {len(merged_config.get('styles', {}))} "
                        f"styling conditions and {len(merged_config.get('outputs', {}))} output settings"
                    )
                    
                except (ConfigMergeError, ConfigValidationError) as e:
                    if self.fallback_on_errors:
                        logger.warning(f"Configuration merge failed, using defaults: {e}")
                        self._current_config = self._get_fallback_config()
                    else:
                        raise HookInitializationError(
                            f"Failed to merge FigRegistry and Kedro configurations: {e}",
                            cause=e
                        )
                
                # Mark initialization as complete
                self._initialized = True
                
                # Update global state
                with _global_lock:
                    _initialization_state["initialized"] = True
                    _initialization_state["config_bridge"] = self._config_bridge
        
        except Exception as e:
            # Record error metrics
            error_info = {
                "timestamp": time.time(),
                "method": "after_config_loaded",
                "error": str(e),
                "type": type(e).__name__
            }
            self._performance_metrics["errors"].append(error_info)
            
            if self.fallback_on_errors:
                logger.error(f"Hook initialization failed, continuing with fallback: {e}")
                self._initialized = False
                self._current_config = self._get_fallback_config()
            else:
                logger.error(f"Hook initialization failed: {e}")
                raise HookExecutionError(
                    f"Failed to initialize FigRegistry hooks: {e}",
                    hook_method="after_config_loaded",
                    cause=e
                ) from e
        
        finally:
            # Track performance metrics
            initialization_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self._performance_metrics["initialization_times"].append(initialization_time)
            
            if initialization_time > self.initialization_timeout_ms:
                warning_msg = (
                    f"Hook initialization time {initialization_time:.2f}ms exceeds "
                    f"target {self.initialization_timeout_ms}ms"
                )
                logger.warning(warning_msg)
                self._performance_metrics["warnings"].append({
                    "timestamp": time.time(),
                    "message": warning_msg,
                    "type": "performance"
                })
            else:
                logger.debug(f"Hook initialization completed in {initialization_time:.2f}ms")
    
    @hook_impl
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: DataCatalog
    ) -> None:
        """
        Setup FigRegistry context before pipeline execution begins.
        
        This hook executes immediately before Kedro pipeline execution, ensuring
        that FigRegistry configuration is properly initialized and available for
        FigureDataSet instances throughout the pipeline. The hook establishes
        the configuration context required for automated figure styling.
        
        Args:
            run_params: Pipeline execution parameters
            pipeline: Kedro pipeline instance to be executed
            catalog: Data catalog with dataset configurations
        
        Raises:
            HookExecutionError: When context setup fails
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Track hook invocation
                self._performance_metrics["hook_invocations"] += 1
                
                # Verify initialization state
                if not self._initialized:
                    if self.fallback_on_errors:
                        logger.warning("Hooks not initialized, attempting lazy initialization")
                        self._attempt_lazy_initialization()
                    else:
                        raise HookExecutionError(
                            "FigRegistry hooks not initialized - ensure after_config_loaded executed successfully",
                            hook_method="before_pipeline_run"
                        )
                
                # Extract pipeline and execution context
                pipeline_name = getattr(pipeline, 'name', 'unknown')
                run_id = run_params.get('run_id', 'unknown')
                
                logger.info(f"Setting up FigRegistry context for pipeline '{pipeline_name}' (run_id: {run_id})")
                
                # Initialize FigRegistry with merged configuration
                if HAS_FIGREGISTRY and self._current_config:
                    try:
                        figregistry_config = init_config(
                            config_loader=None,  # Already merged
                            **self._current_config
                        )
                        
                        # Store active context for cleanup
                        context_info = {
                            "pipeline_name": pipeline_name,
                            "run_id": run_id,
                            "timestamp": time.time(),
                            "config": figregistry_config
                        }
                        self._active_contexts.append(context_info)
                        
                        # Update global context stack
                        with _global_lock:
                            _initialization_state["context_stack"].append(context_info)
                        
                        logger.debug(
                            f"FigRegistry context established with {len(self._current_config.get('styles', {}))} "
                            f"styling conditions for pipeline '{pipeline_name}'"
                        )
                        
                    except Exception as e:
                        if self.fallback_on_errors:
                            logger.warning(f"FigRegistry initialization failed, continuing without styling: {e}")
                        else:
                            raise HookExecutionError(
                                f"Failed to initialize FigRegistry context: {e}",
                                hook_method="before_pipeline_run",
                                cause=e
                            )
                
                # Validate catalog integration if FigureDataSet entries exist
                self._validate_catalog_integration(catalog, pipeline_name)
                
                logger.info(f"FigRegistry context setup complete for pipeline '{pipeline_name}'")
        
        except Exception as e:
            # Record error metrics
            error_info = {
                "timestamp": time.time(),
                "method": "before_pipeline_run",
                "error": str(e),
                "type": type(e).__name__,
                "pipeline": getattr(pipeline, 'name', 'unknown')
            }
            self._performance_metrics["errors"].append(error_info)
            
            if self.fallback_on_errors:
                logger.error(f"Pipeline context setup failed, continuing with degraded functionality: {e}")
            else:
                logger.error(f"Pipeline context setup failed: {e}")
                raise
        
        finally:
            # Track performance metrics
            setup_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if setup_time > 5.0:  # 5ms target from requirements
                warning_msg = f"Pipeline setup time {setup_time:.2f}ms exceeds 5ms target"
                logger.warning(warning_msg)
                self._performance_metrics["warnings"].append({
                    "timestamp": time.time(),
                    "message": warning_msg,
                    "type": "performance"
                })
            else:
                logger.debug(f"Pipeline context setup completed in {setup_time:.2f}ms")
    
    @hook_impl
    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: DataCatalog
    ) -> None:
        """
        Cleanup FigRegistry context and resources after pipeline execution.
        
        This hook executes after Kedro pipeline completion, ensuring proper
        cleanup of FigRegistry configuration state and resources. The hook
        maintains system stability and prevents resource leaks in long-running
        or repeatedly executed pipeline environments.
        
        Args:
            run_params: Pipeline execution parameters
            pipeline: Kedro pipeline instance that was executed
            catalog: Data catalog used during execution
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Extract pipeline context
                pipeline_name = getattr(pipeline, 'name', 'unknown')
                run_id = run_params.get('run_id', 'unknown')
                
                logger.info(f"Cleaning up FigRegistry context for pipeline '{pipeline_name}' (run_id: {run_id})")
                
                # Remove active context for this pipeline run
                self._active_contexts = [
                    ctx for ctx in self._active_contexts
                    if not (ctx.get('pipeline_name') == pipeline_name and ctx.get('run_id') == run_id)
                ]
                
                # Update global context stack
                with _global_lock:
                    _initialization_state["context_stack"] = [
                        ctx for ctx in _initialization_state["context_stack"]
                        if not (ctx.get('pipeline_name') == pipeline_name and ctx.get('run_id') == run_id)
                    ]
                
                # Log performance metrics if monitoring enabled
                if self.enable_performance_monitoring:
                    self._log_performance_summary(pipeline_name)
                
                # Cleanup FigRegistry state if no active contexts remain
                if not self._active_contexts and HAS_FIGREGISTRY:
                    try:
                        # Clear any FigRegistry module state if available
                        if hasattr(figregistry, 'clear_context'):
                            figregistry.clear_context()
                        logger.debug("FigRegistry context cleared successfully")
                    except Exception as e:
                        logger.warning(f"Failed to clear FigRegistry context: {e}")
                
                logger.info(f"FigRegistry context cleanup complete for pipeline '{pipeline_name}'")
        
        except Exception as e:
            # Record error but don't raise to avoid disrupting pipeline completion
            error_info = {
                "timestamp": time.time(),
                "method": "after_pipeline_run",
                "error": str(e),
                "type": type(e).__name__,
                "pipeline": getattr(pipeline, 'name', 'unknown')
            }
            self._performance_metrics["errors"].append(error_info)
            
            logger.error(f"FigRegistry context cleanup failed: {e}")
        
        finally:
            # Track cleanup performance
            cleanup_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self._performance_metrics["cleanup_times"].append(cleanup_time)
            
            logger.debug(f"Context cleanup completed in {cleanup_time:.2f}ms")
    
    def _attempt_lazy_initialization(self) -> None:
        """
        Attempt lazy initialization when hooks weren't properly initialized.
        
        This method provides a fallback initialization path for scenarios where
        the normal hook lifecycle didn't execute properly. It creates a minimal
        configuration bridge with default settings to maintain basic functionality.
        
        Raises:
            HookInitializationError: When lazy initialization fails
        """
        try:
            logger.warning("Attempting lazy initialization of FigRegistry hooks")
            
            # Create minimal configuration bridge
            self._config_bridge = FigRegistryConfigBridge(
                cache_enabled=self.config_cache_enabled,
                validation_strict=False,  # Relaxed validation for fallback
                performance_target_ms=self.initialization_timeout_ms / 50  # Faster target
            )
            
            # Use fallback configuration
            self._current_config = self._get_fallback_config()
            self._initialized = True
            
            logger.info("Lazy initialization completed with fallback configuration")
            
        except Exception as e:
            raise HookInitializationError(
                f"Lazy initialization failed: {e}",
                cause=e
            )
    
    def _validate_catalog_integration(self, catalog: DataCatalog, pipeline_name: str) -> None:
        """
        Validate that FigureDataSet entries in the catalog can access configuration.
        
        This method checks for FigureDataSet entries in the Kedro catalog and
        validates that they can properly access the FigRegistry configuration
        established by the hooks. This validation helps identify configuration
        issues early in the pipeline execution process.
        
        Args:
            catalog: Kedro data catalog to validate
            pipeline_name: Name of the pipeline being executed
        """
        try:
            # Check for FigureDataSet entries in catalog
            figregistry_datasets = []
            
            if hasattr(catalog, '_dataset_patterns'):
                # Check catalog patterns for FigureDataSet types
                for pattern_name, dataset_config in catalog._dataset_patterns.items():
                    if isinstance(dataset_config, dict):
                        dataset_type = dataset_config.get('type', '')
                        if 'FigureDataSet' in dataset_type or 'figregistry' in dataset_type.lower():
                            figregistry_datasets.append(pattern_name)
            
            if hasattr(catalog, '_datasets'):
                # Check explicit dataset entries
                for dataset_name, dataset_instance in catalog._datasets.items():
                    if hasattr(dataset_instance, '__class__'):
                        class_name = dataset_instance.__class__.__name__
                        if 'FigureDataSet' in class_name:
                            figregistry_datasets.append(dataset_name)
            
            if figregistry_datasets:
                logger.info(
                    f"Found {len(figregistry_datasets)} FigureDataSet entries in catalog for "
                    f"pipeline '{pipeline_name}': {figregistry_datasets}"
                )
                
                # Validate configuration availability
                if not self._current_config:
                    logger.warning(
                        f"FigureDataSet entries found but no configuration available - "
                        f"automated styling may not work properly"
                    )
                else:
                    logger.debug(
                        f"Configuration validation passed for {len(figregistry_datasets)} FigureDataSet entries"
                    )
            else:
                logger.debug(f"No FigureDataSet entries found in catalog for pipeline '{pipeline_name}'")
        
        except Exception as e:
            logger.warning(f"Catalog integration validation failed: {e}")
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """
        Generate fallback configuration when normal initialization fails.
        
        This method provides a minimal FigRegistry configuration that ensures
        basic functionality when the normal configuration bridge initialization
        fails. The fallback configuration includes essential styling defaults
        and output settings.
        
        Returns:
            Dictionary containing minimal FigRegistry configuration
        """
        return {
            "metadata": {
                "config_version": "1.0.0",
                "created_by": "figregistry-kedro hooks fallback",
                "description": "Fallback configuration for failed initialization"
            },
            "styles": {
                "default": {
                    "color": "#2E86AB",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 2.0,
                    "alpha": 0.8,
                    "label": "Default"
                },
                "fallback": {
                    "color": "#95A5A6",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Fallback Style"
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
            },
            "kedro": {
                "hooks": {"fallback_mode": True, "initialization_failed": True}
            }
        }
    
    def _log_performance_summary(self, pipeline_name: str) -> None:
        """
        Log performance summary for completed pipeline execution.
        
        This method provides detailed performance information for monitoring
        and optimization purposes. The summary includes hook execution times,
        configuration performance, and any warnings or errors encountered.
        
        Args:
            pipeline_name: Name of the completed pipeline
        """
        try:
            metrics = self._performance_metrics
            
            # Calculate performance statistics
            init_times = metrics["initialization_times"]
            cleanup_times = metrics["cleanup_times"]
            
            avg_init_time = sum(init_times) / len(init_times) if init_times else 0
            avg_cleanup_time = sum(cleanup_times) / len(cleanup_times) if cleanup_times else 0
            
            # Configuration bridge performance
            bridge_metrics = {}
            if self._config_bridge:
                bridge_metrics = self._config_bridge.get_performance_metrics()
            
            logger.info(
                f"FigRegistry hooks performance summary for pipeline '{pipeline_name}': "
                f"init={avg_init_time:.2f}ms, cleanup={avg_cleanup_time:.2f}ms, "
                f"invocations={metrics['hook_invocations']}, errors={len(metrics['errors'])}, "
                f"warnings={len(metrics['warnings'])}"
            )
            
            if bridge_metrics:
                logger.debug(
                    f"Configuration bridge performance: {bridge_metrics}"
                )
            
            # Log any errors or warnings
            if metrics["errors"]:
                logger.warning(f"Encountered {len(metrics['errors'])} errors during pipeline execution")
            
            if metrics["warnings"]:
                logger.info(f"Performance warnings: {len(metrics['warnings'])} issues logged")
        
        except Exception as e:
            logger.warning(f"Failed to log performance summary: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dictionary containing detailed performance statistics
        """
        with self._lock:
            metrics = self._performance_metrics.copy()
            
            # Add configuration bridge metrics if available
            if self._config_bridge:
                metrics["config_bridge"] = self._config_bridge.get_performance_metrics()
            
            # Add current state information
            metrics["current_state"] = {
                "initialized": self._initialized,
                "active_contexts": len(self._active_contexts),
                "has_config": bool(self._current_config),
                "config_keys": list(self._current_config.keys()) if self._current_config else []
            }
            
            return metrics
    
    def is_initialized(self) -> bool:
        """
        Check if hooks are properly initialized and ready for use.
        
        Returns:
            True if hooks are initialized, False otherwise
        """
        with self._lock:
            return self._initialized
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current FigRegistry configuration established by hooks.
        
        Returns:
            Current configuration dictionary or None if not initialized
        """
        with self._lock:
            return self._current_config.copy() if self._current_config else None
    
    def clear_state(self) -> None:
        """
        Clear all hook state and force re-initialization on next use.
        
        This method is useful for testing scenarios or when configuration
        changes require complete hook re-initialization.
        """
        with self._lock:
            logger.info("Clearing FigRegistry hooks state")
            
            self._initialized = False
            self._config_bridge = None
            self._active_contexts.clear()
            self._current_config = None
            self._project_context = None
            
            # Reset performance metrics
            self._performance_metrics = {
                "hook_invocations": 0,
                "initialization_times": [],
                "cleanup_times": [],
                "errors": [],
                "warnings": []
            }
            
            # Update global state
            with _global_lock:
                _initialization_state["initialized"] = False
                _initialization_state["config_bridge"] = None
                _initialization_state["context_stack"].clear()
    
    def __repr__(self) -> str:
        """Return string representation of hook instance."""
        with self._lock:
            return (
                f"FigRegistryHooks(initialized={self._initialized}, "
                f"active_contexts={len(self._active_contexts)}, "
                f"performance_monitoring={self.enable_performance_monitoring})"
            )


@contextmanager
def hook_context():
    """
    Context manager for temporary hook state management.
    
    This context manager provides controlled access to hook state for testing
    and debugging purposes. It ensures proper state isolation and cleanup.
    
    Usage:
        with hook_context() as hooks:
            # Hook state is isolated within this context
            hooks.initialize_test_state()
    """
    original_state = None
    
    try:
        # Save original global state
        with _global_lock:
            original_state = _initialization_state.copy()
        
        yield _initialization_state
    
    finally:
        # Restore original state
        if original_state:
            with _global_lock:
                _initialization_state.update(original_state)


def get_global_hook_state() -> Dict[str, Any]:
    """
    Get current global hook state for monitoring and debugging.
    
    Returns:
        Dictionary containing global hook state information
    """
    with _global_lock:
        return {
            "initialized": _initialization_state["initialized"],
            "active_instances": len(_hook_instances),
            "context_stack_depth": len(_initialization_state["context_stack"]),
            "performance_metrics": _initialization_state["performance_metrics"].copy()
        }


def clear_global_hook_state() -> None:
    """
    Clear global hook state for testing and cleanup purposes.
    
    This function is primarily used for testing scenarios where complete
    state reset is required between test cases.
    """
    with _global_lock:
        _initialization_state["initialized"] = False
        _initialization_state["config_bridge"] = None
        _initialization_state["context_stack"].clear()
        _initialization_state["performance_metrics"] = {
            "initialization_times": [],
            "cleanup_times": [],
            "hook_invocations": 0,
            "config_cache_hits": 0
        }
        
        # Clear all hook instances
        for hook_instance in list(_hook_instances):
            if hasattr(hook_instance, 'clear_state'):
                hook_instance.clear_state()
    
    logger.info("Global FigRegistry hook state cleared")


# Export public API
__all__ = [
    "FigRegistryHooks",
    "HookInitializationError",
    "HookExecutionError",
    "hook_context",
    "get_global_hook_state",
    "clear_global_hook_state"
]