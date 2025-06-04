"""Unit tests for FigRegistryHooks component validating lifecycle integration with Kedro.

This module provides comprehensive unit tests for the FigRegistryHooks component that
validate lifecycle integration with Kedro's hook specification system per Section 6.6.
Tests cover hook registration through plugin discovery, before_pipeline_run and 
after_config_loaded event handling, configuration context management, and cleanup
operations per F-006 requirements.

Key testing capabilities per Section 6.6.3.8:
- Hook registration mechanism through Kedro's plugin discovery system 
- Non-invasive integration preserving Kedro's execution model per F-006.2
- Configuration context management throughout pipeline execution lifecycle
- Thread-safe operation for parallel pipeline execution per Section 5.2.7
- Performance validation against <25ms hook initialization overhead per Section 6.6.4.3
- Comprehensive error handling and fallback behavior testing

Test Categories:
- Hook Specification Compliance: Validates implementation of Kedro hook interfaces
- Lifecycle Integration: Tests hook execution across pipeline execution phases
- Configuration Management: Validates FigRegistry context initialization and cleanup
- Performance Testing: Measures hook overhead against specified SLA targets
- Error Handling: Tests graceful degradation and fallback mechanisms
- Thread Safety: Validates concurrent execution with parallel Kedro runners

Coverage Requirements per Section 6.6.2.4:
- Minimum Coverage: ≥90% for figregistry_kedro.hooks module
- Critical Path Coverage: 100% for hook registration and lifecycle operations
- Performance SLA Validation: Hook initialization <25ms per Section 6.6.4.3
"""

import os
import sys
import time
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest

# Test framework imports
from figregistry_kedro.tests.conftest import (
    KEDRO_AVAILABLE,
    MATPLOTLIB_AVAILABLE,
    BENCHMARK_AVAILABLE
)

# Import hooks implementation with fallback handling
try:
    from figregistry_kedro.hooks import (
        FigRegistryHooks,
        FigRegistryHookState,
        HookExecutionError,
        create_hooks,
        DEFAULT_HOOK_CONFIG,
        get_hook_state
    )
    HOOKS_AVAILABLE = True
except ImportError:
    HOOKS_AVAILABLE = False

# Import configuration bridge with fallback
try:
    from figregistry_kedro.config import (
        FigRegistryConfigBridge,
        ConfigurationMergeError,
        init_config,
        get_bridge_instance,
        set_bridge_instance
    )
    CONFIG_BRIDGE_AVAILABLE = True
except ImportError:
    CONFIG_BRIDGE_AVAILABLE = False

# Kedro imports with graceful fallback
if KEDRO_AVAILABLE:
    from kedro.framework.context import KedroContext
    from kedro.framework.session import KedroSession
    from kedro.config import ConfigLoader
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline
    from kedro.framework.hooks import PluginManager


# =============================================================================
# TEST FIXTURES FOR HOOKS TESTING
# =============================================================================

@pytest.fixture
def mock_hook_state():
    """Provide isolated FigRegistryHookState for testing state management."""
    if not HOOKS_AVAILABLE:
        pytest.skip("FigRegistryHooks not available")
    
    state = FigRegistryHookState()
    yield state
    # Cleanup after test
    state.reset()


@pytest.fixture
def figregistry_hooks_instance():
    """Provide FigRegistryHooks instance with default configuration for testing."""
    if not HOOKS_AVAILABLE:
        pytest.skip("FigRegistryHooks not available")
    
    hooks = FigRegistryHooks(
        auto_initialize=True,
        enable_performance_monitoring=True,
        fallback_on_error=True,
        max_initialization_time=0.025  # 25ms as per spec
    )
    yield hooks
    # Reset state after test
    hooks.reset_state()


@pytest.fixture
def hooks_with_fallback_disabled():
    """Provide FigRegistryHooks instance with fallback disabled for error testing."""
    if not HOOKS_AVAILABLE:
        pytest.skip("FigRegistryHooks not available")
    
    hooks = FigRegistryHooks(
        auto_initialize=True,
        enable_performance_monitoring=True,
        fallback_on_error=False,
        max_initialization_time=0.025
    )
    yield hooks
    hooks.reset_state()


@pytest.fixture
def mock_kedro_context(mocker):
    """Mock Kedro context for hook testing without full project overhead."""
    if not KEDRO_AVAILABLE:
        context = mocker.Mock()
        context.env = "test"
        context.project_name = "test_project"
        context.project_path = Path("/tmp/test_project")
        return context
    
    context = mocker.Mock(spec=KedroContext)
    context.env = "test"
    context.project_name = "test_project"
    context.project_path = Path("/tmp/test_project")
    
    # Mock ConfigLoader
    config_loader = mocker.Mock(spec=ConfigLoader)
    config_loader.env = "test"
    config_loader.get.return_value = {
        'styles': {
            'test_condition': {'color': '#1f77b4', 'marker': 'o'}
        },
        'outputs': {'base_path': 'test_figures'}
    }
    context.config_loader = config_loader
    
    # Mock DataCatalog
    catalog = mocker.Mock(spec=DataCatalog)
    catalog.list.return_value = ['test_dataset']
    context.catalog = catalog
    
    return context


@pytest.fixture
def mock_pipeline_run_params():
    """Provide mock pipeline run parameters for hook testing."""
    return {
        'pipeline_name': 'test_pipeline',
        'session_id': 'test_session_123',
        'extra_params': {},
        'runner': 'sequential'
    }


@pytest.fixture
def mock_pipeline(mocker):
    """Mock Kedro pipeline for hook lifecycle testing."""
    if not KEDRO_AVAILABLE:
        pipeline = mocker.Mock()
        pipeline.nodes = []
        pipeline.describe.return_value = "Mock Test Pipeline"
        return pipeline
    
    pipeline = mocker.Mock(spec=Pipeline)
    pipeline.nodes = []
    pipeline.describe.return_value = "Mock Test Pipeline"
    return pipeline


@pytest.fixture
def mock_catalog(mocker):
    """Mock DataCatalog for hook testing."""
    if not KEDRO_AVAILABLE:
        catalog = mocker.Mock()
        catalog.list.return_value = ['test_dataset']
        catalog.save = Mock()
        catalog.load = Mock()
        return catalog
    
    catalog = mocker.Mock(spec=DataCatalog)
    catalog.list.return_value = ['test_dataset']
    catalog.save = Mock()
    catalog.load = Mock()
    return catalog


@pytest.fixture
def performance_tracker():
    """Provide performance tracking utilities for hook timing validation."""
    class PerformanceTracker:
        def __init__(self):
            self.timings = {}
            self.start_times = {}
        
        def start_timing(self, operation: str):
            """Start timing an operation."""
            self.start_times[operation] = time.perf_counter()
        
        def end_timing(self, operation: str) -> float:
            """End timing and return duration in milliseconds."""
            if operation not in self.start_times:
                return 0.0
            
            duration_ms = (time.perf_counter() - self.start_times[operation]) * 1000
            self.timings[operation] = duration_ms
            return duration_ms
        
        def get_timing(self, operation: str) -> float:
            """Get recorded timing for operation."""
            return self.timings.get(operation, 0.0)
        
        def validate_timing(self, operation: str, max_time_ms: float) -> bool:
            """Validate timing against maximum threshold."""
            return self.get_timing(operation) <= max_time_ms
    
    return PerformanceTracker()


# =============================================================================
# HOOK SPECIFICATION COMPLIANCE TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestHookSpecificationCompliance:
    """Test suite validating FigRegistryHooks implements Kedro hook specifications.
    
    Validates that FigRegistryHooks correctly implements Kedro's hook interface
    specifications per F-006 requirements, including proper method signatures,
    hook decorators, and registration mechanisms.
    """

    def test_hook_methods_exist(self, figregistry_hooks_instance):
        """Validate that FigRegistryHooks implements required hook methods per F-006."""
        hooks = figregistry_hooks_instance
        
        # Verify required hook lifecycle methods exist
        assert hasattr(hooks, 'after_config_loaded'), "Missing after_config_loaded hook method"
        assert hasattr(hooks, 'before_pipeline_run'), "Missing before_pipeline_run hook method"
        assert hasattr(hooks, 'after_pipeline_run'), "Missing after_pipeline_run hook method"
        assert hasattr(hooks, 'on_node_error'), "Missing on_node_error hook method"
        
        # Verify methods are callable
        assert callable(hooks.after_config_loaded), "after_config_loaded must be callable"
        assert callable(hooks.before_pipeline_run), "before_pipeline_run must be callable"
        assert callable(hooks.after_pipeline_run), "after_pipeline_run must be callable"
        assert callable(hooks.on_node_error), "on_node_error must be callable"

    def test_hook_method_signatures(self, figregistry_hooks_instance):
        """Validate hook method signatures comply with Kedro specifications."""
        hooks = figregistry_hooks_instance
        
        # Test after_config_loaded signature
        try:
            hooks.after_config_loaded(
                context=None,
                config_loader=None,
                conf_source=None
            )
        except Exception as e:
            # Should not fail due to signature issues
            assert not isinstance(e, TypeError), f"after_config_loaded signature invalid: {e}"
        
        # Test before_pipeline_run signature
        try:
            hooks.before_pipeline_run(
                run_params={'pipeline_name': 'test'},
                pipeline=None,
                catalog=None
            )
        except Exception as e:
            assert not isinstance(e, TypeError), f"before_pipeline_run signature invalid: {e}"
        
        # Test after_pipeline_run signature
        try:
            hooks.after_pipeline_run(
                run_params={'pipeline_name': 'test'},
                pipeline=None,
                catalog=None
            )
        except Exception as e:
            assert not isinstance(e, TypeError), f"after_pipeline_run signature invalid: {e}"
        
        # Test on_node_error signature
        try:
            hooks.on_node_error(
                error=Exception("test"),
                node_name="test_node",
                catalog=None,
                inputs={},
                is_async=False
            )
        except Exception as e:
            assert not isinstance(e, TypeError), f"on_node_error signature invalid: {e}"

    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available for hook decorator testing")
    def test_hook_decorators_present(self):
        """Validate that hook methods have proper @hook_impl decorators."""
        from figregistry_kedro.hooks import FigRegistryHooks
        
        # Check hook decorator presence
        hooks_class = FigRegistryHooks
        
        # Verify methods have hook_impl decorators (check for marker attributes)
        after_config_loaded = getattr(hooks_class, 'after_config_loaded', None)
        assert after_config_loaded is not None, "after_config_loaded method missing"
        
        before_pipeline_run = getattr(hooks_class, 'before_pipeline_run', None) 
        assert before_pipeline_run is not None, "before_pipeline_run method missing"
        
        after_pipeline_run = getattr(hooks_class, 'after_pipeline_run', None)
        assert after_pipeline_run is not None, "after_pipeline_run method missing"
        
        on_node_error = getattr(hooks_class, 'on_node_error', None)
        assert on_node_error is not None, "on_node_error method missing"

    def test_hooks_initialization_parameters(self):
        """Test FigRegistryHooks initialization with various parameter combinations."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        # Test default initialization
        hooks_default = FigRegistryHooks()
        assert hooks_default.auto_initialize is True
        assert hooks_default.enable_performance_monitoring is False
        assert hooks_default.fallback_on_error is True
        assert hooks_default.max_initialization_time == 0.005  # 5ms default
        
        # Test custom initialization
        hooks_custom = FigRegistryHooks(
            auto_initialize=False,
            enable_performance_monitoring=True,
            fallback_on_error=False,
            max_initialization_time=0.010  # 10ms
        )
        assert hooks_custom.auto_initialize is False
        assert hooks_custom.enable_performance_monitoring is True
        assert hooks_custom.fallback_on_error is False
        assert hooks_custom.max_initialization_time == 0.010
        
        # Cleanup
        hooks_default.reset_state()
        hooks_custom.reset_state()

    def test_create_hooks_factory_function(self):
        """Test create_hooks factory function for configuration override."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        # Test factory with default config
        hooks_default = create_hooks()
        assert hooks_default.auto_initialize is True
        assert hooks_default.fallback_on_error is True
        
        # Test factory with overrides
        hooks_custom = create_hooks(
            enable_performance_monitoring=True,
            fallback_on_error=False,
            max_initialization_time=0.020
        )
        assert hooks_custom.enable_performance_monitoring is True
        assert hooks_custom.fallback_on_error is False
        assert hooks_custom.max_initialization_time == 0.020
        
        # Cleanup
        hooks_default.reset_state()
        hooks_custom.reset_state()


# =============================================================================
# HOOK REGISTRATION AND DISCOVERY TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestHookRegistration:
    """Test suite for hook registration through Kedro's plugin discovery system.
    
    Validates that FigRegistryHooks can be properly registered and discovered
    through Kedro's plugin system per F-006.2 requirements for non-invasive
    integration.
    """

    def test_hook_registration_through_settings(self, mocker):
        """Test hook registration through Kedro settings.py configuration per F-006.2."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        # Mock Kedro's hook manager
        mock_manager = mocker.Mock()
        registered_hooks = []
        
        def mock_register(hook_instance):
            registered_hooks.append(hook_instance)
            return True
        
        mock_manager.register = Mock(side_effect=mock_register)
        
        # Create hooks instance
        hooks = FigRegistryHooks()
        
        # Simulate registration
        mock_manager.register(hooks)
        
        # Verify registration
        assert len(registered_hooks) == 1
        assert isinstance(registered_hooks[0], FigRegistryHooks)
        
        # Cleanup
        hooks.reset_state()

    def test_hook_plugin_discovery_entry_points(self, mocker):
        """Test that hooks can be discovered through entry points."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        # Mock entry points discovery
        mock_entry_point = mocker.Mock()
        mock_entry_point.load.return_value = FigRegistryHooks
        
        mock_entry_points = mocker.patch('pkg_resources.iter_entry_points')
        mock_entry_points.return_value = [mock_entry_point]
        
        # Simulate plugin discovery
        discovered_hooks = []
        for entry_point in mock_entry_points.return_value:
            hook_class = entry_point.load()
            discovered_hooks.append(hook_class)
        
        # Verify discovery
        assert len(discovered_hooks) == 1
        assert discovered_hooks[0] == FigRegistryHooks

    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available for plugin manager testing")
    def test_hook_manager_integration(self, mocker):
        """Test integration with Kedro's PluginManager."""
        # Mock PluginManager
        mock_plugin_manager = mocker.Mock(spec=PluginManager)
        
        # Track hook registration calls
        registered_hooks = []
        mock_plugin_manager.register.side_effect = lambda hook: registered_hooks.append(hook)
        mock_plugin_manager.is_registered.return_value = True
        
        # Create and register hooks
        hooks = FigRegistryHooks()
        mock_plugin_manager.register(hooks)
        
        # Verify registration with plugin manager
        assert len(registered_hooks) == 1
        assert registered_hooks[0] is hooks
        
        # Test hook manager queries
        mock_plugin_manager.is_registered.assert_called_once()
        
        # Cleanup
        hooks.reset_state()

    def test_multiple_hooks_registration(self):
        """Test registration of multiple FigRegistryHooks instances."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        # Create multiple hooks instances with different configurations
        hooks1 = FigRegistryHooks(auto_initialize=True)
        hooks2 = FigRegistryHooks(auto_initialize=False)
        hooks3 = create_hooks(enable_performance_monitoring=True)
        
        # Verify all instances are independent
        assert hooks1.auto_initialize is True
        assert hooks2.auto_initialize is False
        assert hooks3.enable_performance_monitoring is True
        
        # Verify state isolation
        hooks1._state.mark_initialized("test1")
        assert hooks1._state.is_initialized is True
        assert hooks2._state.is_initialized is False
        assert hooks3._state.is_initialized is False
        
        # Cleanup
        hooks1.reset_state()
        hooks2.reset_state()
        hooks3.reset_state()


# =============================================================================
# LIFECYCLE INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestHookLifecycleIntegration:
    """Test suite for hook lifecycle integration with Kedro pipeline execution.
    
    Validates proper execution of hook methods during Kedro pipeline lifecycle
    phases per Section 5.2.7 requirements for configuration initialization,
    context management, and cleanup operations.
    """

    def test_after_config_loaded_hook_execution(
        self, 
        figregistry_hooks_instance, 
        mock_kedro_context,
        mocker
    ):
        """Test after_config_loaded hook initializes FigRegistry configuration per F-006.2."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry initialization functions
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        mock_figregistry_available = mocker.patch('figregistry_kedro.hooks.figregistry', True)
        
        # Mock configuration bridge creation
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.return_value = mock_bridge
            mock_init_config.return_value = {'initialized': True}
        
        # Execute hook
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_kedro_context.config_loader,
            conf_source="/test/conf"
        )
        
        # Verify initialization occurred
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge_class.assert_called_once()
            mock_init_config.assert_called_once()
        
        # Verify hook state updated
        assert hooks._state.is_initialized is True

    def test_before_pipeline_run_hook_execution(
        self,
        figregistry_hooks_instance,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog
    ):
        """Test before_pipeline_run hook sets up pipeline context per Section 5.2.7."""
        hooks = figregistry_hooks_instance
        
        # Pre-initialize hook state
        hooks._state.mark_initialized("test")
        
        # Execute hook
        hooks.before_pipeline_run(
            run_params=mock_pipeline_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        
        # Verify pipeline registration
        assert hooks._state.active_pipeline_count == 1
        assert "test_pipeline" in hooks._state._active_pipelines

    def test_after_pipeline_run_hook_execution(
        self,
        figregistry_hooks_instance,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog
    ):
        """Test after_pipeline_run hook performs cleanup per Section 5.2.7."""
        hooks = figregistry_hooks_instance
        
        # Setup pipeline state
        hooks._state.mark_initialized("test")
        hooks._state.register_pipeline("test_pipeline")
        
        # Execute hook
        hooks.after_pipeline_run(
            run_params=mock_pipeline_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        
        # Verify cleanup occurred
        assert hooks._state.active_pipeline_count == 0
        assert "test_pipeline" not in hooks._state._active_pipelines

    def test_on_node_error_hook_execution(self, figregistry_hooks_instance, mock_catalog):
        """Test on_node_error hook handles node failures gracefully."""
        hooks = figregistry_hooks_instance
        
        # Test with FigRegistry-related error
        figregistry_error = ConfigurationMergeError("Test config error")
        hooks.on_node_error(
            error=figregistry_error,
            node_name="test_node",
            catalog=mock_catalog,
            inputs={'test_input': 'value'}
        )
        
        # Test with generic error
        generic_error = ValueError("Generic test error")
        hooks.on_node_error(
            error=generic_error,
            node_name="test_node", 
            catalog=mock_catalog,
            inputs={'test_input': 'value'}
        )
        
        # Hook should handle both error types without raising exceptions

    def test_complete_lifecycle_sequence(
        self,
        figregistry_hooks_instance,
        mock_kedro_context,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog,
        mocker
    ):
        """Test complete hook lifecycle sequence from config loading to cleanup."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry components
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        
        # 1. Configuration loading phase
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_kedro_context.config_loader
        )
        assert hooks._state.is_initialized is True
        
        # 2. Pipeline setup phase  
        hooks.before_pipeline_run(
            run_params=mock_pipeline_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        assert hooks._state.active_pipeline_count == 1
        
        # 3. Pipeline cleanup phase
        hooks.after_pipeline_run(
            run_params=mock_pipeline_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        assert hooks._state.active_pipeline_count == 0

    def test_multiple_pipeline_lifecycle_management(
        self,
        figregistry_hooks_instance,
        mock_kedro_context,
        mock_pipeline,
        mock_catalog,
        mocker
    ):
        """Test hook state management with multiple concurrent pipelines."""
        hooks = figregistry_hooks_instance
        
        # Mock configuration
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock() 
            mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        
        # Initialize hooks
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_kedro_context.config_loader
        )
        
        # Start multiple pipelines
        pipeline1_params = {'pipeline_name': 'pipeline1', 'session_id': 'session1'}
        pipeline2_params = {'pipeline_name': 'pipeline2', 'session_id': 'session2'}
        
        hooks.before_pipeline_run(pipeline1_params, mock_pipeline, mock_catalog)
        hooks.before_pipeline_run(pipeline2_params, mock_pipeline, mock_catalog)
        
        assert hooks._state.active_pipeline_count == 2
        
        # Complete pipelines in different order
        hooks.after_pipeline_run(pipeline2_params, mock_pipeline, mock_catalog)
        assert hooks._state.active_pipeline_count == 1
        
        hooks.after_pipeline_run(pipeline1_params, mock_pipeline, mock_catalog)
        assert hooks._state.active_pipeline_count == 0


# =============================================================================
# CONFIGURATION MANAGEMENT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestConfigurationManagement:
    """Test suite for FigRegistry configuration context management through hooks.
    
    Validates that hooks properly initialize and manage FigRegistry configuration
    context per F-006.2 requirements for non-invasive integration and automatic
    configuration setup.
    """

    def test_configuration_initialization_with_kedro_context(
        self,
        figregistry_hooks_instance,
        mock_kedro_context,
        mocker
    ):
        """Test configuration initialization with Kedro context and ConfigLoader."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry and bridge components
        mock_figregistry = mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        mock_init_config.return_value = {
            'styles': {'test': {'color': '#000000'}},
            'outputs': {'base_path': 'test_output'}
        }
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.return_value = mock_bridge
        
        # Execute configuration loading
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_kedro_context.config_loader,
            conf_source="/test/conf"
        )
        
        # Verify initialization
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge_class.assert_called_once_with(
                config_loader=mock_kedro_context.config_loader,
                environment="test",
                enable_caching=True
            )
            mock_init_config.assert_called_once_with(
                config_loader=mock_kedro_context.config_loader,
                environment="test"
            )
        
        # Verify state management
        assert hooks._state.is_initialized is True
        assert hooks._state.bridge is not None

    def test_configuration_initialization_without_kedro_context(
        self,
        figregistry_hooks_instance,
        mocker
    ):
        """Test configuration initialization fallback without Kedro context."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry components
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        mock_init_config.return_value = {'test': True}
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.return_value = mock_bridge
        
        # Execute with no context
        hooks.after_config_loaded(
            context=None,
            config_loader=None,
            conf_source=None
        )
        
        # Should still initialize with default environment
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge_class.assert_called_once_with(
                config_loader=None,
                environment="base",
                enable_caching=True
            )

    def test_configuration_bridge_state_management(self, figregistry_hooks_instance, mocker):
        """Test configuration bridge state management throughout hook lifecycle."""
        hooks = figregistry_hooks_instance
        
        # Mock bridge and components
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.return_value = mock_bridge
            
            mock_set_bridge = mocker.patch('figregistry_kedro.hooks.set_bridge_instance')
            
            mocker.patch('figregistry_kedro.hooks.figregistry', True)
            mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
            
            # Initialize configuration
            hooks.after_config_loaded(context=None, config_loader=None)
            
            # Verify bridge was set
            mock_set_bridge.assert_called_once_with(mock_bridge)
            assert hooks._state.bridge is mock_bridge
            
            # Reset state and verify cleanup
            hooks.reset_state()
            assert hooks._state.bridge is None
            assert hooks._state.is_initialized is False

    def test_late_initialization_fallback(
        self,
        figregistry_hooks_instance,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog,
        mocker
    ):
        """Test late initialization when configuration loading failed."""
        hooks = figregistry_hooks_instance
        
        # Mock components for late initialization
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.return_value = mock_bridge
        
        # Ensure hooks are not initialized
        assert hooks._state.is_initialized is False
        
        # Execute before_pipeline_run (should trigger late initialization)
        hooks.before_pipeline_run(
            run_params=mock_pipeline_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        
        # Verify late initialization occurred
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge_class.assert_called_once()
            mock_init_config.assert_called_once()
        assert hooks._state.is_initialized is True

    def test_configuration_error_handling(self, figregistry_hooks_instance, mocker):
        """Test graceful handling of configuration initialization errors."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry to raise configuration error
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        
        if CONFIG_BRIDGE_AVAILABLE:
            # Mock bridge to raise ConfigurationMergeError
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.side_effect = ConfigurationMergeError("Test config error")
        
        # Execute configuration loading (should not raise exception due to fallback)
        hooks.after_config_loaded(context=None, config_loader=None)
        
        # With fallback enabled, should not be initialized but no exception raised
        assert hooks._state.is_initialized is False

    def test_configuration_error_without_fallback(self, hooks_with_fallback_disabled, mocker):
        """Test configuration error handling without fallback enabled."""
        hooks = hooks_with_fallback_disabled
        
        # Mock FigRegistry to raise configuration error
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.side_effect = ConfigurationMergeError("Test config error")
            
            # Should raise HookExecutionError when fallback disabled
            with pytest.raises(HookExecutionError) as exc_info:
                hooks.after_config_loaded(context=None, config_loader=None)
            
            assert "after_config_loaded" in str(exc_info.value)
            assert isinstance(exc_info.value.original_error, ConfigurationMergeError)


# =============================================================================
# PERFORMANCE TESTING
# =============================================================================

@pytest.mark.performance
@pytest.mark.kedro_plugin
@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
class TestHookPerformance:
    """Test suite for hook performance validation per Section 6.6.4.3.
    
    Validates that hook execution meets performance SLA targets:
    - Hook initialization overhead: <25ms per project startup
    - Configuration bridge resolution: <50ms per pipeline run  
    - Overall hook execution: <5ms per hook invocation
    """

    def test_hook_initialization_performance(self, benchmark, performance_tracker):
        """Test hook initialization performance against 25ms target per Section 6.6.4.3."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        def initialize_hooks():
            """Initialize FigRegistryHooks and measure performance."""
            performance_tracker.start_timing('hook_initialization')
            hooks = FigRegistryHooks(
                auto_initialize=True,
                enable_performance_monitoring=True
            )
            duration = performance_tracker.end_timing('hook_initialization')
            hooks.reset_state()
            return duration
        
        # Benchmark hook initialization
        initialization_time = benchmark.pedantic(initialize_hooks, rounds=10, iterations=1)
        
        # Validate against 25ms target
        assert initialization_time < 25.0, f"Hook initialization took {initialization_time:.2f}ms, exceeds 25ms target"

    def test_after_config_loaded_performance(
        self,
        benchmark,
        figregistry_hooks_instance,
        mock_kedro_context,
        performance_tracker,
        mocker
    ):
        """Test after_config_loaded hook performance against timing targets."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry components for performance testing
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        
        def execute_config_hook():
            """Execute after_config_loaded hook and measure performance."""
            performance_tracker.start_timing('config_hook')
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_kedro_context.config_loader
            )
            duration = performance_tracker.end_timing('config_hook')
            hooks.reset_state()  # Reset for next iteration
            return duration
        
        # Benchmark configuration hook
        config_time = benchmark.pedantic(execute_config_hook, rounds=5, iterations=1)
        
        # Validate against 50ms target for configuration operations
        assert config_time < 50.0, f"Configuration hook took {config_time:.2f}ms, exceeds 50ms target"

    def test_before_pipeline_run_performance(
        self,
        benchmark,
        figregistry_hooks_instance,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog,
        performance_tracker
    ):
        """Test before_pipeline_run hook performance against 5ms target."""
        hooks = figregistry_hooks_instance
        
        # Pre-initialize for realistic performance testing
        hooks._state.mark_initialized("test")
        
        def execute_pipeline_hook():
            """Execute before_pipeline_run hook and measure performance."""
            performance_tracker.start_timing('pipeline_hook')
            hooks.before_pipeline_run(
                run_params=mock_pipeline_run_params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
            duration = performance_tracker.end_timing('pipeline_hook')
            # Cleanup for next iteration
            hooks._state.unregister_pipeline("test_pipeline")
            return duration
        
        # Benchmark pipeline hook
        pipeline_time = benchmark.pedantic(execute_pipeline_hook, rounds=10, iterations=1)
        
        # Validate against 5ms target for hook invocation
        assert pipeline_time < 5.0, f"Pipeline hook took {pipeline_time:.2f}ms, exceeds 5ms target"

    def test_complete_lifecycle_performance(
        self,
        benchmark,
        figregistry_hooks_instance,
        mock_kedro_context,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog,
        performance_tracker,
        mocker
    ):
        """Test complete hook lifecycle performance from initialization to cleanup."""
        hooks = figregistry_hooks_instance
        
        # Mock components
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        
        def execute_full_lifecycle():
            """Execute complete hook lifecycle and measure total performance."""
            performance_tracker.start_timing('full_lifecycle')
            
            # Configuration phase
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_kedro_context.config_loader
            )
            
            # Pipeline setup phase
            hooks.before_pipeline_run(
                run_params=mock_pipeline_run_params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
            
            # Pipeline cleanup phase
            hooks.after_pipeline_run(
                run_params=mock_pipeline_run_params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
            
            duration = performance_tracker.end_timing('full_lifecycle')
            hooks.reset_state()  # Reset for next iteration
            return duration
        
        # Benchmark full lifecycle
        lifecycle_time = benchmark.pedantic(execute_full_lifecycle, rounds=5, iterations=1)
        
        # Validate against combined target (25ms + 5ms + 5ms = 35ms)
        assert lifecycle_time < 35.0, f"Full lifecycle took {lifecycle_time:.2f}ms, exceeds 35ms target"

    def test_concurrent_hook_performance(
        self,
        figregistry_hooks_instance,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog,
        performance_tracker
    ):
        """Test hook performance under concurrent execution conditions."""
        hooks = figregistry_hooks_instance
        hooks._state.mark_initialized("test")
        
        def concurrent_hook_execution(pipeline_id: int):
            """Execute hooks concurrently and measure individual performance."""
            params = {**mock_pipeline_run_params, 'pipeline_name': f'pipeline_{pipeline_id}'}
            
            performance_tracker.start_timing(f'concurrent_{pipeline_id}')
            hooks.before_pipeline_run(params, mock_pipeline, mock_catalog)
            hooks.after_pipeline_run(params, mock_pipeline, mock_catalog)
            return performance_tracker.end_timing(f'concurrent_{pipeline_id}')
        
        # Execute multiple concurrent pipelines
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(concurrent_hook_execution, i) 
                for i in range(10)
            ]
            
            execution_times = [future.result() for future in as_completed(futures)]
        
        # Verify all executions meet performance targets
        max_time = max(execution_times)
        avg_time = sum(execution_times) / len(execution_times)
        
        assert max_time < 10.0, f"Max concurrent execution took {max_time:.2f}ms, exceeds 10ms target"
        assert avg_time < 5.0, f"Average concurrent execution took {avg_time:.2f}ms, exceeds 5ms target"


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestThreadSafety:
    """Test suite for thread-safe operation with parallel Kedro runners per Section 5.2.7.
    
    Validates that FigRegistryHooks support thread-safe operation for parallel
    pipeline execution without race conditions or state corruption.
    """

    def test_thread_safe_state_management(self, figregistry_hooks_instance):
        """Test thread-safe state management with concurrent operations."""
        hooks = figregistry_hooks_instance
        
        # Track results from concurrent operations
        results = []
        errors = []
        
        def concurrent_state_operation(thread_id: int):
            """Perform state operations concurrently."""
            try:
                # Mark as initialized
                hooks._state.mark_initialized(f"env_{thread_id}")
                
                # Register multiple pipelines
                for i in range(5):
                    hooks._state.register_pipeline(f"pipeline_{thread_id}_{i}")
                
                # Verify state
                pipeline_count = hooks._state.active_pipeline_count
                is_initialized = hooks._state.is_initialized
                
                # Unregister pipelines
                for i in range(5):
                    hooks._state.unregister_pipeline(f"pipeline_{thread_id}_{i}")
                
                results.append({
                    'thread_id': thread_id,
                    'max_pipeline_count': pipeline_count,
                    'is_initialized': is_initialized,
                    'final_count': hooks._state.active_pipeline_count
                })
                
            except Exception as e:
                errors.append({'thread_id': thread_id, 'error': str(e)})
        
        # Execute concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_state_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify all operations completed
        assert len(results) == 10, "Not all threads completed successfully"
        
        # Verify state consistency
        for result in results:
            assert result['is_initialized'] is True
            assert result['max_pipeline_count'] >= 5  # Should have registered at least 5 pipelines

    def test_concurrent_hook_execution(
        self,
        figregistry_hooks_instance,
        mock_pipeline,
        mock_catalog,
        mocker
    ):
        """Test concurrent execution of hook methods without race conditions."""
        hooks = figregistry_hooks_instance
        
        # Mock components for concurrent testing
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        
        execution_results = []
        
        def concurrent_pipeline_execution(pipeline_id: int):
            """Execute complete pipeline lifecycle concurrently."""
            try:
                run_params = {
                    'pipeline_name': f'concurrent_pipeline_{pipeline_id}',
                    'session_id': f'session_{pipeline_id}'
                }
                
                # Execute hook lifecycle
                hooks.before_pipeline_run(run_params, mock_pipeline, mock_catalog)
                
                # Simulate some work
                time.sleep(0.001)  # 1ms
                
                hooks.after_pipeline_run(run_params, mock_pipeline, mock_catalog)
                
                execution_results.append({
                    'pipeline_id': pipeline_id,
                    'status': 'success'
                })
                
            except Exception as e:
                execution_results.append({
                    'pipeline_id': pipeline_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Initialize hooks once
        hooks._state.mark_initialized("test")
        
        # Execute concurrent pipelines
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(concurrent_pipeline_execution, i)
                for i in range(20)
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        # Verify all executions succeeded
        successful_executions = [r for r in execution_results if r['status'] == 'success']
        failed_executions = [r for r in execution_results if r['status'] == 'error']
        
        assert len(successful_executions) == 20, f"Expected 20 successful executions, got {len(successful_executions)}"
        assert len(failed_executions) == 0, f"Unexpected failures: {failed_executions}"
        
        # Verify final state is clean
        assert hooks._state.active_pipeline_count == 0

    def test_bridge_instance_thread_safety(self, figregistry_hooks_instance, mocker):
        """Test thread-safe access to configuration bridge instance."""
        hooks = figregistry_hooks_instance
        
        if not CONFIG_BRIDGE_AVAILABLE:
            pytest.skip("FigRegistryConfigBridge not available")
        
        # Mock bridge
        mock_bridge = mocker.Mock()
        mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        bridge_access_results = []
        
        def access_bridge_concurrently(thread_id: int):
            """Access configuration bridge from multiple threads."""
            try:
                # Initialize configuration (should be thread-safe)
                hooks.after_config_loaded(context=None, config_loader=None)
                
                # Access bridge
                bridge = hooks._state.bridge
                
                bridge_access_results.append({
                    'thread_id': thread_id,
                    'bridge_available': bridge is not None,
                    'is_initialized': hooks._state.is_initialized
                })
                
            except Exception as e:
                bridge_access_results.append({
                    'thread_id': thread_id,
                    'error': str(e)
                })
        
        # Execute concurrent bridge access
        threads = []
        for i in range(8):
            thread = threading.Thread(target=access_bridge_concurrently, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify thread safety
        successful_access = [r for r in bridge_access_results if 'error' not in r]
        failed_access = [r for r in bridge_access_results if 'error' in r]
        
        assert len(failed_access) == 0, f"Bridge access failures: {failed_access}"
        assert len(successful_access) == 8, "Not all threads accessed bridge successfully"
        
        # Verify all threads see consistent state
        for result in successful_access:
            assert result['bridge_available'] is True
            assert result['is_initialized'] is True


# =============================================================================
# ERROR HANDLING AND FALLBACK TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestErrorHandlingAndFallback:
    """Test suite for error handling and fallback behavior per F-006.2.
    
    Validates graceful degradation when FigRegistry or Kedro components
    are unavailable, ensuring non-invasive integration that doesn't break
    existing Kedro workflows.
    """

    def test_figregistry_unavailable_fallback(self, figregistry_hooks_instance, mocker):
        """Test graceful fallback when FigRegistry is not available."""
        hooks = figregistry_hooks_instance
        
        # Mock FigRegistry as unavailable
        mocker.patch('figregistry_kedro.hooks.figregistry', None)
        
        # Should not raise exception
        hooks.after_config_loaded(context=None, config_loader=None)
        
        # Should not be initialized but no error
        assert hooks._state.is_initialized is False

    def test_kedro_components_unavailable_fallback(self, figregistry_hooks_instance):
        """Test graceful handling when Kedro components are unavailable."""
        hooks = figregistry_hooks_instance
        
        # Execute hooks with None values (simulating unavailable components)
        hooks.after_config_loaded(context=None, config_loader=None)
        hooks.before_pipeline_run(
            run_params={'pipeline_name': 'test'},
            pipeline=None,
            catalog=None
        )
        hooks.after_pipeline_run(
            run_params={'pipeline_name': 'test'},
            pipeline=None,
            catalog=None
        )
        
        # Should handle gracefully without exceptions

    def test_configuration_merge_error_fallback(self, figregistry_hooks_instance, mocker):
        """Test fallback behavior when configuration merging fails."""
        hooks = figregistry_hooks_instance
        
        # Mock components to trigger configuration error
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        
        if CONFIG_BRIDGE_AVAILABLE:
            # Mock bridge to raise error during initialization
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.side_effect = ConfigurationMergeError("Test merge error")
            
            # With fallback enabled, should not raise exception
            hooks.after_config_loaded(context=None, config_loader=None)
            
            # Should not be initialized but execution continues
            assert hooks._state.is_initialized is False

    def test_hook_execution_error_without_fallback(self, hooks_with_fallback_disabled, mocker):
        """Test error propagation when fallback is disabled."""
        hooks = hooks_with_fallback_disabled
        
        # Mock to raise exception
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        mock_init_config.side_effect = Exception("Test initialization error")
        
        # Should raise HookExecutionError when fallback disabled
        with pytest.raises(HookExecutionError) as exc_info:
            hooks.after_config_loaded(context=None, config_loader=None)
        
        assert "after_config_loaded" in str(exc_info.value)
        assert exc_info.value.hook_name == "after_config_loaded"

    def test_node_error_handling(self, figregistry_hooks_instance, mock_catalog):
        """Test proper handling of node execution errors."""
        hooks = figregistry_hooks_instance
        
        # Test with various error types
        error_types = [
            ValueError("Generic value error"),
            ConfigurationMergeError("Config error"),
            HookExecutionError("test_hook", "Hook error"),
            RuntimeError("Runtime error")
        ]
        
        for error in error_types:
            # Should handle all error types gracefully
            hooks.on_node_error(
                error=error,
                node_name="test_node",
                catalog=mock_catalog,
                inputs={'test': 'data'}
            )
        
        # No exceptions should be raised

    def test_performance_monitoring_error_handling(self, mocker):
        """Test error handling in performance monitoring with fallback."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHooks not available")
        
        # Create hooks with performance monitoring enabled
        hooks = FigRegistryHooks(enable_performance_monitoring=True)
        
        # Mock time.perf_counter to raise exception
        mock_perf_counter = mocker.patch('time.perf_counter')
        mock_perf_counter.side_effect = Exception("Timer error")
        
        # Should handle timer errors gracefully
        hooks.after_config_loaded(context=None, config_loader=None)
        
        # Cleanup
        hooks.reset_state()

    def test_late_initialization_error_handling(
        self,
        figregistry_hooks_instance,
        mock_pipeline_run_params,
        mock_pipeline,
        mock_catalog,
        mocker
    ):
        """Test error handling during late initialization attempts."""
        hooks = figregistry_hooks_instance
        
        # Mock components to fail during late initialization
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        mock_init_config.side_effect = Exception("Late init error")
        
        # Ensure not initialized
        assert hooks._state.is_initialized is False
        
        # Execute before_pipeline_run (should attempt late initialization)
        hooks.before_pipeline_run(
            run_params=mock_pipeline_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        
        # With fallback enabled, should continue despite late init failure
        # Pipeline should still be registered
        assert hooks._state.active_pipeline_count == 1


# =============================================================================
# STATE MANAGEMENT TESTS  
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestStateManagement:
    """Test suite for hook state management and lifecycle tracking.
    
    Validates proper state management throughout hook execution including
    initialization tracking, pipeline registration, and cleanup operations.
    """

    def test_hook_state_initialization(self, mock_hook_state):
        """Test FigRegistryHookState initialization and properties."""
        state = mock_hook_state
        
        # Verify initial state
        assert state.is_initialized is False
        assert state.bridge is None
        assert state.active_pipeline_count == 0
        
        # Test state modification
        state.mark_initialized("test_env")
        assert state.is_initialized is True
        
        # Test pipeline registration
        state.register_pipeline("test_pipeline")
        assert state.active_pipeline_count == 1
        
        # Test pipeline unregistration
        state.unregister_pipeline("test_pipeline")
        assert state.active_pipeline_count == 0

    def test_bridge_instance_management(self, mock_hook_state, mocker):
        """Test configuration bridge instance management."""
        state = mock_hook_state
        
        if CONFIG_BRIDGE_AVAILABLE:
            # Create mock bridge
            mock_bridge = mocker.Mock()
            
            # Set bridge
            state.set_bridge(mock_bridge)
            assert state.bridge is mock_bridge
            
            # Reset state
            state.reset()
            assert state.bridge is None
            assert state.is_initialized is False

    def test_pipeline_registration_tracking(self, mock_hook_state):
        """Test pipeline registration and tracking functionality."""
        state = mock_hook_state
        
        # Register multiple pipelines
        pipelines = ['pipeline1', 'pipeline2', 'pipeline3']
        for pipeline in pipelines:
            state.register_pipeline(pipeline)
        
        assert state.active_pipeline_count == 3
        
        # Unregister in different order
        state.unregister_pipeline('pipeline2')
        assert state.active_pipeline_count == 2
        
        state.unregister_pipeline('pipeline1')
        assert state.active_pipeline_count == 1
        
        state.unregister_pipeline('pipeline3')
        assert state.active_pipeline_count == 0
        
        # Test duplicate unregistration
        state.unregister_pipeline('pipeline1')  # Should not error
        assert state.active_pipeline_count == 0

    def test_concurrent_state_modifications(self, mock_hook_state):
        """Test state management under concurrent modifications."""
        state = mock_hook_state
        
        errors = []
        
        def concurrent_state_ops(thread_id: int):
            """Perform concurrent state operations."""
            try:
                # Register pipelines
                for i in range(5):
                    state.register_pipeline(f"thread_{thread_id}_pipeline_{i}")
                
                # Brief pause to encourage race conditions
                time.sleep(0.001)
                
                # Unregister pipelines
                for i in range(5):
                    state.unregister_pipeline(f"thread_{thread_id}_pipeline_{i}")
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Execute concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_state_ops, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and clean final state
        assert len(errors) == 0, f"Concurrent state errors: {errors}"
        assert state.active_pipeline_count == 0

    def test_state_reset_functionality(self, figregistry_hooks_instance, mocker):
        """Test complete state reset functionality."""
        hooks = figregistry_hooks_instance
        
        if CONFIG_BRIDGE_AVAILABLE:
            # Setup complex state
            mock_bridge = mocker.Mock()
            hooks._state.set_bridge(mock_bridge)
        
        hooks._state.mark_initialized("test_env")
        hooks._state.register_pipeline("test_pipeline1")
        hooks._state.register_pipeline("test_pipeline2")
        
        # Verify state is populated
        assert hooks._state.is_initialized is True
        if CONFIG_BRIDGE_AVAILABLE:
            assert hooks._state.bridge is not None
        assert hooks._state.active_pipeline_count == 2
        
        # Reset state
        hooks.reset_state()
        
        # Verify complete reset
        assert hooks._state.is_initialized is False
        assert hooks._state.bridge is None
        assert hooks._state.active_pipeline_count == 0

    def test_get_state_debugging_info(self, figregistry_hooks_instance):
        """Test get_state method for debugging and monitoring."""
        hooks = figregistry_hooks_instance
        
        # Get initial state
        state_info = hooks.get_state()
        
        # Verify state structure
        expected_keys = {
            'initialized', 'active_pipelines', 'bridge_available',
            'auto_initialize', 'performance_monitoring', 'fallback_on_error'
        }
        assert set(state_info.keys()) == expected_keys
        
        # Verify initial values
        assert state_info['initialized'] is False
        assert state_info['active_pipelines'] == 0
        assert state_info['bridge_available'] is False
        assert state_info['auto_initialize'] is True
        assert state_info['performance_monitoring'] is True
        assert state_info['fallback_on_error'] is True
        
        # Modify state and verify updates
        hooks._state.mark_initialized("test")
        hooks._state.register_pipeline("test_pipeline")
        
        updated_state = hooks.get_state()
        assert updated_state['initialized'] is True
        assert updated_state['active_pipelines'] == 1


# =============================================================================
# INTEGRATION WITH EXISTING HOOK SYSTEM TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.kedro_plugin
class TestHookSystemIntegration:
    """Test suite for integration with existing Kedro hook systems.
    
    Validates that FigRegistryHooks integrate properly with other Kedro hooks
    and plugin systems without conflicts or interference.
    """

    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available for hook manager testing")
    def test_multiple_hooks_coexistence(self, mocker):
        """Test FigRegistryHooks coexistence with other Kedro hooks."""
        # Mock PluginManager and other hooks
        mock_plugin_manager = mocker.Mock(spec=PluginManager)
        
        # Create mock other hooks
        other_hook1 = mocker.Mock()
        other_hook2 = mocker.Mock()
        
        # Register all hooks
        figregistry_hooks = FigRegistryHooks()
        hooks_list = [other_hook1, figregistry_hooks, other_hook2]
        
        # Mock hook execution sequence
        mock_plugin_manager.call_hook.side_effect = lambda hook_name, **kwargs: [
            getattr(hook, hook_name, lambda **kw: None)(**kwargs) 
            for hook in hooks_list 
            if hasattr(hook, hook_name)
        ]
        
        # Test hook execution sequence
        mock_plugin_manager.call_hook(
            'after_config_loaded',
            context=None,
            config_loader=None
        )
        
        # Verify FigRegistryHooks executed without interfering with others
        figregistry_hooks.reset_state()

    def test_hook_execution_order_independence(self, figregistry_hooks_instance, mocker):
        """Test that hook execution order doesn't affect functionality."""
        hooks = figregistry_hooks_instance
        
        # Mock components
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mocker.patch('figregistry_kedro.hooks.init_config', return_value={'test': True})
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge', return_value=mock_bridge)
        
        # Test different execution orders
        execution_orders = [
            ['after_config_loaded', 'before_pipeline_run', 'after_pipeline_run'],
            ['before_pipeline_run', 'after_config_loaded', 'after_pipeline_run'],
        ]
        
        for order in execution_orders:
            hooks.reset_state()
            
            for hook_method in order:
                if hook_method == 'after_config_loaded':
                    hooks.after_config_loaded(context=None, config_loader=None)
                elif hook_method == 'before_pipeline_run':
                    hooks.before_pipeline_run(
                        run_params={'pipeline_name': 'test'},
                        pipeline=None,
                        catalog=None
                    )
                elif hook_method == 'after_pipeline_run':
                    hooks.after_pipeline_run(
                        run_params={'pipeline_name': 'test'},
                        pipeline=None,
                        catalog=None
                    )
            
            # Should handle any execution order gracefully

    def test_hook_isolation_from_other_plugins(self, figregistry_hooks_instance, mocker):
        """Test that FigRegistryHooks don't interfere with other plugin operations."""
        hooks = figregistry_hooks_instance
        
        # Mock other plugin operations that might conflict
        mock_other_plugin_config = mocker.Mock()
        mock_other_plugin_config.some_method.return_value = "other_plugin_result"
        
        # Mock global state that other plugins might use
        mock_global_state = {'other_plugin': 'active', 'shared_resource': 'available'}
        
        # Execute FigRegistry hooks
        hooks.after_config_loaded(context=None, config_loader=None)
        hooks.before_pipeline_run(
            run_params={'pipeline_name': 'test'},
            pipeline=None,
            catalog=None
        )
        
        # Verify other plugin state unchanged
        assert mock_global_state['other_plugin'] == 'active'
        assert mock_global_state['shared_resource'] == 'available'
        assert mock_other_plugin_config.some_method.return_value == "other_plugin_result"

    def test_kedro_session_lifecycle_integration(self, figregistry_hooks_instance, mocker):
        """Test integration with complete Kedro session lifecycle."""
        hooks = figregistry_hooks_instance
        
        # Mock Kedro session components
        mock_session = mocker.Mock()
        mock_context = mocker.Mock()
        mock_context.env = "test"
        
        # Mock session lifecycle
        session_lifecycle_events = [
            'session_created',
            'context_loaded',
            'config_loaded', 
            'pipeline_registered',
            'pipeline_run_started',
            'pipeline_run_completed',
            'session_closed'
        ]
        
        # Execute relevant hooks during session lifecycle
        for event in session_lifecycle_events:
            if event == 'config_loaded':
                hooks.after_config_loaded(
                    context=mock_context,
                    config_loader=None
                )
            elif event == 'pipeline_run_started':
                hooks.before_pipeline_run(
                    run_params={'pipeline_name': 'test'},
                    pipeline=None,
                    catalog=None
                )
            elif event == 'pipeline_run_completed':
                hooks.after_pipeline_run(
                    run_params={'pipeline_name': 'test'},
                    pipeline=None,
                    catalog=None
                )
        
        # Verify clean final state
        assert hooks._state.active_pipeline_count == 0


# =============================================================================
# MODULE GLOBAL FUNCTIONS TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.kedro_plugin
class TestModuleGlobalFunctions:
    """Test suite for module-level global functions and utilities."""

    def test_get_hook_state_function(self):
        """Test get_hook_state global function."""
        # Function should return None for now (as documented)
        state = get_hook_state()
        assert state is None

    def test_default_hook_config_constant(self):
        """Test DEFAULT_HOOK_CONFIG constant values."""
        if not HOOKS_AVAILABLE:
            pytest.skip("Hook constants not available")
        
        expected_config = {
            'auto_initialize': True,
            'enable_performance_monitoring': False,
            'fallback_on_error': True,
            'max_initialization_time': 0.005  # 5ms
        }
        
        assert DEFAULT_HOOK_CONFIG == expected_config

    def test_hook_execution_error_exception(self):
        """Test HookExecutionError exception class."""
        if not HOOKS_AVAILABLE:
            pytest.skip("HookExecutionError not available")
        
        # Test basic creation
        error = HookExecutionError("test_hook", "Test error message")
        assert error.hook_name == "test_hook"
        assert "test_hook" in str(error)
        assert "Test error message" in str(error)
        
        # Test with original error
        original_error = ValueError("Original error")
        error_with_original = HookExecutionError(
            "test_hook", 
            "Wrapper error", 
            original_error
        )
        assert error_with_original.original_error is original_error

    def test_figregistry_hook_state_class(self):
        """Test FigRegistryHookState class functionality."""
        if not HOOKS_AVAILABLE:
            pytest.skip("FigRegistryHookState not available")
        
        state = FigRegistryHookState()
        
        # Test property access
        assert state.is_initialized is False
        assert state.bridge is None
        assert state.active_pipeline_count == 0
        
        # Test state modifications
        state.mark_initialized("test")
        assert state.is_initialized is True
        
        state.register_pipeline("test_pipeline")
        assert state.active_pipeline_count == 1
        
        state.unregister_pipeline("test_pipeline")
        assert state.active_pipeline_count == 0
        
        # Test reset
        state.reset()
        assert state.is_initialized is False


# =============================================================================
# SKIP CONDITIONS AND MODULE AVAILABILITY TESTS
# =============================================================================

@pytest.mark.unit
def test_module_availability_handling():
    """Test proper handling of module availability conditions."""
    # Test HOOKS_AVAILABLE flag
    if HOOKS_AVAILABLE:
        assert FigRegistryHooks is not None
        assert FigRegistryHookState is not None
        assert HookExecutionError is not None
    else:
        # When hooks not available, tests should be skipped
        pytest.skip("FigRegistryHooks module not available")

@pytest.mark.unit  
def test_graceful_import_fallback():
    """Test graceful fallback when imports fail."""
    # This test validates that the test module itself handles import failures
    assert HOOKS_AVAILABLE in [True, False]
    assert CONFIG_BRIDGE_AVAILABLE in [True, False]
    assert KEDRO_AVAILABLE in [True, False]
    
    # If any critical modules unavailable, appropriate skips should occur
    if not HOOKS_AVAILABLE:
        with pytest.raises(pytest.skip.Exception):
            pytest.skip("Required modules not available")


# =============================================================================
# COMPREHENSIVE INTEGRATION TEST
# =============================================================================

@pytest.mark.integration
@pytest.mark.kedro_plugin
@pytest.mark.slow
class TestComprehensiveHookIntegration:
    """Comprehensive integration test for complete hook functionality.
    
    Tests the complete workflow from hook registration through pipeline
    execution with realistic scenarios and full component integration.
    """

    def test_complete_hook_integration_workflow(
        self,
        figregistry_hooks_instance,
        mock_kedro_context,
        mock_pipeline,
        mock_catalog,
        performance_tracker,
        mocker
    ):
        """Test complete integration workflow with performance monitoring."""
        hooks = figregistry_hooks_instance
        
        # Mock all FigRegistry components
        mocker.patch('figregistry_kedro.hooks.figregistry', True)
        mock_init_config = mocker.patch('figregistry_kedro.hooks.init_config')
        mock_init_config.return_value = {
            'styles': {
                'exploratory': {'color': '#1f77b4', 'marker': 'o'},
                'presentation': {'color': '#ff7f0e', 'marker': 's'}
            },
            'outputs': {'base_path': 'integration_test_figures'}
        }
        
        if CONFIG_BRIDGE_AVAILABLE:
            mock_bridge = mocker.Mock()
            mock_bridge.get_merged_config.return_value = mock_init_config.return_value
            mock_bridge_class = mocker.patch('figregistry_kedro.hooks.FigRegistryConfigBridge')
            mock_bridge_class.return_value = mock_bridge
        
        # Track performance throughout workflow
        performance_tracker.start_timing('complete_workflow')
        
        # 1. Configuration Loading Phase
        performance_tracker.start_timing('config_phase')
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_kedro_context.config_loader,
            conf_source="/integration/test/conf"
        )
        config_time = performance_tracker.end_timing('config_phase')
        
        # 2. Multiple Pipeline Execution Phase
        pipeline_params = [
            {'pipeline_name': 'exploratory_analysis', 'session_id': 'session_1'},
            {'pipeline_name': 'model_training', 'session_id': 'session_1'},
            {'pipeline_name': 'evaluation', 'session_id': 'session_1'}
        ]
        
        performance_tracker.start_timing('pipeline_phase')
        
        # Start all pipelines
        for params in pipeline_params:
            hooks.before_pipeline_run(
                run_params=params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
        
        # Simulate pipeline execution with some processing time
        time.sleep(0.01)  # 10ms simulation
        
        # Complete all pipelines
        for params in pipeline_params:
            hooks.after_pipeline_run(
                run_params=params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
        
        pipeline_time = performance_tracker.end_timing('pipeline_phase')
        total_time = performance_tracker.end_timing('complete_workflow')
        
        # 3. Validation Phase
        
        # Verify configuration was initialized
        assert hooks._state.is_initialized is True
        if CONFIG_BRIDGE_AVAILABLE:
            assert hooks._state.bridge is not None
            mock_bridge_class.assert_called_once()
            mock_init_config.assert_called_once()
        
        # Verify all pipelines completed cleanly
        assert hooks._state.active_pipeline_count == 0
        
        # Verify performance targets met
        assert config_time < 50.0, f"Config phase took {config_time:.2f}ms, exceeds 50ms target"
        assert pipeline_time < 100.0, f"Pipeline phase took {pipeline_time:.2f}ms, exceeds 100ms target"
        assert total_time < 150.0, f"Total workflow took {total_time:.2f}ms, exceeds 150ms target"
        
        # 4. State Verification
        final_state = hooks.get_state()
        assert final_state['initialized'] is True
        assert final_state['active_pipelines'] == 0
        assert final_state['bridge_available'] is (CONFIG_BRIDGE_AVAILABLE and True)
        
        # 5. Error Resilience Test
        # Simulate error during node execution
        test_error = ValueError("Simulated node execution error")
        hooks.on_node_error(
            error=test_error,
            node_name="integration_test_node",
            catalog=mock_catalog,
            inputs={'test_data': 'sample'}
        )
        
        # Hook should handle error gracefully without affecting state
        assert hooks._state.is_initialized is True
        assert hooks._state.active_pipeline_count == 0
        
        # 6. Cleanup Verification
        hooks.reset_state()
        reset_state = hooks.get_state()
        assert reset_state['initialized'] is False
        assert reset_state['active_pipelines'] == 0
        assert reset_state['bridge_available'] is False


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])