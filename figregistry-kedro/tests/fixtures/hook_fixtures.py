"""
Hook lifecycle testing fixtures for FigRegistryHooks validation.

This module provides comprehensive hook lifecycle testing fixtures per Section 5.2.7,
enabling comprehensive validation of FigRegistryHooks registration, initialization,
and lifecycle integration with Kedro's execution model. The fixtures support testing
of hook manager interactions, plugin discovery scenarios, thread-safe operations,
and performance validation for the complete hook system.

Key Features:
- Mock hook manager simulation for registration testing per F-006.2
- Hook lifecycle event simulation (before_pipeline_run, after_config_loaded)
- Hook registration validation through plugin discovery system
- Thread-safe operation testing for parallel pipeline execution per Section 5.2.7
- Performance validation fixtures for <25ms initialization overhead per Section 6.6.4.3
- Non-invasive integration testing preserving Kedro's execution model per F-006.2
- Comprehensive cleanup validation for resource management

The module leverages pytest-mock for comprehensive hook manager simulation and provides
specialized fixtures for validating hook system behavior across different execution
scenarios including parallel runners, multi-environment configurations, and lifecycle
event sequences.

Usage:
    @pytest.fixture
    def test_hook_registration(mock_hook_manager, figregistry_hooks_instance):
        # Test hook registration through mock manager
        
    @pytest.fixture  
    def test_lifecycle_events(hook_lifecycle_events):
        # Validate before_pipeline_run and after_config_loaded scenarios
"""

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Kedro imports with graceful fallback
try:
    from kedro.framework.context import KedroContext
    from kedro.framework.hooks import PluginManager
    from kedro.framework.session import KedroSession
    from kedro.config import ConfigLoader
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline
    from kedro.runner import AbstractRunner, ParallelRunner, SequentialRunner
    HAS_KEDRO = True
except ImportError:
    KedroContext = None
    PluginManager = None
    KedroSession = None  
    ConfigLoader = None
    DataCatalog = None
    Pipeline = None
    AbstractRunner = None
    ParallelRunner = None
    SequentialRunner = None
    HAS_KEDRO = False

# FigRegistry hooks import with fallback
try:
    from figregistry_kedro.hooks import (
        FigRegistryHooks,
        HookInitializationError,
        HookExecutionError,
        hook_context,
        get_global_hook_state,
        clear_global_hook_state
    )
    HAS_FIGREGISTRY_HOOKS = True
except ImportError:
    FigRegistryHooks = None
    HookInitializationError = None
    HookExecutionError = None
    hook_context = None
    get_global_hook_state = None
    clear_global_hook_state = None
    HAS_FIGREGISTRY_HOOKS = False

# Local fixture dependencies
from figregistry_kedro.tests.fixtures.kedro_fixtures import (
    minimal_kedro_context,
    mock_config_loader,
    test_catalog_with_figregistry
)


# =============================================================================
# Hook Manager Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_hook_manager(mocker):
    """
    Mock hook manager for hook registration testing through Kedro's plugin discovery system.
    
    Creates a comprehensive mock of Kedro's PluginManager for testing FigRegistryHooks
    registration, lifecycle event simulation, and plugin discovery validation per 
    Section 5.2.7. The fixture provides tracked hook registration and invocation
    capabilities for comprehensive hook system testing.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        
    Returns:
        Mock PluginManager with hook registration tracking and lifecycle simulation
        
    Validates:
        - Hook registration through plugin discovery system per F-006.2
        - Hook invocation tracking and parameter validation
        - Multiple hook instance management for complex scenarios
        - Hook execution order and lifecycle event sequences
    """
    if not HAS_KEDRO:
        pytest.skip("Kedro not available for hook manager mocking")
    
    # Create mock PluginManager with proper specification
    mock_manager = mocker.Mock(spec=PluginManager)
    
    # Track registered hooks for comprehensive testing
    registered_hooks = []
    hook_invocation_history = []
    hook_discovery_log = []
    
    def mock_register(hook_instance, validate=True):
        """Mock hook registration with validation tracking."""
        registration_record = {
            'hook_instance': hook_instance,
            'hook_type': type(hook_instance).__name__,
            'registration_time': time.time(),
            'validated': validate,
            'hook_methods': _extract_hook_methods(hook_instance)
        }
        
        registered_hooks.append(registration_record)
        hook_discovery_log.append({
            'action': 'register',
            'hook_type': type(hook_instance).__name__,
            'timestamp': time.time(),
            'method_count': len(registration_record['hook_methods'])
        })
        
        return True
    
    def mock_unregister(hook_instance):
        """Mock hook unregistration with tracking."""
        hook_discovery_log.append({
            'action': 'unregister', 
            'hook_type': type(hook_instance).__name__,
            'timestamp': time.time()
        })
        
        # Remove from registered hooks
        registered_hooks[:] = [
            record for record in registered_hooks 
            if record['hook_instance'] is not hook_instance
        ]
        
        return True
    
    def mock_call_hook(hook_name, **kwargs):
        """Mock hook invocation with comprehensive parameter tracking."""
        invocation_start = time.time()
        
        invocation_record = {
            'hook_name': hook_name,
            'kwargs': kwargs.copy(),
            'timestamp': invocation_start,
            'registered_hook_count': len(registered_hooks),
            'results': []
        }
        
        # Simulate hook execution for registered hooks
        for hook_record in registered_hooks:
            hook_instance = hook_record['hook_instance']
            
            if hasattr(hook_instance, hook_name):
                hook_method = getattr(hook_instance, hook_name)
                
                try:
                    # Time individual hook execution
                    method_start = time.time()
                    result = hook_method(**kwargs)
                    method_duration = (time.time() - method_start) * 1000  # Convert to ms
                    
                    invocation_record['results'].append({
                        'hook_type': type(hook_instance).__name__,
                        'result': result,
                        'success': True,
                        'execution_time_ms': method_duration
                    })
                    
                except Exception as e:
                    method_duration = (time.time() - method_start) * 1000
                    
                    invocation_record['results'].append({
                        'hook_type': type(hook_instance).__name__,
                        'error': str(e),
                        'success': False,
                        'execution_time_ms': method_duration
                    })
        
        invocation_record['total_duration_ms'] = (time.time() - invocation_start) * 1000
        hook_invocation_history.append(invocation_record)
        
        return [result['result'] for result in invocation_record['results'] if result['success']]
    
    def mock_is_registered(hook_instance):
        """Check if hook instance is registered."""
        return any(
            record['hook_instance'] is hook_instance 
            for record in registered_hooks
        )
    
    def mock_list_hooks():
        """Get list of all registered hook instances."""
        return [record['hook_instance'] for record in registered_hooks]
    
    def mock_get_hook_callers():
        """Get mapping of hook names to registered callers."""
        hook_callers = {}
        
        for record in registered_hooks:
            for method_name in record['hook_methods']:
                if method_name not in hook_callers:
                    hook_callers[method_name] = []
                hook_callers[method_name].append(record['hook_instance'])
        
        return hook_callers
    
    # Configure mock PluginManager methods
    mock_manager.register = Mock(side_effect=mock_register)
    mock_manager.unregister = Mock(side_effect=mock_unregister)
    mock_manager.call_hook = Mock(side_effect=mock_call_hook)
    mock_manager.is_registered = Mock(side_effect=mock_is_registered)
    mock_manager.list_hooks = Mock(side_effect=mock_list_hooks)
    mock_manager.get_hook_callers = Mock(side_effect=mock_get_hook_callers)
    
    # Add plugin discovery simulation
    mock_manager.discover_hooks = Mock(return_value=[])
    mock_manager.hook_specs = Mock(return_value=['before_pipeline_run', 'after_pipeline_run', 'after_config_loaded'])
    
    # Provide access to tracking data for test validation
    mock_manager._registered_hooks = registered_hooks
    mock_manager._hook_invocation_history = hook_invocation_history
    mock_manager._hook_discovery_log = hook_discovery_log
    
    # Performance tracking utilities
    mock_manager.get_performance_summary = Mock(
        side_effect=lambda: _generate_performance_summary(hook_invocation_history)
    )
    
    return mock_manager


@pytest.fixture
def mock_hook_manager_with_discovery(mock_hook_manager, mocker):
    """
    Enhanced mock hook manager with comprehensive plugin discovery simulation.
    
    Extends the basic mock_hook_manager with realistic plugin discovery behavior
    including entry point scanning, settings.py hook registration, and automatic
    hook instance creation for comprehensive plugin discovery testing.
    
    Args:
        mock_hook_manager: Base mock hook manager fixture
        mocker: pytest-mock fixture
        
    Returns:
        Enhanced mock PluginManager with plugin discovery capabilities
        
    Validates:
        - Entry point based plugin discovery per Section 5.2.7
        - settings.py hook registration validation per F-006.2
        - Automatic hook instance lifecycle management
        - Plugin loading error handling and recovery scenarios
    """
    discovery_simulation = {
        'entry_points': {},
        'settings_hooks': [],
        'auto_discovery_enabled': True,
        'discovery_errors': []
    }
    
    def mock_discover_hooks_from_entrypoints():
        """Simulate entry point based hook discovery."""
        discovered_hooks = []
        
        for entry_point_name, hook_class in discovery_simulation['entry_points'].items():
            try:
                # Simulate entry point instantiation
                hook_instance = hook_class()
                discovered_hooks.append({
                    'source': 'entry_point',
                    'name': entry_point_name,
                    'instance': hook_instance,
                    'hook_class': hook_class
                })
            except Exception as e:
                discovery_simulation['discovery_errors'].append({
                    'source': 'entry_point',
                    'name': entry_point_name,
                    'error': str(e)
                })
        
        return discovered_hooks
    
    def mock_discover_hooks_from_settings():
        """Simulate settings.py hook discovery."""
        discovered_hooks = []
        
        for hook_instance in discovery_simulation['settings_hooks']:
            discovered_hooks.append({
                'source': 'settings',
                'instance': hook_instance,
                'hook_class': type(hook_instance)
            })
        
        return discovered_hooks
    
    def mock_auto_discover_and_register():
        """Simulate complete auto-discovery and registration process."""
        if not discovery_simulation['auto_discovery_enabled']:
            return
        
        # Discover from entry points
        entry_point_hooks = mock_discover_hooks_from_entrypoints()
        settings_hooks = mock_discover_hooks_from_settings()
        
        # Register all discovered hooks
        for hook_info in entry_point_hooks + settings_hooks:
            mock_hook_manager.register(hook_info['instance'])
        
        return {
            'entry_point_hooks': len(entry_point_hooks),
            'settings_hooks': len(settings_hooks),
            'total_registered': len(entry_point_hooks) + len(settings_hooks),
            'errors': len(discovery_simulation['discovery_errors'])
        }
    
    # Add entry point simulation methods
    def register_entry_point(name, hook_class):
        """Register entry point for discovery testing."""
        discovery_simulation['entry_points'][name] = hook_class
    
    def register_settings_hook(hook_instance):
        """Register settings.py hook for discovery testing."""
        discovery_simulation['settings_hooks'].append(hook_instance)
    
    def enable_auto_discovery(enabled=True):
        """Enable/disable auto discovery for testing."""
        discovery_simulation['auto_discovery_enabled'] = enabled
    
    # Enhance mock manager with discovery capabilities
    mock_hook_manager.discover_hooks_from_entrypoints = Mock(
        side_effect=mock_discover_hooks_from_entrypoints
    )
    mock_hook_manager.discover_hooks_from_settings = Mock(
        side_effect=mock_discover_hooks_from_settings
    )
    mock_hook_manager.auto_discover_and_register = Mock(
        side_effect=mock_auto_discover_and_register
    )
    
    # Test utilities for discovery simulation
    mock_hook_manager.register_entry_point = register_entry_point
    mock_hook_manager.register_settings_hook = register_settings_hook  
    mock_hook_manager.enable_auto_discovery = enable_auto_discovery
    
    # Access to discovery simulation state
    mock_hook_manager._discovery_simulation = discovery_simulation
    
    return mock_hook_manager


# =============================================================================
# Hook Lifecycle Event Fixtures
# =============================================================================

@pytest.fixture
def hook_lifecycle_events(mocker, minimal_kedro_context, test_catalog_with_figregistry):
    """
    Hook lifecycle events fixture simulating before_pipeline_run and after_config_loaded scenarios.
    
    Provides comprehensive simulation of Kedro hook lifecycle events for testing
    FigRegistryHooks integration with pipeline execution phases per F-006.2. The
    fixture generates realistic event sequences with proper parameter structures
    and execution contexts for thorough hook validation testing.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        minimal_kedro_context: Mock Kedro context fixture
        test_catalog_with_figregistry: Mock catalog with FigureDataSet entries
        
    Returns:
        Dictionary containing lifecycle event simulation utilities and test scenarios
        
    Validates:
        - Hook execution during before_pipeline_run event per Section 5.2.7
        - Configuration loading during after_config_loaded event
        - Hook parameter validation and context management
        - Event sequence ordering and timing constraints
    """
    lifecycle_simulation = {
        'event_history': [],
        'current_context': None,
        'execution_parameters': {},
        'performance_metrics': {
            'event_timings': {},
            'hook_execution_times': [],
            'context_initialization_time': None
        }
    }
    
    def simulate_after_config_loaded_event(
        context=None,
        config_loader=None,
        conf_source=None,
        environment='test'
    ):
        """
        Simulate after_config_loaded hook event with realistic parameters.
        
        Args:
            context: Mock Kedro context (defaults to minimal_kedro_context)
            config_loader: Mock config loader (defaults to mock_config_loader)
            conf_source: Configuration source path
            environment: Environment name for configuration loading
            
        Returns:
            Dictionary containing event simulation results and timing
        """
        event_start = time.time()
        
        # Use provided context or default
        event_context = context or minimal_kedro_context
        event_config_loader = config_loader or mocker.Mock(spec=ConfigLoader)
        event_conf_source = conf_source or f"conf/{environment}"
        
        # Configure mock config loader responses
        if hasattr(event_config_loader, 'get'):
            event_config_loader.get.return_value = {
                'styles': {
                    'test_condition': {'color': '#1f77b4', 'marker': 'o'},
                    'validation_condition': {'color': '#ff7f0e', 'marker': 's'}
                },
                'outputs': {
                    'base_path': 'test_figures',
                    'versioning': True
                },
                'metadata': {
                    'loaded_from': 'hook_lifecycle_test',
                    'environment': environment
                }
            }
        
        event_record = {
            'event_type': 'after_config_loaded',
            'timestamp': event_start,
            'parameters': {
                'context': event_context,
                'config_loader': event_config_loader,
                'conf_source': event_conf_source
            },
            'environment': environment,
            'execution_duration_ms': None,
            'hook_results': []
        }
        
        # Store current context for subsequent events
        lifecycle_simulation['current_context'] = event_context
        lifecycle_simulation['execution_parameters']['config_loaded'] = True
        lifecycle_simulation['execution_parameters']['environment'] = environment
        
        event_duration = (time.time() - event_start) * 1000
        event_record['execution_duration_ms'] = event_duration
        
        lifecycle_simulation['event_history'].append(event_record)
        lifecycle_simulation['performance_metrics']['event_timings']['after_config_loaded'] = event_duration
        
        return event_record
    
    def simulate_before_pipeline_run_event(
        run_params=None,
        pipeline=None,
        catalog=None,
        pipeline_name='test_pipeline'
    ):
        """
        Simulate before_pipeline_run hook event with pipeline context.
        
        Args:
            run_params: Pipeline run parameters dictionary
            pipeline: Mock pipeline instance
            catalog: Mock data catalog (defaults to test_catalog_with_figregistry)
            pipeline_name: Name of the pipeline being executed
            
        Returns:
            Dictionary containing event simulation results and hook execution data
        """
        event_start = time.time()
        
        # Create realistic run parameters
        event_run_params = run_params or {
            'run_id': f'test_run_{int(time.time())}',
            'pipeline_name': pipeline_name,
            'tags': [],
            'node_names': [],
            'from_nodes': [],
            'to_nodes': [],
            'from_inputs': [],
            'to_outputs': [],
            'environment': lifecycle_simulation['execution_parameters'].get('environment', 'test')
        }
        
        # Create mock pipeline
        event_pipeline = pipeline or mocker.Mock(spec=Pipeline)
        if hasattr(event_pipeline, 'name'):
            event_pipeline.name = pipeline_name
        if hasattr(event_pipeline, 'nodes'):
            event_pipeline.nodes = [
                mocker.Mock(name='test_node_1'),
                mocker.Mock(name='test_node_2')
            ]
        
        # Use catalog or default
        event_catalog = catalog or test_catalog_with_figregistry
        
        event_record = {
            'event_type': 'before_pipeline_run',
            'timestamp': event_start,
            'parameters': {
                'run_params': event_run_params,
                'pipeline': event_pipeline,
                'catalog': event_catalog
            },
            'pipeline_name': pipeline_name,
            'run_id': event_run_params['run_id'],
            'execution_duration_ms': None,
            'hook_results': []
        }
        
        # Update execution state
        lifecycle_simulation['execution_parameters']['pipeline_started'] = True
        lifecycle_simulation['execution_parameters']['current_run_id'] = event_run_params['run_id']
        lifecycle_simulation['execution_parameters']['current_pipeline'] = pipeline_name
        
        event_duration = (time.time() - event_start) * 1000
        event_record['execution_duration_ms'] = event_duration
        
        lifecycle_simulation['event_history'].append(event_record)
        lifecycle_simulation['performance_metrics']['event_timings']['before_pipeline_run'] = event_duration
        
        return event_record
    
    def simulate_after_pipeline_run_event(
        run_params=None,
        pipeline=None,
        catalog=None
    ):
        """
        Simulate after_pipeline_run hook event for cleanup testing.
        
        Args:
            run_params: Pipeline run parameters (uses current execution params if None)
            pipeline: Mock pipeline instance
            catalog: Mock data catalog
            
        Returns:
            Dictionary containing cleanup event simulation results
        """
        event_start = time.time()
        
        # Use current execution parameters if not provided
        current_run_id = lifecycle_simulation['execution_parameters'].get('current_run_id')
        current_pipeline = lifecycle_simulation['execution_parameters'].get('current_pipeline', 'test_pipeline')
        
        event_run_params = run_params or {
            'run_id': current_run_id,
            'pipeline_name': current_pipeline,
            'environment': lifecycle_simulation['execution_parameters'].get('environment', 'test')
        }
        
        event_pipeline = pipeline or mocker.Mock(spec=Pipeline)
        if hasattr(event_pipeline, 'name'):
            event_pipeline.name = current_pipeline
        
        event_catalog = catalog or test_catalog_with_figregistry
        
        event_record = {
            'event_type': 'after_pipeline_run',
            'timestamp': event_start,
            'parameters': {
                'run_params': event_run_params,
                'pipeline': event_pipeline,
                'catalog': event_catalog
            },
            'pipeline_name': current_pipeline,
            'run_id': current_run_id,
            'execution_duration_ms': None,
            'cleanup_performed': True
        }
        
        # Update execution state for cleanup
        lifecycle_simulation['execution_parameters']['pipeline_started'] = False
        lifecycle_simulation['execution_parameters']['cleanup_completed'] = True
        
        event_duration = (time.time() - event_start) * 1000
        event_record['execution_duration_ms'] = event_duration
        
        lifecycle_simulation['event_history'].append(event_record)
        lifecycle_simulation['performance_metrics']['event_timings']['after_pipeline_run'] = event_duration
        
        return event_record
    
    def simulate_complete_lifecycle(pipeline_name='test_lifecycle_pipeline'):
        """
        Simulate complete hook lifecycle from config loading through pipeline cleanup.
        
        Args:
            pipeline_name: Name of the pipeline for the complete lifecycle test
            
        Returns:
            Dictionary containing complete lifecycle simulation results and metrics
        """
        lifecycle_start = time.time()
        
        # Execute complete lifecycle sequence
        config_event = simulate_after_config_loaded_event()
        before_event = simulate_before_pipeline_run_event(pipeline_name=pipeline_name)
        after_event = simulate_after_pipeline_run_event()
        
        total_lifecycle_time = (time.time() - lifecycle_start) * 1000
        
        return {
            'total_lifecycle_time_ms': total_lifecycle_time,
            'events': {
                'after_config_loaded': config_event,
                'before_pipeline_run': before_event,
                'after_pipeline_run': after_event
            },
            'event_count': len(lifecycle_simulation['event_history']),
            'performance_summary': lifecycle_simulation['performance_metrics']
        }
    
    def get_event_sequence_validation():
        """
        Validate the sequence and timing of lifecycle events.
        
        Returns:
            Dictionary containing sequence validation results and timing analysis
        """
        events = lifecycle_simulation['event_history']
        
        if not events:
            return {'valid_sequence': False, 'error': 'No events recorded'}
        
        # Check event sequence ordering
        expected_sequence = ['after_config_loaded', 'before_pipeline_run', 'after_pipeline_run']
        actual_sequence = [event['event_type'] for event in events[-3:]]  # Last 3 events
        
        sequence_valid = actual_sequence == expected_sequence if len(actual_sequence) == 3 else True
        
        # Analyze timing characteristics
        timing_analysis = {}
        for event_type, timing in lifecycle_simulation['performance_metrics']['event_timings'].items():
            timing_analysis[event_type] = {
                'duration_ms': timing,
                'within_25ms_target': timing < 25.0,
                'performance_grade': 'excellent' if timing < 5.0 else 'good' if timing < 15.0 else 'acceptable' if timing < 25.0 else 'slow'
            }
        
        return {
            'valid_sequence': sequence_valid,
            'expected_sequence': expected_sequence,
            'actual_sequence': actual_sequence,
            'timing_analysis': timing_analysis,
            'total_events': len(events),
            'performance_metrics': lifecycle_simulation['performance_metrics']
        }
    
    def reset_lifecycle_state():
        """Reset lifecycle simulation state for clean test runs."""
        lifecycle_simulation['event_history'].clear()
        lifecycle_simulation['current_context'] = None
        lifecycle_simulation['execution_parameters'].clear()
        lifecycle_simulation['performance_metrics'] = {
            'event_timings': {},
            'hook_execution_times': [],
            'context_initialization_time': None
        }
    
    return {
        'simulate_after_config_loaded': simulate_after_config_loaded_event,
        'simulate_before_pipeline_run': simulate_before_pipeline_run_event,
        'simulate_after_pipeline_run': simulate_after_pipeline_run_event,
        'simulate_complete_lifecycle': simulate_complete_lifecycle,
        'get_event_sequence_validation': get_event_sequence_validation,
        'reset_lifecycle_state': reset_lifecycle_state,
        'lifecycle_state': lifecycle_simulation
    }


# =============================================================================
# Hook Registration Fixtures
# =============================================================================

@pytest.fixture
def hook_registration_fixtures(mocker, tmp_path):
    """
    Hook registration fixtures for testing settings.py configuration and entry point validation.
    
    Provides comprehensive testing utilities for validating FigRegistryHooks registration
    through Kedro's plugin discovery system including settings.py configuration,
    entry point validation, and hook instance lifecycle management per Section 5.2.7.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        tmp_path: pytest temporary path fixture for project simulation
        
    Returns:
        Dictionary containing hook registration testing utilities and validation functions
        
    Validates:
        - Hook registration through settings.py configuration per F-006.2
        - Entry point based plugin discovery per Section 5.2.7
        - Hook instance creation and initialization validation
        - Plugin loading error handling and recovery scenarios
    """
    registration_simulation = {
        'temporary_projects': [],
        'settings_configurations': {},
        'entry_point_registrations': {},
        'registration_errors': [],
        'discovered_hooks': []
    }
    
    def create_test_kedro_project(
        project_name='test_hook_registration_project',
        include_figregistry_hooks=True,
        include_malformed_settings=False
    ):
        """
        Create temporary Kedro project for hook registration testing.
        
        Args:
            project_name: Name of the temporary project
            include_figregistry_hooks: Whether to include FigRegistryHooks in settings.py
            include_malformed_settings: Whether to include malformed settings for error testing
            
        Returns:
            Dictionary containing project paths and configuration for testing
        """
        project_path = tmp_path / project_name
        project_path.mkdir(exist_ok=True)
        
        # Create basic project structure
        src_path = project_path / "src" / project_name
        src_path.mkdir(parents=True, exist_ok=True)
        
        conf_path = project_path / "conf" / "base"
        conf_path.mkdir(parents=True, exist_ok=True)
        
        # Create settings.py with or without FigRegistryHooks
        settings_content = _generate_settings_content(
            include_figregistry_hooks=include_figregistry_hooks,
            include_malformed_settings=include_malformed_settings
        )
        
        settings_path = src_path / "settings.py"
        with open(settings_path, 'w') as f:
            f.write(settings_content)
        
        # Create basic figregistry configuration if hooks are included
        if include_figregistry_hooks:
            figregistry_config = {
                'styles': {
                    'test_registration': {
                        'color': '#1f77b4',
                        'marker': 'o'
                    }
                },
                'outputs': {
                    'base_path': 'test_hook_outputs'
                }
            }
            
            import yaml
            figregistry_path = conf_path / "figregistry.yml"
            with open(figregistry_path, 'w') as f:
                yaml.dump(figregistry_config, f)
        
        project_info = {
            'project_name': project_name,
            'project_path': project_path,
            'src_path': src_path,
            'conf_path': conf_path,
            'settings_path': settings_path,
            'include_hooks': include_figregistry_hooks,
            'include_malformed': include_malformed_settings
        }
        
        registration_simulation['temporary_projects'].append(project_info)
        
        return project_info
    
    def simulate_settings_py_discovery(project_info):
        """
        Simulate Kedro's settings.py hook discovery process.
        
        Args:
            project_info: Project information from create_test_kedro_project
            
        Returns:
            Dictionary containing discovery results and any errors encountered
        """
        discovery_start = time.time()
        
        try:
            # Simulate importing settings.py
            settings_path = project_info['settings_path']
            
            # Read settings file content
            with open(settings_path, 'r') as f:
                settings_content = f.read()
            
            # Parse settings for HOOKS configuration
            hooks_discovered = []
            
            if 'FigRegistryHooks' in settings_content and 'HOOKS' in settings_content:
                # Simulate successful hook discovery
                if HAS_FIGREGISTRY_HOOKS:
                    hook_instance = FigRegistryHooks()
                    hooks_discovered.append({
                        'hook_class': FigRegistryHooks,
                        'hook_instance': hook_instance,
                        'source': 'settings.py',
                        'project': project_info['project_name']
                    })
                else:
                    # Simulate mock hook for testing when actual hooks not available
                    mock_hook = mocker.Mock()
                    mock_hook.__class__.__name__ = 'FigRegistryHooks'
                    hooks_discovered.append({
                        'hook_class': type(mock_hook),
                        'hook_instance': mock_hook,
                        'source': 'settings.py',
                        'project': project_info['project_name']
                    })
            
            discovery_result = {
                'success': True,
                'hooks_discovered': hooks_discovered,
                'discovery_time_ms': (time.time() - discovery_start) * 1000,
                'project': project_info['project_name'],
                'settings_path': str(settings_path)
            }
            
            registration_simulation['discovered_hooks'].extend(hooks_discovered)
            
        except Exception as e:
            discovery_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'discovery_time_ms': (time.time() - discovery_start) * 1000,
                'project': project_info['project_name']
            }
            
            registration_simulation['registration_errors'].append(discovery_result)
        
        return discovery_result
    
    def simulate_entry_point_discovery():
        """
        Simulate entry point based hook discovery for plugin testing.
        
        Returns:
            Dictionary containing entry point discovery results and hook instances
        """
        discovery_start = time.time()
        entry_point_hooks = []
        
        # Simulate entry point discovery process
        for entry_name, hook_info in registration_simulation['entry_point_registrations'].items():
            try:
                # Simulate entry point instantiation
                if HAS_FIGREGISTRY_HOOKS and hook_info['hook_class'] == FigRegistryHooks:
                    hook_instance = FigRegistryHooks()
                else:
                    # Create mock hook for testing
                    hook_instance = mocker.Mock()
                    hook_instance.__class__.__name__ = hook_info['hook_class'].__name__
                
                entry_point_hooks.append({
                    'entry_point_name': entry_name,
                    'hook_class': hook_info['hook_class'],
                    'hook_instance': hook_instance,
                    'source': 'entry_point'
                })
                
            except Exception as e:
                registration_simulation['registration_errors'].append({
                    'source': 'entry_point',
                    'entry_point_name': entry_name,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        discovery_result = {
            'success': True,
            'entry_point_hooks': entry_point_hooks,
            'discovery_time_ms': (time.time() - discovery_start) * 1000,
            'total_entry_points': len(registration_simulation['entry_point_registrations'])
        }
        
        registration_simulation['discovered_hooks'].extend(entry_point_hooks)
        
        return discovery_result
    
    def register_entry_point(entry_point_name, hook_class):
        """
        Register entry point for testing entry point discovery.
        
        Args:
            entry_point_name: Name of the entry point
            hook_class: Hook class to register
        """
        registration_simulation['entry_point_registrations'][entry_point_name] = {
            'hook_class': hook_class,
            'registration_time': time.time()
        }
    
    def validate_hook_registration(hook_instance):
        """
        Validate that a hook instance meets registration requirements.
        
        Args:
            hook_instance: Hook instance to validate
            
        Returns:
            Dictionary containing validation results and compliance checks
        """
        validation_start = time.time()
        
        validation_results = {
            'valid_hook': True,
            'validation_errors': [],
            'hook_methods': [],
            'compliance_checks': {}
        }
        
        # Check for required hook methods
        required_methods = ['before_pipeline_run', 'after_config_loaded']
        optional_methods = ['after_pipeline_run', 'before_node_run', 'after_node_run']
        
        for method_name in required_methods:
            if hasattr(hook_instance, method_name):
                validation_results['hook_methods'].append(method_name)
                validation_results['compliance_checks'][method_name] = 'present'
            else:
                validation_results['valid_hook'] = False
                validation_results['validation_errors'].append(f"Missing required method: {method_name}")
                validation_results['compliance_checks'][method_name] = 'missing'
        
        for method_name in optional_methods:
            if hasattr(hook_instance, method_name):
                validation_results['hook_methods'].append(method_name)
                validation_results['compliance_checks'][method_name] = 'present'
            else:
                validation_results['compliance_checks'][method_name] = 'not_implemented'
        
        # Check method signatures if available
        if HAS_FIGREGISTRY_HOOKS and isinstance(hook_instance, FigRegistryHooks):
            validation_results['signature_validation'] = _validate_hook_signatures(hook_instance)
        else:
            validation_results['signature_validation'] = 'skipped_mock_hook'
        
        validation_results['validation_time_ms'] = (time.time() - validation_start) * 1000
        
        return validation_results
    
    def get_registration_summary():
        """
        Get comprehensive summary of hook registration testing results.
        
        Returns:
            Dictionary containing complete registration testing summary and metrics
        """
        return {
            'temporary_projects': len(registration_simulation['temporary_projects']),
            'settings_configurations': len(registration_simulation['settings_configurations']),
            'entry_point_registrations': len(registration_simulation['entry_point_registrations']),
            'discovered_hooks': len(registration_simulation['discovered_hooks']),
            'registration_errors': len(registration_simulation['registration_errors']),
            'project_details': registration_simulation['temporary_projects'],
            'discovery_errors': registration_simulation['registration_errors'],
            'hook_discovery_breakdown': _analyze_hook_discovery(registration_simulation['discovered_hooks'])
        }
    
    def cleanup_registration_test():
        """Clean up temporary projects and reset registration state."""
        # Cleanup is handled automatically by tmp_path fixture
        registration_simulation['temporary_projects'].clear()
        registration_simulation['settings_configurations'].clear()
        registration_simulation['entry_point_registrations'].clear()
        registration_simulation['registration_errors'].clear()
        registration_simulation['discovered_hooks'].clear()
    
    return {
        'create_test_kedro_project': create_test_kedro_project,
        'simulate_settings_py_discovery': simulate_settings_py_discovery,
        'simulate_entry_point_discovery': simulate_entry_point_discovery,
        'register_entry_point': register_entry_point,
        'validate_hook_registration': validate_hook_registration,
        'get_registration_summary': get_registration_summary,
        'cleanup_registration_test': cleanup_registration_test,
        'registration_state': registration_simulation
    }


# =============================================================================
# Thread Safety Fixtures
# =============================================================================

@pytest.fixture
def thread_safety_fixtures(mocker):
    """
    Thread safety fixtures for parallel runner execution testing with mock session contexts.
    
    Provides comprehensive thread safety testing capabilities for FigRegistryHooks
    operation in parallel pipeline execution environments per Section 5.2.7. The
    fixture supports concurrent hook execution, session context isolation, and
    race condition detection for robust parallel execution validation.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        
    Returns:
        Dictionary containing thread safety testing utilities and concurrent execution simulation
        
    Validates:
        - Thread-safe hook operation for parallel pipeline execution per Section 5.2.7
        - Session context isolation across concurrent pipelines
        - Race condition detection and prevention
        - Resource contention management during parallel execution
    """
    thread_safety_state = {
        'concurrent_executions': [],
        'thread_safety_violations': [],
        'resource_conflicts': [],
        'performance_degradation': [],
        'session_contexts': {}
    }
    
    def create_concurrent_session_contexts(num_contexts=4):
        """
        Create multiple isolated session contexts for concurrent testing.
        
        Args:
            num_contexts: Number of concurrent session contexts to create
            
        Returns:
            List of mock session contexts for parallel execution testing
        """
        session_contexts = []
        
        for i in range(num_contexts):
            context_id = f"concurrent_session_{i}"
            
            # Create isolated mock session context
            mock_session = mocker.Mock()
            mock_session.session_id = context_id
            mock_session.store = mocker.Mock()
            mock_session._project_path = f"/tmp/test_project_{i}"
            
            # Create isolated context
            mock_context = mocker.Mock(spec=KedroContext if HAS_KEDRO else object)
            mock_context.project_name = f"test_project_{i}"
            mock_context.env = "test"
            mock_context.project_path = f"/tmp/test_project_{i}"
            
            # Create isolated config loader
            mock_config_loader = mocker.Mock()
            mock_config_loader.get.return_value = {
                'styles': {
                    f'session_{i}_condition': {
                        'color': f'#{"1f77b4" if i % 2 == 0 else "ff7f0e"}',
                        'marker': 'o' if i % 2 == 0 else 's'
                    }
                },
                'outputs': {
                    'base_path': f'concurrent_outputs_{i}'
                }
            }
            mock_context.config_loader = mock_config_loader
            
            # Create isolated catalog
            mock_catalog = mocker.Mock()
            mock_catalog.list.return_value = [f'concurrent_figure_{i}']
            mock_context.catalog = mock_catalog
            
            # Link session and context
            mock_session.load_context = mocker.Mock(return_value=mock_context)
            
            session_info = {
                'session_id': context_id,
                'session': mock_session,
                'context': mock_context,
                'config_loader': mock_config_loader,
                'catalog': mock_catalog,
                'thread_id': None  # Will be set during execution
            }
            
            session_contexts.append(session_info)
            thread_safety_state['session_contexts'][context_id] = session_info
        
        return session_contexts
    
    def simulate_concurrent_hook_execution(
        session_contexts,
        hook_instance=None,
        execution_scenario='basic_parallel'
    ):
        """
        Simulate concurrent hook execution across multiple session contexts.
        
        Args:
            session_contexts: List of session contexts from create_concurrent_session_contexts
            hook_instance: FigRegistryHooks instance (or mock) for testing
            execution_scenario: Type of concurrent execution to simulate
            
        Returns:
            Dictionary containing concurrent execution results and thread safety analysis
        """
        execution_start = time.time()
        
        # Create or use provided hook instance
        if hook_instance is None:
            if HAS_FIGREGISTRY_HOOKS:
                hook_instance = FigRegistryHooks()
            else:
                hook_instance = mocker.Mock()
                hook_instance.__class__.__name__ = 'FigRegistryHooks'
        
        # Thread-safe execution tracking
        execution_results = []
        execution_lock = threading.Lock()
        
        def execute_hook_lifecycle(session_info):
            """Execute hook lifecycle in individual thread."""
            thread_start = time.time()
            thread_id = threading.current_thread().ident
            session_info['thread_id'] = thread_id
            
            thread_result = {
                'session_id': session_info['session_id'],
                'thread_id': thread_id,
                'start_time': thread_start,
                'events': [],
                'errors': [],
                'performance_metrics': {}
            }
            
            try:
                # Simulate after_config_loaded event
                config_start = time.time()
                if hasattr(hook_instance, 'after_config_loaded'):
                    hook_instance.after_config_loaded(
                        context=session_info['context'],
                        config_loader=session_info['config_loader'],
                        conf_source=f"conf/{session_info['session_id']}"
                    )
                config_duration = (time.time() - config_start) * 1000
                
                thread_result['events'].append({
                    'event': 'after_config_loaded',
                    'duration_ms': config_duration
                })
                thread_result['performance_metrics']['config_loaded_ms'] = config_duration
                
                # Simulate before_pipeline_run event
                pipeline_start = time.time()
                if hasattr(hook_instance, 'before_pipeline_run'):
                    mock_run_params = {
                        'run_id': f"{session_info['session_id']}_run",
                        'pipeline_name': f"concurrent_pipeline_{session_info['session_id']}"
                    }
                    mock_pipeline = mocker.Mock()
                    mock_pipeline.name = f"concurrent_pipeline_{session_info['session_id']}"
                    
                    hook_instance.before_pipeline_run(
                        run_params=mock_run_params,
                        pipeline=mock_pipeline,
                        catalog=session_info['catalog']
                    )
                pipeline_duration = (time.time() - pipeline_start) * 1000
                
                thread_result['events'].append({
                    'event': 'before_pipeline_run',
                    'duration_ms': pipeline_duration
                })
                thread_result['performance_metrics']['pipeline_started_ms'] = pipeline_duration
                
                # Simulate concurrent work delay
                if execution_scenario == 'stress_test':
                    time.sleep(0.01)  # 10ms concurrent work simulation
                elif execution_scenario == 'race_condition_test':
                    time.sleep(0.001)  # 1ms for race condition detection
                
                # Simulate after_pipeline_run event  
                cleanup_start = time.time()
                if hasattr(hook_instance, 'after_pipeline_run'):
                    mock_run_params = {
                        'run_id': f"{session_info['session_id']}_run",
                        'pipeline_name': f"concurrent_pipeline_{session_info['session_id']}"
                    }
                    mock_pipeline = mocker.Mock()
                    mock_pipeline.name = f"concurrent_pipeline_{session_info['session_id']}"
                    
                    hook_instance.after_pipeline_run(
                        run_params=mock_run_params,
                        pipeline=mock_pipeline,
                        catalog=session_info['catalog']
                    )
                cleanup_duration = (time.time() - cleanup_start) * 1000
                
                thread_result['events'].append({
                    'event': 'after_pipeline_run',
                    'duration_ms': cleanup_duration
                })
                thread_result['performance_metrics']['cleanup_ms'] = cleanup_duration
                
            except Exception as e:
                thread_result['errors'].append({
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'timestamp': time.time()
                })
                
                # Check for thread safety violations
                if 'race' in str(e).lower() or 'concurrent' in str(e).lower():
                    with execution_lock:
                        thread_safety_state['thread_safety_violations'].append({
                            'session_id': session_info['session_id'],
                            'thread_id': thread_id,
                            'error': str(e),
                            'scenario': execution_scenario
                        })
            
            thread_result['total_duration_ms'] = (time.time() - thread_start) * 1000
            
            # Thread-safe result storage
            with execution_lock:
                execution_results.append(thread_result)
            
            return thread_result
        
        # Execute concurrent hook lifecycles
        with ThreadPoolExecutor(max_workers=len(session_contexts)) as executor:
            future_to_session = {
                executor.submit(execute_hook_lifecycle, session_info): session_info
                for session_info in session_contexts
            }
            
            for future in as_completed(future_to_session):
                session_info = future_to_session[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                except Exception as e:
                    error_record = {
                        'session_id': session_info['session_id'],
                        'thread_execution_error': str(e),
                        'error_type': type(e).__name__
                    }
                    thread_safety_state['resource_conflicts'].append(error_record)
        
        # Analyze concurrent execution results
        total_execution_time = (time.time() - execution_start) * 1000
        
        concurrent_analysis = _analyze_concurrent_execution(
            execution_results, 
            total_execution_time,
            execution_scenario
        )
        
        execution_record = {
            'execution_scenario': execution_scenario,
            'session_count': len(session_contexts),
            'total_execution_time_ms': total_execution_time,
            'individual_results': execution_results,
            'concurrent_analysis': concurrent_analysis,
            'thread_safety_violations': len(thread_safety_state['thread_safety_violations']),
            'resource_conflicts': len(thread_safety_state['resource_conflicts'])
        }
        
        thread_safety_state['concurrent_executions'].append(execution_record)
        
        return execution_record
    
    def detect_race_conditions(execution_results):
        """
        Analyze execution results for potential race conditions.
        
        Args:
            execution_results: Results from simulate_concurrent_hook_execution
            
        Returns:
            Dictionary containing race condition analysis and detection results
        """
        race_condition_analysis = {
            'potential_races_detected': False,
            'timing_anomalies': [],
            'resource_contentions': [],
            'performance_degradations': []
        }
        
        if not execution_results or 'individual_results' not in execution_results:
            return race_condition_analysis
        
        individual_results = execution_results['individual_results']
        
        # Analyze timing patterns for anomalies
        event_timings = {}
        for result in individual_results:
            for event in result.get('events', []):
                event_name = event['event']
                duration = event['duration_ms']
                
                if event_name not in event_timings:
                    event_timings[event_name] = []
                event_timings[event_name].append(duration)
        
        # Detect timing anomalies that might indicate race conditions
        for event_name, durations in event_timings.items():
            if len(durations) > 1:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
                
                # Check for significant timing variations
                variation_threshold = avg_duration * 2.0  # 200% variation threshold
                
                if max_duration > variation_threshold:
                    race_condition_analysis['timing_anomalies'].append({
                        'event': event_name,
                        'avg_duration_ms': avg_duration,
                        'max_duration_ms': max_duration,
                        'min_duration_ms': min_duration,
                        'variation_ratio': max_duration / min_duration,
                        'potential_race': True
                    })
                    race_condition_analysis['potential_races_detected'] = True
        
        # Check for resource contention indicators
        error_patterns = []
        for result in individual_results:
            for error in result.get('errors', []):
                error_patterns.append(error['error_type'])
        
        # Look for error patterns that suggest resource conflicts
        contention_indicators = ['AttributeError', 'RuntimeError', 'ValueError']
        for indicator in contention_indicators:
            if error_patterns.count(indicator) > 1:
                race_condition_analysis['resource_contentions'].append({
                    'error_type': indicator,
                    'occurrence_count': error_patterns.count(indicator),
                    'potential_contention': True
                })
                race_condition_analysis['potential_races_detected'] = True
        
        return race_condition_analysis
    
    def validate_thread_safety_compliance():
        """
        Validate overall thread safety compliance based on all executed tests.
        
        Returns:
            Dictionary containing comprehensive thread safety compliance assessment
        """
        compliance_assessment = {
            'thread_safe': True,
            'compliance_score': 0.0,
            'violation_count': len(thread_safety_state['thread_safety_violations']),
            'conflict_count': len(thread_safety_state['resource_conflicts']),
            'total_executions': len(thread_safety_state['concurrent_executions']),
            'detailed_violations': thread_safety_state['thread_safety_violations'],
            'performance_assessment': {}
        }
        
        # Calculate compliance score
        total_tests = max(1, len(thread_safety_state['concurrent_executions']))
        violation_ratio = len(thread_safety_state['thread_safety_violations']) / total_tests
        conflict_ratio = len(thread_safety_state['resource_conflicts']) / total_tests
        
        # Score calculation (100 = perfect, 0 = completely unsafe)
        compliance_score = max(0.0, 100.0 - (violation_ratio * 50) - (conflict_ratio * 30))
        compliance_assessment['compliance_score'] = compliance_score
        
        # Determine thread safety status
        if violation_ratio > 0 or conflict_ratio > 0.1:  # More than 10% conflicts
            compliance_assessment['thread_safe'] = False
        
        # Performance assessment across concurrent executions
        if thread_safety_state['concurrent_executions']:
            total_times = [
                exec_result['total_execution_time_ms'] 
                for exec_result in thread_safety_state['concurrent_executions']
            ]
            
            compliance_assessment['performance_assessment'] = {
                'avg_execution_time_ms': sum(total_times) / len(total_times),
                'max_execution_time_ms': max(total_times),
                'min_execution_time_ms': min(total_times),
                'performance_consistent': max(total_times) / min(total_times) < 2.0 if min(total_times) > 0 else False
            }
        
        return compliance_assessment
    
    def reset_thread_safety_state():
        """Reset thread safety testing state for clean test runs."""
        thread_safety_state['concurrent_executions'].clear()
        thread_safety_state['thread_safety_violations'].clear()
        thread_safety_state['resource_conflicts'].clear()
        thread_safety_state['performance_degradation'].clear()
        thread_safety_state['session_contexts'].clear()
    
    return {
        'create_concurrent_session_contexts': create_concurrent_session_contexts,
        'simulate_concurrent_hook_execution': simulate_concurrent_hook_execution,
        'detect_race_conditions': detect_race_conditions,
        'validate_thread_safety_compliance': validate_thread_safety_compliance,
        'reset_thread_safety_state': reset_thread_safety_state,
        'thread_safety_state': thread_safety_state
    }


# =============================================================================
# Hook Performance Fixtures
# =============================================================================

@pytest.fixture
def hook_performance_fixtures(mocker):
    """
    Hook performance fixtures for validating <25ms initialization overhead per Section 6.6.4.3.
    
    Provides comprehensive performance measurement and validation utilities for
    FigRegistryHooks to ensure initialization overhead remains within the <25ms
    target specified in Section 6.6.4.3. The fixture supports detailed timing
    analysis, performance regression detection, and benchmark validation.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        
    Returns:
        Dictionary containing performance testing utilities and validation functions
        
    Validates:
        - Hook initialization overhead <25ms per Section 6.6.4.3
        - Performance consistency across multiple executions
        - Memory usage during hook lifecycle operations
        - Performance regression detection and reporting
    """
    performance_state = {
        'measurement_sessions': [],
        'performance_baselines': {},
        'regression_detections': [],
        'memory_usage_samples': []
    }
    
    def measure_hook_initialization_performance(
        hook_instance=None,
        measurement_iterations=10,
        warm_up_iterations=3
    ):
        """
        Measure FigRegistryHooks initialization performance with statistical analysis.
        
        Args:
            hook_instance: FigRegistryHooks instance (created if None)
            measurement_iterations: Number of performance measurement iterations
            warm_up_iterations: Number of warm-up iterations before measurement
            
        Returns:
            Dictionary containing detailed initialization performance metrics
        """
        measurement_start = time.time()
        
        # Create or use provided hook instance
        if hook_instance is None:
            if HAS_FIGREGISTRY_HOOKS:
                hook_instance = FigRegistryHooks(
                    enable_performance_monitoring=True,
                    initialization_timeout_ms=25.0
                )
            else:
                hook_instance = mocker.Mock()
                hook_instance.__class__.__name__ = 'FigRegistryHooks'
        
        initialization_times = []
        
        # Warm-up iterations (not measured)
        for _ in range(warm_up_iterations):
            try:
                if hasattr(hook_instance, 'clear_state'):
                    hook_instance.clear_state()
                
                # Simulate initialization
                if hasattr(hook_instance, 'after_config_loaded'):
                    mock_context = mocker.Mock()
                    mock_config_loader = mocker.Mock()
                    mock_config_loader.get.return_value = {'test': 'config'}
                    
                    hook_instance.after_config_loaded(
                        context=mock_context,
                        config_loader=mock_config_loader,
                        conf_source="test_warmup"
                    )
            except Exception:
                pass  # Ignore warm-up errors
        
        # Actual performance measurements
        for iteration in range(measurement_iterations):
            try:
                # Clear state for clean measurement
                if hasattr(hook_instance, 'clear_state'):
                    hook_instance.clear_state()
                
                # Measure initialization timing
                init_start = time.perf_counter()
                
                if hasattr(hook_instance, 'after_config_loaded'):
                    mock_context = mocker.Mock()
                    mock_context.env = 'test'
                    mock_context.project_path = '/tmp/test_project'
                    
                    mock_config_loader = mocker.Mock()
                    mock_config_loader.get.return_value = {
                        'styles': {'test': {'color': '#1f77b4'}},
                        'outputs': {'base_path': 'test_outputs'}
                    }
                    
                    hook_instance.after_config_loaded(
                        context=mock_context,
                        config_loader=mock_config_loader,
                        conf_source=f"test_iteration_{iteration}"
                    )
                
                init_duration = (time.perf_counter() - init_start) * 1000  # Convert to ms
                initialization_times.append(init_duration)
                
            except Exception as e:
                # Record failed initialization
                initialization_times.append(float('inf'))  # Mark as failed
        
        # Filter out failed measurements
        valid_times = [t for t in initialization_times if t != float('inf')]
        
        if not valid_times:
            return {
                'measurement_success': False,
                'error': 'All initialization measurements failed',
                'iterations_attempted': measurement_iterations
            }
        
        # Statistical analysis
        avg_time = sum(valid_times) / len(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)
        
        # Calculate percentiles
        sorted_times = sorted(valid_times)
        p50_time = sorted_times[len(sorted_times) // 2]
        p95_time = sorted_times[int(len(sorted_times) * 0.95)]
        p99_time = sorted_times[int(len(sorted_times) * 0.99)]
        
        performance_metrics = {
            'measurement_success': True,
            'total_measurement_time_ms': (time.time() - measurement_start) * 1000,
            'iterations_completed': len(valid_times),
            'iterations_failed': measurement_iterations - len(valid_times),
            'statistics': {
                'average_ms': avg_time,
                'minimum_ms': min_time,
                'maximum_ms': max_time,
                'p50_median_ms': p50_time,
                'p95_ms': p95_time,
                'p99_ms': p99_time,
                'standard_deviation_ms': _calculate_standard_deviation(valid_times)
            },
            'target_compliance': {
                'target_threshold_ms': 25.0,
                'average_within_target': avg_time < 25.0,
                'p95_within_target': p95_time < 25.0,
                'max_within_target': max_time < 25.0,
                'compliance_percentage': len([t for t in valid_times if t < 25.0]) / len(valid_times) * 100
            },
            'raw_measurements': valid_times
        }
        
        performance_state['measurement_sessions'].append(performance_metrics)
        
        return performance_metrics
    
    def measure_hook_lifecycle_performance(hook_instance=None):
        """
        Measure complete hook lifecycle performance including all hook events.
        
        Args:
            hook_instance: FigRegistryHooks instance for lifecycle measurement
            
        Returns:
            Dictionary containing complete lifecycle performance analysis
        """
        lifecycle_start = time.perf_counter()
        
        if hook_instance is None:
            if HAS_FIGREGISTRY_HOOKS:
                hook_instance = FigRegistryHooks(enable_performance_monitoring=True)
            else:
                hook_instance = mocker.Mock()
                hook_instance.__class__.__name__ = 'FigRegistryHooks'
        
        lifecycle_events = {}
        
        # Measure after_config_loaded performance
        if hasattr(hook_instance, 'after_config_loaded'):
            config_start = time.perf_counter()
            
            try:
                mock_context = mocker.Mock()
                mock_context.env = 'test'
                mock_config_loader = mocker.Mock()
                mock_config_loader.get.return_value = {
                    'styles': {'lifecycle_test': {'color': '#1f77b4'}},
                    'outputs': {'base_path': 'lifecycle_outputs'}
                }
                
                hook_instance.after_config_loaded(
                    context=mock_context,
                    config_loader=mock_config_loader,
                    conf_source="lifecycle_test"
                )
                
                lifecycle_events['after_config_loaded'] = {
                    'duration_ms': (time.perf_counter() - config_start) * 1000,
                    'success': True
                }
            except Exception as e:
                lifecycle_events['after_config_loaded'] = {
                    'duration_ms': (time.perf_counter() - config_start) * 1000,
                    'success': False,
                    'error': str(e)
                }
        
        # Measure before_pipeline_run performance
        if hasattr(hook_instance, 'before_pipeline_run'):
            pipeline_start = time.perf_counter()
            
            try:
                mock_run_params = {'run_id': 'lifecycle_test_run', 'pipeline_name': 'test_pipeline'}
                mock_pipeline = mocker.Mock()
                mock_pipeline.name = 'test_pipeline'
                mock_catalog = mocker.Mock()
                mock_catalog.list.return_value = ['test_dataset']
                
                hook_instance.before_pipeline_run(
                    run_params=mock_run_params,
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
                
                lifecycle_events['before_pipeline_run'] = {
                    'duration_ms': (time.perf_counter() - pipeline_start) * 1000,
                    'success': True
                }
            except Exception as e:
                lifecycle_events['before_pipeline_run'] = {
                    'duration_ms': (time.perf_counter() - pipeline_start) * 1000,
                    'success': False,
                    'error': str(e)
                }
        
        # Measure after_pipeline_run performance
        if hasattr(hook_instance, 'after_pipeline_run'):
            cleanup_start = time.perf_counter()
            
            try:
                mock_run_params = {'run_id': 'lifecycle_test_run', 'pipeline_name': 'test_pipeline'}
                mock_pipeline = mocker.Mock()
                mock_pipeline.name = 'test_pipeline'
                mock_catalog = mocker.Mock()
                
                hook_instance.after_pipeline_run(
                    run_params=mock_run_params,
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
                
                lifecycle_events['after_pipeline_run'] = {
                    'duration_ms': (time.perf_counter() - cleanup_start) * 1000,
                    'success': True
                }
            except Exception as e:
                lifecycle_events['after_pipeline_run'] = {
                    'duration_ms': (time.perf_counter() - cleanup_start) * 1000,
                    'success': False,
                    'error': str(e)
                }
        
        total_lifecycle_time = (time.perf_counter() - lifecycle_start) * 1000
        
        # Calculate lifecycle performance metrics
        successful_events = [event for event in lifecycle_events.values() if event['success']]
        total_hook_time = sum(event['duration_ms'] for event in successful_events)
        
        lifecycle_metrics = {
            'total_lifecycle_time_ms': total_lifecycle_time,
            'total_hook_execution_time_ms': total_hook_time,
            'hook_overhead_percentage': (total_hook_time / total_lifecycle_time) * 100 if total_lifecycle_time > 0 else 0,
            'individual_events': lifecycle_events,
            'successful_events': len(successful_events),
            'total_events': len(lifecycle_events),
            'performance_targets': {
                'total_within_25ms': total_hook_time < 25.0,
                'individual_events_analysis': _analyze_individual_event_performance(lifecycle_events)
            }
        }
        
        return lifecycle_metrics
    
    def validate_performance_targets(performance_data):
        """
        Validate performance data against established targets and requirements.
        
        Args:
            performance_data: Performance measurement results
            
        Returns:
            Dictionary containing target validation results and compliance assessment
        """
        validation_results = {
            'targets_met': True,
            'validation_details': {},
            'compliance_score': 0.0,
            'performance_grade': 'unknown'
        }
        
        # Primary target: <25ms initialization overhead
        if 'statistics' in performance_data:
            stats = performance_data['statistics']
            
            validation_results['validation_details']['initialization_overhead'] = {
                'target_ms': 25.0,
                'average_ms': stats['average_ms'],
                'p95_ms': stats['p95_ms'],
                'maximum_ms': stats['maximum_ms'],
                'average_compliant': stats['average_ms'] < 25.0,
                'p95_compliant': stats['p95_ms'] < 25.0,
                'max_compliant': stats['maximum_ms'] < 25.0
            }
            
            # Check if targets are met
            if not (stats['average_ms'] < 25.0 and stats['p95_ms'] < 25.0):
                validation_results['targets_met'] = False
        
        # Secondary target: Performance consistency
        if 'target_compliance' in performance_data:
            compliance = performance_data['target_compliance']
            
            validation_results['validation_details']['consistency'] = {
                'compliance_percentage': compliance['compliance_percentage'],
                'consistency_target': 90.0,  # 90% of measurements should be under target
                'consistency_met': compliance['compliance_percentage'] >= 90.0
            }
            
            if compliance['compliance_percentage'] < 90.0:
                validation_results['targets_met'] = False
        
        # Calculate compliance score
        score_components = []
        
        if 'initialization_overhead' in validation_results['validation_details']:
            overhead = validation_results['validation_details']['initialization_overhead']
            if overhead['average_compliant']:
                score_components.append(40)  # 40 points for average compliance
            if overhead['p95_compliant']:
                score_components.append(30)  # 30 points for p95 compliance
            if overhead['max_compliant']:
                score_components.append(20)  # 20 points for max compliance
        
        if 'consistency' in validation_results['validation_details']:
            consistency = validation_results['validation_details']['consistency']
            if consistency['consistency_met']:
                score_components.append(10)  # 10 points for consistency
        
        validation_results['compliance_score'] = sum(score_components)
        
        # Determine performance grade
        if validation_results['compliance_score'] >= 90:
            validation_results['performance_grade'] = 'excellent'
        elif validation_results['compliance_score'] >= 75:
            validation_results['performance_grade'] = 'good'
        elif validation_results['compliance_score'] >= 60:
            validation_results['performance_grade'] = 'acceptable'
        else:
            validation_results['performance_grade'] = 'poor'
        
        return validation_results
    
    def detect_performance_regressions(current_performance, baseline_name='default'):
        """
        Detect performance regressions by comparing against established baselines.
        
        Args:
            current_performance: Current performance measurement results
            baseline_name: Name of the baseline to compare against
            
        Returns:
            Dictionary containing regression analysis and detection results
        """
        if baseline_name not in performance_state['performance_baselines']:
            # Establish baseline if it doesn't exist
            performance_state['performance_baselines'][baseline_name] = current_performance
            return {
                'baseline_established': True,
                'baseline_name': baseline_name,
                'no_regression_detected': True,
                'message': 'Baseline established for future regression detection'
            }
        
        baseline = performance_state['performance_baselines'][baseline_name]
        regression_analysis = {
            'baseline_established': False,
            'baseline_name': baseline_name,
            'regression_detected': False,
            'performance_changes': {},
            'severity': 'none'
        }
        
        # Compare key performance metrics
        if 'statistics' in both_data_present(current_performance, baseline):
            current_stats = current_performance['statistics']
            baseline_stats = baseline['statistics']
            
            regression_analysis['performance_changes']['average_change'] = {
                'current_ms': current_stats['average_ms'],
                'baseline_ms': baseline_stats['average_ms'],
                'change_ms': current_stats['average_ms'] - baseline_stats['average_ms'],
                'change_percentage': ((current_stats['average_ms'] - baseline_stats['average_ms']) / baseline_stats['average_ms']) * 100 if baseline_stats['average_ms'] > 0 else 0
            }
            
            regression_analysis['performance_changes']['p95_change'] = {
                'current_ms': current_stats['p95_ms'],
                'baseline_ms': baseline_stats['p95_ms'],
                'change_ms': current_stats['p95_ms'] - baseline_stats['p95_ms'],
                'change_percentage': ((current_stats['p95_ms'] - baseline_stats['p95_ms']) / baseline_stats['p95_ms']) * 100 if baseline_stats['p95_ms'] > 0 else 0
            }
            
            # Detect significant regressions
            avg_regression_threshold = 20.0  # 20% degradation threshold
            p95_regression_threshold = 25.0  # 25% degradation threshold
            
            avg_change = regression_analysis['performance_changes']['average_change']['change_percentage']
            p95_change = regression_analysis['performance_changes']['p95_change']['change_percentage']
            
            if avg_change > avg_regression_threshold or p95_change > p95_regression_threshold:
                regression_analysis['regression_detected'] = True
                
                if avg_change > 50.0 or p95_change > 50.0:
                    regression_analysis['severity'] = 'critical'
                elif avg_change > 35.0 or p95_change > 35.0:
                    regression_analysis['severity'] = 'major'
                else:
                    regression_analysis['severity'] = 'minor'
                
                # Record regression detection
                regression_record = {
                    'detection_time': time.time(),
                    'baseline_name': baseline_name,
                    'severity': regression_analysis['severity'],
                    'average_change_percentage': avg_change,
                    'p95_change_percentage': p95_change
                }
                performance_state['regression_detections'].append(regression_record)
        
        return regression_analysis
    
    def get_performance_summary():
        """
        Get comprehensive performance testing summary and historical analysis.
        
        Returns:
            Dictionary containing complete performance testing summary and trends
        """
        return {
            'measurement_sessions': len(performance_state['measurement_sessions']),
            'performance_baselines': list(performance_state['performance_baselines'].keys()),
            'regression_detections': len(performance_state['regression_detections']),
            'historical_sessions': performance_state['measurement_sessions'],
            'regression_history': performance_state['regression_detections'],
            'overall_compliance': _calculate_overall_performance_compliance(performance_state['measurement_sessions'])
        }
    
    def reset_performance_state():
        """Reset performance testing state for clean test runs."""
        performance_state['measurement_sessions'].clear()
        performance_state['performance_baselines'].clear()
        performance_state['regression_detections'].clear()
        performance_state['memory_usage_samples'].clear()
    
    return {
        'measure_hook_initialization_performance': measure_hook_initialization_performance,
        'measure_hook_lifecycle_performance': measure_hook_lifecycle_performance,
        'validate_performance_targets': validate_performance_targets,
        'detect_performance_regressions': detect_performance_regressions,
        'get_performance_summary': get_performance_summary,
        'reset_performance_state': reset_performance_state,
        'performance_state': performance_state
    }


# =============================================================================
# Cleanup Validation Fixtures
# =============================================================================

@pytest.fixture
def cleanup_validation_fixtures(mocker):
    """
    Cleanup validation fixtures for testing resource cleanup during pipeline completion.
    
    Provides comprehensive validation utilities for testing FigRegistryHooks resource
    cleanup behavior during pipeline completion per Section 5.2.7. The fixture validates
    proper cleanup of configuration state, memory resources, file handles, and context
    management to ensure no resource leaks during pipeline execution cycles.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        
    Returns:
        Dictionary containing cleanup validation utilities and resource leak detection
        
    Validates:
        - Resource cleanup during pipeline completion per Section 5.2.7
        - Memory leak detection and prevention
        - Configuration state cleanup and reset
        - File handle and context management cleanup validation
    """
    cleanup_state = {
        'cleanup_sessions': [],
        'resource_leak_detections': [],
        'cleanup_failures': [],
        'memory_snapshots': []
    }
    
    def create_resource_snapshot(hook_instance=None, snapshot_name='default'):
        """
        Create snapshot of current resource usage for cleanup validation.
        
        Args:
            hook_instance: FigRegistryHooks instance to snapshot
            snapshot_name: Name for the resource snapshot
            
        Returns:
            Dictionary containing current resource usage snapshot
        """
        snapshot_time = time.time()
        
        resource_snapshot = {
            'snapshot_name': snapshot_name,
            'timestamp': snapshot_time,
            'hook_instance_id': id(hook_instance) if hook_instance else None,
            'hook_state': {},
            'global_state': {},
            'memory_usage': {},
            'open_resources': []
        }
        
        # Capture hook instance state
        if hook_instance:
            try:
                if hasattr(hook_instance, 'is_initialized'):
                    resource_snapshot['hook_state']['initialized'] = hook_instance.is_initialized()
                
                if hasattr(hook_instance, '_active_contexts'):
                    resource_snapshot['hook_state']['active_contexts'] = len(hook_instance._active_contexts)
                
                if hasattr(hook_instance, '_current_config'):
                    resource_snapshot['hook_state']['has_config'] = hook_instance._current_config is not None
                
                if hasattr(hook_instance, 'get_performance_metrics'):
                    metrics = hook_instance.get_performance_metrics()
                    resource_snapshot['hook_state']['performance_metrics'] = metrics
                
            except Exception as e:
                resource_snapshot['hook_state']['snapshot_error'] = str(e)
        
        # Capture global state if available
        if HAS_FIGREGISTRY_HOOKS and get_global_hook_state:
            try:
                global_state = get_global_hook_state()
                resource_snapshot['global_state'] = global_state
            except Exception as e:
                resource_snapshot['global_state'] = {'error': str(e)}
        
        # Basic memory usage snapshot (simplified)
        try:
            import gc
            resource_snapshot['memory_usage'] = {
                'gc_collect_count': len(gc.collect()),
                'reference_count': len(gc.get_objects()),
                'garbage_count': len(gc.garbage)
            }
        except Exception:
            resource_snapshot['memory_usage'] = {'error': 'Memory snapshot failed'}
        
        cleanup_state['memory_snapshots'].append(resource_snapshot)
        
        return resource_snapshot
    
    def simulate_pipeline_cleanup_sequence(hook_instance=None):
        """
        Simulate complete pipeline cleanup sequence and validate resource cleanup.
        
        Args:
            hook_instance: FigRegistryHooks instance for cleanup testing
            
        Returns:
            Dictionary containing cleanup sequence results and validation
        """
        cleanup_start = time.time()
        
        if hook_instance is None:
            if HAS_FIGREGISTRY_HOOKS:
                hook_instance = FigRegistryHooks(enable_performance_monitoring=True)
            else:
                hook_instance = mocker.Mock()
                hook_instance.__class__.__name__ = 'FigRegistryHooks'
        
        cleanup_sequence = {
            'sequence_id': f"cleanup_{int(cleanup_start)}",
            'hook_instance_id': id(hook_instance),
            'steps': [],
            'resource_snapshots': {},
            'cleanup_success': True,
            'cleanup_errors': []
        }
        
        # Step 1: Take initial resource snapshot
        initial_snapshot = create_resource_snapshot(hook_instance, 'initial')
        cleanup_sequence['resource_snapshots']['initial'] = initial_snapshot
        
        # Step 2: Initialize hook with resources
        try:
            step_start = time.time()
            
            if hasattr(hook_instance, 'after_config_loaded'):
                mock_context = mocker.Mock()
                mock_context.env = 'cleanup_test'
                mock_context.project_path = '/tmp/cleanup_test_project'
                
                mock_config_loader = mocker.Mock()
                mock_config_loader.get.return_value = {
                    'styles': {
                        'cleanup_test_condition': {'color': '#1f77b4', 'marker': 'o'}
                    },
                    'outputs': {'base_path': 'cleanup_test_outputs'}
                }
                
                hook_instance.after_config_loaded(
                    context=mock_context,
                    config_loader=mock_config_loader,
                    conf_source="cleanup_test"
                )
            
            cleanup_sequence['steps'].append({
                'step': 'initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            cleanup_sequence['steps'].append({
                'step': 'initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            cleanup_sequence['cleanup_errors'].append(f"Initialization failed: {e}")
        
        # Step 3: Take post-initialization snapshot
        post_init_snapshot = create_resource_snapshot(hook_instance, 'post_initialization')
        cleanup_sequence['resource_snapshots']['post_initialization'] = post_init_snapshot
        
        # Step 4: Simulate pipeline execution
        try:
            step_start = time.time()
            
            if hasattr(hook_instance, 'before_pipeline_run'):
                mock_run_params = {
                    'run_id': 'cleanup_test_run',
                    'pipeline_name': 'cleanup_test_pipeline'
                }
                mock_pipeline = mocker.Mock()
                mock_pipeline.name = 'cleanup_test_pipeline'
                mock_catalog = mocker.Mock()
                mock_catalog.list.return_value = ['cleanup_test_dataset']
                
                hook_instance.before_pipeline_run(
                    run_params=mock_run_params,
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
            
            cleanup_sequence['steps'].append({
                'step': 'pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            cleanup_sequence['steps'].append({
                'step': 'pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            cleanup_sequence['cleanup_errors'].append(f"Pipeline execution failed: {e}")
        
        # Step 5: Take post-execution snapshot
        post_exec_snapshot = create_resource_snapshot(hook_instance, 'post_execution')
        cleanup_sequence['resource_snapshots']['post_execution'] = post_exec_snapshot
        
        # Step 6: Perform cleanup
        try:
            step_start = time.time()
            
            if hasattr(hook_instance, 'after_pipeline_run'):
                mock_run_params = {
                    'run_id': 'cleanup_test_run',
                    'pipeline_name': 'cleanup_test_pipeline'
                }
                mock_pipeline = mocker.Mock()
                mock_pipeline.name = 'cleanup_test_pipeline'
                mock_catalog = mocker.Mock()
                
                hook_instance.after_pipeline_run(
                    run_params=mock_run_params,
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
            
            # Additional cleanup if available
            if hasattr(hook_instance, 'clear_state'):
                hook_instance.clear_state()
            
            cleanup_sequence['steps'].append({
                'step': 'cleanup',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            cleanup_sequence['steps'].append({
                'step': 'cleanup',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            cleanup_sequence['cleanup_errors'].append(f"Cleanup failed: {e}")
            cleanup_sequence['cleanup_success'] = False
        
        # Step 7: Take final snapshot
        final_snapshot = create_resource_snapshot(hook_instance, 'final')
        cleanup_sequence['resource_snapshots']['final'] = final_snapshot
        
        # Step 8: Clear global state if available
        if HAS_FIGREGISTRY_HOOKS and clear_global_hook_state:
            try:
                clear_global_hook_state()
                cleanup_sequence['steps'].append({
                    'step': 'global_cleanup',
                    'duration_ms': 0,  # Usually instant
                    'success': True
                })
            except Exception as e:
                cleanup_sequence['steps'].append({
                    'step': 'global_cleanup',
                    'duration_ms': 0,
                    'success': False,
                    'error': str(e)
                })
        
        cleanup_sequence['total_cleanup_time_ms'] = (time.time() - cleanup_start) * 1000
        cleanup_state['cleanup_sessions'].append(cleanup_sequence)
        
        return cleanup_sequence
    
    def validate_resource_cleanup(cleanup_sequence):
        """
        Validate that resources were properly cleaned up during pipeline completion.
        
        Args:
            cleanup_sequence: Cleanup sequence results from simulate_pipeline_cleanup_sequence
            
        Returns:
            Dictionary containing resource cleanup validation results
        """
        validation_results = {
            'cleanup_validated': True,
            'validation_errors': [],
            'resource_leak_detected': False,
            'cleanup_completeness': {},
            'performance_validation': {}
        }
        
        snapshots = cleanup_sequence.get('resource_snapshots', {})
        
        if 'initial' not in snapshots or 'final' not in snapshots:
            validation_results['cleanup_validated'] = False
            validation_results['validation_errors'].append('Missing required snapshots for validation')
            return validation_results
        
        initial_snapshot = snapshots['initial']
        final_snapshot = snapshots['final']
        
        # Validate hook state cleanup
        if 'hook_state' in both_snapshots_present(initial_snapshot, final_snapshot):
            initial_state = initial_snapshot['hook_state']
            final_state = final_snapshot['hook_state']
            
            validation_results['cleanup_completeness']['hook_state'] = {
                'initial_initialized': initial_state.get('initialized', False),
                'final_initialized': final_state.get('initialized', False),
                'initial_contexts': initial_state.get('active_contexts', 0),
                'final_contexts': final_state.get('active_contexts', 0),
                'initial_config': initial_state.get('has_config', False),
                'final_config': final_state.get('has_config', False)
            }
            
            # Check for proper state cleanup
            if (final_state.get('active_contexts', 0) > 0 or 
                final_state.get('has_config', False)):
                validation_results['resource_leak_detected'] = True
                validation_results['validation_errors'].append('Hook state not properly cleaned up')
        
        # Validate global state cleanup
        if 'global_state' in both_snapshots_present(initial_snapshot, final_snapshot):
            initial_global = initial_snapshot['global_state']
            final_global = final_snapshot['global_state']
            
            validation_results['cleanup_completeness']['global_state'] = {
                'initial_initialized': initial_global.get('initialized', False),
                'final_initialized': final_global.get('initialized', False),
                'initial_instances': initial_global.get('active_instances', 0),
                'final_instances': final_global.get('active_instances', 0)
            }
            
            # Check for global state cleanup
            if (final_global.get('initialized', False) or 
                final_global.get('active_instances', 0) > 0):
                validation_results['resource_leak_detected'] = True
                validation_results['validation_errors'].append('Global state not properly cleaned up')
        
        # Validate memory usage
        if 'memory_usage' in both_snapshots_present(initial_snapshot, final_snapshot):
            initial_memory = initial_snapshot['memory_usage']
            final_memory = final_snapshot['memory_usage']
            
            # Check for significant memory growth (potential leak)
            if ('reference_count' in initial_memory and 'reference_count' in final_memory):
                ref_growth = final_memory['reference_count'] - initial_memory['reference_count']
                growth_threshold = 100  # Allow some growth, but not excessive
                
                validation_results['cleanup_completeness']['memory'] = {
                    'initial_references': initial_memory['reference_count'],
                    'final_references': final_memory['reference_count'],
                    'reference_growth': ref_growth,
                    'growth_within_threshold': ref_growth < growth_threshold
                }
                
                if ref_growth > growth_threshold:
                    validation_results['resource_leak_detected'] = True
                    validation_results['validation_errors'].append(f'Excessive memory growth detected: {ref_growth} references')
        
        # Validate cleanup performance
        cleanup_steps = cleanup_sequence.get('steps', [])
        cleanup_step = next((step for step in cleanup_steps if step['step'] == 'cleanup'), None)
        
        if cleanup_step:
            validation_results['performance_validation'] = {
                'cleanup_duration_ms': cleanup_step['duration_ms'],
                'cleanup_success': cleanup_step['success'],
                'within_performance_target': cleanup_step['duration_ms'] < 50.0  # 50ms cleanup target
            }
            
            if cleanup_step['duration_ms'] > 50.0:
                validation_results['validation_errors'].append(f'Cleanup took too long: {cleanup_step["duration_ms"]:.2f}ms')
        
        # Record any resource leaks
        if validation_results['resource_leak_detected']:
            leak_record = {
                'detection_time': time.time(),
                'sequence_id': cleanup_sequence.get('sequence_id'),
                'validation_errors': validation_results['validation_errors'],
                'cleanup_completeness': validation_results['cleanup_completeness']
            }
            cleanup_state['resource_leak_detections'].append(leak_record)
        
        # Mark cleanup as failed if validation fails
        if validation_results['validation_errors']:
            validation_results['cleanup_validated'] = False
        
        return validation_results
    
    def detect_memory_leaks(memory_snapshots=None):
        """
        Analyze memory snapshots to detect potential memory leaks.
        
        Args:
            memory_snapshots: List of memory snapshots (uses stored snapshots if None)
            
        Returns:
            Dictionary containing memory leak analysis and detection results
        """
        snapshots = memory_snapshots or cleanup_state['memory_snapshots']
        
        if len(snapshots) < 2:
            return {
                'analysis_possible': False,
                'error': 'Need at least 2 snapshots for leak analysis',
                'snapshots_available': len(snapshots)
            }
        
        leak_analysis = {
            'analysis_possible': True,
            'snapshots_analyzed': len(snapshots),
            'memory_trends': {},
            'potential_leaks': [],
            'leak_severity': 'none'
        }
        
        # Analyze memory trends across snapshots
        reference_counts = []
        garbage_counts = []
        
        for snapshot in snapshots:
            memory_data = snapshot.get('memory_usage', {})
            if 'reference_count' in memory_data:
                reference_counts.append(memory_data['reference_count'])
            if 'garbage_count' in memory_data:
                garbage_counts.append(memory_data['garbage_count'])
        
        # Analyze reference count trends
        if len(reference_counts) >= 2:
            ref_growth = reference_counts[-1] - reference_counts[0]
            avg_growth = ref_growth / (len(reference_counts) - 1)
            
            leak_analysis['memory_trends']['reference_counts'] = {
                'initial_count': reference_counts[0],
                'final_count': reference_counts[-1],
                'total_growth': ref_growth,
                'average_growth_per_snapshot': avg_growth,
                'growth_pattern': 'increasing' if ref_growth > 0 else 'stable' if ref_growth == 0 else 'decreasing'
            }
            
            # Detect potential reference leaks
            leak_threshold = 50  # References
            if ref_growth > leak_threshold:
                leak_analysis['potential_leaks'].append({
                    'leak_type': 'reference_leak',
                    'severity': 'major' if ref_growth > 200 else 'minor',
                    'details': f'Reference count increased by {ref_growth}',
                    'growth': ref_growth
                })
        
        # Analyze garbage collection trends
        if len(garbage_counts) >= 2:
            garbage_growth = garbage_counts[-1] - garbage_counts[0]
            
            leak_analysis['memory_trends']['garbage_counts'] = {
                'initial_count': garbage_counts[0],
                'final_count': garbage_counts[-1],
                'total_growth': garbage_growth
            }
            
            # Detect potential garbage accumulation
            if garbage_growth > 10:  # Garbage objects
                leak_analysis['potential_leaks'].append({
                    'leak_type': 'garbage_accumulation',
                    'severity': 'minor',
                    'details': f'Garbage count increased by {garbage_growth}',
                    'growth': garbage_growth
                })
        
        # Determine overall leak severity
        if leak_analysis['potential_leaks']:
            severities = [leak['severity'] for leak in leak_analysis['potential_leaks']]
            if 'major' in severities:
                leak_analysis['leak_severity'] = 'major'
            else:
                leak_analysis['leak_severity'] = 'minor'
        
        return leak_analysis
    
    def get_cleanup_validation_summary():
        """
        Get comprehensive summary of cleanup validation testing results.
        
        Returns:
            Dictionary containing complete cleanup validation summary and metrics
        """
        return {
            'cleanup_sessions': len(cleanup_state['cleanup_sessions']),
            'resource_leaks_detected': len(cleanup_state['resource_leak_detections']),
            'cleanup_failures': len(cleanup_state['cleanup_failures']),
            'memory_snapshots': len(cleanup_state['memory_snapshots']),
            'session_details': cleanup_state['cleanup_sessions'],
            'leak_detections': cleanup_state['resource_leak_detections'],
            'overall_cleanup_success_rate': _calculate_cleanup_success_rate(cleanup_state['cleanup_sessions'])
        }
    
    def reset_cleanup_validation_state():
        """Reset cleanup validation state for clean test runs."""
        cleanup_state['cleanup_sessions'].clear()
        cleanup_state['resource_leak_detections'].clear()
        cleanup_state['cleanup_failures'].clear()
        cleanup_state['memory_snapshots'].clear()
    
    return {
        'create_resource_snapshot': create_resource_snapshot,
        'simulate_pipeline_cleanup_sequence': simulate_pipeline_cleanup_sequence,
        'validate_resource_cleanup': validate_resource_cleanup,
        'detect_memory_leaks': detect_memory_leaks,
        'get_cleanup_validation_summary': get_cleanup_validation_summary,
        'reset_cleanup_validation_state': reset_cleanup_validation_state,
        'cleanup_state': cleanup_state
    }


# =============================================================================
# Non-Invasive Integration Fixtures
# =============================================================================

@pytest.fixture
def non_invasive_integration_fixtures(mocker, minimal_kedro_context):
    """
    Non-invasive integration fixtures for validating preservation of Kedro's execution model.
    
    Provides comprehensive validation utilities for ensuring FigRegistryHooks integration
    preserves Kedro's native execution model per F-006.2. The fixture validates that
    hook integration does not interfere with normal pipeline execution, maintains
    Kedro's architectural principles, and provides transparent operation.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        minimal_kedro_context: Mock Kedro context fixture
        
    Returns:
        Dictionary containing non-invasive integration validation utilities
        
    Validates:
        - Non-invasive integration preserving Kedro's execution model per F-006.2
        - Transparent hook operation without pipeline interference
        - Kedro architectural principle preservation
        - Normal pipeline execution flow maintenance
    """
    integration_state = {
        'baseline_executions': [],
        'hook_enabled_executions': [],
        'interference_detections': [],
        'architecture_violations': []
    }
    
    def simulate_baseline_kedro_execution(pipeline_name='baseline_test_pipeline'):
        """
        Simulate baseline Kedro pipeline execution without FigRegistryHooks.
        
        Args:
            pipeline_name: Name of the pipeline for baseline testing
            
        Returns:
            Dictionary containing baseline execution metrics and behavior
        """
        execution_start = time.time()
        
        baseline_execution = {
            'execution_id': f"baseline_{int(execution_start)}",
            'pipeline_name': pipeline_name,
            'execution_steps': [],
            'performance_metrics': {},
            'kedro_behaviors': {},
            'execution_success': True
        }
        
        # Step 1: Context initialization
        try:
            step_start = time.time()
            
            mock_context = minimal_kedro_context
            mock_context.project_name = f"baseline_project_{pipeline_name}"
            mock_context.env = 'baseline_test'
            
            baseline_execution['execution_steps'].append({
                'step': 'context_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            baseline_execution['execution_steps'].append({
                'step': 'context_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            baseline_execution['execution_success'] = False
        
        # Step 2: Catalog initialization
        try:
            step_start = time.time()
            
            mock_catalog = mocker.Mock()
            mock_catalog.list.return_value = [f'{pipeline_name}_dataset_1', f'{pipeline_name}_dataset_2']
            mock_catalog.load = mocker.Mock(return_value={'test': 'data'})
            mock_catalog.save = mocker.Mock()
            
            baseline_execution['execution_steps'].append({
                'step': 'catalog_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            baseline_execution['execution_steps'].append({
                'step': 'catalog_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            baseline_execution['execution_success'] = False
        
        # Step 3: Pipeline execution simulation
        try:
            step_start = time.time()
            
            # Simulate node execution
            mock_pipeline = mocker.Mock()
            mock_pipeline.name = pipeline_name
            mock_pipeline.nodes = [
                mocker.Mock(name=f'{pipeline_name}_node_1'),
                mocker.Mock(name=f'{pipeline_name}_node_2')
            ]
            
            # Simulate pipeline run parameters
            run_params = {
                'run_id': f'baseline_run_{int(time.time())}',
                'pipeline_name': pipeline_name,
                'environment': 'baseline_test'
            }
            
            baseline_execution['execution_steps'].append({
                'step': 'pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True,
                'run_params': run_params
            })
            
        except Exception as e:
            baseline_execution['execution_steps'].append({
                'step': 'pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            baseline_execution['execution_success'] = False
        
        # Calculate baseline performance metrics
        successful_steps = [step for step in baseline_execution['execution_steps'] if step['success']]
        total_duration = sum(step['duration_ms'] for step in successful_steps)
        
        baseline_execution['performance_metrics'] = {
            'total_duration_ms': total_duration,
            'successful_steps': len(successful_steps),
            'total_steps': len(baseline_execution['execution_steps']),
            'step_breakdown': {step['step']: step['duration_ms'] for step in successful_steps}
        }
        
        # Record Kedro-specific behaviors
        baseline_execution['kedro_behaviors'] = {
            'context_created': True,
            'catalog_accessible': True,
            'pipeline_executable': True,
            'run_params_available': True,
            'standard_execution_flow': baseline_execution['execution_success']
        }
        
        baseline_execution['total_execution_time_ms'] = (time.time() - execution_start) * 1000
        integration_state['baseline_executions'].append(baseline_execution)
        
        return baseline_execution
    
    def simulate_hook_enabled_execution(
        pipeline_name='hook_enabled_test_pipeline',
        hook_instance=None
    ):
        """
        Simulate Kedro pipeline execution with FigRegistryHooks enabled.
        
        Args:
            pipeline_name: Name of the pipeline for hook-enabled testing
            hook_instance: FigRegistryHooks instance (created if None)
            
        Returns:
            Dictionary containing hook-enabled execution metrics and behavior
        """
        execution_start = time.time()
        
        if hook_instance is None:
            if HAS_FIGREGISTRY_HOOKS:
                hook_instance = FigRegistryHooks(
                    enable_performance_monitoring=True,
                    fallback_on_errors=True
                )
            else:
                hook_instance = mocker.Mock()
                hook_instance.__class__.__name__ = 'FigRegistryHooks'
        
        hook_execution = {
            'execution_id': f"hook_enabled_{int(execution_start)}",
            'pipeline_name': pipeline_name,
            'hook_instance_id': id(hook_instance),
            'execution_steps': [],
            'hook_interventions': [],
            'performance_metrics': {},
            'kedro_behaviors': {},
            'execution_success': True
        }
        
        # Step 1: Context initialization (same as baseline)
        try:
            step_start = time.time()
            
            mock_context = minimal_kedro_context
            mock_context.project_name = f"hook_enabled_project_{pipeline_name}"
            mock_context.env = 'hook_test'
            
            hook_execution['execution_steps'].append({
                'step': 'context_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            hook_execution['execution_steps'].append({
                'step': 'context_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            hook_execution['execution_success'] = False
        
        # Step 2: Hook config loading intervention
        try:
            step_start = time.time()
            
            mock_config_loader = mocker.Mock()
            mock_config_loader.get.return_value = {
                'styles': {'hook_test_condition': {'color': '#1f77b4'}},
                'outputs': {'base_path': 'hook_test_outputs'}
            }
            
            # Simulate after_config_loaded hook
            if hasattr(hook_instance, 'after_config_loaded'):
                hook_instance.after_config_loaded(
                    context=mock_context,
                    config_loader=mock_config_loader,
                    conf_source="hook_test"
                )
            
            hook_execution['execution_steps'].append({
                'step': 'hook_config_loading',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
            hook_execution['hook_interventions'].append({
                'hook_method': 'after_config_loaded',
                'timing': 'before_catalog_initialization',
                'duration_ms': (time.time() - step_start) * 1000
            })
            
        except Exception as e:
            hook_execution['execution_steps'].append({
                'step': 'hook_config_loading',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            hook_execution['execution_success'] = False
        
        # Step 3: Catalog initialization (potentially modified by hooks)
        try:
            step_start = time.time()
            
            mock_catalog = mocker.Mock()
            mock_catalog.list.return_value = [f'{pipeline_name}_dataset_1', f'{pipeline_name}_figure_dataset']
            mock_catalog.load = mocker.Mock(return_value={'test': 'data'})
            mock_catalog.save = mocker.Mock()
            
            hook_execution['execution_steps'].append({
                'step': 'catalog_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
        except Exception as e:
            hook_execution['execution_steps'].append({
                'step': 'catalog_initialization',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            hook_execution['execution_success'] = False
        
        # Step 4: Before pipeline run hook intervention
        try:
            step_start = time.time()
            
            run_params = {
                'run_id': f'hook_enabled_run_{int(time.time())}',
                'pipeline_name': pipeline_name,
                'environment': 'hook_test'
            }
            
            mock_pipeline = mocker.Mock()
            mock_pipeline.name = pipeline_name
            mock_pipeline.nodes = [
                mocker.Mock(name=f'{pipeline_name}_node_1'),
                mocker.Mock(name=f'{pipeline_name}_node_2')
            ]
            
            # Simulate before_pipeline_run hook
            if hasattr(hook_instance, 'before_pipeline_run'):
                hook_instance.before_pipeline_run(
                    run_params=run_params,
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
            
            hook_execution['execution_steps'].append({
                'step': 'before_pipeline_hook',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
            hook_execution['hook_interventions'].append({
                'hook_method': 'before_pipeline_run',
                'timing': 'before_pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000
            })
            
        except Exception as e:
            hook_execution['execution_steps'].append({
                'step': 'before_pipeline_hook',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            hook_execution['execution_success'] = False
        
        # Step 5: Pipeline execution (should be unchanged)
        try:
            step_start = time.time()
            
            # Simulate normal pipeline execution
            # This should proceed exactly as in baseline
            
            hook_execution['execution_steps'].append({
                'step': 'pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True,
                'run_params': run_params
            })
            
        except Exception as e:
            hook_execution['execution_steps'].append({
                'step': 'pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            hook_execution['execution_success'] = False
        
        # Step 6: After pipeline run hook intervention
        try:
            step_start = time.time()
            
            # Simulate after_pipeline_run hook
            if hasattr(hook_instance, 'after_pipeline_run'):
                hook_instance.after_pipeline_run(
                    run_params=run_params,
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
            
            hook_execution['execution_steps'].append({
                'step': 'after_pipeline_hook',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': True
            })
            
            hook_execution['hook_interventions'].append({
                'hook_method': 'after_pipeline_run',
                'timing': 'after_pipeline_execution',
                'duration_ms': (time.time() - step_start) * 1000
            })
            
        except Exception as e:
            hook_execution['execution_steps'].append({
                'step': 'after_pipeline_hook',
                'duration_ms': (time.time() - step_start) * 1000,
                'success': False,
                'error': str(e)
            })
            hook_execution['execution_success'] = False
        
        # Calculate performance metrics
        successful_steps = [step for step in hook_execution['execution_steps'] if step['success']]
        total_duration = sum(step['duration_ms'] for step in successful_steps)
        hook_overhead = sum(intervention['duration_ms'] for intervention in hook_execution['hook_interventions'])
        
        hook_execution['performance_metrics'] = {
            'total_duration_ms': total_duration,
            'successful_steps': len(successful_steps),
            'total_steps': len(hook_execution['execution_steps']),
            'hook_overhead_ms': hook_overhead,
            'hook_overhead_percentage': (hook_overhead / total_duration) * 100 if total_duration > 0 else 0,
            'step_breakdown': {step['step']: step['duration_ms'] for step in successful_steps}
        }
        
        # Record Kedro behavior preservation
        hook_execution['kedro_behaviors'] = {
            'context_created': True,
            'catalog_accessible': True,
            'pipeline_executable': True,
            'run_params_available': True,
            'standard_execution_flow': hook_execution['execution_success'],
            'hook_interference_detected': False  # Will be updated by analysis
        }
        
        hook_execution['total_execution_time_ms'] = (time.time() - execution_start) * 1000
        integration_state['hook_enabled_executions'].append(hook_execution)
        
        return hook_execution
    
    def compare_execution_behaviors(baseline_execution, hook_execution):
        """
        Compare baseline and hook-enabled executions to detect interference.
        
        Args:
            baseline_execution: Baseline execution results
            hook_execution: Hook-enabled execution results
            
        Returns:
            Dictionary containing behavior comparison and interference analysis
        """
        comparison_analysis = {
            'non_invasive_compliance': True,
            'behavior_differences': [],
            'performance_impact': {},
            'architecture_preservation': {},
            'interference_detected': False
        }
        
        # Compare core Kedro behaviors
        baseline_behaviors = baseline_execution.get('kedro_behaviors', {})
        hook_behaviors = hook_execution.get('kedro_behaviors', {})
        
        behavior_keys = set(baseline_behaviors.keys()) | set(hook_behaviors.keys())
        
        for behavior_key in behavior_keys:
            baseline_value = baseline_behaviors.get(behavior_key, None)
            hook_value = hook_behaviors.get(behavior_key, None)
            
            if baseline_value != hook_value:
                comparison_analysis['behavior_differences'].append({
                    'behavior': behavior_key,
                    'baseline_value': baseline_value,
                    'hook_value': hook_value,
                    'interference_level': 'critical' if behavior_key == 'standard_execution_flow' else 'minor'
                })
                
                if behavior_key == 'standard_execution_flow':
                    comparison_analysis['interference_detected'] = True
                    comparison_analysis['non_invasive_compliance'] = False
        
        # Compare performance impact
        baseline_metrics = baseline_execution.get('performance_metrics', {})
        hook_metrics = hook_execution.get('performance_metrics', {})
        
        if 'total_duration_ms' in both_metrics_present(baseline_metrics, hook_metrics):
            baseline_time = baseline_metrics['total_duration_ms']
            hook_time = hook_metrics['total_duration_ms']
            overhead_time = hook_metrics.get('hook_overhead_ms', 0)
            
            performance_impact = {
                'baseline_duration_ms': baseline_time,
                'hook_enabled_duration_ms': hook_time,
                'absolute_overhead_ms': hook_time - baseline_time,
                'relative_overhead_percentage': ((hook_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0,
                'hook_overhead_ms': overhead_time,
                'hook_overhead_percentage': hook_metrics.get('hook_overhead_percentage', 0)
            }
            
            comparison_analysis['performance_impact'] = performance_impact
            
            # Check for excessive overhead (>10% performance impact threshold)
            if performance_impact['relative_overhead_percentage'] > 10.0:
                comparison_analysis['behavior_differences'].append({
                    'behavior': 'performance_overhead',
                    'baseline_value': baseline_time,
                    'hook_value': hook_time,
                    'interference_level': 'major'
                })
                comparison_analysis['interference_detected'] = True
                comparison_analysis['non_invasive_compliance'] = False
        
        # Compare execution step structures
        baseline_steps = [step['step'] for step in baseline_execution.get('execution_steps', [])]
        hook_steps = [step['step'] for step in hook_execution.get('execution_steps', []) 
                     if not step['step'].startswith('hook_') and not step['step'].endswith('_hook')]
        
        if baseline_steps != hook_steps:
            comparison_analysis['behavior_differences'].append({
                'behavior': 'execution_flow_structure',
                'baseline_value': baseline_steps,
                'hook_value': hook_steps,
                'interference_level': 'major'
            })
            comparison_analysis['interference_detected'] = True
            comparison_analysis['non_invasive_compliance'] = False
        
        # Validate architecture preservation
        comparison_analysis['architecture_preservation'] = {
            'kedro_context_unchanged': baseline_behaviors.get('context_created') == hook_behaviors.get('context_created'),
            'catalog_accessibility_unchanged': baseline_behaviors.get('catalog_accessible') == hook_behaviors.get('catalog_accessible'),
            'pipeline_executability_unchanged': baseline_behaviors.get('pipeline_executable') == hook_behaviors.get('pipeline_executable'),
            'execution_flow_preserved': baseline_steps == hook_steps
        }
        
        # Record interference if detected
        if comparison_analysis['interference_detected']:
            interference_record = {
                'detection_time': time.time(),
                'baseline_execution_id': baseline_execution.get('execution_id'),
                'hook_execution_id': hook_execution.get('execution_id'),
                'behavior_differences': comparison_analysis['behavior_differences'],
                'performance_impact': comparison_analysis['performance_impact']
            }
            integration_state['interference_detections'].append(interference_record)
        
        return comparison_analysis
    
    def validate_kedro_architecture_preservation():
        """
        Validate that Kedro's architectural principles are preserved with hooks enabled.
        
        Returns:
            Dictionary containing architectural preservation validation results
        """
        if not integration_state['baseline_executions'] or not integration_state['hook_enabled_executions']:
            return {
                'validation_possible': False,
                'error': 'Need both baseline and hook-enabled executions for validation',
                'baseline_count': len(integration_state['baseline_executions']),
                'hook_enabled_count': len(integration_state['hook_enabled_executions'])
            }
        
        architecture_validation = {
            'validation_possible': True,
            'architecture_preserved': True,
            'preservation_score': 0.0,
            'architectural_principles': {},
            'violations_detected': []
        }
        
        # Compare most recent executions
        latest_baseline = integration_state['baseline_executions'][-1]
        latest_hook_enabled = integration_state['hook_enabled_executions'][-1]
        
        comparison = compare_execution_behaviors(latest_baseline, latest_hook_enabled)
        
        # Validate key architectural principles
        principles = {
            'execution_flow_preservation': comparison['architecture_preservation']['execution_flow_preserved'],
            'context_integrity': comparison['architecture_preservation']['kedro_context_unchanged'],
            'catalog_accessibility': comparison['architecture_preservation']['catalog_accessibility_unchanged'],
            'pipeline_executability': comparison['architecture_preservation']['pipeline_executability_unchanged'],
            'minimal_performance_impact': comparison['performance_impact'].get('relative_overhead_percentage', 0) < 5.0 if 'performance_impact' in comparison else True
        }
        
        architecture_validation['architectural_principles'] = principles
        
        # Calculate preservation score
        preserved_principles = sum(1 for preserved in principles.values() if preserved)
        total_principles = len(principles)
        preservation_score = (preserved_principles / total_principles) * 100
        
        architecture_validation['preservation_score'] = preservation_score
        
        # Check for violations
        for principle, preserved in principles.items():
            if not preserved:
                architecture_validation['violations_detected'].append({
                    'principle': principle,
                    'violation_type': 'architectural_integrity',
                    'severity': 'critical' if principle in ['execution_flow_preservation', 'context_integrity'] else 'minor'
                })
                
                if principle in ['execution_flow_preservation', 'context_integrity']:
                    architecture_validation['architecture_preserved'] = False
        
        # Record violations if any
        if architecture_validation['violations_detected']:
            violation_record = {
                'detection_time': time.time(),
                'preservation_score': preservation_score,
                'violations': architecture_validation['violations_detected'],
                'baseline_execution': latest_baseline.get('execution_id'),
                'hook_execution': latest_hook_enabled.get('execution_id')
            }
            integration_state['architecture_violations'].append(violation_record)
        
        return architecture_validation
    
    def get_non_invasive_integration_summary():
        """
        Get comprehensive summary of non-invasive integration validation results.
        
        Returns:
            Dictionary containing complete integration validation summary
        """
        return {
            'baseline_executions': len(integration_state['baseline_executions']),
            'hook_enabled_executions': len(integration_state['hook_enabled_executions']),
            'interference_detections': len(integration_state['interference_detections']),
            'architecture_violations': len(integration_state['architecture_violations']),
            'integration_compliance': len(integration_state['interference_detections']) == 0 and len(integration_state['architecture_violations']) == 0,
            'execution_details': {
                'baseline_executions': integration_state['baseline_executions'],
                'hook_enabled_executions': integration_state['hook_enabled_executions']
            },
            'violation_details': {
                'interference_detections': integration_state['interference_detections'],
                'architecture_violations': integration_state['architecture_violations']
            }
        }
    
    def reset_non_invasive_integration_state():
        """Reset non-invasive integration testing state for clean test runs."""
        integration_state['baseline_executions'].clear()
        integration_state['hook_enabled_executions'].clear()
        integration_state['interference_detections'].clear()
        integration_state['architecture_violations'].clear()
    
    return {
        'simulate_baseline_kedro_execution': simulate_baseline_kedro_execution,
        'simulate_hook_enabled_execution': simulate_hook_enabled_execution,
        'compare_execution_behaviors': compare_execution_behaviors,
        'validate_kedro_architecture_preservation': validate_kedro_architecture_preservation,
        'get_non_invasive_integration_summary': get_non_invasive_integration_summary,
        'reset_non_invasive_integration_state': reset_non_invasive_integration_state,
        'integration_state': integration_state
    }


# =============================================================================
# Comprehensive Hook Fixture Collections
# =============================================================================

@pytest.fixture
def complete_hook_testing_suite(
    mock_hook_manager,
    hook_lifecycle_events,
    hook_registration_fixtures,
    thread_safety_fixtures,
    hook_performance_fixtures,
    cleanup_validation_fixtures,
    non_invasive_integration_fixtures
):
    """
    Complete hook testing suite combining all hook validation fixtures.
    
    Provides comprehensive testing capabilities for FigRegistryHooks validation
    by combining all individual hook testing fixtures into a unified interface.
    This suite enables complete hook system validation through a single fixture
    that provides access to all hook testing capabilities.
    
    Args:
        All individual hook testing fixtures
        
    Returns:
        Dictionary containing unified access to all hook testing utilities
        
    Validates:
        - Complete hook lifecycle testing per Section 5.2.7
        - Hook registration and plugin discovery per F-006.2
        - Thread safety for parallel execution per Section 5.2.7
        - Performance compliance <25ms initialization per Section 6.6.4.3
        - Resource cleanup validation and leak detection
        - Non-invasive integration preserving Kedro execution model per F-006.2
    """
    return {
        'hook_manager': mock_hook_manager,
        'lifecycle_events': hook_lifecycle_events,
        'registration': hook_registration_fixtures,
        'thread_safety': thread_safety_fixtures,
        'performance': hook_performance_fixtures,
        'cleanup_validation': cleanup_validation_fixtures,
        'non_invasive_integration': non_invasive_integration_fixtures,
        
        # Unified testing utilities
        'run_complete_hook_validation': lambda hook_instance=None: _run_complete_hook_validation(
            hook_instance,
            mock_hook_manager,
            hook_lifecycle_events,
            hook_registration_fixtures,
            thread_safety_fixtures,
            hook_performance_fixtures,
            cleanup_validation_fixtures,
            non_invasive_integration_fixtures
        ),
        
        'get_comprehensive_test_report': lambda: _generate_comprehensive_test_report(
            mock_hook_manager,
            hook_lifecycle_events,
            hook_registration_fixtures,
            thread_safety_fixtures,
            hook_performance_fixtures,
            cleanup_validation_fixtures,
            non_invasive_integration_fixtures
        ),
        
        'reset_all_testing_state': lambda: _reset_all_testing_state(
            hook_lifecycle_events,
            hook_registration_fixtures,
            thread_safety_fixtures,
            hook_performance_fixtures,
            cleanup_validation_fixtures,
            non_invasive_integration_fixtures
        )
    }


# =============================================================================
# Utility Helper Functions
# =============================================================================

def _extract_hook_methods(hook_instance):
    """Extract hook method names from hook instance."""
    hook_methods = []
    
    for attr_name in dir(hook_instance):
        if not attr_name.startswith('_') and callable(getattr(hook_instance, attr_name)):
            # Check for hook-like method names
            if any(hook_pattern in attr_name for hook_pattern in ['before_', 'after_', '_hook']):
                hook_methods.append(attr_name)
    
    return hook_methods


def _generate_performance_summary(hook_invocation_history):
    """Generate performance summary from hook invocation history."""
    if not hook_invocation_history:
        return {'summary': 'No invocations recorded'}
    
    total_invocations = len(hook_invocation_history)
    total_time = sum(invocation['total_duration_ms'] for invocation in hook_invocation_history)
    avg_time = total_time / total_invocations if total_invocations > 0 else 0
    
    return {
        'total_invocations': total_invocations,
        'total_time_ms': total_time,
        'average_time_ms': avg_time,
        'performance_grade': 'excellent' if avg_time < 5.0 else 'good' if avg_time < 15.0 else 'acceptable' if avg_time < 25.0 else 'poor'
    }


def _generate_settings_content(include_figregistry_hooks=True, include_malformed_settings=False):
    """Generate settings.py content for testing."""
    base_content = '''"""Project settings."""

'''
    
    if include_figregistry_hooks:
        if include_malformed_settings:
            # Malformed settings for error testing
            base_content += '''# Malformed import for testing
from figregistry_kedro.hooks import NonExistentHooks

HOOKS = (NonExistentHooks(),)
'''
        else:
            # Proper settings
            base_content += '''from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
'''
    else:
        base_content += '''HOOKS = ()
'''
    
    return base_content


def _validate_hook_signatures(hook_instance):
    """Validate hook method signatures for compliance."""
    signature_validation = {
        'valid_signatures': True,
        'method_signatures': {},
        'signature_errors': []
    }
    
    # Check expected hook method signatures
    expected_signatures = {
        'after_config_loaded': ['context', 'config_loader', 'conf_source'],
        'before_pipeline_run': ['run_params', 'pipeline', 'catalog'],
        'after_pipeline_run': ['run_params', 'pipeline', 'catalog']
    }
    
    for method_name, expected_params in expected_signatures.items():
        if hasattr(hook_instance, method_name):
            method = getattr(hook_instance, method_name)
            
            try:
                import inspect
                signature = inspect.signature(method)
                actual_params = list(signature.parameters.keys())
                
                # Remove 'self' parameter
                if 'self' in actual_params:
                    actual_params.remove('self')
                
                signature_validation['method_signatures'][method_name] = {
                    'expected_params': expected_params,
                    'actual_params': actual_params,
                    'signature_match': set(expected_params).issubset(set(actual_params))
                }
                
                if not set(expected_params).issubset(set(actual_params)):
                    signature_validation['valid_signatures'] = False
                    signature_validation['signature_errors'].append(
                        f"Method {method_name} missing expected parameters: {set(expected_params) - set(actual_params)}"
                    )
                    
            except Exception as e:
                signature_validation['signature_errors'].append(f"Failed to inspect {method_name}: {e}")
    
    return signature_validation


def _analyze_hook_discovery(discovered_hooks):
    """Analyze hook discovery results for breakdown."""
    discovery_breakdown = {
        'by_source': {},
        'by_hook_type': {},
        'total_discovered': len(discovered_hooks)
    }
    
    for hook_info in discovered_hooks:
        source = hook_info.get('source', 'unknown')
        hook_type = hook_info.get('hook_class', type(None)).__name__
        
        if source not in discovery_breakdown['by_source']:
            discovery_breakdown['by_source'][source] = 0
        discovery_breakdown['by_source'][source] += 1
        
        if hook_type not in discovery_breakdown['by_hook_type']:
            discovery_breakdown['by_hook_type'][hook_type] = 0
        discovery_breakdown['by_hook_type'][hook_type] += 1
    
    return discovery_breakdown


def _analyze_concurrent_execution(execution_results, total_execution_time, execution_scenario):
    """Analyze concurrent execution results for thread safety."""
    analysis = {
        'execution_scenario': execution_scenario,
        'total_threads': len(execution_results),
        'successful_executions': len([r for r in execution_results if not r.get('errors')]),
        'failed_executions': len([r for r in execution_results if r.get('errors')]),
        'performance_analysis': {},
        'thread_safety_assessment': 'safe'
    }
    
    if execution_results:
        # Performance analysis
        execution_times = [r['total_duration_ms'] for r in execution_results]
        analysis['performance_analysis'] = {
            'average_execution_time_ms': sum(execution_times) / len(execution_times),
            'min_execution_time_ms': min(execution_times),
            'max_execution_time_ms': max(execution_times),
            'time_variation_ratio': max(execution_times) / min(execution_times) if min(execution_times) > 0 else float('inf')
        }
        
        # Thread safety assessment
        error_count = analysis['failed_executions']
        if error_count > 0:
            error_ratio = error_count / len(execution_results)
            if error_ratio > 0.5:
                analysis['thread_safety_assessment'] = 'unsafe'
            elif error_ratio > 0.2:
                analysis['thread_safety_assessment'] = 'questionable'
            else:
                analysis['thread_safety_assessment'] = 'mostly_safe'
    
    return analysis


def _calculate_standard_deviation(values):
    """Calculate standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _analyze_individual_event_performance(lifecycle_events):
    """Analyze individual event performance for targets."""
    event_analysis = {}
    
    for event_name, event_data in lifecycle_events.items():
        if event_data['success']:
            duration = event_data['duration_ms']
            event_analysis[event_name] = {
                'duration_ms': duration,
                'within_5ms_target': duration < 5.0,
                'within_25ms_target': duration < 25.0,
                'performance_grade': 'excellent' if duration < 5.0 else 'good' if duration < 15.0 else 'acceptable' if duration < 25.0 else 'poor'
            }
        else:
            event_analysis[event_name] = {
                'duration_ms': event_data['duration_ms'],
                'within_5ms_target': False,
                'within_25ms_target': False,
                'performance_grade': 'failed',
                'error': event_data.get('error')
            }
    
    return event_analysis


def both_data_present(data1, data2):
    """Check if both data dictionaries have the required keys."""
    return data1 and data2


def both_metrics_present(metrics1, metrics2):
    """Check if both metrics dictionaries are present."""
    return metrics1 and metrics2


def both_snapshots_present(snapshot1, snapshot2):
    """Check if both snapshots are present."""
    return snapshot1 and snapshot2


def _calculate_overall_performance_compliance(measurement_sessions):
    """Calculate overall performance compliance across sessions."""
    if not measurement_sessions:
        return {'compliance_rate': 0.0, 'sessions_analyzed': 0}
    
    compliant_sessions = 0
    total_sessions = len(measurement_sessions)
    
    for session in measurement_sessions:
        if session.get('measurement_success') and session.get('target_compliance', {}).get('average_within_target'):
            compliant_sessions += 1
    
    return {
        'compliance_rate': (compliant_sessions / total_sessions) * 100,
        'sessions_analyzed': total_sessions,
        'compliant_sessions': compliant_sessions
    }


def _calculate_cleanup_success_rate(cleanup_sessions):
    """Calculate cleanup success rate across sessions."""
    if not cleanup_sessions:
        return {'success_rate': 0.0, 'sessions_analyzed': 0}
    
    successful_cleanups = len([session for session in cleanup_sessions if session.get('cleanup_success')])
    total_sessions = len(cleanup_sessions)
    
    return {
        'success_rate': (successful_cleanups / total_sessions) * 100,
        'sessions_analyzed': total_sessions,
        'successful_cleanups': successful_cleanups
    }


def _run_complete_hook_validation(
    hook_instance,
    mock_hook_manager,
    hook_lifecycle_events,
    hook_registration_fixtures,
    thread_safety_fixtures,
    hook_performance_fixtures,
    cleanup_validation_fixtures,
    non_invasive_integration_fixtures
):
    """Run complete hook validation using all testing fixtures."""
    validation_results = {
        'validation_start_time': time.time(),
        'hook_instance_id': id(hook_instance) if hook_instance else None,
        'validation_phases': {},
        'overall_success': True,
        'critical_failures': []
    }
    
    # Phase 1: Hook Registration Validation
    try:
        if hook_instance:
            registration_validation = hook_registration_fixtures['validate_hook_registration'](hook_instance)
            validation_results['validation_phases']['registration'] = registration_validation
            
            if not registration_validation['valid_hook']:
                validation_results['overall_success'] = False
                validation_results['critical_failures'].append('Hook registration validation failed')
    except Exception as e:
        validation_results['validation_phases']['registration'] = {'error': str(e)}
        validation_results['overall_success'] = False
        validation_results['critical_failures'].append(f'Registration validation error: {e}')
    
    # Phase 2: Performance Validation
    try:
        performance_results = hook_performance_fixtures['measure_hook_initialization_performance'](hook_instance)
        validation_results['validation_phases']['performance'] = performance_results
        
        if not performance_results.get('measurement_success') or not performance_results.get('target_compliance', {}).get('average_within_target'):
            validation_results['overall_success'] = False
            validation_results['critical_failures'].append('Performance targets not met')
    except Exception as e:
        validation_results['validation_phases']['performance'] = {'error': str(e)}
        validation_results['overall_success'] = False
        validation_results['critical_failures'].append(f'Performance validation error: {e}')
    
    # Phase 3: Thread Safety Validation
    try:
        session_contexts = thread_safety_fixtures['create_concurrent_session_contexts'](4)
        concurrent_results = thread_safety_fixtures['simulate_concurrent_hook_execution'](session_contexts, hook_instance)
        validation_results['validation_phases']['thread_safety'] = concurrent_results
        
        if concurrent_results.get('thread_safety_violations', 0) > 0:
            validation_results['overall_success'] = False
            validation_results['critical_failures'].append('Thread safety violations detected')
    except Exception as e:
        validation_results['validation_phases']['thread_safety'] = {'error': str(e)}
        validation_results['overall_success'] = False
        validation_results['critical_failures'].append(f'Thread safety validation error: {e}')
    
    # Phase 4: Cleanup Validation
    try:
        cleanup_results = cleanup_validation_fixtures['simulate_pipeline_cleanup_sequence'](hook_instance)
        cleanup_validation = cleanup_validation_fixtures['validate_resource_cleanup'](cleanup_results)
        validation_results['validation_phases']['cleanup'] = cleanup_validation
        
        if not cleanup_validation['cleanup_validated']:
            validation_results['overall_success'] = False
            validation_results['critical_failures'].append('Resource cleanup validation failed')
    except Exception as e:
        validation_results['validation_phases']['cleanup'] = {'error': str(e)}
        validation_results['overall_success'] = False
        validation_results['critical_failures'].append(f'Cleanup validation error: {e}')
    
    # Phase 5: Non-Invasive Integration Validation
    try:
        baseline_execution = non_invasive_integration_fixtures['simulate_baseline_kedro_execution']()
        hook_execution = non_invasive_integration_fixtures['simulate_hook_enabled_execution'](hook_instance=hook_instance)
        integration_comparison = non_invasive_integration_fixtures['compare_execution_behaviors'](baseline_execution, hook_execution)
        validation_results['validation_phases']['non_invasive_integration'] = integration_comparison
        
        if not integration_comparison['non_invasive_compliance']:
            validation_results['overall_success'] = False
            validation_results['critical_failures'].append('Non-invasive integration requirements not met')
    except Exception as e:
        validation_results['validation_phases']['non_invasive_integration'] = {'error': str(e)}
        validation_results['overall_success'] = False
        validation_results['critical_failures'].append(f'Non-invasive integration validation error: {e}')
    
    validation_results['validation_duration_ms'] = (time.time() - validation_results['validation_start_time']) * 1000
    
    return validation_results


def _generate_comprehensive_test_report(
    mock_hook_manager,
    hook_lifecycle_events,
    hook_registration_fixtures,
    thread_safety_fixtures,
    hook_performance_fixtures,
    cleanup_validation_fixtures,
    non_invasive_integration_fixtures
):
    """Generate comprehensive test report from all fixtures."""
    report = {
        'report_generation_time': time.time(),
        'fixture_summaries': {},
        'overall_test_health': 'healthy',
        'recommendations': []
    }
    
    # Collect summaries from each fixture
    try:
        report['fixture_summaries']['registration'] = hook_registration_fixtures['get_registration_summary']()
    except Exception as e:
        report['fixture_summaries']['registration'] = {'error': str(e)}
    
    try:
        report['fixture_summaries']['performance'] = hook_performance_fixtures['get_performance_summary']()
    except Exception as e:
        report['fixture_summaries']['performance'] = {'error': str(e)}
    
    try:
        report['fixture_summaries']['thread_safety'] = thread_safety_fixtures['validate_thread_safety_compliance']()
    except Exception as e:
        report['fixture_summaries']['thread_safety'] = {'error': str(e)}
    
    try:
        report['fixture_summaries']['cleanup'] = cleanup_validation_fixtures['get_cleanup_validation_summary']()
    except Exception as e:
        report['fixture_summaries']['cleanup'] = {'error': str(e)}
    
    try:
        report['fixture_summaries']['non_invasive_integration'] = non_invasive_integration_fixtures['get_non_invasive_integration_summary']()
    except Exception as e:
        report['fixture_summaries']['non_invasive_integration'] = {'error': str(e)}
    
    # Analyze overall test health
    error_count = sum(1 for summary in report['fixture_summaries'].values() if 'error' in summary)
    
    if error_count > 0:
        report['overall_test_health'] = 'degraded' if error_count < 3 else 'unhealthy'
        report['recommendations'].append(f'Address {error_count} fixture errors for complete test coverage')
    
    # Add specific recommendations based on summaries
    performance_summary = report['fixture_summaries'].get('performance', {})
    if isinstance(performance_summary, dict) and performance_summary.get('measurement_sessions', 0) == 0:
        report['recommendations'].append('Run performance measurements to establish baselines')
    
    thread_safety_summary = report['fixture_summaries'].get('thread_safety', {})
    if isinstance(thread_safety_summary, dict) and not thread_safety_summary.get('thread_safe', True):
        report['recommendations'].append('Critical: Address thread safety violations before production use')
    
    return report


def _reset_all_testing_state(
    hook_lifecycle_events,
    hook_registration_fixtures,
    thread_safety_fixtures,
    hook_performance_fixtures,
    cleanup_validation_fixtures,
    non_invasive_integration_fixtures
):
    """Reset all testing fixture state for clean test runs."""
    try:
        hook_lifecycle_events['reset_lifecycle_state']()
    except Exception:
        pass
    
    try:
        hook_registration_fixtures['cleanup_registration_test']()
    except Exception:
        pass
    
    try:
        thread_safety_fixtures['reset_thread_safety_state']()
    except Exception:
        pass
    
    try:
        hook_performance_fixtures['reset_performance_state']()
    except Exception:
        pass
    
    try:
        cleanup_validation_fixtures['reset_cleanup_validation_state']()
    except Exception:
        pass
    
    try:
        non_invasive_integration_fixtures['reset_non_invasive_integration_state']()
    except Exception:
        pass


# Export public fixtures
__all__ = [
    'mock_hook_manager',
    'mock_hook_manager_with_discovery',
    'hook_lifecycle_events',
    'hook_registration_fixtures', 
    'thread_safety_fixtures',
    'hook_performance_fixtures',
    'cleanup_validation_fixtures',
    'non_invasive_integration_fixtures',
    'complete_hook_testing_suite'
]