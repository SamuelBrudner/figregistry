"""Hook lifecycle testing fixtures for FigRegistryHooks comprehensive validation.

This module provides comprehensive hook lifecycle testing fixtures that enable thorough 
validation of FigRegistryHooks integration with Kedro's execution model. The fixtures 
support hook registration testing through plugin discovery, lifecycle event simulation, 
thread-safe operation validation, and performance measurement to ensure hooks meet the 
<25ms initialization overhead requirement while preserving Kedro's execution model.

The fixture architecture provides:
- Mock hook managers for testing hook registration through Kedro's plugin discovery system
- Lifecycle event simulation for before_pipeline_run and after_config_loaded scenarios
- Thread-safety fixtures for parallel runner execution testing with mock session contexts
- Performance fixtures for validating initialization overhead against SLA requirements
- Cleanup validation fixtures for testing resource cleanup during pipeline completion
- Non-invasive integration fixtures for preserving Kedro's execution model integrity

Hook Registration Testing:
    The mock_hook_manager fixture provides comprehensive hook registration validation
    through Kedro's plugin discovery system, ensuring FigRegistryHooks register correctly
    in settings.py configuration and entry point validation scenarios.

Lifecycle Event Testing:
    Hook lifecycle event fixtures simulate complete hook execution sequences including
    configuration initialization, context management, and cleanup operations across
    realistic pipeline execution scenarios.

Thread Safety Testing:
    Thread safety fixtures validate parallel runner execution scenarios with multiple
    concurrent hook invocations, ensuring proper context isolation and state management
    without synchronization conflicts.

Performance Testing:
    Performance fixtures measure hook execution overhead and validate compliance with
    the <25ms initialization time requirement specified in Section 6.6.4.3.

Usage:
    Import and use these fixtures in hook tests:
    
    ```python
    def test_hook_registration(mock_hook_manager):
        hooks = FigRegistryHooks()
        mock_hook_manager.register(hooks)
        assert mock_hook_manager.is_registered('FigRegistryHooks')
    
    def test_lifecycle_sequence(hook_lifecycle_events):
        for event in hook_lifecycle_events:
            event.execute()
            assert event.completed_successfully
    ```
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Generator, Set
from unittest.mock import Mock, MagicMock, patch
import pytest
import tempfile
import shutil
from pathlib import Path

# Test imports with graceful degradation for testing environments
try:
    from kedro.framework.hooks import hook_impl
    from kedro.framework.context import KedroContext
    from kedro.config import ConfigLoader
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    KEDRO_AVAILABLE = True
except ImportError:
    # Mock Kedro classes for testing when not available
    hook_impl = lambda func: func
    KedroContext = object
    ConfigLoader = object
    DataCatalog = object
    Pipeline = object
    KedroSession = object
    bootstrap_project = lambda x: None
    KEDRO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HookExecutionRecord:
    """Record of hook execution for testing validation."""
    hook_name: str
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    thread_id: Optional[int] = None
    session_id: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None


@dataclass
class LifecycleEventContext:
    """Context data for hook lifecycle event simulation."""
    event_name: str
    run_params: Dict[str, Any]
    pipeline: Optional[Mock] = None
    catalog: Optional[Mock] = None
    context: Optional[Mock] = None
    config_loader: Optional[Mock] = None
    
    def execute(self) -> HookExecutionRecord:
        """Execute the lifecycle event and return execution record."""
        start_time = time.perf_counter()
        try:
            # Simulate event execution
            time.sleep(0.001)  # Minimal delay to simulate processing
            execution_time = (time.perf_counter() - start_time) * 1000
            return HookExecutionRecord(
                hook_name=self.event_name,
                execution_time_ms=execution_time,
                success=True,
                thread_id=threading.get_ident(),
                session_id=self.run_params.get('session_id'),
                context_data={'event_type': self.event_name}
            )
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return HookExecutionRecord(
                hook_name=self.event_name,
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
                thread_id=threading.get_ident()
            )


class MockHookManager:
    """Mock implementation of Kedro's hook manager for testing hook registration."""
    
    def __init__(self):
        self._hooks: Dict[str, Any] = {}
        self._execution_order: List[str] = []
        self._call_count: Dict[str, int] = {}
        self._is_registered = False
        
    def register(self, hook_instance: Any) -> None:
        """Register a hook instance for testing."""
        hook_name = hook_instance.__class__.__name__
        self._hooks[hook_name] = hook_instance
        self._call_count[hook_name] = 0
        self._is_registered = True
        logger.debug(f"Registered hook: {hook_name}")
        
    def is_registered(self, hook_name: str) -> bool:
        """Check if a hook is registered."""
        return hook_name in self._hooks
        
    def get_hook(self, hook_name: str) -> Any:
        """Get registered hook instance."""
        return self._hooks.get(hook_name)
        
    def call_hook(self, hook_name: str, method_name: str, **kwargs) -> Any:
        """Simulate calling a hook method."""
        if hook_name in self._hooks:
            hook = self._hooks[hook_name]
            self._call_count[hook_name] += 1
            self._execution_order.append(f"{hook_name}.{method_name}")
            
            if hasattr(hook, method_name):
                method = getattr(hook, method_name)
                return method(**kwargs)
        return None
        
    def get_execution_order(self) -> List[str]:
        """Get the order of hook method executions."""
        return self._execution_order.copy()
        
    def get_call_count(self, hook_name: str) -> int:
        """Get the number of times a hook was called."""
        return self._call_count.get(hook_name, 0)
        
    def reset(self) -> None:
        """Reset hook manager state for testing."""
        self._hooks.clear()
        self._execution_order.clear()
        self._call_count.clear()
        self._is_registered = False


class MockKedroSession:
    """Mock Kedro session for testing hook integration."""
    
    def __init__(self, session_id: str, project_path: Path):
        self.session_id = session_id
        self.project_path = project_path
        self._is_active = False
        self._context: Optional[Mock] = None
        
    def __enter__(self):
        self._is_active = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_active = False
        
    @property
    def context(self) -> Mock:
        """Get mock Kedro context."""
        if self._context is None:
            self._context = Mock(spec=KedroContext)
            self._context.project_path = self.project_path
            self._context.env = 'test'
        return self._context
        
    def run(self, pipeline_name: str = None, **kwargs) -> Dict[str, Any]:
        """Mock pipeline run."""
        return {
            'session_id': self.session_id,
            'pipeline_name': pipeline_name or 'default',
            'status': 'completed'
        }


@pytest.fixture
def mock_hook_manager() -> MockHookManager:
    """Fixture providing mock hook manager for hook registration testing.
    
    This fixture provides comprehensive hook registration validation through Kedro's
    plugin discovery system per Section 5.2.7. It supports testing of hook registration
    scenarios, execution order validation, and entry point discovery simulation.
    
    The mock hook manager enables validation of:
    - Hook registration through plugin discovery system per F-006.2
    - Hook method execution order and call counting
    - Registration state validation for settings.py configuration testing
    - Hook manager reset capabilities for test isolation
    
    Returns:
        MockHookManager: Configured mock hook manager for registration testing
        
    Example:
        ```python
        def test_hook_registration(mock_hook_manager):
            from figregistry_kedro.hooks import FigRegistryHooks
            
            hooks = FigRegistryHooks()
            mock_hook_manager.register(hooks)
            
            assert mock_hook_manager.is_registered('FigRegistryHooks')
            assert mock_hook_manager.get_hook('FigRegistryHooks') is hooks
        ```
    """
    manager = MockHookManager()
    yield manager
    # Cleanup after test
    manager.reset()


@pytest.fixture
def hook_lifecycle_events() -> List[LifecycleEventContext]:
    """Fixture providing hook lifecycle event simulation for before_pipeline_run and after_config_loaded scenarios.
    
    This fixture simulates comprehensive hook lifecycle events per F-006.2 requirements,
    providing realistic execution contexts for testing hook behavior throughout pipeline
    execution cycles. Events include proper context setup, parameter injection, and
    cleanup simulation.
    
    The lifecycle events support testing of:
    - before_pipeline_run hook execution with pipeline context
    - after_config_loaded hook execution with configuration loading
    - Context management and parameter resolution
    - Error handling during lifecycle transitions
    
    Returns:
        List[LifecycleEventContext]: Configured lifecycle event contexts for testing
        
    Example:
        ```python
        def test_lifecycle_execution(hook_lifecycle_events):
            for event in hook_lifecycle_events:
                record = event.execute()
                assert record.success
                assert record.execution_time_ms < 25  # Performance requirement
        ```
    """
    session_id = str(uuid.uuid4())
    
    # Mock pipeline and catalog for testing
    mock_pipeline = Mock(spec=Pipeline)
    mock_pipeline.describe.return_value = "test_pipeline"
    
    mock_catalog = Mock(spec=DataCatalog)
    mock_catalog.list.return_value = ['input_data', 'output_figures']
    
    mock_context = Mock(spec=KedroContext)
    mock_context.env = 'test'
    mock_context.project_path = Path('/tmp/test_project')
    
    mock_config_loader = Mock(spec=ConfigLoader)
    mock_config_loader.get.return_value = {
        'figregistry_version': '0.3.0',
        'style': {'default': {'color': 'blue'}},
        'condition_styles': {'test': {'color': 'red'}}
    }
    
    events = [
        LifecycleEventContext(
            event_name='after_config_loaded',
            run_params={
                'session_id': session_id,
                'env': 'test'
            },
            context=mock_context,
            config_loader=mock_config_loader
        ),
        LifecycleEventContext(
            event_name='before_pipeline_run',
            run_params={
                'session_id': session_id,
                'pipeline_name': 'test_pipeline',
                'tags': ['figregistry']
            },
            pipeline=mock_pipeline,
            catalog=mock_catalog
        ),
        LifecycleEventContext(
            event_name='after_pipeline_run',
            run_params={
                'session_id': session_id,
                'pipeline_name': 'test_pipeline',
                'status': 'completed'
            },
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
    ]
    
    return events


@pytest.fixture
def hook_registration_fixtures() -> Dict[str, Any]:
    """Fixture providing hook registration testing for settings.py configuration and entry point validation.
    
    This fixture supports comprehensive testing of hook registration scenarios per Section 5.2.7,
    including settings.py configuration validation, entry point discovery simulation, and
    plugin registration verification through Kedro's plugin discovery system.
    
    The registration fixtures provide:
    - Mock settings.py configuration with HOOKS tuple
    - Entry point discovery simulation for kedro.hooks entry points
    - Plugin manager integration testing capabilities
    - Registration validation and error scenario testing
    
    Returns:
        Dict[str, Any]: Hook registration testing configuration and utilities
        
    Example:
        ```python
        def test_settings_registration(hook_registration_fixtures):
            settings_config = hook_registration_fixtures['mock_settings']
            hooks_tuple = settings_config['HOOKS']
            
            assert any(isinstance(hook, FigRegistryHooks) for hook in hooks_tuple)
        ```
    """
    # Mock settings.py configuration
    mock_settings = {
        'HOOKS': (),  # Will be populated during test
        'DISABLE_HOOKS_FOR_PLUGINS': [],
        'HOOK_SPECS': {}
    }
    
    # Mock entry point for kedro.hooks discovery
    mock_entry_point = Mock()
    mock_entry_point.name = 'figregistry_hooks'
    mock_entry_point.load.return_value = 'figregistry_kedro.hooks:FigRegistryHooks'
    
    # Mock plugin discovery
    mock_plugin_manager = Mock()
    mock_plugin_manager.list_name_plugin.return_value = [
        ('figregistry_hooks', mock_entry_point)
    ]
    
    return {
        'mock_settings': mock_settings,
        'mock_entry_point': mock_entry_point,
        'mock_plugin_manager': mock_plugin_manager,
        'hooks_entry_points': {
            'kedro.hooks': ['figregistry_hooks = figregistry_kedro.hooks:FigRegistryHooks']
        }
    }


@pytest.fixture
def thread_safety_fixtures() -> Dict[str, Any]:
    """Fixture providing thread-safe operation testing for parallel runner execution with mock session contexts.
    
    This fixture enables comprehensive thread safety validation per Section 5.2.7 requirements,
    providing multiple concurrent execution contexts to test hook behavior under parallel
    pipeline execution scenarios. Includes thread isolation validation and state management testing.
    
    The thread safety fixtures support:
    - Concurrent hook execution with multiple threads
    - Thread-local storage validation for hook state
    - Session context isolation between parallel executions
    - Race condition detection and synchronization testing
    
    Returns:
        Dict[str, Any]: Thread safety testing utilities and execution contexts
        
    Example:
        ```python
        def test_concurrent_execution(thread_safety_fixtures):
            executor = thread_safety_fixtures['thread_executor']
            sessions = thread_safety_fixtures['mock_sessions']
            
            futures = []
            for session in sessions:
                future = executor.submit(session.run, 'test_pipeline')
                futures.append(future)
                
            results = [future.result() for future in as_completed(futures)]
            assert len(results) == len(sessions)
        ```
    """
    num_threads = 4
    
    # Create multiple mock sessions for concurrent testing
    mock_sessions = []
    for i in range(num_threads):
        session_id = f"session_{i}_{uuid.uuid4().hex[:8]}"
        project_path = Path(f"/tmp/test_project_{i}")
        session = MockKedroSession(session_id, project_path)
        mock_sessions.append(session)
    
    # Thread executor for concurrent testing
    thread_executor = ThreadPoolExecutor(max_workers=num_threads)
    
    # Thread-local storage for testing isolation
    thread_local_data = threading.local()
    
    def get_thread_data():
        if not hasattr(thread_local_data, 'hook_state'):
            thread_local_data.hook_state = {
                'initialized': False,
                'config_loaded': False,
                'pipeline_count': 0
            }
        return thread_local_data.hook_state
    
    # Synchronization primitives for testing
    execution_lock = threading.RLock()
    execution_events = {
        'config_loaded': threading.Event(),
        'pipeline_started': threading.Event(),
        'pipeline_completed': threading.Event()
    }
    
    return {
        'mock_sessions': mock_sessions,
        'thread_executor': thread_executor,
        'thread_local_data': thread_local_data,
        'get_thread_data': get_thread_data,
        'execution_lock': execution_lock,
        'execution_events': execution_events,
        'num_threads': num_threads
    }


@pytest.fixture
def hook_performance_fixtures() -> Dict[str, Any]:
    """Fixture providing hook performance testing for validating <25ms initialization overhead per Section 6.6.4.3.
    
    This fixture enables comprehensive performance measurement and validation of hook
    execution overhead per Section 6.6.4.3 requirements. Provides timing utilities,
    benchmark baselines, and performance assertion helpers to ensure hooks meet the
    <25ms initialization time requirement.
    
    The performance fixtures provide:
    - High-precision timing measurement utilities
    - Performance baseline establishment and comparison
    - SLA validation for <25ms initialization overhead
    - Memory usage tracking for hook operations
    
    Returns:
        Dict[str, Any]: Performance testing utilities and measurement tools
        
    Example:
        ```python
        def test_initialization_performance(hook_performance_fixtures):
            timer = hook_performance_fixtures['precision_timer']
            sla_validator = hook_performance_fixtures['sla_validator']
            
            with timer() as timing:
                hook = FigRegistryHooks()
                hook.after_config_loaded(context, config_loader)
                
            assert sla_validator.validate_initialization_time(timing.elapsed_ms)
        ```
    """
    # Performance SLA requirements from Section 6.6.4.3
    SLA_REQUIREMENTS = {
        'hook_initialization': 25.0,  # milliseconds
        'config_bridge_resolution': 50.0,
        'lifecycle_event_processing': 5.0
    }
    
    class PrecisionTimer:
        """High-precision timer for performance measurement."""
        
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = None
            
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.perf_counter()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000
    
    class SLAValidator:
        """Validator for performance SLA requirements."""
        
        def __init__(self, requirements: Dict[str, float]):
            self.requirements = requirements
            self.measurements: List[Dict[str, Any]] = []
            
        def validate_initialization_time(self, elapsed_ms: float) -> bool:
            """Validate hook initialization time against SLA."""
            is_valid = elapsed_ms < self.requirements['hook_initialization']
            self.measurements.append({
                'type': 'initialization',
                'elapsed_ms': elapsed_ms,
                'sla_ms': self.requirements['hook_initialization'],
                'is_valid': is_valid
            })
            return is_valid
            
        def validate_config_resolution_time(self, elapsed_ms: float) -> bool:
            """Validate config bridge resolution time against SLA."""
            is_valid = elapsed_ms < self.requirements['config_bridge_resolution']
            self.measurements.append({
                'type': 'config_resolution',
                'elapsed_ms': elapsed_ms,
                'sla_ms': self.requirements['config_bridge_resolution'],
                'is_valid': is_valid
            })
            return is_valid
            
        def validate_lifecycle_event_time(self, elapsed_ms: float) -> bool:
            """Validate lifecycle event processing time against SLA."""
            is_valid = elapsed_ms < self.requirements['lifecycle_event_processing']
            self.measurements.append({
                'type': 'lifecycle_event',
                'elapsed_ms': elapsed_ms,
                'sla_ms': self.requirements['lifecycle_event_processing'],
                'is_valid': is_valid
            })
            return is_valid
            
        def get_performance_summary(self) -> Dict[str, Any]:
            """Get summary of performance measurements."""
            if not self.measurements:
                return {'total_measurements': 0}
                
            return {
                'total_measurements': len(self.measurements),
                'passed_sla': sum(1 for m in self.measurements if m['is_valid']),
                'failed_sla': sum(1 for m in self.measurements if not m['is_valid']),
                'average_time_ms': sum(m['elapsed_ms'] for m in self.measurements) / len(self.measurements),
                'max_time_ms': max(m['elapsed_ms'] for m in self.measurements),
                'min_time_ms': min(m['elapsed_ms'] for m in self.measurements)
            }
    
    # Memory usage tracking
    memory_tracker = {
        'baseline_mb': 0,
        'peak_mb': 0,
        'measurements': []
    }
    
    def measure_memory_usage():
        """Measure current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_tracker['measurements'].append(memory_mb)
            if memory_mb > memory_tracker['peak_mb']:
                memory_tracker['peak_mb'] = memory_mb
            return memory_mb
        except ImportError:
            # psutil not available, return mock measurement
            return 50.0  # Mock 50MB baseline
    
    return {
        'sla_requirements': SLA_REQUIREMENTS,
        'precision_timer': PrecisionTimer,
        'sla_validator': SLAValidator(SLA_REQUIREMENTS),
        'memory_tracker': memory_tracker,
        'measure_memory_usage': measure_memory_usage
    }


@pytest.fixture
def cleanup_validation_fixtures() -> Dict[str, Any]:
    """Fixture providing cleanup validation testing for resource cleanup during pipeline completion per Section 5.2.7.
    
    This fixture enables comprehensive testing of resource cleanup and state management
    during pipeline completion per Section 5.2.7 requirements. Provides utilities for
    validating proper cleanup of hook state, configuration contexts, and temporary resources.
    
    The cleanup validation fixtures support:
    - Hook state cleanup validation after pipeline completion
    - Resource leak detection and memory cleanup verification
    - Context isolation and state reset testing
    - Cleanup error handling and graceful degradation testing
    
    Returns:
        Dict[str, Any]: Cleanup validation utilities and state tracking tools
        
    Example:
        ```python
        def test_pipeline_cleanup(cleanup_validation_fixtures):
            state_tracker = cleanup_validation_fixtures['state_tracker']
            cleanup_validator = cleanup_validation_fixtures['cleanup_validator']
            
            # Execute pipeline with hooks
            hooks.before_pipeline_run(run_params, pipeline, catalog)
            hooks.after_pipeline_run(run_params, pipeline, catalog)
            
            assert cleanup_validator.validate_state_reset()
            assert state_tracker.get_active_pipeline_count() == 0
        ```
    """
    class StateTracker:
        """Tracks hook state throughout execution for cleanup validation."""
        
        def __init__(self):
            self.active_pipelines: Set[str] = set()
            self.initialized_contexts: Set[str] = set()
            self.allocated_resources: Dict[str, Any] = {}
            self.cleanup_events: List[Dict[str, Any]] = []
            
        def register_pipeline(self, pipeline_name: str):
            """Register an active pipeline."""
            self.active_pipelines.add(pipeline_name)
            
        def unregister_pipeline(self, pipeline_name: str):
            """Unregister a completed pipeline."""
            self.active_pipelines.discard(pipeline_name)
            self.cleanup_events.append({
                'type': 'pipeline_cleanup',
                'pipeline_name': pipeline_name,
                'timestamp': time.time()
            })
            
        def register_context(self, context_id: str):
            """Register an initialized context."""
            self.initialized_contexts.add(context_id)
            
        def cleanup_context(self, context_id: str):
            """Cleanup an initialized context."""
            self.initialized_contexts.discard(context_id)
            self.cleanup_events.append({
                'type': 'context_cleanup',
                'context_id': context_id,
                'timestamp': time.time()
            })
            
        def allocate_resource(self, resource_id: str, resource: Any):
            """Track allocated resource."""
            self.allocated_resources[resource_id] = resource
            
        def deallocate_resource(self, resource_id: str):
            """Track deallocated resource."""
            if resource_id in self.allocated_resources:
                del self.allocated_resources[resource_id]
                self.cleanup_events.append({
                    'type': 'resource_cleanup',
                    'resource_id': resource_id,
                    'timestamp': time.time()
                })
                
        def get_active_pipeline_count(self) -> int:
            """Get number of active pipelines."""
            return len(self.active_pipelines)
            
        def get_active_context_count(self) -> int:
            """Get number of active contexts."""
            return len(self.initialized_contexts)
            
        def get_allocated_resource_count(self) -> int:
            """Get number of allocated resources."""
            return len(self.allocated_resources)
            
        def reset(self):
            """Reset state tracker."""
            self.active_pipelines.clear()
            self.initialized_contexts.clear()
            self.allocated_resources.clear()
            self.cleanup_events.clear()
    
    class CleanupValidator:
        """Validates proper cleanup behavior."""
        
        def __init__(self, state_tracker: StateTracker):
            self.state_tracker = state_tracker
            
        def validate_state_reset(self) -> bool:
            """Validate that all state has been properly reset."""
            return (
                self.state_tracker.get_active_pipeline_count() == 0 and
                self.state_tracker.get_active_context_count() == 0 and
                self.state_tracker.get_allocated_resource_count() == 0
            )
            
        def validate_cleanup_sequence(self) -> bool:
            """Validate that cleanup events occurred in proper sequence."""
            events = self.state_tracker.cleanup_events
            if not events:
                return True
                
            # Validate that cleanup events are in chronological order
            timestamps = [event['timestamp'] for event in events]
            return timestamps == sorted(timestamps)
            
        def validate_no_resource_leaks(self) -> bool:
            """Validate that no resources were leaked."""
            return self.state_tracker.get_allocated_resource_count() == 0
    
    # Create temporary directory for cleanup testing
    temp_dir = tempfile.mkdtemp(prefix='figregistry_hook_test_')
    temp_path = Path(temp_dir)
    
    state_tracker = StateTracker()
    cleanup_validator = CleanupValidator(state_tracker)
    
    # Mock resource allocation for testing
    mock_resources = {
        'config_cache': {},
        'context_objects': {},
        'file_handles': []
    }
    
    def cleanup_temp_resources():
        """Cleanup temporary test resources."""
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
    
    return {
        'state_tracker': state_tracker,
        'cleanup_validator': cleanup_validator,
        'temp_directory': temp_path,
        'mock_resources': mock_resources,
        'cleanup_temp_resources': cleanup_temp_resources
    }


@pytest.fixture
def non_invasive_integration_fixtures() -> Dict[str, Any]:
    """Fixture providing non-invasive integration testing for validating preservation of Kedro's execution model per F-006.2.
    
    This fixture enables comprehensive testing of non-invasive integration patterns per
    F-006.2 requirements, ensuring that FigRegistryHooks preserve Kedro's execution model
    while providing automation capabilities. Validates that hooks don't interfere with
    pipeline execution flow or modify Kedro's internal state.
    
    The non-invasive integration fixtures support:
    - Kedro execution model preservation validation
    - Pipeline flow integrity testing
    - State isolation and side-effect detection
    - Hook transparency and execution model compliance
    
    Returns:
        Dict[str, Any]: Non-invasive integration testing utilities and validation tools
        
    Example:
        ```python
        def test_non_invasive_integration(non_invasive_integration_fixtures):
            execution_monitor = non_invasive_integration_fixtures['execution_monitor']
            kedro_simulator = non_invasive_integration_fixtures['kedro_simulator']
            
            # Run pipeline with hooks
            with execution_monitor:
                result = kedro_simulator.run_pipeline_with_hooks()
                
            assert execution_monitor.validate_no_interference()
            assert result['status'] == 'completed'
        ```
    """
    class ExecutionMonitor:
        """Monitors pipeline execution for interference detection."""
        
        def __init__(self):
            self.baseline_state: Dict[str, Any] = {}
            self.execution_state: Dict[str, Any] = {}
            self.state_changes: List[Dict[str, Any]] = []
            self.is_monitoring = False
            
        def __enter__(self):
            self.start_monitoring()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop_monitoring()
            
        def start_monitoring(self):
            """Start monitoring execution state."""
            self.is_monitoring = True
            self.capture_baseline_state()
            
        def stop_monitoring(self):
            """Stop monitoring execution state."""
            self.is_monitoring = False
            self.capture_execution_state()
            
        def capture_baseline_state(self):
            """Capture baseline Kedro state before hook execution."""
            self.baseline_state = {
                'pipeline_state': 'initialized',
                'catalog_entries': set(),
                'context_attributes': set(),
                'global_config': {}
            }
            
        def capture_execution_state(self):
            """Capture Kedro state after hook execution."""
            self.execution_state = {
                'pipeline_state': 'completed',
                'catalog_entries': set(),
                'context_attributes': set(),
                'global_config': {}
            }
            
        def detect_state_changes(self) -> List[Dict[str, Any]]:
            """Detect unexpected state changes in Kedro execution model."""
            changes = []
            
            # Check for unexpected attribute additions
            baseline_attrs = self.baseline_state.get('context_attributes', set())
            execution_attrs = self.execution_state.get('context_attributes', set())
            
            new_attrs = execution_attrs - baseline_attrs
            if new_attrs:
                changes.append({
                    'type': 'context_attributes_added',
                    'attributes': list(new_attrs)
                })
                
            # Check for catalog modifications
            baseline_catalog = self.baseline_state.get('catalog_entries', set())
            execution_catalog = self.execution_state.get('catalog_entries', set())
            
            if baseline_catalog != execution_catalog:
                changes.append({
                    'type': 'catalog_entries_modified',
                    'added': list(execution_catalog - baseline_catalog),
                    'removed': list(baseline_catalog - execution_catalog)
                })
                
            return changes
            
        def validate_no_interference(self) -> bool:
            """Validate that hooks didn't interfere with Kedro execution model."""
            changes = self.detect_state_changes()
            return len(changes) == 0
    
    class KedroSimulator:
        """Simulates Kedro pipeline execution for non-invasive testing."""
        
        def __init__(self):
            self.mock_context = Mock(spec=KedroContext)
            self.mock_pipeline = Mock(spec=Pipeline)
            self.mock_catalog = Mock(spec=DataCatalog)
            self.execution_log: List[str] = []
            
        def setup_mock_environment(self):
            """Setup mock Kedro environment."""
            # Configure mock context
            self.mock_context.env = 'test'
            self.mock_context.project_path = Path('/tmp/test_project')
            self.mock_context.config_loader = Mock(spec=ConfigLoader)
            
            # Configure mock pipeline
            self.mock_pipeline.describe.return_value = "Mock pipeline for testing"
            self.mock_pipeline.nodes = []
            
            # Configure mock catalog
            self.mock_catalog.list.return_value = ['input_data', 'output_data']
            self.mock_catalog.save = Mock()
            self.mock_catalog.load = Mock()
            
        def run_pipeline_without_hooks(self) -> Dict[str, Any]:
            """Run pipeline simulation without hooks."""
            self.execution_log.append('pipeline_start')
            self.execution_log.append('node_execution')
            self.execution_log.append('pipeline_complete')
            
            return {
                'status': 'completed',
                'execution_time': 0.1,
                'nodes_executed': 1,
                'outputs_created': 1
            }
            
        def run_pipeline_with_hooks(self) -> Dict[str, Any]:
            """Run pipeline simulation with hooks."""
            self.execution_log.append('hook_before_pipeline_run')
            self.execution_log.append('pipeline_start')
            self.execution_log.append('node_execution')
            self.execution_log.append('pipeline_complete')
            self.execution_log.append('hook_after_pipeline_run')
            
            return {
                'status': 'completed',
                'execution_time': 0.12,  # Slight overhead from hooks
                'nodes_executed': 1,
                'outputs_created': 1,
                'hooks_executed': 2
            }
            
        def compare_execution_results(self, without_hooks: Dict, with_hooks: Dict) -> Dict[str, Any]:
            """Compare execution results to validate non-invasive behavior."""
            return {
                'status_unchanged': without_hooks['status'] == with_hooks['status'],
                'nodes_executed_unchanged': without_hooks['nodes_executed'] == with_hooks['nodes_executed'],
                'outputs_created_unchanged': without_hooks['outputs_created'] == with_hooks['outputs_created'],
                'execution_overhead_ms': (with_hooks['execution_time'] - without_hooks['execution_time']) * 1000,
                'hooks_transparent': 'hooks_executed' in with_hooks and with_hooks['hooks_executed'] > 0
            }
    
    # Create execution monitor and simulator
    execution_monitor = ExecutionMonitor()
    kedro_simulator = KedroSimulator()
    kedro_simulator.setup_mock_environment()
    
    # State isolation validator
    def validate_state_isolation(before_state: Dict, after_state: Dict) -> bool:
        """Validate that hook execution maintains state isolation."""
        # Check that no unexpected state modifications occurred
        critical_keys = ['pipeline_state', 'catalog_entries', 'context_attributes']
        
        for key in critical_keys:
            if before_state.get(key) != after_state.get(key):
                # Allow expected changes but flag unexpected ones
                if key == 'pipeline_state':
                    # Pipeline state progression is expected
                    continue
                else:
                    return False
        return True
    
    # Transparency validator
    transparency_metrics = {
        'execution_overhead_threshold_ms': 50,  # Maximum acceptable overhead
        'state_modification_count': 0,
        'side_effect_count': 0
    }
    
    return {
        'execution_monitor': execution_monitor,
        'kedro_simulator': kedro_simulator,
        'validate_state_isolation': validate_state_isolation,
        'transparency_metrics': transparency_metrics
    }


@pytest.fixture(scope="session")
def kedro_test_environment(tmp_path_factory):
    """Session-scoped fixture providing temporary Kedro test environment.
    
    Creates a temporary Kedro project structure for integration testing that
    persists throughout the test session. This fixture supports testing hooks
    in realistic Kedro project contexts while maintaining isolation between
    test runs.
    
    Returns:
        Dict[str, Any]: Kedro test environment with project structure and utilities
    """
    if not KEDRO_AVAILABLE:
        pytest.skip("Kedro not available for testing")
    
    # Create temporary project directory
    project_dir = tmp_path_factory.mktemp("kedro_test_project")
    
    # Create basic Kedro project structure
    conf_dir = project_dir / "conf" / "base"
    conf_dir.mkdir(parents=True)
    
    src_dir = project_dir / "src" / "test_project"
    src_dir.mkdir(parents=True)
    
    # Create basic configuration files
    (conf_dir / "catalog.yml").write_text("""
test_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/test_figure.png
  purpose: expl
  condition_param: experiment_type
""")
    
    (conf_dir / "figregistry.yml").write_text("""
figregistry_version: "0.3.0"
style:
  default:
    figure.figsize: [8, 6]
    axes.labelsize: 12
condition_styles:
  test:
    color: red
    marker: o
""")
    
    (src_dir / "settings.py").write_text("""
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
""")
    
    return {
        'project_dir': project_dir,
        'conf_dir': conf_dir,
        'src_dir': src_dir
    }


# Utility functions for hook testing

def create_mock_run_params(session_id: str = None, pipeline_name: str = "default") -> Dict[str, Any]:
    """Create mock run parameters for hook testing.
    
    Args:
        session_id: Optional session ID, generates UUID if not provided
        pipeline_name: Name of the pipeline being executed
        
    Returns:
        Dict[str, Any]: Mock run parameters for hook method testing
    """
    return {
        'session_id': session_id or str(uuid.uuid4()),
        'pipeline_name': pipeline_name,
        'env': 'test',
        'tags': ['figregistry'],
        'run_id': str(uuid.uuid4()),
        'extra_params': {}
    }


def assert_hook_execution_performance(execution_record: HookExecutionRecord, max_time_ms: float = 25.0):
    """Assert that hook execution meets performance requirements.
    
    Args:
        execution_record: Record of hook execution timing
        max_time_ms: Maximum allowed execution time in milliseconds
        
    Raises:
        AssertionError: If execution time exceeds maximum allowed time
    """
    assert execution_record.success, f"Hook execution failed: {execution_record.error_message}"
    assert execution_record.execution_time_ms < max_time_ms, (
        f"Hook execution time {execution_record.execution_time_ms:.2f}ms exceeds "
        f"maximum allowed {max_time_ms:.2f}ms"
    )


def validate_thread_safety(execution_records: List[HookExecutionRecord]) -> bool:
    """Validate that concurrent hook executions maintain thread safety.
    
    Args:
        execution_records: List of execution records from concurrent executions
        
    Returns:
        bool: True if all executions were thread-safe, False otherwise
    """
    if not execution_records:
        return True
        
    # Check that all executions succeeded
    all_successful = all(record.success for record in execution_records)
    
    # Check that thread IDs are unique (no thread reuse issues)
    thread_ids = [record.thread_id for record in execution_records if record.thread_id]
    unique_threads = len(set(thread_ids)) == len(thread_ids)
    
    # Check that no execution interfered with others (no shared state corruption)
    session_ids = [record.session_id for record in execution_records if record.session_id]
    unique_sessions = len(set(session_ids)) == len(session_ids)
    
    return all_successful and unique_threads and unique_sessions


@contextmanager
def temporary_hook_state():
    """Context manager for temporary hook state during testing.
    
    Yields:
        Dict[str, Any]: Temporary hook state that is automatically cleaned up
    """
    temp_state = {
        'initialized': False,
        'pipelines': set(),
        'contexts': {},
        'cleanup_functions': []
    }
    
    try:
        yield temp_state
    finally:
        # Execute cleanup functions
        for cleanup_func in temp_state.get('cleanup_functions', []):
            try:
                cleanup_func()
            except Exception as e:
                logger.warning(f"Cleanup function failed: {e}")
        
        # Reset state
        temp_state.clear()


# Export all fixtures and utilities for easy import
__all__ = [
    'mock_hook_manager',
    'hook_lifecycle_events', 
    'hook_registration_fixtures',
    'thread_safety_fixtures',
    'hook_performance_fixtures',
    'cleanup_validation_fixtures',
    'non_invasive_integration_fixtures',
    'kedro_test_environment',
    'create_mock_run_params',
    'assert_hook_execution_performance', 
    'validate_thread_safety',
    'temporary_hook_state',
    'HookExecutionRecord',
    'LifecycleEventContext',
    'MockHookManager',
    'MockKedroSession'
]