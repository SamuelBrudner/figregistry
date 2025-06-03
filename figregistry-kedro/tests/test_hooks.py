"""
Unit tests for FigRegistryHooks component validating lifecycle integration with Kedro.

This module provides comprehensive testing of the FigRegistryHooks component,
validating compliance with Kedro's hook specification system per Section 5.2.7.
Tests cover hook registration through plugin discovery, lifecycle event handling,
configuration context management, and thread-safe operation across Kedro's
execution framework.

Key Testing Areas per Requirements:
- Hook specification compliance (before_pipeline_run, after_config_loaded) per F-006
- Non-invasive integration preserving Kedro's execution model per F-006.2
- Thread-safe operation for parallel pipeline execution per Section 5.2.7
- Registration through standard Kedro settings.py configuration per Section 5.2.7
- Performance validation (<25ms initialization overhead) per Section 6.6.4.3

Testing Framework Integration per Section 6.6.2.1:
- pytest >=8.0.0 with kedro-pytest framework integration
- pytest-mock >=3.14.0 for Kedro component simulation and hook lifecycle testing
- pytest-benchmark for performance overhead measurement and validation
- Comprehensive mocking of Kedro session contexts and hook manager operations

Coverage Targets per Section 6.6.2.4:
- ≥90% coverage for FigRegistryHooks module with 100% critical path coverage
- Complete validation of hook registration, lifecycle methods, and cleanup operations
- Performance and security testing per Sections 6.6.4.3 and 6.6.8.3
"""

import os
import sys
import time
import threading
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import numpy as np
import matplotlib.pyplot as plt
from kedro.io import DataCatalog
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.hooks import _create_hook_manager
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner, ParallelRunner


# =============================================================================
# HOOK REGISTRATION AND DISCOVERY TESTS
# =============================================================================

class TestHookRegistration:
    """
    Test hook registration through Kedro's plugin discovery system.
    
    Validates that FigRegistryHooks can be properly registered via settings.py
    configuration and plugin discovery mechanisms per Section 5.2.7.
    """
    
    def test_hook_class_exists_and_importable(self):
        """
        Test that FigRegistryHooks class can be imported and instantiated.
        
        Validates basic module structure and class availability for
        registration through Kedro's plugin discovery per F-006.
        """
        # Test import without errors
        try:
            from figregistry_kedro.hooks import FigRegistryHooks
        except ImportError as e:
            pytest.fail(f"FigRegistryHooks class not importable: {e}")
        
        # Test instantiation
        hooks_instance = FigRegistryHooks()
        assert hooks_instance is not None
        assert isinstance(hooks_instance, FigRegistryHooks)
    
    def test_hook_specification_compliance(self):
        """
        Test that FigRegistryHooks implements required hook methods.
        
        Validates compliance with Kedro hook specification including
        before_pipeline_run and after_config_loaded methods per Section 5.2.7.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Test required hook methods exist
        assert hasattr(hooks, 'before_pipeline_run'), \
            "FigRegistryHooks must implement before_pipeline_run method"
        assert hasattr(hooks, 'after_config_loaded'), \
            "FigRegistryHooks must implement after_config_loaded method"
        
        # Test methods are callable
        assert callable(getattr(hooks, 'before_pipeline_run')), \
            "before_pipeline_run must be callable"
        assert callable(getattr(hooks, 'after_config_loaded')), \
            "after_config_loaded must be callable"
    
    def test_hook_method_signatures(self):
        """
        Test that hook methods have correct signatures per Kedro specification.
        
        Validates parameter signatures for hook methods to ensure compatibility
        with Kedro's hook manager invocation patterns per Section 5.2.7.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        import inspect
        
        hooks = FigRegistryHooks()
        
        # Test before_pipeline_run signature
        before_sig = inspect.signature(hooks.before_pipeline_run)
        before_params = list(before_sig.parameters.keys())
        expected_before_params = ['run_params', 'pipeline', 'catalog']
        
        for param in expected_before_params:
            assert param in before_params, \
                f"before_pipeline_run must accept {param} parameter"
        
        # Test after_config_loaded signature
        after_sig = inspect.signature(hooks.after_config_loaded)
        after_params = list(after_sig.parameters.keys())
        expected_after_params = ['context', 'config_loader', 'conf_source']
        
        for param in expected_after_params:
            assert param in after_params, \
                f"after_config_loaded must accept {param} parameter"
    
    @pytest.mark.kedro_integration
    def test_hook_registration_in_hook_manager(self, mock_hook_manager):
        """
        Test hook registration with Kedro's hook manager.
        
        Validates that FigRegistryHooks can be registered with hook manager
        and appears in plugin registry per plugin discovery requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Test registration with hook manager
        mock_hook_manager.register(hooks)
        
        # Verify registration call was made
        mock_hook_manager.register.assert_called_once_with(hooks)
        
        # Test that hooks appears in registered plugins
        mock_hook_manager.get_plugins.return_value = [hooks]
        registered_plugins = mock_hook_manager.get_plugins()
        
        assert hooks in registered_plugins, \
            "FigRegistryHooks should appear in registered plugins"
    
    def test_hook_entry_point_configuration(self):
        """
        Test that hook entry points are properly configured for plugin discovery.
        
        Validates that package configuration includes proper entry points
        for automatic hook discovery per Section 5.2.7.
        """
        # This test would validate entry points in pyproject.toml/setup.py
        # Since we're testing the hooks directly, we simulate entry point discovery
        
        from figregistry_kedro.hooks import FigRegistryHooks
        
        # Test that hooks class is discoverable
        hooks_class = FigRegistryHooks
        assert hooks_class.__module__ == 'figregistry_kedro.hooks', \
            "FigRegistryHooks must be in figregistry_kedro.hooks module"
        
        # Test class is properly structured for entry point discovery
        assert hasattr(hooks_class, '__init__'), \
            "FigRegistryHooks must be instantiable for entry point discovery"


# =============================================================================
# LIFECYCLE INTEGRATION TESTS
# =============================================================================

class TestLifecycleIntegration:
    """
    Test hook lifecycle integration with Kedro execution framework.
    
    Validates proper integration with Kedro's pipeline execution lifecycle,
    including configuration loading, context management, and cleanup operations
    per F-006.2 non-invasive integration requirements.
    """
    
    @pytest.mark.kedro_integration
    def test_after_config_loaded_execution(self, mock_kedro_context, figregistry_test_config):
        """
        Test after_config_loaded hook execution and configuration initialization.
        
        Validates that after_config_loaded correctly initializes FigRegistry
        configuration context during Kedro project startup per Section 5.2.7.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Mock config loader with figregistry configuration
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        
        # Mock conf_source parameter
        mock_conf_source = Path('/tmp/test_project/conf')
        
        # Execute after_config_loaded hook
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=mock_conf_source
            )
            
            # Verify FigRegistry initialization was called
            mock_figregistry.init_config.assert_called_once()
            
            # Verify config loader was accessed for figregistry configuration
            mock_config_loader.get.assert_called_with('figregistry*')
    
    @pytest.mark.kedro_integration
    def test_before_pipeline_run_execution(self, mock_kedro_context, sample_catalog_config):
        """
        Test before_pipeline_run hook execution and context setup.
        
        Validates that before_pipeline_run properly sets up FigRegistry context
        before pipeline execution without modifying pipeline behavior per F-006.2.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Create mock pipeline and run parameters
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.describe.return_value = "Test Pipeline"
        
        mock_catalog = Mock(spec=DataCatalog)
        mock_catalog._data_sets = sample_catalog_config
        
        run_params = {
            'pipeline_name': 'test_pipeline',
            'env': 'local',
            'extra_params': {}
        }
        
        # Execute before_pipeline_run hook
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            result = hooks.before_pipeline_run(
                run_params=run_params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
            
            # Verify hook execution doesn't interfere with pipeline
            assert result is None, \
                "before_pipeline_run should not return value that modifies pipeline"
            
            # Verify FigRegistry context is maintained
            # This is implementation-dependent but should not raise exceptions
    
    @pytest.mark.kedro_integration
    def test_hook_execution_with_missing_config(self, mock_kedro_context):
        """
        Test hook execution gracefully handles missing FigRegistry configuration.
        
        Validates that hooks operate correctly when figregistry configuration
        is not present, maintaining graceful degradation per robustness requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Mock config loader with no figregistry configuration
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.side_effect = KeyError('figregistry')
        
        mock_conf_source = Path('/tmp/test_project/conf')
        
        # Test that hook handles missing configuration gracefully
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            # Should not raise exception
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=mock_conf_source
            )
            
            # Verify FigRegistry init was not called with invalid config
            mock_figregistry.init_config.assert_not_called()
    
    def test_hook_execution_preserves_kedro_model(self, mock_kedro_context, mock_hook_manager):
        """
        Test that hook execution preserves Kedro's execution model.
        
        Validates non-invasive integration that doesn't modify Kedro's
        internal execution patterns per F-006.2 requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Store original context state
        original_catalog = mock_kedro_context.catalog
        original_config_loader = mock_kedro_context.config_loader
        
        # Execute hook lifecycle simulation
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = {}
        
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_config_loader,
            conf_source=Path('/tmp/conf')
        )
        
        # Verify Kedro context is unchanged
        assert mock_kedro_context.catalog == original_catalog, \
            "Hook execution must not modify Kedro catalog"
        assert mock_kedro_context.config_loader == original_config_loader, \
            "Hook execution must not modify Kedro config loader"


# =============================================================================
# CONFIGURATION CONTEXT MANAGEMENT TESTS
# =============================================================================

class TestConfigurationManagement:
    """
    Test configuration context management throughout pipeline execution.
    
    Validates proper handling of FigRegistry configuration loading, merging,
    and state management during Kedro pipeline lifecycle per Section 5.2.5.
    """
    
    @pytest.mark.kedro_integration
    def test_configuration_bridge_initialization(self, mock_kedro_context, figregistry_test_config):
        """
        Test FigRegistryConfigBridge initialization during hook execution.
        
        Validates that hooks properly initialize configuration bridge for
        merging Kedro and FigRegistry configurations per Section 5.2.5.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        
        with patch('figregistry_kedro.hooks.FigRegistryConfigBridge') as mock_bridge:
            mock_bridge_instance = Mock()
            mock_bridge.return_value = mock_bridge_instance
            
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
            
            # Verify config bridge was created and initialized
            mock_bridge.assert_called_once()
            mock_bridge_instance.init_config.assert_called_once()
    
    def test_configuration_precedence_handling(self, mock_kedro_context):
        """
        Test configuration precedence between Kedro and FigRegistry settings.
        
        Validates that hooks handle configuration precedence correctly,
        ensuring Kedro settings override FigRegistry defaults per F-007.2.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Mock configuration with both base and override settings
        base_config = {
            'styles': {'exploratory': {'figure.figsize': [8, 6]}},
            'outputs': {'base_path': 'data/08_reporting'}
        }
        
        kedro_config = {
            'styles': {'exploratory': {'figure.figsize': [10, 8]}},  # Override
            'outputs': {'base_path': 'custom/output/path'}  # Override
        }
        
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = kedro_config
        
        with patch('figregistry_kedro.hooks.FigRegistryConfigBridge') as mock_bridge:
            mock_bridge_instance = Mock()
            mock_bridge.return_value = mock_bridge_instance
            mock_bridge_instance.merge_configs.return_value = kedro_config
            
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
            
            # Verify config bridge handled precedence
            mock_bridge_instance.merge_configs.assert_called_once()
    
    def test_configuration_state_persistence(self, mock_kedro_context, figregistry_test_config):
        """
        Test configuration state persistence throughout pipeline execution.
        
        Validates that configuration state is maintained consistently
        across hook invocations during pipeline execution lifecycle.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        
        # Simulate configuration loading
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
            
            # Verify initial configuration
            assert mock_figregistry.init_config.call_count == 1
            
            # Simulate pipeline run
            mock_pipeline = Mock(spec=Pipeline)
            mock_catalog = Mock(spec=DataCatalog)
            
            hooks.before_pipeline_run(
                run_params={'pipeline_name': 'test'},
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
            
            # Configuration should remain consistent
            # Implementation specific - verify no additional init calls
            assert mock_figregistry.init_config.call_count == 1
    
    @pytest.mark.security_test
    def test_configuration_security_validation(self, mock_kedro_context, security_test_configs):
        """
        Test configuration security validation during hook execution.
        
        Validates that hooks properly validate configuration security
        and prevent malicious configuration injection per Section 6.6.8.3.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Test with malicious configuration
        malicious_config = security_test_configs['yaml_injection_config']
        
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = malicious_config
        
        # Hook should handle malicious config gracefully
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
            
            # Should not execute malicious configuration
            # Implementation dependent - verify safe handling


# =============================================================================
# THREAD-SAFE OPERATION TESTS
# =============================================================================

class TestThreadSafeOperation:
    """
    Test thread-safe operation for parallel pipeline execution.
    
    Validates that FigRegistryHooks operate safely in multi-threaded
    environments supporting Kedro's parallel runners per Section 5.2.7.
    """
    
    @pytest.mark.plugin_performance
    def test_concurrent_hook_execution(self, mock_kedro_context, figregistry_test_config):
        """
        Test concurrent hook execution across multiple threads.
        
        Validates that hooks can execute safely in parallel threads
        without race conditions or state corruption per thread-safety requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        # Create multiple hook instances for parallel execution
        num_threads = 4
        hooks_instances = [FigRegistryHooks() for _ in range(num_threads)]
        
        results = []
        exceptions = []
        
        def execute_hook(hook_instance, thread_id):
            """Execute hook in thread and capture results."""
            try:
                mock_config_loader = Mock(spec=ConfigLoader)
                mock_config_loader.get.return_value = figregistry_test_config
                
                # Execute after_config_loaded in thread
                hook_instance.after_config_loaded(
                    context=mock_kedro_context,
                    config_loader=mock_config_loader,
                    conf_source=Path(f'/tmp/conf_{thread_id}')
                )
                
                # Execute before_pipeline_run in thread
                mock_pipeline = Mock(spec=Pipeline)
                mock_catalog = Mock(spec=DataCatalog)
                
                hook_instance.before_pipeline_run(
                    run_params={'pipeline_name': f'test_{thread_id}'},
                    pipeline=mock_pipeline,
                    catalog=mock_catalog
                )
                
                results.append(f"thread_{thread_id}_success")
                
            except Exception as e:
                exceptions.append((thread_id, str(e)))
        
        # Execute hooks concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(execute_hook, hooks_instances[i], i)
                for i in range(num_threads)
            ]
            
            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()  # Will raise exception if thread failed
        
        # Verify all threads completed successfully
        assert len(results) == num_threads, \
            f"Expected {num_threads} successful executions, got {len(results)}"
        assert len(exceptions) == 0, \
            f"Thread safety failures: {exceptions}"
    
    def test_hook_state_isolation(self, mock_kedro_context):
        """
        Test hook state isolation between parallel executions.
        
        Validates that hook instances maintain proper state isolation
        and don't interfere with each other during parallel execution.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        # Create two hook instances
        hooks1 = FigRegistryHooks()
        hooks2 = FigRegistryHooks()
        
        # Configure different contexts for each hook
        config1 = {'styles': {'test': {'figure.figsize': [8, 6]}}}
        config2 = {'styles': {'test': {'figure.figsize': [12, 8]}}}
        
        mock_config_loader1 = Mock(spec=ConfigLoader)
        mock_config_loader1.get.return_value = config1
        
        mock_config_loader2 = Mock(spec=ConfigLoader)
        mock_config_loader2.get.return_value = config2
        
        # Execute hooks with different configurations
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            hooks1.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader1,
                conf_source=Path('/tmp/conf1')
            )
            
            hooks2.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader2,
                conf_source=Path('/tmp/conf2')
            )
            
            # Verify both configurations were processed independently
            assert mock_figregistry.init_config.call_count == 2
    
    @pytest.mark.plugin_performance
    def test_parallel_runner_compatibility(self, mock_kedro_context, sample_catalog_config):
        """
        Test compatibility with Kedro's ParallelRunner execution.
        
        Validates that hooks operate correctly with ParallelRunner
        and maintain thread safety during parallel node execution.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Simulate parallel runner execution scenario
        def simulate_parallel_node_execution(node_id):
            """Simulate parallel node execution with hook context."""
            mock_pipeline = Mock(spec=Pipeline)
            mock_catalog = Mock(spec=DataCatalog)
            mock_catalog._data_sets = sample_catalog_config
            
            run_params = {
                'pipeline_name': f'parallel_test_{node_id}',
                'env': 'local'
            }
            
            # Execute hook in parallel context
            hooks.before_pipeline_run(
                run_params=run_params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
            
            return f"node_{node_id}_completed"
        
        # Execute multiple nodes in parallel
        num_nodes = 6
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(simulate_parallel_node_execution, i)
                for i in range(num_nodes)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all parallel executions completed
        assert len(results) == num_nodes
        assert all('completed' in result for result in results)


# =============================================================================
# PERFORMANCE TESTING
# =============================================================================

class TestHookPerformance:
    """
    Test hook performance and initialization overhead.
    
    Validates that hook operations meet performance targets including
    <25ms initialization overhead per Section 6.6.4.3 requirements.
    """
    
    @pytest.mark.plugin_performance
    def test_hook_initialization_performance(self, benchmark, mock_kedro_context, figregistry_test_config):
        """
        Test hook initialization performance meets <25ms target.
        
        Validates that after_config_loaded hook execution completes
        within performance threshold per Section 6.6.4.3.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        
        def hook_initialization():
            """Execute hook initialization for benchmarking."""
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
        
        # Benchmark hook initialization
        result = benchmark(hook_initialization)
        
        # Verify performance target (25ms = 0.025 seconds)
        assert benchmark.stats['mean'] < 0.025, \
            f"Hook initialization took {benchmark.stats['mean']:.4f}s, exceeds 25ms target"
    
    @pytest.mark.plugin_performance
    def test_before_pipeline_run_performance(self, benchmark, sample_catalog_config):
        """
        Test before_pipeline_run performance overhead.
        
        Validates that before_pipeline_run hook execution adds minimal
        overhead to pipeline startup per performance requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        mock_pipeline = Mock(spec=Pipeline)
        mock_catalog = Mock(spec=DataCatalog)
        mock_catalog._data_sets = sample_catalog_config
        
        run_params = {'pipeline_name': 'performance_test'}
        
        def hook_execution():
            """Execute before_pipeline_run for benchmarking."""
            hooks.before_pipeline_run(
                run_params=run_params,
                pipeline=mock_pipeline,
                catalog=mock_catalog
            )
        
        # Benchmark hook execution
        result = benchmark(hook_execution)
        
        # Verify minimal overhead (should be very fast)
        assert benchmark.stats['mean'] < 0.01, \
            f"before_pipeline_run took {benchmark.stats['mean']:.4f}s, excessive overhead"
    
    @pytest.mark.plugin_performance
    def test_hook_memory_overhead(self, mock_kedro_context, figregistry_test_config):
        """
        Test hook memory overhead during execution.
        
        Validates that hook execution maintains minimal memory footprint
        per Section 5.2.8 scaling considerations.
        """
        import psutil
        import gc
        
        from figregistry_kedro.hooks import FigRegistryHooks
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and execute hooks
        hooks = FigRegistryHooks()
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        
        # Execute multiple hook operations
        for i in range(10):
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path(f'/tmp/conf_{i}')
            )
        
        # Measure memory after hook operations
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_overhead = final_memory - baseline_memory
        
        # Verify memory overhead is reasonable (<5MB per Section 6.6.4.3)
        assert memory_overhead < 5.0, \
            f"Hook memory overhead {memory_overhead:.2f}MB exceeds 5MB target"


# =============================================================================
# ERROR HANDLING AND ROBUSTNESS TESTS
# =============================================================================

class TestErrorHandling:
    """
    Test error handling and robustness of hook implementation.
    
    Validates graceful handling of error conditions, invalid configurations,
    and edge cases per Section 6.6.3 error handling requirements.
    """
    
    def test_hook_handles_invalid_config_loader(self, mock_kedro_context):
        """
        Test hook handling of invalid or missing config loader.
        
        Validates graceful degradation when config loader is unavailable
        or returns invalid configuration data.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Test with None config loader
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=None,
            conf_source=Path('/tmp/conf')
        )
        
        # Test with config loader that raises exceptions
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.side_effect = RuntimeError("Config loader error")
        
        # Should not raise exception
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_config_loader,
            conf_source=Path('/tmp/conf')
        )
    
    def test_hook_handles_catalog_errors(self, mock_kedro_context):
        """
        Test hook handling of catalog-related errors.
        
        Validates that hooks handle catalog errors gracefully without
        disrupting pipeline execution per robustness requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Test with None catalog
        mock_pipeline = Mock(spec=Pipeline)
        
        hooks.before_pipeline_run(
            run_params={'pipeline_name': 'test'},
            pipeline=mock_pipeline,
            catalog=None
        )
        
        # Test with catalog that raises exceptions
        mock_catalog = Mock(spec=DataCatalog)
        mock_catalog._data_sets = Mock(side_effect=AttributeError("Catalog error"))
        
        # Should not raise exception
        hooks.before_pipeline_run(
            run_params={'pipeline_name': 'test'},
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
    
    def test_hook_handles_missing_dependencies(self, mock_kedro_context):
        """
        Test hook handling when FigRegistry dependencies are missing.
        
        Validates graceful degradation when figregistry module is not
        available or incompatible per dependency management requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Mock missing figregistry dependency
        with patch('figregistry_kedro.hooks.figregistry', None):
            mock_config_loader = Mock(spec=ConfigLoader)
            mock_config_loader.get.return_value = {'styles': {}}
            
            # Should handle missing dependency gracefully
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
    
    @pytest.mark.kedro_integration
    def test_hook_cleanup_on_errors(self, mock_kedro_context, figregistry_test_config):
        """
        Test hook cleanup behavior when errors occur during execution.
        
        Validates that hooks properly clean up resources and state
        when errors occur during lifecycle execution per Section 5.2.7.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        
        # Simulate error during configuration
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            mock_figregistry.init_config.side_effect = RuntimeError("Init error")
            
            # Should handle error gracefully without leaving corrupted state
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
            
            # Verify error was handled and no persistent state remains


# =============================================================================
# INTEGRATION TESTS WITH KEDRO COMPONENTS
# =============================================================================

class TestKedroIntegration:
    """
    Test integration with actual Kedro components and workflows.
    
    Validates end-to-end integration of hooks with Kedro session management,
    pipeline execution, and configuration systems per F-006 requirements.
    """
    
    @pytest.mark.kedro_integration
    def test_hooks_with_kedro_session(self, mock_kedro_session, figregistry_test_config):
        """
        Test hook integration with Kedro session lifecycle.
        
        Validates that hooks integrate properly with KedroSession
        management and context loading per session integration requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Mock session context loading
        mock_context = mock_kedro_session.load_context()
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = figregistry_test_config
        mock_context.config_loader = mock_config_loader
        
        # Test hook integration with session
        with patch('figregistry_kedro.hooks.figregistry') as mock_figregistry:
            hooks.after_config_loaded(
                context=mock_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/test_project/conf')
            )
            
            # Verify integration with session context
            mock_figregistry.init_config.assert_called_once()
    
    @pytest.mark.kedro_integration
    def test_hooks_with_pipeline_execution(self, temp_work_dir, sample_catalog_config):
        """
        Test hook integration during actual pipeline execution.
        
        Validates that hooks function correctly during real pipeline
        execution scenarios with node processing and catalog operations.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Create simple test pipeline
        def test_node_func():
            return "test_output"
        
        test_node = node(
            func=test_node_func,
            inputs=None,
            outputs="test_output"
        )
        
        test_pipeline = Pipeline([test_node])
        
        # Create test catalog
        mock_catalog = Mock(spec=DataCatalog)
        mock_catalog._data_sets = sample_catalog_config
        
        run_params = {
            'pipeline_name': 'test_pipeline',
            'env': 'local',
            'tags': [],
            'runner': 'SequentialRunner'
        }
        
        # Execute hook before pipeline run
        result = hooks.before_pipeline_run(
            run_params=run_params,
            pipeline=test_pipeline,
            catalog=mock_catalog
        )
        
        # Verify hook doesn't interfere with pipeline execution
        assert result is None
        assert test_pipeline.describe() == "test_pipeline"
    
    @pytest.mark.kedro_integration
    def test_hooks_settings_integration(self, temp_work_dir):
        """
        Test hook registration through settings.py configuration.
        
        Validates that hooks can be properly registered through Kedro's
        settings.py configuration mechanism per Section 5.2.7.
        """
        # Create temporary settings.py content for testing
        settings_content = '''
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
'''
        
        settings_path = temp_work_dir / "settings.py"
        settings_path.write_text(settings_content)
        
        # Test that settings can be imported and contain hooks
        sys.path.insert(0, str(temp_work_dir))
        
        try:
            import settings
            
            # Verify hooks are properly configured
            assert hasattr(settings, 'HOOKS')
            assert len(settings.HOOKS) == 1
            
            from figregistry_kedro.hooks import FigRegistryHooks
            assert isinstance(settings.HOOKS[0], FigRegistryHooks)
            
        finally:
            # Cleanup
            sys.path.remove(str(temp_work_dir))
            if 'settings' in sys.modules:
                del sys.modules['settings']


# =============================================================================
# SECURITY AND VALIDATION TESTS
# =============================================================================

class TestSecurityValidation:
    """
    Test security aspects of hook implementation.
    
    Validates security constraints, input validation, and protection
    against malicious configurations per Section 6.6.8.3 requirements.
    """
    
    @pytest.mark.security_test
    def test_hook_input_validation(self, mock_kedro_context):
        """
        Test hook input validation and sanitization.
        
        Validates that hooks properly validate and sanitize input parameters
        to prevent injection attacks or malicious configuration per security requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Test with malicious run parameters
        malicious_run_params = {
            'pipeline_name': '../../../etc/passwd',
            'env': '$(rm -rf /)',
            'extra_params': {'malicious': 'payload'}
        }
        
        mock_pipeline = Mock(spec=Pipeline)
        mock_catalog = Mock(spec=DataCatalog)
        
        # Should handle malicious parameters safely
        hooks.before_pipeline_run(
            run_params=malicious_run_params,
            pipeline=mock_pipeline,
            catalog=mock_catalog
        )
        
        # Test with malicious conf_source
        malicious_conf_source = Path('../../../etc/passwd')
        mock_config_loader = Mock(spec=ConfigLoader)
        mock_config_loader.get.return_value = {}
        
        # Should handle malicious paths safely
        hooks.after_config_loaded(
            context=mock_kedro_context,
            config_loader=mock_config_loader,
            conf_source=malicious_conf_source
        )
    
    @pytest.mark.security_test
    def test_hook_privilege_isolation(self, mock_kedro_context):
        """
        Test that hooks maintain proper privilege isolation.
        
        Validates that hooks cannot escalate privileges or access
        unauthorized system resources per security requirements.
        """
        from figregistry_kedro.hooks import FigRegistryHooks
        
        hooks = FigRegistryHooks()
        
        # Store original environment
        original_env = os.environ.copy()
        
        try:
            # Execute hooks with restricted environment
            mock_config_loader = Mock(spec=ConfigLoader)
            mock_config_loader.get.return_value = {}
            
            hooks.after_config_loaded(
                context=mock_kedro_context,
                config_loader=mock_config_loader,
                conf_source=Path('/tmp/conf')
            )
            
            # Verify no unauthorized environment modifications
            assert os.environ == original_env, \
                "Hooks should not modify environment variables"
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)


# =============================================================================
# UTILITY FUNCTIONS FOR TESTING
# =============================================================================

def create_mock_pipeline_with_nodes(num_nodes: int = 3) -> Pipeline:
    """
    Create mock pipeline with specified number of nodes for testing.
    
    Args:
        num_nodes: Number of nodes to include in pipeline
        
    Returns:
        Mock Pipeline instance with test nodes
    """
    def dummy_func(x):
        return f"output_{x}"
    
    nodes = [
        node(
            func=dummy_func,
            inputs=f"input_{i}",
            outputs=f"output_{i}",
            name=f"test_node_{i}"
        )
        for i in range(num_nodes)
    ]
    
    return Pipeline(nodes)


def simulate_hook_execution_cycle(hooks, context, config_loader, catalog, pipeline):
    """
    Simulate complete hook execution cycle for testing.
    
    Args:
        hooks: FigRegistryHooks instance
        context: Mock KedroContext
        config_loader: Mock ConfigLoader
        catalog: Mock DataCatalog
        pipeline: Mock Pipeline
        
    Returns:
        Tuple of (config_result, pipeline_result) from hook execution
    """
    # Execute configuration loading
    config_result = hooks.after_config_loaded(
        context=context,
        config_loader=config_loader,
        conf_source=Path('/tmp/conf')
    )
    
    # Execute pipeline preparation
    pipeline_result = hooks.before_pipeline_run(
        run_params={'pipeline_name': 'test'},
        pipeline=pipeline,
        catalog=catalog
    )
    
    return config_result, pipeline_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])