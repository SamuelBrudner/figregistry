"""Core Kedro testing fixtures for FigRegistry-Kedro plugin validation.

This module provides comprehensive Kedro component mocking for plugin testing per 
Section 6.6.2.3, enabling isolated testing of figregistry-kedro components without 
full Kedro pipeline overhead. Supports kedro-pytest framework integration with 
TestKedro fixtures per Section 6.6.2.1.

The fixtures provide:
- Mock ProjectContext, DataCatalog, ConfigLoader, and session simulation
- Comprehensive plugin testing without full Kedro overhead per testing strategy
- Support for temporary project creation for realistic plugin testing scenarios
- Hook manager simulation for lifecycle testing per Section 6.6.3.8
- TestKedro integration for kedro-pytest framework compatibility

Key testing capabilities:
- minimal_kedro_context: ProjectContext simulation with ConfigLoader and DataCatalog mocks
- test_catalog_with_figregistry: DataCatalog with FigureDataSet entries for integration testing  
- mock_kedro_session: Complete Kedro session simulation with temporary project setup
- figregistry_config_bridge: Pre-configured FigRegistryConfigBridge for testing
- mock_hook_manager: Hook registration and lifecycle testing per Section 6.6.3.8
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
import pytest
import yaml

# Kedro imports with graceful fallback for environments without Kedro
try:
    from kedro.framework.context import KedroContext
    from kedro.framework.project import configure_project
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    from kedro.config import ConfigLoader, OmegaConfigLoader
    from kedro.io import DataCatalog, AbstractDataSet
    from kedro.pipeline import Pipeline
    from kedro.runner import AbstractRunner
    from kedro.framework.hooks import PluginManager
    KEDRO_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments without Kedro
    KedroContext = None
    KedroSession = None
    ConfigLoader = None
    OmegaConfigLoader = None
    DataCatalog = None
    AbstractDataSet = None
    Pipeline = None
    AbstractRunner = None
    PluginManager = None
    KEDRO_AVAILABLE = False

# kedro-pytest integration with fallback
try:
    from kedro_pytest import TestKedro
    KEDRO_PYTEST_AVAILABLE = True
except ImportError:
    TestKedro = None
    KEDRO_PYTEST_AVAILABLE = False

# Local test fixture dependencies
from figregistry_kedro.tests.fixtures.config_fixtures import (
    base_figregistry_config,
    local_override_config,
    minimal_figregistry_config
)


# =============================================================================
# Core Kedro Component Mock Fixtures
# =============================================================================

@pytest.fixture
def minimal_kedro_context(mocker) -> Mock:
    """
    Provides minimal ProjectContext for plugin testing per Section 6.6.2.6.
    
    Creates a lightweight mock of Kedro's ProjectContext with ConfigLoader and 
    DataCatalog mocks for isolated component testing without full project overhead.
    Essential for validating FigRegistryHooks and FigRegistryConfigBridge behavior
    in controlled testing environments.
    
    Args:
        mocker: pytest-mock fixture for creating mocks
        
    Returns:
        Mock ProjectContext with essential components configured
    """
    if not KEDRO_AVAILABLE:
        pytest.skip("Kedro not available for context mocking")
    
    # Create mock ProjectContext with proper specification
    mock_context = mocker.Mock(spec=KedroContext)
    
    # Mock ConfigLoader with basic figregistry configuration loading
    mock_config_loader = mocker.Mock(spec=ConfigLoader)
    mock_config_loader.get.return_value = {
        'styles': {
            'default': {'color': '#1f77b4', 'marker': 'o'},
            'test_condition': {'color': '#ff7f0e', 'marker': 's'}
        },
        'outputs': {
            'base_path': 'figures',
            'dpi': 300
        }
    }
    mock_context.config_loader = mock_config_loader
    
    # Mock DataCatalog with basic dataset support
    mock_catalog = mocker.Mock(spec=DataCatalog)
    mock_catalog.list.return_value = ['test_dataset', 'figregistry_plot']
    mock_catalog.exists.return_value = True
    mock_catalog.save = mocker.Mock()
    mock_catalog.load = mocker.Mock()
    mock_context.catalog = mock_catalog
    
    # Mock essential context properties
    mock_context.project_name = "test_figregistry_project"
    mock_context.project_version = "0.1.0"
    mock_context.project_path = Path("/tmp/test_project")
    mock_context.env = "test"
    
    # Mock pipeline and runner for complete context simulation
    mock_pipeline = mocker.Mock(spec=Pipeline)
    mock_pipeline.nodes = []
    mock_context.pipelines = {'__default__': mock_pipeline}
    
    return mock_context


@pytest.fixture
def mock_config_loader(mocker, base_figregistry_config) -> Mock:
    """
    Mock Kedro ConfigLoader for configuration bridge testing.
    
    Provides a mock ConfigLoader that simulates Kedro's environment-specific
    configuration loading for testing FigRegistryConfigBridge merging behavior.
    
    Args:
        mocker: pytest-mock fixture
        base_figregistry_config: Base FigRegistry configuration fixture
        
    Returns:
        Mock ConfigLoader with pre-configured responses
    """
    if not KEDRO_AVAILABLE:
        pytest.skip("Kedro not available for ConfigLoader mocking")
    
    mock_loader = mocker.Mock(spec=ConfigLoader)
    
    def mock_get(pattern: str, *args, **kwargs):
        """Mock get method that returns configuration based on pattern."""
        if "figregistry" in pattern.lower():
            return base_figregistry_config
        elif "catalog" in pattern.lower():
            return {
                'test_figure': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': 'data/08_reporting/test_figure.png',
                    'purpose': 'exploratory'
                }
            }
        elif "parameters" in pattern.lower():
            return {
                'experiment_type': 'test_experiment',
                'environment': 'test'
            }
        return {}
    
    mock_loader.get = Mock(side_effect=mock_get)
    mock_loader.config_patterns = {
        'figregistry': ['figregistry*', 'figregistry/**'],
        'catalog': ['catalog*'],
        'parameters': ['parameters*']
    }
    
    return mock_loader


@pytest.fixture 
def test_catalog_with_figregistry(mocker) -> Mock:
    """
    DataCatalog with FigureDataSet entries for integration testing per Section 6.6.3.7.
    
    Creates a mock DataCatalog pre-configured with FigureDataSet entries to validate
    catalog integration, parameter extraction, and save operation workflows without
    requiring full Kedro project setup.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock DataCatalog with FigureDataSet entries configured
    """
    if not KEDRO_AVAILABLE:
        pytest.skip("Kedro not available for DataCatalog mocking")
    
    mock_catalog = mocker.Mock(spec=DataCatalog)
    
    # Pre-configured FigureDataSet entries for testing
    figregistry_datasets = {
        'exploratory_plot': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/figures/exploratory/plot.png',
            'purpose': 'exploratory',
            'condition_param': 'experiment_type',
            'style_params': {'condition': 'exploratory'}
        },
        'presentation_chart': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/figures/presentation/chart.png',
            'purpose': 'presentation',
            'condition_param': 'experiment_type',
            'style_params': {'condition': 'presentation'},
            'versioned': True
        },
        'publication_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/figures/publication/figure.pdf',
            'purpose': 'publication',
            'condition_param': 'experiment_type',
            'style_params': {'condition': 'publication', 'format': 'pdf'},
            'save_args': {'dpi': 300, 'bbox_inches': 'tight'}
        }
    }
    
    # Mock catalog behavior for FigureDataSet operations
    mock_catalog.list.return_value = list(figregistry_datasets.keys())
    mock_catalog.exists = Mock(return_value=True)
    
    def mock_load(dataset_name: str):
        """Mock load operation - not typically used for figures."""
        if dataset_name in figregistry_datasets:
            return None  # Figures typically only support save operations
        raise KeyError(f"Dataset {dataset_name} not found")
    
    def mock_save(dataset_name: str, data):
        """Mock save operation for figure datasets."""
        if dataset_name in figregistry_datasets:
            # Simulate successful save operation
            return True
        raise KeyError(f"Dataset {dataset_name} not found")
    
    mock_catalog.load = Mock(side_effect=mock_load)
    mock_catalog.save = Mock(side_effect=mock_save)
    
    # Provide access to dataset configurations for testing
    mock_catalog._datasets = figregistry_datasets
    mock_catalog.get_dataset_config = Mock(
        side_effect=lambda name: figregistry_datasets.get(name, {})
    )
    
    return mock_catalog


@pytest.fixture
def mock_kedro_session(mocker, tmp_path) -> Mock:
    """
    Complete Kedro session simulation with temporary project setup per Section 6.6.2.6.
    
    Provides comprehensive KedroSession simulation including ProjectContext,
    ConfigLoader, and DataCatalog components for end-to-end plugin testing
    without the overhead of full project creation.
    
    Args:
        mocker: pytest-mock fixture
        tmp_path: pytest temporary path fixture
        
    Returns:
        Mock KedroSession with complete component stack
    """
    if not KEDRO_AVAILABLE:
        pytest.skip("Kedro not available for session mocking")
    
    mock_session = mocker.Mock(spec=KedroSession)
    
    # Create temporary project structure for realistic session simulation
    project_path = tmp_path / "mock_kedro_project"
    project_path.mkdir()
    
    # Create minimal conf structure
    conf_path = project_path / "conf"
    conf_path.mkdir()
    (conf_path / "base").mkdir()
    (conf_path / "local").mkdir()
    
    # Create basic figregistry configuration
    figregistry_config = {
        'styles': {
            'test_condition': {'color': '#1f77b4', 'marker': 'o'},
            'mock_condition': {'color': '#ff7f0e', 'marker': 's'}
        },
        'outputs': {'base_path': 'mock_figures'}
    }
    
    with open(conf_path / "base" / "figregistry.yml", 'w') as f:
        yaml.dump(figregistry_config, f)
    
    # Mock session properties
    mock_session.store = mocker.Mock()
    mock_session._project_path = project_path
    
    # Create integrated context with all components
    mock_context = mocker.Mock(spec=KedroContext)
    mock_context.project_path = project_path
    mock_context.project_name = "mock_kedro_project"
    mock_context.env = "test"
    
    # Configure ConfigLoader for session
    mock_config_loader = mocker.Mock(spec=ConfigLoader)
    mock_config_loader.get.return_value = figregistry_config
    mock_context.config_loader = mock_config_loader
    
    # Configure DataCatalog for session
    mock_catalog = mocker.Mock(spec=DataCatalog)
    mock_catalog.list.return_value = ['mock_figure_dataset']
    mock_context.catalog = mock_catalog
    
    # Session creation methods
    mock_session.load_context = Mock(return_value=mock_context)
    
    def mock_run(pipeline_name=None, **kwargs):
        """Mock pipeline run that simulates successful execution."""
        return {'status': 'success', 'pipeline': pipeline_name}
    
    mock_session.run = Mock(side_effect=mock_run)
    
    # Session lifecycle management
    mock_session.close = Mock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    
    return mock_session


@pytest.fixture
def mock_hook_manager(mocker) -> Mock:
    """
    Mock hook manager for hook registration and lifecycle testing per Section 6.6.3.8.
    
    Provides PluginManager simulation for testing FigRegistryHooks registration,
    lifecycle execution, and integration with Kedro's hook system without requiring
    full Kedro project initialization.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock PluginManager for hook testing
    """
    if not KEDRO_AVAILABLE:
        pytest.skip("Kedro not available for hook manager mocking")
    
    mock_manager = mocker.Mock(spec=PluginManager)
    
    # Track registered hooks for testing
    registered_hooks = []
    
    def mock_register(hook_instance):
        """Mock hook registration tracking."""
        registered_hooks.append(hook_instance)
        return True
    
    def mock_call_hook(hook_name, **kwargs):
        """Mock hook invocation with parameter tracking."""
        # Simulate hook execution by calling registered hooks
        results = []
        for hook in registered_hooks:
            if hasattr(hook, hook_name):
                method = getattr(hook, hook_name)
                try:
                    result = method(**kwargs)
                    results.append(result)
                except Exception as e:
                    # Track hook execution errors
                    results.append({'error': str(e)})
        return results
    
    # Hook lifecycle methods
    mock_manager.register = Mock(side_effect=mock_register)
    mock_manager.unregister = Mock()
    mock_manager.call_hook = Mock(side_effect=mock_call_hook)
    mock_manager.is_registered = Mock(return_value=True)
    
    # Hook discovery and management
    mock_manager.list_hooks = Mock(return_value=registered_hooks)
    mock_manager.get_hook_callers = Mock(return_value={})
    
    # Provide access to registered hooks for testing
    mock_manager._registered_hooks = registered_hooks
    
    return mock_manager


# =============================================================================
# FigRegistry Integration Fixtures
# =============================================================================

@pytest.fixture
def figregistry_config_bridge(mocker, base_figregistry_config, local_override_config):
    """
    Pre-configured FigRegistryConfigBridge for testing per configuration bridge requirements.
    
    Provides a FigRegistryConfigBridge instance with pre-loaded configurations
    for testing configuration merging, validation, and bridge functionality
    without requiring full Kedro project setup.
    
    Args:
        mocker: pytest-mock fixture
        base_figregistry_config: Base FigRegistry configuration
        local_override_config: Local environment overrides
        
    Returns:
        Configured FigRegistryConfigBridge instance for testing
    """
    try:
        from figregistry_kedro.config import FigRegistryConfigBridge
    except ImportError:
        pytest.skip("FigRegistryConfigBridge not available")
    
    # Create mock Kedro ConfigLoader
    mock_config_loader = mocker.Mock()
    mock_config_loader.get.return_value = local_override_config
    
    # Initialize bridge with test configurations
    bridge = FigRegistryConfigBridge(
        config_loader=mock_config_loader,
        base_config=base_figregistry_config
    )
    
    # Pre-configure merged configuration for testing
    bridge._merged_config = {
        **base_figregistry_config,
        **local_override_config
    }
    
    return bridge


@pytest.fixture
def mock_figregistry_hooks(mocker):
    """
    Mock FigRegistryHooks instance for lifecycle testing.
    
    Provides a mock of FigRegistryHooks with tracked method calls for testing
    hook registration, initialization sequences, and lifecycle integration.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock FigRegistryHooks instance
    """
    try:
        from figregistry_kedro.hooks import FigRegistryHooks
        mock_hooks = mocker.Mock(spec=FigRegistryHooks)
    except ImportError:
        # Fallback to generic mock if hooks not available
        mock_hooks = mocker.Mock()
    
    # Track hook lifecycle calls
    call_history = []
    
    def track_calls(method_name):
        def wrapper(*args, **kwargs):
            call_history.append({
                'method': method_name,
                'args': args,
                'kwargs': kwargs,
                'timestamp': mocker.Mock()
            })
            return mocker.Mock()
        return wrapper
    
    # Mock hook lifecycle methods with call tracking
    mock_hooks.before_pipeline_run = Mock(side_effect=track_calls('before_pipeline_run'))
    mock_hooks.after_pipeline_run = Mock(side_effect=track_calls('after_pipeline_run'))
    mock_hooks.before_node_run = Mock(side_effect=track_calls('before_node_run'))
    mock_hooks.after_node_run = Mock(side_effect=track_calls('after_node_run'))
    mock_hooks.on_pipeline_error = Mock(side_effect=track_calls('on_pipeline_error'))
    
    # Provide access to call history for testing
    mock_hooks._call_history = call_history
    
    return mock_hooks


@pytest.fixture
def mock_figure_dataset(mocker):
    """
    Mock FigureDataSet for dataset testing without file operations.
    
    Provides a mock implementation of FigureDataSet for testing dataset
    behavior, parameter handling, and catalog integration without actual
    file system operations.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock FigureDataSet instance
    """
    try:
        from figregistry_kedro.datasets import FigureDataSet
        mock_dataset = mocker.Mock(spec=FigureDataSet)
    except ImportError:
        # Fallback to AbstractDataSet spec if FigureDataSet not available
        if KEDRO_AVAILABLE:
            mock_dataset = mocker.Mock(spec=AbstractDataSet)
        else:
            mock_dataset = mocker.Mock()
    
    # Mock dataset configuration
    mock_dataset._filepath = Path("test_figure.png")
    mock_dataset._purpose = "test"
    mock_dataset._condition_param = "experiment_type"
    mock_dataset._style_params = {"condition": "test"}
    
    # Mock dataset operations
    def mock_save(data):
        """Mock save operation tracking."""
        mock_dataset._last_saved_data = data
        return True
    
    def mock_load():
        """Mock load operation - typically not used for figures."""
        return None
    
    def mock_describe():
        """Mock dataset description."""
        return {
            'filepath': str(mock_dataset._filepath),
            'purpose': mock_dataset._purpose,
            'condition_param': mock_dataset._condition_param
        }
    
    mock_dataset._save = Mock(side_effect=mock_save)
    mock_dataset._load = Mock(side_effect=mock_load)
    mock_dataset._describe = Mock(side_effect=mock_describe)
    mock_dataset.save = Mock(side_effect=mock_save)
    mock_dataset.load = Mock(side_effect=mock_load)
    mock_dataset.describe = Mock(side_effect=mock_describe)
    
    return mock_dataset


# =============================================================================
# TestKedro Integration Fixtures
# =============================================================================

@pytest.fixture
def test_kedro_instance(tmp_path):
    """
    TestKedro integration for kedro-pytest framework compatibility per Section 6.6.2.1.
    
    Provides TestKedro fixture integration enabling kedro-pytest framework
    compatibility for comprehensive plugin testing with in-process pipeline
    context simulation.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        TestKedro instance for kedro-pytest integration
    """
    if not KEDRO_PYTEST_AVAILABLE:
        pytest.skip("kedro-pytest not available for TestKedro integration")
    
    # Create minimal project structure for TestKedro
    project_path = tmp_path / "test_kedro_project"
    project_path.mkdir()
    
    # Create basic configuration structure
    conf_path = project_path / "conf"
    conf_path.mkdir()
    (conf_path / "base").mkdir()
    
    # Basic catalog configuration with FigureDataSet
    catalog_config = {
        'test_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/test_figure.png',
            'purpose': 'test'
        }
    }
    
    with open(conf_path / "base" / "catalog.yml", 'w') as f:
        yaml.dump(catalog_config, f)
    
    # Initialize TestKedro instance
    test_kedro = TestKedro(
        project_path=project_path,
        package_name="test_kedro_project"
    )
    
    return test_kedro


@pytest.fixture
def kedro_pytest_session(test_kedro_instance):
    """
    TestKedro session with FigRegistry plugin configuration.
    
    Extends TestKedro with FigRegistry-specific configuration for testing
    complete plugin integration within kedro-pytest framework.
    
    Args:
        test_kedro_instance: TestKedro instance fixture
        
    Returns:
        Configured TestKedro session with FigRegistry integration
    """
    if not KEDRO_PYTEST_AVAILABLE:
        pytest.skip("kedro-pytest not available for session testing")
    
    # Configure FigRegistry settings for TestKedro session
    figregistry_config = {
        'styles': {
            'test_session': {
                'figure.figsize': [8, 6],
                'axes.labelsize': 10
            }
        },
        'outputs': {
            'base_path': 'test_outputs'
        }
    }
    
    # Add FigRegistry configuration to TestKedro
    conf_path = test_kedro_instance.project_path / "conf" / "base"
    with open(conf_path / "figregistry.yml", 'w') as f:
        yaml.dump(figregistry_config, f)
    
    # Configure hooks in TestKedro settings
    settings_content = """
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
"""
    
    src_path = test_kedro_instance.project_path / "src" / "test_kedro_project"
    src_path.mkdir(parents=True, exist_ok=True)
    
    with open(src_path / "settings.py", 'w') as f:
        f.write(settings_content)
    
    return test_kedro_instance


# =============================================================================
# Project Scaffolding Fixtures
# =============================================================================

@pytest.fixture
def minimal_project_scaffolding(tmp_path):
    """
    Minimal Kedro project scaffolding for temporary project creation per Section 6.6.7.2.
    
    Creates the minimal directory structure and configuration files required
    for a functional Kedro project with FigRegistry plugin integration,
    suitable for lightweight testing scenarios.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Dictionary with project paths and configuration for testing
    """
    project_path = tmp_path / "minimal_scaffolded_project"
    project_path.mkdir()
    
    # Create essential directory structure
    (project_path / "conf" / "base").mkdir(parents=True)
    (project_path / "conf" / "local").mkdir(parents=True)
    (project_path / "data" / "01_raw").mkdir(parents=True)
    (project_path / "data" / "08_reporting").mkdir(parents=True)
    (project_path / "src" / "minimal_project").mkdir(parents=True)
    
    # Create minimal catalog configuration
    catalog_config = {
        'test_plot': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/test_plot.png',
            'purpose': 'exploratory'
        }
    }
    
    catalog_path = project_path / "conf" / "base" / "catalog.yml"
    with open(catalog_path, 'w') as f:
        yaml.dump(catalog_config, f)
    
    # Create minimal FigRegistry configuration
    figregistry_config = {
        'styles': {
            'exploratory': {
                'figure.figsize': [8, 6],
                'axes.labelsize': 10
            }
        },
        'outputs': {
            'base_path': 'data/08_reporting'
        }
    }
    
    figregistry_path = project_path / "conf" / "base" / "figregistry.yml"
    with open(figregistry_path, 'w') as f:
        yaml.dump(figregistry_config, f)
    
    # Create settings.py with hook registration
    settings_content = '''"""Project settings."""
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
'''
    
    settings_path = project_path / "src" / "minimal_project" / "settings.py"
    with open(settings_path, 'w') as f:
        f.write(settings_content)
    
    return {
        'project_path': project_path,
        'conf_path': project_path / "conf",
        'catalog_path': catalog_path,
        'figregistry_path': figregistry_path,
        'settings_path': settings_path,
        'data_path': project_path / "data"
    }


@pytest.fixture
def project_scaffolding_factory(tmp_path):
    """
    Factory for creating multiple scaffolded projects with different configurations.
    
    Provides a callable factory that creates isolated Kedro project structures
    with customizable FigRegistry configurations for testing different scenarios.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Callable factory for creating scaffolded projects
    """
    created_projects = []
    
    def create_project(
        name: str = "test_project",
        config_scenario: str = "basic",
        include_hooks: bool = True,
        include_datasets: bool = True
    ) -> Dict[str, Any]:
        """
        Create a scaffolded project with specified configuration.
        
        Args:
            name: Project name
            config_scenario: Configuration scenario (basic, advanced, minimal)
            include_hooks: Whether to include FigRegistryHooks registration
            include_datasets: Whether to include FigureDataSet in catalog
            
        Returns:
            Dictionary with project paths and configuration
        """
        project_path = tmp_path / name
        project_path.mkdir()
        
        # Create directory structure
        (project_path / "conf" / "base").mkdir(parents=True)
        (project_path / "conf" / "local").mkdir(parents=True)
        (project_path / "data" / "08_reporting").mkdir(parents=True)
        (project_path / "src" / name).mkdir(parents=True)
        
        # Configure based on scenario
        if config_scenario == "basic":
            config = {
                'styles': {'default': {'figure.figsize': [8, 6]}},
                'outputs': {'base_path': 'data/08_reporting'}
            }
            catalog = {
                'basic_plot': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': 'data/08_reporting/basic_plot.png'
                }
            } if include_datasets else {}
            
        elif config_scenario == "advanced":
            config = {
                'styles': {
                    'exploratory': {'figure.figsize': [10, 6]},
                    'presentation': {'figure.figsize': [12, 8]}
                },
                'outputs': {'versioning': True}
            }
            catalog = {
                'advanced_plot': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': 'data/08_reporting/advanced_plot.png',
                    'versioned': True
                }
            } if include_datasets else {}
            
        else:  # minimal
            config = {'styles': {'minimal': {}}}
            catalog = {}
        
        # Write configuration files
        if config:
            config_path = project_path / "conf" / "base" / "figregistry.yml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        
        if catalog:
            catalog_path = project_path / "conf" / "base" / "catalog.yml"
            with open(catalog_path, 'w') as f:
                yaml.dump(catalog, f)
        
        # Create settings.py with optional hooks
        if include_hooks:
            settings_content = '''"""Project settings."""
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)
'''
        else:
            settings_content = '''"""Project settings."""

HOOKS = ()
'''
        
        settings_path = project_path / "src" / name / "settings.py"
        with open(settings_path, 'w') as f:
            f.write(settings_content)
        
        project_info = {
            'name': name,
            'path': project_path,
            'config_scenario': config_scenario,
            'include_hooks': include_hooks,
            'include_datasets': include_datasets
        }
        
        created_projects.append(project_info)
        return project_info
    
    yield create_project
    
    # Cleanup created projects
    for project_info in created_projects:
        try:
            import shutil
            shutil.rmtree(project_info['path'])
        except Exception:
            pass  # Ignore cleanup errors in tests


# =============================================================================
# Performance and Validation Fixtures
# =============================================================================

@pytest.fixture
def hook_performance_tracker(mocker):
    """
    Performance tracking for hook operations to validate <25ms initialization requirements.
    
    Provides performance tracking utilities for validating that FigRegistryHooks
    meet the <25ms initialization overhead requirements specified in Section 6.6.4.3.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Performance tracking utilities for hook testing
    """
    import time
    
    timing_data = []
    
    def track_hook_timing(hook_method, *args, **kwargs):
        """Track execution time for hook methods."""
        start_time = time.perf_counter()
        try:
            result = hook_method(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            timing_data.append({
                'method': getattr(hook_method, '__name__', 'unknown'),
                'execution_time_ms': execution_time,
                'success': success,
                'error': error,
                'timestamp': start_time
            })
        
        return result
    
    def get_timing_summary():
        """Get performance summary for analysis."""
        if not timing_data:
            return {'total_methods': 0, 'avg_time_ms': 0, 'max_time_ms': 0}
        
        times = [entry['execution_time_ms'] for entry in timing_data]
        return {
            'total_methods': len(timing_data),
            'avg_time_ms': sum(times) / len(times),
            'max_time_ms': max(times),
            'min_time_ms': min(times),
            'total_time_ms': sum(times),
            'detailed_timings': timing_data
        }
    
    def validate_performance_targets():
        """Validate performance against specified targets."""
        summary = get_timing_summary()
        if summary['total_methods'] == 0:
            return {'status': 'no_data', 'message': 'No timing data available'}
        
        # Check against 25ms hook initialization target
        max_time = summary['max_time_ms']
        avg_time = summary['avg_time_ms']
        
        validation = {
            'status': 'pass' if max_time < 25.0 else 'fail',
            'max_time_target': 25.0,
            'actual_max_time': max_time,
            'avg_time': avg_time,
            'message': f"Max time: {max_time:.2f}ms (target: <25ms)"
        }
        
        return validation
    
    return {
        'track_timing': track_hook_timing,
        'get_summary': get_timing_summary,
        'validate_targets': validate_performance_targets,
        'timing_data': timing_data
    }


@pytest.fixture
def mock_matplotlib_figure(mocker):
    """
    Mock matplotlib figure for testing without GUI dependencies.
    
    Provides a mock matplotlib Figure instance for testing FigureDataSet
    operations without requiring full matplotlib GUI stack or figure rendering.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock matplotlib Figure for testing
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        # Create real figure for testing if matplotlib available
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([1, 2, 3], [1, 4, 2], label="Test Data")
        ax.set_title("Mock Test Figure")
        ax.legend()
        
        return fig
    except ImportError:
        # Fallback to mock if matplotlib not available
        mock_figure = mocker.Mock()
        mock_figure.savefig = mocker.Mock()
        mock_figure.get_size_inches = mocker.Mock(return_value=(8, 6))
        mock_figure.get_dpi = mocker.Mock(return_value=100)
        mock_figure.axes = [mocker.Mock()]
        
        return mock_figure


# =============================================================================
# Validation and Assertion Helpers
# =============================================================================

@pytest.fixture
def kedro_integration_validators():
    """
    Validation utilities for Kedro integration testing.
    
    Provides helper functions for validating proper integration between
    FigRegistry components and Kedro framework elements.
    
    Returns:
        Dictionary of validation functions
    """
    
    def validate_hook_registration(hook_manager, hook_instance):
        """Validate that hooks are properly registered."""
        if hasattr(hook_manager, '_registered_hooks'):
            return hook_instance in hook_manager._registered_hooks
        return hasattr(hook_manager, 'register') and callable(hook_manager.register)
    
    def validate_dataset_interface(dataset_instance):
        """Validate AbstractDataSet interface compliance."""
        required_methods = ['_save', '_load', '_describe']
        return all(hasattr(dataset_instance, method) for method in required_methods)
    
    def validate_config_bridge_initialization(config_bridge):
        """Validate configuration bridge is properly initialized."""
        return (
            hasattr(config_bridge, '_merged_config') or
            hasattr(config_bridge, 'get_merged_config') or
            callable(getattr(config_bridge, 'init_config', None))
        )
    
    def validate_catalog_integration(catalog, dataset_names):
        """Validate that datasets are properly integrated in catalog."""
        catalog_datasets = getattr(catalog, 'list', lambda: [])()
        return all(name in catalog_datasets for name in dataset_names)
    
    def validate_session_context(session):
        """Validate that session has required context components."""
        if hasattr(session, 'load_context'):
            context = session.load_context()
            return (
                hasattr(context, 'config_loader') and
                hasattr(context, 'catalog') and
                hasattr(context, 'project_path')
            )
        return hasattr(session, '_project_path')
    
    return {
        'validate_hook_registration': validate_hook_registration,
        'validate_dataset_interface': validate_dataset_interface,
        'validate_config_bridge': validate_config_bridge_initialization,
        'validate_catalog_integration': validate_catalog_integration,
        'validate_session_context': validate_session_context
    }


# =============================================================================
# Fixture Groups and Collections
# =============================================================================

@pytest.fixture
def complete_kedro_mock_stack(
    minimal_kedro_context,
    test_catalog_with_figregistry,
    mock_kedro_session,
    mock_hook_manager,
    figregistry_config_bridge
):
    """
    Complete Kedro component mock stack for comprehensive testing.
    
    Combines all essential Kedro component mocks into a single fixture
    for comprehensive plugin testing scenarios requiring full component
    integration without real Kedro project overhead.
    
    Args:
        All individual component fixtures
        
    Returns:
        Dictionary containing all mocked Kedro components
    """
    return {
        'context': minimal_kedro_context,
        'catalog': test_catalog_with_figregistry,
        'session': mock_kedro_session,
        'hook_manager': mock_hook_manager,
        'config_bridge': figregistry_config_bridge
    }


@pytest.fixture
def kedro_testing_utilities(
    kedro_integration_validators,
    hook_performance_tracker,
    mock_matplotlib_figure
):
    """
    Collection of testing utilities for Kedro plugin validation.
    
    Combines validation helpers, performance tracking, and test data
    into a comprehensive testing utility collection for plugin testing.
    
    Args:
        Individual utility fixtures
        
    Returns:
        Dictionary of testing utilities
    """
    return {
        'validators': kedro_integration_validators,
        'performance': hook_performance_tracker,
        'test_figure': mock_matplotlib_figure
    }