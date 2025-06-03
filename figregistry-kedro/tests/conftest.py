"""
Main pytest configuration file for figregistry-kedro test suite.

This module provides comprehensive test configuration, shared fixtures, and setup
patterns for testing the figregistry-kedro plugin. It establishes testing patterns
used across all test modules and ensures proper isolation and cleanup for reliable
test execution.

Key Responsibilities:
- Configure pytest for comprehensive plugin testing with kedro-pytest framework
- Provide shared fixtures for Kedro session simulation and plugin component testing
- Implement test isolation and cleanup patterns for reliable test execution
- Setup performance testing infrastructure with pytest-benchmark integration
- Configure comprehensive mocking of Kedro components and FigRegistry API calls

Testing Framework Integration per Section 6.6.2.1:
- pytest >=8.0.0 as core test runner with advanced fixture support
- pytest-cov >=6.1.0 for code coverage measurement and reporting
- pytest-mock >=3.14.0 for mocking capabilities with Kedro components
- kedro-pytest >=0.1.3 for Kedro plugin testing with TestKedro fixtures
- pytest-benchmark for performance overhead measurement

Coverage Targets per Section 6.6.2.4:
- ≥90% coverage for all figregistry_kedro modules
- 100% coverage for critical paths (config merge, hook registration, dataset operations)
- Comprehensive validation of plugin integration without core system modifications
"""

import os
import sys
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union
from unittest.mock import Mock, MagicMock, patch

import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kedro.io import DataCatalog
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.session import KedroSession
from kedro.pipeline import Pipeline

# Test isolation and performance requirements
matplotlib.use('Agg')  # Non-interactive backend for headless testing
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='kedro')


# =============================================================================
# PYTEST CONFIGURATION AND HOOKS
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest for comprehensive figregistry-kedro plugin testing.
    
    Sets up testing environment with enhanced coverage settings, performance
    monitoring, and plugin-specific validation patterns per Section 6.6.2.1.
    """
    # Configure matplotlib for consistent test behavior
    matplotlib.rcdefaults()
    plt.ioff()  # Turn off interactive mode
    
    # Set up test environment variables
    os.environ['FIGREGISTRY_TEST_MODE'] = 'true'
    os.environ['KEDRO_DISABLE_TELEMETRY'] = 'true'
    
    # Configure kedro-pytest settings per Section 6.6.2.1
    config.option.kedro_project_template = 'minimal'
    config.option.kedro_enable_hooks = True
    config.option.kedro_catalog_fixtures = True
    
    # Add custom markers for plugin testing
    config.addinivalue_line(
        "markers",
        "kedro_integration: marks tests as Kedro integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "plugin_performance: marks tests as plugin performance benchmarks"
    )
    config.addinivalue_line(
        "markers",
        "security_test: marks tests as security validation tests"
    )
    config.addinivalue_line(
        "markers",
        "cross_platform: marks tests for cross-platform compatibility"
    )


def pytest_unconfigure(config):
    """
    Clean up test configuration and ensure proper resource cleanup.
    
    Performs comprehensive cleanup of test artifacts, temporary directories,
    and plugin state to prevent cross-test contamination per Section 6.6.5.6.
    """
    # Reset matplotlib state
    matplotlib.rcdefaults()
    plt.close('all')
    
    # Clean up environment variables
    os.environ.pop('FIGREGISTRY_TEST_MODE', None)
    os.environ.pop('KEDRO_DISABLE_TELEMETRY', None)
    
    # Clear any cached module state
    _clear_plugin_module_cache()


def pytest_runtest_setup(item):
    """
    Set up individual test execution with proper isolation.
    
    Ensures each test starts with clean matplotlib state and isolated
    temporary directories per Section 6.6.5.6 isolation requirements.
    """
    # Reset matplotlib to default state before each test
    matplotlib.rcdefaults()
    plt.close('all')
    
    # Clear any figure registry module caches
    _clear_figregistry_cache()


def pytest_runtest_teardown(item, nextitem):
    """
    Clean up after individual test execution.
    
    Ensures proper cleanup of matplotlib figures, temporary files,
    and plugin state between tests per Section 6.6.5.6.
    """
    # Close all matplotlib figures to prevent memory leaks
    plt.close('all')
    
    # Reset any modified matplotlib rcParams
    matplotlib.rcdefaults()
    
    # Clear plugin state for next test
    _clear_plugin_state()


# =============================================================================
# CORE TESTING INFRASTRUCTURE FIXTURES
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """
    Session-wide test setup and configuration.
    
    Establishes global test environment settings, performance baselines,
    and security constraints for the entire test session per testing
    strategy requirements.
    """
    # Ensure matplotlib backend consistency
    matplotlib.use('Agg')
    plt.ioff()
    
    # Set global test configuration
    os.environ['FIGREGISTRY_LOG_LEVEL'] = 'DEBUG'
    os.environ['KEDRO_LOGGING_CONFIG'] = str(
        Path(__file__).parent / "data" / "logging.yml"
    )
    
    # Create session-wide temporary directory
    session_temp_dir = tempfile.mkdtemp(prefix="figregistry_kedro_test_")
    
    yield session_temp_dir
    
    # Cleanup session resources
    shutil.rmtree(session_temp_dir, ignore_errors=True)
    

@pytest.fixture(autouse=True)
def reset_matplotlib_state():
    """
    Reset matplotlib state between tests for isolation.
    
    Ensures clean matplotlib environment for each test execution,
    preventing cross-test contamination per Section 6.6.5.6.
    """
    # Store initial rcParams state
    initial_rcparams = matplotlib.rcParams.copy()
    
    # Reset to defaults before test
    matplotlib.rcdefaults()
    plt.close('all')
    
    yield
    
    # Restore initial state and close figures
    matplotlib.rcParams.update(initial_rcparams)
    plt.close('all')


@pytest.fixture
def temp_work_dir():
    """
    Provide isolated temporary working directory for test operations.
    
    Creates unique temporary directory for each test to ensure file
    system isolation and prevent cross-test contamination.
    """
    with tempfile.TemporaryDirectory(prefix="kedro_test_") as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        yield Path(temp_dir)
        
        os.chdir(original_cwd)


@pytest.fixture
def mock_file_operations(mocker):
    """
    Mock file system operations for testing isolation.
    
    Provides controlled file system mocking for testing file operations
    without actual disk I/O, supporting security testing per Section 6.6.8.2.
    """
    # Mock pathlib operations
    mock_path = mocker.patch('pathlib.Path')
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.is_file.return_value = True
    mock_path.return_value.mkdir.return_value = None
    
    # Mock file operations
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    
    return {
        'path': mock_path,
        'open': mock_open,
        'mkdir': mock_path.return_value.mkdir,
        'exists': mock_path.return_value.exists,
    }


# =============================================================================
# KEDRO TESTING INFRASTRUCTURE FIXTURES
# =============================================================================

@pytest.fixture
def mock_kedro_context(mocker):
    """
    Provide mock Kedro ProjectContext for plugin testing.
    
    Creates comprehensive mock of Kedro ProjectContext with ConfigLoader,
    DataCatalog, and session components for isolated plugin testing
    per Section 6.6.2.3 mocking strategy.
    """
    # Create mock context with required attributes
    context = mocker.Mock(spec=KedroContext)
    
    # Mock ConfigLoader with figregistry configuration
    config_loader = mocker.Mock(spec=ConfigLoader)
    config_loader.get.return_value = {
        'styles': {
            'exploratory': {'figure.figsize': [10, 6]},
            'presentation': {'figure.figsize': [12, 8]},
            'publication': {'figure.figsize': [8, 6]}
        },
        'outputs': {
            'base_path': 'data/08_reporting/figures',
            'timestamp_format': '%Y%m%d_%H%M%S'
        }
    }
    context.config_loader = config_loader
    
    # Mock DataCatalog
    catalog = mocker.Mock(spec=DataCatalog)
    catalog.save = mocker.Mock()
    catalog.load = mocker.Mock()
    context.catalog = catalog
    
    # Mock project path and configuration
    context.project_path = Path('/tmp/test_project')
    context.package_name = 'test_project'
    context.project_version = '0.1.0'
    
    return context


@pytest.fixture
def mock_kedro_session(mock_kedro_context, mocker, temp_work_dir):
    """
    Provide mock Kedro session for complete pipeline simulation.
    
    Creates comprehensive Kedro session mock with proper context
    initialization for testing hook lifecycle and plugin integration
    per Section 6.6.2.6 TestKedro fixture requirements.
    """
    # Create mock session
    session = mocker.Mock(spec=KedroSession)
    session.load_context.return_value = mock_kedro_context
    
    # Mock session properties
    session.store = {}
    session._project_path = temp_work_dir
    session._package_name = 'test_project'
    
    # Mock run method for pipeline execution testing
    def mock_run(pipeline_name=None, **kwargs):
        return {'pipeline_output': 'success'}
    
    session.run = mock_run
    
    # Setup session context manager behavior
    session.__enter__ = mocker.Mock(return_value=session)
    session.__exit__ = mocker.Mock(return_value=None)
    
    return session


@pytest.fixture
def mock_hook_manager(mocker):
    """
    Provide mock hook manager for hook registration testing.
    
    Creates mock hook manager for testing FigRegistryHooks registration
    and lifecycle integration per Section 6.6.3.8 hook testing requirements.
    """
    hook_manager = mocker.Mock()
    
    # Mock hook registration methods
    hook_manager.register = mocker.Mock()
    hook_manager.unregister = mocker.Mock()
    hook_manager.get_plugins = mocker.Mock(return_value=[])
    
    # Mock hook execution methods
    hook_manager.hook = mocker.Mock()
    hook_manager.hook.before_pipeline_run = mocker.Mock()
    hook_manager.hook.after_pipeline_run = mocker.Mock()
    hook_manager.hook.after_config_loaded = mocker.Mock()
    
    return hook_manager


@pytest.fixture
def sample_catalog_config():
    """
    Provide sample Kedro catalog configuration with FigureDataSet entries.
    
    Returns comprehensive catalog configuration for testing FigureDataSet
    integration, parameter extraction, and versioning scenarios
    per Section 5.2.6 dataset requirements.
    """
    return {
        'exploratory_plot': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/figures/exploratory_plot.png',
            'purpose': 'exploratory',
            'condition_param': 'experiment_type',
            'save_args': {
                'dpi': 150,
                'bbox_inches': 'tight'
            }
        },
        'presentation_plot': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/figures/presentation_plot.pdf',
            'purpose': 'presentation',
            'condition_param': 'analysis_mode',
            'style_params': {
                'figure.figsize': [12, 8],
                'axes.labelsize': 14
            },
            'versioned': True
        },
        'publication_plot': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': 'data/08_reporting/figures/publication_plot.svg',
            'purpose': 'publication',
            'condition_param': 'publication_target',
            'save_args': {
                'format': 'svg',
                'transparent': True
            }
        }
    }


# =============================================================================
# MATPLOTLIB AND FIGREGISTRY TESTING FIXTURES
# =============================================================================

@pytest.fixture
def sample_matplotlib_figure():
    """
    Create sample matplotlib figure for dataset testing.
    
    Provides standard matplotlib figure object with basic plot content
    for testing FigureDataSet save/load operations and styling application
    per Section 5.2.6 testing requirements.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create plot
    ax.plot(x, y, label='sin(x)')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.set_title('Sample Plot for Testing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


@pytest.fixture
def complex_matplotlib_figure():
    """
    Create complex matplotlib figure for advanced testing scenarios.
    
    Provides matplotlib figure with subplots, multiple data series,
    and complex styling for testing performance and advanced features.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.exp(-x/5)
    
    # Plot 1: Line plot
    ax1.plot(x, y1, 'b-', label='sin(x)')
    ax1.plot(x, y2, 'r--', label='cos(x)')
    ax1.set_title('Trigonometric Functions')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Scatter plot
    np.random.seed(42)
    x_scatter = np.random.randn(100)
    y_scatter = np.random.randn(100)
    ax2.scatter(x_scatter, y_scatter, alpha=0.6)
    ax2.set_title('Random Scatter')
    ax2.grid(True)
    
    # Plot 3: Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    ax3.bar(categories, values, color='green', alpha=0.7)
    ax3.set_title('Category Data')
    ax3.set_ylabel('Values')
    
    # Plot 4: Exponential decay
    ax4.plot(x, y3, 'purple', linewidth=2)
    ax4.set_title('Exponential Decay')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig


@pytest.fixture
def figregistry_test_config():
    """
    Provide test FigRegistry configuration for plugin testing.
    
    Returns valid FigRegistry configuration dictionary for testing
    configuration bridge, style resolution, and plugin integration
    per Section 5.2.5 configuration requirements.
    """
    return {
        'styles': {
            'exploratory': {
                'figure.figsize': [10, 6],
                'axes.grid': True,
                'axes.grid.alpha': 0.3,
                'font.size': 10
            },
            'presentation': {
                'figure.figsize': [12, 8],
                'axes.grid': True,
                'axes.grid.alpha': 0.2,
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16
            },
            'publication': {
                'figure.figsize': [8, 6],
                'axes.grid': False,
                'font.size': 11,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'legend.fontsize': 10
            }
        },
        'outputs': {
            'base_path': 'data/08_reporting/figures',
            'timestamp_format': '%Y%m%d_%H%M%S',
            'path_aliases': {
                'expl': 'exploratory',
                'pres': 'presentation',
                'pub': 'publication'
            }
        },
        'conditions': {
            'experiment_type': {
                'baseline': 'exploratory',
                'optimization': 'presentation',
                'final_results': 'publication'
            },
            'analysis_mode': {
                'development': 'exploratory',
                'review': 'presentation',
                'publication': 'publication'
            }
        }
    }


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def performance_baseline():
    """
    Provide performance baseline measurements for plugin overhead testing.
    
    Establishes baseline timing measurements for configuration loading,
    style resolution, and dataset operations per Section 6.6.4.3
    performance requirements.
    """
    return {
        'config_load_time': 0.050,  # 50ms target
        'style_resolution_time': 0.001,  # 1ms target
        'dataset_save_overhead': 0.200,  # 200ms target
        'hook_initialization_time': 0.025,  # 25ms target
        'memory_overhead_mb': 5.0  # 5MB target
    }


@pytest.fixture
def benchmark_config(benchmark):
    """
    Configure pytest-benchmark for plugin performance testing.
    
    Sets up benchmark configuration for measuring plugin operations
    with appropriate warmup rounds and timing precision per performance
    testing requirements.
    """
    # Configure benchmark settings
    benchmark.weave = lambda func, *args, **kwargs: func(*args, **kwargs)
    benchmark.group = "figregistry_kedro_plugin"
    
    # Set performance thresholds
    benchmark.stats = {
        'rounds': 100,
        'warmup_rounds': 10,
        'disable_gc': True,
        'timer': 'time.perf_counter'
    }
    
    return benchmark


# =============================================================================
# SECURITY TESTING FIXTURES
# =============================================================================

@pytest.fixture
def security_test_configs():
    """
    Provide malicious configuration test data for security validation.
    
    Returns configuration scenarios for testing path traversal prevention,
    YAML injection protection, and configuration validation security
    per Section 6.6.8.1 security requirements.
    """
    return {
        'path_traversal_config': {
            'outputs': {
                'base_path': '../../../etc/passwd',
                'path_aliases': {
                    'evil': '../../home/user/.ssh/id_rsa'
                }
            }
        },
        'yaml_injection_config': {
            'styles': {
                '!!python/object/apply:os.system': ['rm -rf /']
            }
        },
        'oversized_config': {
            'styles': {f'condition_{i}': {'figure.figsize': [10, 6]} 
                      for i in range(10000)}
        },
        'invalid_types_config': {
            'styles': {
                'test': {
                    'figure.figsize': 'not_a_list',
                    'axes.grid': 'not_a_boolean'
                }
            }
        }
    }


@pytest.fixture
def catalog_security_test_data():
    """
    Provide malicious catalog configurations for security testing.
    
    Returns catalog entry configurations for testing parameter validation,
    path security, and injection prevention per Section 6.6.8.2.
    """
    return {
        'path_traversal_catalog': {
            'malicious_dataset': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': '../../../etc/passwd',
                'purpose': 'publication'
            }
        },
        'parameter_injection_catalog': {
            'injection_dataset': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/test.png',
                'condition_param': '"; rm -rf /; echo "',
                'purpose': 'exploratory'
            }
        },
        'oversized_parameters_catalog': {
            'oversized_dataset': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/test.png',
                'style_params': {f'param_{i}': f'value_{i}' 
                               for i in range(10000)}
            }
        }
    }


# =============================================================================
# CROSS-PLATFORM TESTING FIXTURES
# =============================================================================

@pytest.fixture
def cross_platform_test_env(monkeypatch):
    """
    Provide cross-platform testing environment simulation.
    
    Simulates different operating systems and path conventions for
    testing cross-platform compatibility per Section 6.6.1.4.
    """
    platforms = {
        'windows': {
            'os.name': 'nt',
            'os.sep': '\\',
            'pathlib.Path.home': lambda: Path('C:/Users/testuser')
        },
        'linux': {
            'os.name': 'posix',
            'os.sep': '/',
            'pathlib.Path.home': lambda: Path('/home/testuser')
        },
        'macos': {
            'os.name': 'posix',
            'os.sep': '/',
            'pathlib.Path.home': lambda: Path('/Users/testuser')
        }
    }
    
    def set_platform(platform_name):
        if platform_name not in platforms:
            raise ValueError(f"Unsupported platform: {platform_name}")
        
        platform_config = platforms[platform_name]
        for attr, value in platform_config.items():
            if '.' in attr:
                module_name, attr_name = attr.rsplit('.', 1)
                monkeypatch.setattr(f"{module_name}.{attr_name}", value)
            else:
                monkeypatch.setattr(attr, value)
    
    return set_platform


# =============================================================================
# UTILITY FUNCTIONS FOR TEST CLEANUP AND STATE MANAGEMENT
# =============================================================================

def _clear_plugin_module_cache():
    """
    Clear cached plugin modules to prevent cross-test contamination.
    
    Removes figregistry_kedro modules from sys.modules cache to ensure
    clean import state for each test execution.
    """
    modules_to_clear = [
        mod for mod in sys.modules.keys() 
        if mod.startswith('figregistry_kedro')
    ]
    
    for module in modules_to_clear:
        sys.modules.pop(module, None)


def _clear_figregistry_cache():
    """
    Clear FigRegistry internal caches and state.
    
    Resets FigRegistry configuration cache and style resolution cache
    to ensure clean state for each test.
    """
    try:
        import figregistry
        # Clear configuration cache if available
        if hasattr(figregistry, '_config_cache'):
            figregistry._config_cache.clear()
        
        # Clear style cache if available  
        if hasattr(figregistry, '_style_cache'):
            figregistry._style_cache.clear()
            
    except ImportError:
        # FigRegistry not available, skip clearing
        pass


def _clear_plugin_state():
    """
    Clear plugin-specific state and caches.
    
    Ensures plugin components start with clean state for each test,
    preventing cross-test contamination per Section 6.6.5.6.
    """
    try:
        # Clear any plugin-specific caches
        import figregistry_kedro
        
        # Clear hook state if available
        if hasattr(figregistry_kedro, '_hook_state'):
            figregistry_kedro._hook_state.clear()
            
        # Clear dataset cache if available
        if hasattr(figregistry_kedro, '_dataset_cache'):
            figregistry_kedro._dataset_cache.clear()
            
        # Clear config bridge cache if available
        if hasattr(figregistry_kedro, '_bridge_cache'):
            figregistry_kedro._bridge_cache.clear()
            
    except ImportError:
        # Plugin not available, skip clearing
        pass


# =============================================================================
# TEST DATA IMPORT AND INTEGRATION
# =============================================================================

# Import shared test fixtures from dedicated modules
try:
    from .fixtures.kedro_fixtures import *
    from .fixtures.config_fixtures import *
    from .fixtures.dataset_fixtures import *
    from .fixtures.hook_fixtures import *
    from .fixtures.matplotlib_fixtures import *
    from .fixtures.project_fixtures import *
except ImportError as e:
    # Graceful handling for missing fixture modules during development
    warnings.warn(f"Could not import test fixtures: {e}", UserWarning)

# Import test data modules
try:
    from .data.config_test_data import *
except ImportError as e:
    warnings.warn(f"Could not import test data: {e}", UserWarning)


# =============================================================================
# PYTEST COLLECTION AND EXECUTION HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify collected test items for enhanced plugin testing.
    
    Adds automatic markers based on test patterns and configures
    test execution order for optimal performance and reliability.
    """
    for item in items:
        # Add kedro_integration marker for kedro-related tests
        if 'kedro' in item.nodeid.lower():
            item.add_marker(pytest.mark.kedro_integration)
        
        # Add plugin_performance marker for benchmark tests
        if 'benchmark' in item.nodeid.lower() or 'performance' in item.nodeid.lower():
            item.add_marker(pytest.mark.plugin_performance)
        
        # Add security_test marker for security validation
        if 'security' in item.nodeid.lower() or 'malicious' in item.nodeid.lower():
            item.add_marker(pytest.mark.security_test)
        
        # Add cross_platform marker for platform compatibility tests
        if 'cross_platform' in item.nodeid.lower() or 'platform' in item.nodeid.lower():
            item.add_marker(pytest.mark.cross_platform)


def pytest_sessionfinish(session, exitstatus):
    """
    Clean up after complete test session.
    
    Performs final cleanup of session-wide resources and generates
    summary reports for plugin testing validation.
    """
    # Generate test summary for plugin validation
    if hasattr(session.config, 'figregistry_test_summary'):
        summary = session.config.figregistry_test_summary
        print(f"\nFigRegistry-Kedro Plugin Test Summary:")
        print(f"  Total Tests: {summary.get('total', 0)}")
        print(f"  Plugin Tests: {summary.get('plugin_tests', 0)}")
        print(f"  Performance Tests: {summary.get('performance_tests', 0)}")
        print(f"  Security Tests: {summary.get('security_tests', 0)}")
    
    # Final cleanup
    _clear_plugin_module_cache()
    _clear_figregistry_cache()
    _clear_plugin_state()
    
    # Reset matplotlib
    matplotlib.rcdefaults()
    plt.close('all')