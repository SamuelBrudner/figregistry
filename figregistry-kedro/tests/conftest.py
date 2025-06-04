"""Main pytest configuration file for figregistry-kedro test suite.

This module provides comprehensive test configuration for the figregistry-kedro plugin
testing per Section 6.6.2.1 with kedro-pytest framework integration. Establishes
shared fixtures, test setup, configuration management, and testing patterns used
across all test modules.

Key capabilities per Section 6.6 Testing Strategy:
- pytest configuration for plugin testing with coverage settings for both core and plugin modules
- Shared fixtures for matplotlib figure mocking, temporary directory management, and test data generation
- kedro-pytest integration with TestKedro fixtures for in-process pipeline context simulation
- pytest-mock configuration for comprehensive mocking of Kedro components and FigRegistry API calls
- Test isolation and cleanup patterns to prevent cross-test contamination per Section 6.6.5.6
- Performance testing infrastructure with pytest-benchmark integration for plugin overhead measurement

Testing Framework Stack (Section 6.6.2.1):
- pytest >=8.0.0: Core test runner with advanced fixture support
- pytest-cov >=6.1.0: Code coverage measurement and reporting
- pytest-mock >=3.14.0: Mocking capabilities for external dependencies
- hypothesis >=6.0.0: Property-based testing for configuration validation
- kedro-pytest >=0.1.3: Kedro plugin testing framework providing TestKedro fixtures
- pytest-benchmark: Performance testing infrastructure for plugin overhead measurement

Performance Requirements (Section 6.6.4.3):
- Plugin Pipeline Execution Overhead: <200ms per FigureDataSet save
- Configuration Bridge Merge Time: <50ms per pipeline run
- Hook Initialization Overhead: <25ms per project startup

Coverage Requirements (Section 6.6.2.4):
- Minimum Coverage: 90% overall coverage across all figregistry_kedro modules
- Critical Path Coverage: 100% coverage for configuration bridge, hook registration, dataset save operations
- Module-Specific Targets: ≥90% for all figregistry_kedro modules with 100% for critical paths
"""

import os
import sys
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Iterator, Union, Callable
from unittest.mock import Mock, MagicMock, patch
import pytest
import logging

# Configure warnings for clean test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest for figregistry-kedro plugin testing per Section 6.6.2.1.
    
    Sets up comprehensive testing environment including coverage measurement,
    performance monitoring, and kedro-pytest integration for plugin validation.
    """
    # Configure coverage for both core and plugin modules
    config.option.cov = ["figregistry", "figregistry_kedro"]
    config.option.cov_report = ["term-missing", "html", "xml"]
    config.option.cov_fail_under = 90.0
    
    # Configure test markers for organized test execution
    config.addinivalue_line(
        "markers", 
        "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", 
        "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", 
        "kedro_plugin: Tests specific to Kedro plugin functionality"
    )
    config.addinivalue_line(
        "markers", 
        "performance: Performance tests with timing requirements"
    )
    config.addinivalue_line(
        "markers", 
        "security: Security tests for injection prevention"
    )
    config.addinivalue_line(
        "markers", 
        "slow: Tests that take longer than 10 seconds"
    )
    config.addinivalue_line(
        "markers", 
        "kedro_version: Tests for specific Kedro version compatibility"
    )
    
    # Configure logging for test debugging
    logging.basicConfig(
        level=logging.DEBUG if config.option.verbose > 1 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress matplotlib backend warnings in tests
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection for performance and organization per Section 6.6.7.
    
    Organizes test execution with proper markers and performance considerations.
    """
    for item in items:
        # Mark tests based on file location
        if "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        if "test_security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        if "kedro" in str(item.fspath).lower():
            item.add_marker(pytest.mark.kedro_plugin)
        
        # Mark slow tests
        if "slow" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.slow)


def pytest_benchmark_group_stats(benchmarks):
    """Configure benchmark grouping for performance analysis per Section 6.6.4.3."""
    return {
        'group': 'plugin_component',
        'param': benchmarks[0]['param'] if benchmarks else None
    }


# =============================================================================
# KEDRO FRAMEWORK IMPORTS AND AVAILABILITY
# =============================================================================

# Import Kedro components with graceful fallback
try:
    from kedro.framework.context import KedroContext
    from kedro.framework.session import KedroSession
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

# Hypothesis for property-based testing
try:
    import hypothesis
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# pytest-benchmark for performance testing
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Matplotlib for figure testing
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    matplotlib.use('Agg')  # Non-interactive backend for testing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# FIXTURE IMPORTS FROM DEPENDENCY MODULES
# =============================================================================

# Import all fixtures from kedro_fixtures.py
from figregistry_kedro.tests.fixtures.kedro_fixtures import (
    # Core Kedro component mocks
    minimal_kedro_context,
    mock_config_loader,
    test_catalog_with_figregistry,
    mock_kedro_session,
    mock_hook_manager,
    
    # FigRegistry integration fixtures
    figregistry_config_bridge,
    mock_figregistry_hooks,
    mock_figure_dataset,
    
    # TestKedro integration
    test_kedro_instance,
    kedro_pytest_session,
    
    # Project scaffolding
    minimal_project_scaffolding,
    project_scaffolding_factory,
    
    # Performance and validation
    hook_performance_tracker,
    mock_matplotlib_figure,
    kedro_integration_validators,
    
    # Fixture collections
    complete_kedro_mock_stack,
    kedro_testing_utilities
)

# Import configuration test data generators
from figregistry_kedro.tests.data.config_test_data import (
    # Configuration generators
    generate_baseline_config,
    generate_kedro_specific_config,
    generate_environment_configs,
    
    # Invalid configuration generators  
    generate_invalid_config_scenarios,
    generate_malformed_yaml_strings,
    
    # Merge testing
    generate_merge_test_scenarios,
    
    # Security testing
    generate_security_test_configs,
    generate_yaml_injection_vectors,
    
    # Performance testing
    generate_performance_config_datasets,
    generate_concurrent_access_configs,
    
    # Cross-platform testing
    generate_cross_platform_config_variations,
    generate_filesystem_edge_cases,
    
    # Utilities
    create_temporary_config_file,
    create_temporary_directory_structure,
    validate_config_against_schema,
    generate_test_report_summary,
    
    # Hypothesis strategies
    yaml_config_strategy,
    kedro_config_strategy,
    valid_color_strategy,
    valid_marker_strategy,
    style_dict_strategy
)


# =============================================================================
# SHARED PYTEST FIXTURES FOR TEST INFRASTRUCTURE
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure test environment for reliable execution per Section 6.6.5.6.
    
    Sets up session-wide test configuration including environment variables,
    logging, and infrastructure required for comprehensive plugin testing.
    """
    # Set environment variables for testing
    os.environ["FIGREGISTRY_ENV"] = "test"
    os.environ["KEDRO_ENV"] = "test"
    os.environ["FIGREGISTRY_DISABLE_TELEMETRY"] = "1"
    
    # Configure matplotlib for testing
    if MATPLOTLIB_AVAILABLE:
        import matplotlib
        matplotlib.use('Agg')
        plt.ioff()  # Turn off interactive mode
    
    # Configure logging for tests
    logging.getLogger("figregistry").setLevel(logging.WARNING)
    logging.getLogger("kedro").setLevel(logging.WARNING)
    
    # Ensure clean test state
    yield
    
    # Cleanup after all tests
    for key in ["FIGREGISTRY_ENV", "KEDRO_ENV", "FIGREGISTRY_DISABLE_TELEMETRY"]:
        os.environ.pop(key, None)


@pytest.fixture(scope="function", autouse=True)
def test_isolation():
    """Implement test isolation patterns per Section 6.6.5.6.
    
    Ensures each test runs in isolation with clean state and proper cleanup
    to prevent cross-test contamination of plugin components.
    """
    # Store original state
    original_cwd = os.getcwd()
    original_sys_path = sys.path.copy()
    
    # Reset matplotlib state if available
    if MATPLOTLIB_AVAILABLE:
        plt.close('all')  # Close all figures
        plt.rcdefaults()  # Reset rcParams to defaults
    
    yield
    
    # Restore original state
    os.chdir(original_cwd)
    sys.path = original_sys_path
    
    # Clean up matplotlib state
    if MATPLOTLIB_AVAILABLE:
        plt.close('all')
        plt.rcdefaults()
    
    # Force garbage collection for memory management
    import gc
    gc.collect()


@pytest.fixture
def temp_directory():
    """Provide temporary directory with automatic cleanup per Section 6.6.2.6.
    
    Creates isolated temporary directory for each test with guaranteed cleanup
    to prevent filesystem contamination during testing.
    
    Returns:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="figregistry_test_"))
    try:
        yield temp_dir
    finally:
        # Ensure cleanup even if test fails
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_project_directory(temp_directory):
    """Create temporary Kedro project directory structure per Section 6.6.7.2.
    
    Provides realistic Kedro project structure for plugin testing with
    proper directory hierarchy and configuration placeholders.
    
    Args:
        temp_directory: Base temporary directory fixture
        
    Returns:
        Dict with project paths and configuration
    """
    project_path = temp_directory / "test_kedro_project"
    project_path.mkdir()
    
    # Create standard Kedro directory structure
    directories = [
        "conf/base",
        "conf/local", 
        "data/01_raw",
        "data/02_intermediate",
        "data/03_primary",
        "data/08_reporting",
        "src/test_kedro_project",
        "logs"
    ]
    
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
    
    # Create minimal pyproject.toml
    pyproject_content = '''[tool.kedro]
package_name = "test_kedro_project"
project_name = "Test Kedro Project"
kedro_init_version = "0.19.0"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
'''
    
    with open(project_path / "pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    return {
        "project_path": project_path,
        "conf_path": project_path / "conf",
        "data_path": project_path / "data",
        "src_path": project_path / "src",
        "logs_path": project_path / "logs"
    }


@pytest.fixture
def mock_matplotlib_rcparams():
    """Mock matplotlib rcParams for testing without GUI dependencies.
    
    Provides controlled matplotlib rcParams environment for testing
    style application without requiring full matplotlib GUI stack.
    
    Returns:
        Mock rcParams object for testing
    """
    if not MATPLOTLIB_AVAILABLE:
        # Create mock rcParams when matplotlib not available
        mock_rcparams = Mock()
        mock_rcparams.update = Mock()
        mock_rcparams.get = Mock(return_value="default_value")
        mock_rcparams.__getitem__ = Mock(return_value="default_value")
        mock_rcparams.__setitem__ = Mock()
        return mock_rcparams
    
    # Use real rcParams but reset after test
    original_params = plt.rcParams.copy()
    try:
        yield plt.rcParams
    finally:
        plt.rcParams.update(original_params)


@pytest.fixture
def sample_matplotlib_figure():
    """Generate sample matplotlib figure for dataset testing.
    
    Creates a representative matplotlib figure for testing FigureDataSet
    operations, styling application, and save functionality.
    
    Returns:
        Matplotlib Figure object for testing
    """
    if not MATPLOTLIB_AVAILABLE:
        # Create mock figure when matplotlib not available
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        mock_figure.get_size_inches = Mock(return_value=(8, 6))
        mock_figure.get_dpi = Mock(return_value=100)
        mock_figure.axes = [Mock()]
        return mock_figure
    
    # Create real matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Add sample data for realistic testing
    import numpy as np
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    ax.plot(x, y, 'b-', label='Sample Data', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Sample Figure for Testing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def base_figregistry_config():
    """Provide base FigRegistry configuration for testing.
    
    Returns standard baseline configuration for component testing
    with comprehensive style definitions and output settings.
    
    Returns:
        Dict containing base FigRegistry configuration
    """
    return generate_baseline_config()


@pytest.fixture
def local_override_config():
    """Provide local environment configuration overrides.
    
    Returns configuration overrides for testing environment-specific
    behavior and configuration merging functionality.
    
    Returns:
        Dict containing local configuration overrides
    """
    return generate_environment_configs()["development"]


@pytest.fixture
def minimal_figregistry_config():
    """Provide minimal FigRegistry configuration for lightweight testing.
    
    Returns:
        Dict containing minimal valid configuration
    """
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "test": {"color": "#1f77b4", "marker": "o"}
        },
        "outputs": {
            "base_path": "test_outputs"
        }
    }


@pytest.fixture
def invalid_config_scenarios():
    """Provide invalid configuration scenarios for error testing.
    
    Returns:
        Dict mapping scenario names to invalid configurations
    """
    return generate_invalid_config_scenarios()


@pytest.fixture
def security_test_configs():
    """Provide security test configurations per Section 6.6.8.1.
    
    Returns malicious configuration structures for testing security
    validation including path traversal and injection prevention.
    
    Returns:
        Dict mapping attack vector names to malicious configurations
    """
    return generate_security_test_configs()


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_config():
    """Configure benchmark settings for performance testing per Section 6.6.4.3.
    
    Returns benchmark configuration for measuring plugin performance
    against specified targets (<200ms dataset save, <50ms config merge, <25ms hook init).
    
    Returns:
        Dict containing benchmark configuration
    """
    if not BENCHMARK_AVAILABLE:
        pytest.skip("pytest-benchmark not available for performance testing")
    
    return {
        'min_rounds': 5,
        'max_time': 10.0,  # Maximum 10 seconds per benchmark
        'timer': 'time.perf_counter',
        'disable_gc': True,
        'warmup': True,
        'warmup_iterations': 2
    }


@pytest.fixture
def performance_config_datasets():
    """Provide configuration datasets for performance benchmarking.
    
    Returns iterator of configuration datasets with varying complexity
    for measuring configuration bridge performance.
    
    Returns:
        Iterator of (test_name, config_dict, expected_time_ms) tuples
    """
    return generate_performance_config_datasets()


@pytest.fixture
def performance_validator():
    """Provide performance validation utilities.
    
    Returns utilities for validating plugin performance against
    specified targets from Section 6.6.4.3.
    
    Returns:
        Dict containing performance validation functions
    """
    def validate_dataset_save_time(execution_time_ms: float) -> bool:
        """Validate FigureDataSet save operation against 200ms target."""
        return execution_time_ms < 200.0
    
    def validate_config_bridge_time(execution_time_ms: float) -> bool:
        """Validate config bridge merge against 50ms target."""
        return execution_time_ms < 50.0
    
    def validate_hook_init_time(execution_time_ms: float) -> bool:
        """Validate hook initialization against 25ms target."""
        return execution_time_ms < 25.0
    
    def create_performance_report(results: Dict[str, float]) -> str:
        """Create performance validation report."""
        report_lines = ["Performance Validation Report", "=" * 30]
        
        for metric, time_ms in results.items():
            if "dataset_save" in metric:
                status = "PASS" if validate_dataset_save_time(time_ms) else "FAIL"
                target = "200ms"
            elif "config_bridge" in metric:
                status = "PASS" if validate_config_bridge_time(time_ms) else "FAIL"
                target = "50ms"
            elif "hook_init" in metric:
                status = "PASS" if validate_hook_init_time(time_ms) else "FAIL"
                target = "25ms"
            else:
                status = "UNKNOWN"
                target = "N/A"
            
            report_lines.append(f"{metric}: {time_ms:.2f}ms (target: <{target}) [{status}]")
        
        return "\n".join(report_lines)
    
    return {
        'validate_dataset_save': validate_dataset_save_time,
        'validate_config_bridge': validate_config_bridge_time,
        'validate_hook_init': validate_hook_init_time,
        'create_report': create_performance_report
    }


# =============================================================================
# MOCK CONFIGURATION AND UTILITIES
# =============================================================================

@pytest.fixture
def comprehensive_mock_stack(
    mocker,
    minimal_kedro_context,
    test_catalog_with_figregistry,
    mock_kedro_session,
    mock_hook_manager,
    figregistry_config_bridge
):
    """Comprehensive mock stack for plugin testing per Section 6.6.2.3.
    
    Combines all essential Kedro component mocks with FigRegistry integration
    for comprehensive plugin testing without full framework overhead.
    
    Args:
        mocker: pytest-mock fixture
        Various component fixtures from kedro_fixtures.py
        
    Returns:
        Dict containing complete mock infrastructure
    """
    # Create additional mocks for comprehensive coverage
    mock_pipeline = mocker.Mock(spec=Pipeline if KEDRO_AVAILABLE else None)
    mock_pipeline.nodes = []
    mock_pipeline.describe = Mock(return_value="Mock Pipeline")
    
    mock_runner = mocker.Mock(spec=AbstractRunner if KEDRO_AVAILABLE else None)
    mock_runner.run = Mock(return_value={})
    
    # Mock FigRegistry core components
    mock_figregistry_init = mocker.patch('figregistry.init_config')
    mock_figregistry_get_style = mocker.patch('figregistry.get_style')
    mock_figregistry_save_figure = mocker.patch('figregistry.save_figure')
    
    mock_figregistry_get_style.return_value = {
        'color': '#1f77b4',
        'marker': 'o', 
        'linestyle': '-'
    }
    mock_figregistry_save_figure.return_value = "test_figure.png"
    
    return {
        # Kedro components
        'context': minimal_kedro_context,
        'catalog': test_catalog_with_figregistry,
        'session': mock_kedro_session,
        'hook_manager': mock_hook_manager,
        'pipeline': mock_pipeline,
        'runner': mock_runner,
        
        # FigRegistry integration
        'config_bridge': figregistry_config_bridge,
        'figregistry_init': mock_figregistry_init,
        'figregistry_get_style': mock_figregistry_get_style,
        'figregistry_save_figure': mock_figregistry_save_figure,
        
        # Utilities
        'mocker': mocker
    }


@pytest.fixture
def mock_figregistry_api(mocker):
    """Mock FigRegistry API calls for isolated testing.
    
    Provides comprehensive mocking of FigRegistry's public API to enable
    plugin testing without dependencies on core FigRegistry functionality.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Dict containing mocked FigRegistry API functions
    """
    # Mock core FigRegistry functions
    mock_init_config = mocker.patch('figregistry.init_config')
    mock_get_style = mocker.patch('figregistry.get_style')
    mock_save_figure = mocker.patch('figregistry.save_figure')
    mock_load_config = mocker.patch('figregistry.config.load_config')
    mock_validate_config = mocker.patch('figregistry.config.validate_config')
    
    # Configure mock return values
    mock_get_style.return_value = {
        'figure.figsize': [8, 6],
        'axes.labelsize': 12,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'axes.prop_cycle': "cycler('color', ['#1f77b4', '#ff7f0e'])"
    }
    
    mock_save_figure.return_value = Path("test_output.png")
    mock_load_config.return_value = generate_baseline_config()
    mock_validate_config.return_value = True
    
    return {
        'init_config': mock_init_config,
        'get_style': mock_get_style,
        'save_figure': mock_save_figure,
        'load_config': mock_load_config,
        'validate_config': mock_validate_config
    }


# =============================================================================
# HYPOTHESIS CONFIGURATION FOR PROPERTY-BASED TESTING
# =============================================================================

if HYPOTHESIS_AVAILABLE:
    # Configure Hypothesis for property-based testing per Section 6.6.2.6
    from hypothesis import settings, HealthCheck
    
    @pytest.fixture
    def hypothesis_config():
        """Configure Hypothesis for property-based testing.
        
        Returns:
            Hypothesis settings for configuration validation testing
        """
        return settings(
            max_examples=50,
            deadline=5000,  # 5 second deadline per test
            suppress_health_check=[HealthCheck.too_slow],
            verbosity=hypothesis.Verbosity.verbose if pytest.config.option.verbose > 1 else hypothesis.Verbosity.normal
        )
    
    @pytest.fixture
    def config_generation_strategies():
        """Provide Hypothesis strategies for configuration generation.
        
        Returns:
            Dict containing Hypothesis strategies for testing
        """
        return {
            'yaml_config': yaml_config_strategy,
            'kedro_config': kedro_config_strategy,
            'valid_color': valid_color_strategy,
            'valid_marker': valid_marker_strategy,
            'style_dict': style_dict_strategy
        }

else:
    @pytest.fixture
    def hypothesis_config():
        """Fallback when Hypothesis not available."""
        pytest.skip("Hypothesis not available for property-based testing")
    
    @pytest.fixture
    def config_generation_strategies():
        """Fallback when Hypothesis not available."""
        pytest.skip("Hypothesis strategies not available")


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def comprehensive_test_data():
    """Provide comprehensive test data for all testing scenarios.
    
    Combines all test data generators into a single fixture for convenient
    access to baseline configs, invalid scenarios, security tests, and
    performance datasets.
    
    Returns:
        Dict containing all test data categories
    """
    return {
        'baseline_configs': {
            'basic': generate_baseline_config(),
            'kedro_specific': generate_kedro_specific_config(),
            'minimal': {
                "figregistry_version": "0.3.0",
                "styles": {"test": {"color": "#000000"}},
                "outputs": {"base_path": "test"}
            }
        },
        'environment_configs': generate_environment_configs(),
        'invalid_configs': generate_invalid_config_scenarios(),
        'malformed_yaml': generate_malformed_yaml_strings(),
        'merge_scenarios': generate_merge_test_scenarios(),
        'security_configs': generate_security_test_configs(),
        'yaml_injection_vectors': generate_yaml_injection_vectors(),
        'cross_platform_configs': generate_cross_platform_config_variations(),
        'filesystem_edge_cases': generate_filesystem_edge_cases()
    }


@pytest.fixture
def config_file_factory(temp_directory):
    """Factory for creating temporary configuration files.
    
    Provides a callable factory for creating temporary YAML configuration
    files with automatic cleanup for configuration testing scenarios.
    
    Args:
        temp_directory: Base temporary directory fixture
        
    Returns:
        Callable factory for creating config files
    """
    created_files = []
    
    def create_config_file(config_dict: Dict[str, Any], filename: str = "test_config.yml") -> Path:
        """Create temporary configuration file.
        
        Args:
            config_dict: Configuration dictionary to write
            filename: Name for the configuration file
            
        Returns:
            Path to created configuration file
        """
        config_path = temp_directory / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        created_files.append(config_path)
        return config_path
    
    yield create_config_file
    
    # Cleanup created files (temp_directory cleanup handles parent directory)
    for file_path in created_files:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors


# =============================================================================
# VALIDATION AND ASSERTION UTILITIES
# =============================================================================

@pytest.fixture
def validation_utilities():
    """Provide validation utilities for comprehensive testing.
    
    Returns utilities for validating plugin behavior, configuration
    correctness, and integration compliance across test scenarios.
    
    Returns:
        Dict containing validation utility functions
    """
    def validate_figregistry_config(config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate FigRegistry configuration structure and content."""
        return validate_config_against_schema(config_dict)
    
    def validate_kedro_integration(component_name: str, component) -> bool:
        """Validate Kedro component integration compliance."""
        if component_name == "dataset":
            required_methods = ['_save', '_load', '_describe']
            return all(hasattr(component, method) for method in required_methods)
        elif component_name == "hook":
            lifecycle_methods = ['before_pipeline_run', 'after_pipeline_run']
            return any(hasattr(component, method) for method in lifecycle_methods)
        elif component_name == "config_bridge":
            bridge_methods = ['get_merged_config', 'init_config']
            return any(hasattr(component, method) for method in bridge_methods)
        return False
    
    def validate_plugin_performance(timings: Dict[str, float]) -> Dict[str, bool]:
        """Validate plugin performance against targets."""
        results = {}
        if 'dataset_save' in timings:
            results['dataset_save'] = timings['dataset_save'] < 200.0
        if 'config_bridge' in timings:
            results['config_bridge'] = timings['config_bridge'] < 50.0
        if 'hook_init' in timings:
            results['hook_init'] = timings['hook_init'] < 25.0
        return results
    
    def validate_security_protection(test_name: str, config_dict: Dict[str, Any]) -> bool:
        """Validate security protection against malicious configurations."""
        # Check for path traversal patterns
        if any("../" in str(value) or "..\\" in str(value) 
               for value in str(config_dict).split()):
            return False
        
        # Check for injection patterns
        injection_patterns = ["!!python", "!!map", "!!set", "__import__"]
        if any(pattern in str(config_dict) for pattern in injection_patterns):
            return False
        
        return True
    
    return {
        'validate_figregistry_config': validate_figregistry_config,
        'validate_kedro_integration': validate_kedro_integration,
        'validate_plugin_performance': validate_plugin_performance,
        'validate_security_protection': validate_security_protection
    }


# =============================================================================
# SESSION AND MODULE CLEANUP
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_session():
    """Session-level cleanup for comprehensive test isolation.
    
    Ensures clean session state before and after test execution
    with proper resource cleanup and state reset.
    """
    # Pre-test session setup
    import gc
    gc.collect()
    
    yield
    
    # Post-test session cleanup
    if MATPLOTLIB_AVAILABLE:
        plt.close('all')
        plt.rcdefaults()
    
    # Force garbage collection
    gc.collect()


# =============================================================================
# PYTEST PLUGIN REGISTRATION
# =============================================================================

def pytest_plugins():
    """Register pytest plugins for enhanced testing capabilities.
    
    Returns:
        List of pytest plugin names for automatic loading
    """
    plugins = ['pytest_mock']
    
    if BENCHMARK_AVAILABLE:
        plugins.append('pytest_benchmark')
    
    if HYPOTHESIS_AVAILABLE:
        plugins.append('hypothesis.extra.pytest')
    
    return plugins


# =============================================================================
# MODULE EXPORTS AND AVAILABILITY FLAGS
# =============================================================================

# Export availability flags for conditional test execution
__all__ = [
    # Availability flags
    'KEDRO_AVAILABLE',
    'KEDRO_PYTEST_AVAILABLE', 
    'HYPOTHESIS_AVAILABLE',
    'BENCHMARK_AVAILABLE',
    'MATPLOTLIB_AVAILABLE',
    
    # Core fixtures (re-exported from kedro_fixtures.py)
    'minimal_kedro_context',
    'mock_config_loader',
    'test_catalog_with_figregistry',
    'mock_kedro_session',
    'mock_hook_manager',
    'figregistry_config_bridge',
    'mock_figregistry_hooks',
    'mock_figure_dataset',
    'test_kedro_instance',
    'kedro_pytest_session',
    'minimal_project_scaffolding',
    'project_scaffolding_factory',
    'hook_performance_tracker',
    'mock_matplotlib_figure',
    'kedro_integration_validators',
    'complete_kedro_mock_stack',
    'kedro_testing_utilities',
    
    # Configuration fixtures
    'base_figregistry_config',
    'local_override_config',
    'minimal_figregistry_config',
    'invalid_config_scenarios',
    'security_test_configs',
    
    # Infrastructure fixtures
    'temp_directory',
    'temp_project_directory',
    'mock_matplotlib_rcparams',
    'sample_matplotlib_figure',
    
    # Performance fixtures
    'benchmark_config',
    'performance_config_datasets',
    'performance_validator',
    
    # Mock fixtures
    'comprehensive_mock_stack',
    'mock_figregistry_api',
    
    # Test data fixtures
    'comprehensive_test_data',
    'config_file_factory',
    
    # Validation fixtures
    'validation_utilities'
]

# Add conditional exports based on availability
if HYPOTHESIS_AVAILABLE:
    __all__.extend(['hypothesis_config', 'config_generation_strategies'])

# Module information for test reporting
TEST_MODULE_INFO = {
    'version': '1.0.0',
    'description': 'Main pytest configuration for figregistry-kedro testing',
    'kedro_available': KEDRO_AVAILABLE,
    'kedro_pytest_available': KEDRO_PYTEST_AVAILABLE,
    'hypothesis_available': HYPOTHESIS_AVAILABLE,
    'benchmark_available': BENCHMARK_AVAILABLE,
    'matplotlib_available': MATPLOTLIB_AVAILABLE,
    'fixture_count': len(__all__),
    'testing_frameworks': [
        'pytest >=8.0.0',
        'pytest-cov >=6.1.0', 
        'pytest-mock >=3.14.0'
    ] + (['kedro-pytest >=0.1.3'] if KEDRO_PYTEST_AVAILABLE else []) +
        (['hypothesis >=6.0.0'] if HYPOTHESIS_AVAILABLE else []) +
        (['pytest-benchmark'] if BENCHMARK_AVAILABLE else [])
}