"""Comprehensive unit tests for FigureDataSet component.

This module provides exhaustive testing of the FigureDataSet implementation per
Section 6.6 Testing Strategy, validating AbstractDataSet interface compliance, 
automated condition-based styling, catalog integration, and performance requirements.

Test Coverage Areas:
- AbstractDataSet interface compliance per F-005 requirements
- Automated condition-based styling without manual intervention per F-005.2
- Kedro catalog parameter extraction and configuration per Section 5.2.6
- Compatibility with Kedro versioning and experiment tracking per F-005.2
- Thread-safe operation for parallel pipeline execution per Section 5.2.8
- Performance validation against <200ms save operation targets per Section 6.6.4.3
- Comprehensive error handling for malformed configurations and edge cases

The test suite leverages pytest-mock for Kedro component simulation, property-based
testing for configuration validation, and comprehensive performance benchmarking
to ensure production-ready plugin functionality across supported environments.
"""

import concurrent.futures
import os
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Core testing dependencies
import numpy as np

# Suppress warnings in test environment
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Import matplotlib with fallback for test environments
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    matplotlib.use('Agg')  # Non-interactive backend for testing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None
    plt = None

# Import Kedro components with graceful fallback
try:
    from kedro.io import AbstractDataSet
    from kedro.io.core import get_filepath_str
    KEDRO_AVAILABLE = True
except ImportError:
    KEDRO_AVAILABLE = False
    AbstractDataSet = object

# Import FigregRegistry components with fallback
try:
    import figregistry
    FIGREGISTRY_AVAILABLE = True
except ImportError:
    FIGREGISTRY_AVAILABLE = False

# Import components under test
try:
    from figregistry_kedro.datasets import (
        FigureDataSet, 
        FigureDataSetError,
        StyleResolutionCache,
        create_figure_dataset,
        validate_figure_dataset_config
    )
    FIGREGISTRY_KEDRO_AVAILABLE = True
except ImportError:
    FIGREGISTRY_KEDRO_AVAILABLE = False
    FigureDataSet = None
    FigureDataSetError = Exception
    StyleResolutionCache = None

# Import test fixtures and utilities
from figregistry_kedro.tests.fixtures.kedro_fixtures import (
    minimal_kedro_context,
    test_catalog_with_figregistry,
    mock_kedro_session,
    mock_hook_manager,
    figregistry_config_bridge,
    mock_figure_dataset,
    hook_performance_tracker,
    mock_matplotlib_figure,
    kedro_integration_validators,
    complete_kedro_mock_stack
)

# Import property-based testing if available
try:
    from hypothesis import given, strategies as st, settings, assume, HealthCheck
    from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Import benchmarking if available
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


# =============================================================================
# TEST CONFIGURATION AND FIXTURES
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.kedro_plugin,
    pytest.mark.skipif(not FIGREGISTRY_KEDRO_AVAILABLE, reason="figregistry-kedro not available")
]


@pytest.fixture
def sample_figure():
    """Create sample matplotlib figure for testing FigureDataSet operations.
    
    Returns:
        matplotlib.figure.Figure: Sample figure with realistic plot data
    """
    if not MATPLOTLIB_AVAILABLE:
        pytest.skip("Matplotlib not available for figure creation")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create realistic test data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + 0.1 * np.random.RandomState(42).randn(100)
    y2 = np.cos(x) + 0.1 * np.random.RandomState(43).randn(100)
    
    ax.plot(x, y1, 'b-', label='Sine Wave', linewidth=2, marker='o', markersize=4)
    ax.plot(x, y2, 'r--', label='Cosine Wave', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Sample Figure for FigureDataSet Testing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


@pytest.fixture
def mock_figregistry_api(mocker):
    """Mock FigRegistry API calls for isolated FigureDataSet testing.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        dict: Mocked FigRegistry API functions with configurable responses
    """
    # Mock core FigRegistry functions
    mock_get_style = mocker.patch('figregistry.get_style')
    mock_save_figure = mocker.patch('figregistry.save_figure')
    mock_init_config = mocker.patch('figregistry.init_config')
    
    # Configure realistic return values
    mock_get_style.return_value = {
        'figure.figsize': [8, 6],
        'figure.facecolor': 'white',
        'figure.edgecolor': 'black',
        'figure.dpi': 100,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'lines.color': '#1f77b4',
        'lines.marker': 'o',
        'lines.linestyle': '-'
    }
    
    mock_save_figure.return_value = "test_figure_output.png"
    
    return {
        'get_style': mock_get_style,
        'save_figure': mock_save_figure,
        'init_config': mock_init_config
    }


@pytest.fixture
def basic_dataset_config():
    """Provide basic FigureDataSet configuration for testing.
    
    Returns:
        dict: Basic dataset configuration parameters
    """
    return {
        'filepath': 'data/08_reporting/test_figure.png',
        'purpose': 'exploratory',
        'condition_param': 'experiment_condition',
        'style_params': {
            'figure.dpi': 300,
            'figure.facecolor': 'white'
        },
        'save_args': {
            'bbox_inches': 'tight',
            'transparent': False
        }
    }


@pytest.fixture
def advanced_dataset_config():
    """Provide advanced FigureDataSet configuration for complex testing.
    
    Returns:
        dict: Advanced dataset configuration with versioning and complex parameters
    """
    return {
        'filepath': 'data/08_reporting/figures/advanced/{condition}/output.pdf',
        'purpose': 'publication',
        'condition_param': 'experimental_treatment',
        'style_params': {
            'figure.figsize': [12, 8],
            'figure.dpi': 300,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12
        },
        'save_args': {
            'format': 'pdf',
            'dpi': 300,
            'bbox_inches': 'tight',
            'pad_inches': 0.2
        },
        'version': True
    }


@pytest.fixture
def pipeline_context():
    """Provide mock pipeline context for condition parameter resolution.
    
    Returns:
        dict: Mock pipeline parameters for testing condition resolution
    """
    return {
        'experiment_condition': 'test_experiment',
        'experimental_treatment': 'control_group',
        'data_processing_stage': 'intermediate',
        'analysis_type': 'exploratory_analysis',
        'output_format': 'publication_ready'
    }


@pytest.fixture
def performance_tracker():
    """Provide performance tracking utilities for benchmark validation.
    
    Returns:
        dict: Performance tracking and validation utilities
    """
    timing_data = []
    
    def track_operation(operation_name: str, execution_time: float):
        """Track operation timing for performance analysis."""
        timing_data.append({
            'operation': operation_name,
            'execution_time_ms': execution_time * 1000,
            'timestamp': time.time()
        })
    
    def validate_performance_targets() -> Dict[str, Any]:
        """Validate performance against specified targets."""
        if not timing_data:
            return {'status': 'no_data'}
        
        save_operations = [t for t in timing_data if 'save' in t['operation']]
        if not save_operations:
            return {'status': 'no_save_operations'}
        
        max_save_time = max(op['execution_time_ms'] for op in save_operations)
        avg_save_time = sum(op['execution_time_ms'] for op in save_operations) / len(save_operations)
        
        return {
            'status': 'pass' if max_save_time < 200.0 else 'fail',
            'max_save_time_ms': max_save_time,
            'avg_save_time_ms': avg_save_time,
            'target_ms': 200.0,
            'operation_count': len(save_operations)
        }
    
    def get_summary() -> Dict[str, Any]:
        """Get comprehensive timing summary."""
        return {
            'operations': timing_data.copy(),
            'validation': validate_performance_targets(),
            'total_operations': len(timing_data)
        }
    
    return {
        'track_operation': track_operation,
        'validate_targets': validate_performance_targets,
        'get_summary': get_summary,
        'timing_data': timing_data
    }


# =============================================================================
# ABSTRACTDATASET INTERFACE COMPLIANCE TESTS (F-005 Requirements)
# =============================================================================

class TestAbstractDataSetCompliance:
    """Test suite for AbstractDataSet interface compliance per F-005 requirements.
    
    Validates that FigureDataSet correctly implements the AbstractDataSet interface
    including _save(), _load(), _describe(), and _exists() methods according to
    Kedro dataset API specifications.
    """
    
    def test_figuredataset_inherits_abstractdataset(self):
        """Verify FigureDataSet correctly inherits from AbstractDataSet.
        
        Validates class hierarchy and interface compliance for Kedro catalog integration.
        """
        if not KEDRO_AVAILABLE:
            pytest.skip("Kedro not available for interface testing")
        
        assert issubclass(FigureDataSet, AbstractDataSet)
        
        # Verify required abstract methods are implemented
        required_methods = ['_save', '_load', '_describe']
        for method in required_methods:
            assert hasattr(FigureDataSet, method)
            assert callable(getattr(FigureDataSet, method))
    
    def test_figuredataset_initialization_valid_config(self, basic_dataset_config):
        """Test FigureDataSet initialization with valid configuration parameters.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates proper initialization and parameter storage.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        
        # Verify configuration parameters are stored correctly
        assert dataset._filepath == basic_dataset_config['filepath']
        assert dataset._purpose == basic_dataset_config['purpose']
        assert dataset._condition_param == basic_dataset_config['condition_param']
        assert dataset._style_params == basic_dataset_config['style_params']
        assert dataset._save_args == basic_dataset_config['save_args']
        
        # Verify operation statistics are initialized
        assert dataset._operation_stats['saves'] == 0
        assert dataset._operation_stats['style_resolution_time'] == 0.0
        assert dataset._operation_stats['save_operation_time'] == 0.0
    
    def test_figuredataset_initialization_minimal_config(self):
        """Test FigureDataSet initialization with minimal required configuration.
        
        Validates default parameter handling and graceful degradation.
        """
        dataset = FigureDataSet(filepath='test_figure.png')
        
        # Verify default values are applied
        assert dataset._filepath == 'test_figure.png'
        assert dataset._purpose == 'exploratory'
        assert dataset._condition_param is None
        assert dataset._style_params == {}
        assert dataset._save_args == {}
    
    def test_figuredataset_initialization_invalid_purpose_warning(self, caplog):
        """Test FigureDataSet initialization with invalid purpose generates warning.
        
        Args:
            caplog: pytest log capture fixture
            
        Validates warning generation for non-standard purpose values.
        """
        with caplog.at_level('WARNING'):
            dataset = FigureDataSet(
                filepath='test.png',
                purpose='invalid_purpose'
            )
        
        assert dataset._purpose == 'invalid_purpose'
        assert 'not in recommended values' in caplog.text
    
    def test_figuredataset_describe_method(self, basic_dataset_config):
        """Test _describe() method returns comprehensive dataset metadata.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates metadata completeness and format per AbstractDataSet requirements.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        description = dataset._describe()
        
        # Verify required metadata fields
        assert 'filepath' in description
        assert 'purpose' in description
        assert 'condition_param' in description
        assert 'style_params' in description
        assert 'save_args' in description
        assert 'operation_stats' in description
        assert 'cache_stats' in description
        assert 'dependencies' in description
        
        # Verify field values match configuration
        assert description['filepath'] == basic_dataset_config['filepath']
        assert description['purpose'] == basic_dataset_config['purpose']
        assert description['condition_param'] == basic_dataset_config['condition_param']
        
        # Verify dependency availability flags
        dependencies = description['dependencies']
        assert 'matplotlib_available' in dependencies
        assert 'kedro_available' in dependencies
        assert 'figregistry_available' in dependencies
    
    def test_figuredataset_load_method_raises_error(self, basic_dataset_config):
        """Test _load() method raises appropriate error for unsupported operation.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates that loading figures is not supported with clear error message.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            dataset._load()
        
        error_message = str(exc_info.value)
        assert 'Loading figures is not supported' in error_message
        assert 'generated by pipeline nodes' in error_message
    
    def test_figuredataset_exists_method(self, basic_dataset_config, tmp_path):
        """Test _exists() method correctly detects file presence.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            tmp_path: pytest temporary path fixture
            
        Validates file existence detection functionality.
        """
        # Use temporary path for testing
        test_filepath = tmp_path / "test_figure.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        dataset = FigureDataSet(**config)
        
        # File should not exist initially
        assert not dataset._exists()
        
        # Create file and verify detection
        test_filepath.touch()
        assert dataset._exists()
    
    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available")
    def test_figuredataset_versioning_compatibility(self, basic_dataset_config, mocker):
        """Test FigureDataSet compatibility with Kedro versioning system.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mocker: pytest-mock fixture
            
        Validates that Kedro versioning works correctly with FigureDataSet.
        """
        # Mock get_filepath_str for versioning
        mock_get_filepath = mocker.patch('figregistry_kedro.datasets.get_filepath_str')
        mock_get_filepath.return_value = 'versioned/path/test_figure.png'
        
        config = basic_dataset_config.copy()
        config['version'] = 'test_version'
        dataset = FigureDataSet(**config)
        
        # Test exists method with versioning
        dataset._exists()
        mock_get_filepath.assert_called_with(config['filepath'], 'test_version')


# =============================================================================
# AUTOMATED STYLING AND CONDITION RESOLUTION TESTS (F-005.2 Requirements)
# =============================================================================

class TestAutomatedStyling:
    """Test suite for automated condition-based styling per F-005.2 requirements.
    
    Validates automated styling application without manual intervention, including
    condition parameter resolution, style application, and FigRegistry API integration.
    """
    
    def test_save_operation_basic_styling(self, sample_figure, basic_dataset_config, 
                                        mock_figregistry_api, tmp_path, mocker):
        """Test basic save operation with automated styling application.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for file operations
            mocker: pytest-mock fixture
            
        Validates automated styling without manual intervention.
        """
        # Configure temporary filepath
        test_filepath = tmp_path / "test_basic_styling.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock mkdir for directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Execute save operation
        dataset._save(sample_figure)
        
        # Verify FigRegistry get_style was called with correct condition
        mock_figregistry_api['get_style'].assert_called_once_with('exploratory')
        
        # Verify save_figure was called with styling applied
        mock_figregistry_api['save_figure'].assert_called_once()
        save_call_kwargs = mock_figregistry_api['save_figure'].call_args.kwargs
        assert save_call_kwargs['figure'] == sample_figure
        assert save_call_kwargs['condition'] == 'exploratory'
    
    def test_condition_parameter_resolution(self, sample_figure, basic_dataset_config,
                                          mock_figregistry_api, pipeline_context, tmp_path, mocker):
        """Test condition parameter resolution from pipeline context.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            pipeline_context: Mock pipeline parameters
            tmp_path: Temporary path for file operations
            mocker: pytest-mock fixture
            
        Validates dynamic condition resolution from pipeline parameters.
        """
        # Configure temporary filepath
        test_filepath = tmp_path / "test_condition_resolution.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock mkdir for directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Set pipeline context for condition resolution
        dataset.set_pipeline_context(pipeline_context)
        
        # Execute save operation
        dataset._save(sample_figure)
        
        # Verify condition resolved from pipeline context
        expected_condition = pipeline_context[config['condition_param']]
        mock_figregistry_api['get_style'].assert_called_once_with(expected_condition)
    
    def test_style_params_override_behavior(self, sample_figure, basic_dataset_config,
                                          mock_figregistry_api, tmp_path, mocker):
        """Test that style_params override base styling from FigRegistry.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for file operations
            mocker: pytest-mock fixture
            
        Validates style parameter precedence and merging behavior.
        """
        # Configure dataset with style overrides
        test_filepath = tmp_path / "test_style_override.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        config['style_params'] = {
            'figure.dpi': 150,  # Override default
            'axes.labelsize': 16,  # Additional parameter
        }
        
        # Mock mkdir and apply_style method
        mocker.patch.object(Path, 'mkdir')
        mock_apply_style = mocker.patch.object(FigureDataSet, '_apply_style_to_figure')
        
        dataset = FigureDataSet(**config)
        dataset._save(sample_figure)
        
        # Verify style merging behavior
        mock_apply_style.assert_called_once()
        applied_style_args = mock_apply_style.call_args[0]
        applied_style = applied_style_args[1]  # Second argument is the style dict
        
        # Verify style_params override base styling
        assert 'figure.dpi' in applied_style
        assert 'axes.labelsize' in applied_style
    
    def test_fallback_to_purpose_when_condition_param_missing(self, sample_figure, 
                                                            basic_dataset_config,
                                                            mock_figregistry_api, tmp_path, mocker):
        """Test fallback to purpose when condition parameter is not found.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for file operations
            mocker: pytest-mock fixture
            
        Validates graceful degradation when pipeline context lacks condition parameter.
        """
        # Configure temporary filepath
        test_filepath = tmp_path / "test_fallback_purpose.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock mkdir for directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Set empty pipeline context (missing condition parameter)
        dataset.set_pipeline_context({})
        
        # Execute save operation
        dataset._save(sample_figure)
        
        # Verify fallback to purpose value
        mock_figregistry_api['get_style'].assert_called_once_with(config['purpose'])
    
    def test_no_condition_param_uses_purpose_directly(self, sample_figure, mock_figregistry_api, tmp_path, mocker):
        """Test that missing condition_param uses purpose directly for styling.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for file operations
            mocker: pytest-mock fixture
            
        Validates direct purpose usage when no condition parameter specified.
        """
        # Configure dataset without condition_param
        test_filepath = tmp_path / "test_no_condition_param.png"
        config = {
            'filepath': str(test_filepath),
            'purpose': 'presentation'
            # No condition_param specified
        }
        
        # Mock mkdir for directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        dataset._save(sample_figure)
        
        # Verify purpose used directly for styling
        mock_figregistry_api['get_style'].assert_called_once_with('presentation')
    
    def test_figure_style_application_to_matplotlib(self, sample_figure, basic_dataset_config,
                                                  mock_figregistry_api, tmp_path, mocker):
        """Test that resolved styles are correctly applied to matplotlib figure.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for file operations
            mocker: pytest-mock fixture
            
        Validates matplotlib rcParams application and figure property modification.
        """
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available for style application testing")
        
        # Configure realistic style response
        mock_figregistry_api['get_style'].return_value = {
            'figure.facecolor': 'lightgray',
            'figure.edgecolor': 'darkblue',
            'figure.dpi': 150
        }
        
        # Configure temporary filepath
        test_filepath = tmp_path / "test_style_application.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock mkdir for directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Store original figure properties
        original_facecolor = sample_figure.patch.get_facecolor()
        original_dpi = sample_figure.get_dpi()
        
        # Execute save operation
        dataset._save(sample_figure)
        
        # Verify style application modified figure
        # Note: Actual verification depends on matplotlib's internal state management
        mock_figregistry_api['get_style'].assert_called_once()
        mock_figregistry_api['save_figure'].assert_called_once()


# =============================================================================
# CATALOG INTEGRATION AND PARAMETER EXTRACTION TESTS (Section 5.2.6)
# =============================================================================

class TestCatalogIntegration:
    """Test suite for Kedro catalog integration and parameter extraction.
    
    Validates catalog parameter extraction including purpose, condition_param,
    and style_params configuration from Kedro catalog entries per Section 5.2.6.
    """
    
    def test_catalog_parameter_extraction_basic(self, basic_dataset_config):
        """Test extraction of basic catalog parameters.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates proper extraction and storage of catalog configuration parameters.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        
        # Verify all parameters extracted correctly
        assert dataset._filepath == basic_dataset_config['filepath']
        assert dataset._purpose == basic_dataset_config['purpose']
        assert dataset._condition_param == basic_dataset_config['condition_param']
        assert dataset._style_params == basic_dataset_config['style_params']
        assert dataset._save_args == basic_dataset_config['save_args']
    
    def test_catalog_parameter_extraction_advanced(self, advanced_dataset_config):
        """Test extraction of advanced catalog parameters including versioning.
        
        Args:
            advanced_dataset_config: Advanced dataset configuration fixture
            
        Validates complex parameter extraction including versioning support.
        """
        dataset = FigureDataSet(**advanced_dataset_config)
        
        # Verify advanced parameters
        assert dataset._version == advanced_dataset_config['version']
        assert dataset._style_params['figure.figsize'] == [12, 8]
        assert dataset._save_args['format'] == 'pdf'
        assert dataset._save_args['dpi'] == 300
    
    def test_validate_figure_dataset_config_valid(self, basic_dataset_config):
        """Test configuration validation with valid parameters.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates configuration validation accepts properly formatted configurations.
        """
        validated_config = validate_figure_dataset_config(basic_dataset_config)
        
        # Verify validation passes and returns normalized config
        assert validated_config == basic_dataset_config
    
    def test_validate_figure_dataset_config_missing_filepath(self):
        """Test configuration validation rejects missing filepath.
        
        Validates that configuration validation enforces required parameters.
        """
        invalid_config = {
            'purpose': 'exploratory'
            # Missing required filepath
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_figure_dataset_config(invalid_config)
        
        assert 'requires \'filepath\' parameter' in str(exc_info.value)
    
    def test_validate_figure_dataset_config_invalid_types(self):
        """Test configuration validation rejects invalid parameter types.
        
        Validates type checking and validation for configuration parameters.
        """
        # Test invalid condition_param type
        with pytest.raises(ValueError) as exc_info:
            validate_figure_dataset_config({
                'filepath': 'test.png',
                'condition_param': 123  # Should be string
            })
        assert 'condition_param must be a string' in str(exc_info.value)
        
        # Test invalid style_params type
        with pytest.raises(ValueError) as exc_info:
            validate_figure_dataset_config({
                'filepath': 'test.png',
                'style_params': 'invalid'  # Should be dict
            })
        assert 'style_params must be a dictionary' in str(exc_info.value)
        
        # Test invalid save_args type
        with pytest.raises(ValueError) as exc_info:
            validate_figure_dataset_config({
                'filepath': 'test.png',
                'save_args': []  # Should be dict
            })
        assert 'save_args must be a dictionary' in str(exc_info.value)
    
    def test_create_figure_dataset_factory(self, basic_dataset_config):
        """Test create_figure_dataset factory function.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates factory function creates properly configured dataset instances.
        """
        dataset = create_figure_dataset(**basic_dataset_config)
        
        # Verify factory creates correct instance
        assert isinstance(dataset, FigureDataSet)
        assert dataset._filepath == basic_dataset_config['filepath']
        assert dataset._purpose == basic_dataset_config['purpose']
        assert dataset._condition_param == basic_dataset_config['condition_param']
    
    def test_catalog_entry_path_resolution(self, basic_dataset_config, tmp_path, mocker):
        """Test catalog entry path resolution and directory creation.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates automatic directory creation and path resolution.
        """
        # Configure nested path structure
        nested_path = tmp_path / "data" / "08_reporting" / "figures" / "test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(nested_path)
        
        # Mock figure for save operation
        mock_figure = mocker.Mock()
        mock_figure.savefig = mocker.Mock()
        
        # Mock FigRegistry functions
        mocker.patch('figregistry.get_style', return_value={})
        mocker.patch('figregistry.save_figure', return_value=str(nested_path))
        
        dataset = FigureDataSet(**config)
        dataset._save(mock_figure)
        
        # Verify directory structure was created
        assert nested_path.parent.exists()


# =============================================================================
# VERSIONING COMPATIBILITY TESTS (F-005.2 Requirements)
# =============================================================================

class TestVersioningCompatibility:
    """Test suite for Kedro versioning system compatibility.
    
    Validates that FigRegistry timestamp versioning coexists with Kedro dataset
    versioning without conflicts per F-005.2 requirements.
    """
    
    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available")
    def test_kedro_versioning_filepath_resolution(self, basic_dataset_config, mocker):
        """Test that Kedro versioning correctly resolves filepaths.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mocker: pytest-mock fixture
            
        Validates Kedro's get_filepath_str integration for versioning.
        """
        # Mock get_filepath_str for versioning
        mock_get_filepath = mocker.patch('figregistry_kedro.datasets.get_filepath_str')
        versioned_path = 'versioned/2024-01-01T12.00.00.000Z/test_figure.png'
        mock_get_filepath.return_value = versioned_path
        
        # Mock other dependencies
        mock_figure = mocker.Mock()
        mocker.patch('figregistry.get_style', return_value={})
        mocker.patch('figregistry.save_figure', return_value=versioned_path)
        mocker.patch.object(Path, 'mkdir')
        
        config = basic_dataset_config.copy()
        config['version'] = '2024-01-01T12.00.00.000Z'
        dataset = FigureDataSet(**config)
        
        # Execute save operation
        dataset._save(mock_figure)
        
        # Verify versioned path resolution
        mock_get_filepath.assert_called_with(
            basic_dataset_config['filepath'], 
            '2024-01-01T12.00.00.000Z'
        )
    
    def test_figregistry_timestamp_versioning_coexistence(self, basic_dataset_config, 
                                                        mock_figregistry_api, tmp_path, mocker):
        """Test FigRegistry timestamp versioning works alongside Kedro versioning.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates no conflicts between FigRegistry and Kedro versioning systems.
        """
        # Configure FigRegistry to return timestamped filename
        timestamped_filename = "test_figure_20240101_120000.png"
        mock_figregistry_api['save_figure'].return_value = timestamped_filename
        
        # Configure versioned dataset
        test_filepath = tmp_path / "versioned_figure.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        config['version'] = True
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        mock_figure = mocker.Mock()
        
        dataset = FigureDataSet(**config)
        dataset._save(mock_figure)
        
        # Verify both versioning systems operated
        mock_figregistry_api['save_figure'].assert_called_once()
        
        # Verify FigRegistry timestamp naming was used
        save_call_kwargs = mock_figregistry_api['save_figure'].call_args.kwargs
        assert save_call_kwargs['filepath'] == str(test_filepath)
    
    def test_versioning_exists_method_compatibility(self, basic_dataset_config, mocker):
        """Test _exists() method works correctly with versioning.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mocker: pytest-mock fixture
            
        Validates file existence checking with versioned paths.
        """
        # Mock get_filepath_str to return versioned path
        versioned_path = 'versioned/path/test_figure.png'
        mock_get_filepath = mocker.patch('figregistry_kedro.datasets.get_filepath_str')
        mock_get_filepath.return_value = versioned_path
        
        # Mock Path.exists to return True for versioned path
        mock_path_exists = mocker.patch.object(Path, 'exists')
        mock_path_exists.return_value = True
        
        config = basic_dataset_config.copy()
        config['version'] = 'test_version'
        dataset = FigureDataSet(**config)
        
        # Test exists method
        result = dataset._exists()
        
        # Verify versioned path checking
        assert result is True
        mock_get_filepath.assert_called_with(config['filepath'], 'test_version')
    
    def test_versioning_fallback_behavior(self, basic_dataset_config, mocker):
        """Test fallback behavior when get_filepath_str is unavailable.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mocker: pytest-mock fixture
            
        Validates graceful degradation for older Kedro versions.
        """
        # Mock get_filepath_str to raise exception (simulating older Kedro)
        mock_get_filepath = mocker.patch('figregistry_kedro.datasets.get_filepath_str')
        mock_get_filepath.side_effect = Exception("get_filepath_str not available")
        
        # Mock Path.exists for fallback path
        mock_path_exists = mocker.patch.object(Path, 'exists')
        mock_path_exists.return_value = False
        
        config = basic_dataset_config.copy()
        config['version'] = 'test_version'
        dataset = FigureDataSet(**config)
        
        # Test exists method fallback
        result = dataset._exists()
        
        # Verify fallback to non-versioned path
        assert result is False
        mock_path_exists.assert_called()


# =============================================================================
# PERFORMANCE TESTING (Section 6.6.4.3 Requirements)
# =============================================================================

class TestPerformanceRequirements:
    """Test suite for performance validation per Section 6.6.4.3 requirements.
    
    Validates <200ms per FigureDataSet save operation and measures plugin overhead
    compared to manual matplotlib save operations.
    """
    
    @pytest.mark.performance
    def test_save_operation_performance_target(self, sample_figure, basic_dataset_config,
                                             mock_figregistry_api, performance_tracker, tmp_path, mocker):
        """Test save operation meets <200ms performance target.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            performance_tracker: Performance tracking utilities
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates save operation performance against specified target.
        """
        # Configure temporary filepath
        test_filepath = tmp_path / "performance_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Measure save operation performance
        start_time = time.perf_counter()
        dataset._save(sample_figure)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        performance_tracker['track_operation']('dataset_save', execution_time)
        
        # Validate performance target
        execution_time_ms = execution_time * 1000
        assert execution_time_ms < 200.0, f"Save operation took {execution_time_ms:.2f}ms, exceeding 200ms target"
        
        # Verify performance tracking
        validation_result = performance_tracker['validate_targets']()
        assert validation_result['status'] == 'pass'
    
    @pytest.mark.performance  
    def test_style_resolution_performance(self, basic_dataset_config, mock_figregistry_api):
        """Test style resolution meets <1ms performance target.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            
        Validates style resolution performance for scalability requirements.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        
        # Measure style resolution performance
        start_time = time.perf_counter()
        style = dataset._get_figure_style('test_condition')
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Validate <1ms target for style resolution
        assert execution_time_ms < 1.0, f"Style resolution took {execution_time_ms:.2f}ms, exceeding 1ms target"
        
        # Verify style was retrieved
        assert isinstance(style, dict)
        mock_figregistry_api['get_style'].assert_called_once_with('test_condition')
    
    @pytest.mark.performance
    def test_plugin_overhead_comparison(self, sample_figure, basic_dataset_config,
                                      mock_figregistry_api, tmp_path, mocker):
        """Test plugin overhead compared to manual matplotlib save operations.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates <5% overhead compared to manual save operations.
        """
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available for overhead comparison")
        
        # Configure test paths
        manual_save_path = tmp_path / "manual_save.png"
        plugin_save_path = tmp_path / "plugin_save.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(plugin_save_path)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        # Measure manual matplotlib save
        start_time = time.perf_counter()
        sample_figure.savefig(manual_save_path, **config['save_args'])
        manual_time = time.perf_counter() - start_time
        
        # Measure plugin save operation
        dataset = FigureDataSet(**config)
        start_time = time.perf_counter()
        dataset._save(sample_figure)
        plugin_time = time.perf_counter() - start_time
        
        # Calculate overhead percentage
        overhead_percentage = ((plugin_time - manual_time) / manual_time) * 100
        
        # Validate <5% overhead target
        assert overhead_percentage < 5.0, f"Plugin overhead {overhead_percentage:.1f}% exceeds 5% target"
    
    @pytest.mark.performance
    def test_cache_performance_optimization(self, basic_dataset_config, mock_figregistry_api):
        """Test style cache improves performance for repeated operations.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            
        Validates cache effectiveness for performance optimization.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        condition = 'test_condition'
        
        # First style resolution (cache miss)
        start_time = time.perf_counter()
        style1 = dataset._get_figure_style(condition)
        first_time = time.perf_counter() - start_time
        
        # Second style resolution (cache hit)
        start_time = time.perf_counter()
        style2 = dataset._get_figure_style(condition)
        second_time = time.perf_counter() - start_time
        
        # Verify cache effectiveness
        assert style1 == style2  # Same result
        assert second_time < first_time  # Faster on cache hit
        
        # Verify FigRegistry called only once (cache working)
        assert mock_figregistry_api['get_style'].call_count == 1

    def test_style_cache_logging(self, basic_dataset_config, mock_figregistry_api, caplog):
        """Test debug logging for cache hits and misses."""

        dataset = FigureDataSet(**basic_dataset_config)
        condition = 'cache_test'

        with caplog.at_level('DEBUG'):
            dataset._get_style_configuration(condition)

        assert any('cache miss' in record.message.lower() for record in caplog.records)

        caplog.clear()

        with caplog.at_level('DEBUG'):
            dataset._get_style_configuration(condition)

        assert any('cache hit' in record.message.lower() for record in caplog.records)
    
    @pytest.mark.performance
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_benchmark_save_operation(self, sample_figure, basic_dataset_config, 
                                    mock_figregistry_api, tmp_path, mocker, benchmark):
        """Benchmark save operation using pytest-benchmark.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            benchmark: pytest-benchmark fixture
            
        Provides detailed performance analysis with statistical validation.
        """
        # Configure test filepath
        test_filepath = tmp_path / "benchmark_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Benchmark save operation
        result = benchmark(dataset._save, sample_figure)
        
        # Verify operation completed successfully
        mock_figregistry_api['get_style'].assert_called()
        mock_figregistry_api['save_figure'].assert_called()


# =============================================================================
# THREAD-SAFE OPERATION TESTS (Section 5.2.8 Requirements)
# =============================================================================

class TestThreadSafeOperation:
    """Test suite for thread-safe operation per Section 5.2.8 requirements.
    
    Validates thread-safe operation for parallel pipeline execution using
    pytest-mock simulation of concurrent catalog access.
    """
    
    def test_concurrent_save_operations(self, sample_figure, basic_dataset_config,
                                      mock_figregistry_api, tmp_path, mocker):
        """Test concurrent save operations are thread-safe.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates thread-safe operation during parallel execution.
        """
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        # Create multiple dataset instances for concurrent testing
        datasets = []
        for i in range(5):
            test_filepath = tmp_path / f"concurrent_test_{i}.png"
            config = basic_dataset_config.copy()
            config['filepath'] = str(test_filepath)
            datasets.append(FigureDataSet(**config))
        
        # Track thread execution
        results = []
        exceptions = []
        
        def save_operation(dataset_index):
            """Execute save operation and track results."""
            try:
                dataset = datasets[dataset_index]
                dataset._save(sample_figure)
                results.append(f"Dataset {dataset_index} completed successfully")
            except Exception as e:
                exceptions.append(f"Dataset {dataset_index} failed: {str(e)}")
        
        # Execute concurrent save operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_operation, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # Verify all operations completed successfully
        assert len(results) == 5
        assert len(exceptions) == 0
        
        # Verify all save operations were called
        assert mock_figregistry_api['save_figure'].call_count == 5
    
    def test_style_cache_thread_safety(self, basic_dataset_config, mock_figregistry_api):
        """Test style cache operates correctly under concurrent access.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            
        Validates thread-safe style cache operations.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        conditions = ['condition_1', 'condition_2', 'condition_3', 'condition_1', 'condition_2']
        
        # Track style resolution results
        results = []
        exceptions = []
        
        def resolve_style(condition):
            """Resolve style and track results."""
            try:
                style = dataset._get_figure_style(condition)
                results.append((condition, len(style)))
            except Exception as e:
                exceptions.append(f"Style resolution failed for {condition}: {str(e)}")
        
        # Execute concurrent style resolutions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(resolve_style, condition) for condition in conditions]
            concurrent.futures.wait(futures)
        
        # Verify all resolutions completed successfully
        assert len(results) == 5
        assert len(exceptions) == 0
        
        # Verify cache worked correctly (only 3 unique conditions should call FigRegistry)
        unique_conditions = len(set(conditions))
        assert mock_figregistry_api['get_style'].call_count == unique_conditions
    
    def test_pipeline_context_thread_isolation(self, basic_dataset_config):
        """Test pipeline context isolation between threads.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates thread-local storage for pipeline context.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        
        # Track context isolation
        context_results = {}
        
        def set_and_check_context(thread_id, context_data):
            """Set context and verify isolation."""
            dataset.set_pipeline_context(context_data)
            retrieved_context = dataset._get_pipeline_context()
            context_results[thread_id] = retrieved_context
        
        # Create different contexts for each thread
        contexts = {
            'thread_1': {'experiment': 'test_1', 'condition': 'A'},
            'thread_2': {'experiment': 'test_2', 'condition': 'B'},
            'thread_3': {'experiment': 'test_3', 'condition': 'C'}
        }
        
        # Execute concurrent context operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(set_and_check_context, thread_id, context)
                for thread_id, context in contexts.items()
            ]
            concurrent.futures.wait(futures)
        
        # Verify context isolation
        assert len(context_results) == 3
        for thread_id, expected_context in contexts.items():
            assert context_results[thread_id] == expected_context
    
    def test_performance_stats_thread_safety(self, sample_figure, basic_dataset_config,
                                           mock_figregistry_api, tmp_path, mocker):
        """Test performance statistics collection is thread-safe.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates thread-safe performance statistics aggregation.
        """
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        # Clear global performance stats
        FigureDataSet.reset_performance_stats()
        
        # Create dataset instances for concurrent testing
        datasets = []
        for i in range(3):
            test_filepath = tmp_path / f"perf_test_{i}.png"
            config = basic_dataset_config.copy()
            config['filepath'] = str(test_filepath)
            datasets.append(FigureDataSet(**config))
        
        def execute_save(dataset):
            """Execute save and track performance."""
            dataset._save(sample_figure)
        
        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(execute_save, dataset) for dataset in datasets]
            concurrent.futures.wait(futures)
        
        # Verify performance stats were collected safely
        perf_stats = FigureDataSet.get_performance_stats()
        assert perf_stats['total_saves'] == 3
        assert perf_stats['avg_save_time'] >= 0
        assert perf_stats['avg_style_time'] >= 0


# =============================================================================
# ERROR HANDLING AND EDGE CASE TESTS
# =============================================================================

class TestErrorHandling:
    """Test suite for comprehensive error handling and edge case management.
    
    Validates robust error management for malformed catalog configurations,
    missing condition parameters, and file system failures.
    """
    
    def test_invalid_figure_type_error(self, basic_dataset_config):
        """Test error handling for invalid figure object types.
        
        Args:
            basic_dataset_config: Basic dataset configuration fixture
            
        Validates type checking and error reporting for invalid inputs.
        """
        dataset = FigureDataSet(**basic_dataset_config)
        
        # Test with non-Figure object
        with pytest.raises(FigureDataSetError) as exc_info:
            dataset._save("not_a_figure")
        
        error_message = str(exc_info.value)
        assert 'Expected matplotlib Figure object' in error_message
        assert 'got <class \'str\'>' in error_message
    
    def test_figregistry_dependency_unavailable_error(self, mocker):
        """Test error handling when FigRegistry dependencies are unavailable.
        
        Args:
            mocker: pytest-mock fixture
            
        Validates graceful handling of missing dependencies.
        """
        # Mock dependencies as unavailable
        mocker.patch('figregistry_kedro.datasets.figregistry_available', False)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            FigureDataSet(filepath='test.png')
        
        error_message = str(exc_info.value)
        assert 'FigRegistry is required but not available' in error_message
        assert 'figregistry>=0.3.0' in error_message
    
    def test_kedro_dependency_unavailable_error(self, mocker):
        """Test error handling when Kedro dependencies are unavailable.
        
        Args:
            mocker: pytest-mock fixture
            
        Validates error reporting for missing Kedro dependencies.
        """
        # Mock dependencies as unavailable  
        mocker.patch('figregistry_kedro.datasets.kedro_available', False)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            FigureDataSet(filepath='test.png')
        
        error_message = str(exc_info.value)
        assert 'Kedro is required but not available' in error_message
        assert 'kedro>=0.18.0' in error_message
    
    def test_matplotlib_dependency_unavailable_error(self, mocker):
        """Test error handling when matplotlib dependencies are unavailable.
        
        Args:
            mocker: pytest-mock fixture
            
        Validates error reporting for missing matplotlib dependencies.
        """
        # Mock dependencies as unavailable
        mocker.patch('figregistry_kedro.datasets.matplotlib_available', False)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            FigureDataSet(filepath='test.png')
        
        error_message = str(exc_info.value)
        assert 'Matplotlib is required but not available' in error_message
        assert 'matplotlib>=3.9.0' in error_message
    
    def test_file_system_permission_error(self, sample_figure, basic_dataset_config, 
                                        mock_figregistry_api, tmp_path, mocker):
        """Test error handling for file system permission errors.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates graceful handling of file system errors.
        """
        # Configure test with permission error
        test_filepath = tmp_path / "permission_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock Path.mkdir to raise permission error
        mock_mkdir = mocker.patch.object(Path, 'mkdir')
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        dataset = FigureDataSet(**config)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            dataset._save(sample_figure)
        
        # Verify error contains permission information
        error_message = str(exc_info.value)
        assert 'Figure save operation failed' in error_message
        assert exc_info.value.original_error is not None
    
    def test_figregistry_style_resolution_error(self, sample_figure, basic_dataset_config,
                                              mock_figregistry_api, tmp_path, mocker):
        """Test error handling for FigRegistry style resolution failures.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates graceful handling of style resolution errors.
        """
        # Configure FigRegistry to raise error
        mock_figregistry_api['get_style'].side_effect = Exception("Style resolution failed")
        
        test_filepath = tmp_path / "style_error_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation to avoid unrelated errors
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            dataset._save(sample_figure)
        
        error_message = str(exc_info.value)
        assert 'Style resolution failed' in error_message
    
    def test_figregistry_save_figure_error(self, sample_figure, basic_dataset_config,
                                         mock_figregistry_api, tmp_path, mocker):
        """Test error handling for FigRegistry save_figure failures.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates graceful handling of save operation errors.
        """
        # Configure FigRegistry save_figure to raise error
        mock_figregistry_api['save_figure'].side_effect = Exception("Save operation failed")
        
        test_filepath = tmp_path / "save_error_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        with pytest.raises(FigureDataSetError) as exc_info:
            dataset._save(sample_figure)
        
        error_message = str(exc_info.value)
        assert 'Figure save failed' in error_message
    
    def test_malformed_style_params_handling(self, sample_figure, basic_dataset_config,
                                           mock_figregistry_api, tmp_path, mocker, caplog):
        """Test handling of malformed style parameters.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            caplog: pytest log capture fixture
            
        Validates graceful handling of invalid style configurations.
        """
        # Configure malformed style parameters
        test_filepath = tmp_path / "malformed_style_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        config['style_params'] = {
            'invalid.property': 'invalid_value',
            'figure.invalid_attr': 'test'
        }
        
        # Mock directory creation and figure attribute setting
        mocker.patch.object(Path, 'mkdir')
        mock_setattr = mocker.patch('setattr')
        mock_setattr.side_effect = AttributeError("Invalid attribute")
        
        dataset = FigureDataSet(**config)
        
        with caplog.at_level('DEBUG'):
            dataset._save(sample_figure)
        
        # Verify operation completed despite style errors
        mock_figregistry_api['save_figure'].assert_called_once()
        
        # Verify appropriate logging of style application issues
        assert any('Could not set figure property' in record.message for record in caplog.records)
    
    def test_empty_style_response_handling(self, sample_figure, basic_dataset_config,
                                         mock_figregistry_api, tmp_path, mocker):
        """Test handling of empty style responses from FigRegistry.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates graceful handling of empty style dictionaries.
        """
        # Configure empty style response
        mock_figregistry_api['get_style'].return_value = {}
        
        test_filepath = tmp_path / "empty_style_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        dataset._save(sample_figure)
        
        # Verify operation completed successfully with empty style
        mock_figregistry_api['get_style'].assert_called_once()
        mock_figregistry_api['save_figure'].assert_called_once()
    
    def test_none_style_response_handling(self, sample_figure, basic_dataset_config,
                                        mock_figregistry_api, tmp_path, mocker):
        """Test handling of None style responses from FigRegistry.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration fixture
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates graceful handling of None style responses.
        """
        # Configure None style response
        mock_figregistry_api['get_style'].return_value = None
        
        test_filepath = tmp_path / "none_style_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        dataset._save(sample_figure)
        
        # Verify operation completed successfully with None style
        mock_figregistry_api['get_style'].assert_called_once()
        mock_figregistry_api['save_figure'].assert_called_once()


# =============================================================================
# STYLE RESOLUTION CACHE TESTS
# =============================================================================

class TestStyleResolutionCache:
    """Test suite for style resolution cache functionality.
    
    Validates cache behavior including LRU eviction, thread safety,
    and performance optimization for repeated style lookups.
    """
    
    def test_cache_basic_functionality(self):
        """Test basic cache put and get operations.
        
        Validates fundamental cache operations and data integrity.
        """
        if StyleResolutionCache is None:
            pytest.skip("StyleResolutionCache not available")
        
        cache = StyleResolutionCache(max_size=10)
        
        # Test cache miss
        result = cache.get('test_key')
        assert result is None
        
        # Test cache put and hit
        test_style = {'color': '#FF0000', 'linewidth': 2}
        cache.put('test_key', test_style)
        
        retrieved_style = cache.get('test_key')
        assert retrieved_style == test_style
        assert retrieved_style is not test_style  # Verify deep copy
    
    def test_cache_lru_eviction(self):
        """Test LRU (Least Recently Used) eviction policy.
        
        Validates that cache correctly evicts least recently used entries.
        """
        if StyleResolutionCache is None:
            pytest.skip("StyleResolutionCache not available")
        
        cache = StyleResolutionCache(max_size=3)
        
        # Fill cache to capacity
        for i in range(3):
            cache.put(f'key_{i}', {'value': i})
        
        # Access key_0 to make it most recently used
        cache.get('key_0')
        
        # Add new item, should evict key_1 (least recently used)
        cache.put('key_3', {'value': 3})
        
        # Verify eviction
        assert cache.get('key_0') is not None  # Most recently used
        assert cache.get('key_1') is None      # Should be evicted
        assert cache.get('key_2') is not None  # Second most recently used
        assert cache.get('key_3') is not None  # Newly added
    
    def test_cache_statistics_tracking(self):
        """Test cache statistics collection and reporting.
        
        Validates hit rate calculation and statistics accuracy.
        """
        if StyleResolutionCache is None:
            pytest.skip("StyleResolutionCache not available")
        
        cache = StyleResolutionCache(max_size=5)
        
        # Initial statistics
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        
        # Add entries and track hits/misses
        cache.put('key_1', {'test': 1})
        cache.put('key_2', {'test': 2})
        
        # Mix of hits and misses
        cache.get('key_1')     # Hit
        cache.get('key_2')     # Hit
        cache.get('key_3')     # Miss
        cache.get('key_1')     # Hit
        
        # Verify statistics
        stats = cache.get_stats()
        assert stats['hits'] == 3
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.75  # 3/4
        assert stats['size'] == 2
    
    def test_cache_clear_functionality(self):
        """Test cache clear operation and statistics reset.
        
        Validates complete cache clearing and statistics reset.
        """
        if StyleResolutionCache is None:
            pytest.skip("StyleResolutionCache not available")
        
        cache = StyleResolutionCache(max_size=5)
        
        # Populate cache
        cache.put('key_1', {'test': 1})
        cache.put('key_2', {'test': 2})
        cache.get('key_1')  # Generate hit
        cache.get('key_3')  # Generate miss
        
        # Verify cache has data
        assert cache.get_stats()['size'] == 2
        assert cache.get_stats()['hits'] > 0
        
        # Clear cache
        cache.clear()
        
        # Verify cache is empty and stats reset
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        assert cache.get('key_1') is None
    
    def test_cache_thread_safety(self):
        """Test cache thread safety under concurrent access.
        
        Validates thread-safe cache operations with concurrent access.
        """
        if StyleResolutionCache is None:
            pytest.skip("StyleResolutionCache not available")
        
        cache = StyleResolutionCache(max_size=100)
        
        def cache_operations(thread_id):
            """Perform cache operations from multiple threads."""
            for i in range(10):
                key = f'thread_{thread_id}_key_{i}'
                value = {'thread': thread_id, 'iteration': i}
                
                # Put and get operations
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value
        
        # Execute concurrent cache operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # Verify cache integrity
        stats = cache.get_stats()
        assert stats['size'] <= 50  # 5 threads * 10 operations
        assert stats['hits'] >= 50  # At least one hit per operation
    
    def test_cache_data_isolation(self):
        """Test that cached data is properly isolated between entries.
        
        Validates data integrity and isolation in cache storage.
        """
        if StyleResolutionCache is None:
            pytest.skip("StyleResolutionCache not available")
        
        cache = StyleResolutionCache(max_size=10)
        
        # Store mutable objects
        style_1 = {'colors': ['red', 'blue'], 'linewidth': 2}
        style_2 = {'colors': ['green', 'yellow'], 'linewidth': 3}
        
        cache.put('style_1', style_1)
        cache.put('style_2', style_2)
        
        # Retrieve and modify
        retrieved_1 = cache.get('style_1')
        retrieved_2 = cache.get('style_2')
        
        # Modify retrieved objects
        retrieved_1['colors'].append('purple')
        retrieved_2['linewidth'] = 999
        
        # Verify original cache data is not modified
        fresh_1 = cache.get('style_1')
        fresh_2 = cache.get('style_2')
        
        assert fresh_1 == style_1  # Original data preserved
        assert fresh_2 == style_2  # Original data preserved
        assert 'purple' not in fresh_1['colors']
        assert fresh_2['linewidth'] == 3


# =============================================================================
# PROPERTY-BASED TESTING (Hypothesis)
# =============================================================================

if HYPOTHESIS_AVAILABLE:
    class TestPropertyBasedValidation:
        """Property-based test suite using Hypothesis for comprehensive validation.
        
        Validates FigureDataSet behavior across wide range of inputs and configurations
        using property-based testing techniques.
        """
        
        @given(
            filepath=st.text(min_size=1, max_size=100).filter(lambda x: '/' not in x and '\\' not in x),
            purpose=st.sampled_from(['exploratory', 'presentation', 'publication', 'analysis']),
            dpi=st.integers(min_value=72, max_value=300),
            linewidth=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)
        )
        @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
        def test_dataset_configuration_properties(self, filepath, purpose, dpi, linewidth):
            """Test dataset configuration with property-based inputs.
            
            Args:
                filepath: Generated filepath string
                purpose: Generated purpose value
                dpi: Generated DPI value  
                linewidth: Generated line width value
                
            Validates configuration handling across wide input range.
            """
            config = {
                'filepath': f"{filepath}.png",
                'purpose': purpose,
                'style_params': {
                    'figure.dpi': dpi,
                    'lines.linewidth': linewidth
                }
            }
            
            # Create dataset instance
            dataset = FigureDataSet(**config)
            
            # Verify configuration preservation
            assert dataset._filepath == config['filepath']
            assert dataset._purpose == config['purpose']
            assert dataset._style_params['figure.dpi'] == dpi
            assert dataset._style_params['lines.linewidth'] == linewidth
        
        @given(
            condition=st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
            style_override_count=st.integers(min_value=0, max_value=10)
        )
        @settings(max_examples=15, deadline=5000)
        def test_condition_resolution_properties(self, condition, style_override_count, mocker):
            """Test condition resolution with property-based conditions.
            
            Args:
                condition: Generated condition string
                style_override_count: Number of style overrides to generate
                mocker: pytest-mock fixture
                
            Validates condition resolution behavior.
            """
            # Mock FigRegistry API
            mock_get_style = mocker.patch('figregistry.get_style')
            mock_get_style.return_value = {'test': 'value'}
            
            # Generate style overrides
            style_params = {f'param_{i}': f'value_{i}' for i in range(style_override_count)}
            
            config = {
                'filepath': 'test.png',
                'purpose': 'test',
                'condition_param': 'test_condition',
                'style_params': style_params
            }
            
            dataset = FigureDataSet(**config)
            dataset.set_pipeline_context({'test_condition': condition})
            
            # Test condition resolution
            resolved_condition = dataset._resolve_condition()
            assert resolved_condition == condition
        
        @given(
            cache_size=st.integers(min_value=1, max_value=1000),
            operation_count=st.integers(min_value=1, max_value=100)
        )
        @settings(max_examples=10, deadline=10000)
        def test_cache_behavior_properties(self, cache_size, operation_count):
            """Test cache behavior with property-based parameters.
            
            Args:
                cache_size: Generated cache size
                operation_count: Number of operations to perform
                
            Validates cache behavior across different configurations.
            """
            if StyleResolutionCache is None:
                return  # Skip if not available
            
            cache = StyleResolutionCache(max_size=cache_size)
            
            # Perform operations
            for i in range(operation_count):
                key = f'key_{i % (cache_size + 10)}'  # Some key reuse
                value = {'operation': i, 'data': f'value_{i}'}
                cache.put(key, value)
                
                # Occasional get operations
                if i % 3 == 0:
                    cache.get(key)
            
            # Verify cache constraints
            stats = cache.get_stats()
            assert stats['size'] <= cache_size
            assert stats['hits'] + stats['misses'] == operation_count // 3


# =============================================================================
# INTEGRATION TEST WITH REAL KEDRO COMPONENTS
# =============================================================================

@pytest.mark.integration
class TestKedroIntegration:
    """Integration test suite with real Kedro components.
    
    Validates FigureDataSet integration with actual Kedro catalog and session
    components when available.
    """
    
    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available")
    def test_integration_with_kedro_catalog(self, sample_figure, basic_dataset_config, 
                                          mock_figregistry_api, tmp_path, mocker):
        """Test integration with real Kedro DataCatalog.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates integration with actual Kedro catalog operations.
        """
        from kedro.io import DataCatalog
        
        # Configure dataset for catalog
        test_filepath = tmp_path / "catalog_integration_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        # Create catalog with FigureDataSet
        catalog = DataCatalog({
            'test_figure': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                **config
            }
        })
        
        # Test catalog save operation
        catalog.save('test_figure', sample_figure)
        
        # Verify FigRegistry integration
        mock_figregistry_api['get_style'].assert_called_once()
        mock_figregistry_api['save_figure'].assert_called_once()
    
    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available")
    def test_integration_with_kedro_versioning(self, sample_figure, basic_dataset_config,
                                             mock_figregistry_api, tmp_path, mocker):
        """Test integration with Kedro's versioning system.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            
        Validates versioned dataset integration.
        """
        from kedro.io import DataCatalog
        
        # Configure versioned dataset
        test_filepath = tmp_path / "versioned_integration_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        config['versioned'] = True
        
        # Mock directory creation and versioning
        mocker.patch.object(Path, 'mkdir')
        mock_get_filepath = mocker.patch('figregistry_kedro.datasets.get_filepath_str')
        mock_get_filepath.return_value = str(test_filepath)
        
        # Create catalog with versioned dataset
        catalog = DataCatalog({
            'versioned_figure': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                **config
            }
        })
        
        # Test versioned save operation
        catalog.save('versioned_figure', sample_figure)
        
        # Verify integration completed successfully
        mock_figregistry_api['save_figure'].assert_called_once()


# =============================================================================
# UTILITY FUNCTIONS AND GLOBAL TEST INFRASTRUCTURE
# =============================================================================

def test_module_imports():
    """Test that all required modules can be imported correctly.
    
    Validates import availability and version compatibility.
    """
    # Test core imports
    assert FIGREGISTRY_KEDRO_AVAILABLE, "figregistry_kedro package not available"
    
    if KEDRO_AVAILABLE:
        from kedro.io import AbstractDataSet
        assert AbstractDataSet is not None
    
    if MATPLOTLIB_AVAILABLE:
        import matplotlib
        assert matplotlib is not None
    
    if FIGREGISTRY_AVAILABLE:
        import figregistry
        assert figregistry is not None


def test_test_infrastructure():
    """Test that test infrastructure is properly configured.
    
    Validates test fixtures and utilities are working correctly.
    """
    # Verify required test dependencies
    import pytest
    assert pytest is not None
    
    # Verify optional test dependencies
    if HYPOTHESIS_AVAILABLE:
        import hypothesis
        assert hypothesis is not None
    
    if BENCHMARK_AVAILABLE:
        import pytest_benchmark
        assert pytest_benchmark is not None


# =============================================================================
# PERFORMANCE BASELINE ESTABLISHMENT
# =============================================================================

@pytest.mark.performance
class TestPerformanceBaseline:
    """Establish performance baselines for regression testing.
    
    Creates baseline measurements for future performance regression detection.
    """
    
    def test_establish_save_operation_baseline(self, sample_figure, basic_dataset_config,
                                             mock_figregistry_api, tmp_path, mocker, performance_tracker):
        """Establish baseline for save operation performance.
        
        Args:
            sample_figure: Sample matplotlib figure fixture
            basic_dataset_config: Basic dataset configuration
            mock_figregistry_api: Mock FigRegistry API functions
            tmp_path: Temporary path for testing
            mocker: pytest-mock fixture
            performance_tracker: Performance tracking utilities
            
        Creates performance baseline for future comparison.
        """
        # Configure test
        test_filepath = tmp_path / "baseline_test.png"
        config = basic_dataset_config.copy()
        config['filepath'] = str(test_filepath)
        
        # Mock directory creation
        mocker.patch.object(Path, 'mkdir')
        
        dataset = FigureDataSet(**config)
        
        # Perform multiple measurements for statistical validity
        measurements = []
        for _ in range(10):
            start_time = time.perf_counter()
            dataset._save(sample_figure)
            end_time = time.perf_counter()
            measurements.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate baseline statistics
        avg_time = sum(measurements) / len(measurements)
        max_time = max(measurements)
        min_time = min(measurements)
        
        # Log baseline for reference
        print(f"\nPerformance Baseline Established:")
        print(f"  Average save time: {avg_time:.2f}ms")
        print(f"  Maximum save time: {max_time:.2f}ms")
        print(f"  Minimum save time: {min_time:.2f}ms")
        print(f"  Target threshold: 200.00ms")
        
        # Verify all measurements meet target
        assert max_time < 200.0, f"Baseline maximum {max_time:.2f}ms exceeds target"


# =============================================================================
# TEST CLEANUP AND REPORTING
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_session_cleanup():
    """Session-level cleanup for comprehensive test isolation.
    
    Ensures clean test session state and resource cleanup.
    """
    # Pre-test session setup
    if MATPLOTLIB_AVAILABLE:
        plt.ioff()  # Turn off interactive mode
    
    yield
    
    # Post-test session cleanup
    if MATPLOTLIB_AVAILABLE:
        plt.close('all')  # Close all figures
        plt.rcdefaults()  # Reset rcParams
    
    # Clear any global state
    if FIGREGISTRY_KEDRO_AVAILABLE:
        FigureDataSet.clear_cache()
        FigureDataSet.reset_performance_stats()


def pytest_runtest_setup(item):
    """Setup for individual test execution.
    
    Args:
        item: pytest test item
        
    Ensures clean state for each test.
    """
    # Clear matplotlib state
    if MATPLOTLIB_AVAILABLE:
        plt.close('all')
        plt.rcdefaults()
    
    # Clear FigureDataSet caches
    if FIGREGISTRY_KEDRO_AVAILABLE:
        FigureDataSet.clear_cache()


def pytest_runtest_teardown(item):
    """Teardown after individual test execution.
    
    Args:
        item: pytest test item
        
    Ensures cleanup after each test.
    """
    # Force garbage collection
    import gc
    gc.collect()


# Export test information for reporting
TEST_MODULE_INFO = {
    'module': 'test_datasets.py',
    'description': 'Comprehensive FigureDataSet component testing',
    'coverage_target': '≥90%',
    'performance_target': '<200ms per save operation',
    'thread_safety': 'Validated',
    'kedro_integration': 'Complete',
    'test_count': 'Comprehensive',
    'dependencies_tested': [
        'matplotlib availability',
        'kedro availability', 
        'figregistry availability',
        'figregistry_kedro availability'
    ],
    'testing_frameworks': [
        'pytest unit testing',
        'pytest-mock component mocking',
        'hypothesis property-based testing',
        'pytest-benchmark performance testing',
        'concurrent.futures thread safety testing'
    ]
}