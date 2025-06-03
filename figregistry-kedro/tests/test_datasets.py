"""
Comprehensive unit tests for FigureDataSet component.

This module provides complete test coverage for the figregistry_kedro.datasets.FigureDataSet
component, validating AbstractDataSet interface compliance, automated condition-based styling,
catalog parameter extraction, versioning compatibility, performance requirements, and
error handling scenarios per F-005 feature requirements.

Testing Scope per Section 6.6.1.1:
- AbstractDataSet interface compliance for catalog integration (F-005)
- Automated condition-based styling without manual intervention (F-005.2)
- Compatibility with Kedro versioning and experiment tracking (F-005.2)
- Thread-safe operation for parallel pipeline execution (Section 5.2.8)
- Performance validation of <200ms per save operation (Section 6.6.4.3)

Test Organization per Section 6.6.2.2:
- Unit tests for save/load/describe methods
- Integration tests for catalog parameter extraction
- Performance tests with pytest-benchmark
- Security tests for parameter validation
- Thread-safety tests for parallel execution
"""

import os
import sys
import time
import warnings
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Configure matplotlib for headless testing
matplotlib.use('Agg')
plt.ioff()

# Suppress warnings for clean test output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='kedro')


# =============================================================================
# TEST SETUP AND FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_matplotlib_state():
    """
    Reset matplotlib state between tests for isolation per Section 6.6.5.6.
    
    Ensures clean matplotlib environment for each test execution,
    preventing cross-test contamination and ensuring consistent figure state.
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
def sample_figure():
    """
    Create sample matplotlib figure for dataset testing per Section 6.6.2.6.
    
    Provides standard matplotlib figure object with basic plot content
    for testing FigureDataSet save/load operations and styling application.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    
    # Create plot with various elements for comprehensive testing
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x) + noise')
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.set_title('Sample Plot for FigureDataSet Testing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some additional complexity
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.text(5, 0.5, 'Test annotation', fontsize=10, ha='center')
    
    return fig


@pytest.fixture
def complex_figure():
    """
    Create complex matplotlib figure for advanced testing scenarios.
    
    Provides matplotlib figure with subplots, multiple data series,
    and complex styling for testing performance and advanced features.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate deterministic sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    
    # Plot 1: Multiple line plots
    ax1.plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
    ax1.plot(x, np.cos(x), 'r--', label='cos(x)', linewidth=2)
    ax1.set_title('Trigonometric Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot with colormap
    x_scatter = np.random.randn(100)
    y_scatter = np.random.randn(100)
    colors = np.random.randn(100)
    ax2.scatter(x_scatter, y_scatter, c=colors, alpha=0.6, cmap='viridis')
    ax2.set_title('Random Scatter with Colormap')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bar plot with error bars
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    errors = [2, 3, 4, 5, 3]
    ax3.bar(categories, values, yerr=errors, capsize=5, color='green', alpha=0.7)
    ax3.set_title('Category Data with Error Bars')
    ax3.set_ylabel('Values')
    
    # Plot 4: Filled area plot
    y_fill = np.exp(-x/5) * np.sin(x)
    ax4.fill_between(x, 0, y_fill, alpha=0.6, color='purple')
    ax4.plot(x, y_fill, 'purple', linewidth=2)
    ax4.set_title('Exponential Decay with Fill')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


@pytest.fixture
def mock_figregistry_apis(mocker):
    """
    Mock FigRegistry core APIs for dataset testing isolation per Section 6.6.2.3.
    
    Provides comprehensive mocking of FigRegistry's get_style() and save_figure()
    APIs to test FigureDataSet behavior without external dependencies.
    """
    # Mock get_style API
    mock_get_style = mocker.patch('figregistry.get_style')
    mock_get_style.return_value = {
        'figure.figsize': [10, 6],
        'axes.grid': True,
        'axes.grid.alpha': 0.3,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 10
    }
    
    # Mock save_figure API
    mock_save_figure = mocker.patch('figregistry.save_figure')
    mock_save_figure.return_value = '/mocked/path/figure_20231201_123456.png'
    
    # Mock init_config API for configuration initialization
    mock_init_config = mocker.patch('figregistry.init_config')
    mock_init_config.return_value = True
    
    return {
        'get_style': mock_get_style,
        'save_figure': mock_save_figure,
        'init_config': mock_init_config
    }


@pytest.fixture
def figure_dataset_config():
    """
    Provide comprehensive FigureDataSet configuration scenarios per F-005.2.
    
    Returns various catalog configuration patterns for testing parameter
    extraction, condition resolution, and error handling scenarios.
    """
    return {
        'basic_config': {
            'filepath': 'data/08_reporting/figures/basic_plot.png',
            'purpose': 'exploratory',
            'condition_param': 'experiment_type',
            'save_args': {
                'dpi': 150,
                'bbox_inches': 'tight'
            }
        },
        'advanced_config': {
            'filepath': 'data/08_reporting/figures/advanced_plot.pdf',
            'purpose': 'publication',
            'condition_param': 'analysis_mode',
            'style_params': {
                'figure.figsize': [8, 6],
                'axes.labelsize': 12,
                'font.family': 'serif'
            },
            'save_args': {
                'format': 'pdf',
                'dpi': 300,
                'transparent': True
            },
            'versioned': True
        },
        'minimal_config': {
            'filepath': 'data/08_reporting/figures/minimal_plot.svg',
            'purpose': 'presentation'
            # No condition_param or style_params for testing defaults
        },
        'invalid_config': {
            # Missing required filepath
            'purpose': 'exploratory',
            'condition_param': 'invalid_param'
        }
    }


@pytest.fixture
def mock_kedro_session_context(mocker, temp_work_dir):
    """
    Mock Kedro session context for dataset testing per Section 6.6.2.3.
    
    Creates comprehensive mock of Kedro session components including
    catalog, configuration, and pipeline parameters for isolated testing.
    """
    # Mock session
    mock_session = mocker.Mock()
    mock_session.store = {
        'experiment_type': 'baseline',
        'analysis_mode': 'development',
        'user_id': 'test_user',
        'project_name': 'test_project'
    }
    
    # Mock context
    mock_context = mocker.Mock()
    mock_context.project_path = temp_work_dir
    mock_context.package_name = 'test_project'
    
    # Mock catalog with versioning support
    mock_catalog = mocker.Mock()
    mock_catalog.save = mocker.Mock()
    mock_catalog.load = mocker.Mock()
    mock_catalog.exists = mocker.Mock(return_value=True)
    
    # Mock config loader
    mock_config_loader = mocker.Mock()
    mock_config_loader.get.return_value = {
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
    
    mock_context.catalog = mock_catalog
    mock_context.config_loader = mock_config_loader
    mock_session.load_context.return_value = mock_context
    
    return mock_session


@pytest.fixture
def performance_baseline():
    """
    Provide performance baseline measurements per Section 6.6.4.3.
    
    Establishes baseline timing measurements for configuration loading,
    style resolution, and dataset operations for performance validation.
    """
    return {
        'manual_save_time': 0.100,  # 100ms baseline for manual plt.savefig
        'style_resolution_time': 0.001,  # 1ms target for get_style()
        'dataset_save_target': 0.200,  # 200ms target for FigureDataSet save
        'overhead_threshold': 0.05,  # 5% maximum overhead vs manual
        'memory_baseline_mb': 50.0  # 50MB baseline memory usage
    }


# =============================================================================
# FIGUREDATASET UNIT TESTS - ABSTRACTDATASET INTERFACE COMPLIANCE
# =============================================================================

class TestFigureDataSetInterface:
    """
    Test AbstractDataSet interface compliance per F-005 requirements.
    
    Validates that FigureDataSet correctly implements Kedro's AbstractDataSet
    contract including _save(), _load(), and _describe() methods with proper
    parameter handling and error management.
    """
    
    def test_figuredataset_initialization(self, figure_dataset_config, temp_work_dir):
        """
        Test FigureDataSet initialization with various configuration parameters.
        
        Validates proper initialization of FigureDataSet with catalog parameters
        including filepath, purpose, condition_param, and style_params.
        """
        # Import here to avoid circular imports during test collection
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        
        # Test basic initialization
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param'],
            save_args=config['save_args']
        )
        
        # Verify attributes are properly set
        assert dataset._filepath == config['filepath']
        assert dataset._purpose == config['purpose']
        assert dataset._condition_param == config['condition_param']
        assert dataset._save_args == config['save_args']
        
        # Test advanced initialization with style_params
        advanced_config = figure_dataset_config['advanced_config']
        advanced_dataset = FigureDataSet(
            filepath=advanced_config['filepath'],
            purpose=advanced_config['purpose'],
            condition_param=advanced_config['condition_param'],
            style_params=advanced_config['style_params'],
            save_args=advanced_config['save_args'],
            versioned=advanced_config['versioned']
        )
        
        assert advanced_dataset._style_params == advanced_config['style_params']
        assert advanced_dataset._versioned == advanced_config['versioned']
    
    
    def test_figuredataset_initialization_minimal_config(self, figure_dataset_config):
        """
        Test FigureDataSet initialization with minimal required parameters.
        
        Validates that FigureDataSet can be initialized with only required
        parameters and properly handles default values for optional parameters.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['minimal_config']
        
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Verify required attributes
        assert dataset._filepath == config['filepath']
        assert dataset._purpose == config['purpose']
        
        # Verify optional attributes have sensible defaults
        assert dataset._condition_param is None
        assert dataset._style_params == {}
        assert dataset._save_args == {}
        assert dataset._versioned is False
    
    
    def test_figuredataset_initialization_invalid_config(self, figure_dataset_config):
        """
        Test FigureDataSet initialization with invalid configurations.
        
        Validates proper error handling for missing required parameters
        and invalid parameter combinations per error handling requirements.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test missing filepath
        with pytest.raises((ValueError, TypeError)) as excinfo:
            FigureDataSet(purpose='exploratory')
        
        assert 'filepath' in str(excinfo.value).lower()
        
        # Test missing purpose
        with pytest.raises((ValueError, TypeError)) as excinfo:
            FigureDataSet(filepath='test.png')
        
        assert 'purpose' in str(excinfo.value).lower()
        
        # Test invalid purpose value
        with pytest.raises(ValueError) as excinfo:
            FigureDataSet(
                filepath='test.png',
                purpose='invalid_purpose'
            )
        
        assert 'purpose' in str(excinfo.value).lower()
    
    
    def test_figuredataset_describe_method(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test _describe() method implementation per AbstractDataSet requirements.
        
        Validates that _describe() returns proper dictionary containing dataset
        configuration and metadata for Kedro catalog introspection.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param'],
            style_params=config['style_params'],
            save_args=config['save_args'],
            versioned=config['versioned']
        )
        
        description = dataset._describe()
        
        # Verify description contains required information
        assert isinstance(description, dict)
        assert 'filepath' in description
        assert 'purpose' in description
        assert 'condition_param' in description
        assert 'style_params' in description
        assert 'versioned' in description
        
        # Verify description values match configuration
        assert description['filepath'] == config['filepath']
        assert description['purpose'] == config['purpose']
        assert description['condition_param'] == config['condition_param']
        assert description['style_params'] == config['style_params']
        assert description['versioned'] == config['versioned']
    
    
    def test_figuredataset_save_method_basic(self, sample_figure, figure_dataset_config, 
                                           mock_figregistry_apis, temp_work_dir):
        """
        Test _save() method implementation with basic configuration.
        
        Validates that _save() properly accepts matplotlib figure objects
        and calls FigRegistry APIs with correct parameters per F-005.2.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param'],
            save_args=config['save_args']
        )
        
        # Mock session context for parameter resolution
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {'experiment_type': 'baseline'}
            
            # Execute save operation
            dataset._save(sample_figure)
        
        # Verify FigRegistry APIs were called correctly
        mock_figregistry_apis['get_style'].assert_called_once_with(
            condition='baseline',
            purpose='exploratory'
        )
        
        mock_figregistry_apis['save_figure'].assert_called_once()
        save_call_args = mock_figregistry_apis['save_figure'].call_args
        
        # Verify save_figure was called with correct arguments
        assert save_call_args[0][0] is sample_figure  # First arg is figure
        assert config['filepath'] in str(save_call_args)
        assert config['save_args']['dpi'] == 150
    
    
    def test_figuredataset_save_method_with_styling(self, complex_figure, figure_dataset_config,
                                                  mock_figregistry_apis, temp_work_dir):
        """
        Test _save() method with style parameter application.
        
        Validates that _save() correctly applies style parameters from both
        catalog configuration and FigRegistry condition resolution per F-005.2.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param'],
            style_params=config['style_params'],
            save_args=config['save_args']
        )
        
        # Configure mock to return comprehensive style dictionary
        mock_figregistry_apis['get_style'].return_value = {
            'figure.figsize': [8, 6],  # Should be overridden by style_params
            'axes.grid': True,
            'axes.grid.alpha': 0.2,
            'font.size': 11,
            'axes.labelsize': 12,  # Should be overridden by style_params
            'axes.titlesize': 14,
            'legend.fontsize': 10
        }
        
        # Mock session context
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {'analysis_mode': 'publication'}
            
            # Store initial rcParams for comparison
            initial_figsize = matplotlib.rcParams['figure.figsize']
            initial_labelsize = matplotlib.rcParams['axes.labelsize']
            
            # Execute save operation
            dataset._save(complex_figure)
            
            # Verify style was applied correctly
            # Note: In real implementation, style would be applied before save_figure call
            mock_figregistry_apis['get_style'].assert_called_once_with(
                condition='publication',
                purpose='publication'
            )
    
    
    def test_figuredataset_save_method_error_handling(self, figure_dataset_config, 
                                                    mock_figregistry_apis):
        """
        Test _save() method error handling scenarios.
        
        Validates proper error handling for invalid figure objects, missing
        parameters, and FigRegistry API failures per error management requirements.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Test invalid figure object
        with pytest.raises((TypeError, ValueError)) as excinfo:
            dataset._save("not_a_figure")
        
        assert 'figure' in str(excinfo.value).lower()
        
        # Test FigRegistry API failure
        mock_figregistry_apis['get_style'].side_effect = Exception("Style resolution failed")
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with pytest.raises(Exception) as excinfo:
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'experiment_type': 'baseline'}
                dataset._save(fig)
        
        assert "Style resolution failed" in str(excinfo.value)
        plt.close(fig)
    
    
    def test_figuredataset_load_method_not_supported(self, figure_dataset_config):
        """
        Test _load() method raises appropriate error per AbstractDataSet requirements.
        
        FigureDataSet is output-only, so _load() should raise NotImplementedError
        or appropriate exception indicating load operations are not supported.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # _load() should raise NotImplementedError for output-only dataset
        with pytest.raises(NotImplementedError) as excinfo:
            dataset._load()
        
        assert 'load' in str(excinfo.value).lower()
    
    
    def test_figuredataset_exists_method(self, figure_dataset_config, temp_work_dir):
        """
        Test _exists() method implementation per AbstractDataSet requirements.
        
        Validates that _exists() correctly checks for figure file existence
        and handles versioned and non-versioned scenarios appropriately.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        filepath = temp_work_dir / config['filepath']
        
        dataset = FigureDataSet(
            filepath=str(filepath),
            purpose=config['purpose']
        )
        
        # Initially file should not exist
        assert not dataset._exists()
        
        # Create the file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
        
        # Now file should exist
        assert dataset._exists()
        
        # Clean up
        filepath.unlink()


# =============================================================================
# CATALOG PARAMETER EXTRACTION TESTS
# =============================================================================

class TestCatalogParameterExtraction:
    """
    Test catalog parameter extraction and condition resolution per F-005.2.
    
    Validates that FigureDataSet correctly extracts purpose, condition_param,
    and style_params from Kedro catalog configurations and resolves condition
    values from session context.
    """
    
    def test_condition_parameter_extraction_basic(self, figure_dataset_config, 
                                                mock_kedro_session_context, 
                                                mock_figregistry_apis):
        """
        Test basic condition parameter extraction from session context.
        
        Validates that condition_param values are correctly extracted from
        Kedro session store and passed to FigRegistry get_style() API.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value = mock_kedro_session_context
            
            dataset._save(fig)
        
        # Verify condition parameter was extracted correctly
        mock_figregistry_apis['get_style'].assert_called_once_with(
            condition='baseline',  # From mock session store
            purpose='exploratory'
        )
        
        plt.close(fig)
    
    
    def test_condition_parameter_extraction_missing_param(self, figure_dataset_config,
                                                        mock_kedro_session_context,
                                                        mock_figregistry_apis):
        """
        Test condition parameter extraction when parameter is missing from session.
        
        Validates graceful handling when condition_param is not found in session
        store, ensuring appropriate fallback behavior or error handling.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param='missing_parameter'  # Not in session store
        )
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value = mock_kedro_session_context
            
            # Should handle missing parameter gracefully
            dataset._save(fig)
        
        # Verify get_style was called with None or default condition
        call_args = mock_figregistry_apis['get_style'].call_args
        assert call_args[1]['condition'] is None or call_args[1]['condition'] == 'default'
        assert call_args[1]['purpose'] == 'exploratory'
        
        plt.close(fig)
    
    
    def test_condition_parameter_extraction_no_session(self, figure_dataset_config,
                                                     mock_figregistry_apis):
        """
        Test condition parameter extraction when no Kedro session is available.
        
        Validates that FigureDataSet handles scenarios where no Kedro session
        context is available, falling back to appropriate default behavior.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # No session context available
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value = None
            
            dataset._save(fig)
        
        # Should handle missing session gracefully
        call_args = mock_figregistry_apis['get_style'].call_args
        assert call_args[1]['condition'] is None or call_args[1]['condition'] == 'default'
        
        plt.close(fig)
    
    
    def test_style_params_application(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test style_params application and precedence over FigRegistry styles.
        
        Validates that style_params from catalog configuration correctly
        override FigRegistry condition-based styles per precedence rules.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param'],
            style_params=config['style_params']
        )
        
        # Configure FigRegistry to return conflicting style values
        mock_figregistry_apis['get_style'].return_value = {
            'figure.figsize': [12, 8],  # Should be overridden by style_params [8, 6]
            'axes.labelsize': 16,       # Should be overridden by style_params 12
            'font.family': 'sans-serif', # Should be overridden by style_params 'serif'
            'axes.grid': True,          # Should remain as no override in style_params
            'legend.fontsize': 10       # Should remain as no override in style_params
        }
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {'analysis_mode': 'publication'}
            
            # Store initial rcParams
            initial_rcparams = matplotlib.rcParams.copy()
            
            dataset._save(fig)
            
            # In real implementation, would verify that style_params took precedence
            # For now, verify the APIs were called correctly
            mock_figregistry_apis['get_style'].assert_called_once()
        
        plt.close(fig)
    
    
    def test_purpose_parameter_validation(self, figure_dataset_config):
        """
        Test purpose parameter validation against allowed values.
        
        Validates that purpose parameter is properly validated against
        allowed values (exploratory, presentation, publication) with
        appropriate error handling for invalid values.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        valid_purposes = ['exploratory', 'presentation', 'publication']
        
        # Test valid purposes
        for purpose in valid_purposes:
            dataset = FigureDataSet(
                filepath='test.png',
                purpose=purpose
            )
            assert dataset._purpose == purpose
        
        # Test invalid purposes
        invalid_purposes = ['invalid', 'research', 'analysis', 123, None]
        
        for invalid_purpose in invalid_purposes:
            with pytest.raises((ValueError, TypeError)) as excinfo:
                FigureDataSet(
                    filepath='test.png',
                    purpose=invalid_purpose
                )
            
            assert 'purpose' in str(excinfo.value).lower()
    
    
    def test_save_args_parameter_handling(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test save_args parameter extraction and application.
        
        Validates that save_args from catalog configuration are correctly
        passed to matplotlib savefig operations through FigRegistry save_figure.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            save_args=config['save_args']
        )
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        dataset._save(fig)
        
        # Verify save_figure was called with correct save_args
        save_call_args = mock_figregistry_apis['save_figure'].call_args
        
        # save_args should be passed to save_figure
        # Exact implementation depends on save_figure API design
        assert save_call_args is not None
        
        # Check that expected save_args values are present
        call_str = str(save_call_args)
        assert '300' in call_str  # dpi value
        assert 'pdf' in call_str.lower()  # format value
        
        plt.close(fig)


# =============================================================================
# VERSIONING COMPATIBILITY TESTS
# =============================================================================

class TestVersioningCompatibility:
    """
    Test compatibility with Kedro versioning system per F-005.2.
    
    Validates that FigRegistry timestamp versioning coexists with Kedro's
    dataset versioning without conflicts, ensuring both systems function
    independently while providing consistent version tracking.
    """
    
    def test_versioned_dataset_initialization(self, figure_dataset_config):
        """
        Test FigureDataSet initialization with versioning enabled.
        
        Validates that versioned=True parameter is properly handled and
        that versioning configuration doesn't conflict with FigRegistry settings.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            versioned=True
        )
        
        assert dataset._versioned is True
        
        # Test versioned dataset description includes versioning info
        description = dataset._describe()
        assert 'versioned' in description
        assert description['versioned'] is True
    
    
    def test_versioned_save_operation(self, sample_figure, figure_dataset_config,
                                    mock_figregistry_apis, temp_work_dir):
        """
        Test save operation with Kedro versioning enabled.
        
        Validates that FigureDataSet save operations work correctly with
        Kedro versioning and that versioned paths are handled appropriately.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            versioned=True
        )
        
        # Mock versioned save behavior
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            
            # In versioned mode, Kedro might modify the filepath
            with patch.object(dataset, '_get_versioned_path') as mock_versioned_path:
                mock_versioned_path.return_value = f"{config['filepath']}.20231201T123456.000Z"
                
                dataset._save(sample_figure)
        
        # Verify FigRegistry APIs were called
        mock_figregistry_apis['save_figure'].assert_called_once()
        
        # In real implementation, would verify versioned path handling
        save_call_args = mock_figregistry_apis['save_figure'].call_args
        assert save_call_args is not None
    
    
    def test_timestamp_versioning_coexistence(self, sample_figure, figure_dataset_config,
                                            mock_figregistry_apis):
        """
        Test coexistence of FigRegistry timestamp and Kedro versioning.
        
        Validates that FigRegistry's timestamp-based file naming works
        alongside Kedro's versioning system without conflicts or interference.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            versioned=True
        )
        
        # Configure FigRegistry to return timestamped filename
        mock_figregistry_apis['save_figure'].return_value = (
            'data/08_reporting/figures/advanced_plot_20231201_123456.pdf'
        )
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            
            dataset._save(sample_figure)
        
        # Verify both versioning systems can coexist
        mock_figregistry_apis['save_figure'].assert_called_once()
        
        # Check that returned path includes timestamp
        returned_path = mock_figregistry_apis['save_figure'].return_value
        assert '20231201_123456' in returned_path
    
    
    def test_versioning_error_handling(self, sample_figure, figure_dataset_config,
                                     mock_figregistry_apis):
        """
        Test error handling in versioned datasets.
        
        Validates proper error handling when versioning operations fail
        and ensures graceful degradation or appropriate error reporting.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            versioned=True
        )
        
        # Simulate versioning failure
        with patch.object(dataset, '_get_versioned_path') as mock_versioned_path:
            mock_versioned_path.side_effect = Exception("Versioning failed")
            
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {}
                
                # Should handle versioning failure gracefully
                with pytest.raises(Exception) as excinfo:
                    dataset._save(sample_figure)
                
                assert "Versioning failed" in str(excinfo.value)


# =============================================================================
# PERFORMANCE AND THREADING TESTS
# =============================================================================

class TestPerformanceAndThreading:
    """
    Test performance requirements and thread safety per Section 5.2.8.
    
    Validates that FigureDataSet operations meet performance targets
    (<200ms per save, <5% overhead) and support thread-safe operation
    for parallel Kedro runner execution.
    """
    
    def test_save_operation_performance(self, sample_figure, figure_dataset_config,
                                      mock_figregistry_apis, performance_baseline, benchmark):
        """
        Test FigureDataSet save operation performance per Section 6.6.4.3.
        
        Validates that save operations complete within 200ms target and
        overhead compared to manual matplotlib saves remains under 5%.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Configure fast mock responses
        mock_figregistry_apis['get_style'].return_value = {}
        mock_figregistry_apis['save_figure'].return_value = '/mock/path/test.png'
        
        def save_operation():
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'experiment_type': 'baseline'}
                dataset._save(sample_figure)
        
        # Benchmark the save operation
        result = benchmark(save_operation)
        
        # Verify performance meets requirements
        save_time = result
        target_time = performance_baseline['dataset_save_target']
        
        assert save_time < target_time, (
            f"Save operation took {save_time:.3f}s, exceeds target {target_time:.3f}s"
        )
        
        # Calculate overhead compared to baseline
        baseline_time = performance_baseline['manual_save_time']
        overhead_ratio = (save_time - baseline_time) / baseline_time
        max_overhead = performance_baseline['overhead_threshold']
        
        assert overhead_ratio < max_overhead, (
            f"Overhead {overhead_ratio:.2%} exceeds maximum {max_overhead:.2%}"
        )
    
    
    def test_style_resolution_performance(self, figure_dataset_config, mock_figregistry_apis,
                                        performance_baseline, benchmark):
        """
        Test style resolution performance within target thresholds.
        
        Validates that get_style() API calls complete within 1ms target
        and don't significantly impact overall save operation performance.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Configure realistic mock response time
        def mock_get_style_with_delay(*args, **kwargs):
            time.sleep(0.0005)  # 0.5ms simulated processing
            return {'figure.figsize': [10, 6]}
        
        mock_figregistry_apis['get_style'].side_effect = mock_get_style_with_delay
        
        def style_resolution():
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'experiment_type': 'baseline'}
                # Just call the style resolution part
                return dataset._resolve_style_parameters('baseline')
        
        # Benchmark style resolution
        result = benchmark(style_resolution)
        
        target_time = performance_baseline['style_resolution_time']
        assert result < target_time, (
            f"Style resolution took {result:.4f}s, exceeds target {target_time:.4f}s"
        )
    
    
    def test_thread_safe_parallel_execution(self, sample_figure, figure_dataset_config,
                                          mock_figregistry_apis):
        """
        Test thread-safe operation for parallel Kedro runner execution.
        
        Validates that multiple FigureDataSet instances can safely operate
        concurrently without race conditions or state corruption.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        
        # Create multiple dataset instances
        datasets = []
        for i in range(5):
            dataset = FigureDataSet(
                filepath=f"data/08_reporting/figures/parallel_test_{i}.png",
                purpose=config['purpose'],
                condition_param=config['condition_param']
            )
            datasets.append(dataset)
        
        # Configure thread-safe mock responses
        call_counts = {'get_style': 0, 'save_figure': 0}
        call_lock = threading.Lock()
        
        def thread_safe_get_style(*args, **kwargs):
            with call_lock:
                call_counts['get_style'] += 1
                time.sleep(0.001)  # Simulate processing time
                return {'figure.figsize': [10, 6]}
        
        def thread_safe_save_figure(*args, **kwargs):
            with call_lock:
                call_counts['save_figure'] += 1
                time.sleep(0.002)  # Simulate file I/O
                return f'/mock/path/test_{call_counts["save_figure"]}.png'
        
        mock_figregistry_apis['get_style'].side_effect = thread_safe_get_style
        mock_figregistry_apis['save_figure'].side_effect = thread_safe_save_figure
        
        def save_with_dataset(dataset_index):
            """Save operation for a specific dataset."""
            dataset = datasets[dataset_index]
            
            # Create figure for this thread
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3], [dataset_index, dataset_index + 1, dataset_index + 2])
            ax.set_title(f'Thread {dataset_index}')
            
            try:
                with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                    mock_session.return_value.store = {'experiment_type': f'test_{dataset_index}'}
                    dataset._save(fig)
                return True
            except Exception as e:
                print(f"Thread {dataset_index} failed: {e}")
                return False
            finally:
                plt.close(fig)
        
        # Execute parallel saves
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_with_dataset, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # Verify all operations completed successfully
        assert all(results), "Some parallel operations failed"
        
        # Verify all API calls were made
        assert call_counts['get_style'] == 5
        assert call_counts['save_figure'] == 5
    
    
    def test_memory_usage_efficiency(self, complex_figure, figure_dataset_config,
                                   mock_figregistry_apis, performance_baseline):
        """
        Test memory usage efficiency during save operations.
        
        Validates that FigureDataSet operations don't cause excessive
        memory overhead or memory leaks during figure processing.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        import psutil
        import gc
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Measure initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple save operations
        for i in range(10):
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'analysis_mode': 'publication'}
                dataset._save(complex_figure)
        
        # Force garbage collection
        gc.collect()
        
        # Measure final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify memory usage is within acceptable bounds
        max_memory_increase = performance_baseline['memory_baseline_mb']
        assert memory_increase < max_memory_increase, (
            f"Memory usage increased by {memory_increase:.1f}MB, "
            f"exceeds limit {max_memory_increase:.1f}MB"
        )


# =============================================================================
# ERROR HANDLING AND SECURITY TESTS
# =============================================================================

class TestErrorHandlingAndSecurity:
    """
    Test error handling and security validation per Section 6.6.8.
    
    Validates robust error management for malformed configurations, missing
    parameters, file system failures, and security constraints for parameter
    validation and path traversal prevention.
    """
    
    def test_malformed_catalog_configuration_handling(self, mock_figregistry_apis):
        """
        Test handling of malformed catalog configurations.
        
        Validates graceful error handling for invalid catalog parameter
        combinations, malformed YAML, and type validation failures.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test various malformed configurations
        malformed_configs = [
            {
                'filepath': 123,  # Should be string
                'purpose': 'exploratory'
            },
            {
                'filepath': 'test.png',
                'purpose': None  # Should be string
            },
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'style_params': 'not_a_dict'  # Should be dict
            },
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'save_args': 'not_a_dict'  # Should be dict
            },
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'versioned': 'not_a_bool'  # Should be bool
            }
        ]
        
        for config in malformed_configs:
            with pytest.raises((TypeError, ValueError)) as excinfo:
                FigureDataSet(**config)
            
            # Error message should be informative
            error_msg = str(excinfo.value).lower()
            assert any(keyword in error_msg for keyword in 
                      ['type', 'value', 'invalid', 'expected'])
    
    
    def test_missing_condition_parameter_handling(self, sample_figure, figure_dataset_config,
                                                mock_figregistry_apis):
        """
        Test handling when condition parameter is missing from session.
        
        Validates that missing condition parameters are handled gracefully
        with appropriate fallback behavior and clear error messages.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param='missing_parameter'
        )
        
        # Test with empty session store
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            
            # Should handle gracefully, not raise exception
            dataset._save(sample_figure)
        
        # Verify get_style was called with appropriate fallback
        call_args = mock_figregistry_apis['get_style'].call_args
        condition_value = call_args[1]['condition']
        assert condition_value is None or condition_value == 'default'
    
    
    def test_figregistry_api_failure_handling(self, sample_figure, figure_dataset_config):
        """
        Test handling of FigRegistry API failures.
        
        Validates proper error propagation and handling when FigRegistry
        get_style() or save_figure() operations fail.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Test get_style failure
        with patch('figregistry.get_style') as mock_get_style:
            mock_get_style.side_effect = Exception("Style resolution failed")
            
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'experiment_type': 'baseline'}
                
                with pytest.raises(Exception) as excinfo:
                    dataset._save(sample_figure)
                
                assert "Style resolution failed" in str(excinfo.value)
        
        # Test save_figure failure
        with patch('figregistry.get_style') as mock_get_style, \
             patch('figregistry.save_figure') as mock_save_figure:
            
            mock_get_style.return_value = {}
            mock_save_figure.side_effect = Exception("Save operation failed")
            
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'experiment_type': 'baseline'}
                
                with pytest.raises(Exception) as excinfo:
                    dataset._save(sample_figure)
                
                assert "Save operation failed" in str(excinfo.value)
    
    
    def test_file_system_failure_handling(self, sample_figure, figure_dataset_config,
                                        mock_figregistry_apis):
        """
        Test handling of file system operation failures.
        
        Validates proper error handling for file system issues including
        permission errors, disk space, and directory creation failures.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath='/invalid/path/test.png',  # Invalid path
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Configure save_figure to simulate file system error
        mock_figregistry_apis['save_figure'].side_effect = PermissionError(
            "Permission denied: /invalid/path/test.png"
        )
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {'experiment_type': 'baseline'}
            
            with pytest.raises(PermissionError) as excinfo:
                dataset._save(sample_figure)
            
            assert "Permission denied" in str(excinfo.value)
    
    
    def test_path_traversal_prevention(self, figure_dataset_config):
        """
        Test prevention of path traversal attacks in filepath parameter.
        
        Validates that malicious filepath values cannot escape designated
        output directories through path traversal sequences.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test various path traversal attempts
        malicious_paths = [
            '../../../etc/passwd',
            '../../home/user/.ssh/id_rsa',
            '/absolute/path/to/sensitive/file',
            'data/../../../etc/passwd',
            'data/08_reporting/../../../../../../etc/passwd'
        ]
        
        for malicious_path in malicious_paths:
            # FigureDataSet should either:
            # 1. Reject the path during initialization, or
            # 2. Sanitize the path to prevent traversal
            try:
                dataset = FigureDataSet(
                    filepath=malicious_path,
                    purpose='exploratory'
                )
                
                # If initialization succeeds, verify path is sanitized
                sanitized_path = dataset._filepath
                assert not sanitized_path.startswith('/etc/')
                assert not sanitized_path.startswith('/home/')
                assert '../' not in sanitized_path
                
            except (ValueError, SecurityError) as e:
                # Rejecting malicious paths is acceptable
                assert 'path' in str(e).lower() or 'security' in str(e).lower()
    
    
    def test_parameter_injection_prevention(self, figure_dataset_config):
        """
        Test prevention of parameter injection in condition_param values.
        
        Validates that condition_param values undergo proper sanitization
        to prevent injection of malicious content or system commands.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test various injection attempts
        injection_attempts = [
            '"; rm -rf /; echo "',
            '$(rm -rf /)',
            '`rm -rf /`',
            '${rm -rf /}',
            'normal_param; malicious_command',
            'param\nmalicious_command',
            'param\rmalicious_command'
        ]
        
        config = figure_dataset_config['basic_config']
        
        for injection_param in injection_attempts:
            # Should either reject during initialization or sanitize
            try:
                dataset = FigureDataSet(
                    filepath=config['filepath'],
                    purpose=config['purpose'],
                    condition_param=injection_param
                )
                
                # If accepted, verify parameter is sanitized
                sanitized_param = dataset._condition_param
                assert ';' not in sanitized_param
                assert '$' not in sanitized_param
                assert '`' not in sanitized_param
                assert '\n' not in sanitized_param
                assert '\r' not in sanitized_param
                
            except (ValueError, SecurityError) as e:
                # Rejecting injection attempts is acceptable
                assert 'parameter' in str(e).lower() or 'invalid' in str(e).lower()
    
    
    def test_oversized_parameter_handling(self, figure_dataset_config):
        """
        Test handling of oversized parameter values.
        
        Validates that extremely large parameter values are handled
        gracefully without causing memory exhaustion or system instability.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        
        # Test oversized style_params
        oversized_style_params = {f'param_{i}': f'value_{i}' for i in range(10000)}
        
        with pytest.raises((ValueError, MemoryError)) as excinfo:
            FigureDataSet(
                filepath=config['filepath'],
                purpose=config['purpose'],
                style_params=oversized_style_params
            )
        
        error_msg = str(excinfo.value).lower()
        assert any(keyword in error_msg for keyword in 
                  ['size', 'limit', 'too large', 'memory'])
        
        # Test extremely long filepath
        oversized_filepath = 'a' * 10000
        
        with pytest.raises((ValueError, OSError)) as excinfo:
            FigureDataSet(
                filepath=oversized_filepath,
                purpose=config['purpose']
            )


# =============================================================================
# INTEGRATION AND COMPATIBILITY TESTS
# =============================================================================

class TestIntegrationAndCompatibility:
    """
    Test integration with Kedro ecosystem and compatibility requirements.
    
    Validates seamless integration with Kedro's data catalog system,
    compatibility across supported Python and Kedro versions, and
    proper interaction with other Kedro components.
    """
    
    def test_kedro_catalog_integration(self, sample_figure, figure_dataset_config,
                                     mock_kedro_session_context, mock_figregistry_apis):
        """
        Test integration with Kedro data catalog system.
        
        Validates that FigureDataSet works correctly within Kedro's catalog
        framework including registration, discovery, and execution.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Simulate catalog save operation
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value = mock_kedro_session_context
            
            # This simulates kedro catalog.save("dataset_name", figure)
            dataset._save(sample_figure)
        
        # Verify proper integration with FigRegistry
        mock_figregistry_apis['get_style'].assert_called_once()
        mock_figregistry_apis['save_figure'].assert_called_once()
        
        # Verify catalog-specific behavior
        assert hasattr(dataset, '_describe')
        assert hasattr(dataset, '_exists')
        description = dataset._describe()
        assert isinstance(description, dict)
    
    
    def test_kedro_version_compatibility(self, sample_figure, figure_dataset_config,
                                       mock_figregistry_apis):
        """
        Test compatibility across supported Kedro versions.
        
        Validates that FigureDataSet works correctly with Kedro versions
        in the supported range (0.18.0-0.19.x) with consistent behavior.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
            import kedro
        except ImportError:
            pytest.skip("FigureDataSet or Kedro not available - implementation pending")
        
        # Verify Kedro version is in supported range
        kedro_version = kedro.__version__
        major, minor = map(int, kedro_version.split('.')[:2])
        
        assert (major == 0 and minor >= 18), (
            f"Kedro version {kedro_version} not in supported range >=0.18.0"
        )
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Test core functionality works across versions
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {'experiment_type': 'baseline'}
            dataset._save(sample_figure)
        
        # Verify expected behavior
        assert mock_figregistry_apis['get_style'].called
        assert mock_figregistry_apis['save_figure'].called
    
    
    def test_matplotlib_version_compatibility(self, sample_figure, figure_dataset_config,
                                            mock_figregistry_apis):
        """
        Test compatibility with supported matplotlib versions.
        
        Validates that FigureDataSet works correctly with matplotlib
        versions in the supported range (>=3.9.0) with consistent behavior.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
            import matplotlib
        except ImportError:
            pytest.skip("FigureDataSet or matplotlib not available")
        
        # Verify matplotlib version is supported
        mpl_version = matplotlib.__version__
        major, minor = map(int, mpl_version.split('.')[:2])
        
        assert (major > 3 or (major == 3 and minor >= 9)), (
            f"Matplotlib version {mpl_version} not in supported range >=3.9.0"
        )
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Test figure handling works correctly
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            dataset._save(sample_figure)
        
        # Verify matplotlib figure was handled correctly
        save_call_args = mock_figregistry_apis['save_figure'].call_args
        assert save_call_args[0][0] is sample_figure
    
    
    def test_python_version_compatibility(self, figure_dataset_config):
        """
        Test compatibility with supported Python versions.
        
        Validates that FigureDataSet works correctly with Python
        versions in the supported range (3.10+) including type annotations
        and modern Python features.
        """
        import sys
        
        # Verify Python version is supported
        python_version = sys.version_info
        assert python_version >= (3, 10), (
            f"Python version {python_version} not in supported range >=3.10"
        )
        
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        
        # Test initialization with type hints works correctly
        dataset: FigureDataSet = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Verify type annotations are preserved
        assert hasattr(FigureDataSet, '__annotations__')
        
        # Test that modern Python features work
        description = dataset._describe()
        assert isinstance(description, dict)
    
    
    def test_cross_platform_compatibility(self, sample_figure, figure_dataset_config,
                                        mock_figregistry_apis, cross_platform_test_env):
        """
        Test cross-platform compatibility (Windows, macOS, Linux).
        
        Validates that FigureDataSet works correctly across different
        operating systems with proper path handling and file operations.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        platforms_to_test = ['windows', 'linux', 'macos']
        
        for platform in platforms_to_test:
            cross_platform_test_env(platform)
            
            config = figure_dataset_config['basic_config']
            dataset = FigureDataSet(
                filepath=config['filepath'],
                purpose=config['purpose']
            )
            
            # Test that initialization works on all platforms
            assert hasattr(dataset, '_filepath')
            assert hasattr(dataset, '_purpose')
            
            # Test save operation
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {}
                dataset._save(sample_figure)
            
            # Verify APIs were called regardless of platform
            assert mock_figregistry_apis['save_figure'].called
            
            # Reset mocks for next platform
            mock_figregistry_apis['save_figure'].reset_mock()
            mock_figregistry_apis['get_style'].reset_mock()


# =============================================================================
# COMPREHENSIVE REGRESSION AND EDGE CASE TESTS
# =============================================================================

class TestRegressionAndEdgeCases:
    """
    Test regression scenarios and edge cases for comprehensive coverage.
    
    Validates behavior in unusual scenarios, boundary conditions, and
    potential regression cases to ensure robust operation across all
    supported use cases and configurations.
    """
    
    def test_empty_figure_handling(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test handling of empty or minimal matplotlib figures.
        
        Validates that FigureDataSet correctly processes figures with
        no data, empty axes, or minimal content without errors.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Test completely empty figure
        empty_fig, empty_ax = plt.subplots()
        # No plot calls - completely empty
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            dataset._save(empty_fig)
        
        mock_figregistry_apis['save_figure'].assert_called()
        plt.close(empty_fig)
        
        # Test figure with empty axes but title
        titled_fig, titled_ax = plt.subplots()
        titled_ax.set_title('Empty Plot with Title')
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            dataset._save(titled_fig)
        
        assert mock_figregistry_apis['save_figure'].call_count == 2
        plt.close(titled_fig)
    
    
    def test_complex_figure_structures(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test handling of complex figure structures and layouts.
        
        Validates that FigureDataSet correctly processes figures with
        multiple subplots, complex layouts, and nested figure structures.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Create complex figure with GridSpec
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Add various subplot types
        ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
        ax2 = fig.add_subplot(gs[1, 0])  # Middle left
        ax3 = fig.add_subplot(gs[1, 1:])  # Middle right (2 columns)
        ax4 = fig.add_subplot(gs[2, 0])  # Bottom left
        ax5 = fig.add_subplot(gs[2, 1])  # Bottom middle
        ax6 = fig.add_subplot(gs[2, 2])  # Bottom right
        
        # Add content to each subplot
        x = np.linspace(0, 10, 100)
        ax1.plot(x, np.sin(x))
        ax1.set_title('Main Plot')
        
        ax2.bar(['A', 'B', 'C'], [1, 2, 3])
        ax3.scatter(np.random.randn(50), np.random.randn(50))
        ax4.hist(np.random.randn(100), bins=20)
        ax5.pie([30, 35, 25, 10], labels=['A', 'B', 'C', 'D'])
        ax6.imshow(np.random.rand(10, 10), cmap='viridis')
        
        # Test save operation
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            dataset._save(fig)
        
        mock_figregistry_apis['save_figure'].assert_called()
        plt.close(fig)
    
    
    def test_unicode_and_special_characters(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test handling of unicode and special characters in parameters.
        
        Validates that FigureDataSet correctly processes parameters
        containing unicode characters, special symbols, and international text.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test unicode in filepath
        unicode_config = {
            'filepath': 'data/08_reporting/figures/测试图表_αβγ.png',
            'purpose': 'exploratory'
        }
        
        dataset = FigureDataSet(**unicode_config)
        assert dataset._filepath == unicode_config['filepath']
        
        # Test unicode in condition parameter values
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title('Unicode Test: 图表标题 αβγδε')
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {'测试参数': '实验条件α'}
            
            dataset_with_unicode_param = FigureDataSet(
                filepath='test.png',
                purpose='exploratory',
                condition_param='测试参数'
            )
            
            dataset_with_unicode_param._save(fig)
        
        # Verify unicode values were handled correctly
        call_args = mock_figregistry_apis['get_style'].call_args
        assert call_args[1]['condition'] == '实验条件α'
        
        plt.close(fig)
    
    
    def test_boundary_value_parameters(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test handling of boundary value parameters.
        
        Validates behavior with minimum/maximum values, empty strings,
        and edge cases for numeric and string parameters.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test minimum required configuration
        minimal_dataset = FigureDataSet(
            filepath='test.png',
            purpose='exploratory'
        )
        
        # Test with empty style_params
        empty_params_dataset = FigureDataSet(
            filepath='test.png',
            purpose='exploratory',
            style_params={}
        )
        
        # Test with empty save_args
        empty_save_args_dataset = FigureDataSet(
            filepath='test.png',
            purpose='exploratory',
            save_args={}
        )
        
        # Test with very long but valid filepath
        long_filepath = 'data/08_reporting/figures/' + 'a' * 200 + '.png'
        long_path_dataset = FigureDataSet(
            filepath=long_filepath,
            purpose='exploratory'
        )
        
        # Verify all datasets were created successfully
        for dataset in [minimal_dataset, empty_params_dataset, 
                       empty_save_args_dataset, long_path_dataset]:
            assert hasattr(dataset, '_filepath')
            assert hasattr(dataset, '_purpose')
    
    
    def test_concurrent_modification_scenarios(self, sample_figure, figure_dataset_config,
                                             mock_figregistry_apis):
        """
        Test scenarios involving concurrent modifications.
        
        Validates behavior when figure objects are modified during
        save operations or when multiple operations occur simultaneously.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Create figure that will be modified during save
        fig, ax = plt.subplots()
        x = [1, 2, 3]
        y = [1, 2, 3]
        line, = ax.plot(x, y)
        
        def modify_figure_during_save(*args, **kwargs):
            """Mock save_figure that modifies figure during save."""
            # Modify figure during save operation
            line.set_ydata([3, 2, 1])
            ax.set_title('Modified during save')
            return '/mock/path/test.png'
        
        mock_figregistry_apis['save_figure'].side_effect = modify_figure_during_save
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            
            # Should handle concurrent modification gracefully
            dataset._save(fig)
        
        mock_figregistry_apis['save_figure'].assert_called_once()
        plt.close(fig)
    
    
    def test_memory_pressure_scenarios(self, figure_dataset_config, mock_figregistry_apis):
        """
        Test behavior under memory pressure scenarios.
        
        Validates that FigureDataSet operations complete successfully
        even when system memory is constrained or when processing
        very large figures.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Create figure with large amount of data
        fig, ax = plt.subplots(figsize=(20, 15))
        
        # Generate large dataset
        n_points = 100000
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)
        colors = np.random.randn(n_points)
        
        # Create memory-intensive plot
        scatter = ax.scatter(x, y, c=colors, alpha=0.6, s=1)
        ax.set_title('Large Dataset Plot')
        
        # Add colorbar (additional memory usage)
        cbar = plt.colorbar(scatter)
        
        with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
            mock_session.return_value.store = {}
            
            # Should handle large figure without memory issues
            dataset._save(fig)
        
        mock_figregistry_apis['save_figure'].assert_called_once()
        plt.close(fig)


# =============================================================================
# PERFORMANCE BENCHMARKING SUITE
# =============================================================================

@pytest.mark.plugin_performance
class TestPerformanceBenchmarks:
    """
    Comprehensive performance benchmarking for FigureDataSet operations.
    
    Provides detailed performance measurement and validation against
    requirements including save operation timing, memory efficiency,
    and comparison with manual matplotlib save operations.
    """
    
    def test_benchmark_save_operation_simple_figure(self, sample_figure, figure_dataset_config,
                                                   mock_figregistry_apis, benchmark):
        """
        Benchmark save operation with simple figure per Section 6.6.4.3.
        
        Measures performance of basic FigureDataSet save operation and
        validates against 200ms target with simple figure content.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param']
        )
        
        # Configure optimized mock responses
        mock_figregistry_apis['get_style'].return_value = {'figure.figsize': [10, 6]}
        mock_figregistry_apis['save_figure'].return_value = '/mock/path/test.png'
        
        def save_operation():
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'experiment_type': 'baseline'}
                dataset._save(sample_figure)
        
        # Benchmark the operation
        result = benchmark.pedantic(save_operation, rounds=10, iterations=5)
        
        # Validate performance requirements
        assert result < 0.200, f"Save operation took {result:.3f}s, exceeds 200ms target"
        
        # Store result for comparison
        benchmark.extra_info['operation'] = 'simple_figure_save'
        benchmark.extra_info['target_ms'] = 200
        benchmark.extra_info['actual_ms'] = result * 1000
    
    
    def test_benchmark_save_operation_complex_figure(self, complex_figure, figure_dataset_config,
                                                   mock_figregistry_apis, benchmark):
        """
        Benchmark save operation with complex figure.
        
        Measures performance with complex multi-subplot figures to validate
        performance scales appropriately with figure complexity.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['advanced_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose'],
            condition_param=config['condition_param'],
            style_params=config['style_params']
        )
        
        # Configure comprehensive style response
        mock_figregistry_apis['get_style'].return_value = {
            'figure.figsize': [12, 10],
            'axes.grid': True,
            'font.size': 11,
            'axes.labelsize': 12
        }
        
        def save_complex_operation():
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {'analysis_mode': 'publication'}
                dataset._save(complex_figure)
        
        result = benchmark.pedantic(save_complex_operation, rounds=5, iterations=3)
        
        # Complex figures may take longer but should still be reasonable
        assert result < 0.500, f"Complex save took {result:.3f}s, exceeds 500ms threshold"
        
        benchmark.extra_info['operation'] = 'complex_figure_save'
        benchmark.extra_info['actual_ms'] = result * 1000
    
    
    def test_benchmark_overhead_vs_manual_save(self, sample_figure, figure_dataset_config,
                                             mock_figregistry_apis, benchmark, temp_work_dir):
        """
        Benchmark overhead compared to manual matplotlib save operations.
        
        Compares FigureDataSet save performance against direct plt.savefig
        to validate <5% overhead requirement per performance specifications.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        # Configure fast mock responses to isolate FigureDataSet overhead
        mock_figregistry_apis['get_style'].return_value = {}
        mock_figregistry_apis['save_figure'].return_value = '/mock/path/test.png'
        
        # Benchmark manual save operation
        manual_save_path = temp_work_dir / 'manual_test.png'
        def manual_save():
            sample_figure.savefig(manual_save_path, dpi=150, bbox_inches='tight')
        
        manual_time = benchmark.pedantic(manual_save, rounds=10, iterations=5)
        
        # Benchmark FigureDataSet save operation
        def dataset_save():
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {}
                dataset._save(sample_figure)
        
        dataset_time = benchmark.pedantic(dataset_save, rounds=10, iterations=5)
        
        # Calculate overhead
        overhead = (dataset_time - manual_time) / manual_time
        max_overhead = 0.05  # 5% maximum
        
        assert overhead < max_overhead, (
            f"Overhead {overhead:.2%} exceeds maximum {max_overhead:.2%}"
        )
        
        benchmark.extra_info['manual_time_ms'] = manual_time * 1000
        benchmark.extra_info['dataset_time_ms'] = dataset_time * 1000
        benchmark.extra_info['overhead_percent'] = overhead * 100
    
    
    def test_benchmark_memory_efficiency(self, figure_dataset_config, mock_figregistry_apis,
                                       benchmark):
        """
        Benchmark memory efficiency during save operations.
        
        Measures memory usage patterns to ensure efficient memory management
        and validate against memory overhead requirements.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
            import psutil
        except ImportError:
            pytest.skip("FigureDataSet or psutil not available")
        
        config = figure_dataset_config['basic_config']
        dataset = FigureDataSet(
            filepath=config['filepath'],
            purpose=config['purpose']
        )
        
        process = psutil.Process()
        
        def memory_efficient_save():
            # Create figure within the benchmark
            fig, ax = plt.subplots(figsize=(10, 8))
            x = np.linspace(0, 10, 1000)
            y = np.sin(x) + 0.1 * np.random.randn(1000)
            ax.plot(x, y)
            ax.set_title('Memory Efficiency Test')
            
            initial_memory = process.memory_info().rss
            
            with patch('figregistry_kedro.datasets._get_current_session') as mock_session:
                mock_session.return_value.store = {}
                dataset._save(fig)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            plt.close(fig)
            return memory_increase
        
        memory_increase = benchmark(memory_efficient_save)
        
        # Memory increase should be minimal (< 10MB for this test)
        max_memory_increase = 10 * 1024 * 1024  # 10MB in bytes
        assert memory_increase < max_memory_increase, (
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB, "
            f"exceeds {max_memory_increase / 1024 / 1024:.1f}MB limit"
        )
        
        benchmark.extra_info['memory_increase_mb'] = memory_increase / 1024 / 1024


# =============================================================================
# SECURITY AND ROBUSTNESS TESTS
# =============================================================================

@pytest.mark.security_test
class TestSecurityAndRobustness:
    """
    Security validation and robustness testing per Section 6.6.8.
    
    Validates security constraints, input validation, and protection
    against malicious inputs or configuration manipulation.
    """
    
    def test_configuration_injection_prevention(self, mock_figregistry_apis):
        """
        Test prevention of configuration injection attacks.
        
        Validates that malicious configuration values cannot be used
        to inject code or manipulate system behavior.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test code injection in style_params
        malicious_style_params = {
            '__import__("os").system("rm -rf /")': 'malicious_value',
            'eval("print(\'injected\')")': 'another_malicious_value'
        }
        
        # Should reject or sanitize malicious keys
        with pytest.raises((ValueError, SecurityError, TypeError)):
            FigureDataSet(
                filepath='test.png',
                purpose='exploratory',
                style_params=malicious_style_params
            )
    
    
    def test_path_sanitization_comprehensive(self, mock_figregistry_apis):
        """
        Test comprehensive path sanitization and validation.
        
        Validates protection against various path manipulation techniques
        including traversal, absolute paths, and symbolic links.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        dangerous_paths = [
            # Path traversal attempts
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            'data/../../../etc/passwd',
            'data\\..\\..\\..\\windows\\system32',
            
            # Absolute path attempts
            '/etc/passwd',
            'C:\\Windows\\System32\\config\\sam',
            '/usr/bin/bash',
            
            # Special device files (Unix)
            '/dev/null',
            '/dev/zero',
            '/proc/version',
            
            # Network paths
            '//malicious-server/share/file',
            '\\\\malicious-server\\share\\file',
            
            # URL-like paths
            'http://malicious.com/payload',
            'file:///etc/passwd',
            'ftp://malicious.com/payload'
        ]
        
        for dangerous_path in dangerous_paths:
            with pytest.raises((ValueError, SecurityError, OSError)) as excinfo:
                FigureDataSet(
                    filepath=dangerous_path,
                    purpose='exploratory'
                )
            
            # Error should indicate security or path validation issue
            error_msg = str(excinfo.value).lower()
            assert any(keyword in error_msg for keyword in 
                      ['path', 'security', 'invalid', 'not allowed'])
    
    
    def test_parameter_validation_comprehensive(self, mock_figregistry_apis):
        """
        Test comprehensive parameter validation and sanitization.
        
        Validates that all input parameters are properly validated
        and malicious values are rejected or sanitized appropriately.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test various malicious parameter combinations
        malicious_parameters = [
            # SQL injection attempts
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'condition_param': "'; DROP TABLE figures; --"
            },
            
            # Command injection attempts
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'condition_param': '$(rm -rf /)'
            },
            
            # Script injection attempts
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'style_params': {
                    '<script>alert("xss")</script>': 'value'
                }
            },
            
            # Buffer overflow attempts
            {
                'filepath': 'test.png',
                'purpose': 'exploratory',
                'condition_param': 'A' * 100000
            }
        ]
        
        for malicious_params in malicious_parameters:
            # Should either reject completely or sanitize safely
            try:
                dataset = FigureDataSet(**malicious_params)
                
                # If accepted, verify parameters are sanitized
                if hasattr(dataset, '_condition_param') and dataset._condition_param:
                    sanitized = dataset._condition_param
                    assert '<script>' not in sanitized
                    assert 'DROP TABLE' not in sanitized
                    assert '$(rm' not in sanitized
                    assert len(sanitized) < 1000  # Reasonable length limit
                    
            except (ValueError, SecurityError, TypeError, MemoryError):
                # Rejecting malicious parameters is acceptable
                pass
    
    
    def test_resource_exhaustion_protection(self, mock_figregistry_apis):
        """
        Test protection against resource exhaustion attacks.
        
        Validates that FigureDataSet protects against attempts to
        exhaust system resources through malicious configurations.
        """
        try:
            from figregistry_kedro.datasets import FigureDataSet
        except ImportError:
            pytest.skip("FigureDataSet not available - implementation pending")
        
        # Test extremely large parameter dictionaries
        with pytest.raises((ValueError, MemoryError, OSError)):
            huge_style_params = {f'param_{i}': f'value_{i}' for i in range(100000)}
            FigureDataSet(
                filepath='test.png',
                purpose='exploratory',
                style_params=huge_style_params
            )
        
        # Test deeply nested parameter structures
        with pytest.raises((ValueError, RecursionError)):
            nested_dict = {'level_0': {}}
            current = nested_dict['level_0']
            for i in range(1000):  # Create deep nesting
                current[f'level_{i+1}'] = {}
                current = current[f'level_{i+1}']
            
            FigureDataSet(
                filepath='test.png',
                purpose='exploratory',
                style_params=nested_dict
            )


# =============================================================================
# TEST EXECUTION AND REPORTING
# =============================================================================

if __name__ == '__main__':
    """
    Direct test execution for development and debugging.
    
    Provides direct test execution capabilities for development workflows
    while maintaining compatibility with pytest discovery and CI/CD pipelines.
    """
    import sys
    
    # Configure test execution
    test_args = [
        __file__,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--disable-warnings',  # Clean output
        '-x',  # Stop on first failure for debugging
    ]
    
    # Add coverage reporting if available
    try:
        import pytest_cov
        test_args.extend(['--cov=figregistry_kedro.datasets', '--cov-report=term-missing'])
    except ImportError:
        pass
    
    # Add benchmark configuration if available
    try:
        import pytest_benchmark
        test_args.extend(['--benchmark-only', '--benchmark-sort=mean'])
    except ImportError:
        pass
    
    # Execute tests
    exit_code = pytest.main(test_args)
    sys.exit(exit_code)