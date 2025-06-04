"""
Matplotlib figure generation fixtures providing various figure types, plot configurations, 
and graphics testing data for FigRegistry styling validation.

This module provides comprehensive matplotlib fixtures including:
- Sample plots (line, bar, scatter, subplots) for styling validation
- Figure object creation and manipulation fixtures  
- rcParams testing fixtures for style application validation
- Cross-platform graphics compatibility testing data
- Graphics state isolation for reliable test execution
- Performance benchmarking fixtures for overhead measurement

Fixtures ensure reliable test execution through matplotlib state isolation 
and cleanup between tests per Section 6.6.5.6.
"""

import copy
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Ensure matplotlib uses a non-interactive backend for testing
matplotlib.use('Agg')


class GraphicsStateManager:
    """
    Context manager for matplotlib state isolation between tests.
    
    Ensures complete matplotlib state reset and cleanup to prevent 
    cross-test contamination per Section 6.6.5.6 requirements.
    """
    
    def __init__(self):
        self.original_rcparams = None
        self.original_backend = None
        
    def __enter__(self):
        """Save current matplotlib state and prepare clean environment."""
        # Save original rcParams
        self.original_rcparams = matplotlib.rcParams.copy()
        self.original_backend = matplotlib.get_backend()
        
        # Reset to default state
        matplotlib.rcdefaults()
        matplotlib.use('Agg')
        
        # Clear any existing figures
        plt.close('all')
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original matplotlib state and cleanup."""
        # Close any figures created during test
        plt.close('all')
        
        # Restore original rcParams
        if self.original_rcparams is not None:
            matplotlib.rcParams.clear()
            matplotlib.rcParams.update(self.original_rcparams)
            
        # Restore original backend
        if self.original_backend is not None:
            matplotlib.use(self.original_backend)


@pytest.fixture(scope="function")
def graphics_state_isolation():
    """
    Provides matplotlib state isolation for individual test functions.
    
    Ensures each test starts with clean matplotlib state and performs
    complete cleanup upon completion, preventing state leakage between tests.
    
    Returns:
        GraphicsStateManager: Context manager for state isolation
    """
    with GraphicsStateManager() as manager:
        yield manager


@pytest.fixture(scope="function")
def clean_matplotlib_state(graphics_state_isolation):
    """
    Ensures matplotlib starts in a clean state for each test.
    
    This fixture automatically applies graphics state isolation and
    provides additional validation that matplotlib is properly reset.
    
    Returns:
        dict: Current matplotlib rcParams after cleanup
    """
    # Verify matplotlib is in clean state
    assert len(plt.get_fignums()) == 0, "Matplotlib figures not properly cleaned"
    
    # Return current rcParams for test inspection
    return matplotlib.rcParams.copy()


@pytest.fixture(scope="function")
def sample_data():
    """
    Provides sample datasets for figure generation across different plot types.
    
    Returns:
        dict: Collection of numpy arrays and pandas DataFrames for plotting
    """
    # Set random seed for reproducible test data
    np.random.seed(42)
    
    # Time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    time_series = np.cumsum(np.random.randn(100)) + 100
    
    # Categorical data
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    values = [23, 45, 56, 78, 32]
    
    # Scatter plot data
    x_scatter = np.random.randn(50)
    y_scatter = x_scatter * 2 + np.random.randn(50) * 0.5
    colors = np.random.rand(50)
    
    # Multi-dimensional data for subplots
    x_multi = np.linspace(0, 10, 100)
    y1_multi = np.sin(x_multi)
    y2_multi = np.cos(x_multi)
    y3_multi = np.tan(x_multi / 2)
    
    return {
        'time_series': {
            'dates': dates,
            'values': time_series
        },
        'categorical': {
            'categories': categories,
            'values': values
        },
        'scatter': {
            'x': x_scatter,
            'y': y_scatter,
            'colors': colors
        },
        'multi_dimensional': {
            'x': x_multi,
            'y1': y1_multi,
            'y2': y2_multi,
            'y3': y3_multi
        }
    }


@pytest.fixture(scope="function")
def sample_figure_fixtures(clean_matplotlib_state, sample_data):
    """
    Creates various matplotlib figure types for styling testing.
    
    Generates line plots, bar charts, scatter plots, and histograms
    with consistent data for reproducible styling validation.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        sample_data: Test data for figure generation
        
    Returns:
        dict: Collection of matplotlib figure objects by type
    """
    figures = {}
    
    # Line plot figure
    fig_line, ax_line = plt.subplots(figsize=(8, 6))
    ax_line.plot(sample_data['time_series']['dates'], 
                sample_data['time_series']['values'],
                label='Time Series Data')
    ax_line.set_title('Sample Line Plot')
    ax_line.set_xlabel('Date')
    ax_line.set_ylabel('Value')
    ax_line.legend()
    ax_line.grid(True)
    figures['line_plot'] = fig_line
    
    # Bar chart figure
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    bars = ax_bar.bar(sample_data['categorical']['categories'],
                      sample_data['categorical']['values'])
    ax_bar.set_title('Sample Bar Chart')
    ax_bar.set_xlabel('Categories')
    ax_bar.set_ylabel('Values')
    ax_bar.tick_params(axis='x', rotation=45)
    figures['bar_chart'] = fig_bar
    
    # Scatter plot figure
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    scatter = ax_scatter.scatter(sample_data['scatter']['x'],
                               sample_data['scatter']['y'],
                               c=sample_data['scatter']['colors'],
                               alpha=0.6)
    ax_scatter.set_title('Sample Scatter Plot')
    ax_scatter.set_xlabel('X Values')
    ax_scatter.set_ylabel('Y Values')
    plt.colorbar(scatter, ax=ax_scatter)
    figures['scatter_plot'] = fig_scatter
    
    # Histogram figure
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    ax_hist.hist(sample_data['scatter']['x'], bins=20, alpha=0.7, color='skyblue')
    ax_hist.set_title('Sample Histogram')
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True, alpha=0.3)
    figures['histogram'] = fig_hist
    
    return figures


@pytest.fixture(scope="function")
def subplot_fixtures(clean_matplotlib_state, sample_data):
    """
    Creates complex figure layouts with multiple subplots for testing.
    
    Generates multi-panel visualizations including 2x2 grids, shared axes,
    and complex layouts for comprehensive styling validation.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        sample_data: Test data for subplot generation
        
    Returns:
        dict: Collection of complex subplot figures
    """
    figures = {}
    
    # 2x2 subplot grid
    fig_grid, axes_grid = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Line plot
    axes_grid[0, 0].plot(sample_data['multi_dimensional']['x'],
                        sample_data['multi_dimensional']['y1'])
    axes_grid[0, 0].set_title('Sine Wave')
    axes_grid[0, 0].grid(True)
    
    # Top-right: Cosine plot
    axes_grid[0, 1].plot(sample_data['multi_dimensional']['x'],
                        sample_data['multi_dimensional']['y2'], 'r-')
    axes_grid[0, 1].set_title('Cosine Wave')
    axes_grid[0, 1].grid(True)
    
    # Bottom-left: Bar chart
    axes_grid[1, 0].bar(sample_data['categorical']['categories'][:3],
                       sample_data['categorical']['values'][:3])
    axes_grid[1, 0].set_title('Category Subset')
    axes_grid[1, 0].tick_params(axis='x', rotation=45)
    
    # Bottom-right: Scatter plot
    axes_grid[1, 1].scatter(sample_data['scatter']['x'][:25],
                           sample_data['scatter']['y'][:25])
    axes_grid[1, 1].set_title('Scatter Subset')
    
    plt.tight_layout()
    figures['grid_2x2'] = fig_grid
    
    # Shared axes figure
    fig_shared, (ax1_shared, ax2_shared) = plt.subplots(2, 1, 
                                                       figsize=(10, 8),
                                                       sharex=True)
    
    ax1_shared.plot(sample_data['multi_dimensional']['x'],
                   sample_data['multi_dimensional']['y1'], 'b-', label='Sine')
    ax1_shared.plot(sample_data['multi_dimensional']['x'],
                   sample_data['multi_dimensional']['y2'], 'r-', label='Cosine')
    ax1_shared.set_title('Trigonometric Functions')
    ax1_shared.legend()
    ax1_shared.grid(True)
    
    ax2_shared.plot(sample_data['multi_dimensional']['x'],
                   sample_data['multi_dimensional']['y3'], 'g-', label='Tangent')
    ax2_shared.set_title('Tangent Function')
    ax2_shared.set_xlabel('X Values')
    ax2_shared.legend()
    ax2_shared.grid(True)
    
    plt.tight_layout()
    figures['shared_axes'] = fig_shared
    
    # Complex mixed layout
    fig_complex = plt.figure(figsize=(14, 10))
    
    # Large subplot spanning top
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax_main.plot(sample_data['time_series']['dates'],
                sample_data['time_series']['values'])
    ax_main.set_title('Main Time Series')
    ax_main.grid(True)
    
    # Bottom left subplot
    ax_bl = plt.subplot2grid((3, 3), (1, 0), rowspan=2)
    ax_bl.bar(sample_data['categorical']['categories'],
             sample_data['categorical']['values'])
    ax_bl.set_title('Categories')
    ax_bl.tick_params(axis='x', rotation=90)
    
    # Bottom middle subplot
    ax_bm = plt.subplot2grid((3, 3), (1, 1))
    ax_bm.hist(sample_data['scatter']['x'], bins=15)
    ax_bm.set_title('Distribution')
    
    # Bottom right subplot
    ax_br = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax_br.scatter(sample_data['scatter']['x'],
                 sample_data['scatter']['y'])
    ax_br.set_title('Correlation')
    
    # Small subplot
    ax_small = plt.subplot2grid((3, 3), (2, 1))
    ax_small.pie([1, 2, 3], labels=['A', 'B', 'C'])
    ax_small.set_title('Pie Chart')
    
    plt.tight_layout()
    figures['complex_layout'] = fig_complex
    
    return figures


@pytest.fixture(scope="function")
def rcparams_testing_fixtures(clean_matplotlib_state):
    """
    Provides rcParams configurations for testing style application.
    
    Creates sets of matplotlib parameters for validating that FigRegistry
    styling operations correctly apply and modify rcParams during figure
    generation and saving operations.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        
    Returns:
        dict: Collection of rcParams configurations for testing
    """
    # Default matplotlib rcParams (baseline)
    default_params = matplotlib.rcParams.copy()
    
    # Scientific publication style
    publication_params = {
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif'
    }
    
    # Presentation style
    presentation_params = {
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 24,
        'lines.linewidth': 3.0,
        'axes.linewidth': 2.0,
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.weight': 'bold'
    }
    
    # Exploratory analysis style
    exploratory_params = {
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 14,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.7,
        'font.family': 'monospace',
        'axes.facecolor': '#f8f8f8'
    }
    
    # High contrast style for accessibility
    high_contrast_params = {
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2.5,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.8,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'text.color': 'black'
    }
    
    return {
        'default': default_params,
        'publication': publication_params,
        'presentation': presentation_params,
        'exploratory': exploratory_params,
        'high_contrast': high_contrast_params
    }


@pytest.fixture(scope="function")
def figure_format_fixtures(clean_matplotlib_state):
    """
    Provides figure format specifications for cross-platform testing.
    
    Creates format configurations for PNG, PDF, SVG output testing
    ensuring consistent behavior across operating systems and matplotlib
    backends during FigRegistry save operations.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        
    Returns:
        dict: Format specifications and parameters for testing
    """
    return {
        'png': {
            'format': 'png',
            'dpi': 300,
            'transparent': False,
            'facecolor': 'white',
            'edgecolor': 'none',
            'bbox_inches': 'tight',
            'pad_inches': 0.1
        },
        'png_transparent': {
            'format': 'png',
            'dpi': 300,
            'transparent': True,
            'facecolor': 'none',
            'edgecolor': 'none',
            'bbox_inches': 'tight',
            'pad_inches': 0.0
        },
        'pdf': {
            'format': 'pdf',
            'dpi': 300,
            'transparent': False,
            'facecolor': 'white',
            'edgecolor': 'none',
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'metadata': {
                'Title': 'FigRegistry Test Output',
                'Subject': 'Automated Figure Generation',
                'Creator': 'FigRegistry-Kedro Plugin'
            }
        },
        'svg': {
            'format': 'svg',
            'transparent': False,
            'facecolor': 'white',
            'edgecolor': 'none',
            'bbox_inches': 'tight',
            'pad_inches': 0.0,
            'metadata': {
                'Creator': 'FigRegistry-Kedro Plugin'
            }
        },
        'pdf_vector': {
            'format': 'pdf',
            'dpi': 'figure',  # Use figure's native DPI
            'transparent': True,
            'facecolor': 'none',
            'edgecolor': 'none',
            'bbox_inches': None,  # Use figure's original bbox
            'metadata': {
                'Title': 'Vector Output Test',
                'Creator': 'FigRegistry-Kedro Plugin'
            }
        }
    }


@pytest.fixture(scope="function")
def performance_figure_fixtures(clean_matplotlib_state):
    """
    Creates figures for benchmarking FigRegistry performance overhead.
    
    Generates figures of varying complexity for measuring styling and
    save operation performance, enabling validation that plugin overhead
    remains within acceptable thresholds.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        
    Returns:
        dict: Collection of figures with performance timing data
    """
    figures = {}
    
    # Simple figure for baseline performance
    start_time = time.perf_counter()
    fig_simple, ax_simple = plt.subplots(figsize=(6, 4))
    x_simple = np.linspace(0, 10, 100)
    y_simple = np.sin(x_simple)
    ax_simple.plot(x_simple, y_simple)
    ax_simple.set_title('Simple Performance Test')
    creation_time_simple = time.perf_counter() - start_time
    
    figures['simple'] = {
        'figure': fig_simple,
        'creation_time': creation_time_simple,
        'complexity': 'low',
        'elements': 100
    }
    
    # Medium complexity figure
    start_time = time.perf_counter()
    fig_medium, axes_medium = plt.subplots(2, 2, figsize=(10, 8))
    
    for i, ax in enumerate(axes_medium.flat):
        x = np.linspace(0, 10, 500)
        y = np.sin(x + i) * np.cos(x / 2)
        ax.plot(x, y, linewidth=1.5)
        ax.set_title(f'Subplot {i+1}')
        ax.grid(True, alpha=0.3)
        ax.legend([f'Series {i+1}'])
    
    plt.tight_layout()
    creation_time_medium = time.perf_counter() - start_time
    
    figures['medium'] = {
        'figure': fig_medium,
        'creation_time': creation_time_medium,
        'complexity': 'medium',
        'elements': 2000
    }
    
    # Complex figure for stress testing
    start_time = time.perf_counter()
    fig_complex, ax_complex = plt.subplots(figsize=(12, 9))
    
    # Multiple data series
    x_complex = np.linspace(0, 20, 2000)
    for i in range(10):
        y = np.sin(x_complex + i * 0.5) * np.exp(-x_complex / 20)
        ax_complex.plot(x_complex, y, alpha=0.7, linewidth=2,
                       label=f'Series {i+1}')
    
    # Add scatter overlay
    x_scatter = np.random.uniform(0, 20, 500)
    y_scatter = np.random.uniform(-1, 1, 500)
    ax_complex.scatter(x_scatter, y_scatter, alpha=0.3, s=10)
    
    # Styling elements
    ax_complex.set_title('Complex Performance Test Figure')
    ax_complex.set_xlabel('X Values')
    ax_complex.set_ylabel('Y Values')
    ax_complex.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_complex.grid(True, alpha=0.2)
    
    plt.tight_layout()
    creation_time_complex = time.perf_counter() - start_time
    
    figures['complex'] = {
        'figure': fig_complex,
        'creation_time': creation_time_complex,
        'complexity': 'high',
        'elements': 20500
    }
    
    return figures


@pytest.fixture(scope="function")
def cross_platform_graphics_fixtures(clean_matplotlib_state):
    """
    Provides graphics configurations for cross-platform compatibility testing.
    
    Creates test scenarios that validate consistent matplotlib behavior
    across Windows, macOS, and Linux environments, ensuring FigRegistry
    plugin operations produce identical results regardless of platform.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        
    Returns:
        dict: Platform-specific configurations and test data
    """
    # Font configurations that should work across platforms
    cross_platform_fonts = {
        'serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'serif'],
        'sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif'],
        'monospace': ['DejaVu Sans Mono', 'Liberation Mono', 'Courier New', 'monospace']
    }
    
    # Color specifications that render consistently
    standard_colors = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }
    
    # DPI settings for different platforms
    platform_dpi_settings = {
        'windows_standard': 96,
        'windows_high_dpi': 144,
        'macos_retina': 144,
        'linux_standard': 96,
        'linux_high_dpi': 192
    }
    
    # Backend-specific configurations
    backend_configs = {
        'agg': {
            'backend': 'Agg',
            'supports_transparency': True,
            'vector_output': False,
            'interactive': False
        },
        'svg': {
            'backend': 'SVG',
            'supports_transparency': True,
            'vector_output': True,
            'interactive': False
        },
        'pdf': {
            'backend': 'PDF',
            'supports_transparency': True,
            'vector_output': True,
            'interactive': False
        }
    }
    
    # Create test figure for platform validation
    fig_platform, ax_platform = plt.subplots(figsize=(8, 6))
    
    # Use cross-platform compatible elements
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    line1 = ax_platform.plot(x, y1, color=standard_colors['primary'], 
                            linewidth=2, label='Sin(x)')
    line2 = ax_platform.plot(x, y2, color=standard_colors['secondary'], 
                            linewidth=2, label='Cos(x)')
    
    ax_platform.set_title('Cross-Platform Compatibility Test')
    ax_platform.set_xlabel('X Values')
    ax_platform.set_ylabel('Y Values')
    ax_platform.legend()
    ax_platform.grid(True, alpha=0.3)
    
    return {
        'fonts': cross_platform_fonts,
        'colors': standard_colors,
        'dpi_settings': platform_dpi_settings,
        'backend_configs': backend_configs,
        'test_figure': fig_platform,
        'validation_data': {
            'x': x,
            'y1': y1,
            'y2': y2
        }
    }


@pytest.fixture(scope="function")
def temporary_output_directory(tmp_path):
    """
    Provides temporary directory for figure output testing.
    
    Creates isolated temporary directories for each test to ensure
    clean file system state and prevent cross-test contamination
    during figure save operations.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Path: Temporary directory path for figure outputs
    """
    # Create subdirectory structure for organized testing
    output_dir = tmp_path / "figure_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Create format-specific subdirectories
    (output_dir / "png").mkdir(exist_ok=True)
    (output_dir / "pdf").mkdir(exist_ok=True)
    (output_dir / "svg").mkdir(exist_ok=True)
    
    return output_dir


@pytest.fixture(scope="function")
def figure_save_benchmark():
    """
    Provides timing utilities for benchmarking figure save operations.
    
    Returns timing functions and performance thresholds for validating
    that FigRegistry styling and save operations meet performance
    requirements across different figure types and output formats.
    
    Returns:
        dict: Benchmarking utilities and performance thresholds
    """
    class SaveBenchmark:
        def __init__(self):
            self.save_times = []
            self.style_times = []
            
        def time_save_operation(self, figure, filepath, **save_kwargs):
            """Time a complete figure save operation."""
            start_time = time.perf_counter()
            figure.savefig(filepath, **save_kwargs)
            end_time = time.perf_counter()
            
            save_time = end_time - start_time
            self.save_times.append(save_time)
            return save_time
            
        def time_style_application(self, style_params):
            """Time matplotlib rcParams application."""
            start_time = time.perf_counter()
            for param, value in style_params.items():
                plt.rcParams[param] = value
            end_time = time.perf_counter()
            
            style_time = end_time - start_time
            self.style_times.append(style_time)
            return style_time
            
        def get_statistics(self):
            """Calculate performance statistics."""
            return {
                'save_times': {
                    'count': len(self.save_times),
                    'mean': np.mean(self.save_times) if self.save_times else 0,
                    'median': np.median(self.save_times) if self.save_times else 0,
                    'max': np.max(self.save_times) if self.save_times else 0,
                    'min': np.min(self.save_times) if self.save_times else 0
                },
                'style_times': {
                    'count': len(self.style_times),
                    'mean': np.mean(self.style_times) if self.style_times else 0,
                    'median': np.median(self.style_times) if self.style_times else 0,
                    'max': np.max(self.style_times) if self.style_times else 0,
                    'min': np.min(self.style_times) if self.style_times else 0
                }
            }
    
    # Performance thresholds from technical specification
    thresholds = {
        'max_save_time': 0.5,  # 500ms per figure
        'max_style_time': 0.01,  # 10ms per style application
        'max_plugin_overhead': 0.2,  # 200ms for FigureDataSet save
        'max_config_bridge_time': 0.05  # 50ms for config bridge resolution
    }
    
    return {
        'benchmark': SaveBenchmark(),
        'thresholds': thresholds
    }


@pytest.fixture(scope="function")
def matplotlib_error_scenarios(clean_matplotlib_state):
    """
    Provides error scenario configurations for robust testing.
    
    Creates invalid figure configurations and malformed parameters
    for testing error handling in FigRegistry styling operations
    and ensuring graceful degradation under error conditions.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        
    Returns:
        dict: Error scenario configurations and test cases
    """
    return {
        'invalid_rcparams': {
            'font.size': 'invalid_size',  # Should be numeric
            'figure.figsize': 'not_a_tuple',  # Should be tuple
            'axes.linewidth': -1,  # Should be positive
            'nonexistent.param': 'value'  # Parameter doesn't exist
        },
        'invalid_save_params': {
            'dpi': 'not_numeric',
            'format': 'unsupported_format',
            'transparent': 'not_boolean',
            'bbox_inches': 'invalid_bbox'
        },
        'malformed_figure_data': {
            'empty_data': ([], []),
            'mismatched_lengths': ([1, 2, 3], [1, 2]),
            'non_numeric_data': (['a', 'b', 'c'], [1, 2, 3]),
            'infinite_values': ([1, 2, np.inf], [1, 2, 3]),
            'nan_values': ([1, np.nan, 3], [1, 2, 3])
        },
        'resource_constraints': {
            'excessive_figure_size': (100, 100),  # Very large figure
            'excessive_dpi': 10000,  # Unreasonably high DPI
            'memory_intensive_data': np.random.random((10000, 10000))
        }
    }


# Additional utility fixtures for comprehensive testing

@pytest.fixture(scope="session", autouse=True)
def matplotlib_test_setup():
    """
    Session-level setup for matplotlib testing environment.
    
    Configures matplotlib for optimal testing behavior and
    ensures consistent behavior across all test sessions.
    """
    # Configure matplotlib for testing
    matplotlib.use('Agg')  # Non-interactive backend
    plt.ioff()  # Turn off interactive mode
    
    # Set reasonable defaults for testing
    plt.rcParams['figure.max_open_warning'] = 0  # Disable warnings for many figures
    plt.rcParams['font.size'] = 10  # Consistent font size
    
    yield
    
    # Cleanup after all tests
    plt.close('all')


@pytest.fixture(scope="function", autouse=True)
def figure_cleanup():
    """
    Function-level automatic cleanup for matplotlib figures.
    
    Ensures that each test function starts and ends with no
    open matplotlib figures, preventing memory leaks and
    cross-test contamination.
    """
    # Pre-test cleanup
    plt.close('all')
    
    yield
    
    # Post-test cleanup
    plt.close('all')


@pytest.fixture(scope="function")
def kedro_compatible_figures(sample_figure_fixtures):
    """
    Provides figures specifically configured for Kedro integration testing.
    
    Creates figures that match expected patterns for FigureDataSet
    usage in Kedro pipelines, including proper metadata and configuration
    for catalog integration testing.
    
    Args:
        sample_figure_fixtures: Basic figure fixtures
        
    Returns:
        dict: Kedro-compatible figure objects with metadata
    """
    kedro_figures = {}
    
    for fig_type, figure in sample_figure_fixtures.items():
        # Add metadata that FigureDataSet might use
        figure._figregistry_metadata = {
            'purpose': 'testing',
            'condition_param': 'test_condition',
            'created_by': 'pytest',
            'figure_type': fig_type
        }
        
        kedro_figures[fig_type] = {
            'figure': figure,
            'catalog_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': f'data/08_reporting/{fig_type}.png',
                'purpose': 'testing',
                'condition_param': 'experiment_type',
                'style_params': {
                    'font.size': 12,
                    'figure.dpi': 300
                }
            }
        }
    
    return kedro_figures