"""
Performance testing data generators and utilities for figregistry-kedro plugin.

This module provides comprehensive data generation functions and utilities for performance 
testing and benchmarking of figregistry-kedro plugin operations. It supports validation 
of plugin overhead targets including <200ms FigureDataSet overhead, <50ms configuration 
bridge resolution, and <25ms hook initialization per technical specification requirements.

Key Performance Targets (per Section 6.6.4.3):
- Configuration Bridge Merge Time: < 50ms per pipeline run
- FigureDataSet Save Overhead: < 200ms per save operation
- Hook Initialization Overhead: < 25ms per project startup
- Plugin Memory Overhead: < 5MB total footprint
- Pipeline Execution Overhead: < 200ms per FigureDataSet save

The module is structured to support comprehensive performance validation across:
- Large-scale configuration scenarios for testing bridge merge performance
- High-volume catalog entries for concurrent execution testing
- Complex figure objects for dataset save performance benchmarking
- Concurrent execution scenarios for parallel Kedro runner validation
- Memory usage patterns for plugin footprint validation
- Stress testing data for high-load scenario validation

Dependencies:
- numpy: For data generation and mathematical operations
- matplotlib: For complex figure generation and performance testing
- pandas: For structured data creation and manipulation
- scipy: For scientific computing patterns in test data
- typing: For comprehensive type hints and annotations
"""

import time
import psutil
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Generator, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Performance measurement utilities
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# Type definitions for performance testing
ConfigDict = Dict[str, Any]
CatalogConfig = Dict[str, Dict[str, Any]]
FigureObject = matplotlib.figure.Figure
TimingResult = Dict[str, float]
MemoryResult = Dict[str, Union[float, int]]
PerformanceMetrics = Dict[str, Union[float, int, str]]


# =============================================================================
# PERFORMANCE BASELINE AND TARGET DEFINITIONS
# =============================================================================

@dataclass
class PerformanceTargets:
    """
    Performance targets for figregistry-kedro plugin operations.
    
    These targets align with technical specification requirements in Section 6.6.4.3
    and ensure plugin integration maintains acceptable overhead for scientific
    computing workflows.
    """
    # Configuration bridge performance targets
    config_bridge_merge_time_ms: float = 50.0  # Maximum merge time per pipeline run
    config_load_time_ms: float = 100.0  # Maximum configuration loading time
    config_validation_time_ms: float = 25.0  # Maximum validation time
    
    # FigureDataSet performance targets  
    figure_dataset_save_overhead_ms: float = 200.0  # Maximum save overhead
    style_resolution_time_ms: float = 10.0  # Maximum style resolution time
    figure_processing_time_ms: float = 100.0  # Maximum figure processing time
    
    # Hook initialization performance targets
    hook_initialization_time_ms: float = 25.0  # Maximum hook init time
    hook_registration_time_ms: float = 5.0  # Maximum registration time
    hook_execution_overhead_ms: float = 5.0  # Maximum per-hook execution overhead
    
    # Memory usage targets
    plugin_memory_overhead_mb: float = 5.0  # Maximum plugin memory footprint
    config_cache_memory_mb: float = 2.0  # Maximum configuration cache memory
    per_figure_memory_overhead_mb: float = 0.5  # Maximum per-figure memory overhead
    
    # Pipeline execution targets
    pipeline_execution_overhead_ms: float = 200.0  # Maximum per-FigureDataSet overhead
    concurrent_execution_degradation_percent: float = 10.0  # Maximum performance degradation


@dataclass  
class StressTestLimits:
    """
    Stress testing limits for validating plugin behavior under high-load scenarios.
    
    These limits define the boundary conditions for stress testing to ensure
    plugin reliability and graceful degradation under extreme conditions.
    """
    # Configuration stress limits
    max_config_size_mb: float = 10.0  # Maximum configuration file size
    max_style_conditions: int = 1000  # Maximum number of style conditions
    max_config_nesting_levels: int = 10  # Maximum configuration nesting depth
    
    # Catalog stress limits
    max_catalog_entries: int = 500  # Maximum FigureDataSet catalog entries
    max_concurrent_saves: int = 20  # Maximum concurrent figure save operations
    max_parallel_pipelines: int = 8  # Maximum parallel pipeline executions
    
    # Figure complexity limits
    max_figure_subplots: int = 20  # Maximum subplots per figure
    max_data_points_per_plot: int = 100000  # Maximum data points per plot
    max_figure_memory_mb: float = 50.0  # Maximum figure memory usage
    
    # Memory stress limits
    max_total_memory_mb: float = 1000.0  # Maximum total memory during stress test
    memory_leak_tolerance_mb: float = 5.0  # Acceptable memory increase per operation


# Global performance targets instance
PERFORMANCE_TARGETS = PerformanceTargets()
STRESS_TEST_LIMITS = StressTestLimits()


# =============================================================================
# LARGE CONFIGURATION GENERATORS
# =============================================================================

def generate_large_figregistry_config(
    num_conditions: int = 100,
    num_style_properties: int = 50,
    config_complexity: str = "medium"
) -> ConfigDict:
    """
    Generate large-scale FigRegistry configuration for testing configuration bridge merge performance.
    
    Creates complex configuration scenarios with numerous conditions, style mappings,
    and nested structures to validate that configuration bridge merge operations
    remain within the 50ms target per Section 6.6.4.3.
    
    Args:
        num_conditions: Number of experimental conditions to generate
        num_style_properties: Number of style properties per condition
        config_complexity: Complexity level ('simple', 'medium', 'complex')
        
    Returns:
        Large configuration dictionary suitable for bridge merge testing
        
    Examples:
        >>> config = generate_large_figregistry_config(num_conditions=200)
        >>> len(config['conditions'])
        200
        >>> 'styles' in config and 'outputs' in config
        True
    """
    complexity_multipliers = {
        'simple': 1.0,
        'medium': 2.0,  
        'complex': 5.0
    }
    
    multiplier = complexity_multipliers.get(config_complexity, 2.0)
    effective_conditions = int(num_conditions * multiplier)
    effective_properties = int(num_style_properties * multiplier)
    
    # Generate base style properties
    base_style_properties = [
        'figure.figsize', 'axes.grid', 'axes.grid.alpha', 'font.size',
        'axes.labelsize', 'axes.titlesize', 'legend.fontsize', 'xtick.labelsize',
        'ytick.labelsize', 'axes.linewidth', 'grid.linewidth', 'lines.linewidth',
        'figure.facecolor', 'axes.facecolor', 'axes.edgecolor', 'text.color',
        'axes.labelcolor', 'xtick.color', 'ytick.color', 'legend.frameon',
        'legend.fancybox', 'legend.shadow', 'axes.spines.top', 'axes.spines.right',
        'axes.spines.bottom', 'axes.spines.left', 'savefig.dpi', 'savefig.bbox',
        'savefig.pad_inches', 'savefig.facecolor', 'savefig.edgecolor'
    ]
    
    # Extend with generated properties if needed
    if effective_properties > len(base_style_properties):
        for i in range(len(base_style_properties), effective_properties):
            base_style_properties.append(f'custom.property_{i}')
    
    style_properties = base_style_properties[:effective_properties]
    
    # Generate styles section
    styles = {}
    for i in range(effective_conditions):
        condition_name = f"condition_{i}"
        style_dict = {}
        
        for j, prop in enumerate(style_properties):
            if 'figsize' in prop:
                style_dict[prop] = [8 + (i % 4), 6 + (i % 3)]
            elif 'size' in prop and 'fig' not in prop:
                style_dict[prop] = 10 + (i % 8)
            elif 'alpha' in prop or 'width' in prop:
                style_dict[prop] = 0.1 + (i % 10) * 0.1
            elif 'color' in prop:
                colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown']
                style_dict[prop] = colors[i % len(colors)]
            elif prop.endswith('on') or 'grid' in prop:
                style_dict[prop] = bool(i % 2)
            elif 'dpi' in prop:
                style_dict[prop] = 100 + (i % 5) * 50
            elif 'pad' in prop:
                style_dict[prop] = 0.1 + (i % 5) * 0.05
            else:
                style_dict[prop] = f"value_{i}_{j}"
        
        styles[condition_name] = style_dict
    
    # Generate conditions section with pattern mapping
    conditions = {}
    for i in range(effective_conditions):
        experiment_type = f"experiment_type_{i}"
        condition_mapping = {}
        
        for j in range(min(5, effective_conditions // 10)):  # Up to 5 values per condition
            key = f"value_{j}"
            target_condition = f"condition_{(i + j) % effective_conditions}"
            condition_mapping[key] = target_condition
            
        conditions[experiment_type] = condition_mapping
    
    # Generate outputs section
    outputs = {
        'base_path': f'data/08_reporting/figures/performance_test_{effective_conditions}',
        'timestamp_format': '%Y%m%d_%H%M%S_%f',
        'path_aliases': {
            f'alias_{i}': f'condition_{i}' for i in range(min(20, effective_conditions))
        },
        'versioning': {
            'enabled': True,
            'strategy': 'timestamp',
            'format': 'iso8601'
        }
    }
    
    # Add nested configuration structures for complexity testing
    if config_complexity == 'complex':
        outputs['advanced_settings'] = {
            'performance_monitoring': {
                'enabled': True,
                'metrics': ['timing', 'memory', 'file_size'],
                'thresholds': {
                    'max_save_time_ms': 200,
                    'max_memory_mb': 5,
                    'max_file_size_mb': 10
                }
            },
            'cache_settings': {
                'style_cache_size': 1000,
                'config_cache_ttl': 3600,
                'memory_cache_limit_mb': 50
            },
            'integration_settings': {
                'kedro_compatibility': {
                    'version_range': '>=0.18.0,<0.20.0',
                    'hook_priority': 100,
                    'dataset_registration': 'automatic'
                }
            }
        }
    
    return {
        'styles': styles,
        'conditions': conditions, 
        'outputs': outputs,
        'metadata': {
            'generated_conditions': effective_conditions,
            'generated_properties': effective_properties,
            'complexity_level': config_complexity,
            'generation_timestamp': time.time()
        }
    }


def generate_kedro_environment_configs(
    base_config: ConfigDict,
    num_environments: int = 5
) -> Dict[str, ConfigDict]:
    """
    Generate multiple Kedro environment-specific configurations for merge testing.
    
    Creates environment-specific configuration overrides that test the configuration
    bridge's ability to efficiently merge multiple configuration sources while
    maintaining the 50ms merge time target.
    
    Args:
        base_config: Base FigRegistry configuration to override
        num_environments: Number of environment configurations to generate
        
    Returns:
        Dictionary mapping environment names to configuration overrides
        
    Examples:
        >>> base = generate_large_figregistry_config(num_conditions=50)
        >>> envs = generate_kedro_environment_configs(base, num_environments=3)
        >>> 'local' in envs and 'staging' in envs and 'production' in envs
        True
    """
    environments = ['local', 'staging', 'production', 'testing', 'development']
    environment_configs = {}
    
    for i in range(min(num_environments, len(environments))):
        env_name = environments[i]
        
        # Create environment-specific overrides
        env_config = {
            'outputs': {
                'base_path': f'data/08_reporting/figures/{env_name}',
                'timestamp_format': base_config['outputs']['timestamp_format'],
                'environment': env_name
            },
            'styles': {},
            'conditions': {}
        }
        
        # Override subset of styles for environment-specific behavior
        base_styles = base_config.get('styles', {})
        for j, (style_name, style_dict) in enumerate(list(base_styles.items())[:10]):
            if j % (i + 1) == 0:  # Vary which styles get overridden
                env_style = style_dict.copy()
                
                # Environment-specific modifications
                if env_name == 'production':
                    env_style['figure.figsize'] = [12, 8]  # Larger production figures
                    env_style['savefig.dpi'] = 300  # Higher DPI for production
                elif env_name == 'local':
                    env_style['figure.figsize'] = [8, 6]  # Smaller local figures
                    env_style['savefig.dpi'] = 100  # Lower DPI for local development
                elif env_name == 'staging':
                    env_style['axes.grid'] = True  # Always show grid in staging
                    env_style['axes.grid.alpha'] = 0.5
                
                env_config['styles'][style_name] = env_style
        
        # Add environment-specific conditions
        env_config['conditions'][f'{env_name}_experiments'] = {
            'baseline': f'condition_{i}',
            'optimized': f'condition_{i + 10}',
            'final': f'condition_{i + 20}'
        }
        
        environment_configs[env_name] = env_config
    
    return environment_configs


def generate_complex_merge_scenarios(
    base_conditions: int = 100,
    merge_complexity: str = "high"
) -> List[Tuple[ConfigDict, List[ConfigDict]]]:
    """
    Generate complex configuration merge scenarios for stress testing bridge performance.
    
    Creates challenging merge scenarios with overlapping keys, nested structures,
    and type conflicts to validate configuration bridge robustness and performance
    under complex merge operations.
    
    Args:
        base_conditions: Number of base conditions in primary config
        merge_complexity: Complexity level for merge scenarios
        
    Returns:
        List of (base_config, override_configs) tuples for merge testing
        
    Examples:
        >>> scenarios = generate_complex_merge_scenarios(base_conditions=50)
        >>> len(scenarios) > 0
        True
        >>> all(len(scenario[1]) > 0 for scenario in scenarios)
        True
    """
    scenarios = []
    
    complexity_settings = {
        'low': {'num_scenarios': 3, 'overrides_per_scenario': 2, 'conflict_rate': 0.1},
        'medium': {'num_scenarios': 5, 'overrides_per_scenario': 3, 'conflict_rate': 0.3},
        'high': {'num_scenarios': 8, 'overrides_per_scenario': 5, 'conflict_rate': 0.5}
    }
    
    settings = complexity_settings.get(merge_complexity, complexity_settings['medium'])
    
    for scenario_idx in range(settings['num_scenarios']):
        # Generate base configuration
        base_config = generate_large_figregistry_config(
            num_conditions=base_conditions,
            config_complexity='medium'
        )
        
        override_configs = []
        
        for override_idx in range(settings['overrides_per_scenario']):
            override_config = {'styles': {}, 'conditions': {}, 'outputs': {}}
            
            # Create overlapping style definitions with conflicts
            base_styles = base_config.get('styles', {})
            for style_name, style_dict in list(base_styles.items())[:20]:
                if np.random.random() < settings['conflict_rate']:
                    # Create conflicting override
                    conflicting_style = {}
                    for prop, value in style_dict.items():
                        if 'figsize' in prop and isinstance(value, list):
                            conflicting_style[prop] = [value[0] + 2, value[1] + 1]
                        elif isinstance(value, (int, float)):
                            conflicting_style[prop] = value * 1.5
                        elif isinstance(value, bool):
                            conflicting_style[prop] = not value
                        elif isinstance(value, str):
                            conflicting_style[prop] = f"override_{value}"
                        else:
                            conflicting_style[prop] = value
                    
                    override_config['styles'][style_name] = conflicting_style
            
            # Add new style definitions
            for new_idx in range(5):
                new_style_name = f"override_style_{scenario_idx}_{override_idx}_{new_idx}"
                override_config['styles'][new_style_name] = {
                    'figure.figsize': [10 + new_idx, 8 + new_idx],
                    'font.size': 12 + new_idx,
                    'axes.grid': bool(new_idx % 2)
                }
            
            # Create complex nested structure conflicts
            override_config['outputs'] = {
                'base_path': f'override_path_{scenario_idx}_{override_idx}',
                'advanced_settings': {
                    'performance_monitoring': {
                        'enabled': False,  # Conflict with base
                        'custom_metrics': [f'metric_{i}' for i in range(override_idx + 1)]
                    }
                }
            }
            
            override_configs.append(override_config)
        
        scenarios.append((base_config, override_configs))
    
    return scenarios


# =============================================================================
# HIGH VOLUME CATALOG GENERATORS  
# =============================================================================

def generate_high_volume_catalog_config(
    num_figure_datasets: int = 100,
    concurrent_access_pattern: str = "mixed"
) -> CatalogConfig:
    """
    Generate high-volume Kedro catalog configurations with multiple FigureDataSet entries.
    
    Creates large catalog configurations for testing concurrent execution performance
    and validating that plugin operations scale efficiently with catalog size while
    maintaining per-dataset performance targets.
    
    Args:
        num_figure_datasets: Number of FigureDataSet entries to generate
        concurrent_access_pattern: Access pattern ('sequential', 'parallel', 'mixed')
        
    Returns:
        Kedro catalog configuration with multiple FigureDataSet entries
        
    Examples:
        >>> catalog = generate_high_volume_catalog_config(num_figure_datasets=50)
        >>> len([k for k in catalog.keys() if 'figure' in k.lower()]) == 50
        True
    """
    catalog_config = {}
    
    # Define figure output categories
    categories = ['exploratory', 'presentation', 'publication', 'diagnostic', 'comparison']
    formats = ['png', 'pdf', 'svg', 'eps']
    condition_params = ['experiment_type', 'analysis_mode', 'data_version', 'model_variant']
    
    for i in range(num_figure_datasets):
        dataset_name = f"figure_dataset_{i:03d}"
        category = categories[i % len(categories)]
        format_ext = formats[i % len(formats)]
        condition_param = condition_params[i % len(condition_params)]
        
        # Create dataset configuration
        dataset_config = {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': f'data/08_reporting/figures/{category}/figure_{i:03d}.{format_ext}',
            'purpose': category,
            'condition_param': condition_param,
            'save_args': {
                'dpi': 150 + (i % 3) * 50,  # Vary DPI: 150, 200, 250
                'bbox_inches': 'tight',
                'transparent': bool(i % 2)
            }
        }
        
        # Add versioning for subset of datasets
        if i % 3 == 0:
            dataset_config['versioned'] = True
            
        # Add style parameter overrides for complexity
        if i % 5 == 0:
            dataset_config['style_params'] = {
                'figure.figsize': [8 + (i % 4), 6 + (i % 3)],
                'font.size': 10 + (i % 6),
                'axes.grid': bool(i % 2)
            }
            
        # Configure for concurrent access patterns
        if concurrent_access_pattern == "parallel":
            dataset_config['load_args'] = {'thread_safe': True}
        elif concurrent_access_pattern == "mixed":
            if i % 2 == 0:
                dataset_config['load_args'] = {'thread_safe': True}
        
        catalog_config[dataset_name] = dataset_config
    
    # Add non-figure datasets for realistic catalog simulation
    for i in range(num_figure_datasets // 5):  # 20% non-figure datasets
        dataset_name = f"data_input_{i:03d}"
        catalog_config[dataset_name] = {
            'type': 'pandas.CSVDataSet',
            'filepath': f'data/01_raw/input_{i:03d}.csv'
        }
        
        dataset_name = f"processed_data_{i:03d}"
        catalog_config[dataset_name] = {
            'type': 'pandas.ParquetDataSet', 
            'filepath': f'data/03_primary/processed_{i:03d}.parquet'
        }
    
    return catalog_config


def generate_concurrent_catalog_scenarios(
    base_catalog_size: int = 50,
    concurrency_levels: List[int] = None
) -> List[Tuple[CatalogConfig, int]]:
    """
    Generate catalog configurations for concurrent execution testing.
    
    Creates multiple catalog scenarios with different concurrency characteristics
    to validate plugin performance under parallel Kedro runner execution patterns.
    
    Args:
        base_catalog_size: Base number of datasets in each catalog
        concurrency_levels: List of concurrency levels to test
        
    Returns:
        List of (catalog_config, concurrency_level) tuples for testing
        
    Examples:
        >>> scenarios = generate_concurrent_catalog_scenarios(base_catalog_size=20)
        >>> len(scenarios) > 0
        True
        >>> all(isinstance(scenario[1], int) for scenario in scenarios)
        True
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]
    
    scenarios = []
    
    for concurrency_level in concurrency_levels:
        # Adjust catalog size based on concurrency level
        adjusted_catalog_size = min(base_catalog_size * concurrency_level, 
                                   STRESS_TEST_LIMITS.max_catalog_entries)
        
        # Generate catalog with concurrency-appropriate configuration
        catalog_config = generate_high_volume_catalog_config(
            num_figure_datasets=adjusted_catalog_size,
            concurrent_access_pattern="parallel" if concurrency_level > 1 else "sequential"
        )
        
        # Add concurrency-specific configuration
        for dataset_name, dataset_config in catalog_config.items():
            if 'figregistry_kedro.datasets.FigureDataSet' in dataset_config.get('type', ''):
                # Add concurrency hints
                dataset_config['metadata'] = {
                    'concurrency_level': concurrency_level,
                    'thread_safety': concurrency_level > 1,
                    'performance_priority': 'throughput' if concurrency_level > 4 else 'latency'
                }
        
        scenarios.append((catalog_config, concurrency_level))
    
    return scenarios


def generate_catalog_stress_scenarios(
    max_datasets: int = None
) -> List[CatalogConfig]:
    """
    Generate stress test catalog configurations for validating plugin limits.
    
    Creates increasingly large catalog configurations to identify performance
    boundaries and validate graceful degradation under extreme load conditions.
    
    Args:
        max_datasets: Maximum number of datasets (defaults to stress test limit)
        
    Returns:
        List of stress test catalog configurations
        
    Examples:
        >>> stress_catalogs = generate_catalog_stress_scenarios()
        >>> len(stress_catalogs) > 0
        True
        >>> all(len(catalog) > 50 for catalog in stress_catalogs)
        True
    """
    if max_datasets is None:
        max_datasets = STRESS_TEST_LIMITS.max_catalog_entries
    
    stress_scenarios = []
    
    # Progressive stress levels
    stress_levels = [50, 100, 200, 350, max_datasets]
    
    for stress_level in stress_levels:
        if stress_level > max_datasets:
            continue
            
        catalog_config = generate_high_volume_catalog_config(
            num_figure_datasets=stress_level,
            concurrent_access_pattern="mixed"
        )
        
        # Add stress-specific configuration
        for dataset_name, dataset_config in catalog_config.items():
            if 'figregistry_kedro.datasets.FigureDataSet' in dataset_config.get('type', ''):
                # Configure for stress conditions
                dataset_config['stress_test'] = {
                    'target_datasets': stress_level,
                    'memory_monitoring': True,
                    'performance_tracking': True
                }
                
                # Add complex style parameters for higher stress
                if stress_level > 100:
                    dataset_config['style_params'] = {
                        'figure.figsize': [12, 10],
                        'axes.grid': True,
                        'axes.grid.alpha': 0.3,
                        'font.size': 12,
                        'axes.labelsize': 14,
                        'axes.titlesize': 16,
                        'legend.fontsize': 11,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10
                    }
        
        stress_scenarios.append(catalog_config)
    
    return stress_scenarios


# =============================================================================
# COMPLEX FIGURE GENERATORS
# =============================================================================

def generate_simple_figure(
    figsize: Tuple[int, int] = (8, 6),
    data_points: int = 100
) -> FigureObject:
    """
    Generate simple matplotlib figure for baseline performance testing.
    
    Creates basic figure with single plot for measuring baseline FigureDataSet
    save performance without complex rendering overhead.
    
    Args:
        figsize: Figure size in inches (width, height)
        data_points: Number of data points in the plot
        
    Returns:
        Matplotlib figure object for performance testing
        
    Examples:
        >>> fig = generate_simple_figure(figsize=(10, 8), data_points=50)
        >>> fig.get_size_inches()[0] == 10.0
        True
        >>> len(fig.axes) == 1
        True
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate simple data
    x = np.linspace(0, 10, data_points)
    y = np.sin(x) + np.random.normal(0, 0.1, data_points)
    
    # Create basic plot
    ax.plot(x, y, 'b-', linewidth=2, label='Data')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values') 
    ax.set_title('Simple Performance Test Figure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def generate_complex_figure(
    complexity_level: str = "medium",
    num_subplots: int = None,
    data_points_per_subplot: int = None
) -> FigureObject:
    """
    Generate complex matplotlib figure for advanced performance testing.
    
    Creates multi-subplot figures with various plot types to test FigureDataSet
    performance with complex rendering and styling requirements.
    
    Args:
        complexity_level: Complexity level ('low', 'medium', 'high', 'extreme')
        num_subplots: Number of subplots (auto-determined by complexity if None)
        data_points_per_subplot: Data points per subplot (auto-determined if None)
        
    Returns:
        Complex matplotlib figure object for stress testing
        
    Examples:
        >>> fig = generate_complex_figure(complexity_level="medium")
        >>> len(fig.axes) >= 4
        True
        >>> fig.get_size_inches()[0] >= 10
        True
    """
    complexity_settings = {
        'low': {'subplots': 2, 'points': 500, 'figsize': (10, 6)},
        'medium': {'subplots': 4, 'points': 1000, 'figsize': (12, 8)}, 
        'high': {'subplots': 9, 'points': 2000, 'figsize': (15, 12)},
        'extreme': {'subplots': 16, 'points': 5000, 'figsize': (20, 16)}
    }
    
    settings = complexity_settings.get(complexity_level, complexity_settings['medium'])
    
    if num_subplots is not None:
        settings['subplots'] = min(num_subplots, STRESS_TEST_LIMITS.max_figure_subplots)
    if data_points_per_subplot is not None:
        settings['points'] = min(data_points_per_subplot, 
                               STRESS_TEST_LIMITS.max_data_points_per_plot)
    
    # Calculate subplot layout
    rows = int(np.ceil(np.sqrt(settings['subplots'])))
    cols = int(np.ceil(settings['subplots'] / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=settings['figsize'])
    if settings['subplots'] == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    # Plot types for variety
    plot_types = ['line', 'scatter', 'bar', 'histogram', 'heatmap', 'contour']
    
    for i in range(settings['subplots']):
        if i >= len(axes):
            break
            
        ax = axes[i]
        plot_type = plot_types[i % len(plot_types)]
        
        # Generate data based on plot type
        if plot_type == 'line':
            x = np.linspace(0, 10, settings['points'])
            y = np.sin(x * (i + 1)) + np.random.normal(0, 0.1, settings['points'])
            ax.plot(x, y, linewidth=1.5, label=f'Series {i+1}')
            
        elif plot_type == 'scatter':
            x = np.random.randn(settings['points'])
            y = np.random.randn(settings['points'])
            ax.scatter(x, y, alpha=0.6, s=20)
            
        elif plot_type == 'bar':
            categories = [f'Cat{j}' for j in range(min(20, settings['points'] // 50))]
            values = np.random.randint(10, 100, len(categories))
            ax.bar(categories, values, alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
            
        elif plot_type == 'histogram':
            data = np.random.normal(0, 1, settings['points'])
            ax.hist(data, bins=50, alpha=0.7, density=True)
            
        elif plot_type == 'heatmap':
            size = min(50, int(np.sqrt(settings['points'])))
            data = np.random.randn(size, size)
            im = ax.imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax)
            
        elif plot_type == 'contour':
            size = min(50, int(np.sqrt(settings['points'])))
            x = np.linspace(-3, 3, size)
            y = np.linspace(-3, 3, size)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y) + np.random.normal(0, 0.1, (size, size))
            ax.contour(X, Y, Z, levels=10)
        
        # Add labels and formatting
        ax.set_title(f'{plot_type.title()} Plot {i+1}')
        ax.grid(True, alpha=0.3)
        
        if plot_type in ['line']:
            ax.legend()
    
    # Remove empty subplots
    for i in range(settings['subplots'], len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig


def generate_memory_intensive_figure(
    memory_target_mb: float = 10.0
) -> FigureObject:
    """
    Generate memory-intensive figure for memory usage testing.
    
    Creates figure designed to use specific amount of memory to test plugin
    memory overhead calculations and validate memory usage targets.
    
    Args:
        memory_target_mb: Target memory usage in megabytes
        
    Returns:
        Memory-intensive matplotlib figure for memory testing
        
    Examples:
        >>> fig = generate_memory_intensive_figure(memory_target_mb=5.0)
        >>> fig is not None
        True
    """
    # Estimate data points needed for target memory
    # Rough estimate: 8 bytes per float64 point * 2 (x,y) * overhead factor
    bytes_per_point = 16  # Conservative estimate including overhead
    target_bytes = memory_target_mb * 1024 * 1024
    estimated_points = int(target_bytes / bytes_per_point)
    
    # Cap at stress test limits
    estimated_points = min(estimated_points, STRESS_TEST_LIMITS.max_data_points_per_plot)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    points_per_subplot = estimated_points // 4
    
    for i, ax in enumerate(axes):
        # Generate large datasets
        x = np.linspace(0, 100, points_per_subplot)
        y = np.sin(x * 0.1) + np.random.normal(0, 0.1, points_per_subplot)
        
        if i == 0:
            # Dense line plot
            ax.plot(x, y, linewidth=0.5, alpha=0.8)
        elif i == 1:
            # Dense scatter plot
            ax.scatter(x, y, s=1, alpha=0.5)
        elif i == 2:
            # High-resolution histogram
            ax.hist(y, bins=200, alpha=0.7, density=True)
        else:
            # High-resolution heatmap
            size = int(np.sqrt(points_per_subplot))
            if size > 0:
                data = np.random.randn(size, size)
                im = ax.imshow(data, aspect='auto', cmap='plasma', interpolation='bilinear')
                plt.colorbar(im, ax=ax)
        
        ax.set_title(f'Memory Test Subplot {i+1}')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f'Memory Intensive Figure (Target: {memory_target_mb:.1f} MB)')
    plt.tight_layout()
    
    return fig


def generate_figure_dataset_batch(
    batch_size: int = 10,
    complexity_distribution: List[str] = None
) -> List[FigureObject]:
    """
    Generate batch of figures for concurrent execution testing.
    
    Creates multiple figures of varying complexity for testing concurrent
    FigureDataSet save operations and parallel pipeline execution scenarios.
    
    Args:
        batch_size: Number of figures to generate
        complexity_distribution: List of complexity levels for each figure
        
    Returns:
        List of matplotlib figure objects for batch testing
        
    Examples:
        >>> figures = generate_figure_dataset_batch(batch_size=5)
        >>> len(figures) == 5
        True
        >>> all(isinstance(fig, matplotlib.figure.Figure) for fig in figures)
        True
    """
    if complexity_distribution is None:
        complexity_distribution = ['low', 'medium', 'high'] * (batch_size // 3 + 1)
    
    figures = []
    
    for i in range(batch_size):
        complexity = complexity_distribution[i % len(complexity_distribution)]
        
        if complexity == 'simple':
            fig = generate_simple_figure(
                figsize=(8 + i % 4, 6 + i % 3),
                data_points=100 + i * 50
            )
        else:
            fig = generate_complex_figure(
                complexity_level=complexity,
                num_subplots=None,  # Auto-determined by complexity
                data_points_per_subplot=None  # Auto-determined by complexity
            )
        
        # Add batch metadata for tracking
        fig.batch_index = i
        fig.complexity_level = complexity
        fig.generation_timestamp = time.time()
        
        figures.append(fig)
    
    return figures


# =============================================================================
# CONCURRENT EXECUTION DATA
# =============================================================================

def generate_concurrent_execution_scenarios(
    max_concurrency: int = None
) -> List[Dict[str, Any]]:
    """
    Generate concurrent execution scenarios for parallel Kedro runner testing.
    
    Creates test scenarios with different concurrency patterns to validate
    plugin thread-safety and performance under parallel execution conditions.
    
    Args:
        max_concurrency: Maximum concurrency level (defaults to system CPU count)
        
    Returns:
        List of concurrent execution scenario configurations
        
    Examples:
        >>> scenarios = generate_concurrent_execution_scenarios(max_concurrency=4)
        >>> len(scenarios) > 0
        True
        >>> all('concurrency_level' in scenario for scenario in scenarios)
        True
    """
    if max_concurrency is None:
        max_concurrency = min(psutil.cpu_count(), STRESS_TEST_LIMITS.max_parallel_pipelines)
    
    scenarios = []
    
    # Define concurrency patterns
    concurrency_levels = [1, 2, 4, 8, 16]
    execution_patterns = ['sequential', 'burst', 'sustained', 'mixed']
    
    for concurrency in concurrency_levels:
        if concurrency > max_concurrency:
            continue
            
        for pattern in execution_patterns:
            scenario = {
                'concurrency_level': concurrency,
                'execution_pattern': pattern,
                'total_operations': concurrency * 10,  # 10 operations per thread
                'operation_duration': 1.0,  # 1 second per operation
                'figure_complexity': 'medium',
                'memory_limit_mb': 100,
                'timeout_seconds': 60
            }
            
            # Pattern-specific configuration
            if pattern == 'sequential':
                scenario['delay_between_operations'] = 0.1
                scenario['concurrent_batches'] = 1
            elif pattern == 'burst':
                scenario['delay_between_operations'] = 0.0
                scenario['concurrent_batches'] = concurrency
                scenario['burst_size'] = concurrency
            elif pattern == 'sustained':
                scenario['delay_between_operations'] = 0.05
                scenario['concurrent_batches'] = max(1, concurrency // 2)
                scenario['duration_seconds'] = 30
            elif pattern == 'mixed':
                scenario['delay_between_operations'] = 0.02
                scenario['concurrent_batches'] = concurrency
                scenario['burst_intervals'] = [2, 5, 10]  # Burst every N operations
            
            # Add performance expectations
            scenario['performance_expectations'] = {
                'max_operation_time_ms': PERFORMANCE_TARGETS.figure_dataset_save_overhead_ms,
                'max_memory_per_thread_mb': PERFORMANCE_TARGETS.plugin_memory_overhead_mb,
                'max_total_memory_mb': PERFORMANCE_TARGETS.plugin_memory_overhead_mb * concurrency,
                'throughput_degradation_threshold': PERFORMANCE_TARGETS.concurrent_execution_degradation_percent
            }
            
            scenarios.append(scenario)
    
    return scenarios


def generate_thread_safety_test_data(
    num_threads: int = 8,
    operations_per_thread: int = 50
) -> Dict[str, Any]:
    """
    Generate test data for thread safety validation.
    
    Creates shared data structures and execution patterns for testing plugin
    thread safety under concurrent access to configuration, styling, and
    file operations.
    
    Args:
        num_threads: Number of concurrent threads to simulate
        operations_per_thread: Number of operations per thread
        
    Returns:
        Thread safety test configuration and data
        
    Examples:
        >>> test_data = generate_thread_safety_test_data(num_threads=4)
        >>> test_data['num_threads'] == 4
        True
        >>> 'shared_resources' in test_data
        True
    """
    # Generate shared configuration data
    shared_config = generate_large_figregistry_config(
        num_conditions=100,
        config_complexity='medium'
    )
    
    # Generate shared catalog configuration
    shared_catalog = generate_high_volume_catalog_config(
        num_figure_datasets=num_threads * 5,
        concurrent_access_pattern='parallel'
    )
    
    # Generate figure data for each thread
    thread_figures = {}
    for thread_id in range(num_threads):
        thread_figures[thread_id] = generate_figure_dataset_batch(
            batch_size=operations_per_thread,
            complexity_distribution=['low', 'medium'] * (operations_per_thread // 2 + 1)
        )
    
    # Define shared resources and conflict scenarios
    shared_resources = {
        'output_directories': [
            f'data/08_reporting/thread_{i}' for i in range(num_threads)
        ],
        'shared_output_directory': 'data/08_reporting/shared',
        'configuration_cache': shared_config,
        'style_cache': {},  # To be populated during testing
        'file_locks': {},  # To track file access conflicts
    }
    
    # Define conflict scenarios
    conflict_scenarios = [
        {
            'name': 'same_file_different_threads',
            'threads': list(range(num_threads)),
            'target_file': 'data/08_reporting/shared/conflict_test.png',
            'expected_behavior': 'serialized_access'
        },
        {
            'name': 'same_config_different_conditions',
            'threads': list(range(min(4, num_threads))),
            'conditions': [f'condition_{i}' for i in range(min(4, num_threads))],
            'expected_behavior': 'parallel_resolution'
        },
        {
            'name': 'cache_invalidation_race',
            'threads': [0, 1],  # Two threads only
            'cache_operations': ['read', 'write', 'invalidate'],
            'expected_behavior': 'consistent_state'
        }
    ]
    
    # Performance benchmarks for thread safety
    performance_benchmarks = {
        'single_thread_baseline': {
            'operations': operations_per_thread,
            'expected_time_ms': operations_per_thread * 200,  # 200ms per operation
            'max_memory_mb': 10
        },
        'multi_thread_target': {
            'operations': operations_per_thread * num_threads,
            'max_time_ms': operations_per_thread * 250,  # 25% overhead acceptable
            'max_memory_mb': 10 * num_threads,
            'scalability_efficiency': 0.8  # 80% efficiency target
        }
    }
    
    return {
        'num_threads': num_threads,
        'operations_per_thread': operations_per_thread,
        'shared_config': shared_config,
        'shared_catalog': shared_catalog,
        'thread_figures': thread_figures,
        'shared_resources': shared_resources,
        'conflict_scenarios': conflict_scenarios,
        'performance_benchmarks': performance_benchmarks,
        'test_duration_seconds': 30,
        'monitoring_interval_ms': 100
    }


def generate_parallel_pipeline_scenarios(
    num_pipelines: int = 4,
    pipeline_complexity: str = "medium"
) -> List[Dict[str, Any]]:
    """
    Generate parallel Kedro pipeline execution scenarios.
    
    Creates multiple pipeline configurations for testing concurrent Kedro
    pipeline execution with FigRegistry plugin integration.
    
    Args:
        num_pipelines: Number of parallel pipelines to configure
        pipeline_complexity: Complexity level of each pipeline
        
    Returns:
        List of pipeline configuration scenarios
        
    Examples:
        >>> pipelines = generate_parallel_pipeline_scenarios(num_pipelines=3)
        >>> len(pipelines) == 3
        True
        >>> all('pipeline_id' in pipeline for pipeline in pipelines)
        True
    """
    complexity_settings = {
        'simple': {'nodes': 3, 'figures_per_node': 1, 'processing_time': 5},
        'medium': {'nodes': 5, 'figures_per_node': 2, 'processing_time': 10},
        'complex': {'nodes': 8, 'figures_per_node': 3, 'processing_time': 20}
    }
    
    settings = complexity_settings.get(pipeline_complexity, complexity_settings['medium'])
    scenarios = []
    
    for pipeline_id in range(num_pipelines):
        # Generate pipeline-specific catalog
        pipeline_catalog = generate_high_volume_catalog_config(
            num_figure_datasets=settings['nodes'] * settings['figures_per_node'],
            concurrent_access_pattern='parallel'
        )
        
        # Add pipeline-specific namespace
        namespaced_catalog = {}
        for dataset_name, dataset_config in pipeline_catalog.items():
            namespaced_name = f"pipeline_{pipeline_id}_{dataset_name}"
            namespaced_config = dataset_config.copy()
            
            # Update file paths to prevent conflicts
            if 'filepath' in namespaced_config:
                original_path = namespaced_config['filepath']
                namespaced_config['filepath'] = original_path.replace(
                    'data/08_reporting', 
                    f'data/08_reporting/pipeline_{pipeline_id}'
                )
            
            namespaced_catalog[namespaced_name] = namespaced_config
        
        # Generate pipeline configuration
        pipeline_scenario = {
            'pipeline_id': pipeline_id,
            'pipeline_name': f'parallel_test_pipeline_{pipeline_id}',
            'catalog_config': namespaced_catalog,
            'num_nodes': settings['nodes'],
            'figures_per_node': settings['figures_per_node'],
            'estimated_execution_time': settings['processing_time'],
            'complexity_level': pipeline_complexity,
            'resource_requirements': {
                'memory_mb': 50 + pipeline_id * 10,
                'cpu_cores': 1,
                'disk_space_mb': 100
            },
            'dependencies': [],  # No inter-pipeline dependencies for parallel testing
            'performance_targets': {
                'max_execution_time_seconds': settings['processing_time'] * 2,
                'max_memory_usage_mb': 100,
                'min_throughput_figures_per_second': 1.0
            }
        }
        
        # Add inter-pipeline dependencies for some scenarios
        if pipeline_id > 0 and pipeline_complexity == 'complex':
            # Complex scenarios may have dependencies
            dependency_probability = 0.3
            if np.random.random() < dependency_probability:
                dependency_id = np.random.randint(0, pipeline_id)
                pipeline_scenario['dependencies'].append(f'parallel_test_pipeline_{dependency_id}')
        
        scenarios.append(pipeline_scenario)
    
    return scenarios


# =============================================================================
# MEMORY USAGE SCENARIOS
# =============================================================================

def generate_memory_usage_scenarios(
    target_memory_levels: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Generate memory usage test scenarios for plugin footprint validation.
    
    Creates scenarios with different memory usage patterns to validate plugin
    memory overhead remains within the 5MB target per technical specifications.
    
    Args:
        target_memory_levels: List of target memory levels in MB
        
    Returns:
        List of memory usage test scenario configurations
        
    Examples:
        >>> scenarios = generate_memory_usage_scenarios()
        >>> len(scenarios) > 0
        True
        >>> all('target_memory_mb' in scenario for scenario in scenarios)
        True
    """
    if target_memory_levels is None:
        target_memory_levels = [1.0, 2.5, 5.0, 7.5, 10.0]  # MB
    
    scenarios = []
    
    for target_memory in target_memory_levels:
        # Create memory usage scenario
        scenario = {
            'target_memory_mb': target_memory,
            'scenario_name': f'memory_test_{target_memory:.1f}mb',
            'memory_allocation_strategy': 'gradual',
            'measurement_interval_ms': 100,
            'test_duration_seconds': 30
        }
        
        # Configure memory usage patterns
        if target_memory <= 2.0:
            # Low memory scenario
            scenario.update({
                'config_complexity': 'simple',
                'num_catalog_entries': 10,
                'figure_complexity': 'low',
                'concurrent_operations': 1,
                'cache_size_limit': '500KB'
            })
        elif target_memory <= 5.0:
            # Target memory scenario (plugin target)
            scenario.update({
                'config_complexity': 'medium',
                'num_catalog_entries': 50,
                'figure_complexity': 'medium',
                'concurrent_operations': 2,
                'cache_size_limit': '2MB'
            })
        else:
            # High memory scenario (stress testing)
            scenario.update({
                'config_complexity': 'complex',
                'num_catalog_entries': 200,
                'figure_complexity': 'high',
                'concurrent_operations': 4,
                'cache_size_limit': '5MB'
            })
        
        # Add memory monitoring configuration
        scenario['memory_monitoring'] = {
            'track_plugin_memory': True,
            'track_process_memory': True,
            'track_system_memory': True,
            'memory_profiling_enabled': MEMORY_PROFILER_AVAILABLE,
            'gc_monitoring': True,
            'allocation_tracking': True
        }
        
        # Define memory test operations
        scenario['test_operations'] = [
            {
                'operation': 'load_large_config',
                'params': {'num_conditions': 100 * int(target_memory)},
                'expected_memory_delta_mb': target_memory * 0.2
            },
            {
                'operation': 'generate_complex_figures',
                'params': {'batch_size': int(target_memory * 2)},
                'expected_memory_delta_mb': target_memory * 0.4
            },
            {
                'operation': 'concurrent_saves',
                'params': {'concurrency': min(4, int(target_memory))},
                'expected_memory_delta_mb': target_memory * 0.3
            },
            {
                'operation': 'cache_stress',
                'params': {'cache_operations': int(target_memory * 100)},
                'expected_memory_delta_mb': target_memory * 0.1
            }
        ]
        
        scenarios.append(scenario)
    
    return scenarios


def generate_memory_leak_detection_data(
    num_iterations: int = 100,
    operation_types: List[str] = None
) -> Dict[str, Any]:
    """
    Generate test data for memory leak detection.
    
    Creates repetitive operation patterns to detect memory leaks in plugin
    components through sustained operation monitoring.
    
    Args:
        num_iterations: Number of iterations to run each operation
        operation_types: List of operation types to test for leaks
        
    Returns:
        Memory leak detection test configuration
        
    Examples:
        >>> leak_test = generate_memory_leak_detection_data(num_iterations=50)
        >>> leak_test['num_iterations'] == 50
        True
        >>> 'operations' in leak_test
        True
    """
    if operation_types is None:
        operation_types = [
            'config_loading',
            'style_resolution', 
            'figure_saving',
            'cache_operations',
            'concurrent_access'
        ]
    
    # Generate test operations for each type
    operations = {}
    
    for op_type in operation_types:
        if op_type == 'config_loading':
            operations[op_type] = {
                'function': 'load_and_merge_config',
                'params': {
                    'config_size': 'medium',
                    'environments': ['local', 'staging']
                },
                'cleanup_required': True,
                'expected_memory_stable': True
            }
            
        elif op_type == 'style_resolution':
            operations[op_type] = {
                'function': 'resolve_style_conditions',
                'params': {
                    'conditions': [f'condition_{i}' for i in range(20)],
                    'cache_enabled': True
                },
                'cleanup_required': False,
                'expected_memory_stable': True
            }
            
        elif op_type == 'figure_saving':
            operations[op_type] = {
                'function': 'save_figure_with_styling',
                'params': {
                    'figure_complexity': 'medium',
                    'formats': ['png', 'pdf'],
                    'cleanup_files': True
                },
                'cleanup_required': True,
                'expected_memory_stable': True
            }
            
        elif op_type == 'cache_operations':
            operations[op_type] = {
                'function': 'cache_stress_test',
                'params': {
                    'cache_size': 1000,
                    'invalidation_rate': 0.1
                },
                'cleanup_required': True,
                'expected_memory_stable': True
            }
            
        elif op_type == 'concurrent_access':
            operations[op_type] = {
                'function': 'concurrent_operation_test',
                'params': {
                    'num_threads': 4,
                    'operations_per_thread': 10
                },
                'cleanup_required': True,
                'expected_memory_stable': True
            }
    
    # Memory leak detection configuration
    detection_config = {
        'baseline_measurements': 10,  # Initial measurements for baseline
        'measurement_interval': 5,   # Measure every N iterations
        'leak_threshold_mb': STRESS_TEST_LIMITS.memory_leak_tolerance_mb,
        'trend_analysis_window': 20,  # Analyze trend over N measurements
        'gc_frequency': 10,  # Force GC every N iterations
        'memory_profiling': MEMORY_PROFILER_AVAILABLE
    }
    
    return {
        'num_iterations': num_iterations,
        'operations': operations,
        'detection_config': detection_config,
        'performance_targets': {
            'max_memory_increase_mb': STRESS_TEST_LIMITS.memory_leak_tolerance_mb,
            'max_iteration_time_ms': 500,
            'gc_efficiency_threshold': 0.9
        },
        'monitoring_settings': {
            'track_rss_memory': True,
            'track_heap_memory': True,
            'track_gc_collections': True,
            'track_object_counts': True,
            'detailed_profiling': False  # Enable for detailed analysis
        }
    }


def calculate_memory_footprint_baseline() -> Dict[str, float]:
    """
    Calculate baseline memory footprint for plugin components.
    
    Measures baseline memory usage of plugin components to establish
    reference values for memory overhead calculations.
    
    Returns:
        Dictionary with baseline memory measurements in MB
        
    Examples:
        >>> baseline = calculate_memory_footprint_baseline()
        >>> 'process_baseline_mb' in baseline
        True
        >>> baseline['process_baseline_mb'] > 0
        True
    """
    import gc
    
    # Force garbage collection for clean baseline
    gc.collect()
    
    # Get process memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    
    baseline = {
        'process_baseline_mb': memory_info.rss / (1024 * 1024),
        'virtual_memory_mb': memory_info.vms / (1024 * 1024),
        'shared_memory_mb': getattr(memory_info, 'shared', 0) / (1024 * 1024),
        'system_available_mb': psutil.virtual_memory().available / (1024 * 1024),
        'system_total_mb': psutil.virtual_memory().total / (1024 * 1024),
        'cpu_count': psutil.cpu_count(),
        'python_objects': len(gc.get_objects()),
        'gc_collections': sum(gc.get_counts())
    }
    
    # Add matplotlib baseline if imported
    if 'matplotlib.pyplot' in globals():
        baseline['matplotlib_figures'] = len(plt.get_fignums())
    
    return baseline


# =============================================================================
# BENCHMARK TIMING UTILITIES
# =============================================================================

class PerformanceTimer:
    """
    High-precision timing utility for plugin performance measurement.
    
    Provides context manager and decorator interfaces for measuring operation
    timing with microsecond precision and statistical analysis capabilities.
    """
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.measurements = []
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.measurements.append(self.duration_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary of all measurements."""
        if not self.measurements:
            return {}
        
        measurements = np.array(self.measurements)
        return {
            'count': len(measurements),
            'mean_ms': float(np.mean(measurements)),
            'median_ms': float(np.median(measurements)),
            'std_ms': float(np.std(measurements)),
            'min_ms': float(np.min(measurements)),
            'max_ms': float(np.max(measurements)),
            'p95_ms': float(np.percentile(measurements, 95)),
            'p99_ms': float(np.percentile(measurements, 99))
        }


def time_function_calls(
    func: Callable,
    args_list: List[Tuple],
    num_warmup: int = 5,
    num_measurements: int = 20
) -> Dict[str, Any]:
    """
    Time function calls with statistical analysis.
    
    Measures function execution time across multiple calls with warmup
    iterations and statistical analysis of timing results.
    
    Args:
        func: Function to time
        args_list: List of argument tuples for function calls
        num_warmup: Number of warmup iterations
        num_measurements: Number of timing measurements
        
    Returns:
        Timing results with statistical analysis
        
    Examples:
        >>> def test_func(x): return x * 2
        >>> results = time_function_calls(test_func, [(1,), (2,)], num_warmup=2, num_measurements=5)
        >>> 'mean_ms' in results
        True
    """
    # Warmup iterations
    for i in range(num_warmup):
        args = args_list[i % len(args_list)]
        func(*args)
    
    # Timing measurements
    measurements = []
    
    for i in range(num_measurements):
        args = args_list[i % len(args_list)]
        
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        measurements.append(duration_ms)
    
    # Statistical analysis
    measurements = np.array(measurements)
    
    return {
        'function_name': func.__name__,
        'num_calls': num_measurements,
        'measurements_ms': measurements.tolist(),
        'mean_ms': float(np.mean(measurements)),
        'median_ms': float(np.median(measurements)),
        'std_ms': float(np.std(measurements)),
        'min_ms': float(np.min(measurements)),
        'max_ms': float(np.max(measurements)),
        'p95_ms': float(np.percentile(measurements, 95)),
        'p99_ms': float(np.percentile(measurements, 99)),
        'coefficient_of_variation': float(np.std(measurements) / np.mean(measurements)),
        'outliers': _detect_outliers(measurements).tolist()
    }


def benchmark_plugin_operations(
    operation_configs: List[Dict[str, Any]],
    iterations: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple plugin operations for performance validation.
    
    Runs comprehensive benchmarks across plugin operations to validate
    performance targets and identify bottlenecks.
    
    Args:
        operation_configs: List of operation configurations to benchmark
        iterations: Number of iterations per operation
        
    Returns:
        Benchmark results for all operations
        
    Examples:
        >>> configs = [{'name': 'test_op', 'func': lambda: None, 'args': []}]
        >>> results = benchmark_plugin_operations(configs, iterations=10)
        >>> len(results) > 0
        True
    """
    benchmark_results = {}
    
    for config in operation_configs:
        operation_name = config['name']
        operation_func = config['func']
        operation_args = config.get('args', [])
        
        # Time the operation
        with PerformanceTimer(operation_name) as timer:
            for _ in range(iterations):
                if operation_args:
                    operation_func(*operation_args)
                else:
                    operation_func()
        
        # Store results
        benchmark_results[operation_name] = timer.get_stats()
        
        # Add performance target comparison
        target_key = f"{operation_name}_target_ms"
        if hasattr(PERFORMANCE_TARGETS, target_key.replace('_target_ms', '_time_ms')):
            target_value = getattr(PERFORMANCE_TARGETS, target_key.replace('_target_ms', '_time_ms'))
            benchmark_results[operation_name]['target_ms'] = target_value
            benchmark_results[operation_name]['meets_target'] = (
                benchmark_results[operation_name]['mean_ms'] <= target_value
            )
    
    return benchmark_results


@contextmanager
def memory_monitoring(
    operation_name: str = "operation",
    sampling_interval: float = 0.1
):
    """
    Context manager for memory usage monitoring during operations.
    
    Monitors memory usage throughout operation execution with periodic
    sampling to detect memory leaks and validate memory targets.
    
    Args:
        operation_name: Name of operation being monitored
        sampling_interval: Memory sampling interval in seconds
        
    Yields:
        Memory monitoring results dictionary
        
    Examples:
        >>> with memory_monitoring("test_op") as monitor:
        ...     pass  # Some operation
        >>> 'peak_memory_mb' in monitor
        True
    """
    process = psutil.Process()
    
    # Initial memory measurement
    initial_memory = process.memory_info().rss / (1024 * 1024)
    
    monitoring_data = {
        'operation_name': operation_name,
        'initial_memory_mb': initial_memory,
        'memory_samples': [],
        'timestamps': []
    }
    
    def memory_sampler():
        start_time = time.time()
        while getattr(memory_sampler, 'running', True):
            current_memory = process.memory_info().rss / (1024 * 1024)
            current_time = time.time()
            
            monitoring_data['memory_samples'].append(current_memory)
            monitoring_data['timestamps'].append(current_time - start_time)
            
            time.sleep(sampling_interval)
    
    # Start memory sampling in background thread
    memory_sampler.running = True
    sampling_thread = threading.Thread(target=memory_sampler, daemon=True)
    sampling_thread.start()
    
    try:
        yield monitoring_data
    finally:
        # Stop memory sampling
        memory_sampler.running = False
        sampling_thread.join(timeout=1.0)
        
        # Final memory measurement
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        # Calculate memory statistics
        if monitoring_data['memory_samples']:
            memory_samples = np.array(monitoring_data['memory_samples'])
            monitoring_data.update({
                'final_memory_mb': final_memory,
                'peak_memory_mb': float(np.max(memory_samples)),
                'min_memory_mb': float(np.min(memory_samples)),
                'mean_memory_mb': float(np.mean(memory_samples)),
                'memory_delta_mb': final_memory - initial_memory,
                'max_memory_delta_mb': float(np.max(memory_samples) - initial_memory),
                'num_samples': len(memory_samples)
            })


def _detect_outliers(data: np.ndarray, z_threshold: float = 2.0) -> np.ndarray:
    """
    Detect outliers in timing data using z-score method.
    
    Args:
        data: Array of timing measurements
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        Array of outlier indices
    """
    if len(data) < 3:
        return np.array([])
    
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    outlier_indices = np.where(z_scores > z_threshold)[0]
    
    return outlier_indices


# =============================================================================
# STRESS TEST DATA GENERATORS
# =============================================================================

def generate_stress_test_configurations(
    stress_level: str = "medium"
) -> List[Dict[str, Any]]:
    """
    Generate stress test configurations for high-load scenario validation.
    
    Creates configurations designed to stress plugin components to validate
    graceful degradation and identify performance boundaries under extreme load.
    
    Args:
        stress_level: Stress level ('low', 'medium', 'high', 'extreme')
        
    Returns:
        List of stress test configuration scenarios
        
    Examples:
        >>> stress_configs = generate_stress_test_configurations(stress_level="medium")
        >>> len(stress_configs) > 0
        True
        >>> all('stress_type' in config for config in stress_configs)
        True
    """
    stress_settings = {
        'low': {
            'config_conditions': 200,
            'catalog_entries': 100,
            'concurrent_operations': 4,
            'memory_target_mb': 25,
            'duration_seconds': 60
        },
        'medium': {
            'config_conditions': 500,
            'catalog_entries': 250,
            'concurrent_operations': 8,
            'memory_target_mb': 50,
            'duration_seconds': 120
        },
        'high': {
            'config_conditions': 1000,
            'catalog_entries': 500,
            'concurrent_operations': 16,
            'memory_target_mb': 100,
            'duration_seconds': 300
        },
        'extreme': {
            'config_conditions': 2000,
            'catalog_entries': 1000,
            'concurrent_operations': 32,
            'memory_target_mb': 200,
            'duration_seconds': 600
        }
    }
    
    settings = stress_settings.get(stress_level, stress_settings['medium'])
    stress_configurations = []
    
    # Configuration stress test
    config_stress = {
        'stress_type': 'configuration_complexity',
        'name': f'{stress_level}_config_stress',
        'description': 'Test configuration bridge performance with large configurations',
        'parameters': {
            'num_conditions': settings['config_conditions'],
            'num_environments': 5,
            'nesting_depth': 8,
            'merge_operations': 100
        },
        'targets': {
            'max_merge_time_ms': PERFORMANCE_TARGETS.config_bridge_merge_time_ms * 2,
            'max_memory_mb': settings['memory_target_mb'] * 0.3,
            'success_rate': 0.95
        },
        'monitoring': {
            'track_timing': True,
            'track_memory': True,
            'track_errors': True
        }
    }
    stress_configurations.append(config_stress)
    
    # Catalog volume stress test
    catalog_stress = {
        'stress_type': 'catalog_volume',
        'name': f'{stress_level}_catalog_stress',
        'description': 'Test plugin performance with high-volume catalog configurations',
        'parameters': {
            'num_datasets': settings['catalog_entries'],
            'concurrent_saves': settings['concurrent_operations'],
            'figure_complexity': 'high',
            'versioning_enabled': True
        },
        'targets': {
            'max_save_time_ms': PERFORMANCE_TARGETS.figure_dataset_save_overhead_ms * 1.5,
            'max_memory_per_dataset_mb': 2.0,
            'success_rate': 0.90
        },
        'monitoring': {
            'track_throughput': True,
            'track_memory_per_operation': True,
            'track_file_system_pressure': True
        }
    }
    stress_configurations.append(catalog_stress)
    
    # Concurrent execution stress test
    concurrency_stress = {
        'stress_type': 'concurrent_execution',
        'name': f'{stress_level}_concurrency_stress',
        'description': 'Test plugin thread safety under high concurrency',
        'parameters': {
            'num_threads': settings['concurrent_operations'],
            'operations_per_thread': 50,
            'shared_resources': True,
            'contention_rate': 0.3
        },
        'targets': {
            'max_operation_time_ms': PERFORMANCE_TARGETS.figure_dataset_save_overhead_ms * 2,
            'thread_safety_violations': 0,
            'deadlock_timeout_seconds': 30
        },
        'monitoring': {
            'track_thread_contention': True,
            'track_race_conditions': True,
            'track_resource_conflicts': True
        }
    }
    stress_configurations.append(concurrency_stress)
    
    # Memory pressure stress test
    memory_stress = {
        'stress_type': 'memory_pressure',
        'name': f'{stress_level}_memory_stress',
        'description': 'Test plugin behavior under memory pressure',
        'parameters': {
            'target_memory_mb': settings['memory_target_mb'],
            'allocation_rate': 'aggressive',
            'gc_frequency': 'reduced',
            'memory_fragments': True
        },
        'targets': {
            'max_total_memory_mb': settings['memory_target_mb'],
            'memory_leak_tolerance_mb': STRESS_TEST_LIMITS.memory_leak_tolerance_mb,
            'gc_efficiency': 0.8
        },
        'monitoring': {
            'track_memory_allocation': True,
            'track_gc_performance': True,
            'track_memory_fragmentation': True
        }
    }
    stress_configurations.append(memory_stress)
    
    # Sustained load stress test
    sustained_load_stress = {
        'stress_type': 'sustained_load',
        'name': f'{stress_level}_sustained_load',
        'description': 'Test plugin stability under sustained load',
        'parameters': {
            'duration_seconds': settings['duration_seconds'],
            'operations_per_second': 10,
            'load_variation': 'random',
            'spike_frequency': 60  # Seconds between load spikes
        },
        'targets': {
            'uptime_percentage': 99.5,
            'performance_degradation_max': 20.0,  # Percent
            'error_rate_max': 1.0  # Percent
        },
        'monitoring': {
            'track_performance_trends': True,
            'track_stability_metrics': True,
            'track_resource_exhaustion': True
        }
    }
    stress_configurations.append(sustained_load_stress)
    
    return stress_configurations


def generate_performance_regression_data(
    baseline_metrics: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Generate test data for performance regression detection.
    
    Creates test scenarios for detecting performance regressions in plugin
    operations by comparing against established baseline metrics.
    
    Args:
        baseline_metrics: Baseline performance metrics for comparison
        
    Returns:
        Performance regression test configuration
        
    Examples:
        >>> regression_data = generate_performance_regression_data()
        >>> 'baseline_metrics' in regression_data
        True
        >>> 'regression_tests' in regression_data
        True
    """
    if baseline_metrics is None:
        baseline_metrics = {
            'config_merge_time_ms': PERFORMANCE_TARGETS.config_bridge_merge_time_ms * 0.8,
            'figure_save_time_ms': PERFORMANCE_TARGETS.figure_dataset_save_overhead_ms * 0.8,
            'hook_init_time_ms': PERFORMANCE_TARGETS.hook_initialization_time_ms * 0.8,
            'memory_usage_mb': PERFORMANCE_TARGETS.plugin_memory_overhead_mb * 0.8,
            'style_resolution_time_ms': PERFORMANCE_TARGETS.style_resolution_time_ms * 0.8
        }
    
    # Define regression test scenarios
    regression_tests = [
        {
            'test_name': 'config_bridge_regression',
            'operation': 'configuration_merge',
            'baseline_metric': 'config_merge_time_ms',
            'regression_threshold_percent': 15.0,  # 15% increase triggers alert
            'test_parameters': {
                'num_configs': 5,
                'config_size': 'large',
                'merge_complexity': 'high'
            }
        },
        {
            'test_name': 'figure_dataset_regression',
            'operation': 'figure_save_with_styling',
            'baseline_metric': 'figure_save_time_ms',
            'regression_threshold_percent': 10.0,
            'test_parameters': {
                'figure_complexity': 'medium',
                'style_complexity': 'high',
                'file_formats': ['png', 'pdf']
            }
        },
        {
            'test_name': 'hook_initialization_regression',
            'operation': 'hook_registration_and_init',
            'baseline_metric': 'hook_init_time_ms',
            'regression_threshold_percent': 20.0,
            'test_parameters': {
                'num_hooks': 3,
                'config_complexity': 'medium',
                'kedro_context_size': 'large'
            }
        },
        {
            'test_name': 'memory_usage_regression',
            'operation': 'sustained_plugin_usage',
            'baseline_metric': 'memory_usage_mb',
            'regression_threshold_percent': 25.0,
            'test_parameters': {
                'duration_seconds': 300,
                'operations_per_second': 2,
                'concurrent_threads': 4
            }
        },
        {
            'test_name': 'style_resolution_regression',
            'operation': 'style_condition_resolution',
            'baseline_metric': 'style_resolution_time_ms',
            'regression_threshold_percent': 30.0,
            'test_parameters': {
                'num_conditions': 100,
                'cache_enabled': True,
                'pattern_complexity': 'high'
            }
        }
    ]
    
    # Generate test data for each regression test
    test_data = {}
    for test in regression_tests:
        test_name = test['test_name']
        params = test['test_parameters']
        
        if 'config' in test_name:
            test_data[test_name] = generate_large_figregistry_config(
                num_conditions=params.get('num_configs', 5) * 100,
                config_complexity=params.get('config_size', 'medium')
            )
        elif 'figure' in test_name:
            test_data[test_name] = generate_figure_dataset_batch(
                batch_size=10,
                complexity_distribution=[params.get('figure_complexity', 'medium')] * 10
            )
        elif 'memory' in test_name:
            test_data[test_name] = generate_memory_usage_scenarios(
                target_memory_levels=[5.0, 10.0, 15.0]
            )
        else:
            test_data[test_name] = {}
    
    return {
        'baseline_metrics': baseline_metrics,
        'regression_tests': regression_tests,
        'test_data': test_data,
        'monitoring_config': {
            'measurement_frequency': 'per_test',
            'statistical_analysis': True,
            'trend_detection': True,
            'alert_thresholds': {
                'performance_degradation': 15.0,  # Percent
                'memory_increase': 25.0,  # Percent
                'error_rate_increase': 5.0  # Percent
            }
        },
        'reporting_config': {
            'generate_trend_charts': True,
            'compare_to_baseline': True,
            'include_statistical_significance': True,
            'export_raw_data': True
        }
    }


def create_load_testing_suite(
    test_duration_minutes: int = 30
) -> Dict[str, Any]:
    """
    Create comprehensive load testing suite for plugin validation.
    
    Combines multiple stress test scenarios into a comprehensive load testing
    suite for validating plugin behavior under sustained production-like loads.
    
    Args:
        test_duration_minutes: Total duration for load testing suite
        
    Returns:
        Complete load testing suite configuration
        
    Examples:
        >>> load_suite = create_load_testing_suite(test_duration_minutes=15)
        >>> 'test_phases' in load_suite
        True
        >>> len(load_suite['test_phases']) > 0
        True
    """
    total_duration_seconds = test_duration_minutes * 60
    
    # Define load testing phases
    test_phases = [
        {
            'phase_name': 'warmup',
            'duration_seconds': total_duration_seconds * 0.1,  # 10% of total time
            'description': 'System warmup with baseline load',
            'load_characteristics': {
                'concurrent_operations': 2,
                'operations_per_second': 1,
                'figure_complexity': 'low',
                'config_size': 'small'
            }
        },
        {
            'phase_name': 'ramp_up',
            'duration_seconds': total_duration_seconds * 0.2,  # 20% of total time
            'description': 'Gradual load increase to target levels',
            'load_characteristics': {
                'concurrent_operations': 'increase_from_2_to_8',
                'operations_per_second': 'increase_from_1_to_5',
                'figure_complexity': 'medium',
                'config_size': 'medium'
            }
        },
        {
            'phase_name': 'sustained_load',
            'duration_seconds': total_duration_seconds * 0.5,  # 50% of total time
            'description': 'Sustained high load testing',
            'load_characteristics': {
                'concurrent_operations': 8,
                'operations_per_second': 5,
                'figure_complexity': 'high',
                'config_size': 'large'
            }
        },
        {
            'phase_name': 'spike_testing',
            'duration_seconds': total_duration_seconds * 0.1,  # 10% of total time
            'description': 'Load spikes to test elasticity',
            'load_characteristics': {
                'concurrent_operations': 16,
                'operations_per_second': 10,
                'figure_complexity': 'extreme',
                'config_size': 'extreme',
                'spike_pattern': 'random'
            }
        },
        {
            'phase_name': 'cool_down',
            'duration_seconds': total_duration_seconds * 0.1,  # 10% of total time
            'description': 'Gradual load reduction and cleanup',
            'load_characteristics': {
                'concurrent_operations': 'decrease_from_8_to_1',
                'operations_per_second': 'decrease_from_5_to_1',
                'figure_complexity': 'low',
                'config_size': 'small'
            }
        }
    ]
    
    # Generate test data for each phase
    phase_test_data = {}
    for phase in test_phases:
        phase_name = phase['phase_name']
        load_chars = phase['load_characteristics']
        
        # Generate configuration data
        config_size = load_chars.get('config_size', 'medium')
        config_complexity_map = {
            'small': 50, 'medium': 200, 'large': 500, 'extreme': 1000
        }
        
        phase_test_data[phase_name] = {
            'config_data': generate_large_figregistry_config(
                num_conditions=config_complexity_map.get(config_size, 200),
                config_complexity=config_size
            ),
            'catalog_data': generate_high_volume_catalog_config(
                num_figure_datasets=config_complexity_map.get(config_size, 200) // 4,
                concurrent_access_pattern='parallel'
            ),
            'figure_data': generate_figure_dataset_batch(
                batch_size=20,
                complexity_distribution=[load_chars.get('figure_complexity', 'medium')] * 20
            )
        }
    
    # Performance monitoring configuration
    monitoring_config = {
        'metrics_collection_interval_seconds': 10,
        'detailed_profiling_phases': ['sustained_load', 'spike_testing'],
        'memory_monitoring': {
            'enabled': True,
            'sampling_rate_hz': 1,
            'alert_threshold_mb': 100
        },
        'performance_monitoring': {
            'enabled': True,
            'track_response_times': True,
            'track_throughput': True,
            'track_error_rates': True
        },
        'resource_monitoring': {
            'cpu_usage': True,
            'memory_usage': True,
            'disk_io': True,
            'file_system_pressure': True
        }
    }
    
    # Success criteria and thresholds
    success_criteria = {
        'overall_success_rate': 95.0,  # Percent
        'max_response_time_ms': PERFORMANCE_TARGETS.figure_dataset_save_overhead_ms * 2,
        'max_memory_usage_mb': PERFORMANCE_TARGETS.plugin_memory_overhead_mb * 5,
        'max_error_rate': 2.0,  # Percent
        'resource_utilization_limits': {
            'cpu_percent': 80.0,
            'memory_percent': 70.0,
            'disk_io_mbps': 100.0
        }
    }
    
    return {
        'test_phases': test_phases,
        'total_duration_seconds': total_duration_seconds,
        'phase_test_data': phase_test_data,
        'monitoring_config': monitoring_config,
        'success_criteria': success_criteria,
        'cleanup_config': {
            'cleanup_between_phases': True,
            'force_gc_between_phases': True,
            'reset_caches_between_phases': True,
            'cleanup_temp_files': True
        },
        'reporting_config': {
            'generate_phase_reports': True,
            'generate_overall_summary': True,
            'include_performance_graphs': True,
            'export_raw_metrics': True
        }
    }


# =============================================================================
# MODULE INITIALIZATION AND EXPORTS
# =============================================================================

# Export all public functions and classes
__all__ = [
    # Performance targets and limits
    'PerformanceTargets',
    'StressTestLimits', 
    'PERFORMANCE_TARGETS',
    'STRESS_TEST_LIMITS',
    
    # Large configuration generators
    'generate_large_figregistry_config',
    'generate_kedro_environment_configs',
    'generate_complex_merge_scenarios',
    
    # High volume catalog generators
    'generate_high_volume_catalog_config',
    'generate_concurrent_catalog_scenarios', 
    'generate_catalog_stress_scenarios',
    
    # Complex figure generators
    'generate_simple_figure',
    'generate_complex_figure',
    'generate_memory_intensive_figure',
    'generate_figure_dataset_batch',
    
    # Concurrent execution data
    'generate_concurrent_execution_scenarios',
    'generate_thread_safety_test_data',
    'generate_parallel_pipeline_scenarios',
    
    # Memory usage scenarios
    'generate_memory_usage_scenarios',
    'generate_memory_leak_detection_data',
    'calculate_memory_footprint_baseline',
    
    # Benchmark timing utilities  
    'PerformanceTimer',
    'time_function_calls',
    'benchmark_plugin_operations',
    'memory_monitoring',
    
    # Stress test data generators
    'generate_stress_test_configurations',
    'generate_performance_regression_data', 
    'create_load_testing_suite',
    
    # Type definitions
    'ConfigDict',
    'CatalogConfig',
    'FigureObject',
    'TimingResult',
    'MemoryResult',
    'PerformanceMetrics'
]


# Module metadata
__version__ = "1.0.0"
__author__ = "FigRegistry Development Team"
__description__ = "Performance testing data generators for figregistry-kedro plugin"