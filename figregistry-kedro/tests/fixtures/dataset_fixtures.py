"""FigureDataSet testing fixtures for comprehensive dataset validation.

This module provides specialized fixtures for testing FigureDataSet functionality,
including matplotlib figure mocks, catalog entry configurations, parameter testing
scenarios, and AbstractDataSet interface compliance validation.

Key fixture categories:
- Sample matplotlib figures with FigureDataSet integration metadata
- Catalog configuration templates with comprehensive parameter coverage
- Dataset parameter scenarios for condition resolution testing
- Versioned dataset fixtures for Kedro compatibility validation
- Error scenario fixtures for robust error handling testing
- Abstract dataset compliance fixtures for interface validation
- Performance testing fixtures for overhead measurement and validation

All fixtures support the requirement for <5% performance overhead compared to 
manual matplotlib operations and ensure comprehensive coverage of FigureDataSet
functionality per Section 5.2.6 requirements.
"""

import copy
import time
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Import the modules under test
from figregistry_kedro.datasets import (
    FigureDataSet, 
    FigureDataSetError,
    StyleResolutionCache,
    create_figure_dataset,
    validate_figure_dataset_config
)

# Import matplotlib fixtures for base functionality
from .matplotlib_fixtures import (
    clean_matplotlib_state,
    sample_figure_fixtures,
    subplot_fixtures,
    rcparams_testing_fixtures,
    figure_format_fixtures,
    performance_figure_fixtures,
    temporary_output_directory
)


class MockKedroContext:
    """Mock Kedro context for testing dataset integration."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.catalog = Mock()
        self.config_loader = Mock()
        
    def get_config_loader(self):
        return self.config_loader


class MockFigRegistryBridge:
    """Mock FigRegistry configuration bridge for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'condition_styles': {
                'exploratory': {'color': 'blue', 'linestyle': '-'},
                'presentation': {'color': 'red', 'linestyle': '--'},
                'publication': {'color': 'black', 'linestyle': '-'}
            },
            'paths': {
                'exploratory': 'data/01_raw',
                'presentation': 'data/08_reporting',
                'publication': 'outputs/figures'
            }
        }
    
    def get_style(self, condition: str) -> Dict[str, Any]:
        """Mock get_style implementation."""
        return self.config.get('condition_styles', {}).get(condition, {})
    
    def init_config(self, **kwargs):
        """Mock init_config implementation."""
        pass


@pytest.fixture(scope="function")
def sample_matplotlib_figure(clean_matplotlib_state, sample_figure_fixtures):
    """
    Comprehensive matplotlib figure fixtures for dataset save/load testing.
    
    Provides various figure types (line plots, bar charts, scatter plots, histograms,
    subplots) with consistent styling and metadata for testing FigureDataSet
    functionality across different visualization scenarios.
    
    Supports requirement: Section 5.2.6 - comprehensive figure type coverage for
    dataset testing with automated styling application validation.
    
    Args:
        clean_matplotlib_state: Ensures clean matplotlib environment
        sample_figure_fixtures: Base matplotlib figures for enhancement
        
    Returns:
        dict: Enhanced figure objects with dataset-specific metadata and configurations
    """
    enhanced_figures = {}
    
    # Enhance base figures with dataset-specific metadata
    for fig_type, figure in sample_figure_fixtures.items():
        # Add FigureDataSet-compatible metadata
        figure._figregistry_dataset_metadata = {
            'purpose': 'testing',
            'condition_param': 'experiment_type',
            'figure_type': fig_type,
            'test_scenario': 'sample_data',
            'creation_timestamp': time.time(),
            'expected_style_properties': ['color', 'linestyle', 'linewidth'],
            'validation_checkpoints': {
                'pre_style': True,
                'post_style': False,
                'saved': False
            }
        }
        
        enhanced_figures[fig_type] = {
            'figure': figure,
            'metadata': figure._figregistry_dataset_metadata,
            'expected_outputs': {
                'exploratory': f'data/01_raw/{fig_type}_exploratory.png',
                'presentation': f'data/08_reporting/{fig_type}_presentation.png',
                'publication': f'outputs/figures/{fig_type}_publication.pdf'
            },
            'test_conditions': {
                'exploratory': {'experiment_type': 'exploratory'},
                'presentation': {'experiment_type': 'presentation'},
                'publication': {'experiment_type': 'publication'},
                'custom_condition': {'experiment_type': 'custom_experiment_1'}
            },
            'performance_baseline': {
                'matplotlib_save_time': None,  # To be measured during tests
                'figregistry_save_time': None,  # To be measured during tests
                'overhead_percentage': None  # To be calculated
            }
        }
    
    # Add specialized figures for edge cases
    
    # Empty figure for edge case testing
    fig_empty, ax_empty = plt.subplots(figsize=(6, 4))
    ax_empty.set_title('Empty Figure Test')
    enhanced_figures['empty'] = {
        'figure': fig_empty,
        'metadata': {
            'purpose': 'edge_case_testing',
            'condition_param': 'test_condition',
            'figure_type': 'empty',
            'test_scenario': 'edge_case'
        },
        'expected_behavior': 'should_save_successfully',
        'test_conditions': {'test_condition': 'empty_figure'}
    }
    
    # Large figure for performance testing
    fig_large, ax_large = plt.subplots(figsize=(20, 15))
    x_large = np.linspace(0, 100, 10000)
    y_large = np.sin(x_large) * np.exp(-x_large/50)
    ax_large.plot(x_large, y_large, linewidth=0.5)
    ax_large.set_title('Large Figure Performance Test')
    enhanced_figures['large'] = {
        'figure': fig_large,
        'metadata': {
            'purpose': 'performance_testing',
            'condition_param': 'performance_test',
            'figure_type': 'large',
            'test_scenario': 'performance_validation'
        },
        'performance_requirements': {
            'max_save_time': 1.0,  # 1 second max
            'max_overhead_percentage': 5.0
        },
        'test_conditions': {'performance_test': 'large_figure'}
    }
    
    # Figure with complex styling for validation
    fig_complex_style, ax_complex = plt.subplots(figsize=(10, 8))
    x = np.linspace(0, 10, 100)
    for i, style in enumerate(['-', '--', '-.', ':']):
        y = np.sin(x + i * 0.5)
        ax_complex.plot(x, y, linestyle=style, linewidth=2, label=f'Style {i+1}')
    ax_complex.set_title('Complex Styling Test Figure')
    ax_complex.legend()
    ax_complex.grid(True, alpha=0.3)
    enhanced_figures['complex_style'] = {
        'figure': fig_complex_style,
        'metadata': {
            'purpose': 'style_validation',
            'condition_param': 'style_test',
            'figure_type': 'complex_style',
            'test_scenario': 'style_application'
        },
        'style_requirements': {
            'must_preserve_legend': True,
            'must_preserve_grid': True,
            'must_apply_condition_styles': True
        },
        'test_conditions': {'style_test': 'complex_styling'}
    }
    
    return enhanced_figures


@pytest.fixture(scope="function")
def catalog_with_figuredataset(temporary_output_directory):
    """
    Catalog configuration fixture with comprehensive FigureDataSet entries.
    
    Provides catalog configurations covering all FigureDataSet parameters including
    purpose, condition_param, style_params, save_args, and versioning settings
    for testing catalog integration and parameter validation.
    
    Supports requirement: F-005.2 - catalog configuration testing with complete
    parameter coverage and validation scenarios.
    
    Args:
        temporary_output_directory: Pytest temporary directory for outputs
        
    Returns:
        dict: Comprehensive catalog configurations for FigureDataSet testing
    """
    base_output_dir = temporary_output_directory
    
    # Core catalog entries with different parameter combinations
    catalog_entries = {
        # Basic configuration
        'basic_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'basic' / 'basic_figure.png'),
            'purpose': 'exploratory'
        },
        
        # Full parameter configuration
        'full_config_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'full' / 'full_config_figure.png'),
            'purpose': 'presentation',
            'condition_param': 'experiment_condition',
            'style_params': {
                'font.size': 14,
                'figure.dpi': 300,
                'axes.linewidth': 1.5,
                'figure.facecolor': 'white'
            },
            'save_args': {
                'bbox_inches': 'tight',
                'transparent': False,
                'pad_inches': 0.1
            }
        },
        
        # Publication quality configuration
        'publication_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'publication' / 'publication_figure.pdf'),
            'purpose': 'publication',
            'condition_param': 'publication_type',
            'style_params': {
                'font.family': 'serif',
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'legend.fontsize': 10,
                'figure.dpi': 600,
                'lines.linewidth': 1.0,
                'mathtext.fontset': 'dejavuserif'
            },
            'save_args': {
                'format': 'pdf',
                'bbox_inches': 'tight',
                'pad_inches': 0.05,
                'metadata': {
                    'Title': 'Publication Figure',
                    'Creator': 'FigRegistry-Kedro Plugin'
                }
            }
        },
        
        # Versioned dataset configuration
        'versioned_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'versioned' / 'versioned_figure.png'),
            'purpose': 'exploratory',
            'condition_param': 'version_test',
            'versioned': True
        },
        
        # Multi-format output configuration
        'multi_format_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'multi' / 'multi_format_figure.svg'),
            'purpose': 'presentation',
            'condition_param': 'format_test',
            'style_params': {
                'figure.dpi': 'figure',  # Use figure's native DPI for vector
                'savefig.transparent': True
            },
            'save_args': {
                'format': 'svg',
                'bbox_inches': 'tight',
                'transparent': True
            }
        },
        
        # High-DPI configuration
        'high_dpi_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'high_dpi' / 'high_dpi_figure.png'),
            'purpose': 'presentation',
            'condition_param': 'dpi_test',
            'style_params': {
                'figure.dpi': 300,
                'savefig.dpi': 300
            },
            'save_args': {
                'dpi': 300,
                'bbox_inches': 'tight'
            }
        },
        
        # Custom styling configuration
        'custom_style_figure': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'custom' / 'custom_style_figure.png'),
            'purpose': 'exploratory',
            'condition_param': 'custom_style_test',
            'style_params': {
                'axes.facecolor': '#f0f0f0',
                'axes.edgecolor': '#333333',
                'axes.linewidth': 2.0,
                'grid.alpha': 0.5,
                'grid.color': '#888888',
                'font.family': 'monospace',
                'font.size': 11
            }
        }
    }
    
    # Environment-specific catalog configurations
    environment_catalogs = {
        'development': {
            **catalog_entries,
            # Override some settings for development
            'dev_figure': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_output_dir / 'dev' / 'dev_figure.png'),
                'purpose': 'exploratory',
                'condition_param': 'dev_condition',
                'style_params': {
                    'font.size': 10,
                    'figure.dpi': 100  # Lower DPI for faster development
                }
            }
        },
        
        'production': {
            **catalog_entries,
            # Override some settings for production
            'prod_figure': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_output_dir / 'prod' / 'prod_figure.png'),
                'purpose': 'publication',
                'condition_param': 'prod_condition',
                'style_params': {
                    'font.size': 12,
                    'figure.dpi': 300,  # High DPI for production
                    'savefig.bbox': 'tight'
                },
                'save_args': {
                    'bbox_inches': 'tight',
                    'transparent': False,
                    'optimize': True
                }
            }
        }
    }
    
    # Parameter validation test cases
    parameter_validation_cases = {
        'minimal_valid': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'minimal' / 'minimal.png')
        },
        
        'all_optional_params': {
            'type': 'figregistry_kedro.datasets.FigureDataSet',
            'filepath': str(base_output_dir / 'all_optional' / 'all_optional.png'),
            'purpose': 'testing',
            'condition_param': 'test_condition',
            'style_params': {'font.size': 12},
            'save_args': {'dpi': 150},
            'load_args': {},
            'version': '1.0.0'
        }
    }
    
    return {
        'base_entries': catalog_entries,
        'environment_catalogs': environment_catalogs,
        'validation_cases': parameter_validation_cases,
        'output_directory': base_output_dir,
        'supported_formats': ['png', 'pdf', 'svg', 'jpg', 'eps'],
        'required_directories': [
            'basic', 'full', 'publication', 'versioned', 
            'multi', 'high_dpi', 'custom', 'dev', 'prod',
            'minimal', 'all_optional'
        ]
    }


@pytest.fixture(scope="function")
def dataset_parameter_scenarios():
    """
    Parameter testing scenarios for condition extraction and style application.
    
    Provides comprehensive scenarios for testing parameter resolution, condition
    extraction from pipeline context, and style application logic within
    FigureDataSet operations.
    
    Supports requirement: Dynamic condition resolution and parameter validation
    for pipeline context integration.
    
    Returns:
        dict: Parameter scenarios covering various pipeline context configurations
    """
    # Basic parameter resolution scenarios
    basic_scenarios = {
        'simple_condition': {
            'pipeline_params': {'experiment_type': 'exploratory'},
            'dataset_config': {
                'condition_param': 'experiment_type',
                'purpose': 'exploratory'
            },
            'expected_condition': 'exploratory',
            'expected_style_calls': 1
        },
        
        'nested_parameter': {
            'pipeline_params': {
                'experiment': {
                    'type': 'presentation',
                    'variant': 'high_contrast'
                }
            },
            'dataset_config': {
                'condition_param': 'experiment.type',
                'purpose': 'presentation'
            },
            'expected_condition': 'presentation',
            'expected_style_calls': 1
        },
        
        'fallback_to_purpose': {
            'pipeline_params': {'other_param': 'value'},
            'dataset_config': {
                'condition_param': 'missing_param',
                'purpose': 'publication'
            },
            'expected_condition': 'publication',  # Falls back to purpose
            'expected_style_calls': 1
        },
        
        'no_condition_param': {
            'pipeline_params': {'experiment_type': 'exploratory'},
            'dataset_config': {
                'purpose': 'presentation'
            },
            'expected_condition': 'presentation',  # Uses purpose as condition
            'expected_style_calls': 1
        }
    }
    
    # Complex parameter scenarios
    complex_scenarios = {
        'dynamic_condition_resolution': {
            'pipeline_params': {
                'experiment_metadata': {
                    'condition': 'custom_experiment_1',
                    'parameters': {
                        'sample_size': 1000,
                        'treatment': 'variant_a'
                    }
                }
            },
            'dataset_config': {
                'condition_param': 'experiment_metadata.condition',
                'purpose': 'exploratory',
                'style_params': {
                    'figure.dpi': 300,
                    'font.size': 12
                }
            },
            'expected_condition': 'custom_experiment_1',
            'expected_style_calls': 1
        },
        
        'conditional_style_override': {
            'pipeline_params': {'experiment_type': 'high_dpi_presentation'},
            'dataset_config': {
                'condition_param': 'experiment_type',
                'purpose': 'presentation',
                'style_params': {
                    'figure.dpi': 600,  # Override default
                    'font.size': 16
                }
            },
            'expected_condition': 'high_dpi_presentation',
            'expected_style_calls': 1,
            'expected_style_merge': True
        },
        
        'multi_level_parameters': {
            'pipeline_params': {
                'experiment': {
                    'config': {
                        'visualization': {
                            'type': 'publication_ready',
                            'format': 'pdf'
                        }
                    }
                }
            },
            'dataset_config': {
                'condition_param': 'experiment.config.visualization.type',
                'purpose': 'publication'
            },
            'expected_condition': 'publication_ready',
            'expected_style_calls': 1
        }
    }
    
    # Error and edge case scenarios
    edge_case_scenarios = {
        'empty_parameters': {
            'pipeline_params': {},
            'dataset_config': {
                'condition_param': 'missing_param',
                'purpose': 'exploratory'
            },
            'expected_condition': 'exploratory',  # Fallback to purpose
            'expected_style_calls': 1
        },
        
        'none_parameter_value': {
            'pipeline_params': {'experiment_type': None},
            'dataset_config': {
                'condition_param': 'experiment_type',
                'purpose': 'presentation'
            },
            'expected_condition': 'None',  # Converted to string
            'expected_style_calls': 1
        },
        
        'numeric_condition': {
            'pipeline_params': {'experiment_id': 12345},
            'dataset_config': {
                'condition_param': 'experiment_id',
                'purpose': 'exploratory'
            },
            'expected_condition': '12345',  # Converted to string
            'expected_style_calls': 1
        },
        
        'boolean_condition': {
            'pipeline_params': {'use_enhanced_style': True},
            'dataset_config': {
                'condition_param': 'use_enhanced_style',
                'purpose': 'presentation'
            },
            'expected_condition': 'True',  # Converted to string
            'expected_style_calls': 1
        }
    }
    
    # Performance testing scenarios
    performance_scenarios = {
        'cache_hit_scenario': {
            'pipeline_params': {'experiment_type': 'cached_condition'},
            'dataset_config': {
                'condition_param': 'experiment_type',
                'purpose': 'exploratory'
            },
            'expected_condition': 'cached_condition',
            'cache_setup': True,
            'expected_cache_hits': 1
        },
        
        'cache_miss_scenario': {
            'pipeline_params': {'experiment_type': 'new_condition'},
            'dataset_config': {
                'condition_param': 'experiment_type',
                'purpose': 'presentation'
            },
            'expected_condition': 'new_condition',
            'cache_setup': False,
            'expected_cache_misses': 1
        },
        
        'rapid_parameter_changes': {
            'parameter_sequence': [
                {'experiment_type': 'condition_1'},
                {'experiment_type': 'condition_2'},
                {'experiment_type': 'condition_1'},  # Should hit cache
                {'experiment_type': 'condition_3'},
                {'experiment_type': 'condition_2'}   # Should hit cache
            ],
            'dataset_config': {
                'condition_param': 'experiment_type',
                'purpose': 'exploratory'
            },
            'expected_cache_hits': 2,
            'expected_cache_misses': 3
        }
    }
    
    return {
        'basic': basic_scenarios,
        'complex': complex_scenarios,
        'edge_cases': edge_case_scenarios,
        'performance': performance_scenarios,
        'validation_rules': {
            'condition_must_be_string': True,
            'missing_params_fallback_to_purpose': True,
            'style_params_override_base_styles': True,
            'cache_key_includes_all_relevant_params': True
        }
    }


@pytest.fixture(scope="function")
def versioned_dataset_fixtures(temporary_output_directory):
    """
    Kedro versioning compatibility testing fixtures.
    
    Provides comprehensive scenarios for testing FigureDataSet integration with
    Kedro's versioning system, including version timestamp generation, version
    comparison, and versioned catalog behavior.
    
    Supports requirement: Kedro versioning system compatibility per dataset
    versioning workflow requirements.
    
    Args:
        temporary_output_directory: Pytest temporary directory for versioned outputs
        
    Returns:
        dict: Versioning scenarios and configuration for Kedro compatibility testing
    """
    base_dir = temporary_output_directory / 'versioned'
    base_dir.mkdir(exist_ok=True)
    
    # Basic versioning scenarios
    versioning_scenarios = {
        'timestamp_versioning': {
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_dir / 'timestamp' / 'figure.png'),
                'purpose': 'exploratory',
                'versioned': True
            },
            'expected_behavior': {
                'creates_version_directory': True,
                'uses_timestamp_format': True,
                'version_format': 'YYYY-MM-DD_HH-mm-ss.sss'
            }
        },
        
        'explicit_version': {
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_dir / 'explicit' / 'figure.png'),
                'purpose': 'publication',
                'version': '2023-12-01T10:30:00.000Z'
            },
            'expected_behavior': {
                'uses_explicit_version': True,
                'version_string': '2023-12-01T10:30:00.000Z'
            }
        },
        
        'version_comparison': {
            'versions': [
                '2023-12-01T10:00:00.000Z',
                '2023-12-01T10:30:00.000Z',
                '2023-12-01T11:00:00.000Z'
            ],
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_dir / 'comparison' / 'figure.png'),
                'purpose': 'presentation'
            },
            'expected_behavior': {
                'versions_sortable': True,
                'latest_version_accessible': True
            }
        }
    }
    
    # Advanced versioning features
    advanced_versioning = {
        'concurrent_versioning': {
            'scenario': 'multiple_writers_same_time',
            'dataset_configs': [
                {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': str(base_dir / 'concurrent' / f'figure_worker_{i}.png'),
                    'purpose': 'exploratory',
                    'versioned': True
                }
                for i in range(5)
            ],
            'expected_behavior': {
                'no_version_collisions': True,
                'all_versions_unique': True,
                'thread_safe_operations': True
            }
        },
        
        'version_metadata': {
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_dir / 'metadata' / 'figure.png'),
                'purpose': 'publication',
                'versioned': True,
                'version_metadata': {
                    'experiment_id': 'exp_001',
                    'researcher': 'test_user',
                    'pipeline_run_id': 'run_12345'
                }
            },
            'expected_behavior': {
                'metadata_preserved': True,
                'metadata_accessible': True
            }
        }
    }
    
    # Version lifecycle testing
    lifecycle_scenarios = {
        'version_creation': {
            'test_sequence': [
                'create_initial_version',
                'verify_version_exists',
                'create_second_version',
                'verify_both_versions_exist',
                'access_latest_version'
            ],
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_dir / 'lifecycle' / 'figure.png'),
                'purpose': 'exploratory',
                'versioned': True
            }
        },
        
        'version_loading': {
            'test_sequence': [
                'save_versioned_figure',
                'load_specific_version',
                'load_latest_version',
                'verify_content_consistency'
            ],
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(base_dir / 'loading' / 'figure.png'),
                'purpose': 'presentation',
                'versioned': True
            }
        }
    }
    
    # Integration with Kedro catalog versioning
    catalog_integration = {
        'catalog_versioned_dataset': {
            'catalog_entry': {
                'versioned_figure': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': str(base_dir / 'catalog' / 'versioned_figure.png'),
                    'purpose': 'exploratory',
                    'condition_param': 'version_test',
                    'versioned': True
                }
            },
            'pipeline_params': {'version_test': 'catalog_versioning'},
            'expected_behavior': {
                'integrates_with_catalog': True,
                'version_aware_loading': True,
                'consistent_versioning': True
            }
        },
        
        'mixed_versioned_unversioned': {
            'catalog_entries': {
                'versioned_figure': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': str(base_dir / 'mixed' / 'versioned.png'),
                    'purpose': 'publication',
                    'versioned': True
                },
                'unversioned_figure': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': str(base_dir / 'mixed' / 'unversioned.png'),
                    'purpose': 'exploratory',
                    'versioned': False
                }
            },
            'expected_behavior': {
                'both_work_independently': True,
                'no_cross_interference': True
            }
        }
    }
    
    # Performance implications of versioning
    performance_considerations = {
        'versioning_overhead': {
            'test_scenarios': [
                {
                    'name': 'unversioned_baseline',
                    'config': {
                        'versioned': False,
                        'filepath': str(base_dir / 'perf' / 'unversioned.png')
                    }
                },
                {
                    'name': 'versioned_performance',
                    'config': {
                        'versioned': True,
                        'filepath': str(base_dir / 'perf' / 'versioned.png')
                    }
                }
            ],
            'performance_requirements': {
                'max_versioning_overhead_ms': 20,  # Max 20ms overhead for versioning
                'max_overhead_percentage': 10  # Max 10% overhead
            }
        }
    }
    
    return {
        'basic_scenarios': versioning_scenarios,
        'advanced_features': advanced_versioning,
        'lifecycle_testing': lifecycle_scenarios,
        'catalog_integration': catalog_integration,
        'performance_considerations': performance_considerations,
        'test_utilities': {
            'version_pattern_regex': r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}',
            'timestamp_format': '%Y-%m-%d_%H-%M-%S.%f',
            'version_directory_structure': str(base_dir)
        }
    }


@pytest.fixture(scope="function")
def error_scenario_fixtures(temporary_output_directory):
    """
    Comprehensive error handling testing scenarios.
    
    Provides error scenarios including malformed catalog configurations, invalid
    parameters, file system errors, and dependency issues for testing robust
    error handling in FigureDataSet operations.
    
    Supports requirement: Robust error handling testing per Section 5.2.6
    requirements for comprehensive dataset validation.
    
    Args:
        temporary_output_directory: Pytest temporary directory for error testing
        
    Returns:
        dict: Error scenarios and configurations for robust error handling testing
    """
    error_dir = temporary_output_directory / 'errors'
    error_dir.mkdir(exist_ok=True)
    
    # Configuration validation errors
    config_errors = {
        'missing_filepath': {
            'invalid_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'purpose': 'exploratory'
                # Missing required 'filepath'
            },
            'expected_error': 'ValueError',
            'error_message_contains': 'filepath',
            'test_phase': 'initialization'
        },
        
        'invalid_purpose': {
            'invalid_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'invalid_purpose.png'),
                'purpose': 123  # Should be string
            },
            'expected_error': 'TypeError',
            'error_message_contains': 'purpose',
            'test_phase': 'initialization'
        },
        
        'invalid_condition_param': {
            'invalid_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'invalid_condition.png'),
                'condition_param': ['not', 'a', 'string']  # Should be string
            },
            'expected_error': 'TypeError',
            'error_message_contains': 'condition_param',
            'test_phase': 'initialization'
        },
        
        'invalid_style_params': {
            'invalid_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'invalid_style.png'),
                'style_params': 'not_a_dict'  # Should be dictionary
            },
            'expected_error': 'TypeError',
            'error_message_contains': 'style_params',
            'test_phase': 'initialization'
        },
        
        'malformed_save_args': {
            'invalid_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'malformed_save.png'),
                'save_args': {
                    'dpi': 'not_numeric',
                    'format': 'unsupported_format',
                    'transparent': 'not_boolean'
                }
            },
            'expected_error': 'ValueError',
            'error_message_contains': 'save_args',
            'test_phase': 'save_operation'
        }
    }
    
    # Dependency and import errors
    dependency_errors = {
        'missing_matplotlib': {
            'mock_scenario': 'matplotlib_not_available',
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'matplotlib',
            'test_phase': 'initialization'
        },
        
        'missing_kedro': {
            'mock_scenario': 'kedro_not_available',
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'kedro',
            'test_phase': 'initialization'
        },
        
        'missing_figregistry': {
            'mock_scenario': 'figregistry_not_available',
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'figregistry',
            'test_phase': 'initialization'
        },
        
        'corrupted_figregistry_config': {
            'mock_scenario': 'figregistry_config_error',
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'configuration',
            'test_phase': 'style_resolution'
        }
    }
    
    # File system and I/O errors
    filesystem_errors = {
        'readonly_directory': {
            'error_setup': {
                'create_readonly_dir': str(error_dir / 'readonly'),
                'target_file': str(error_dir / 'readonly' / 'figure.png')
            },
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'readonly' / 'figure.png'),
                'purpose': 'exploratory'
            },
            'expected_error': 'PermissionError',
            'error_message_contains': 'permission',
            'test_phase': 'save_operation'
        },
        
        'invalid_filepath': {
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': '/invalid/path/that/does/not/exist/figure.png',
                'purpose': 'exploratory'
            },
            'expected_error': 'FileNotFoundError',
            'error_message_contains': 'path',
            'test_phase': 'save_operation'
        },
        
        'disk_full_simulation': {
            'mock_scenario': 'disk_full',
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'disk_full' / 'figure.png'),
                'purpose': 'presentation'
            },
            'expected_error': 'OSError',
            'error_message_contains': 'disk',
            'test_phase': 'save_operation'
        }
    }
    
    # Data validation errors
    data_errors = {
        'invalid_figure_object': {
            'invalid_data': 'not_a_figure',
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'invalid_data.png'),
                'purpose': 'exploratory'
            },
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'Figure object',
            'test_phase': 'save_operation'
        },
        
        'none_figure': {
            'invalid_data': None,
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'none_data.png'),
                'purpose': 'exploratory'
            },
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'Figure object',
            'test_phase': 'save_operation'
        },
        
        'corrupted_figure': {
            'mock_scenario': 'corrupted_figure_object',
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'corrupted.png'),
                'purpose': 'presentation'
            },
            'expected_error': 'FigureDataSetError',
            'error_message_contains': 'corrupted',
            'test_phase': 'save_operation'
        }
    }
    
    # Style resolution and application errors
    style_errors = {
        'invalid_style_parameters': {
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'invalid_style.png'),
                'purpose': 'exploratory',
                'style_params': {
                    'font.size': 'invalid_size',
                    'nonexistent.param': 'value'
                }
            },
            'expected_error': 'ValueError',
            'error_message_contains': 'style',
            'test_phase': 'style_application'
        },
        
        'style_resolution_failure': {
            'mock_scenario': 'figregistry_get_style_fails',
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'style_fail.png'),
                'purpose': 'presentation',
                'condition_param': 'failing_condition'
            },
            'expected_behavior': 'graceful_degradation',
            'fallback_style': {},
            'test_phase': 'style_resolution'
        }
    }
    
    # Performance and resource errors
    resource_errors = {
        'memory_exhaustion': {
            'mock_scenario': 'memory_limit_exceeded',
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'memory_error.png'),
                'purpose': 'exploratory'
            },
            'expected_error': 'MemoryError',
            'test_phase': 'save_operation'
        },
        
        'timeout_scenario': {
            'mock_scenario': 'operation_timeout',
            'timeout_seconds': 1.0,
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'timeout.png'),
                'purpose': 'presentation'
            },
            'expected_error': 'TimeoutError',
            'test_phase': 'save_operation'
        }
    }
    
    # Error recovery scenarios
    recovery_scenarios = {
        'retry_after_transient_error': {
            'error_sequence': [
                {'error': 'TemporaryFileSystemError', 'should_retry': True},
                {'error': None, 'should_succeed': True}
            ],
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'retry.png'),
                'purpose': 'exploratory'
            },
            'expected_behavior': 'eventual_success'
        },
        
        'fallback_behavior': {
            'primary_failure': 'figregistry_save_failure',
            'fallback_method': 'matplotlib_direct_save',
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(error_dir / 'fallback.png'),
                'purpose': 'presentation'
            },
            'expected_behavior': 'fallback_success'
        }
    }
    
    return {
        'config_errors': config_errors,
        'dependency_errors': dependency_errors,
        'filesystem_errors': filesystem_errors,
        'data_errors': data_errors,
        'style_errors': style_errors,
        'resource_errors': resource_errors,
        'recovery_scenarios': recovery_scenarios,
        'error_test_utilities': {
            'mock_helpers': {
                'mock_readonly_directory': lambda path: path.chmod(0o444),
                'mock_missing_dependency': lambda module: patch(module, None),
                'mock_filesystem_error': lambda: patch('builtins.open', side_effect=OSError)
            },
            'validation_helpers': {
                'assert_error_message_quality': lambda msg: len(msg) > 10 and 'Error' in msg,
                'assert_proper_error_type': lambda exc, expected: isinstance(exc, expected),
                'assert_error_context_preserved': lambda exc: hasattr(exc, 'original_error')
            }
        }
    }


@pytest.fixture(scope="function")
def abstract_dataset_compliance_fixtures():
    """
    AbstractDataSet interface compliance validation fixtures.
    
    Provides comprehensive test scenarios for validating that FigureDataSet
    correctly implements the Kedro AbstractDataSet interface, including
    all required methods, proper error handling, and interface contracts.
    
    Supports requirement: F-005 - AbstractDataSet interface compliance
    validation for Kedro integration.
    
    Returns:
        dict: Interface compliance testing scenarios and validation utilities
    """
    # Required AbstractDataSet methods
    required_methods = {
        '_save': {
            'method_name': '_save',
            'required': True,
            'signature': 'def _save(self, data) -> None',
            'test_scenarios': [
                {
                    'name': 'valid_figure_save',
                    'input_type': 'matplotlib.figure.Figure',
                    'expected_behavior': 'successful_save'
                },
                {
                    'name': 'invalid_input_type',
                    'input_type': 'str',
                    'expected_behavior': 'raises_error'
                }
            ]
        },
        
        '_load': {
            'method_name': '_load',
            'required': True,
            'signature': 'def _load(self)',
            'test_scenarios': [
                {
                    'name': 'load_not_supported',
                    'expected_behavior': 'raises_error',
                    'expected_error': 'FigureDataSetError'
                }
            ]
        },
        
        '_describe': {
            'method_name': '_describe',
            'required': True,
            'signature': 'def _describe(self) -> Dict[str, Any]',
            'test_scenarios': [
                {
                    'name': 'description_completeness',
                    'expected_keys': [
                        'filepath', 'purpose', 'condition_param',
                        'style_params', 'save_args', 'version',
                        'operation_stats', 'cache_stats', 'dependencies'
                    ],
                    'expected_behavior': 'returns_dict'
                }
            ]
        },
        
        '_exists': {
            'method_name': '_exists',
            'required': True,
            'signature': 'def _exists(self) -> bool',
            'test_scenarios': [
                {
                    'name': 'file_exists_check',
                    'setup': 'create_test_file',
                    'expected_behavior': 'returns_true'
                },
                {
                    'name': 'file_not_exists_check',
                    'setup': 'no_file',
                    'expected_behavior': 'returns_false'
                }
            ]
        }
    }
    
    # Optional AbstractDataSet methods (if implemented)
    optional_methods = {
        '_release': {
            'method_name': '_release',
            'required': False,
            'signature': 'def _release(self) -> None',
            'test_scenarios': [
                {
                    'name': 'cleanup_resources',
                    'expected_behavior': 'no_error'
                }
            ]
        },
        
        '_invalidate_cache': {
            'method_name': '_invalidate_cache',
            'required': False,
            'signature': 'def _invalidate_cache(self) -> None',
            'test_scenarios': [
                {
                    'name': 'cache_invalidation',
                    'expected_behavior': 'cache_cleared'
                }
            ]
        }
    }
    
    # Interface contract validation
    interface_contracts = {
        'initialization_contract': {
            'description': 'Dataset must initialize with required parameters',
            'test_cases': [
                {
                    'name': 'minimal_initialization',
                    'params': {'filepath': '/tmp/test.png'},
                    'expected_behavior': 'successful_init'
                },
                {
                    'name': 'full_initialization',
                    'params': {
                        'filepath': '/tmp/test.png',
                        'purpose': 'exploratory',
                        'condition_param': 'test_condition',
                        'style_params': {'font.size': 12},
                        'save_args': {'dpi': 300},
                        'version': '1.0.0'
                    },
                    'expected_behavior': 'successful_init'
                }
            ]
        },
        
        'save_contract': {
            'description': 'Save operation must handle data correctly',
            'test_cases': [
                {
                    'name': 'save_matplotlib_figure',
                    'data_type': 'matplotlib.figure.Figure',
                    'expected_behavior': 'file_created'
                },
                {
                    'name': 'save_invalid_data',
                    'data_type': 'str',
                    'expected_behavior': 'raises_error'
                }
            ]
        },
        
        'error_handling_contract': {
            'description': 'Errors must be handled gracefully with informative messages',
            'test_cases': [
                {
                    'name': 'invalid_filepath',
                    'error_scenario': 'filesystem_error',
                    'expected_behavior': 'informative_error'
                },
                {
                    'name': 'invalid_data_type',
                    'error_scenario': 'type_error',
                    'expected_behavior': 'clear_error_message'
                }
            ]
        }
    }
    
    # Kedro integration compliance
    kedro_integration = {
        'catalog_integration': {
            'description': 'Dataset must work within Kedro catalog system',
            'test_scenarios': [
                {
                    'name': 'catalog_registration',
                    'catalog_config': {
                        'test_figure': {
                            'type': 'figregistry_kedro.datasets.FigureDataSet',
                            'filepath': 'data/test_figure.png'
                        }
                    },
                    'expected_behavior': 'successful_registration'
                },
                {
                    'name': 'versioned_catalog_entry',
                    'catalog_config': {
                        'versioned_figure': {
                            'type': 'figregistry_kedro.datasets.FigureDataSet',
                            'filepath': 'data/versioned_figure.png',
                            'versioned': True
                        }
                    },
                    'expected_behavior': 'version_support'
                }
            ]
        },
        
        'pipeline_integration': {
            'description': 'Dataset must integrate with Kedro pipeline execution',
            'test_scenarios': [
                {
                    'name': 'pipeline_output',
                    'pipeline_context': {
                        'parameters': {'experiment_type': 'test'},
                        'run_id': 'test_run_001'
                    },
                    'expected_behavior': 'context_aware_save'
                }
            ]
        }
    }
    
    # Performance compliance
    performance_compliance = {
        'response_time_requirements': {
            'save_operation': {
                'max_time_ms': 200,  # 200ms max for save operation
                'measurement_method': 'average_over_10_runs'
            },
            'describe_operation': {
                'max_time_ms': 10,  # 10ms max for describe
                'measurement_method': 'single_call'
            },
            'exists_check': {
                'max_time_ms': 50,  # 50ms max for exists check
                'measurement_method': 'average_over_5_runs'
            }
        },
        
        'memory_requirements': {
            'max_memory_overhead_mb': 10,  # 10MB max overhead
            'memory_leak_tolerance': 0,  # No memory leaks allowed
            'measurement_method': 'before_after_comparison'
        }
    }
    
    # Compatibility matrix
    compatibility_matrix = {
        'kedro_versions': [
            {'version': '0.18.0', 'supported': True, 'test_required': True},
            {'version': '0.18.14', 'supported': True, 'test_required': True},
            {'version': '0.19.0', 'supported': True, 'test_required': True},
            {'version': '0.19.8', 'supported': True, 'test_required': False}  # Latest
        ],
        
        'python_versions': [
            {'version': '3.10', 'supported': True, 'test_required': True},
            {'version': '3.11', 'supported': True, 'test_required': True},
            {'version': '3.12', 'supported': True, 'test_required': True}
        ]
    }
    
    # Validation utilities
    validation_utilities = {
        'method_signature_validator': {
            'check_method_exists': lambda obj, method: hasattr(obj, method),
            'check_method_callable': lambda obj, method: callable(getattr(obj, method, None)),
            'check_signature_compliance': lambda method, expected: True  # Mock implementation
        },
        
        'behavior_validators': {
            'assert_successful_save': lambda result: result is None,
            'assert_error_raised': lambda func, error_type: pytest.raises(error_type),
            'assert_return_type': lambda result, expected_type: isinstance(result, expected_type)
        },
        
        'performance_validators': {
            'measure_execution_time': lambda func: time.perf_counter,
            'measure_memory_usage': lambda func: None,  # Mock implementation
            'assert_within_threshold': lambda measured, threshold: measured <= threshold
        }
    }
    
    return {
        'required_methods': required_methods,
        'optional_methods': optional_methods,
        'interface_contracts': interface_contracts,
        'kedro_integration': kedro_integration,
        'performance_compliance': performance_compliance,
        'compatibility_matrix': compatibility_matrix,
        'validation_utilities': validation_utilities,
        'compliance_checklist': {
            'implements_required_methods': False,  # To be validated
            'handles_errors_gracefully': False,
            'integrates_with_catalog': False,
            'supports_versioning': False,
            'meets_performance_requirements': False,
            'maintains_backward_compatibility': False
        }
    }


@pytest.fixture(scope="function")
def performance_testing_fixtures(sample_matplotlib_figure, temporary_output_directory):
    """
    Performance testing fixtures for validating <5% overhead requirement.
    
    Provides comprehensive performance testing scenarios including baseline
    measurements, overhead calculations, and validation that FigureDataSet
    operations maintain <5% overhead compared to manual matplotlib operations.
    
    Supports requirement: Section 5.2.8 - <5% overhead validation compared
    to manual matplotlib operations.
    
    Args:
        sample_matplotlib_figure: Sample figures for performance testing
        temporary_output_directory: Directory for performance test outputs
        
    Returns:
        dict: Performance testing scenarios and measurement utilities
    """
    perf_dir = temporary_output_directory / 'performance'
    perf_dir.mkdir(exist_ok=True)
    
    # Performance measurement utilities
    class PerformanceMeasurement:
        def __init__(self):
            self.measurements = []
            self.baseline_times = {}
            self.figregistry_times = {}
            
        def measure_baseline_operation(self, operation_name: str, operation_func, *args, **kwargs):
            """Measure baseline matplotlib operation time."""
            start_time = time.perf_counter()
            result = operation_func(*args, **kwargs)
            end_time = time.perf_counter()
            
            operation_time = end_time - start_time
            self.baseline_times[operation_name] = operation_time
            
            return result, operation_time
        
        def measure_figregistry_operation(self, operation_name: str, operation_func, *args, **kwargs):
            """Measure FigureDataSet operation time."""
            start_time = time.perf_counter()
            result = operation_func(*args, **kwargs)
            end_time = time.perf_counter()
            
            operation_time = end_time - start_time
            self.figregistry_times[operation_name] = operation_time
            
            return result, operation_time
        
        def calculate_overhead(self, operation_name: str) -> float:
            """Calculate overhead percentage for an operation."""
            baseline = self.baseline_times.get(operation_name, 0)
            figregistry = self.figregistry_times.get(operation_name, 0)
            
            if baseline == 0:
                return 0.0
            
            overhead = ((figregistry - baseline) / baseline) * 100
            return overhead
        
        def get_summary(self) -> Dict[str, Any]:
            """Get performance measurement summary."""
            summary = {
                'baseline_times': self.baseline_times.copy(),
                'figregistry_times': self.figregistry_times.copy(),
                'overhead_percentages': {},
                'compliance': {}
            }
            
            for operation in self.baseline_times:
                if operation in self.figregistry_times:
                    overhead = self.calculate_overhead(operation)
                    summary['overhead_percentages'][operation] = overhead
                    summary['compliance'][operation] = overhead <= 5.0  # 5% threshold
            
            return summary
    
    # Performance test scenarios
    performance_scenarios = {
        'simple_figure_save': {
            'description': 'Basic figure save operation performance',
            'test_figure': 'line_plot',  # From sample_matplotlib_figure
            'baseline_operation': lambda fig, path: fig.savefig(path),
            'figregistry_operation': lambda dataset, fig: dataset._save(fig),
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(perf_dir / 'simple_save.png'),
                'purpose': 'exploratory'
            },
            'performance_requirements': {
                'max_overhead_percentage': 5.0,
                'max_absolute_overhead_ms': 50
            }
        },
        
        'complex_figure_save': {
            'description': 'Complex subplot figure save performance',
            'test_figure': 'complex_style',
            'baseline_operation': lambda fig, path: fig.savefig(path, dpi=300, bbox_inches='tight'),
            'figregistry_operation': lambda dataset, fig: dataset._save(fig),
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(perf_dir / 'complex_save.png'),
                'purpose': 'presentation',
                'style_params': {'figure.dpi': 300},
                'save_args': {'bbox_inches': 'tight'}
            },
            'performance_requirements': {
                'max_overhead_percentage': 5.0,
                'max_absolute_overhead_ms': 100
            }
        },
        
        'high_dpi_figure_save': {
            'description': 'High DPI figure save performance',
            'test_figure': 'large',
            'baseline_operation': lambda fig, path: fig.savefig(path, dpi=600),
            'figregistry_operation': lambda dataset, fig: dataset._save(fig),
            'dataset_config': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': str(perf_dir / 'high_dpi_save.png'),
                'purpose': 'publication',
                'style_params': {'figure.dpi': 600}
            },
            'performance_requirements': {
                'max_overhead_percentage': 5.0,
                'max_absolute_overhead_ms': 200
            }
        }
    }
    
    # Style application performance
    style_performance_scenarios = {
        'simple_style_application': {
            'description': 'Basic style parameter application',
            'style_params': {
                'font.size': 12,
                'figure.facecolor': 'white'
            },
            'baseline_operation': lambda params: plt.rcParams.update(params),
            'figregistry_operation': lambda dataset, condition: dataset._get_figure_style(condition),
            'performance_requirements': {
                'max_time_ms': 10,
                'max_overhead_percentage': 5.0
            }
        },
        
        'complex_style_application': {
            'description': 'Complex style parameter application',
            'style_params': {
                'font.size': 14, 'font.family': 'serif',
                'axes.titlesize': 16, 'axes.labelsize': 14,
                'xtick.labelsize': 12, 'ytick.labelsize': 12,
                'legend.fontsize': 12, 'figure.dpi': 300,
                'lines.linewidth': 1.5, 'axes.linewidth': 1.0
            },
            'baseline_operation': lambda params: plt.rcParams.update(params),
            'figregistry_operation': lambda dataset, condition: dataset._get_figure_style(condition),
            'performance_requirements': {
                'max_time_ms': 20,
                'max_overhead_percentage': 5.0
            }
        }
    }
    
    # Cache performance scenarios
    cache_performance_scenarios = {
        'cache_hit_performance': {
            'description': 'Style cache hit performance',
            'test_sequence': [
                'prime_cache_with_condition',
                'measure_cache_hit_time',
                'verify_sub_millisecond_response'
            ],
            'performance_requirements': {
                'max_cache_hit_time_ms': 1.0,
                'cache_hit_improvement_factor': 5.0  # Should be 5x faster than miss
            }
        },
        
        'cache_miss_performance': {
            'description': 'Style cache miss performance',
            'test_sequence': [
                'clear_cache',
                'measure_cache_miss_time',
                'verify_acceptable_miss_time'
            ],
            'performance_requirements': {
                'max_cache_miss_time_ms': 10.0
            }
        }
    }
    
    # Concurrent operation performance
    concurrent_performance_scenarios = {
        'parallel_save_operations': {
            'description': 'Multiple figures saved concurrently',
            'concurrency_levels': [1, 2, 4, 8],
            'test_figures': ['line_plot', 'bar_chart', 'scatter_plot', 'histogram'],
            'performance_requirements': {
                'linear_scaling_tolerance': 1.2,  # Up to 20% degradation acceptable
                'no_deadlocks': True,
                'thread_safety': True
            }
        },
        
        'concurrent_style_resolution': {
            'description': 'Style resolution under concurrent load',
            'concurrent_conditions': [
                'exploratory', 'presentation', 'publication',
                'custom_condition_1', 'custom_condition_2'
            ],
            'performance_requirements': {
                'cache_consistency': True,
                'no_race_conditions': True,
                'performance_degradation_max': 1.5  # Max 50% degradation
            }
        }
    }
    
    # Memory performance scenarios
    memory_performance_scenarios = {
        'memory_usage_tracking': {
            'description': 'Memory overhead during operations',
            'test_scenarios': [
                {
                    'name': 'baseline_memory',
                    'operation': 'matplotlib_save_only'
                },
                {
                    'name': 'figregistry_memory',
                    'operation': 'figregistry_dataset_save'
                }
            ],
            'performance_requirements': {
                'max_memory_overhead_mb': 10,
                'no_memory_leaks': True
            }
        },
        
        'large_figure_memory': {
            'description': 'Memory efficiency with large figures',
            'figure_sizes': [(20, 15), (30, 20), (40, 30)],
            'performance_requirements': {
                'memory_scaling_linear': True,
                'max_memory_multiplier': 2.0  # Max 2x memory usage
            }
        }
    }
    
    # Performance regression testing
    regression_scenarios = {
        'version_comparison': {
            'description': 'Performance comparison across versions',
            'baseline_version': '0.1.0',
            'current_version': '0.2.0',
            'performance_requirements': {
                'no_regression_tolerance': 1.1,  # Max 10% regression allowed
                'improvement_expectations': {
                    'cache_hit_time': 0.9,  # Expect 10% improvement
                    'style_resolution': 0.95  # Expect 5% improvement
                }
            }
        }
    }
    
    # Benchmarking utilities
    benchmarking_utilities = {
        'measurement_tools': {
            'timer': PerformanceMeasurement(),
            'memory_profiler': None,  # Mock for now
            'cpu_profiler': None      # Mock for now
        },
        
        'statistical_analysis': {
            'min_sample_size': 10,
            'confidence_level': 0.95,
            'outlier_removal': True,
            'statistical_significance_test': 't_test'
        },
        
        'reporting': {
            'generate_performance_report': lambda results: {
                'summary': results,
                'compliance': all(r['overhead'] <= 5.0 for r in results.values()),
                'recommendations': []
            },
            'create_performance_charts': lambda data: None,  # Mock implementation
            'export_benchmark_data': lambda data, path: None  # Mock implementation
        }
    }
    
    return {
        'measurement_utility': PerformanceMeasurement(),
        'performance_scenarios': performance_scenarios,
        'style_performance': style_performance_scenarios,
        'cache_performance': cache_performance_scenarios,
        'concurrent_performance': concurrent_performance_scenarios,
        'memory_performance': memory_performance_scenarios,
        'regression_scenarios': regression_scenarios,
        'benchmarking_utilities': benchmarking_utilities,
        'performance_thresholds': {
            'max_overhead_percentage': 5.0,
            'max_style_resolution_ms': 1.0,
            'max_save_operation_ms': 200.0,
            'max_memory_overhead_mb': 10.0,
            'min_cache_hit_rate': 0.8
        },
        'test_output_directory': perf_dir
    }


# Additional utility fixtures for comprehensive testing

@pytest.fixture(scope="function")
def mock_figregistry_bridge():
    """Mock FigRegistry configuration bridge for testing."""
    return MockFigRegistryBridge()


@pytest.fixture(scope="function")
def mock_kedro_context():
    """Mock Kedro context for testing dataset integration."""
    return MockKedroContext()


@pytest.fixture(scope="function")
def dataset_factory():
    """Factory function for creating FigureDataSet instances for testing."""
    def _create_dataset(**kwargs):
        default_config = {
            'filepath': '/tmp/test_figure.png',
            'purpose': 'testing'
        }
        config = {**default_config, **kwargs}
        return FigureDataSet(**config)
    return _create_dataset


@pytest.fixture(scope="function", autouse=True)
def cleanup_dataset_cache():
    """Automatically clear dataset cache between tests."""
    # Pre-test cleanup
    FigureDataSet.clear_cache()
    FigureDataSet.reset_performance_stats()
    
    yield
    
    # Post-test cleanup
    FigureDataSet.clear_cache()
    FigureDataSet.reset_performance_stats()


@pytest.fixture(scope="function")
def thread_pool_executor():
    """Thread pool executor for concurrent testing scenarios."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        yield executor


# Export all fixtures for test discovery
__all__ = [
    'sample_matplotlib_figure',
    'catalog_with_figuredataset',
    'dataset_parameter_scenarios',
    'versioned_dataset_fixtures',
    'error_scenario_fixtures',
    'abstract_dataset_compliance_fixtures',
    'performance_testing_fixtures',
    'mock_figregistry_bridge',
    'mock_kedro_context',
    'dataset_factory',
    'cleanup_dataset_cache',
    'thread_pool_executor'
]