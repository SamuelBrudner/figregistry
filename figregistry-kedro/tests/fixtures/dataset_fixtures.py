"""
FigureDataSet testing fixtures providing comprehensive test scenarios for automated 
matplotlib figure styling and Kedro catalog integration validation.

This module provides specialized fixtures for testing the FigureDataSet implementation
including catalog configuration scenarios, parameter validation, versioning compatibility,
error handling, and performance benchmarking. All fixtures ensure compliance with
Kedro's AbstractDataSet interface requirements and FigRegistry styling automation.

Key Fixture Categories:
- Sample matplotlib figure objects for dataset save/load testing per Section 5.2.6  
- Catalog entry configurations with FigureDataSet parameters per F-005.2
- Dataset parameter scenarios for condition parameter extraction and style application
- Versioned dataset fixtures for Kedro versioning compatibility testing per requirements
- Error scenario fixtures for robust error handling including malformed configurations
- AbstractDataSet interface compliance fixtures for validation testing per F-005
- Performance testing fixtures for validating <5% overhead per Section 5.2.8

All fixtures maintain isolation between tests and provide comprehensive coverage
of FigureDataSet functionality across different usage scenarios.
"""

import copy
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Ensure matplotlib uses non-interactive backend for testing
matplotlib.use('Agg')

# Import the FigureDataSet for testing
try:
    from figregistry_kedro.datasets import FigureDataSet, FigureDatasetError
    HAS_FIGUREDATASET = True
except ImportError:
    HAS_FIGUREDATASET = False
    FigureDataSet = None
    FigureDatasetError = Exception

# Import testing utilities from matplotlib fixtures
from .matplotlib_fixtures import GraphicsStateManager


class MockKedroContext:
    """
    Mock Kedro context for testing dataset integration.
    
    Simulates Kedro's context object providing pipeline parameters,
    configuration management, and versioning support for comprehensive
    FigureDataSet testing without requiring full Kedro installation.
    """
    
    def __init__(self, project_path: Path = None, env: str = "base"):
        self.project_path = project_path or Path.cwd()
        self.env = env
        self.catalog = MockDataCatalog()
        self.params = {}
        self.config_loader = MockConfigLoader()
        
    def get_config_loader(self):
        return self.config_loader


class MockDataCatalog:
    """
    Mock Kedro DataCatalog for testing catalog integration.
    
    Provides minimal catalog functionality for testing dataset
    configuration, versioning, and metadata management.
    """
    
    def __init__(self):
        self._datasets = {}
        self._versions = {}
        
    def add(self, data_set_name: str, data_set: Any, replace: bool = False):
        """Add dataset to catalog."""
        if data_set_name in self._datasets and not replace:
            raise ValueError(f"Dataset {data_set_name} already exists")
        self._datasets[data_set_name] = data_set
        
    def list(self) -> List[str]:
        """List all dataset names in catalog."""
        return list(self._datasets.keys())


class MockConfigLoader:
    """
    Mock Kedro ConfigLoader for testing configuration management.
    
    Simulates Kedro's configuration loading system including
    environment-specific configurations and parameter resolution.
    """
    
    def __init__(self):
        self.config_patterns = {}
        self.base_env = "base"
        self.run_env = "local"
        
    def get(self, *patterns: str) -> Dict[str, Any]:
        """Get configuration for specified patterns."""
        config = {}
        for pattern in patterns:
            if pattern in self.config_patterns:
                config.update(self.config_patterns[pattern])
        return config


@pytest.fixture(scope="function")
def mock_kedro_context(tmp_path):
    """
    Provides mock Kedro context for dataset testing.
    
    Creates a minimal Kedro context simulation that supports
    the essential interfaces required by FigureDataSet without
    requiring full Kedro installation.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        MockKedroContext: Mock context object for testing
    """
    context = MockKedroContext(project_path=tmp_path)
    
    # Add common pipeline parameters
    context.params = {
        "experiment_condition": "exploratory",
        "model_type": "linear_regression", 
        "dataset_version": "v1.0",
        "analysis_type": "statistical"
    }
    
    # Configure mock config loader with FigRegistry settings
    context.config_loader.config_patterns = {
        "figregistry": {
            "condition_styles": {
                "exploratory": {
                    "color": "#A8E6CF",
                    "linewidth": 1.5,
                    "marker": "o"
                },
                "presentation": {
                    "color": "#FFB6C1",
                    "linewidth": 2.0,
                    "marker": "s"
                },
                "publication": {
                    "color": "#1A1A1A",
                    "linewidth": 2.5,
                    "marker": "^"
                }
            },
            "paths": {
                "exploratory": "data/08_reporting/exploratory",
                "presentation": "data/08_reporting/presentation",
                "publication": "data/08_reporting/publication"
            }
        }
    }
    
    return context


@pytest.fixture(scope="function")
def sample_matplotlib_figure():
    """
    Creates sample matplotlib figure with various plot types for dataset testing.
    
    Generates a comprehensive figure containing multiple plot elements
    including lines, scatter points, and annotations that can be used
    to validate FigRegistry styling application and save operations.
    
    Returns:
        matplotlib.figure.Figure: Sample figure object for testing
    """
    with GraphicsStateManager():
        # Create figure with subplots for comprehensive testing
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Generate test data
        np.random.seed(42)  # Reproducible test data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
        y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
        
        # Top-left: Line plot with multiple series
        axes[0, 0].plot(x, y1, 'b-', label='Sin + noise', linewidth=2)
        axes[0, 0].plot(x, y2, 'r--', label='Cos + noise', linewidth=2)
        axes[0, 0].set_title('Time Series Data')
        axes[0, 0].set_xlabel('X Values')
        axes[0, 0].set_ylabel('Y Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-right: Scatter plot
        scatter_x = np.random.randn(50)
        scatter_y = scatter_x * 2 + np.random.randn(50) * 0.5
        colors = np.random.rand(50)
        scatter = axes[0, 1].scatter(scatter_x, scatter_y, c=colors, alpha=0.6)
        axes[0, 1].set_title('Scatter Plot')
        axes[0, 1].set_xlabel('X Values')
        axes[0, 1].set_ylabel('Y Values')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Bottom-left: Bar chart
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 56, 78, 32]
        bars = axes[1, 0].bar(categories, values, color='lightblue')
        axes[1, 0].set_title('Category Analysis')
        axes[1, 0].set_xlabel('Categories')
        axes[1, 0].set_ylabel('Values')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           str(value), ha='center', va='bottom')
        
        # Bottom-right: Histogram
        hist_data = np.random.normal(0, 1, 1000)
        axes[1, 1].hist(hist_data, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Distribution Analysis')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(hist_data), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(hist_data):.2f}')
        axes[1, 1].legend()
        
        # Add overall figure title and adjust layout
        fig.suptitle('Sample Multi-Panel Figure for Dataset Testing', fontsize=14)
        plt.tight_layout()
        
        # Add metadata for testing
        fig._figregistry_test_metadata = {
            'purpose': 'testing',
            'complexity': 'medium',
            'elements': ['line', 'scatter', 'bar', 'histogram'],
            'created_by': 'dataset_fixtures'
        }
        
        return fig


@pytest.fixture(scope="function")
def catalog_with_figuredataset(tmp_path):
    """
    Provides catalog entries with FigureDataSet configurations for testing.
    
    Creates comprehensive catalog configurations covering different
    FigureDataSet parameter combinations including purpose categories,
    condition parameters, styling overrides, and versioning options.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        dict: Catalog configurations with FigureDataSet entries
    """
    base_output_path = tmp_path / "data" / "08_reporting"
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    catalog_configs = {
        # Basic configuration with minimal parameters
        "basic_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": str(base_output_path / "basic_figure.png"),
            "purpose": "exploratory"
        },
        
        # Standard configuration with condition parameter
        "analysis_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet", 
            "filepath": str(base_output_path / "analysis_figure.png"),
            "purpose": "presentation",
            "condition_param": "experiment_condition",
            "format_kwargs": {
                "dpi": 200,
                "bbox_inches": "tight"
            }
        },
        
        # Advanced configuration with styling overrides
        "publication_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": str(base_output_path / "publication_figure.pdf"),
            "purpose": "publication",
            "condition_param": "model_type",
            "style_params": {
                "font.size": 14,
                "axes.linewidth": 1.5,
                "figure.dpi": 300
            },
            "format_kwargs": {
                "dpi": 300,
                "bbox_inches": "tight",
                "transparent": False,
                "metadata": {
                    "Title": "Publication Figure",
                    "Creator": "FigRegistry-Kedro Plugin"
                }
            }
        },
        
        # Versioned configuration
        "versioned_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": str(base_output_path / "versioned_figure.png"),
            "purpose": "exploratory",
            "condition_param": "dataset_version",
            "versioned": True,
            "metadata": {
                "description": "Versioned figure for experiment tracking",
                "tags": ["experiment", "versioned"]
            }
        },
        
        # Configuration with all optional parameters
        "comprehensive_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": str(base_output_path / "comprehensive_figure.svg"),
            "purpose": "presentation", 
            "condition_param": "analysis_type",
            "style_params": {
                "color": "#2E86AB",
                "linewidth": 2.5,
                "marker": "o",
                "alpha": 0.8
            },
            "format_kwargs": {
                "transparent": True,
                "bbox_inches": "tight"
            },
            "enable_caching": True,
            "metadata": {
                "project": "figregistry-kedro-testing",
                "version": "1.0.0"
            }
        }
    }
    
    return catalog_configs


@pytest.fixture(scope="function")
def dataset_parameter_scenarios():
    """
    Provides parameter scenarios for testing condition parameter extraction and style application.
    
    Creates comprehensive test scenarios covering valid and edge cases
    for FigureDataSet parameter handling, including condition resolution,
    style parameter validation, and error scenarios.
    
    Returns:
        dict: Parameter scenarios organized by test category
    """
    return {
        # Valid parameter combinations
        "valid_scenarios": {
            "minimal_config": {
                "filepath": "data/08_reporting/minimal.png",
                "purpose": "exploratory"
            },
            "standard_config": {
                "filepath": "data/08_reporting/standard.png", 
                "purpose": "presentation",
                "condition_param": "experiment_type"
            },
            "comprehensive_config": {
                "filepath": "data/08_reporting/comprehensive.png",
                "purpose": "publication",
                "condition_param": "model_version", 
                "style_params": {
                    "font.size": 12,
                    "linewidth": 2.0,
                    "color": "#FF5733"
                },
                "format_kwargs": {
                    "dpi": 300,
                    "bbox_inches": "tight"
                },
                "versioned": True
            }
        },
        
        # Purpose category scenarios
        "purpose_scenarios": {
            "exploratory": {
                "filepath": "data/08_reporting/exploratory.png",
                "purpose": "exploratory",
                "expected_style": {
                    "color": "#A8E6CF",
                    "linewidth": 1.5,
                    "marker": "o"
                }
            },
            "presentation": {
                "filepath": "data/08_reporting/presentation.png", 
                "purpose": "presentation",
                "expected_style": {
                    "color": "#FFB6C1",
                    "linewidth": 2.0,
                    "marker": "s"
                }
            },
            "publication": {
                "filepath": "data/08_reporting/publication.png",
                "purpose": "publication",
                "expected_style": {
                    "color": "#1A1A1A",
                    "linewidth": 2.5,
                    "marker": "^"
                }
            }
        },
        
        # Condition parameter scenarios
        "condition_scenarios": {
            "simple_condition": {
                "filepath": "data/08_reporting/simple.png",
                "purpose": "exploratory",
                "condition_param": "experiment_type",
                "mock_context_params": {
                    "experiment_type": "baseline"
                }
            },
            "complex_condition": {
                "filepath": "data/08_reporting/complex.png",
                "purpose": "presentation", 
                "condition_param": "model_config",
                "mock_context_params": {
                    "model_config": "random_forest_optimized"
                }
            },
            "missing_condition": {
                "filepath": "data/08_reporting/missing.png",
                "purpose": "exploratory",
                "condition_param": "missing_param",
                "mock_context_params": {
                    "other_param": "value"
                },
                "expected_fallback": "exploratory"  # Should fall back to purpose
            }
        },
        
        # Style parameter override scenarios  
        "style_override_scenarios": {
            "color_override": {
                "filepath": "data/08_reporting/color_override.png",
                "purpose": "exploratory",
                "style_params": {
                    "color": "#FF0000"  # Override default color
                },
                "expected_merged_style": {
                    "color": "#FF0000",
                    "linewidth": 1.5,  # From base style
                    "marker": "o"      # From base style
                }
            },
            "multiple_overrides": {
                "filepath": "data/08_reporting/multiple_overrides.png",
                "purpose": "presentation",
                "style_params": {
                    "color": "#0000FF",
                    "linewidth": 3.0,
                    "alpha": 0.7
                },
                "expected_merged_style": {
                    "color": "#0000FF",
                    "linewidth": 3.0,
                    "alpha": 0.7,
                    "marker": "s"  # From base style
                }
            }
        }
    }


@pytest.fixture(scope="function") 
def versioned_dataset_fixtures(tmp_path):
    """
    Creates versioned dataset fixtures for Kedro versioning compatibility testing.
    
    Provides comprehensive test scenarios for Kedro's dataset versioning
    system integration including load/save version specifications,
    version timestamp handling, and versioned catalog interactions.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        dict: Versioned dataset configurations and test scenarios
    """
    versioned_output_path = tmp_path / "data" / "08_reporting" / "versioned"
    versioned_output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate mock version timestamps
    base_timestamp = "2024-01-15T10-30-45.123Z"
    version_timestamps = [
        "2024-01-15T10-30-45.123Z",
        "2024-01-15T10-35-12.456Z", 
        "2024-01-15T10-40-33.789Z"
    ]
    
    return {
        # Basic versioning scenarios
        "basic_versioning": {
            "config": {
                "type": "figregistry_kedro.datasets.FigureDataSet",
                "filepath": str(versioned_output_path / "basic.png"),
                "purpose": "exploratory",
                "versioned": True
            },
            "test_versions": version_timestamps,
            "expected_files": [
                f"basic_{ts}.png" for ts in version_timestamps
            ]
        },
        
        # Load version specification
        "load_version_test": {
            "config": {
                "type": "figregistry_kedro.datasets.FigureDataSet", 
                "filepath": str(versioned_output_path / "load_test.png"),
                "purpose": "presentation",
                "versioned": True,
                "load_version": version_timestamps[1]  # Specific version to load
            },
            "expected_load_file": f"load_test_{version_timestamps[1]}.png"
        },
        
        # Save version specification  
        "save_version_test": {
            "config": {
                "type": "figregistry_kedro.datasets.FigureDataSet",
                "filepath": str(versioned_output_path / "save_test.png"),
                "purpose": "publication", 
                "versioned": True,
                "save_version": version_timestamps[2]  # Specific version to save
            },
            "expected_save_file": f"save_test_{version_timestamps[2]}.png"
        },
        
        # Versioning with metadata
        "metadata_versioning": {
            "config": {
                "type": "figregistry_kedro.datasets.FigureDataSet",
                "filepath": str(versioned_output_path / "metadata.png"),
                "purpose": "exploratory",
                "versioned": True,
                "metadata": {
                    "experiment_id": "exp_001",
                    "model_version": "v1.2.3",
                    "dataset_hash": "abc123def456"
                }
            },
            "test_metadata": {
                "experiment_id": "exp_001", 
                "model_version": "v1.2.3",
                "dataset_hash": "abc123def456"
            }
        },
        
        # Version compatibility scenarios
        "compatibility_scenarios": {
            "kedro_catalog_versioning": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": str(versioned_output_path / "catalog.png"),
                    "purpose": "presentation",
                    "versioned": True
                },
                "kedro_versioning_enabled": True,
                "expected_integration": "seamless"
            },
            "manual_version_override": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet", 
                    "filepath": str(versioned_output_path / "manual.png"),
                    "purpose": "publication",
                    "versioned": False,  # Versioning disabled
                    "save_version": version_timestamps[0]  # Should be ignored
                },
                "expected_behavior": "no_versioning"
            }
        }
    }


@pytest.fixture(scope="function")
def error_scenario_fixtures():
    """
    Provides error scenarios for robust error handling testing.
    
    Creates comprehensive error scenarios including malformed catalog
    configurations, invalid parameters, file system errors, and
    edge cases to validate FigureDataSet error handling robustness.
    
    Returns:
        dict: Error scenarios organized by error category
    """
    return {
        # Parameter validation errors
        "parameter_errors": {
            "missing_filepath": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    # Missing required filepath parameter
                    "purpose": "exploratory"
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "filepath parameter is required"
            },
            "invalid_purpose": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "data/08_reporting/invalid.png",
                    "purpose": "invalid_purpose"
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "purpose must be one of"
            },
            "invalid_condition_param": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "data/08_reporting/invalid.png", 
                    "purpose": "exploratory",
                    "condition_param": "123invalid"  # Invalid identifier
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "valid Python identifier"
            },
            "invalid_style_params": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "data/08_reporting/invalid.png",
                    "purpose": "exploratory", 
                    "style_params": "not_a_dict"  # Should be dictionary
                },
                "expected_error": "FigureDatasetError", 
                "error_message_contains": "style_params must be a dictionary"
            }
        },
        
        # Figure object validation errors
        "figure_errors": {
            "non_figure_object": {
                "invalid_figure": "not_a_figure",
                "expected_error": "FigureDatasetError",
                "error_message_contains": "Expected matplotlib Figure object"
            },
            "none_figure": {
                "invalid_figure": None,
                "expected_error": "FigureDatasetError",
                "error_message_contains": "Expected matplotlib Figure object"
            },
            "corrupt_figure": {
                "invalid_figure": Mock(),  # Mock object that's not a real figure
                "expected_error": "FigureDatasetError",
                "error_message_contains": "Expected matplotlib Figure object"
            }
        },
        
        # File system errors
        "filesystem_errors": {
            "invalid_output_path": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "/invalid/path/that/does/not/exist/figure.png",
                    "purpose": "exploratory"
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "Failed to create output directory"
            },
            "permission_denied": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet", 
                    "filepath": "/root/protected/figure.png",  # Likely no permission
                    "purpose": "exploratory"
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "Failed to save figure"
            }
        },
        
        # Configuration integration errors
        "config_errors": {
            "malformed_style_config": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "data/08_reporting/malformed.png",
                    "purpose": "exploratory",
                    "style_params": {
                        "invalid.rcparam": "value",  # Invalid rcParam
                        "font.size": "not_a_number"  # Invalid value type
                    }
                },
                "expected_behavior": "graceful_degradation",
                "expected_warning": "Failed to apply styling"
            },
            "missing_condition_config": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "data/08_reporting/missing.png", 
                    "purpose": "exploratory",
                    "condition_param": "nonexistent_param"
                },
                "mock_context_params": {},  # Empty context
                "expected_behavior": "fallback_to_purpose",
                "expected_warning": "Failed to resolve condition parameter"
            }
        },
        
        # Edge cases and boundary conditions
        "edge_cases": {
            "empty_filepath": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "",  # Empty string
                    "purpose": "exploratory"
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "filepath parameter is required"
            },
            "whitespace_only_condition": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "data/08_reporting/whitespace.png",
                    "purpose": "exploratory", 
                    "condition_param": "   "  # Whitespace only
                },
                "expected_error": "FigureDatasetError",
                "error_message_contains": "non-empty string when provided"
            },
            "extremely_long_filepath": {
                "config": {
                    "type": "figregistry_kedro.datasets.FigureDataSet",
                    "filepath": "a" * 1000 + ".png",  # Very long path
                    "purpose": "exploratory"
                },
                "expected_behavior": "system_dependent",
                "notes": "May succeed or fail depending on filesystem limits"
            }
        }
    }


@pytest.fixture(scope="function")
def abstract_dataset_compliance_fixtures():
    """
    Creates fixtures for AbstractDataSet interface compliance validation.
    
    Provides comprehensive test scenarios to validate that FigureDataSet
    correctly implements Kedro's AbstractDataSet interface including
    required methods, proper inheritance, and interface contracts.
    
    Returns:
        dict: Interface compliance test scenarios and validation criteria
    """
    return {
        # Required method implementations
        "required_methods": {
            "_save": {
                "method_name": "_save",
                "required": True,
                "signature": "data: Figure",
                "return_type": "None",
                "description": "Save matplotlib figure with FigRegistry styling"
            },
            "_load": {
                "method_name": "_load", 
                "required": True,
                "signature": "() -> Figure",
                "return_type": "Figure",
                "description": "Load matplotlib figure from file"
            },
            "_describe": {
                "method_name": "_describe",
                "required": True, 
                "signature": "() -> Dict[str, Any]",
                "return_type": "Dict[str, Any]",
                "description": "Return dataset configuration description"
            },
            "_exists": {
                "method_name": "_exists",
                "required": True,
                "signature": "() -> bool", 
                "return_type": "bool",
                "description": "Check if dataset file exists"
            }
        },
        
        # Class attributes validation
        "class_attributes": {
            "_EPHEMERAL": {
                "attribute_name": "_EPHEMERAL",
                "expected_value": False,
                "description": "Figures are persistent, not ephemeral"
            },
            "_SINGLE_PROCESS": {
                "attribute_name": "_SINGLE_PROCESS", 
                "expected_value": False,
                "description": "Support parallel execution"
            }
        },
        
        # Interface contract validation
        "interface_contracts": {
            "initialization": {
                "test_scenario": "valid_initialization",
                "required_parameters": ["filepath"],
                "optional_parameters": [
                    "purpose", "condition_param", "style_params", 
                    "format_kwargs", "versioned", "metadata"
                ],
                "validation_criteria": [
                    "Initializes without errors with required parameters",
                    "Accepts all optional parameters",
                    "Validates parameter types and values"
                ]
            },
            "save_operation": {
                "test_scenario": "save_interface_compliance",
                "input_type": "matplotlib.figure.Figure",
                "expected_behavior": [
                    "Accepts matplotlib Figure objects",
                    "Applies FigRegistry styling", 
                    "Saves to configured filepath",
                    "Handles versioning if enabled",
                    "Raises appropriate errors for invalid inputs"
                ]
            },
            "load_operation": {
                "test_scenario": "load_interface_compliance", 
                "return_type": "matplotlib.figure.Figure",
                "expected_behavior": [
                    "Returns matplotlib Figure object",
                    "Handles missing files gracefully",
                    "Supports versioned file loading",
                    "Raises appropriate errors for corrupted files"
                ]
            },
            "describe_operation": {
                "test_scenario": "describe_interface_compliance",
                "return_type": "dict",
                "required_keys": [
                    "filepath", "purpose", "protocol", "versioned"
                ],
                "optional_keys": [
                    "condition_param", "style_params", "format_kwargs",
                    "metadata", "load_version", "save_version"
                ]
            }
        },
        
        # Inheritance validation  
        "inheritance_validation": {
            "abstract_dataset_inheritance": {
                "base_class": "kedro.io.AbstractDataset",
                "generic_types": ["matplotlib.figure.Figure", "matplotlib.figure.Figure"],
                "validation_criteria": [
                    "Inherits from AbstractDataset", 
                    "Implements generic type parameters correctly",
                    "Overrides required abstract methods"
                ]
            }
        },
        
        # Error handling compliance
        "error_handling_compliance": {
            "dataset_error_inheritance": {
                "custom_exception": "FigureDatasetError",
                "base_exception": "kedro.io.core.DatasetError", 
                "validation_criteria": [
                    "Custom exception inherits from DatasetError",
                    "Provides descriptive error messages",
                    "Includes relevant error context"
                ]
            },
            "error_scenarios": [
                {
                    "scenario": "invalid_input_type",
                    "expected_exception": "FigureDatasetError",
                    "trigger": "Pass non-Figure object to _save()"
                },
                {
                    "scenario": "file_not_found",
                    "expected_exception": "FileNotFoundError", 
                    "trigger": "Call _load() when file doesn't exist"
                },
                {
                    "scenario": "permission_error",
                    "expected_exception": "FigureDatasetError",
                    "trigger": "Save to directory without write permissions"
                }
            ]
        }
    }


@pytest.fixture(scope="function")
def performance_testing_fixtures(sample_matplotlib_figure):
    """
    Creates performance testing fixtures for validating <5% overhead.
    
    Provides comprehensive performance benchmarking utilities and
    test scenarios to validate that FigureDataSet operations maintain
    performance overhead under 5% compared to manual matplotlib operations.
    
    Args:
        sample_matplotlib_figure: Sample figure for performance testing
        
    Returns:
        dict: Performance testing utilities and benchmark scenarios
    """
    class PerformanceBenchmark:
        """Performance benchmarking utility for dataset operations."""
        
        def __init__(self):
            self.results = {
                "manual_save_times": [],
                "dataset_save_times": [],
                "style_application_times": [],
                "configuration_load_times": []
            }
            
        def benchmark_manual_save(self, figure, filepath, **save_kwargs):
            """Benchmark manual matplotlib savefig operation."""
            start_time = time.perf_counter()
            figure.savefig(filepath, **save_kwargs)
            end_time = time.perf_counter()
            
            save_time = (end_time - start_time) * 1000  # Convert to milliseconds
            self.results["manual_save_times"].append(save_time)
            return save_time
            
        def benchmark_dataset_save(self, dataset, figure):
            """Benchmark FigureDataSet save operation."""
            start_time = time.perf_counter()
            dataset._save(figure)
            end_time = time.perf_counter()
            
            save_time = (end_time - start_time) * 1000  # Convert to milliseconds
            self.results["dataset_save_times"].append(save_time)
            return save_time
            
        def benchmark_style_application(self, figure, style_config):
            """Benchmark style application to figure."""
            start_time = time.perf_counter()
            
            # Simulate style application
            for ax in figure.get_axes():
                for line in ax.get_lines():
                    if "color" in style_config:
                        line.set_color(style_config["color"])
                    if "linewidth" in style_config:
                        line.set_linewidth(style_config["linewidth"])
                        
            end_time = time.perf_counter()
            
            style_time = (end_time - start_time) * 1000
            self.results["style_application_times"].append(style_time)
            return style_time
            
        def calculate_overhead_percentage(self):
            """Calculate overhead percentage compared to manual operations."""
            if not self.results["manual_save_times"] or not self.results["dataset_save_times"]:
                return None
                
            avg_manual = np.mean(self.results["manual_save_times"])
            avg_dataset = np.mean(self.results["dataset_save_times"])
            
            overhead_percentage = ((avg_dataset - avg_manual) / avg_manual) * 100
            return overhead_percentage
            
        def get_performance_summary(self):
            """Get comprehensive performance summary."""
            return {
                "manual_operations": {
                    "count": len(self.results["manual_save_times"]),
                    "mean_ms": np.mean(self.results["manual_save_times"]) if self.results["manual_save_times"] else 0,
                    "median_ms": np.median(self.results["manual_save_times"]) if self.results["manual_save_times"] else 0,
                    "std_ms": np.std(self.results["manual_save_times"]) if self.results["manual_save_times"] else 0
                },
                "dataset_operations": {
                    "count": len(self.results["dataset_save_times"]),
                    "mean_ms": np.mean(self.results["dataset_save_times"]) if self.results["dataset_save_times"] else 0,
                    "median_ms": np.median(self.results["dataset_save_times"]) if self.results["dataset_save_times"] else 0,
                    "std_ms": np.std(self.results["dataset_save_times"]) if self.results["dataset_save_times"] else 0
                },
                "style_operations": {
                    "count": len(self.results["style_application_times"]),
                    "mean_ms": np.mean(self.results["style_application_times"]) if self.results["style_application_times"] else 0,
                    "median_ms": np.median(self.results["style_application_times"]) if self.results["style_application_times"] else 0
                },
                "overhead_analysis": {
                    "overhead_percentage": self.calculate_overhead_percentage(),
                    "within_target": self.calculate_overhead_percentage() < 5.0 if self.calculate_overhead_percentage() else None
                }
            }
    
    # Performance test scenarios
    performance_scenarios = {
        # Basic performance validation
        "basic_performance": {
            "description": "Basic save operation performance comparison",
            "test_iterations": 10,
            "figure_complexity": "simple",
            "expected_overhead_max": 5.0,  # 5% maximum overhead
            "performance_targets": {
                "max_save_time_ms": 200,
                "max_style_time_ms": 10,
                "max_config_time_ms": 50
            }
        },
        
        # Stress testing scenarios
        "stress_testing": {
            "description": "High-load performance validation",
            "test_iterations": 100,
            "figure_complexity": "complex",
            "expected_overhead_max": 5.0,
            "performance_targets": {
                "max_save_time_ms": 500,
                "max_style_time_ms": 25,
                "max_config_time_ms": 100
            }
        },
        
        # Configuration impact testing
        "config_impact": {
            "description": "Configuration loading performance impact",
            "test_scenarios": [
                {"config_size": "small", "max_time_ms": 25},
                {"config_size": "medium", "max_time_ms": 50}, 
                {"config_size": "large", "max_time_ms": 100}
            ]
        },
        
        # Caching effectiveness testing
        "caching_effectiveness": {
            "description": "Style caching performance validation",
            "cache_enabled_iterations": 50,
            "cache_disabled_iterations": 50,
            "expected_cache_speedup": 2.0  # Expect 2x speedup with caching
        }
    }
    
    # Test figures with different complexity levels
    test_figures = {
        "simple": sample_matplotlib_figure,
        "medium": sample_matplotlib_figure,  # Could be replaced with more complex
        "complex": sample_matplotlib_figure   # Could be replaced with most complex
    }
    
    # Style configurations for testing
    test_style_configs = {
        "minimal_style": {
            "color": "#FF0000"
        },
        "standard_style": {
            "color": "#00FF00",
            "linewidth": 2.0,
            "marker": "o"
        },
        "comprehensive_style": {
            "color": "#0000FF",
            "linewidth": 2.5,
            "marker": "s",
            "alpha": 0.7,
            "linestyle": "--"
        }
    }
    
    return {
        "benchmark_utility": PerformanceBenchmark(),
        "test_scenarios": performance_scenarios,
        "test_figures": test_figures,
        "test_styles": test_style_configs,
        "performance_thresholds": {
            "max_overhead_percentage": 5.0,
            "max_save_time_ms": 200,
            "max_style_time_ms": 10,
            "max_config_time_ms": 50
        }
    }


@pytest.fixture(scope="function")
def dataset_factory():
    """
    Provides factory function for creating FigureDataSet instances with various configurations.
    
    Utility factory for creating FigureDataSet instances with different
    parameter combinations during testing, enabling systematic validation
    of dataset behavior across parameter spaces.
    
    Returns:
        callable: Factory function for creating configured FigureDataSet instances
    """
    def create_dataset(
        filepath: str,
        purpose: str = "exploratory",
        condition_param: Optional[str] = None,
        style_params: Optional[Dict[str, Any]] = None,
        format_kwargs: Optional[Dict[str, Any]] = None,
        versioned: bool = False,
        **kwargs
    ):
        """
        Create FigureDataSet instance with specified configuration.
        
        Args:
            filepath: Output file path for the dataset
            purpose: Figure purpose (exploratory, presentation, publication)
            condition_param: Parameter name for condition resolution
            style_params: Style parameter overrides
            format_kwargs: Format arguments for savefig
            versioned: Enable Kedro versioning
            **kwargs: Additional dataset parameters
            
        Returns:
            FigureDataSet: Configured dataset instance
        """
        if not HAS_FIGUREDATASET:
            pytest.skip("FigureDataSet not available for testing")
            
        config = {
            "filepath": filepath,
            "purpose": purpose,
            "condition_param": condition_param,
            "style_params": style_params or {},
            "format_kwargs": format_kwargs or {},
            "versioned": versioned,
            **kwargs
        }
        
        # Remove None values to use defaults
        config = {k: v for k, v in config.items() if v is not None}
        
        return FigureDataSet(**config)
    
    return create_dataset


@pytest.fixture(scope="function")
def mock_figregistry_integration():
    """
    Provides mock FigRegistry integration for isolated dataset testing.
    
    Creates comprehensive mocks for FigRegistry functionality including
    configuration management, style resolution, and save operations
    to enable testing of FigureDataSet behavior without FigRegistry dependencies.
    
    Returns:
        dict: Mock objects and utilities for FigRegistry integration testing
    """
    # Mock configuration object
    mock_config = {
        "condition_styles": {
            "exploratory": {
                "color": "#A8E6CF",
                "linewidth": 1.5,
                "marker": "o"
            },
            "presentation": {
                "color": "#FFB6C1", 
                "linewidth": 2.0,
                "marker": "s"
            },
            "publication": {
                "color": "#1A1A1A",
                "linewidth": 2.5,
                "marker": "^"
            }
        },
        "paths": {
            "exploratory": "data/08_reporting/exploratory",
            "presentation": "data/08_reporting/presentation", 
            "publication": "data/08_reporting/publication"
        }
    }
    
    # Mock FigRegistry functions
    class MockFigRegistry:
        def __init__(self, config):
            self.config = config
            
        def init_config(self, config_path=None):
            """Mock init_config function.""" 
            return self.config
            
        def get_style(self, condition):
            """Mock get_style function."""
            return self.config["condition_styles"].get(
                condition, 
                self.config["condition_styles"]["exploratory"]
            )
            
        def save_figure(self, filepath, fig=None, **kwargs):
            """Mock save_figure function."""
            if fig is not None:
                fig.savefig(filepath, **kwargs)
            return filepath
    
    # Mock configuration bridge
    class MockConfigBridge:
        def __init__(self, figregistry_mock):
            self.figregistry = figregistry_mock
            
        def get_merged_config(self):
            """Get merged configuration."""
            return self.figregistry.config
            
        def resolve_condition_param(self, condition_param, context_params):
            """Resolve condition parameter from context."""
            return context_params.get(condition_param, "exploratory")
    
    figregistry_mock = MockFigRegistry(mock_config)
    config_bridge_mock = MockConfigBridge(figregistry_mock)
    
    return {
        "figregistry_mock": figregistry_mock,
        "config_bridge_mock": config_bridge_mock,
        "mock_config": mock_config,
        "mock_functions": {
            "init_config": figregistry_mock.init_config,
            "get_style": figregistry_mock.get_style,
            "save_figure": figregistry_mock.save_figure
        }
    }


# Utility fixtures for comprehensive testing support

@pytest.fixture(scope="function")
def temporary_figure_outputs(tmp_path):
    """
    Creates temporary directory structure for figure output testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Path: Temporary output directory with proper structure
    """
    output_dir = tmp_path / "figure_outputs"
    
    # Create purpose-specific subdirectories
    for purpose in ["exploratory", "presentation", "publication"]:
        (output_dir / purpose).mkdir(parents=True, exist_ok=True)
        
    # Create versioned output directory
    (output_dir / "versioned").mkdir(parents=True, exist_ok=True)
    
    return output_dir


@pytest.fixture(scope="function") 
def dataset_test_suite():
    """
    Provides comprehensive test suite utilities for dataset validation.
    
    Returns:
        dict: Test utilities and validation functions
    """
    def validate_dataset_interface(dataset_instance):
        """Validate that dataset implements required interface."""
        required_methods = ["_save", "_load", "_describe", "_exists"]
        missing_methods = []
        
        for method_name in required_methods:
            if not hasattr(dataset_instance, method_name):
                missing_methods.append(method_name)
                
        return len(missing_methods) == 0, missing_methods
    
    def validate_save_operation(dataset_instance, figure, expected_filepath):
        """Validate save operation behavior."""
        try:
            dataset_instance._save(figure)
            file_exists = Path(expected_filepath).exists()
            return True, file_exists, None
        except Exception as e:
            return False, False, str(e)
    
    def validate_describe_output(dataset_instance):
        """Validate _describe() method output."""
        try:
            description = dataset_instance._describe()
            
            # Check required keys
            required_keys = ["filepath", "purpose"]
            missing_keys = [key for key in required_keys if key not in description]
            
            return len(missing_keys) == 0, description, missing_keys
        except Exception as e:
            return False, {}, [str(e)]
    
    return {
        "validate_interface": validate_dataset_interface,
        "validate_save": validate_save_operation,
        "validate_describe": validate_describe_output
    }


# Export all fixtures for use in tests
__all__ = [
    "mock_kedro_context",
    "sample_matplotlib_figure", 
    "catalog_with_figuredataset",
    "dataset_parameter_scenarios",
    "versioned_dataset_fixtures",
    "error_scenario_fixtures",
    "abstract_dataset_compliance_fixtures",
    "performance_testing_fixtures",
    "dataset_factory",
    "mock_figregistry_integration",
    "temporary_figure_outputs",
    "dataset_test_suite"
]