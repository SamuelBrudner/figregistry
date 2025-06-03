"""Configuration testing fixtures for FigRegistry-Kedro integration.

This module provides comprehensive YAML configuration test data for validating
the FigRegistryConfigBridge functionality, including environment-specific overrides,
configuration merging scenarios, Pydantic validation testing, and error handling
for various configuration edge cases.

The fixtures support testing of:
- Configuration bridge between Kedro ConfigLoader and FigRegistry YAML system
- Environment-specific configuration precedence rules per F-007.2
- Pydantic validation for type safety across both systems per Section 5.2.5
- Configuration merging with proper precedence rules per F-007
- Performance validation for <10ms merge time requirements
- Error handling for malformed YAML and schema violations
"""

import pytest
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import yaml
from datetime import datetime
from unittest.mock import Mock


# =============================================================================
# Base FigRegistry Configuration Fixtures
# =============================================================================

@pytest.fixture
def base_figregistry_config() -> Dict[str, Any]:
    """Standard FigRegistry YAML configuration for baseline testing per Section 5.2.5.
    
    Provides a complete, valid FigRegistry configuration with all major sections
    populated for use as the foundation for testing configuration merging,
    validation, and bridge functionality.
    
    Returns:
        Complete FigRegistry configuration dictionary with styles, palettes,
        outputs, and defaults sections properly structured.
    """
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {
                "color": "#1f77b4",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 2.0,
                "markersize": 6,
                "alpha": 0.8,
                "label": "Control"
            },
            "treatment_a": {
                "color": "#ff7f0e", 
                "marker": "s",
                "linestyle": "--",
                "linewidth": 2.5,
                "markersize": 7,
                "alpha": 0.9,
                "label": "Treatment A"
            },
            "treatment_b": {
                "color": "#2ca02c",
                "marker": "^",
                "linestyle": "-.",
                "linewidth": 2.0,
                "markersize": 8,
                "alpha": 0.85,
                "label": "Treatment B"
            },
            "exploratory_*": {
                "color": "#d62728",
                "marker": "x",
                "linestyle": ":",
                "linewidth": 1.5,
                "markersize": 5,
                "alpha": 0.7,
                "label": "Exploratory"
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "colorblind_safe": ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161"],
            "publication": ["#000000", "#404040", "#808080", "#b0b0b0", "#d0d0d0"],
            "high_contrast": ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"]
        },
        "outputs": {
            "base_path": "figures",
            "path_aliases": {
                "expl": "exploratory",
                "pres": "presentation", 
                "pub": "publication"
            },
            "timestamp_format": "{name}_{ts}",
            "default_format": "png",
            "dpi": 300,
            "bbox_inches": "tight",
            "pad_inches": 0.1
        },
        "defaults": {
            "figure_size": [10, 6],
            "font_family": "sans-serif",
            "font_size": 12,
            "line_width": 1.5,
            "marker_size": 6,
            "color_cycle": "default",
            "grid": True,
            "spine_visibility": {
                "top": False,
                "right": False,
                "bottom": True,
                "left": True
            }
        },
        "metadata": {
            "project_name": "figregistry-kedro-test",
            "description": "Test configuration for FigRegistry-Kedro integration",
            "version": "1.0.0",
            "created": "2024-01-01T00:00:00Z"
        }
    }


@pytest.fixture
def minimal_figregistry_config() -> Dict[str, Any]:
    """Minimal valid FigRegistry configuration for basic testing.
    
    Provides the bare minimum required configuration fields to test
    configuration loading and validation with default value handling.
    
    Returns:
        Minimal FigRegistry configuration with only required fields.
    """
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "default": {
                "color": "#1f77b4",
                "marker": "o"
            }
        }
    }


@pytest.fixture 
def comprehensive_figregistry_config() -> Dict[str, Any]:
    """Comprehensive FigRegistry configuration with advanced features.
    
    Includes complex styling patterns, multiple palettes, advanced output
    configurations, and extensive metadata for testing full feature support.
    
    Returns:
        Comprehensive configuration dictionary with all optional fields.
    """
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "baseline": {
                "color": "#1f77b4",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 2.0,
                "markersize": 6,
                "alpha": 0.8,
                "label": "Baseline",
                "zorder": 1
            },
            "intervention_high": {
                "color": "#ff7f0e",
                "marker": "s", 
                "linestyle": "--",
                "linewidth": 3.0,
                "markersize": 8,
                "alpha": 0.9,
                "label": "High Dose",
                "zorder": 2
            },
            "intervention_low": {
                "color": "#ffbb78",
                "marker": "s",
                "linestyle": "--", 
                "linewidth": 2.0,
                "markersize": 6,
                "alpha": 0.7,
                "label": "Low Dose",
                "zorder": 2
            },
            "exploratory_*": {
                "color": "#d62728",
                "marker": "x",
                "linestyle": ":",
                "linewidth": 1.0,
                "markersize": 4,
                "alpha": 0.5,
                "label": "Exploratory",
                "zorder": 0
            },
            "validation_*": {
                "color": "#9467bd",
                "marker": "v",
                "linestyle": "-.",
                "linewidth": 1.5,
                "markersize": 5,
                "alpha": 0.6,
                "label": "Validation",
                "zorder": 1
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "qualitative": ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"],
            "sequential_blue": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd"],
            "diverging": ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a"],
            "colorblind_safe": ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161"],
            "publication_bw": ["#000000", "#404040", "#666666", "#999999", "#cccccc", "#ffffff"],
            "high_contrast": ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        },
        "outputs": {
            "base_path": "outputs/figures",
            "path_aliases": {
                "expl": "01_exploratory",
                "eda": "01_exploratory", 
                "pres": "02_presentation",
                "present": "02_presentation",
                "pub": "03_publication",
                "publish": "03_publication",
                "temp": "temp",
                "debug": "debug"
            },
            "timestamp_format": "{name}_{ts:%Y%m%d_%H%M%S}",
            "slug_format": "{purpose}_{name}_{condition}",
            "default_format": "png",
            "formats": ["png", "pdf", "svg"],
            "dpi": 300,
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "facecolor": "white",
            "edgecolor": "none",
            "transparent": False,
            "metadata": True
        },
        "defaults": {
            "figure_size": [12, 8],
            "font_family": "DejaVu Sans", 
            "font_size": 14,
            "title_size": 16,
            "label_size": 12,
            "tick_size": 10,
            "legend_size": 11,
            "line_width": 2.0,
            "marker_size": 8,
            "color_cycle": "default",
            "grid": True,
            "grid_alpha": 0.3,
            "spine_visibility": {
                "top": False,
                "right": False,
                "bottom": True,
                "left": True
            },
            "spine_linewidth": 1.0,
            "tick_direction": "out",
            "legend_frameon": True,
            "legend_fancybox": True,
            "legend_shadow": False
        },
        "metadata": {
            "project_name": "advanced-kedro-figregistry-project",
            "description": "Comprehensive test configuration for advanced FigRegistry-Kedro integration",
            "version": "2.1.0",
            "author": "Test Suite",
            "created": "2024-01-01T00:00:00Z",
            "updated": "2024-01-15T12:30:00Z",
            "tags": ["kedro", "figregistry", "testing", "integration"],
            "environment": "test"
        }
    }


# =============================================================================
# Kedro Configuration Override Fixtures  
# =============================================================================

@pytest.fixture
def local_override_config() -> Dict[str, Any]:
    """Environment-specific configuration testing per F-007.2.
    
    Provides Kedro local environment overrides that should take precedence
    over base FigRegistry configuration during merging operations.
    
    Returns:
        Kedro local environment configuration overrides.
    """
    return {
        "styles": {
            "control": {
                "color": "#0066cc",  # Override base color
                "linewidth": 3.0,   # Override base linewidth
                "label": "Local Control"  # Override base label
            },
            "local_condition": {
                "color": "#ff6600",
                "marker": "D",
                "linestyle": "-",
                "linewidth": 2.0,
                "alpha": 0.8,
                "label": "Local Only"
            }
        },
        "outputs": {
            "base_path": "local_figures",  # Override base path
            "dpi": 150,  # Override DPI for faster local testing
            "default_format": "svg"  # Override format for dev
        },
        "defaults": {
            "figure_size": [8, 5],  # Smaller for local development
            "font_size": 10,  # Smaller font for local
            "grid": False  # Disable grid locally
        },
        "kedro": {
            "enable_versioning": False,
            "parallel_execution": False,
            "debug_mode": True,
            "cache_styling": True
        }
    }


@pytest.fixture
def staging_override_config() -> Dict[str, Any]:
    """Staging environment configuration for multi-environment testing.
    
    Returns:
        Staging environment specific overrides.
    """
    return {
        "styles": {
            "control": {
                "alpha": 0.9  # Higher alpha for staging visibility
            },
            "staging_validation": {
                "color": "#9900cc",
                "marker": "v", 
                "linestyle": "-.",
                "label": "Staging Validation"
            }
        },
        "outputs": {
            "base_path": "staging/figures",
            "dpi": 200,
            "metadata": True,
            "timestamp_format": "staging_{name}_{ts:%Y%m%d_%H%M}"
        },
        "defaults": {
            "figure_size": [10, 6],
            "font_size": 12,
            "grid": True
        },
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "debug_mode": False,
            "performance_monitoring": True
        }
    }


@pytest.fixture
def production_override_config() -> Dict[str, Any]:
    """Production environment configuration for deployment testing.
    
    Returns:
        Production environment specific overrides with optimized settings.
    """
    return {
        "styles": {
            "control": {
                "linewidth": 2.5,  # Thicker lines for production
                "alpha": 1.0  # Full opacity for production
            }
        },
        "outputs": {
            "base_path": "/opt/kedro/production/figures",
            "dpi": 300,
            "formats": ["png", "pdf"],
            "bbox_inches": "tight",
            "pad_inches": 0.2,
            "metadata": True,
            "timestamp_format": "prod_{name}_{ts:%Y%m%d_%H%M%S}"
        },
        "defaults": {
            "figure_size": [12, 8],
            "font_size": 14,
            "line_width": 2.0,
            "grid": True,
            "grid_alpha": 0.2
        },
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "debug_mode": False,
            "performance_monitoring": True,
            "cache_styling": True,
            "enable_concurrent_access": True,
            "validation_enabled": True
        }
    }


# =============================================================================
# Configuration Merging Test Scenarios
# =============================================================================

@pytest.fixture
def merged_config_scenarios() -> List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """Various configuration merge combinations for comprehensive testing.
    
    Provides test scenarios for validating configuration merging logic,
    precedence rules, and edge cases in the FigRegistryConfigBridge.
    
    Returns:
        List of tuples containing (scenario_name, base_config, override_config, expected_merged).
    """
    scenarios = []
    
    # Scenario 1: Simple override
    base_simple = {
        "styles": {"control": {"color": "#1f77b4", "marker": "o"}},
        "outputs": {"base_path": "figures", "dpi": 300}
    }
    override_simple = {
        "styles": {"control": {"color": "#ff0000"}},
        "outputs": {"dpi": 150}
    }
    expected_simple = {
        "styles": {"control": {"color": "#ff0000", "marker": "o"}},
        "outputs": {"base_path": "figures", "dpi": 150}
    }
    scenarios.append(("simple_override", base_simple, override_simple, expected_simple))
    
    # Scenario 2: Deep merge with new sections
    base_deep = {
        "styles": {
            "control": {"color": "#1f77b4", "marker": "o"},
            "treatment": {"color": "#ff7f0e", "marker": "s"}
        },
        "defaults": {"figure_size": [10, 6], "font_size": 12}
    }
    override_deep = {
        "styles": {
            "control": {"linewidth": 2.0},
            "new_condition": {"color": "#00ff00", "marker": "^"}
        },
        "defaults": {"font_size": 14},
        "kedro": {"debug_mode": True}
    }
    expected_deep = {
        "styles": {
            "control": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0},
            "treatment": {"color": "#ff7f0e", "marker": "s"},
            "new_condition": {"color": "#00ff00", "marker": "^"}
        },
        "defaults": {"figure_size": [10, 6], "font_size": 14},
        "kedro": {"debug_mode": True}
    }
    scenarios.append(("deep_merge", base_deep, override_deep, expected_deep))
    
    # Scenario 3: List replacement (not merging)
    base_list = {
        "palettes": {"default": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
        "outputs": {"formats": ["png", "pdf"]}
    }
    override_list = {
        "palettes": {"default": ["#000000", "#ffffff"]},
        "outputs": {"formats": ["svg"]}
    }
    expected_list = {
        "palettes": {"default": ["#000000", "#ffffff"]},
        "outputs": {"formats": ["svg"]}
    }
    scenarios.append(("list_replacement", base_list, override_list, expected_list))
    
    # Scenario 4: Environment-specific precedence
    base_env = {
        "styles": {"control": {"color": "#1f77b4", "alpha": 0.8}},
        "outputs": {"base_path": "figures"},
        "environment": "base"
    }
    override_env = {
        "styles": {"control": {"alpha": 1.0}},
        "outputs": {"base_path": "local_figures"},
        "environment": "local",
        "kedro": {"enable_versioning": False}
    }
    expected_env = {
        "styles": {"control": {"color": "#1f77b4", "alpha": 1.0}},
        "outputs": {"base_path": "local_figures"},
        "environment": "local",
        "kedro": {"enable_versioning": False}
    }
    scenarios.append(("environment_precedence", base_env, override_env, expected_env))
    
    return scenarios


@pytest.fixture
def config_merge_test_cases() -> List[Dict[str, Any]]:
    """Precedence rule validation scenarios per Section 5.2.5.
    
    Provides test cases for validating that configuration merging follows
    the correct precedence rules and handles edge cases appropriately.
    
    Returns:
        List of test case dictionaries with configuration scenarios.
    """
    return [
        {
            "name": "kedro_overrides_figregistry",
            "description": "Kedro configurations should override FigRegistry base settings",
            "figregistry_config": {
                "styles": {"control": {"color": "#1f77b4", "linewidth": 1.0}},
                "outputs": {"dpi": 300}
            },
            "kedro_config": {
                "styles": {"control": {"color": "#ff0000"}},
                "outputs": {"dpi": 150}
            },
            "expected_precedence": {
                "styles.control.color": "#ff0000",  # Kedro wins
                "styles.control.linewidth": 1.0,    # FigRegistry preserved
                "outputs.dpi": 150                  # Kedro wins
            }
        },
        {
            "name": "environment_specific_override",
            "description": "Environment-specific configs override base configs",
            "base_config": {
                "styles": {"control": {"color": "#1f77b4"}},
                "outputs": {"base_path": "figures"},
                "environment": "base"
            },
            "env_config": {
                "styles": {"control": {"color": "#00ff00"}},
                "outputs": {"base_path": "local_figures"},
                "environment": "local"
            },
            "expected_precedence": {
                "styles.control.color": "#00ff00",
                "outputs.base_path": "local_figures",
                "environment": "local"
            }
        },
        {
            "name": "nested_dict_deep_merge",
            "description": "Nested dictionaries should merge deeply, not replace",
            "base_config": {
                "defaults": {
                    "spine_visibility": {"top": False, "right": False, "bottom": True, "left": True},
                    "figure_size": [10, 6]
                }
            },
            "override_config": {
                "defaults": {
                    "spine_visibility": {"top": True},  # Only override top
                    "font_size": 12  # Add new field
                }
            },
            "expected_result": {
                "defaults": {
                    "spine_visibility": {"top": True, "right": False, "bottom": True, "left": True},
                    "figure_size": [10, 6],
                    "font_size": 12
                }
            }
        },
        {
            "name": "list_complete_replacement",
            "description": "Lists should be replaced completely, not merged",
            "base_config": {
                "palettes": {"default": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
                "outputs": {"formats": ["png", "pdf", "svg"]}
            },
            "override_config": {
                "palettes": {"default": ["#000000", "#ffffff"]},
                "outputs": {"formats": ["png"]}
            },
            "expected_result": {
                "palettes": {"default": ["#000000", "#ffffff"]},
                "outputs": {"formats": ["png"]}
            }
        }
    ]


# =============================================================================
# Invalid Configuration Fixtures
# =============================================================================

@pytest.fixture
def invalid_config_fixtures() -> Dict[str, Dict[str, Any]]:
    """Error handling validation including malformed YAML and schema violations.
    
    Provides various invalid configuration scenarios to test error handling
    and validation in the configuration bridge and Pydantic models.
    
    Returns:
        Dictionary of invalid configuration scenarios mapped by error type.
    """
    return {
        "missing_version": {
            "config": {
                "styles": {"control": {"color": "#1f77b4"}}
                # Missing figregistry_version
            },
            "expected_error": "figregistry_version",
            "error_type": "ValidationError"
        },
        "invalid_color_format": {
            "config": {
                "figregistry_version": "0.3.0",
                "styles": {
                    "control": {
                        "color": "not_a_color",  # Invalid color format
                        "marker": "o"
                    }
                }
            },
            "expected_error": "color",
            "error_type": "ValidationError"
        },
        "invalid_marker": {
            "config": {
                "figregistry_version": "0.3.0", 
                "styles": {
                    "control": {
                        "color": "#1f77b4",
                        "marker": "invalid_marker"  # Invalid matplotlib marker
                    }
                }
            },
            "expected_error": "marker",
            "error_type": "ValidationError"
        },
        "invalid_numeric_type": {
            "config": {
                "figregistry_version": "0.3.0",
                "styles": {
                    "control": {
                        "color": "#1f77b4",
                        "linewidth": "not_a_number"  # Should be numeric
                    }
                }
            },
            "expected_error": "linewidth",
            "error_type": "ValidationError"
        },
        "invalid_path_type": {
            "config": {
                "figregistry_version": "0.3.0",
                "outputs": {
                    "base_path": 12345  # Should be string or Path
                }
            },
            "expected_error": "base_path",
            "error_type": "ValidationError"
        },
        "invalid_boolean_type": {
            "config": {
                "figregistry_version": "0.3.0",
                "defaults": {
                    "grid": "maybe"  # Should be boolean
                }
            },
            "expected_error": "grid",
            "error_type": "ValidationError"
        },
        "negative_numeric_values": {
            "config": {
                "figregistry_version": "0.3.0",
                "styles": {
                    "control": {
                        "color": "#1f77b4",
                        "linewidth": -1.5,  # Negative linewidth
                        "markersize": -5,   # Negative markersize
                        "alpha": -0.5       # Alpha out of range
                    }
                }
            },
            "expected_error": "negative values",
            "error_type": "ValidationError"
        },
        "alpha_out_of_range": {
            "config": {
                "figregistry_version": "0.3.0",
                "styles": {
                    "control": {
                        "color": "#1f77b4",
                        "alpha": 1.5  # Alpha > 1.0
                    }
                }
            },
            "expected_error": "alpha",
            "error_type": "ValidationError"
        },
        "empty_styles_dict": {
            "config": {
                "figregistry_version": "0.3.0",
                "styles": {}  # Empty styles not allowed
            },
            "expected_error": "styles",
            "error_type": "ValidationError"
        },
        "non_dict_styles": {
            "config": {
                "figregistry_version": "0.3.0",
                "styles": "not_a_dict"  # Should be dictionary
            },
            "expected_error": "styles",
            "error_type": "ValidationError"
        }
    }


@pytest.fixture
def malformed_yaml_fixtures() -> Dict[str, str]:
    """Malformed YAML test cases for parsing error validation.
    
    Returns:
        Dictionary of malformed YAML strings mapped by error type.
    """
    return {
        "invalid_syntax": """
figregistry_version: "0.3.0"
styles:
  control:
    color: "#1f77b4"
    marker: "o"
  # Missing closing quote and invalid indentation
  treatment:
    color: "#ff7f0e
      marker: "s"
""",
        "duplicate_keys": """
figregistry_version: "0.3.0"
styles:
  control:
    color: "#1f77b4"
    marker: "o"
    color: "#ff0000"  # Duplicate key
""",
        "invalid_indentation": """
figregistry_version: "0.3.0"
styles:
control:  # Wrong indentation
  color: "#1f77b4"
""",
        "unmatched_brackets": """
figregistry_version: "0.3.0"
styles:
  control:
    color: "#1f77b4"
    properties: [
      "linewidth": 2.0
      "marker": "o"  # Missing comma and bracket
""",
        "invalid_unicode": """
figregistry_version: "0.3.0"
styles:
  control:
    label: "\uDC80"  # Invalid Unicode surrogate
""",
        "tab_indentation": """
figregistry_version: "0.3.0"
styles:
\tcontrol:  # Tab instead of spaces
\t\tcolor: "#1f77b4"
"""
    }


# =============================================================================
# Pydantic Validation Fixtures  
# =============================================================================

@pytest.fixture
def pydantic_validation_fixtures() -> Dict[str, Dict[str, Any]]:
    """Type safety testing across configuration bridge operations.
    
    Provides test cases specifically for validating Pydantic model validation
    across the FigRegistryKedroConfig model to ensure type safety.
    
    Returns:
        Dictionary of validation test scenarios.
    """
    return {
        "valid_full_config": {
            "config": {
                "styles": {
                    "control": {
                        "color": "#1f77b4",
                        "marker": "o",
                        "linewidth": 2.0,
                        "alpha": 0.8
                    }
                },
                "palettes": {
                    "default": ["#1f77b4", "#ff7f0e", "#2ca02c"]
                },
                "outputs": {
                    "base_path": "figures",
                    "dpi": 300
                },
                "defaults": {
                    "figure_size": [10, 6],
                    "font_size": 12,
                    "grid": True
                },
                "kedro": {
                    "enable_versioning": True,
                    "debug_mode": False
                },
                "environment": "test",
                "enable_concurrent_access": True,
                "validation_enabled": True
            },
            "should_validate": True
        },
        "minimal_valid_config": {
            "config": {
                "styles": {
                    "default": {"color": "#1f77b4"}
                }
            },
            "should_validate": True
        },
        "invalid_styles_type": {
            "config": {
                "styles": "not_a_dict"
            },
            "should_validate": False,
            "expected_error": "styles must be a dictionary"
        },
        "invalid_outputs_type": {
            "config": {
                "styles": {"default": {"color": "#1f77b4"}},
                "outputs": "not_a_dict"
            },
            "should_validate": False,
            "expected_error": "outputs must be a dictionary"
        },
        "invalid_boolean_field": {
            "config": {
                "styles": {"default": {"color": "#1f77b4"}},
                "enable_concurrent_access": "not_a_boolean"
            },
            "should_validate": False,
            "expected_error": "enable_concurrent_access"
        },
        "invalid_string_field": {
            "config": {
                "styles": {"default": {"color": "#1f77b4"}},
                "environment": 123
            },
            "should_validate": False,
            "expected_error": "environment"
        },
        "extra_fields_allowed": {
            "config": {
                "styles": {"default": {"color": "#1f77b4"}},
                "custom_field": "should_be_allowed",
                "another_extra": {"nested": "value"}
            },
            "should_validate": True,
            "description": "Extra fields should be allowed per Pydantic config"
        }
    }


# =============================================================================
# Environment-Specific Configuration Fixtures
# =============================================================================

@pytest.fixture
def environment_specific_configs() -> Dict[str, Dict[str, Any]]:
    """Development, staging, and production scenarios per F-007.2.
    
    Provides complete environment-specific configuration sets for testing
    the full lifecycle of configuration management across deployment stages.
    
    Returns:
        Dictionary of environment configurations mapped by environment name.
    """
    return {
        "development": {
            "environment": "development",
            "styles": {
                "control": {
                    "color": "#1f77b4",
                    "marker": "o",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Control (Dev)"
                },
                "treatment": {
                    "color": "#ff7f0e", 
                    "marker": "s",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Treatment (Dev)"
                },
                "debug_*": {
                    "color": "#d62728",
                    "marker": "x",
                    "linestyle": ":",
                    "linewidth": 1.0,
                    "alpha": 0.5,
                    "label": "Debug"
                }
            },
            "outputs": {
                "base_path": "dev_figures",
                "dpi": 100,  # Lower DPI for faster iteration
                "default_format": "png",
                "timestamp_format": "dev_{name}_{ts:%H%M%S}"
            },
            "defaults": {
                "figure_size": [8, 5],  # Smaller for faster rendering
                "font_size": 10,
                "grid": False,  # Cleaner for development
                "line_width": 1.0
            },
            "kedro": {
                "enable_versioning": False,
                "parallel_execution": False,
                "debug_mode": True,
                "cache_styling": False,  # Disable caching for fresh styles
                "performance_monitoring": False
            }
        },
        "staging": {
            "environment": "staging",
            "styles": {
                "control": {
                    "color": "#1f77b4",
                    "marker": "o", 
                    "linewidth": 2.0,
                    "alpha": 0.8,
                    "label": "Control"
                },
                "treatment": {
                    "color": "#ff7f0e",
                    "marker": "s",
                    "linewidth": 2.0,
                    "alpha": 0.8,
                    "label": "Treatment"
                },
                "validation_*": {
                    "color": "#9467bd",
                    "marker": "v",
                    "linestyle": "-.",
                    "linewidth": 1.5,
                    "alpha": 0.6,
                    "label": "Validation"
                }
            },
            "outputs": {
                "base_path": "staging/figures",
                "dpi": 200,  # Medium quality for staging
                "default_format": "png",
                "formats": ["png", "svg"],
                "timestamp_format": "staging_{name}_{ts:%Y%m%d_%H%M}",
                "metadata": True
            },
            "defaults": {
                "figure_size": [10, 6],
                "font_size": 12,
                "grid": True,
                "grid_alpha": 0.3,
                "line_width": 1.5
            },
            "kedro": {
                "enable_versioning": True,
                "parallel_execution": True,
                "debug_mode": False,
                "cache_styling": True,
                "performance_monitoring": True,
                "validation_enabled": True
            }
        },
        "production": {
            "environment": "production",
            "styles": {
                "control": {
                    "color": "#1f77b4",
                    "marker": "o",
                    "linewidth": 2.5,  # Thicker for production quality
                    "alpha": 1.0,      # Full opacity for final output
                    "label": "Control"
                },
                "treatment": {
                    "color": "#ff7f0e",
                    "marker": "s", 
                    "linewidth": 2.5,
                    "alpha": 1.0,
                    "label": "Treatment"
                },
                "final_*": {
                    "color": "#2ca02c",
                    "marker": "^",
                    "linestyle": "-",
                    "linewidth": 2.0,
                    "alpha": 0.9,
                    "label": "Final"
                }
            },
            "outputs": {
                "base_path": "/opt/kedro/production/figures",
                "dpi": 300,  # High quality for production
                "default_format": "pdf",  # Vector format for production
                "formats": ["pdf", "png", "svg"],
                "bbox_inches": "tight",
                "pad_inches": 0.2,
                "timestamp_format": "prod_{name}_{ts:%Y%m%d_%H%M%S}",
                "metadata": True,
                "facecolor": "white",
                "transparent": False
            },
            "defaults": {
                "figure_size": [12, 8],  # Larger for publication quality
                "font_size": 14,
                "title_size": 16,
                "label_size": 12,
                "grid": True,
                "grid_alpha": 0.2,
                "line_width": 2.0,
                "spine_visibility": {
                    "top": False,
                    "right": False,
                    "bottom": True,
                    "left": True
                }
            },
            "kedro": {
                "enable_versioning": True,
                "parallel_execution": True,
                "debug_mode": False,
                "cache_styling": True,
                "performance_monitoring": True,
                "validation_enabled": True,
                "enable_concurrent_access": True
            }
        }
    }


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_config_scenarios() -> Dict[str, Dict[str, Any]]:
    """Configuration scenarios for testing <10ms merge time requirements.
    
    Provides configurations of various complexities to validate that
    configuration merging meets the performance target of <10ms.
    
    Returns:
        Dictionary of performance test scenarios.
    """
    return {
        "small_config": {
            "description": "Small configuration with minimal sections",
            "base_config": {
                "styles": {"control": {"color": "#1f77b4"}},
                "outputs": {"base_path": "figures"}
            },
            "override_config": {
                "styles": {"control": {"marker": "o"}},
                "outputs": {"dpi": 300}
            },
            "expected_merge_time_ms": 1.0
        },
        "medium_config": {
            "description": "Medium configuration with moderate complexity",
            "base_config": {
                "styles": {f"condition_{i}": {"color": f"#{'%06x' % (i * 1111 % 16777215)}", "marker": "o"} 
                          for i in range(20)},
                "palettes": {"default": [f"#{'%06x' % (i * 2222 % 16777215)}" for i in range(10)]},
                "outputs": {"base_path": "figures", "dpi": 300, "formats": ["png", "pdf"]},
                "defaults": {"figure_size": [10, 6], "font_size": 12}
            },
            "override_config": {
                "styles": {f"condition_{i}": {"linewidth": 2.0} for i in range(10)},
                "outputs": {"dpi": 150},
                "kedro": {"debug_mode": True}
            },
            "expected_merge_time_ms": 5.0
        },
        "large_config": {
            "description": "Large configuration with high complexity",
            "base_config": {
                "styles": {f"condition_{i}": {
                    "color": f"#{'%06x' % (i * 3333 % 16777215)}",
                    "marker": ["o", "s", "^", "v", "D"][i % 5],
                    "linewidth": 1.0 + (i % 3),
                    "alpha": 0.5 + (i % 5) * 0.1
                } for i in range(100)},
                "palettes": {f"palette_{i}": [f"#{'%06x' % (j * 4444 % 16777215)}" for j in range(i, i+5)] 
                           for i in range(20)},
                "outputs": {
                    "base_path": "figures",
                    "path_aliases": {f"alias_{i}": f"path_{i}" for i in range(50)},
                    "dpi": 300,
                    "formats": ["png", "pdf", "svg", "eps"]
                },
                "defaults": {
                    "figure_size": [12, 8],
                    "font_size": 14,
                    "spine_visibility": {"top": False, "right": False, "bottom": True, "left": True}
                }
            },
            "override_config": {
                "styles": {f"condition_{i}": {"alpha": 0.9} for i in range(0, 100, 2)},
                "outputs": {"dpi": 200, "base_path": "override_figures"},
                "kedro": {
                    "enable_versioning": True,
                    "parallel_execution": True,
                    "performance_monitoring": True
                }
            },
            "expected_merge_time_ms": 8.0
        }
    }


# =============================================================================
# Mock and Utility Fixtures
# =============================================================================

@pytest.fixture
def mock_kedro_config_loader():
    """Mock Kedro ConfigLoader for testing configuration bridge.
    
    Returns:
        Mock ConfigLoader instance with configurable behavior.
    """
    mock_loader = Mock()
    
    def mock_get(pattern):
        """Mock get method that returns configuration based on pattern."""
        if "figregistry" in pattern:
            return {
                "styles": {"mock_condition": {"color": "#123456"}},
                "outputs": {"base_path": "mock_figures"}
            }
        return {}
    
    mock_loader.get = Mock(side_effect=mock_get)
    return mock_loader


@pytest.fixture
def temp_config_files(tmp_path):
    """Temporary configuration files for filesystem testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Dictionary of temporary file paths and their contents.
    """
    config_files = {}
    
    # Base FigRegistry config
    figregistry_config = {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#1f77b4", "marker": "o"}},
        "outputs": {"base_path": "figures", "dpi": 300}
    }
    figregistry_path = tmp_path / "figregistry.yaml"
    with open(figregistry_path, 'w') as f:
        yaml.dump(figregistry_config, f)
    config_files["figregistry"] = figregistry_path
    
    # Kedro config directory structure
    kedro_conf = tmp_path / "conf"
    kedro_conf.mkdir()
    
    base_conf = kedro_conf / "base"
    base_conf.mkdir()
    
    local_conf = kedro_conf / "local" 
    local_conf.mkdir()
    
    # Base figregistry config in Kedro
    kedro_base_config = {
        "styles": {"kedro_condition": {"color": "#ff0000", "marker": "s"}},
        "outputs": {"base_path": "kedro_figures"}
    }
    kedro_base_path = base_conf / "figregistry.yml"
    with open(kedro_base_path, 'w') as f:
        yaml.dump(kedro_base_config, f)
    config_files["kedro_base"] = kedro_base_path
    
    # Local override config
    local_override_config = {
        "styles": {"kedro_condition": {"linewidth": 2.0}},
        "outputs": {"dpi": 150}
    }
    local_override_path = local_conf / "figregistry.yml"
    with open(local_override_path, 'w') as f:
        yaml.dump(local_override_config, f)
    config_files["kedro_local"] = local_override_path
    
    config_files["conf_dir"] = kedro_conf
    return config_files


@pytest.fixture
def sample_figure():
    """Sample matplotlib figure for testing dataset operations.
    
    Returns:
        Matplotlib figure instance for testing.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x), label="Test Data")
        ax.set_xlabel("X values")
        ax.set_ylabel("Y values") 
        ax.set_title("Test Figure")
        ax.legend()
        
        return fig
    except ImportError:
        # Return mock figure if matplotlib not available
        return Mock()


# =============================================================================
# Comprehensive Test Scenario Fixtures
# =============================================================================

@pytest.fixture
def comprehensive_test_scenarios():
    """Complete test scenarios combining multiple fixture types.
    
    Provides end-to-end test scenarios that combine configuration fixtures
    with expected behaviors for comprehensive integration testing.
    
    Returns:
        Dictionary of comprehensive test scenarios.
    """
    return {
        "full_integration_scenario": {
            "description": "Complete integration test with all components",
            "base_figregistry_config": {
                "figregistry_version": "0.3.0",
                "styles": {
                    "baseline": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0},
                    "treatment": {"color": "#ff7f0e", "marker": "s", "linewidth": 2.0}
                },
                "palettes": {"default": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
                "outputs": {"base_path": "figures", "dpi": 300, "default_format": "png"},
                "defaults": {"figure_size": [10, 6], "font_size": 12, "grid": True}
            },
            "kedro_overrides": {
                "styles": {
                    "baseline": {"alpha": 0.8},  # Partial override
                    "kedro_specific": {"color": "#2ca02c", "marker": "^"}  # New condition
                },
                "outputs": {"base_path": "kedro_outputs", "dpi": 200},  # Path and DPI override
                "kedro": {"enable_versioning": True, "debug_mode": False}  # Kedro-specific config
            },
            "expected_merged_config": {
                "styles": {
                    "baseline": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0, "alpha": 0.8},
                    "treatment": {"color": "#ff7f0e", "marker": "s", "linewidth": 2.0},
                    "kedro_specific": {"color": "#2ca02c", "marker": "^"}
                },
                "palettes": {"default": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
                "outputs": {"base_path": "kedro_outputs", "dpi": 200, "default_format": "png"},
                "defaults": {"figure_size": [10, 6], "font_size": 12, "grid": True},
                "kedro": {"enable_versioning": True, "debug_mode": False}
            },
            "validation_checks": [
                "styles.baseline.color == '#1f77b4'",  # Base preserved
                "styles.baseline.alpha == 0.8",       # Override applied
                "outputs.base_path == 'kedro_outputs'", # Override applied
                "outputs.dpi == 200",                  # Override applied
                "kedro.enable_versioning == True"      # Kedro-specific added
            ]
        }
    }