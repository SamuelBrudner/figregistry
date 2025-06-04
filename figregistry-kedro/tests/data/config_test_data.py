"""
Configuration test data generators and utilities for FigRegistry-Kedro integration testing.

This module provides comprehensive test data generation capabilities for validating
FigRegistryConfigBridge functionality, including environment-specific configuration
scenarios, configuration merging test cases, Pydantic validation testing data,
and security testing scenarios for robust configuration handling.

Key Features:
- Environment-specific configuration generators for development, staging, and production
- Invalid configuration generators for comprehensive error handling validation
- Merged configuration scenarios for testing precedence rules per Section 5.2.5
- Hypothesis strategies for property-based testing with comprehensive coverage
- Security test configurations for injection prevention per Section 6.6.8.1
- Performance testing datasets for configuration merge benchmarking
- Cross-platform configuration variations ensuring consistent behavior

The module supports testing requirements per F-007.2 multi-environment configuration
management and Section 6.6.2.6 property-based testing integration with comprehensive
validation coverage for the FigRegistryConfigBridge component.
"""

import copy
import string
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
from uuid import uuid4

try:
    from hypothesis import strategies as st
    from hypothesis.strategies import composite
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    st = None
    composite = None

# Type hints for configuration structures
ConfigDict = Dict[str, Any]
EnvironmentConfig = Dict[str, ConfigDict]
MergeScenario = Tuple[str, ConfigDict, ConfigDict, ConfigDict]
ValidationScenario = Dict[str, Any]

# Constants for configuration generation
VALID_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d3", "#c7c7c7"
]

VALID_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "H", "*", "+", "x", "|", "_"]
VALID_LINESTYLES = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
VALID_FORMATS = ["png", "pdf", "svg", "eps", "tiff", "jpg", "jpeg"]


# =============================================================================
# Environment-Specific Configuration Generators per F-007.2
# =============================================================================

def generate_development_config() -> ConfigDict:
    """
    Generate development environment configuration for testing F-007.2 requirements.
    
    Provides optimized settings for rapid development iteration including reduced
    DPI for faster rendering, simplified styling, and debug-friendly configurations.
    
    Returns:
        Development-optimized configuration dictionary with debug features enabled.
    """
    return {
        "figregistry_version": "0.3.0",
        "environment": "development",
        "metadata": {
            "config_version": "1.0.0",
            "environment": "development",
            "created": datetime.now(timezone.utc).isoformat(),
            "description": "Development configuration for rapid iteration",
            "debug_enabled": True,
            "performance_monitoring": False
        },
        "styles": {
            "control": {
                "color": "#1f77b4",
                "marker": "o",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Control (Dev)",
                "markersize": 5
            },
            "treatment": {
                "color": "#ff7f0e",
                "marker": "s",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Treatment (Dev)",
                "markersize": 5
            },
            "debug_*": {
                "color": "#d62728",
                "marker": "x",
                "linestyle": ":",
                "linewidth": 1.0,
                "alpha": 0.5,
                "label": "Debug",
                "markersize": 4
            },
            "experimental_*": {
                "color": "#9467bd",
                "marker": "D",
                "linestyle": "--",
                "linewidth": 1.0,
                "alpha": 0.6,
                "label": "Experimental",
                "markersize": 4
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            "debug": ["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
            "grayscale": ["#000000", "#404040", "#808080", "#c0c0c0"]
        },
        "outputs": {
            "base_path": "dev_figures",
            "dpi": 72,  # Low DPI for faster iteration
            "default_format": "png",
            "formats": ["png", "svg"],  # Limited formats for speed
            "timestamp_format": "dev_{name}_{ts:%H%M%S}",
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "facecolor": "white",
            "transparent": False,
            "metadata": True,
            "path_aliases": {
                "debug": "debug",
                "temp": "temp",
                "test": "test"
            }
        },
        "defaults": {
            "figure_size": [8, 5],  # Smaller for faster rendering
            "font_size": 10,
            "title_size": 12,
            "label_size": 9,
            "tick_size": 8,
            "line_width": 1.0,
            "marker_size": 5,
            "grid": False,  # Cleaner for development
            "grid_alpha": 0.2,
            "spine_visibility": {
                "top": False,
                "right": False,
                "bottom": True,
                "left": True
            },
            "legend_frameon": False,
            "legend_fancybox": False
        },
        "kedro": {
            "enable_versioning": False,
            "parallel_execution": False,
            "debug_mode": True,
            "cache_styling": False,  # Fresh styles for development
            "performance_monitoring": False,
            "validation_enabled": False,  # Relaxed validation for speed
            "hook_execution_timing": True
        },
        "performance": {
            "cache_enabled": False,
            "lazy_loading": False,
            "validation_strict": False,
            "performance_target_ms": 50.0  # Relaxed for development
        }
    }


def generate_staging_config() -> ConfigDict:
    """
    Generate staging environment configuration for pre-production testing.
    
    Provides production-like settings with enhanced monitoring and validation
    for comprehensive testing before production deployment.
    
    Returns:
        Staging configuration dictionary with production-like features.
    """
    return {
        "figregistry_version": "0.3.0",
        "environment": "staging",
        "metadata": {
            "config_version": "1.0.0",
            "environment": "staging",
            "created": datetime.now(timezone.utc).isoformat(),
            "description": "Staging configuration for pre-production validation",
            "debug_enabled": False,
            "performance_monitoring": True
        },
        "styles": {
            "control": {
                "color": "#1f77b4",
                "marker": "o",
                "linewidth": 2.0,
                "alpha": 0.8,
                "label": "Control",
                "markersize": 6,
                "zorder": 2
            },
            "treatment": {
                "color": "#ff7f0e",
                "marker": "s",
                "linewidth": 2.0,
                "alpha": 0.8,
                "label": "Treatment",
                "markersize": 6,
                "zorder": 2
            },
            "validation_*": {
                "color": "#9467bd",
                "marker": "v",
                "linestyle": "-.",
                "linewidth": 1.5,
                "alpha": 0.6,
                "label": "Validation",
                "markersize": 5,
                "zorder": 1
            },
            "baseline_*": {
                "color": "#2ca02c",
                "marker": "^",
                "linestyle": "-",
                "linewidth": 1.8,
                "alpha": 0.75,
                "label": "Baseline",
                "markersize": 5,
                "zorder": 1
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "validation": ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
            "colorblind_safe": ["#0173b2", "#de8f05", "#029e73", "#cc78bc"]
        },
        "outputs": {
            "base_path": "staging/figures",
            "dpi": 200,  # Medium quality for staging
            "default_format": "png",
            "formats": ["png", "pdf", "svg"],
            "timestamp_format": "staging_{name}_{ts:%Y%m%d_%H%M}",
            "bbox_inches": "tight",
            "pad_inches": 0.15,
            "facecolor": "white",
            "transparent": False,
            "metadata": True,
            "path_aliases": {
                "val": "validation",
                "base": "baseline",
                "test": "testing",
                "review": "review"
            }
        },
        "defaults": {
            "figure_size": [10, 6],
            "font_size": 12,
            "title_size": 14,
            "label_size": 11,
            "tick_size": 9,
            "line_width": 1.5,
            "marker_size": 6,
            "grid": True,
            "grid_alpha": 0.3,
            "spine_visibility": {
                "top": False,
                "right": False,
                "bottom": True,
                "left": True
            },
            "legend_frameon": True,
            "legend_fancybox": True,
            "legend_shadow": False
        },
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "debug_mode": False,
            "cache_styling": True,
            "performance_monitoring": True,
            "validation_enabled": True,
            "hook_execution_timing": True,
            "enable_concurrent_access": True
        },
        "performance": {
            "cache_enabled": True,
            "lazy_loading": True,
            "validation_strict": True,
            "performance_target_ms": 20.0,
            "max_cache_size": 500
        },
        "validation": {
            "schema_validation": True,
            "type_checking": True,
            "parameter_validation": True,
            "path_validation": True
        }
    }


def generate_production_config() -> ConfigDict:
    """
    Generate production environment configuration for deployment.
    
    Provides optimal settings for production deployment including high-quality
    output, comprehensive monitoring, and robust error handling.
    
    Returns:
        Production configuration dictionary with enterprise-grade features.
    """
    return {
        "figregistry_version": "0.3.0",
        "environment": "production",
        "metadata": {
            "config_version": "1.0.0",
            "environment": "production",
            "created": datetime.now(timezone.utc).isoformat(),
            "description": "Production configuration for enterprise deployment",
            "debug_enabled": False,
            "performance_monitoring": True,
            "security_enabled": True
        },
        "styles": {
            "control": {
                "color": "#1f77b4",
                "marker": "o",
                "linewidth": 2.5,  # Thicker for production quality
                "alpha": 1.0,      # Full opacity for final output
                "label": "Control",
                "markersize": 7,
                "zorder": 3
            },
            "treatment": {
                "color": "#ff7f0e",
                "marker": "s",
                "linewidth": 2.5,
                "alpha": 1.0,
                "label": "Treatment",
                "markersize": 7,
                "zorder": 3
            },
            "final_*": {
                "color": "#2ca02c",
                "marker": "^",
                "linestyle": "-",
                "linewidth": 2.0,
                "alpha": 0.9,
                "label": "Final",
                "markersize": 6,
                "zorder": 2
            },
            "publication_*": {
                "color": "#000000",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 2.5,
                "alpha": 1.0,
                "label": "Publication",
                "markersize": 6,
                "zorder": 4
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            "publication": ["#000000", "#404040", "#666666", "#999999", "#cccccc"],
            "colorblind_safe": ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc"],
            "high_contrast": ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"]
        },
        "outputs": {
            "base_path": "/opt/kedro/production/figures",
            "dpi": 300,  # High quality for production
            "default_format": "pdf",  # Vector format for production
            "formats": ["pdf", "png", "svg", "eps"],
            "timestamp_format": "prod_{name}_{ts:%Y%m%d_%H%M%S}",
            "bbox_inches": "tight",
            "pad_inches": 0.2,
            "facecolor": "white",
            "transparent": False,
            "metadata": True,
            "path_aliases": {
                "final": "final_output",
                "pub": "publication",
                "archive": "archived",
                "report": "reporting"
            }
        },
        "defaults": {
            "figure_size": [12, 8],  # Larger for publication quality
            "font_size": 14,
            "title_size": 16,
            "label_size": 12,
            "tick_size": 10,
            "line_width": 2.0,
            "marker_size": 7,
            "grid": True,
            "grid_alpha": 0.2,
            "spine_visibility": {
                "top": False,
                "right": False,
                "bottom": True,
                "left": True
            },
            "spine_linewidth": 1.2,
            "legend_frameon": True,
            "legend_fancybox": True,
            "legend_shadow": True
        },
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "debug_mode": False,
            "cache_styling": True,
            "performance_monitoring": True,
            "validation_enabled": True,
            "hook_execution_timing": False,  # Disabled for production performance
            "enable_concurrent_access": True,
            "error_handling": "strict"
        },
        "performance": {
            "cache_enabled": True,
            "lazy_loading": True,
            "validation_strict": True,
            "performance_target_ms": 10.0,  # Strict production target
            "max_cache_size": 1000
        },
        "validation": {
            "schema_validation": True,
            "type_checking": True,
            "parameter_validation": True,
            "path_validation": True,
            "security_validation": True
        },
        "monitoring": {
            "performance_tracking": True,
            "error_reporting": True,
            "usage_analytics": True,
            "health_checks": True
        }
    }


def generate_baseline_config() -> ConfigDict:
    """
    Generate baseline configuration for standard testing scenarios.
    
    Provides a comprehensive but neutral configuration that serves as the
    foundation for testing configuration merging and precedence rules.
    
    Returns:
        Baseline configuration dictionary with all major sections populated.
    """
    return {
        "figregistry_version": "0.3.0",
        "environment": "base",
        "metadata": {
            "config_version": "1.0.0",
            "environment": "base",
            "created": datetime.now(timezone.utc).isoformat(),
            "description": "Baseline configuration for testing",
            "author": "FigRegistry Test Suite"
        },
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
        }
    }


# =============================================================================
# Invalid Configuration Generators for Error Handling Testing
# =============================================================================

def generate_invalid_config_missing_version() -> ConfigDict:
    """Generate configuration missing required figregistry_version field."""
    return {
        "styles": {"control": {"color": "#1f77b4", "marker": "o"}},
        "outputs": {"base_path": "figures"}
        # Missing figregistry_version
    }


def generate_invalid_config_malformed_styles() -> ConfigDict:
    """Generate configuration with malformed styles section."""
    return {
        "figregistry_version": "0.3.0",
        "styles": "not_a_dict",  # Should be dictionary
        "outputs": {"base_path": "figures"}
    }


def generate_invalid_config_invalid_colors() -> ConfigDict:
    """Generate configuration with invalid color specifications."""
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {
                "color": "invalid_color_format",  # Invalid color
                "marker": "o"
            },
            "treatment": {
                "color": "#gggggg",  # Invalid hex color
                "marker": "s"
            },
            "invalid_alpha": {
                "color": "#1f77b4",
                "alpha": 1.5  # Alpha > 1.0
            }
        }
    }


def generate_invalid_config_negative_values() -> ConfigDict:
    """Generate configuration with negative numeric values."""
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {
                "color": "#1f77b4",
                "linewidth": -1.5,  # Negative linewidth
                "markersize": -5,   # Negative markersize
                "alpha": -0.5       # Negative alpha
            }
        },
        "outputs": {
            "dpi": -100  # Negative DPI
        }
    }


def generate_invalid_config_wrong_types() -> ConfigDict:
    """Generate configuration with incorrect data types."""
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {
                "color": "#1f77b4",
                "linewidth": "not_a_number",  # Should be numeric
                "alpha": "maybe",             # Should be float
                "markersize": True            # Should be numeric
            }
        },
        "outputs": {
            "base_path": 12345,    # Should be string
            "dpi": "high",         # Should be numeric
            "formats": "png"       # Should be list
        },
        "defaults": {
            "grid": "sometimes",   # Should be boolean
            "figure_size": "large" # Should be list/tuple
        }
    }


def generate_invalid_config_empty_required_sections() -> ConfigDict:
    """Generate configuration with empty required sections."""
    return {
        "figregistry_version": "0.3.0",
        "styles": {},  # Empty styles not allowed
        "outputs": {}  # Empty outputs might be problematic
    }


def generate_invalid_config_circular_references() -> ConfigDict:
    """Generate configuration attempting circular references."""
    config = {
        "figregistry_version": "0.3.0",
        "styles": {
            "ref_style": {
                "color": "#1f77b4",
                "inherits_from": "ref_style"  # Self-reference
            }
        }
    }
    return config


def generate_invalid_config_path_traversal() -> ConfigDict:
    """Generate configuration with path traversal attempts."""
    return {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#1f77b4"}},
        "outputs": {
            "base_path": "../../../etc/passwd",  # Path traversal attempt
            "path_aliases": {
                "malicious": "../../../../system",
                "exploit": "/etc/hosts"
            }
        }
    }


# =============================================================================
# Configuration Merging Test Scenarios per Section 5.2.5
# =============================================================================

def generate_merged_config_scenarios() -> List[MergeScenario]:
    """
    Generate comprehensive configuration merge scenarios for precedence testing.
    
    Provides test cases for validating configuration merging logic, precedence
    rules, and edge cases in the FigRegistryConfigBridge per Section 5.2.5.
    
    Returns:
        List of tuples containing (scenario_name, base_config, override_config, expected_merged).
    """
    scenarios = []
    
    # Scenario 1: Simple field override
    base_simple = {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#1f77b4", "marker": "o", "linewidth": 1.0}},
        "outputs": {"base_path": "figures", "dpi": 300}
    }
    override_simple = {
        "styles": {"control": {"color": "#ff0000", "linewidth": 2.0}},
        "outputs": {"dpi": 150}
    }
    expected_simple = {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#ff0000", "marker": "o", "linewidth": 2.0}},
        "outputs": {"base_path": "figures", "dpi": 150}
    }
    scenarios.append(("simple_override", base_simple, override_simple, expected_simple))
    
    # Scenario 2: Deep merge with new sections
    base_deep = {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {"color": "#1f77b4", "marker": "o"},
            "treatment": {"color": "#ff7f0e", "marker": "s"}
        },
        "defaults": {"figure_size": [10, 6], "font_size": 12}
    }
    override_deep = {
        "styles": {
            "control": {"linewidth": 2.0, "alpha": 0.8},
            "new_condition": {"color": "#00ff00", "marker": "^"}
        },
        "defaults": {"font_size": 14},
        "kedro": {"debug_mode": True}
    }
    expected_deep = {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0, "alpha": 0.8},
            "treatment": {"color": "#ff7f0e", "marker": "s"},
            "new_condition": {"color": "#00ff00", "marker": "^"}
        },
        "defaults": {"figure_size": [10, 6], "font_size": 14},
        "kedro": {"debug_mode": True}
    }
    scenarios.append(("deep_merge", base_deep, override_deep, expected_deep))
    
    # Scenario 3: List replacement (not merging)
    base_list = {
        "figregistry_version": "0.3.0",
        "palettes": {"default": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
        "outputs": {"formats": ["png", "pdf"]}
    }
    override_list = {
        "palettes": {"default": ["#000000", "#ffffff"]},
        "outputs": {"formats": ["svg"]}
    }
    expected_list = {
        "figregistry_version": "0.3.0",
        "palettes": {"default": ["#000000", "#ffffff"]},
        "outputs": {"formats": ["svg"]}
    }
    scenarios.append(("list_replacement", base_list, override_list, expected_list))
    
    # Scenario 4: Environment-specific precedence
    base_env = {
        "figregistry_version": "0.3.0",
        "environment": "base",
        "styles": {"control": {"color": "#1f77b4", "alpha": 0.8}},
        "outputs": {"base_path": "figures", "dpi": 300}
    }
    override_env = {
        "environment": "local",
        "styles": {"control": {"alpha": 1.0, "linewidth": 2.5}},
        "outputs": {"base_path": "local_figures"},
        "kedro": {"enable_versioning": False}
    }
    expected_env = {
        "figregistry_version": "0.3.0",
        "environment": "local",
        "styles": {"control": {"color": "#1f77b4", "alpha": 1.0, "linewidth": 2.5}},
        "outputs": {"base_path": "local_figures", "dpi": 300},
        "kedro": {"enable_versioning": False}
    }
    scenarios.append(("environment_precedence", base_env, override_env, expected_env))
    
    # Scenario 5: Complex nested override
    base_nested = {
        "figregistry_version": "0.3.0",
        "defaults": {
            "spine_visibility": {"top": False, "right": False, "bottom": True, "left": True},
            "figure_size": [10, 6],
            "grid_settings": {"enabled": True, "alpha": 0.3, "color": "#cccccc"}
        }
    }
    override_nested = {
        "defaults": {
            "spine_visibility": {"top": True},  # Only override top
            "font_size": 14,  # Add new field
            "grid_settings": {"alpha": 0.5}  # Partial override
        }
    }
    expected_nested = {
        "figregistry_version": "0.3.0",
        "defaults": {
            "spine_visibility": {"top": True, "right": False, "bottom": True, "left": True},
            "figure_size": [10, 6],
            "font_size": 14,
            "grid_settings": {"enabled": True, "alpha": 0.5, "color": "#cccccc"}
        }
    }
    scenarios.append(("nested_merge", base_nested, override_nested, expected_nested))
    
    # Scenario 6: Kedro-specific configuration addition
    base_kedro = {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#1f77b4"}},
        "outputs": {"base_path": "figures"}
    }
    override_kedro = {
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "datasets": {"default_purpose": "exploratory"},
            "hooks": {"execution_timing": True}
        }
    }
    expected_kedro = {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#1f77b4"}},
        "outputs": {"base_path": "figures"},
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "datasets": {"default_purpose": "exploratory"},
            "hooks": {"execution_timing": True}
        }
    }
    scenarios.append(("kedro_addition", base_kedro, override_kedro, expected_kedro))
    
    return scenarios


def generate_precedence_test_cases() -> List[ValidationScenario]:
    """
    Generate test cases for validating configuration precedence rules.
    
    Returns:
        List of test case dictionaries with configuration scenarios and expected outcomes.
    """
    return [
        {
            "name": "kedro_overrides_figregistry",
            "description": "Kedro configurations should override FigRegistry base settings",
            "figregistry_config": {
                "figregistry_version": "0.3.0",
                "styles": {"control": {"color": "#1f77b4", "linewidth": 1.0}},
                "outputs": {"dpi": 300}
            },
            "kedro_config": {
                "styles": {"control": {"color": "#ff0000"}},
                "outputs": {"dpi": 150}
            },
            "expected_precedence": {
                "styles.control.color": "#ff0000",    # Kedro wins
                "styles.control.linewidth": 1.0,      # FigRegistry preserved
                "outputs.dpi": 150                    # Kedro wins
            }
        },
        {
            "name": "environment_specific_override",
            "description": "Environment-specific configs override base configs",
            "base_config": {
                "figregistry_version": "0.3.0",
                "environment": "base",
                "styles": {"control": {"color": "#1f77b4"}},
                "outputs": {"base_path": "figures"}
            },
            "env_config": {
                "environment": "local",
                "styles": {"control": {"color": "#00ff00"}},
                "outputs": {"base_path": "local_figures"}
            },
            "expected_precedence": {
                "environment": "local",
                "styles.control.color": "#00ff00",
                "outputs.base_path": "local_figures"
            }
        },
        {
            "name": "parameter_override_priority",
            "description": "Direct parameter overrides have highest priority",
            "base_config": {
                "figregistry_version": "0.3.0",
                "styles": {"control": {"color": "#1f77b4", "alpha": 0.8}}
            },
            "kedro_config": {
                "styles": {"control": {"color": "#ff0000", "alpha": 0.9}}
            },
            "parameter_overrides": {
                "styles": {"control": {"alpha": 1.0}}
            },
            "expected_precedence": {
                "styles.control.color": "#ff0000",  # Kedro wins
                "styles.control.alpha": 1.0         # Parameter override wins
            }
        }
    ]


# =============================================================================
# Property-Based Testing with Hypothesis Strategies per Section 6.6.2.6
# =============================================================================

if HAS_HYPOTHESIS:
    @composite
    def valid_color_strategy(draw):
        """Generate valid color specifications for property-based testing."""
        color_type = draw(st.sampled_from(["hex", "named", "rgb"]))
        
        if color_type == "hex":
            # Generate valid hex colors
            hex_value = draw(st.integers(min_value=0, max_value=0xFFFFFF))
            return f"#{hex_value:06x}"
        elif color_type == "named":
            # Use predefined valid colors
            return draw(st.sampled_from(VALID_COLORS))
        else:  # rgb
            r = draw(st.floats(min_value=0.0, max_value=1.0))
            g = draw(st.floats(min_value=0.0, max_value=1.0))
            b = draw(st.floats(min_value=0.0, max_value=1.0))
            return (r, g, b)

    @composite
    def valid_style_strategy(draw):
        """Generate valid style configurations for property-based testing."""
        return {
            "color": draw(valid_color_strategy()),
            "marker": draw(st.sampled_from(VALID_MARKERS)),
            "linestyle": draw(st.sampled_from(VALID_LINESTYLES)),
            "linewidth": draw(st.floats(min_value=0.1, max_value=10.0)),
            "markersize": draw(st.floats(min_value=1.0, max_value=20.0)),
            "alpha": draw(st.floats(min_value=0.0, max_value=1.0)),
            "label": draw(st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + string.digits + " _-"))
        }

    @composite
    def valid_outputs_strategy(draw):
        """Generate valid outputs configuration for property-based testing."""
        return {
            "base_path": draw(st.text(min_size=1, max_size=100, alphabet=string.ascii_letters + string.digits + "_-/")),
            "dpi": draw(st.integers(min_value=50, max_value=600)),
            "default_format": draw(st.sampled_from(VALID_FORMATS)),
            "formats": draw(st.lists(st.sampled_from(VALID_FORMATS), min_size=1, max_size=5, unique=True)),
            "bbox_inches": draw(st.sampled_from(["tight", "standard"])),
            "pad_inches": draw(st.floats(min_value=0.0, max_value=1.0))
        }

    @composite
    def valid_figregistry_config_strategy(draw):
        """Generate complete valid FigRegistry configuration for property-based testing."""
        num_styles = draw(st.integers(min_value=1, max_value=10))
        styles = {}
        for i in range(num_styles):
            style_name = f"style_{i}" if i < 5 else f"pattern_{i}_*"
            styles[style_name] = draw(valid_style_strategy())

        return {
            "figregistry_version": draw(st.sampled_from(["0.3.0", "0.3.1", "0.4.0"])),
            "styles": styles,
            "outputs": draw(valid_outputs_strategy()),
            "defaults": {
                "figure_size": draw(st.lists(st.floats(min_value=4.0, max_value=20.0), min_size=2, max_size=2)),
                "font_size": draw(st.integers(min_value=8, max_value=24)),
                "grid": draw(st.booleans())
            }
        }

    @composite
    def malformed_config_strategy(draw):
        """Generate malformed configurations for property-based error testing."""
        error_type = draw(st.sampled_from([
            "missing_version", "invalid_styles", "negative_values", 
            "wrong_types", "empty_sections", "invalid_paths"
        ]))
        
        if error_type == "missing_version":
            return {"styles": {"test": {"color": "#1f77b4"}}}
        elif error_type == "invalid_styles":
            return {
                "figregistry_version": "0.3.0",
                "styles": draw(st.text())  # Should be dict
            }
        elif error_type == "negative_values":
            return {
                "figregistry_version": "0.3.0",
                "styles": {
                    "test": {
                        "color": "#1f77b4",
                        "linewidth": draw(st.floats(max_value=-0.1)),
                        "alpha": draw(st.floats(max_value=-0.1))
                    }
                }
            }
        elif error_type == "wrong_types":
            return {
                "figregistry_version": "0.3.0",
                "styles": {
                    "test": {
                        "color": draw(st.integers()),  # Should be string
                        "linewidth": draw(st.text()),  # Should be float
                        "alpha": draw(st.text())       # Should be float
                    }
                }
            }
        elif error_type == "empty_sections":
            return {
                "figregistry_version": "0.3.0",
                "styles": {}  # Empty styles
            }
        else:  # invalid_paths
            return {
                "figregistry_version": "0.3.0",
                "styles": {"test": {"color": "#1f77b4"}},
                "outputs": {
                    "base_path": "../" * draw(st.integers(min_value=1, max_value=10)) + "etc/passwd"
                }
            }

else:
    # Fallback functions when Hypothesis is not available
    def valid_color_strategy():
        """Fallback for valid color generation without Hypothesis."""
        return random.choice(VALID_COLORS)

    def valid_style_strategy():
        """Fallback for valid style generation without Hypothesis."""
        return {
            "color": random.choice(VALID_COLORS),
            "marker": random.choice(VALID_MARKERS),
            "linestyle": random.choice(VALID_LINESTYLES),
            "linewidth": random.uniform(0.5, 5.0),
            "alpha": random.uniform(0.1, 1.0),
            "label": f"Style_{random.randint(1, 100)}"
        }


# =============================================================================
# Security Testing Configurations per Section 6.6.8.1
# =============================================================================

def generate_yaml_injection_config() -> str:
    """
    Generate YAML configuration with potential injection vectors.
    
    Returns:
        YAML string containing potential security vulnerabilities for testing.
    """
    return """
figregistry_version: "0.3.0"
styles:
  control:
    color: "#1f77b4"
    # Attempt to inject Python code
    exec_attempt: !!python/object/apply:os.system ["rm -rf /"]
    
  # Attempt code execution through YAML tags
  malicious: !!python/object/apply:subprocess.check_output [["cat", "/etc/passwd"]]

outputs:
  # Path traversal attempt
  base_path: "../../../etc"
  
# Attempt to load external YAML files
external_ref: &external_file
  !!python/object/apply:yaml.load [!!python/object/apply:urllib.request.urlopen ["file:///etc/passwd"]]
"""


def generate_path_traversal_configs() -> List[ConfigDict]:
    """
    Generate configurations with path traversal attempts.
    
    Returns:
        List of configuration dictionaries containing path traversal vectors.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "../../../etc/passwd",
                "path_aliases": {
                    "exploit": "../../../../system",
                    "backdoor": "/etc/shadow"
                }
            }
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "figures/../../../home/user/.ssh/id_rsa"
            }
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "kedro": {
                "config_path": "../../../etc/kedro_secrets.yml"
            }
        }
    ]


def generate_malicious_parameter_configs() -> List[ConfigDict]:
    """
    Generate configurations with malicious parameter injection attempts.
    
    Returns:
        List of configurations attempting parameter injection.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {
                "control": {
                    "color": "#1f77b4",
                    "command_injection": "; rm -rf /; echo 'pwned'",
                    "script_injection": "<script>alert('xss')</script>"
                }
            }
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {
                "$(whoami)": {"color": "#ff0000"},  # Command substitution attempt
                "`cat /etc/passwd`": {"color": "#00ff00"}  # Backtick injection
            }
        },
        {
            "figregistry_version": "0.3.0",
            "outputs": {
                "base_path": "figures",
                "filename_template": "{name}; cat /etc/passwd > /tmp/pwned; {ts}"
            }
        }
    ]


def generate_unicode_exploit_configs() -> List[ConfigDict]:
    """
    Generate configurations with Unicode and encoding exploits.
    
    Returns:
        List of configurations attempting Unicode-based attacks.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {
                "control": {
                    "color": "#1f77b4",
                    "unicode_exploit": "\uFEFF\u202E\u200B",  # BOM + RTL override + ZWSP
                    "surrogate_pair": "\uD800\uDC00"  # Surrogate pair
                }
            }
        },
        {
            "figregistry_version": "0.3.0",
            "metadata": {
                "author": "\x00\x01\x02\x03",  # Null bytes and control chars
                "description": "Normal\u0000Hidden\u0000Text"  # Null byte injection
            }
        }
    ]


def generate_configuration_bomb_scenarios() -> List[ConfigDict]:
    """
    Generate configurations that could cause resource exhaustion.
    
    Returns:
        List of configurations designed to test resource limits.
    """
    # Generate large configuration for memory testing
    large_config = {
        "figregistry_version": "0.3.0",
        "styles": {},
        "palettes": {}
    }
    
    # Create many style entries to test memory limits
    for i in range(10000):
        large_config["styles"][f"style_{i}"] = {
            "color": f"#{i:06x}",
            "marker": "o",
            "linewidth": 1.0,
            "data": "x" * 1000  # Large string to consume memory
        }
    
    # Create deeply nested structure
    deeply_nested = {
        "figregistry_version": "0.3.0",
        "styles": {"control": {"color": "#1f77b4"}}
    }
    current = deeply_nested
    for i in range(1000):  # Deep nesting
        current[f"level_{i}"] = {}
        current = current[f"level_{i}"]
    
    return [large_config, deeply_nested]


# =============================================================================
# Performance Testing Configurations per Section 6.6.4.3
# =============================================================================

def generate_performance_config_small() -> ConfigDict:
    """
    Generate small configuration for performance baseline testing.
    
    Returns:
        Minimal configuration for measuring baseline performance.
    """
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "control": {"color": "#1f77b4", "marker": "o"},
            "treatment": {"color": "#ff7f0e", "marker": "s"}
        },
        "outputs": {"base_path": "figures", "dpi": 300},
        "defaults": {"figure_size": [10, 6]}
    }


def generate_performance_config_medium() -> ConfigDict:
    """
    Generate medium-complexity configuration for performance testing.
    
    Returns:
        Moderately complex configuration targeting <20ms merge time.
    """
    config = {
        "figregistry_version": "0.3.0",
        "styles": {},
        "palettes": {},
        "outputs": {
            "base_path": "figures",
            "dpi": 300,
            "formats": ["png", "pdf", "svg"],
            "path_aliases": {}
        },
        "defaults": {
            "figure_size": [10, 6],
            "font_size": 12,
            "spine_visibility": {
                "top": False, "right": False, "bottom": True, "left": True
            }
        }
    }
    
    # Generate 50 style conditions
    for i in range(50):
        config["styles"][f"condition_{i}"] = {
            "color": VALID_COLORS[i % len(VALID_COLORS)],
            "marker": VALID_MARKERS[i % len(VALID_MARKERS)],
            "linewidth": 1.0 + (i % 3),
            "alpha": 0.5 + (i % 5) * 0.1
        }
    
    # Generate 10 color palettes
    for i in range(10):
        config["palettes"][f"palette_{i}"] = VALID_COLORS[i:i+5]
    
    # Generate 20 path aliases
    for i in range(20):
        config["outputs"]["path_aliases"][f"alias_{i}"] = f"path_{i}"
    
    return config


def generate_performance_config_large() -> ConfigDict:
    """
    Generate large configuration for stress testing performance limits.
    
    Returns:
        Complex configuration targeting <50ms merge time under stress.
    """
    config = {
        "figregistry_version": "0.3.0",
        "metadata": {
            "config_version": "2.0.0",
            "environment": "performance_test",
            "description": "Large configuration for performance stress testing",
            "tags": [f"tag_{i}" for i in range(20)]
        },
        "styles": {},
        "palettes": {},
        "outputs": {
            "base_path": "performance_figures",
            "dpi": 300,
            "formats": ["png", "pdf", "svg", "eps", "tiff"],
            "path_aliases": {},
            "timestamp_format": "perf_{name}_{condition}_{ts:%Y%m%d_%H%M%S}",
            "metadata": True
        },
        "defaults": {
            "figure_size": [12, 8],
            "font_size": 14,
            "title_size": 16,
            "label_size": 12,
            "tick_size": 10,
            "legend_size": 11,
            "line_width": 2.0,
            "marker_size": 8,
            "grid": True,
            "grid_alpha": 0.3,
            "spine_visibility": {
                "top": False, "right": False, "bottom": True, "left": True
            },
            "spine_linewidth": 1.0,
            "legend_frameon": True,
            "legend_fancybox": True,
            "legend_shadow": False
        },
        "kedro": {
            "enable_versioning": True,
            "parallel_execution": True,
            "debug_mode": False,
            "cache_styling": True,
            "performance_monitoring": True,
            "validation_enabled": True,
            "datasets": {"default_purpose": "exploratory"}
        }
    }
    
    # Generate 200 style conditions
    for i in range(200):
        config["styles"][f"condition_{i}"] = {
            "color": f"#{(i * 12345) % 0xFFFFFF:06x}",
            "marker": VALID_MARKERS[i % len(VALID_MARKERS)],
            "linestyle": VALID_LINESTYLES[i % len(VALID_LINESTYLES)],
            "linewidth": 0.5 + (i % 10) * 0.3,
            "markersize": 3 + (i % 8),
            "alpha": 0.3 + (i % 7) * 0.1,
            "label": f"Condition {i}",
            "zorder": i % 5
        }
        
        # Add wildcard patterns every 10th style
        if i % 10 == 0:
            config["styles"][f"pattern_{i}_*"] = {
                "color": f"#{(i * 54321) % 0xFFFFFF:06x}",
                "marker": "x",
                "linestyle": ":",
                "alpha": 0.4
            }
    
    # Generate 30 color palettes
    for i in range(30):
        palette_size = 5 + (i % 8)
        config["palettes"][f"palette_{i}"] = [
            f"#{(j * i * 7919) % 0xFFFFFF:06x}" for j in range(palette_size)
        ]
    
    # Generate 100 path aliases
    for i in range(100):
        config["outputs"]["path_aliases"][f"alias_{i}"] = f"path/to/output/{i}"
    
    return config


def generate_performance_benchmark_configs() -> Dict[str, ConfigDict]:
    """
    Generate complete suite of performance benchmark configurations.
    
    Returns:
        Dictionary mapping benchmark names to configuration dictionaries.
    """
    return {
        "minimal": generate_performance_config_small(),
        "moderate": generate_performance_config_medium(),
        "complex": generate_performance_config_large(),
        "merge_intensive": {
            "figregistry_version": "0.3.0",
            "styles": {f"style_{i}": {"color": "#1f77b4"} for i in range(1000)},
            "deep_nesting": {
                "level_1": {
                    "level_2": {
                        "level_3": {
                            "level_4": {
                                "level_5": {
                                    "config": "deep_value"
                                }
                            }
                        }
                    }
                }
            }
        }
    }


# =============================================================================
# Cross-Platform Configuration Variations per Section 6.6.1.4
# =============================================================================

def generate_windows_config_variations() -> List[ConfigDict]:
    """
    Generate Windows-specific configuration variations for cross-platform testing.
    
    Returns:
        List of configuration dictionaries with Windows-specific path formats.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "C:\\Users\\User\\Documents\\figures",
                "path_aliases": {
                    "temp": "C:\\temp\\figures",
                    "shared": "\\\\server\\share\\figures"
                }
            }
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "figures",  # Relative path
                "filename_template": "{name}_{ts:%Y-%m-%d_%H-%M-%S}.png"  # Windows-safe timestamps
            }
        }
    ]


def generate_unix_config_variations() -> List[ConfigDict]:
    """
    Generate Unix/Linux-specific configuration variations for cross-platform testing.
    
    Returns:
        List of configuration dictionaries with Unix-specific path formats.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "/home/user/figures",
                "path_aliases": {
                    "tmp": "/tmp/figures",
                    "shared": "/mnt/shared/figures"
                }
            }
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "~/figures",  # Home directory expansion
                "filename_template": "{name}_{ts:%Y%m%d_%H%M%S}.png"
            }
        }
    ]


def generate_macos_config_variations() -> List[ConfigDict]:
    """
    Generate macOS-specific configuration variations for cross-platform testing.
    
    Returns:
        List of configuration dictionaries with macOS-specific path formats.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "/Users/user/Documents/figures",
                "path_aliases": {
                    "desktop": "/Users/user/Desktop/figures",
                    "applications": "/Applications/Kedro/figures"
                }
            }
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {
                "base_path": "figures",
                "filename_template": "{name}_{ts:%Y%m%d_%H%M%S}.png"
            }
        }
    ]


def generate_path_separator_configs() -> List[ConfigDict]:
    """
    Generate configurations testing various path separator handling.
    
    Returns:
        List of configurations with different path separator styles.
    """
    return [
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {"base_path": "figures/output/final"}  # Forward slashes
        },
        {
            "figregistry_version": "0.3.0", 
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {"base_path": "figures\\output\\final"}  # Backslashes
        },
        {
            "figregistry_version": "0.3.0",
            "styles": {"control": {"color": "#1f77b4"}},
            "outputs": {"base_path": "figures/output\\mixed/separators"}  # Mixed
        }
    ]


# =============================================================================
# Utility Functions for Test Data Generation
# =============================================================================

def generate_random_config_id() -> str:
    """Generate unique configuration identifier for test isolation."""
    return f"test_config_{uuid4().hex[:8]}"


def generate_timestamp_config() -> ConfigDict:
    """Generate configuration with current timestamp for temporal testing."""
    timestamp = datetime.now(timezone.utc)
    return {
        "figregistry_version": "0.3.0",
        "metadata": {
            "created": timestamp.isoformat(),
            "test_id": generate_random_config_id()
        },
        "styles": {"control": {"color": "#1f77b4"}},
        "outputs": {
            "base_path": f"figures_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        }
    }


def apply_config_overrides(base_config: ConfigDict, overrides: ConfigDict) -> ConfigDict:
    """
    Apply configuration overrides using deep merge logic.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    return deep_merge(base_config, overrides)


def validate_config_structure(config: ConfigDict) -> List[str]:
    """
    Validate configuration structure and return list of issues.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check required fields
    if "figregistry_version" not in config:
        errors.append("Missing required field: figregistry_version")
    
    # Check styles section
    if "styles" not in config:
        errors.append("Missing required section: styles")
    elif not isinstance(config["styles"], dict):
        errors.append("styles section must be a dictionary")
    elif not config["styles"]:
        errors.append("styles section cannot be empty")
    
    # Validate individual styles
    if "styles" in config and isinstance(config["styles"], dict):
        for style_name, style_config in config["styles"].items():
            if not isinstance(style_config, dict):
                errors.append(f"Style '{style_name}' must be a dictionary")
                continue
            
            if "color" not in style_config:
                errors.append(f"Style '{style_name}' missing required 'color' field")
    
    # Check outputs section if present
    if "outputs" in config:
        outputs = config["outputs"]
        if not isinstance(outputs, dict):
            errors.append("outputs section must be a dictionary")
        elif "base_path" not in outputs:
            errors.append("outputs section missing 'base_path' field")
    
    return errors


def get_all_test_config_generators() -> Dict[str, callable]:
    """
    Get dictionary of all available configuration generators.
    
    Returns:
        Dictionary mapping generator names to generator functions
    """
    return {
        # Environment-specific generators
        "development": generate_development_config,
        "staging": generate_staging_config,
        "production": generate_production_config,
        "baseline": generate_baseline_config,
        
        # Invalid configuration generators
        "invalid_missing_version": generate_invalid_config_missing_version,
        "invalid_malformed_styles": generate_invalid_config_malformed_styles,
        "invalid_colors": generate_invalid_config_invalid_colors,
        "invalid_negative_values": generate_invalid_config_negative_values,
        "invalid_wrong_types": generate_invalid_config_wrong_types,
        "invalid_empty_sections": generate_invalid_config_empty_required_sections,
        "invalid_circular_refs": generate_invalid_config_circular_references,
        "invalid_path_traversal": generate_invalid_config_path_traversal,
        
        # Performance generators
        "performance_small": generate_performance_config_small,
        "performance_medium": generate_performance_config_medium,
        "performance_large": generate_performance_config_large,
        
        # Utility generators
        "timestamp": generate_timestamp_config
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Environment-specific generators
    "generate_development_config",
    "generate_staging_config", 
    "generate_production_config",
    "generate_baseline_config",
    
    # Invalid configuration generators
    "generate_invalid_config_missing_version",
    "generate_invalid_config_malformed_styles",
    "generate_invalid_config_invalid_colors",
    "generate_invalid_config_negative_values",
    "generate_invalid_config_wrong_types",
    "generate_invalid_config_empty_required_sections",
    "generate_invalid_config_circular_references",
    "generate_invalid_config_path_traversal",
    
    # Merge scenario generators
    "generate_merged_config_scenarios",
    "generate_precedence_test_cases",
    
    # Security testing
    "generate_yaml_injection_config",
    "generate_path_traversal_configs",
    "generate_malicious_parameter_configs",
    "generate_unicode_exploit_configs",
    "generate_configuration_bomb_scenarios",
    
    # Performance testing
    "generate_performance_config_small",
    "generate_performance_config_medium",
    "generate_performance_config_large",
    "generate_performance_benchmark_configs",
    
    # Cross-platform variations
    "generate_windows_config_variations",
    "generate_unix_config_variations",
    "generate_macos_config_variations",
    "generate_path_separator_configs",
    
    # Utility functions
    "generate_random_config_id",
    "generate_timestamp_config",
    "apply_config_overrides",
    "validate_config_structure",
    "get_all_test_config_generators",
    
    # Type hints
    "ConfigDict",
    "EnvironmentConfig",
    "MergeScenario",
    "ValidationScenario"
]

# Hypothesis strategies (if available)
if HAS_HYPOTHESIS:
    __all__.extend([
        "valid_color_strategy",
        "valid_style_strategy", 
        "valid_outputs_strategy",
        "valid_figregistry_config_strategy",
        "malformed_config_strategy"
    ])