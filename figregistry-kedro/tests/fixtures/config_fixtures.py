"""
Configuration Testing Fixtures for FigRegistry-Kedro Integration

This module provides comprehensive YAML configuration test data for testing the 
FigRegistryConfigBridge component. Fixtures support environment-specific configuration 
override testing, configuration merging validation, Pydantic validation testing, 
and error handling scenarios as required by F-007 and F-007.2 functional requirements.

The fixtures enable comprehensive testing of:
- Base FigRegistry configuration templates for baseline testing (Section 5.2.5)
- Environment-specific configuration overrides (F-007.2)
- Configuration merging with proper precedence rules (F-007)
- Pydantic validation for type safety across both systems (Section 5.2.5)
- Error handling for malformed configurations and schema violations
- Performance requirements validation (<10ms merge time targets)

Key Testing Scenarios:
- Valid configuration merging between Kedro and FigRegistry systems
- Environment-specific override precedence (local, staging, production)
- Schema validation failures and error aggregation
- Configuration caching and performance benchmarking
- Thread-safety testing for concurrent access patterns
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def base_figregistry_config() -> Dict[str, Any]:
    """
    Provide standard FigRegistry YAML configuration for baseline testing per Section 5.2.5.
    
    This fixture contains a comprehensive base configuration that represents a typical
    FigRegistry setup with all required and optional sections properly configured.
    Used as the foundation for testing configuration merging, validation, and 
    environment-specific overrides.
    
    Returns:
        Dict containing complete FigRegistry configuration with all sections
    """
    return {
        "figregistry_version": ">=0.3.0",
        "metadata": {
            "config_version": "1.0.0",
            "created_by": "figregistry-kedro test suite",
            "description": "Base FigRegistry configuration for testing",
            "last_updated": "2024-01-15T10:30:00Z",
            "project_name": "figregistry-kedro-test"
        },
        "styles": {
            "exploratory": {
                "color": "#A8E6CF",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Exploratory Analysis",
                "markersize": 6
            },
            "presentation": {
                "color": "#FFB6C1", 
                "marker": "s",
                "linestyle": "-",
                "linewidth": 2.0,
                "alpha": 0.8,
                "label": "Presentation Ready",
                "markersize": 8
            },
            "publication": {
                "color": "#1A1A1A",
                "marker": "^",
                "linestyle": "-",
                "linewidth": 2.5,
                "alpha": 1.0,
                "label": "Publication Quality",
                "markersize": 10
            },
            "baseline": {
                "color": "#2E86AB",
                "marker": "o",
                "linestyle": "--",
                "linewidth": 2.0,
                "alpha": 0.9,
                "label": "Baseline Condition"
            },
            "treatment": {
                "color": "#F24236",
                "marker": "D",
                "linestyle": "-",
                "linewidth": 2.0,
                "alpha": 0.9,
                "label": "Treatment Condition"
            }
        },
        "palettes": {
            "default": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
            "colorblind_safe": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            "publication": {
                "primary": "#000000",
                "secondary": "#666666", 
                "accent": "#2E86AB",
                "highlight": "#F24236"
            }
        },
        "defaults": {
            "figure": {
                "figsize": [10, 8],
                "dpi": 150,
                "facecolor": "white",
                "edgecolor": "none"
            },
            "line": {
                "color": "#2E86AB",
                "linewidth": 2.0,
                "alpha": 0.8
            },
            "scatter": {
                "s": 50,
                "alpha": 0.7,
                "edgecolors": "black",
                "linewidth": 0.5
            },
            "fallback_style": {
                "color": "#95A5A6",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Unknown Condition"
            }
        },
        "outputs": {
            "base_path": "data/08_reporting",
            "naming": {
                "template": "{name}_{condition}_{ts}",
                "timestamp_format": "%Y%m%d_%H%M%S"
            },
            "formats": {
                "defaults": {
                    "exploratory": ["png"],
                    "presentation": ["png", "pdf"],
                    "publication": ["pdf", "svg"]
                },
                "resolution": {
                    "png": {"dpi": 300},
                    "pdf": {"dpi": 300},
                    "svg": {"dpi": 300}
                }
            },
            "paths": {
                "exploratory": "exploratory",
                "presentation": "presentation",
                "publication": "publication"
            }
        },
        "style_inheritance": {
            "enabled": True,
            "hierarchy": ["specific", "category", "defaults"],
            "merge_strategy": "deep"
        },
        "conditional_rules": {
            "wildcard_patterns": ["*_control", "*_treatment"],
            "partial_matching": True,
            "case_sensitive": False
        },
        "performance": {
            "cache_enabled": True,
            "max_cache_size": 1000,
            "target_merge_time_ms": 10.0
        },
        "validation": {
            "strict_mode": True,
            "required_fields": ["styles", "defaults", "outputs"],
            "schema_version": "1.0"
        }
    }


@pytest.fixture  
def local_override_config() -> Dict[str, Any]:
    """
    Provide environment-specific configuration testing for local development per F-007.2.
    
    This fixture represents Kedro local environment configuration overrides that should
    take precedence over base FigRegistry configurations during merging. Tests the
    precedence rules where Kedro-specific parameters override FigRegistry defaults.
    
    Returns:
        Dict containing local environment configuration overrides
    """
    return {
        "figregistry": {
            "outputs": {
                "base_path": "data/01_raw/debug_figures",
                "naming": {
                    "template": "local_{name}_{ts}"
                },
                "formats": {
                    "defaults": {
                        "exploratory": ["png"]  # Only PNG for local development
                    }
                }
            },
            "defaults": {
                "figure": {
                    "figsize": [8, 6],  # Smaller figures for local development
                    "dpi": 100  # Lower DPI for faster rendering
                },
                "line": {
                    "linewidth": 1.0  # Thinner lines for debugging
                }
            },
            "performance": {
                "cache_enabled": True,
                "target_merge_time_ms": 5.0  # Faster target for local
            },
            "validation": {
                "strict_mode": False  # Relaxed validation for development
            }
        },
        "parameters": {
            "experiment_condition": "local_test",
            "experiment_phase": "development",
            "analysis_stage": "exploratory",
            "model_type": "prototype",
            "plot_settings": {
                "figure_size": [6, 4],
                "dpi": 72
            },
            "execution_config": {
                "output_base_path": "outputs/local",
                "figure_formats": ["png"]
            }
        }
    }


@pytest.fixture
def environment_specific_configs() -> Dict[str, Dict[str, Any]]:
    """
    Provide environment-specific configurations supporting development, staging, and 
    production scenarios per F-007.2.
    
    This fixture provides a complete set of environment configurations that test
    the configuration bridge's ability to handle different deployment environments
    with appropriate overrides for each stage of the development lifecycle.
    
    Returns:
        Dict mapping environment names to their specific configurations
    """
    return {
        "base": {
            "figregistry": {
                "metadata": {
                    "environment": "base",
                    "config_level": "baseline"
                },
                "performance": {
                    "target_merge_time_ms": 10.0
                }
            }
        },
        "local": {
            "figregistry": {
                "metadata": {
                    "environment": "local",
                    "config_level": "development"
                },
                "outputs": {
                    "base_path": "data/01_raw/local_figures",
                    "formats": {
                        "defaults": {
                            "exploratory": ["png"]
                        }
                    }
                },
                "defaults": {
                    "figure": {"dpi": 100},
                    "line": {"linewidth": 1.0}
                },
                "performance": {
                    "target_merge_time_ms": 5.0,
                    "cache_enabled": True
                },
                "validation": {
                    "strict_mode": False
                }
            },
            "parameters": {
                "experiment_condition": "local_dev",
                "debug_mode": True
            }
        },
        "staging": {
            "figregistry": {
                "metadata": {
                    "environment": "staging", 
                    "config_level": "testing"
                },
                "outputs": {
                    "base_path": "data/08_reporting/staging",
                    "formats": {
                        "defaults": {
                            "exploratory": ["png"],
                            "presentation": ["png", "pdf"]
                        }
                    }
                },
                "defaults": {
                    "figure": {"dpi": 200},
                    "line": {"linewidth": 1.5}
                },
                "performance": {
                    "target_merge_time_ms": 15.0,
                    "cache_enabled": True
                },
                "validation": {
                    "strict_mode": True
                }
            },
            "parameters": {
                "experiment_condition": "staging_test",
                "validation_enabled": True
            }
        },
        "production": {
            "figregistry": {
                "metadata": {
                    "environment": "production",
                    "config_level": "release"
                },
                "outputs": {
                    "base_path": "data/08_reporting/production",
                    "formats": {
                        "defaults": {
                            "exploratory": ["png"],
                            "presentation": ["png", "pdf"],
                            "publication": ["pdf", "svg", "eps"]
                        },
                        "resolution": {
                            "png": {"dpi": 300},
                            "pdf": {"dpi": 300},
                            "svg": {"dpi": 300},
                            "eps": {"dpi": 300}
                        }
                    }
                },
                "defaults": {
                    "figure": {"dpi": 300},
                    "line": {"linewidth": 2.0}
                },
                "performance": {
                    "target_merge_time_ms": 25.0,  # Allow more time for production quality
                    "cache_enabled": True,
                    "max_cache_size": 5000
                },
                "validation": {
                    "strict_mode": True
                }
            },
            "parameters": {
                "experiment_condition": "production_run",
                "quality_assurance": True,
                "audit_trail": True
            }
        }
    }


@pytest.fixture
def merged_config_scenarios() -> List[Dict[str, Any]]:
    """
    Provide various configuration merge combinations for comprehensive testing.
    
    This fixture provides test scenarios that validate configuration merging logic
    with different combinations of base configs, environment overrides, and
    direct parameter overrides to ensure precedence rules work correctly.
    
    Returns:
        List of test scenarios, each containing input configs and expected results
    """
    return [
        {
            "name": "base_only_merge",
            "description": "Merge with only base FigRegistry configuration",
            "inputs": {
                "figregistry_config": {
                    "styles": {"test": {"color": "#FF0000"}},
                    "defaults": {"figure": {"figsize": [10, 8]}}
                },
                "kedro_config": {},
                "overrides": {}
            },
            "expected": {
                "styles": {"test": {"color": "#FF0000"}},
                "defaults": {"figure": {"figsize": [10, 8]}},
                "outputs": {"base_path": "data/08_reporting"}  # Default from required sections
            }
        },
        {
            "name": "kedro_override_merge", 
            "description": "Kedro parameters override FigRegistry defaults",
            "inputs": {
                "figregistry_config": {
                    "styles": {"test": {"color": "#FF0000"}},
                    "defaults": {"figure": {"figsize": [10, 8], "dpi": 150}}
                },
                "kedro_config": {
                    "figregistry": {
                        "defaults": {"figure": {"dpi": 300}}  # Override DPI
                    },
                    "parameters": {
                        "experiment_condition": "test_condition"
                    }
                },
                "overrides": {}
            },
            "expected": {
                "styles": {"test": {"color": "#FF0000"}},
                "defaults": {"figure": {"figsize": [10, 8], "dpi": 300}},  # Kedro override
                "condition_parameters": {"experiment_condition": "test_condition"}
            }
        },
        {
            "name": "direct_override_merge",
            "description": "Direct overrides take highest precedence",
            "inputs": {
                "figregistry_config": {
                    "styles": {"test": {"color": "#FF0000"}},
                    "defaults": {"figure": {"figsize": [10, 8]}}
                },
                "kedro_config": {
                    "figregistry": {
                        "defaults": {"figure": {"figsize": [12, 10]}}
                    }
                },
                "overrides": {
                    "defaults": {"figure": {"figsize": [8, 6]}}  # Highest precedence
                }
            },
            "expected": {
                "styles": {"test": {"color": "#FF0000"}},
                "defaults": {"figure": {"figsize": [8, 6]}}  # Direct override wins
            }
        },
        {
            "name": "complex_deep_merge",
            "description": "Complex deep merge with multiple override levels",
            "inputs": {
                "figregistry_config": {
                    "styles": {
                        "baseline": {"color": "#FF0000", "linewidth": 1.0},
                        "treatment": {"color": "#00FF00", "linewidth": 2.0}
                    },
                    "defaults": {
                        "figure": {"figsize": [10, 8], "dpi": 150},
                        "line": {"alpha": 0.8}
                    },
                    "outputs": {
                        "base_path": "original/path",
                        "naming": {"template": "orig_{name}"}
                    }
                },
                "kedro_config": {
                    "figregistry": {
                        "styles": {
                            "baseline": {"linewidth": 1.5},  # Override linewidth only
                            "new_style": {"color": "#0000FF"}  # Add new style
                        },
                        "outputs": {
                            "base_path": "kedro/path"  # Override path
                        }
                    }
                },
                "overrides": {
                    "defaults": {
                        "line": {"alpha": 1.0}  # Override alpha
                    }
                }
            },
            "expected": {
                "styles": {
                    "baseline": {"color": "#FF0000", "linewidth": 1.5},  # Merged
                    "treatment": {"color": "#00FF00", "linewidth": 2.0},  # Unchanged
                    "new_style": {"color": "#0000FF"}  # Added from Kedro
                },
                "defaults": {
                    "figure": {"figsize": [10, 8], "dpi": 150},  # Unchanged
                    "line": {"alpha": 1.0}  # Override wins
                },
                "outputs": {
                    "base_path": "kedro/path",  # Kedro override
                    "naming": {"template": "orig_{name}"}  # Unchanged
                }
            }
        }
    ]


@pytest.fixture
def config_merge_test_cases() -> List[Dict[str, Any]]:
    """
    Provide precedence rule validation scenarios per Section 5.2.5.
    
    This fixture tests the specific precedence rules defined in the configuration
    bridge: Override parameters > Kedro config > FigRegistry config > Defaults.
    Each test case validates that the precedence hierarchy is correctly implemented.
    
    Returns:
        List of test cases validating configuration precedence rules
    """
    return [
        {
            "test_id": "precedence_override_wins",
            "description": "Direct override parameters take highest precedence",
            "precedence_level": "override",
            "base_config": {"defaults": {"figure": {"dpi": 100}}},
            "kedro_config": {"figregistry": {"defaults": {"figure": {"dpi": 200}}}},
            "override_params": {"defaults": {"figure": {"dpi": 300}}},
            "expected_value": 300,
            "field_path": ["defaults", "figure", "dpi"]
        },
        {
            "test_id": "precedence_kedro_over_base",
            "description": "Kedro config overrides base FigRegistry config",
            "precedence_level": "kedro",
            "base_config": {"outputs": {"base_path": "base/path"}},
            "kedro_config": {"figregistry": {"outputs": {"base_path": "kedro/path"}}},
            "override_params": {},
            "expected_value": "kedro/path",
            "field_path": ["outputs", "base_path"]
        },
        {
            "test_id": "precedence_base_when_no_override",
            "description": "Base config used when no overrides present",
            "precedence_level": "base",
            "base_config": {"styles": {"test": {"color": "#FF0000"}}},
            "kedro_config": {},
            "override_params": {},
            "expected_value": "#FF0000", 
            "field_path": ["styles", "test", "color"]
        },
        {
            "test_id": "precedence_defaults_when_missing",
            "description": "Required defaults applied when sections missing",
            "precedence_level": "defaults",
            "base_config": {"styles": {}},  # Missing required sections
            "kedro_config": {},
            "override_params": {},
            "expected_value": "data/08_reporting",  # Default base_path
            "field_path": ["outputs", "base_path"]
        },
        {
            "test_id": "precedence_partial_override",
            "description": "Partial overrides preserve other fields",
            "precedence_level": "partial",
            "base_config": {
                "defaults": {
                    "figure": {"figsize": [10, 8], "dpi": 150, "facecolor": "white"}
                }
            },
            "kedro_config": {
                "figregistry": {
                    "defaults": {"figure": {"dpi": 300}}  # Only override DPI
                }
            },
            "override_params": {},
            "expected_values": {
                "figsize": [10, 8],  # Preserved from base
                "dpi": 300,  # Overridden by Kedro
                "facecolor": "white"  # Preserved from base
            },
            "field_path": ["defaults", "figure"]
        },
        {
            "test_id": "precedence_parameter_injection",
            "description": "Kedro parameters correctly injected into condition context",
            "precedence_level": "parameter_injection",
            "base_config": {"styles": {"test": {"color": "#FF0000"}}},
            "kedro_config": {
                "parameters": {
                    "experiment_condition": "baseline",
                    "experiment_phase": "training",
                    "model_type": "neural_network"
                }
            },
            "override_params": {},
            "expected_values": {
                "experiment_condition": "baseline",
                "experiment_phase": "training", 
                "model_type": "neural_network"
            },
            "field_path": ["condition_parameters"]
        }
    ]


@pytest.fixture
def invalid_config_fixtures() -> Dict[str, Dict[str, Any]]:
    """
    Provide error handling validation including malformed YAML and schema violations.
    
    This fixture provides various invalid configuration scenarios to test the
    configuration bridge's error handling, validation, and error aggregation
    capabilities. Tests both YAML parsing errors and Pydantic schema violations.
    
    Returns:
        Dict mapping error types to invalid configuration examples
    """
    return {
        "malformed_yaml": {
            "description": "YAML syntax errors that should be caught during parsing",
            "configs": [
                {
                    "name": "invalid_indentation",
                    "yaml_content": """
styles:
baseline:  # Missing proper indentation
  color: "#FF0000"
    """,
                    "expected_error": "yaml.scanner.ScannerError"
                },
                {
                    "name": "invalid_syntax",
                    "yaml_content": """
styles:
  baseline: {
    color: "#FF0000"
    # Missing closing brace
    """,
                    "expected_error": "yaml.scanner.ScannerError"
                },
                {
                    "name": "duplicate_keys",
                    "yaml_content": """
styles:
  baseline:
    color: "#FF0000"
  baseline:  # Duplicate key
    color: "#00FF00"
    """,
                    "expected_error": "yaml.constructor.ConstructorError"
                }
            ]
        },
        "schema_violations": {
            "description": "Configurations that violate Pydantic schema validation",
            "configs": [
                {
                    "name": "invalid_version_constraint",
                    "config": {
                        "figregistry_version": "invalid_version_format",
                        "styles": {"test": {"color": "#FF0000"}}
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "figregistry_version"
                },
                {
                    "name": "missing_required_color",
                    "config": {
                        "styles": {
                            "test": {"linewidth": 2.0}  # Missing required color
                        }
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "styles.test.color"
                },
                {
                    "name": "invalid_marker_type",
                    "config": {
                        "styles": {
                            "test": {
                                "color": "#FF0000",
                                "marker": 123  # Invalid marker type (should be string)
                            }
                        }
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "styles.test.marker"
                },
                {
                    "name": "missing_output_base_path",
                    "config": {
                        "outputs": {
                            "naming": {"template": "{name}_{ts}"}
                            # Missing required base_path
                        }
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "outputs.base_path"
                }
            ]
        },
        "merge_conflicts": {
            "description": "Configuration merge scenarios that should fail validation",
            "configs": [
                {
                    "name": "conflicting_types",
                    "base_config": {
                        "defaults": {"figure": {"figsize": [10, 8]}}
                    },
                    "override_config": {
                        "defaults": {"figure": {"figsize": "invalid_string"}}  # Type conflict
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "defaults.figure.figsize"
                },
                {
                    "name": "invalid_nested_structure",
                    "base_config": {
                        "styles": {"test": {"color": "#FF0000"}}
                    },
                    "override_config": {
                        "styles": "invalid_string_instead_of_dict"  # Structure conflict
                    },
                    "expected_error": "ConfigMergeError"
                }
            ]
        },
        "performance_violations": {
            "description": "Configurations that might cause performance issues",
            "configs": [
                {
                    "name": "excessive_cache_size",
                    "config": {
                        "performance": {
                            "max_cache_size": -1  # Invalid negative cache size
                        }
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "performance.max_cache_size"
                },
                {
                    "name": "invalid_target_time",
                    "config": {
                        "performance": {
                            "target_merge_time_ms": "not_a_number"  # Invalid type
                        }
                    },
                    "expected_error": "ConfigValidationError",
                    "expected_field": "performance.target_merge_time_ms"
                }
            ]
        }
    }


@pytest.fixture
def pydantic_validation_fixtures() -> Dict[str, Any]:
    """
    Provide type safety testing across configuration bridge operations.
    
    This fixture provides specific test cases for Pydantic validation features
    including type coercion, field validation, and custom validators defined
    in the FigRegistryConfigSchema class.
    
    Returns:
        Dict containing validation test scenarios and expected behaviors
    """
    return {
        "valid_type_coercion": {
            "description": "Valid type coercion scenarios that should pass",
            "test_cases": [
                {
                    "name": "string_to_number_coercion",
                    "input": {"defaults": {"line": {"linewidth": "2.0"}}},
                    "expected": {"defaults": {"line": {"linewidth": 2.0}}},
                    "field": "defaults.line.linewidth"
                },
                {
                    "name": "list_to_tuple_coercion",
                    "input": {"defaults": {"figure": {"figsize": (10, 8)}}},
                    "expected": {"defaults": {"figure": {"figsize": [10, 8]}}},
                    "field": "defaults.figure.figsize"
                }
            ]
        },
        "custom_validator_tests": {
            "description": "Test custom validators in FigRegistryConfigSchema",
            "test_cases": [
                {
                    "name": "styles_color_validation",
                    "input": {
                        "styles": {
                            "valid_style": {"color": "#FF0000", "linewidth": 2.0},
                            "missing_color": {"linewidth": 2.0}  # Should trigger warning
                        }
                    },
                    "expected_warnings": ["Style 'missing_color' missing required 'color' field"],
                    "should_pass": True
                },
                {
                    "name": "outputs_base_path_validation",
                    "input": {
                        "outputs": {
                            "naming": {"template": "{name}_{ts}"}
                            # Missing base_path - should fail validation
                        }
                    },
                    "expected_error": "Output configuration must include 'base_path' field",
                    "should_pass": False
                },
                {
                    "name": "version_constraint_validation",
                    "input": {"figregistry_version": "invalid_no_operator"},
                    "expected_error": "Invalid version constraint format: invalid_no_operator",
                    "should_pass": False
                }
            ]
        },
        "extra_fields_handling": {
            "description": "Test handling of extra fields (allow vs forbid)",
            "test_cases": [
                {
                    "name": "extra_fields_allowed",
                    "input": {
                        "styles": {"test": {"color": "#FF0000"}},
                        "custom_extension": {"user_data": "allowed"}  # Extra field
                    },
                    "expected_behavior": "allowed",
                    "should_pass": True
                },
                {
                    "name": "extra_style_properties",
                    "input": {
                        "styles": {
                            "test": {
                                "color": "#FF0000",
                                "custom_property": "value"  # Extra property in style
                            }
                        }
                    },
                    "expected_behavior": "allowed",
                    "should_pass": True
                }
            ]
        },
        "type_safety_validation": {
            "description": "Strict type validation for critical fields",
            "test_cases": [
                {
                    "name": "figsize_type_validation",
                    "invalid_inputs": [
                        {"defaults": {"figure": {"figsize": "10x8"}}},  # String instead of list
                        {"defaults": {"figure": {"figsize": [10]}}},    # Wrong length
                        {"defaults": {"figure": {"figsize": ["10", "8"]}}},  # String elements
                    ],
                    "field": "defaults.figure.figsize",
                    "expected_error_type": "ValidationError"
                },
                {
                    "name": "color_validation",
                    "valid_inputs": [
                        {"styles": {"test": {"color": "#FF0000"}}},     # Hex color
                        {"styles": {"test": {"color": "red"}}},         # Named color
                        {"styles": {"test": {"color": "rgb(255,0,0)"}}}, # RGB color
                    ],
                    "invalid_inputs": [
                        {"styles": {"test": {"color": 123}}},           # Number
                        {"styles": {"test": {"color": "#GGGGGG"}}},     # Invalid hex
                    ]
                },
                {
                    "name": "boolean_validation",
                    "valid_inputs": [
                        {"performance": {"cache_enabled": True}},
                        {"performance": {"cache_enabled": False}},
                        {"performance": {"cache_enabled": "true"}},     # String coercion
                    ],
                    "invalid_inputs": [
                        {"performance": {"cache_enabled": "maybe"}},    # Invalid string
                        {"performance": {"cache_enabled": 123}},        # Number
                    ]
                }
            ]
        },
        "nested_validation": {
            "description": "Validation of deeply nested configuration structures",
            "test_cases": [
                {
                    "name": "deep_nested_structure",
                    "input": {
                        "outputs": {
                            "formats": {
                                "resolution": {
                                    "png": {"dpi": 300},
                                    "pdf": {"dpi": 300}
                                }
                            }
                        }
                    },
                    "expected_validation": "passes_deep_validation",
                    "validation_depth": 4
                },
                {
                    "name": "invalid_deep_nesting",
                    "input": {
                        "outputs": {
                            "formats": {
                                "resolution": "invalid_string_should_be_dict"
                            }
                        }
                    },
                    "expected_error": "ValidationError",
                    "field": "outputs.formats.resolution"
                }
            ]
        }
    }


@pytest.fixture
def mock_kedro_config_loader():
    """
    Provide mock Kedro ConfigLoader for testing configuration bridge operations.
    
    This fixture creates a mock ConfigLoader that simulates Kedro's configuration
    loading behavior with predefined configurations for different environments.
    Used for testing the configuration bridge without requiring a full Kedro setup.
    
    Returns:
        Mock ConfigLoader with predefined get() method behavior
    """
    mock_loader = MagicMock()
    
    # Define mock responses for different config sections and environments
    mock_responses = {
        ("parameters", "base"): {
            "experiment_condition": "baseline",
            "model_type": "linear_regression"
        },
        ("parameters", "local"): {
            "experiment_condition": "local_test",
            "debug_mode": True,
            "plot_settings": {"figure_size": [6, 4], "dpi": 72}
        },
        ("figregistry", "base"): {
            "styles": {"mock": {"color": "#FF0000"}},
            "defaults": {"figure": {"dpi": 150}}
        },
        ("figregistry", "local"): {
            "outputs": {"base_path": "local/path"},
            "performance": {"target_merge_time_ms": 5.0}
        },
        ("catalog", "base"): {
            "figure_output": {
                "type": "figregistry_kedro.datasets.FigureDataSet",
                "purpose": "exploratory"
            }
        },
        ("logging", "base"): {
            "version": 1,
            "handlers": {"console": {"class": "logging.StreamHandler"}}
        }
    }
    
    def mock_get(section: str, environment: str = "base"):
        """Mock ConfigLoader.get() method."""
        key = (section, environment)
        if key in mock_responses:
            return copy.deepcopy(mock_responses[key])
        return {}
    
    mock_loader.get.side_effect = mock_get
    return mock_loader


@pytest.fixture
def performance_test_configs() -> Dict[str, Any]:
    """
    Provide configuration scenarios for performance benchmarking.
    
    This fixture provides configurations of varying complexity to test
    the <10ms merge time requirement and validate performance characteristics
    under different load scenarios.
    
    Returns:
        Dict containing performance test configurations
    """
    return {
        "minimal_config": {
            "description": "Minimal configuration for baseline performance",
            "config": {
                "styles": {"test": {"color": "#FF0000"}},
                "defaults": {"figure": {"figsize": [10, 8]}},
                "outputs": {"base_path": "data"}
            },
            "expected_merge_time_ms": 2.0
        },
        "moderate_config": {
            "description": "Moderate complexity configuration",
            "config": {
                "styles": {f"style_{i}": {"color": f"#{i:06x}"} for i in range(10)},
                "defaults": {
                    "figure": {"figsize": [10, 8], "dpi": 150},
                    "line": {"linewidth": 2.0, "alpha": 0.8}
                },
                "outputs": {
                    "base_path": "data",
                    "formats": {"defaults": {"exploratory": ["png", "pdf"]}}
                }
            },
            "expected_merge_time_ms": 5.0
        },
        "complex_config": {
            "description": "Complex configuration with many nested sections",
            "config": {
                "styles": {f"style_{i}": {
                    "color": f"#{i:06x}",
                    "marker": "o",
                    "linewidth": i % 3 + 1,
                    "alpha": 0.1 + (i % 10) * 0.1
                } for i in range(100)},
                "palettes": {f"palette_{i}": [f"#{j:06x}" for j in range(i, i+5)] for i in range(0, 50, 5)},
                "defaults": {
                    "figure": {"figsize": [10, 8], "dpi": 150, "facecolor": "white"},
                    "line": {"linewidth": 2.0, "alpha": 0.8, "color": "#000000"},
                    "scatter": {"s": 50, "alpha": 0.7, "edgecolors": "black"}
                },
                "outputs": {
                    "base_path": "data/complex",
                    "formats": {
                        "defaults": {
                            "exploratory": ["png"],
                            "presentation": ["png", "pdf"],
                            "publication": ["pdf", "svg", "eps"]
                        },
                        "resolution": {fmt: {"dpi": 300} for fmt in ["png", "pdf", "svg", "eps"]}
                    },
                    "paths": {purpose: f"output/{purpose}" for purpose in ["exploratory", "presentation", "publication"]}
                }
            },
            "expected_merge_time_ms": 8.0
        }
    }


@pytest.fixture
def thread_safety_test_scenarios() -> List[Dict[str, Any]]:
    """
    Provide test scenarios for validating thread-safe configuration operations.
    
    This fixture provides scenarios for testing concurrent access to the
    configuration bridge, validating cache safety, and ensuring proper
    handling of simultaneous configuration operations.
    
    Returns:
        List of thread safety test scenarios
    """
    return [
        {
            "name": "concurrent_config_loading",
            "description": "Multiple threads loading configurations simultaneously",
            "thread_count": 5,
            "operations_per_thread": 10,
            "config_variations": [
                {"environment": "local", "overrides": {"performance": {"cache_enabled": True}}},
                {"environment": "staging", "overrides": {"validation": {"strict_mode": True}}},
                {"environment": "production", "overrides": {"outputs": {"base_path": "prod"}}}
            ],
            "expected_behavior": "all_operations_succeed"
        },
        {
            "name": "cache_contention_test",
            "description": "Test cache behavior under concurrent access",
            "thread_count": 10,
            "operations_per_thread": 20,
            "cache_operations": ["get", "set", "clear"],
            "expected_behavior": "cache_consistency_maintained"
        },
        {
            "name": "mixed_read_write_operations",
            "description": "Mix of read and write operations to configuration",
            "thread_count": 8,
            "read_operations": 15,
            "write_operations": 5,
            "expected_behavior": "data_integrity_preserved"
        }
    ]


# Helper functions for test data generation and validation

def generate_large_config(num_styles: int = 1000) -> Dict[str, Any]:
    """
    Generate large configuration for stress testing.
    
    Args:
        num_styles: Number of styles to generate
        
    Returns:
        Large configuration dictionary for performance testing
    """
    return {
        "styles": {
            f"style_{i}": {
                "color": f"#{i % 256:02x}{(i*2) % 256:02x}{(i*3) % 256:02x}",
                "marker": ["o", "s", "^", "D", "v"][i % 5],
                "linewidth": 1.0 + (i % 10) * 0.2,
                "alpha": 0.1 + (i % 9) * 0.1,
                "label": f"Condition {i}"
            }
            for i in range(num_styles)
        },
        "defaults": {
            "figure": {"figsize": [10, 8], "dpi": 150},
            "line": {"linewidth": 2.0, "alpha": 0.8}
        },
        "outputs": {
            "base_path": "data/stress_test",
            "naming": {"template": "{name}_{condition}_{ts}"}
        }
    }


def validate_config_structure(config: Dict[str, Any], required_sections: List[str]) -> bool:
    """
    Validate that configuration contains all required sections.
    
    Args:
        config: Configuration dictionary to validate
        required_sections: List of required section names
        
    Returns:
        True if all required sections are present
    """
    return all(section in config for section in required_sections)


def extract_nested_value(config: Dict[str, Any], field_path: List[str]) -> Any:
    """
    Extract nested value from configuration using field path.
    
    Args:
        config: Configuration dictionary
        field_path: List of nested keys to traverse
        
    Returns:
        Value at the specified nested path, or None if not found
    """
    current = config
    for key in field_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


# Export all fixtures for use in tests
__all__ = [
    "base_figregistry_config",
    "local_override_config", 
    "environment_specific_configs",
    "merged_config_scenarios",
    "config_merge_test_cases",
    "invalid_config_fixtures",
    "pydantic_validation_fixtures",
    "mock_kedro_config_loader",
    "performance_test_configs",
    "thread_safety_test_scenarios",
    "generate_large_config",
    "validate_config_structure",
    "extract_nested_value"
]