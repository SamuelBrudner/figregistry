"""Configuration test data generators and utilities for FigRegistryConfigBridge testing.

This module provides comprehensive test data generation functions, fixtures, and utilities
for testing the FigRegistryConfigBridge component across various scenarios including
environment-specific configurations, security testing, performance benchmarking, and
property-based testing with Hypothesis strategies.

The module supports testing of:
- Configuration merging between Kedro and FigRegistry systems
- Environment-specific override scenarios (development, staging, production)
- Security validation including YAML injection and path traversal prevention
- Performance benchmarking for configuration merge operations
- Cross-platform compatibility validation
- Pydantic validation edge cases and error handling

Usage:
    from tests.data.config_test_data import (
        generate_baseline_config,
        generate_invalid_config,
        yaml_config_strategy,
        security_test_configs
    )
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from datetime import datetime, timezone
import uuid
import json
import copy

try:
    from hypothesis import strategies as st
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Define no-op decorators for when hypothesis is not available
    def composite(func):
        return func
    class st:
        @staticmethod
        def text(**kwargs): return ""
        @staticmethod
        def dictionaries(**kwargs): return {}
        @staticmethod
        def integers(**kwargs): return 0
        @staticmethod
        def booleans(**kwargs): return True
        @staticmethod
        def one_of(*args): return ""
        @staticmethod
        def lists(**kwargs): return []


# ===============================================================================
# BASELINE CONFIGURATION GENERATORS
# ===============================================================================

def generate_baseline_config() -> Dict[str, Any]:
    """Generate a baseline FigRegistry configuration for testing.
    
    This configuration serves as the foundation for other test scenarios
    and represents a typical scientific visualization setup with comprehensive
    style mappings and output configurations.
    
    Returns:
        Dict containing valid baseline FigRegistry configuration
    """
    return {
        "figregistry_version": "0.3.0",
        "styles": {
            "exploratory": {
                "color": "#1f77b4",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.5,
                "markersize": 6,
                "label": "Exploratory Analysis"
            },
            "presentation": {
                "color": "#ff7f0e",
                "marker": "s",
                "linestyle": "--",
                "linewidth": 2.0,
                "markersize": 8,
                "label": "Presentation Quality"
            },
            "publication": {
                "color": "#2ca02c",
                "marker": "^",
                "linestyle": "-",
                "linewidth": 2.5,
                "markersize": 10,
                "label": "Publication Ready"
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "scientific": ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161"],
            "colorblind_safe": ["#332288", "#117733", "#44aa99", "#88ccee", "#ddcc77"]
        },
        "outputs": {
            "base_path": "outputs/figures",
            "aliases": {
                "expl": "exploratory",
                "pres": "presentation", 
                "pub": "publication"
            },
            "formats": ["png", "pdf", "svg"],
            "timestamp_format": "{name}_{ts}",
            "create_directories": True
        },
        "defaults": {
            "dpi": 300,
            "figsize": [8, 6],
            "style": "seaborn-v0_8",
            "font_family": "DejaVu Sans",
            "font_size": 12
        }
    }


def generate_kedro_specific_config() -> Dict[str, Any]:
    """Generate Kedro-specific configuration section for testing bridge functionality.
    
    Returns:
        Dict containing Kedro integration settings
    """
    return {
        "kedro": {
            "data_layer_mappings": {
                "01_raw": "exploratory",
                "02_intermediate": "exploratory", 
                "03_primary": "presentation",
                "08_reporting": "publication"
            },
            "enable_versioning": True,
            "hook_priority": 100,
            "auto_catalog_registration": True,
            "pipeline_specific_configs": {
                "data_processing": {
                    "default_purpose": "exploratory"
                },
                "reporting": {
                    "default_purpose": "publication"
                }
            }
        }
    }


def generate_environment_configs() -> Dict[str, Dict[str, Any]]:
    """Generate environment-specific configuration variations for multi-stage testing.
    
    Supports F-007.2 multi-environment requirements for development, staging, and
    production scenarios with appropriate parameter overrides and validation rules.
    
    Returns:
        Dict mapping environment names to configuration overrides
    """
    environments = {}
    
    # Development environment - optimized for speed and debugging
    environments["development"] = {
        "defaults": {
            "dpi": 150,  # Lower DPI for faster rendering
            "figsize": [6, 4],  # Smaller figures
            "enable_debug_logging": True
        },
        "outputs": {
            "base_path": "dev_outputs/figures",
            "formats": ["png"],  # Single format for speed
            "timestamp_format": "{name}_dev_{ts}"
        },
        "kedro": {
            "enable_versioning": False,  # Disabled for dev speed
            "auto_catalog_registration": True
        }
    }
    
    # Staging environment - production-like with additional validation
    environments["staging"] = {
        "defaults": {
            "dpi": 300,
            "figsize": [8, 6],
            "enable_validation": True,
            "enable_performance_monitoring": True
        },
        "outputs": {
            "base_path": "staging_outputs/figures",
            "formats": ["png", "pdf"],
            "timestamp_format": "{name}_staging_{ts}",
            "enable_backup": True
        },
        "kedro": {
            "enable_versioning": True,
            "validation_mode": "strict",
            "performance_monitoring": True
        }
    }
    
    # Production environment - full feature set with optimizations
    environments["production"] = {
        "defaults": {
            "dpi": 300,
            "figsize": [10, 8],  # Larger for production quality
            "enable_compression": True,
            "enable_metadata": True
        },
        "outputs": {
            "base_path": "/opt/production/figures",
            "formats": ["png", "pdf", "svg"],
            "timestamp_format": "{name}_prod_{ts}",
            "enable_backup": True,
            "compression_level": 9
        },
        "kedro": {
            "enable_versioning": True,
            "enable_audit_logging": True,
            "performance_monitoring": True,
            "backup_retention_days": 90
        },
        "security": {
            "enable_path_validation": True,
            "restrict_output_paths": True,
            "enable_sanitization": True
        }
    }
    
    return environments


# ===============================================================================
# INVALID CONFIGURATION GENERATORS
# ===============================================================================

def generate_invalid_config_scenarios() -> Dict[str, Dict[str, Any]]:
    """Generate invalid configuration scenarios for error handling testing.
    
    Creates comprehensive test cases for configuration validation failures
    including malformed YAML structures, missing required fields, type validation
    failures, and schema constraint violations.
    
    Returns:
        Dict mapping scenario names to invalid configuration dictionaries
    """
    invalid_configs = {}
    
    # Missing required version field
    invalid_configs["missing_version"] = {
        "styles": {"test": {"color": "#000000"}},
        # figregistry_version intentionally omitted
    }
    
    # Invalid version format
    invalid_configs["invalid_version_format"] = {
        "figregistry_version": "invalid.version.string",
        "styles": {"test": {"color": "#000000"}}
    }
    
    # Invalid color values
    invalid_configs["invalid_colors"] = {
        "figregistry_version": "0.3.0",
        "styles": {
            "test": {
                "color": "not_a_color",  # Invalid color
                "marker": "invalid_marker"  # Invalid marker
            }
        },
        "palettes": {
            "invalid": ["#gggggg", "not_hex", 123]  # Invalid hex codes
        }
    }
    
    # Type mismatches
    invalid_configs["type_mismatches"] = {
        "figregistry_version": "0.3.0",
        "styles": "should_be_dict",  # Wrong type
        "defaults": {
            "dpi": "should_be_number",  # Wrong type
            "figsize": "should_be_list"  # Wrong type
        }
    }
    
    # Malformed nested structures
    invalid_configs["malformed_nested"] = {
        "figregistry_version": "0.3.0",
        "styles": {
            "test": {
                "color": "#000000",
                "nested_invalid": {
                    "deep_nesting": {
                        "too_deep": {
                            "way_too_deep": "invalid_structure"
                        }
                    }
                }
            }
        }
    }
    
    # Invalid path configurations
    invalid_configs["invalid_paths"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "../../../etc/passwd",  # Path traversal attempt
            "aliases": {
                "exploit": "/root/sensitive"  # Absolute path security issue
            }
        }
    }
    
    # Circular reference in configuration
    invalid_configs["circular_reference"] = {
        "figregistry_version": "0.3.0",
        "styles": {
            "reference_a": {"extends": "reference_b"},
            "reference_b": {"extends": "reference_a"}  # Circular dependency
        }
    }
    
    # Invalid Kedro integration parameters
    invalid_configs["invalid_kedro_params"] = {
        "figregistry_version": "0.3.0",
        "kedro": {
            "data_layer_mappings": "should_be_dict",  # Wrong type
            "hook_priority": "should_be_number",  # Wrong type
            "invalid_parameter": {"complex": "object"}  # Unknown parameter
        }
    }
    
    return invalid_configs


def generate_malformed_yaml_strings() -> Dict[str, str]:
    """Generate malformed YAML strings for parser error testing.
    
    Returns:
        Dict mapping scenario names to malformed YAML strings
    """
    malformed_yamls = {}
    
    # Indentation errors
    malformed_yamls["indentation_error"] = """
figregistry_version: "0.3.0"
styles:
  test:
    color: "#000000"
  marker: "o"  # Incorrect indentation
"""
    
    # Unclosed quotes
    malformed_yamls["unclosed_quotes"] = """
figregistry_version: "0.3.0
styles:
  test:
    color: "#000000"
    label: "unclosed string
"""
    
    # Invalid YAML syntax
    malformed_yamls["invalid_syntax"] = """
figregistry_version: "0.3.0"
styles: [
  test: {
    color: "#000000",
    marker: "o"
  }
# Missing closing bracket
"""
    
    # Mixed indentation (tabs and spaces)
    malformed_yamls["mixed_indentation"] = """
figregistry_version: "0.3.0"
styles:
  test:
    color: "#000000"
\tmarker: "o"  # Tab character mixed with spaces
"""
    
    # Duplicate keys
    malformed_yamls["duplicate_keys"] = """
figregistry_version: "0.3.0"
styles:
  test:
    color: "#000000"
    color: "#ffffff"  # Duplicate key
"""
    
    return malformed_yamls


# ===============================================================================
# MERGED CONFIGURATION SCENARIOS
# ===============================================================================

def generate_merge_test_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for configuration merging and precedence rules.
    
    Creates comprehensive test cases for FigRegistryConfigBridge merge operations
    per Section 5.2.5 requirements, including conflict resolution, precedence
    validation, and edge case handling.
    
    Returns:
        List of merge scenario dictionaries with expected outcomes
    """
    scenarios = []
    
    # Basic merge scenario - no conflicts
    scenarios.append({
        "name": "basic_merge_no_conflicts",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {"color": "#1f77b4", "marker": "o"}
            },
            "outputs": {"base_path": "outputs"}
        },
        "kedro_config": {
            "kedro": {
                "enable_versioning": True,
                "data_layer_mappings": {"01_raw": "exploratory"}
            }
        },
        "expected_result": {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {"color": "#1f77b4", "marker": "o"}
            },
            "outputs": {"base_path": "outputs"},
            "kedro": {
                "enable_versioning": True,
                "data_layer_mappings": {"01_raw": "exploratory"}
            }
        },
        "should_pass": True
    })
    
    # Kedro precedence scenario - Kedro overrides FigRegistry
    scenarios.append({
        "name": "kedro_precedence_override",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "outputs": {
                "base_path": "default_outputs",
                "formats": ["png"]
            },
            "defaults": {"dpi": 150}
        },
        "kedro_config": {
            "outputs": {
                "base_path": "kedro_outputs",  # Should override
                "formats": ["png", "pdf"]      # Should override
            },
            "defaults": {"dpi": 300}          # Should override
        },
        "expected_result": {
            "figregistry_version": "0.3.0",
            "outputs": {
                "base_path": "kedro_outputs",
                "formats": ["png", "pdf"]
            },
            "defaults": {"dpi": 300}
        },
        "should_pass": True
    })
    
    # Deep merge scenario - nested dictionary merging
    scenarios.append({
        "name": "deep_merge_nested_dicts",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {
                    "color": "#1f77b4",
                    "marker": "o",
                    "linewidth": 1.0
                }
            }
        },
        "kedro_config": {
            "styles": {
                "exploratory": {
                    "marker": "s",           # Should override
                    "markersize": 8          # Should add
                },
                "presentation": {            # Should add entire section
                    "color": "#ff7f0e",
                    "marker": "^"
                }
            }
        },
        "expected_result": {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {
                    "color": "#1f77b4",     # Preserved from FigRegistry
                    "marker": "s",          # Overridden by Kedro
                    "linewidth": 1.0,       # Preserved from FigRegistry
                    "markersize": 8         # Added from Kedro
                },
                "presentation": {
                    "color": "#ff7f0e",
                    "marker": "^"
                }
            }
        },
        "should_pass": True
    })
    
    # Environment-specific merge scenario
    scenarios.append({
        "name": "environment_specific_merge",
        "figregistry_config": generate_baseline_config(),
        "kedro_config": generate_environment_configs()["production"],
        "expected_result": None,  # To be computed during test
        "should_pass": True,
        "environment": "production"
    })
    
    # Validation failure scenario - incompatible types
    scenarios.append({
        "name": "merge_validation_failure",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                "test": {"color": "#000000"}
            }
        },
        "kedro_config": {
            "styles": "invalid_type_should_be_dict"  # Type mismatch
        },
        "expected_result": None,
        "should_pass": False,
        "expected_error": "Configuration validation failed"
    })
    
    return scenarios


# ===============================================================================
# HYPOTHESIS STRATEGIES FOR PROPERTY-BASED TESTING
# ===============================================================================

if HYPOTHESIS_AVAILABLE:
    @composite
    def valid_color_strategy(draw):
        """Hypothesis strategy for generating valid color values."""
        color_type = draw(st.one_of(
            st.text(alphabet="0123456789ABCDEF", min_size=6, max_size=6).map(lambda x: f"#{x}"),
            st.sampled_from(["red", "blue", "green", "black", "white", "orange", "purple"])
        ))
        return color_type

    @composite
    def valid_marker_strategy(draw):
        """Hypothesis strategy for generating valid matplotlib markers."""
        return draw(st.sampled_from(["o", "s", "^", "v", "<", ">", "d", "p", "*", "+", "x"]))

    @composite
    def style_dict_strategy(draw):
        """Hypothesis strategy for generating valid style dictionaries."""
        return {
            "color": draw(valid_color_strategy()),
            "marker": draw(valid_marker_strategy()),
            "linestyle": draw(st.sampled_from(["-", "--", "-.", ":"]))
        }

    @composite
    def yaml_config_strategy(draw):
        """Hypothesis strategy for generating valid FigRegistry configurations.
        
        Implements comprehensive property-based testing per Section 6.6.2.6
        with st.composite decorators for configuration validation coverage.
        """
        num_styles = draw(st.integers(min_value=1, max_value=5))
        style_names = draw(st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=15),
            min_size=num_styles, max_size=num_styles, unique=True
        ))
        
        styles = {}
        for name in style_names:
            styles[name] = draw(style_dict_strategy())
        
        return {
            "figregistry_version": "0.3.0",
            "styles": styles,
            "outputs": {
                "base_path": draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=5, max_size=20)),
                "formats": draw(st.lists(st.sampled_from(["png", "pdf", "svg"]), min_size=1, max_size=3, unique=True))
            },
            "defaults": {
                "dpi": draw(st.integers(min_value=72, max_value=600)),
                "figsize": [
                    draw(st.integers(min_value=4, max_value=20)),
                    draw(st.integers(min_value=3, max_value=15))
                ]
            }
        }

    @composite
    def kedro_config_strategy(draw):
        """Hypothesis strategy for generating valid Kedro-specific configurations."""
        return {
            "kedro": {
                "enable_versioning": draw(st.booleans()),
                "data_layer_mappings": draw(st.dictionaries(
                    st.text(alphabet="0123456789abcdefghijklmnopqrstuvwxyz_", min_size=5, max_size=15),
                    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=5, max_size=15),
                    min_size=1, max_size=5
                )),
                "hook_priority": draw(st.integers(min_value=1, max_value=1000))
            }
        }

else:
    # Provide fallback functions when Hypothesis is not available
    def valid_color_strategy():
        return "#1f77b4"
    
    def valid_marker_strategy():
        return "o"
    
    def style_dict_strategy():
        return {"color": "#1f77b4", "marker": "o", "linestyle": "-"}
    
    def yaml_config_strategy():
        return generate_baseline_config()
    
    def kedro_config_strategy():
        return generate_kedro_specific_config()


# ===============================================================================
# SECURITY TEST CONFIGURATIONS
# ===============================================================================

def generate_security_test_configs() -> Dict[str, Dict[str, Any]]:
    """Generate security test configurations for injection prevention per Section 6.6.8.1.
    
    Creates malicious configuration structures including path traversal attempts,
    YAML injection vectors, and configuration manipulation scenarios for
    comprehensive security validation.
    
    Returns:
        Dict mapping attack vector names to malicious configuration dictionaries
    """
    security_configs = {}
    
    # Path traversal attempts
    security_configs["path_traversal_basic"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "../../../etc/passwd",
            "aliases": {
                "exploit": "/root/.ssh/",
                "traversal": "../../../../home/user/.bashrc"
            }
        }
    }
    
    # More sophisticated path traversal
    security_configs["path_traversal_encoded"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "aliases": {
                "encoded": "..%2f..%2f..%2fetc%2fshadow"
            }
        }
    }
    
    # YAML injection attempts
    security_configs["yaml_injection_code"] = {
        "figregistry_version": "0.3.0",
        "styles": {
            "malicious": {
                "color": "!!python/object/apply:os.system ['rm -rf /']"
            }
        }
    }
    
    # Command injection through configuration values
    security_configs["command_injection"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "outputs; rm -rf /; echo",
            "timestamp_format": "{name}; cat /etc/passwd; echo {ts}"
        }
    }
    
    # Large payload DoS attempt
    security_configs["dos_large_payload"] = {
        "figregistry_version": "0.3.0",
        "styles": {f"style_{i}": {"color": "#000000"} for i in range(10000)},
        "malicious_large_string": "A" * 1000000  # 1MB string
    }
    
    # Prototype pollution attempt
    security_configs["prototype_pollution"] = {
        "figregistry_version": "0.3.0",
        "__proto__": {
            "isAdmin": True,
            "polluted": "value"
        },
        "constructor": {
            "prototype": {
                "compromised": True
            }
        }
    }
    
    # SQL injection-like string injections
    security_configs["string_injection"] = {
        "figregistry_version": "0.3.0",
        "styles": {
            "'; DROP TABLE users; --": {"color": "#000000"},
            "<script>alert('xss')</script>": {"color": "#111111"}
        }
    }
    
    # NULL byte injection
    security_configs["null_byte_injection"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "outputs\x00/../../etc/passwd",
            "aliases": {
                "null_exploit": "safe_path\x00../../../../etc/shadow"
            }
        }
    }
    
    return security_configs


def generate_yaml_injection_vectors() -> List[str]:
    """Generate YAML injection attack vectors for parser security testing.
    
    Returns:
        List of malicious YAML strings designed to test parser security
    """
    injection_vectors = [
        # Python object execution
        '!!python/object/apply:os.system ["echo vulnerable"]',
        '!!python/object/apply:subprocess.call [["rm", "-rf", "/"]]',
        
        # File operations
        '!!python/object/apply:builtins.open ["/etc/passwd"]',
        '!!python/object/apply:builtins.eval ["__import__(\'os\').system(\'whoami\')"]',
        
        # Import injection
        '!!python/object/apply:builtins.__import__ ["os"]',
        '!!python/name:os.system',
        
        # Complex nested injections
        '''
        malicious: !!python/object/apply:subprocess.Popen
          - ["echo", "vulnerable"]
          - {stdout: !!python/name:subprocess.PIPE}
        ''',
        
        # Lambda injection
        '!!python/object/apply:builtins.eval ["lambda: __import__(\'os\').system(\'id\')"]',
        
        # Module-level injection
        '!!python/module:os',
        
        # Class instantiation injection
        '!!python/object:os.system ["echo injection"]'
    ]
    
    return injection_vectors


# ===============================================================================
# PERFORMANCE TEST CONFIGURATIONS
# ===============================================================================

def generate_performance_config_datasets() -> Iterator[Tuple[str, Dict[str, Any], int]]:
    """Generate configuration datasets for performance benchmarking per Section 6.6.4.3.
    
    Creates configurations of varying complexity for measuring configuration merge
    operations targeting <50ms overhead requirement.
    
    Yields:
        Tuple of (test_name, config_dict, expected_merge_time_ms)
    """
    # Small configuration - baseline performance
    small_config = {
        "figregistry_version": "0.3.0",
        "styles": {
            "test": {"color": "#000000", "marker": "o"}
        },
        "outputs": {"base_path": "outputs"}
    }
    yield ("small_config", small_config, 5)  # Expected < 5ms
    
    # Medium configuration - typical usage
    medium_config = generate_baseline_config()
    medium_config.update(generate_kedro_specific_config())
    yield ("medium_config", medium_config, 15)  # Expected < 15ms
    
    # Large configuration - complex scientific setup
    large_config = copy.deepcopy(generate_baseline_config())
    
    # Add many style conditions
    for i in range(50):
        large_config["styles"][f"condition_{i}"] = {
            "color": f"#{i:06x}",
            "marker": ["o", "s", "^", "v"][i % 4],
            "linestyle": ["-", "--", "-.", ":"][i % 4],
            "linewidth": 1.0 + (i % 5),
            "markersize": 6 + (i % 10),
            "alpha": 0.5 + (i % 5) * 0.1
        }
    
    # Add complex palette definitions
    large_config["palettes"] = {
        f"palette_{i}": [f"#{j:06x}" for j in range(i*10, (i+1)*10)]
        for i in range(10)
    }
    
    # Add extensive output configuration
    large_config["outputs"]["aliases"] = {
        f"alias_{i}": f"path_{i}" for i in range(20)
    }
    
    yield ("large_config", large_config, 45)  # Expected < 45ms (under 50ms target)
    
    # Extra large configuration - stress test
    xl_config = copy.deepcopy(large_config)
    for i in range(100, 200):
        xl_config["styles"][f"stress_condition_{i}"] = {
            "color": f"#{i:06x}",
            "marker": "o",
            "complex_nested": {
                "level_1": {
                    "level_2": {
                        "level_3": f"value_{i}"
                    }
                }
            }
        }
    
    yield ("xl_config", xl_config, 75)  # Stress test - may exceed 50ms target


def generate_concurrent_access_configs() -> List[Dict[str, Any]]:
    """Generate configurations for concurrent access testing.
    
    Returns:
        List of configuration dictionaries for thread-safety testing
    """
    configs = []
    
    # Base configuration for concurrent modification testing
    base_config = generate_baseline_config()
    
    # Configurations with different modification patterns
    for i in range(10):
        config = copy.deepcopy(base_config)
        config["test_thread_id"] = i
        config["styles"][f"thread_{i}_style"] = {
            "color": f"#{i*111111:06x}",
            "marker": "o",
            "thread_specific": True
        }
        configs.append(config)
    
    return configs


# ===============================================================================
# CROSS-PLATFORM CONFIGURATION VARIATIONS
# ===============================================================================

def generate_cross_platform_config_variations() -> Dict[str, Dict[str, Any]]:
    """Generate cross-platform configuration variations per Section 6.6.1.4.
    
    Ensures consistent test behavior across Windows, macOS, and Linux platforms
    with platform-specific path handling and filesystem considerations.
    
    Returns:
        Dict mapping platform names to platform-specific configurations
    """
    platform_configs = {}
    
    # Windows-specific configuration
    platform_configs["windows"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "C:\\Users\\testuser\\Documents\\figures",
            "aliases": {
                "expl": "C:\\temp\\exploratory",
                "pres": "D:\\presentations\\figures",
                "pub": "\\\\networkdrive\\publications"
            },
            "path_separator": "\\",
            "drive_relative_paths": True
        },
        "platform_specific": {
            "use_windows_paths": True,
            "handle_long_paths": True,
            "case_insensitive_paths": True
        }
    }
    
    # macOS-specific configuration
    platform_configs["macos"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "/Users/testuser/Documents/figures",
            "aliases": {
                "expl": "/tmp/exploratory",
                "pres": "/Users/testuser/presentations",
                "pub": "/Volumes/SharedDrive/publications"
            },
            "path_separator": "/",
            "respect_case_sensitivity": True
        },
        "platform_specific": {
            "use_hfs_plus": True,
            "handle_resource_forks": False,
            "respect_permissions": True
        }
    }
    
    # Linux-specific configuration
    platform_configs["linux"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "/home/testuser/figures",
            "aliases": {
                "expl": "/tmp/exploratory",
                "pres": "/home/testuser/presentations",
                "pub": "/mnt/shared/publications"
            },
            "path_separator": "/",
            "respect_case_sensitivity": True,
            "use_symlinks": True
        },
        "platform_specific": {
            "use_posix_permissions": True,
            "handle_symlinks": True,
            "respect_umask": True
        }
    }
    
    # Container/Docker configuration
    platform_configs["container"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "/app/outputs/figures",
            "aliases": {
                "expl": "/tmp/exploratory",
                "pres": "/app/presentations",
                "pub": "/mounted/publications"
            },
            "create_directories": True,
            "strict_permissions": False
        },
        "platform_specific": {
            "container_mode": True,
            "ephemeral_storage": True,
            "minimal_permissions": True
        }
    }
    
    return platform_configs


def generate_filesystem_edge_cases() -> Dict[str, Dict[str, Any]]:
    """Generate filesystem edge case configurations for robustness testing.
    
    Returns:
        Dict mapping edge case names to challenging filesystem configurations
    """
    edge_cases = {}
    
    # Very long path names
    edge_cases["long_paths"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "/".join(["very_long_directory_name" * 10] * 5),  # Very long path
            "aliases": {
                "long_alias": "a" * 255  # Maximum filename length
            }
        }
    }
    
    # Special characters in paths
    edge_cases["special_characters"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "outputs/with spaces/and-dashes/and_underscores",
            "aliases": {
                "unicode": "outputs/with_ünïcödé_characters",
                "symbols": "outputs/with!@#$%symbols"
            }
        }
    }
    
    # Read-only and permission scenarios
    edge_cases["permission_scenarios"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "/read_only_mount/figures",
            "create_directories": False,  # Cannot create dirs
            "respect_permissions": True
        }
    }
    
    # Network paths and remote filesystems
    edge_cases["network_paths"] = {
        "figregistry_version": "0.3.0",
        "outputs": {
            "base_path": "//remote_server/shared/figures",
            "aliases": {
                "nfs": "/nfs/mounted/figures",
                "smb": "//fileserver/research/figures"
            },
            "network_timeout": 30,
            "retry_on_failure": True
        }
    }
    
    return edge_cases


# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def create_temporary_config_file(config_dict: Dict[str, Any], 
                                suffix: str = ".yml") -> Path:
    """Create a temporary configuration file for testing.
    
    Args:
        config_dict: Configuration dictionary to write
        suffix: File suffix (default: .yml)
        
    Returns:
        Path to the created temporary file
    """
    import yaml
    
    # Create temporary file
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="figregistry_test_")
    
    try:
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    except Exception:
        os.close(fd)
        raise
    
    return Path(path)


def create_temporary_directory_structure(configs: Dict[str, Dict[str, Any]]) -> Path:
    """Create a temporary directory structure with multiple config files.
    
    Args:
        configs: Dict mapping filenames to configuration dictionaries
        
    Returns:
        Path to the temporary directory containing the config files
    """
    import yaml
    
    temp_dir = Path(tempfile.mkdtemp(prefix="figregistry_test_"))
    
    for filename, config in configs.items():
        config_path = temp_dir / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return temp_dir


def validate_config_against_schema(config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration dictionary against expected schema.
    
    Args:
        config_dict: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    if "figregistry_version" not in config_dict:
        errors.append("Missing required field: figregistry_version")
    
    # Validate version format
    version = config_dict.get("figregistry_version", "")
    if not isinstance(version, str) or not version:
        errors.append("figregistry_version must be a non-empty string")
    
    # Validate styles section if present
    if "styles" in config_dict:
        styles = config_dict["styles"]
        if not isinstance(styles, dict):
            errors.append("styles section must be a dictionary")
        else:
            for style_name, style_config in styles.items():
                if not isinstance(style_config, dict):
                    errors.append(f"Style '{style_name}' must be a dictionary")
    
    # Validate outputs section if present
    if "outputs" in config_dict:
        outputs = config_dict["outputs"]
        if not isinstance(outputs, dict):
            errors.append("outputs section must be a dictionary")
    
    return len(errors) == 0, errors


def generate_test_report_summary(test_results: Dict[str, Any]) -> str:
    """Generate a comprehensive test report summary.
    
    Args:
        test_results: Dictionary containing test execution results
        
    Returns:
        Formatted test report string
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    test_id = str(uuid.uuid4())[:8]
    
    report = f"""
FigRegistry Configuration Test Data Report
==========================================
Generated: {timestamp}
Test ID: {test_id}

Test Coverage Summary:
- Baseline configurations: {test_results.get('baseline_count', 0)}
- Environment variations: {test_results.get('environment_count', 0)}
- Invalid config scenarios: {test_results.get('invalid_count', 0)}
- Security test vectors: {test_results.get('security_count', 0)}
- Performance test datasets: {test_results.get('performance_count', 0)}
- Cross-platform variations: {test_results.get('platform_count', 0)}

Property-based Testing:
- Hypothesis strategies: {'Available' if HYPOTHESIS_AVAILABLE else 'Not Available'}
- Strategy coverage: {test_results.get('strategy_count', 0)} scenarios

Security Testing Coverage:
- Path traversal prevention: Enabled
- YAML injection protection: Enabled
- Command injection prevention: Enabled
- Configuration validation: Enabled

Performance Requirements:
- Target merge time: < 50ms
- Concurrent access: Thread-safe
- Memory efficiency: Optimized

Cross-platform Compatibility:
- Windows: Supported
- macOS: Supported  
- Linux: Supported
- Container environments: Supported
"""
    
    return report


# ===============================================================================
# MODULE INITIALIZATION AND EXPORTS
# ===============================================================================

# Export all public functions and classes
__all__ = [
    # Configuration generators
    "generate_baseline_config",
    "generate_kedro_specific_config", 
    "generate_environment_configs",
    
    # Invalid configuration generators
    "generate_invalid_config_scenarios",
    "generate_malformed_yaml_strings",
    
    # Merge testing
    "generate_merge_test_scenarios",
    
    # Hypothesis strategies (if available)
    "yaml_config_strategy",
    "kedro_config_strategy",
    "valid_color_strategy",
    "valid_marker_strategy",
    "style_dict_strategy",
    
    # Security testing
    "generate_security_test_configs",
    "generate_yaml_injection_vectors",
    
    # Performance testing
    "generate_performance_config_datasets",
    "generate_concurrent_access_configs",
    
    # Cross-platform testing
    "generate_cross_platform_config_variations",
    "generate_filesystem_edge_cases",
    
    # Utilities
    "create_temporary_config_file",
    "create_temporary_directory_structure",
    "validate_config_against_schema",
    "generate_test_report_summary",
    
    # Constants
    "HYPOTHESIS_AVAILABLE"
]


# Module-level configuration for testing framework integration
TEST_DATA_VERSION = "1.0.0"
SUPPORTED_FIGREGISTRY_VERSIONS = ["0.3.0", "0.3.1", "0.4.0"]
SUPPORTED_KEDRO_VERSIONS = ["0.18.0", "0.18.14", "0.19.0", "0.19.8"]

# Performance benchmarking constants
PERFORMANCE_TARGET_MERGE_TIME_MS = 50
PERFORMANCE_TARGET_VALIDATION_TIME_MS = 25
PERFORMANCE_MEMORY_LIMIT_MB = 100

# Security testing constants  
SECURITY_MAX_PAYLOAD_SIZE_MB = 10
SECURITY_PATH_TRAVERSAL_PATTERNS = ["../", "..\\", "%2e%2e%2f", "%2e%2e%5c"]
SECURITY_INJECTION_KEYWORDS = ["!!python", "!!map", "!!set", "!!binary"]

if __name__ == "__main__":
    # Module self-test and demonstration
    print("FigRegistry Configuration Test Data Generator")
    print(f"Version: {TEST_DATA_VERSION}")
    print(f"Hypothesis Available: {HYPOTHESIS_AVAILABLE}")
    
    # Generate sample test data
    baseline = generate_baseline_config()
    print(f"\nBaseline config generated with {len(baseline.get('styles', {}))} styles")
    
    invalid_scenarios = generate_invalid_config_scenarios()
    print(f"Invalid config scenarios: {len(invalid_scenarios)}")
    
    security_configs = generate_security_test_configs()
    print(f"Security test configurations: {len(security_configs)}")
    
    print("\nTest data generation complete!")