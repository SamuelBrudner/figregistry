"""Unit tests for FigRegistryConfigBridge component.

This module provides comprehensive unit tests for the FigRegistryConfigBridge that
validates configuration translation between Kedro's ConfigLoader and FigRegistry's
YAML-based system per the technical specification requirements.

Test Coverage Areas per Section 6.6.4:
- Configuration merging logic with precedence rules per F-007.2
- Environment-specific parameter resolution for multi-stage deployments
- Pydantic validation for merged configuration structures per Section 5.2.5
- Performance validation against <50ms configuration merging overhead per Section 6.6.4.3
- Thread-safe concurrent access patterns for parallel Kedro runner execution
- Comprehensive error aggregation for configuration validation failures
- Security validation for YAML injection and path traversal prevention per Section 6.6.8.1

Requirements Validation:
- F-007: Configuration bridge translates Kedro YAML into FigRegistry initialization parameters
- F-007.2: Seamless merging of project-level and traditional FigRegistry settings with precedence rules
- Section 5.2.5: Pydantic validation for type safety and in-memory operation during session lifecycle
- Section 6.6.4.3: <50ms configuration merging overhead performance requirement
- Section 6.6.8.1: Security validation against malicious configuration injection

Performance Targets (Section 6.6.4.3):
- Configuration Bridge Merge Time: <50ms per pipeline run
- Hook Initialization Overhead: <25ms per project startup
- Plugin Pipeline Execution Overhead: <200ms per FigureDataSet save
"""

import os
import sys
import tempfile
import shutil
import time
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest

# Import configuration bridge components
from figregistry_kedro.config import (
    FigRegistryConfigBridge,
    FigRegistryKedroConfig,
    ConfigurationMergeError,
    init_config,
    get_config_bridge,
    get_bridge_instance,
    set_bridge_instance
)

# Import test data generators
from figregistry_kedro.tests.data.config_test_data import (
    generate_baseline_config,
    generate_kedro_specific_config,
    generate_environment_configs,
    generate_invalid_config_scenarios,
    generate_malformed_yaml_strings,
    generate_merge_test_scenarios,
    generate_security_test_configs,
    generate_yaml_injection_vectors,
    generate_performance_config_datasets,
    generate_concurrent_access_configs,
    create_temporary_config_file,
    validate_config_against_schema
)

# Import test configuration from conftest.py
from figregistry_kedro.tests.conftest import (
    KEDRO_AVAILABLE,
    BENCHMARK_AVAILABLE,
    HYPOTHESIS_AVAILABLE
)

# Conditional imports with graceful fallback
if KEDRO_AVAILABLE:
    from kedro.config import ConfigLoader, OmegaConfigLoader
else:
    ConfigLoader = None
    OmegaConfigLoader = None

if HYPOTHESIS_AVAILABLE:
    from hypothesis import given, strategies as st, settings, HealthCheck
    from figregistry_kedro.tests.data.config_test_data import (
        yaml_config_strategy,
        kedro_config_strategy
    )

# Pydantic imports for validation testing
from pydantic import ValidationError
import yaml


class TestFigRegistryConfigBridge:
    """Test suite for FigRegistryConfigBridge configuration translation functionality.
    
    This test class provides comprehensive validation of the configuration bridge
    component per Section 6.6.6 testing requirements with ≥90% coverage target
    and 100% coverage for critical configuration merge operations.
    """

    def test_init_basic_initialization(self):
        """Test basic FigRegistryConfigBridge initialization.
        
        Validates initialization with default parameters and proper
        instance attribute setup per Section 5.2.5 requirements.
        """
        bridge = FigRegistryConfigBridge()
        
        assert bridge.config_loader is None
        assert bridge.environment == "base"
        assert bridge.enable_caching is True
        assert bridge._local_cache is None
        assert bridge._cache_key is None

    def test_init_with_kedro_config_loader(self, mock_config_loader):
        """Test initialization with Kedro ConfigLoader instance.
        
        Args:
            mock_config_loader: Mock ConfigLoader fixture from conftest.py
        """
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            environment="staging",
            enable_caching=False
        )
        
        assert bridge.config_loader == mock_config_loader
        assert bridge.environment == "staging"
        assert bridge.enable_caching is False

    def test_init_environment_parameter_handling(self):
        """Test environment parameter handling and default behavior."""
        # Test default environment
        bridge_default = FigRegistryConfigBridge()
        assert bridge_default.environment == "base"
        
        # Test explicit environment
        bridge_explicit = FigRegistryConfigBridge(environment="production")
        assert bridge_explicit.environment == "production"
        
        # Test None environment fallback
        bridge_none = FigRegistryConfigBridge(environment=None)
        assert bridge_none.environment == "base"

    def test_generate_cache_key_deterministic(self, base_figregistry_config):
        """Test cache key generation is deterministic and consistent.
        
        Args:
            base_figregistry_config: Baseline config fixture from conftest.py
        """
        bridge = FigRegistryConfigBridge(environment="test")
        
        # Generate cache keys for same configuration multiple times
        key1 = bridge._generate_cache_key(base_figregistry_config)
        key2 = bridge._generate_cache_key(base_figregistry_config)
        key3 = bridge._generate_cache_key(base_figregistry_config)
        
        assert key1 == key2 == key3
        assert isinstance(key1, str)
        assert len(key1) > 0

    def test_generate_cache_key_uniqueness(self, base_figregistry_config):
        """Test cache key uniqueness for different configurations."""
        bridge = FigRegistryConfigBridge(environment="test")
        
        # Original config
        key1 = bridge._generate_cache_key(base_figregistry_config)
        
        # Modified config
        modified_config = base_figregistry_config.copy()
        modified_config["test_modification"] = "unique_value"
        key2 = bridge._generate_cache_key(modified_config)
        
        # Different environment
        bridge_diff_env = FigRegistryConfigBridge(environment="production")
        key3 = bridge_diff_env._generate_cache_key(base_figregistry_config)
        
        assert key1 != key2  # Different configs
        assert key1 != key3  # Different environments
        assert key2 != key3  # Both different

    def test_load_figregistry_config_file_exists(self, temp_directory, base_figregistry_config):
        """Test loading standalone figregistry.yaml configuration.
        
        Args:
            temp_directory: Temporary directory fixture
            base_figregistry_config: Baseline configuration fixture
        """
        # Create figregistry.yaml in temporary directory
        config_path = temp_directory / "figregistry.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_figregistry_config, f)
        
        # Change to temp directory for test
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_directory)
            bridge = FigRegistryConfigBridge()
            loaded_config = bridge._load_figregistry_config()
            
            assert loaded_config == base_figregistry_config
            assert "figregistry_version" in loaded_config
            assert "styles" in loaded_config
        finally:
            os.chdir(original_cwd)

    def test_load_figregistry_config_file_missing(self, temp_directory):
        """Test loading figregistry.yaml when file does not exist."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_directory)
            bridge = FigRegistryConfigBridge()
            loaded_config = bridge._load_figregistry_config()
            
            assert loaded_config == {}
        finally:
            os.chdir(original_cwd)

    def test_load_figregistry_config_invalid_yaml(self, temp_directory):
        """Test loading figregistry.yaml with invalid YAML content."""
        # Create invalid YAML file
        config_path = temp_directory / "figregistry.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [\n")  # Malformed YAML
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_directory)
            bridge = FigRegistryConfigBridge()
            
            # Should handle YAML parsing error gracefully
            with pytest.warns(UserWarning):
                loaded_config = bridge._load_figregistry_config()
            
            assert loaded_config == {}
        finally:
            os.chdir(original_cwd)

    def test_load_kedro_figregistry_config_success(self, mock_config_loader, base_figregistry_config):
        """Test loading FigRegistry configuration through Kedro ConfigLoader.
        
        Args:
            mock_config_loader: Mock ConfigLoader fixture
            base_figregistry_config: Baseline configuration fixture
        """
        # Configure mock to return test configuration
        mock_config_loader.get.return_value = base_figregistry_config
        
        bridge = FigRegistryConfigBridge(config_loader=mock_config_loader)
        loaded_config = bridge._load_kedro_figregistry_config()
        
        assert loaded_config == base_figregistry_config
        # Verify ConfigLoader.get was called with expected patterns
        mock_config_loader.get.assert_called()

    def test_load_kedro_figregistry_config_no_loader(self):
        """Test loading Kedro config when no ConfigLoader provided."""
        bridge = FigRegistryConfigBridge(config_loader=None)
        loaded_config = bridge._load_kedro_figregistry_config()
        
        assert loaded_config == {}

    def test_load_kedro_figregistry_config_loader_error(self, mock_config_loader):
        """Test handling ConfigLoader errors during Kedro config loading."""
        # Configure mock to raise exception
        mock_config_loader.get.side_effect = Exception("ConfigLoader error")
        
        bridge = FigRegistryConfigBridge(config_loader=mock_config_loader)
        
        # Should handle error gracefully and return empty dict
        with pytest.warns(UserWarning):
            loaded_config = bridge._load_kedro_figregistry_config()
        
        assert loaded_config == {}

    def test_merge_configurations_basic_no_conflicts(self):
        """Test basic configuration merging without conflicts per F-007.2."""
        bridge = FigRegistryConfigBridge()
        
        figregistry_config = {
            "figregistry_version": "0.3.0",
            "styles": {"exploratory": {"color": "#1f77b4"}},
            "outputs": {"base_path": "outputs"}
        }
        
        kedro_config = {
            "kedro": {"enable_versioning": True},
            "additional_setting": "value"
        }
        
        merged = bridge._merge_configurations(figregistry_config, kedro_config)
        
        # Verify all sections preserved
        assert merged["figregistry_version"] == "0.3.0"
        assert merged["styles"]["exploratory"]["color"] == "#1f77b4"
        assert merged["outputs"]["base_path"] == "outputs"
        assert merged["kedro"]["enable_versioning"] is True
        assert merged["additional_setting"] == "value"
        assert merged["environment"] == "base"  # Added by merge process

    def test_merge_configurations_kedro_precedence(self):
        """Test Kedro configuration precedence over FigRegistry per F-007.2."""
        bridge = FigRegistryConfigBridge()
        
        figregistry_config = {
            "figregistry_version": "0.3.0",
            "outputs": {
                "base_path": "figregistry_outputs",
                "formats": ["png"]
            },
            "defaults": {"dpi": 150}
        }
        
        kedro_config = {
            "outputs": {
                "base_path": "kedro_outputs",  # Should override
                "formats": ["png", "pdf"]      # Should override
            },
            "defaults": {"dpi": 300}          # Should override
        }
        
        merged = bridge._merge_configurations(figregistry_config, kedro_config)
        
        # Verify Kedro values take precedence
        assert merged["outputs"]["base_path"] == "kedro_outputs"
        assert merged["outputs"]["formats"] == ["png", "pdf"]
        assert merged["defaults"]["dpi"] == 300

    def test_merge_configurations_deep_merge_nested_dicts(self):
        """Test deep merging of nested dictionary structures per F-007.2."""
        bridge = FigRegistryConfigBridge()
        
        figregistry_config = {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {
                    "color": "#1f77b4",
                    "marker": "o",
                    "linewidth": 1.0
                }
            }
        }
        
        kedro_config = {
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
        }
        
        merged = bridge._merge_configurations(figregistry_config, kedro_config)
        
        # Verify deep merge behavior
        exploratory_style = merged["styles"]["exploratory"]
        assert exploratory_style["color"] == "#1f77b4"      # Preserved
        assert exploratory_style["marker"] == "s"           # Overridden
        assert exploratory_style["linewidth"] == 1.0        # Preserved
        assert exploratory_style["markersize"] == 8         # Added
        
        # Verify new section added
        presentation_style = merged["styles"]["presentation"]
        assert presentation_style["color"] == "#ff7f0e"
        assert presentation_style["marker"] == "^"

    @pytest.mark.performance
    def test_merge_configurations_performance_requirement(self, base_figregistry_config):
        """Test configuration merging meets <50ms performance requirement per Section 6.6.4.3."""
        bridge = FigRegistryConfigBridge()
        kedro_config = generate_kedro_specific_config()
        
        # Measure merge performance
        start_time = time.perf_counter()
        merged = bridge._merge_configurations(base_figregistry_config, kedro_config)
        end_time = time.perf_counter()
        
        merge_time_ms = (end_time - start_time) * 1000
        
        # Verify performance requirement
        assert merge_time_ms < 50.0, f"Merge time {merge_time_ms:.2f}ms exceeds 50ms target"
        
        # Verify merge correctness wasn't compromised for performance
        assert "figregistry_version" in merged
        assert "kedro" in merged
        assert merged["environment"] == "base"

    def test_validate_configuration_success(self, base_figregistry_config):
        """Test successful configuration validation with Pydantic per Section 5.2.5."""
        bridge = FigRegistryConfigBridge()
        
        validated_config = bridge._validate_configuration(base_figregistry_config)
        
        assert isinstance(validated_config, FigRegistryKedroConfig)
        assert validated_config.styles == base_figregistry_config["styles"]
        assert validated_config.outputs == base_figregistry_config["outputs"]
        assert validated_config.defaults == base_figregistry_config["defaults"]

    def test_validate_configuration_validation_error(self, invalid_config_scenarios):
        """Test configuration validation failure handling per Section 5.2.5.
        
        Args:
            invalid_config_scenarios: Invalid config scenarios fixture
        """
        bridge = FigRegistryConfigBridge()
        
        # Test various invalid configuration scenarios
        for scenario_name, invalid_config in invalid_config_scenarios.items():
            with pytest.raises(ConfigurationMergeError) as exc_info:
                bridge._validate_configuration(invalid_config)
            
            error = exc_info.value
            assert "Configuration validation failed" in str(error)
            assert hasattr(error, 'errors')
            assert len(error.errors) > 0

    def test_validate_configuration_type_safety(self):
        """Test Pydantic type safety validation per Section 5.2.5."""
        bridge = FigRegistryConfigBridge()
        
        # Configuration with type mismatches
        invalid_config = {
            "figregistry_version": "0.3.0",
            "styles": "should_be_dict",  # Wrong type
            "enable_concurrent_access": "should_be_bool",  # Wrong type
            "validation_enabled": 123  # Wrong type
        }
        
        with pytest.raises(ConfigurationMergeError) as exc_info:
            bridge._validate_configuration(invalid_config)
        
        error = exc_info.value
        assert len(error.errors) >= 2  # At least 2 type errors

    def test_get_merged_config_full_workflow(self, mock_config_loader, base_figregistry_config):
        """Test complete get_merged_config workflow per F-007 requirements.
        
        Args:
            mock_config_loader: Mock ConfigLoader fixture
            base_figregistry_config: Baseline configuration fixture
        """
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            environment="production"
        )
        
        # Mock the figregistry config loading to avoid file system dependency
        with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
            merged_config = bridge.get_merged_config()
        
        # Verify complete workflow
        assert isinstance(merged_config, FigRegistryKedroConfig)
        assert merged_config.environment == "production"
        
        # Verify FigRegistry content preserved
        assert "exploratory" in merged_config.styles
        assert merged_config.outputs["base_path"] == "outputs/figures"
        
        # Verify Kedro integration added
        assert merged_config.kedro is not None
        assert "data_layer_mappings" in merged_config.kedro

    def test_get_merged_config_caching_enabled(self, mock_config_loader, base_figregistry_config):
        """Test configuration caching behavior for performance optimization."""
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            enable_caching=True
        )
        
        with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
            # First call - should load and cache
            config1 = bridge.get_merged_config()
            
            # Second call - should use cache
            config2 = bridge.get_merged_config()
        
        # Verify same instance returned (cached)
        assert config1 is config2
        
        # Verify ConfigLoader only called once due to caching
        assert mock_config_loader.get.call_count == 1

    def test_get_merged_config_caching_disabled(self, mock_config_loader, base_figregistry_config):
        """Test configuration behavior with caching disabled."""
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            enable_caching=False
        )
        
        with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
            # Two separate calls without caching
            config1 = bridge.get_merged_config()
            config2 = bridge.get_merged_config()
        
        # Verify separate instances (no caching)
        assert config1 is not config2
        
        # Verify content is identical
        assert config1.dict() == config2.dict()

    def test_get_merged_config_error_handling(self, mock_config_loader):
        """Test comprehensive error handling in get_merged_config per Section 5.2.5."""
        # Configure mock to raise exception
        mock_config_loader.get.side_effect = Exception("Kedro loader failure")
        
        bridge = FigRegistryConfigBridge(config_loader=mock_config_loader)
        
        with pytest.raises(ConfigurationMergeError) as exc_info:
            bridge.get_merged_config()
        
        error = exc_info.value
        assert "Configuration bridge failed" in str(error)

    def test_clear_cache_functionality(self, mock_config_loader, base_figregistry_config):
        """Test cache clearing functionality for forced reload."""
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            enable_caching=True
        )
        
        with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
            # Load and cache configuration
            config1 = bridge.get_merged_config()
            assert bridge._local_cache is not None
            
            # Clear cache
            bridge.clear_cache()
            assert bridge._local_cache is None
            assert bridge._cache_key is None
            
            # Load again - should create new instance
            config2 = bridge.get_merged_config()
            assert config1 is not config2

    def test_clear_global_cache_functionality(self):
        """Test global cache clearing functionality."""
        # Ensure global cache has some content
        FigRegistryConfigBridge._config_cache["test_key"] = Mock()
        assert len(FigRegistryConfigBridge._config_cache) > 0
        
        # Clear global cache
        FigRegistryConfigBridge.clear_global_cache()
        
        assert len(FigRegistryConfigBridge._config_cache) == 0

    @pytest.mark.performance  
    def test_concurrent_access_thread_safety(self, mock_config_loader, base_figregistry_config):
        """Test thread-safe concurrent access patterns per Section 5.2.5.
        
        Validates concurrent access required for parallel Kedro runner execution.
        """
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            enable_caching=True
        )
        
        results = []
        errors = []
        
        def worker_thread():
            """Worker function for concurrent access testing."""
            try:
                with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
                    config = bridge.get_merged_config()
                    results.append(config)
            except Exception as e:
                errors.append(e)
        
        # Execute concurrent access with thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_thread) for _ in range(10)]
            
            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()  # Will raise if thread had exception
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # Verify all threads got valid configurations
        assert len(results) == 10
        for config in results:
            assert isinstance(config, FigRegistryKedroConfig)
            assert config.figregistry_version == "0.3.0"

    @pytest.mark.performance
    def test_performance_with_large_configurations(self, performance_config_datasets):
        """Test performance with large configuration datasets per Section 6.6.4.3.
        
        Args:
            performance_config_datasets: Performance test datasets fixture
        """
        bridge = FigRegistryConfigBridge()
        
        for test_name, config_dict, expected_time_ms in performance_config_datasets:
            # Measure merge performance
            start_time = time.perf_counter()
            merged = bridge._merge_configurations(config_dict, {})
            end_time = time.perf_counter()
            
            actual_time_ms = (end_time - start_time) * 1000
            
            # Verify performance within expected bounds
            assert actual_time_ms < expected_time_ms, (
                f"Test {test_name}: {actual_time_ms:.2f}ms exceeds {expected_time_ms}ms limit"
            )
            
            # Verify merge correctness
            assert "figregistry_version" in merged
            assert merged["environment"] == "base"

    @pytest.mark.security
    def test_security_yaml_injection_prevention(self, security_test_configs):
        """Test YAML injection prevention per Section 6.6.8.1.
        
        Args:
            security_test_configs: Security test configurations fixture
        """
        bridge = FigRegistryConfigBridge()
        
        for attack_name, malicious_config in security_test_configs.items():
            # Attempt to validate malicious configuration
            try:
                validated_config = bridge._validate_configuration(malicious_config)
                
                # If validation succeeds, ensure no dangerous content
                config_dict = validated_config.dict()
                config_str = str(config_dict)
                
                # Check for injection patterns
                dangerous_patterns = ["!!python", "!!map", "!!set", "__import__"]
                for pattern in dangerous_patterns:
                    assert pattern not in config_str, (
                        f"Attack {attack_name}: Dangerous pattern '{pattern}' found in validated config"
                    )
                
            except (ConfigurationMergeError, ValidationError):
                # Validation failure is acceptable for malicious configs
                pass

    @pytest.mark.security
    def test_security_path_traversal_prevention(self, security_test_configs):
        """Test path traversal prevention in configuration values per Section 6.6.8.1."""
        bridge = FigRegistryConfigBridge()
        
        # Focus on path traversal attack configurations
        path_traversal_configs = {
            name: config for name, config in security_test_configs.items()
            if "path_traversal" in name
        }
        
        for attack_name, malicious_config in path_traversal_configs.items():
            try:
                validated_config = bridge._validate_configuration(malicious_config)
                
                # Check outputs section for path traversal attempts
                if validated_config.outputs:
                    base_path = str(validated_config.outputs.get("base_path", ""))
                    aliases = validated_config.outputs.get("aliases", {})
                    
                    # Verify no path traversal patterns
                    dangerous_paths = ["../", "..\\", "%2e%2e%2f", "/etc/", "/root/"]
                    for dangerous in dangerous_paths:
                        assert dangerous not in base_path, (
                            f"Attack {attack_name}: Path traversal in base_path: {base_path}"
                        )
                        
                        for alias_path in aliases.values():
                            assert dangerous not in str(alias_path), (
                                f"Attack {attack_name}: Path traversal in alias: {alias_path}"
                            )
                
            except (ConfigurationMergeError, ValidationError):
                # Validation failure is acceptable for malicious configs
                pass

    def test_environment_specific_configuration_resolution(self):
        """Test environment-specific configuration resolution per F-007.2."""
        environment_configs = generate_environment_configs()
        
        for env_name, env_config in environment_configs.items():
            bridge = FigRegistryConfigBridge(environment=env_name)
            
            base_config = generate_baseline_config()
            merged = bridge._merge_configurations(base_config, env_config)
            
            # Verify environment-specific overrides applied
            assert merged["environment"] == env_name
            
            # Verify specific environment characteristics
            if env_name == "development":
                assert merged["defaults"]["dpi"] == 150  # Lower DPI for dev
            elif env_name == "production":
                assert merged["defaults"]["dpi"] == 300  # Higher DPI for prod
                assert "security" in merged  # Security section added

    def test_configuration_merge_precedence_rules(self):
        """Test comprehensive precedence rules for configuration merging per F-007.2."""
        bridge = FigRegistryConfigBridge()
        
        # Base FigRegistry configuration
        figregistry_config = {
            "figregistry_version": "0.3.0",
            "outputs": {"base_path": "figregistry_path", "formats": ["png"]},
            "styles": {"test": {"color": "#111111", "marker": "o"}},
            "defaults": {"dpi": 150, "figsize": [6, 4]}
        }
        
        # Kedro configuration with overlapping and new settings
        kedro_config = {
            "outputs": {"base_path": "kedro_path", "formats": ["png", "pdf"], "new_setting": True},
            "styles": {"test": {"color": "#222222", "linewidth": 2.0}, "new_style": {"color": "#333333"}},
            "defaults": {"dpi": 300, "font_size": 12},
            "kedro_specific": {"enable_hooks": True}
        }
        
        merged = bridge._merge_configurations(figregistry_config, kedro_config)
        
        # Test precedence rules
        # 1. Kedro overrides FigRegistry for conflicting keys
        assert merged["outputs"]["base_path"] == "kedro_path"
        assert merged["outputs"]["formats"] == ["png", "pdf"]
        assert merged["styles"]["test"]["color"] == "#222222"
        assert merged["defaults"]["dpi"] == 300
        
        # 2. Kedro adds new keys without conflict
        assert merged["outputs"]["new_setting"] is True
        assert merged["styles"]["new_style"]["color"] == "#333333"
        assert merged["defaults"]["font_size"] == 12
        assert merged["kedro_specific"]["enable_hooks"] is True
        
        # 3. FigRegistry values preserved when no conflict
        assert merged["figregistry_version"] == "0.3.0"
        assert merged["styles"]["test"]["marker"] == "o"  # Not overridden
        assert merged["defaults"]["figsize"] == [6, 4]  # Not overridden
        
        # 4. Deep merge preserves nested structure
        assert merged["styles"]["test"]["linewidth"] == 2.0  # Added by Kedro

    @pytest.mark.parametrize("environment", ["development", "staging", "production"])
    def test_multi_environment_configuration_handling(self, environment):
        """Test multi-environment configuration handling per F-007.2.
        
        Args:
            environment: Environment name for parameterized testing
        """
        environment_configs = generate_environment_configs()
        env_specific_config = environment_configs[environment]
        
        bridge = FigRegistryConfigBridge(environment=environment)
        base_config = generate_baseline_config()
        
        merged = bridge._merge_configurations(base_config, env_specific_config)
        validated = bridge._validate_configuration(merged)
        
        # Verify environment applied correctly
        assert validated.environment == environment
        
        # Verify environment-specific validations
        if environment == "development":
            assert validated.defaults["dpi"] == 150
            assert "dev" in validated.outputs["timestamp_format"]
        elif environment == "staging":
            assert validated.defaults.get("enable_validation") is True
            assert "staging" in validated.outputs["timestamp_format"]
        elif environment == "production":
            assert validated.defaults["dpi"] == 300
            assert "prod" in validated.outputs["timestamp_format"]
            assert "security" in validated.dict()

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        figregistry_config=yaml_config_strategy(),
        kedro_config=kedro_config_strategy()
    )
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_property_based_configuration_merging(self, figregistry_config, kedro_config):
        """Property-based test for configuration merging invariants per Section 6.6.2.6.
        
        Uses Hypothesis to test merging behavior across wide range of configurations.
        
        Args:
            figregistry_config: Hypothesis-generated FigRegistry configuration
            kedro_config: Hypothesis-generated Kedro configuration
        """
        bridge = FigRegistryConfigBridge()
        
        try:
            merged = bridge._merge_configurations(figregistry_config, kedro_config)
            
            # Property: Merged config contains all FigRegistry required fields
            assert "figregistry_version" in merged
            assert merged["figregistry_version"] == figregistry_config["figregistry_version"]
            
            # Property: Environment field always added
            assert "environment" in merged
            assert merged["environment"] == "base"
            
            # Property: Kedro config values present when provided
            if "kedro" in kedro_config:
                assert "kedro" in merged
                for key, value in kedro_config["kedro"].items():
                    assert merged["kedro"][key] == value
            
            # Property: Merged config can be validated
            validated = bridge._validate_configuration(merged)
            assert isinstance(validated, FigRegistryKedroConfig)
            
        except Exception as e:
            # Log hypothesis failure details for debugging
            pytest.fail(f"Property-based test failed with configs:\nFigRegistry: {figregistry_config}\nKedro: {kedro_config}\nError: {e}")

    def test_error_aggregation_comprehensive_reporting(self):
        """Test comprehensive error aggregation per Section 5.2.5.
        
        Validates that configuration validation failures provide clear,
        comprehensive error reporting for troubleshooting.
        """
        bridge = FigRegistryConfigBridge()
        
        # Create configuration with multiple validation errors
        multi_error_config = {
            "figregistry_version": 123,  # Wrong type
            "styles": "should_be_dict",  # Wrong type
            "outputs": {"base_path": 456},  # Wrong type for nested field
            "enable_concurrent_access": "not_a_boolean",  # Wrong type
            "validation_enabled": {"complex": "object"}  # Wrong type
        }
        
        with pytest.raises(ConfigurationMergeError) as exc_info:
            bridge._validate_configuration(multi_error_config)
        
        error = exc_info.value
        
        # Verify comprehensive error reporting
        assert "Configuration validation failed" in str(error)
        assert hasattr(error, 'errors')
        assert len(error.errors) >= 4  # Multiple errors captured
        
        # Verify error details structure
        for error_detail in error.errors:
            assert 'field' in error_detail
            assert 'message' in error_detail
            assert 'type' in error_detail

    def test_in_memory_operation_lifecycle(self, mock_config_loader, base_figregistry_config):
        """Test in-memory operation during session lifecycle per Section 5.2.5.
        
        Validates that configuration bridge operates entirely in-memory
        without persistent state between pipeline runs.
        """
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            enable_caching=True
        )
        
        with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
            # First session - load configuration
            config1 = bridge.get_merged_config()
            assert bridge._local_cache is not None
            
            # Verify in-memory state
            original_cache_size = len(FigRegistryConfigBridge._config_cache)
            
            # Simulate session end - clear cache
            bridge.clear_cache()
            assert bridge._local_cache is None
            
            # Second session - should reload from sources
            config2 = bridge.get_merged_config()
            
            # Verify independent sessions
            assert config1 is not config2  # Different instances
            assert config1.dict() == config2.dict()  # Same content
            
            # Verify cache behavior
            assert len(FigRegistryConfigBridge._config_cache) >= original_cache_size


class TestInitConfigFunction:
    """Test suite for init_config() function per F-007 requirements.
    
    Validates initialization function for FigRegistry integration during
    Kedro project startup with validated configuration objects.
    """
    
    def test_init_config_with_kedro_loader(self, mock_config_loader, base_figregistry_config):
        """Test init_config with Kedro ConfigLoader integration."""
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        with patch('figregistry_kedro.config.figregistry') as mock_figregistry:
            with patch('figregistry_kedro.config.FigRegistryConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_class.return_value = mock_config_instance
                
                with patch.object(FigRegistryConfigBridge, '_load_figregistry_config', return_value=base_figregistry_config):
                    result = init_config(
                        config_loader=mock_config_loader,
                        environment="production"
                    )
                
                assert result == mock_config_instance

    def test_init_config_without_kedro_loader(self, base_figregistry_config):
        """Test init_config without Kedro ConfigLoader (standalone mode)."""
        with patch('figregistry_kedro.config.figregistry') as mock_figregistry:
            with patch('figregistry_kedro.config.FigRegistryConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_class.return_value = mock_config_instance
                
                with patch.object(FigRegistryConfigBridge, '_load_figregistry_config', return_value=base_figregistry_config):
                    result = init_config(
                        config_loader=None,
                        environment="base"
                    )
                
                assert result == mock_config_instance

    def test_init_config_figregistry_not_available(self):
        """Test init_config when FigRegistry is not available."""
        with patch('figregistry_kedro.config.figregistry', None):
            with patch('figregistry_kedro.config.load_config', None):
                result = init_config()
                
                assert result is None

    def test_init_config_configuration_merge_error(self, mock_config_loader):
        """Test init_config error handling for configuration merge failures."""
        # Configure mock to return invalid configuration
        mock_config_loader.get.return_value = {"invalid": "config"}
        
        with patch('figregistry_kedro.config.figregistry'):
            with patch.object(FigRegistryConfigBridge, '_load_figregistry_config', return_value={}):
                with pytest.raises(ConfigurationMergeError):
                    init_config(config_loader=mock_config_loader)

    def test_init_config_with_kwargs(self, mock_config_loader, base_figregistry_config):
        """Test init_config with additional keyword arguments."""
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        with patch('figregistry_kedro.config.figregistry') as mock_figregistry:
            with patch.object(FigRegistryConfigBridge, '_load_figregistry_config', return_value=base_figregistry_config):
                result = init_config(
                    config_loader=mock_config_loader,
                    environment="staging",
                    enable_caching=False,
                    custom_param="test_value"
                )
                
                # Verify function completed successfully
                assert result is not None


class TestUtilityFunctions:
    """Test suite for utility functions and module-level functionality."""
    
    def test_get_config_bridge_function(self, mock_config_loader):
        """Test get_config_bridge convenience function."""
        bridge = get_config_bridge(
            config_loader=mock_config_loader,
            environment="test",
            enable_caching=False
        )
        
        assert isinstance(bridge, FigRegistryConfigBridge)
        assert bridge.config_loader == mock_config_loader
        assert bridge.environment == "test"
        assert bridge.enable_caching is False

    def test_bridge_instance_management(self):
        """Test module-level bridge instance management."""
        # Initially no bridge instance
        assert get_bridge_instance() is None
        
        # Set bridge instance
        test_bridge = FigRegistryConfigBridge(environment="test")
        set_bridge_instance(test_bridge)
        
        # Verify instance stored and retrieved
        retrieved_bridge = get_bridge_instance()
        assert retrieved_bridge is test_bridge
        assert retrieved_bridge.environment == "test"
        
        # Clear instance
        set_bridge_instance(None)
        assert get_bridge_instance() is None

    def test_bridge_instance_thread_safety(self):
        """Test thread safety of module-level bridge instance management."""
        results = []
        
        def worker_thread(thread_id):
            """Worker function for thread safety testing."""
            bridge = FigRegistryConfigBridge(environment=f"thread_{thread_id}")
            set_bridge_instance(bridge)
            retrieved = get_bridge_instance()
            results.append((thread_id, retrieved.environment))
        
        # Execute concurrent instance management
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(3)]
            
            for future in as_completed(futures):
                future.result()
        
        # Verify final state is consistent
        final_bridge = get_bridge_instance()
        assert final_bridge is not None
        assert final_bridge.environment.startswith("thread_")


class TestConfigurationMergeError:
    """Test suite for ConfigurationMergeError exception class."""
    
    def test_configuration_merge_error_basic(self):
        """Test basic ConfigurationMergeError creation and attributes."""
        error_message = "Test configuration error"
        error = ConfigurationMergeError(error_message)
        
        assert str(error) == error_message
        assert error.errors == []

    def test_configuration_merge_error_with_errors(self):
        """Test ConfigurationMergeError with detailed error list."""
        error_message = "Multiple validation errors"
        error_details = [
            {"field": "styles", "message": "must be a dict", "type": "type_error"},
            {"field": "version", "message": "field required", "type": "value_error"}
        ]
        
        error = ConfigurationMergeError(error_message, error_details)
        
        assert str(error) == error_message
        assert error.errors == error_details
        assert len(error.errors) == 2

    def test_configuration_merge_error_inheritance(self):
        """Test ConfigurationMergeError inheritance from Exception."""
        error = ConfigurationMergeError("test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationMergeError)


class TestFigRegistryKedroConfigModel:
    """Test suite for FigRegistryKedroConfig Pydantic model per Section 5.2.5."""
    
    def test_pydantic_model_validation_success(self, base_figregistry_config):
        """Test successful Pydantic model validation."""
        config = FigRegistryKedroConfig(**base_figregistry_config)
        
        assert config.styles == base_figregistry_config["styles"]
        assert config.outputs == base_figregistry_config["outputs"]
        assert config.defaults == base_figregistry_config["defaults"]
        assert config.environment == "base"  # Default value
        assert config.enable_concurrent_access is True  # Default value

    def test_pydantic_model_default_values(self):
        """Test Pydantic model default value assignment."""
        minimal_config = {"figregistry_version": "0.3.0"}
        config = FigRegistryKedroConfig(**minimal_config)
        
        assert config.styles == {}
        assert config.palettes == {}
        assert config.outputs == {}
        assert config.defaults == {}
        assert config.kedro == {}
        assert config.environment == "base"
        assert config.enable_concurrent_access is True
        assert config.validation_enabled is True

    def test_pydantic_model_validation_errors(self):
        """Test Pydantic model validation error scenarios."""
        # Invalid types
        with pytest.raises(ValidationError):
            FigRegistryKedroConfig(
                styles="should_be_dict",
                enable_concurrent_access="should_be_bool"
            )

    def test_pydantic_model_custom_validators(self):
        """Test custom Pydantic validators in FigRegistryKedroConfig."""
        # Test styles validator
        with pytest.raises(ValidationError) as exc_info:
            FigRegistryKedroConfig(styles="not_a_dict")
        
        errors = exc_info.value.errors()
        assert any("Styles must be a dictionary" in str(error.get("msg", "")) for error in errors)

    def test_pydantic_model_extra_fields_allowed(self):
        """Test that extra fields are allowed in Pydantic model."""
        config_data = {
            "figregistry_version": "0.3.0",
            "custom_field": "custom_value",
            "extra_section": {"nested": "data"}
        }
        
        config = FigRegistryKedroConfig(**config_data)
        config_dict = config.dict()
        
        assert "custom_field" in config_dict
        assert config_dict["custom_field"] == "custom_value"
        assert "extra_section" in config_dict

    def test_pydantic_model_dict_export(self, base_figregistry_config):
        """Test Pydantic model dictionary export functionality."""
        config = FigRegistryKedroConfig(**base_figregistry_config)
        config_dict = config.dict()
        
        # Verify all expected fields present
        assert "styles" in config_dict
        assert "outputs" in config_dict
        assert "defaults" in config_dict
        assert "environment" in config_dict
        
        # Verify exclude functionality
        limited_dict = config.dict(exclude={"kedro", "environment"})
        assert "kedro" not in limited_dict
        assert "environment" not in limited_dict
        assert "styles" in limited_dict


# =============================================================================
# INTEGRATION AND END-TO-END TESTS
# =============================================================================

@pytest.mark.integration
class TestConfigBridgeIntegration:
    """Integration tests for FigRegistryConfigBridge with realistic scenarios."""
    
    def test_full_kedro_integration_workflow(self, temp_project_directory, mock_config_loader):
        """Test complete integration workflow with realistic Kedro project structure."""
        project_paths = temp_project_directory
        
        # Create figregistry.yml in Kedro project
        figregistry_config = generate_baseline_config()
        figregistry_path = project_paths["conf_path"] / "base" / "figregistry.yml"
        with open(figregistry_path, 'w') as f:
            yaml.dump(figregistry_config, f)
        
        # Create local environment overrides
        local_config = generate_environment_configs()["development"]
        local_path = project_paths["conf_path"] / "local" / "figregistry.yml"
        with open(local_path, 'w') as f:
            yaml.dump(local_config, f)
        
        # Configure mock ConfigLoader to load files
        def mock_get_config(pattern):
            if "figregistry" in pattern:
                return figregistry_config
            return {}
        
        mock_config_loader.get.side_effect = mock_get_config
        
        # Test integration
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            environment="development"
        )
        
        merged_config = bridge.get_merged_config()
        
        # Verify integration success
        assert isinstance(merged_config, FigRegistryKedroConfig)
        assert merged_config.environment == "development"
        assert "exploratory" in merged_config.styles
        
        # Verify local overrides applied
        assert merged_config.defaults["dpi"] == 150  # Development override

    def test_realistic_multi_environment_deployment(self, mock_config_loader):
        """Test realistic multi-environment deployment scenario per F-007.2."""
        base_config = generate_baseline_config()
        environments = generate_environment_configs()
        
        for env_name, env_config in environments.items():
            # Simulate ConfigLoader for specific environment
            mock_config_loader.get.return_value = env_config
            
            bridge = FigRegistryConfigBridge(
                config_loader=mock_config_loader,
                environment=env_name
            )
            
            with patch.object(bridge, '_load_figregistry_config', return_value=base_config):
                merged_config = bridge.get_merged_config()
            
            # Verify environment-specific characteristics
            assert merged_config.environment == env_name
            
            if env_name == "production":
                assert "security" in merged_config.dict()
                assert merged_config.defaults["dpi"] == 300
            elif env_name == "development":
                assert merged_config.defaults["dpi"] == 150
                assert merged_config.kedro.get("enable_versioning") is False

    @pytest.mark.performance
    def test_end_to_end_performance_validation(self, mock_config_loader, performance_validator):
        """Test end-to-end performance validation per Section 6.6.4.3."""
        large_config = generate_baseline_config()
        
        # Create large configuration for stress testing
        for i in range(100):
            large_config["styles"][f"condition_{i}"] = {
                "color": f"#{i:06x}",
                "marker": "o",
                "linewidth": 1.0 + i % 5
            }
        
        kedro_config = generate_kedro_specific_config()
        mock_config_loader.get.return_value = kedro_config
        
        bridge = FigRegistryConfigBridge(
            config_loader=mock_config_loader,
            environment="performance_test"
        )
        
        with patch.object(bridge, '_load_figregistry_config', return_value=large_config):
            # Measure complete workflow performance
            start_time = time.perf_counter()
            merged_config = bridge.get_merged_config()
            end_time = time.perf_counter()
            
            total_time_ms = (end_time - start_time) * 1000
        
        # Validate against performance requirements
        performance_results = {
            "config_bridge_total": total_time_ms
        }
        
        validation_results = performance_validator["validate_config_bridge"](total_time_ms)
        assert validation_results, f"Configuration bridge took {total_time_ms:.2f}ms, exceeding 50ms target"
        
        # Verify functional correctness wasn't compromised
        assert isinstance(merged_config, FigRegistryKedroConfig)
        assert len(merged_config.styles) > 100  # Large config processed correctly


# =============================================================================
# BENCHMARK TESTS (if pytest-benchmark available)
# =============================================================================

if BENCHMARK_AVAILABLE:
    
    @pytest.mark.benchmark
    class TestConfigBridgeBenchmarks:
        """Benchmark tests for FigRegistryConfigBridge performance characteristics."""
        
        def test_benchmark_configuration_merging(self, benchmark, base_figregistry_config):
            """Benchmark configuration merging performance per Section 6.6.4.3."""
            bridge = FigRegistryConfigBridge()
            kedro_config = generate_kedro_specific_config()
            
            result = benchmark(
                bridge._merge_configurations,
                base_figregistry_config,
                kedro_config
            )
            
            # Verify benchmark result correctness
            assert "figregistry_version" in result
            assert "kedro" in result
            assert result["environment"] == "base"
        
        def test_benchmark_validation_performance(self, benchmark, base_figregistry_config):
            """Benchmark Pydantic validation performance."""
            bridge = FigRegistryConfigBridge()
            
            result = benchmark(
                bridge._validate_configuration,
                base_figregistry_config
            )
            
            assert isinstance(result, FigRegistryKedroConfig)
        
        def test_benchmark_cache_performance(self, benchmark, mock_config_loader, base_figregistry_config):
            """Benchmark cache performance for repeated access."""
            kedro_config = generate_kedro_specific_config()
            mock_config_loader.get.return_value = kedro_config
            
            bridge = FigRegistryConfigBridge(
                config_loader=mock_config_loader,
                enable_caching=True
            )
            
            with patch.object(bridge, '_load_figregistry_config', return_value=base_figregistry_config):
                # First call to populate cache
                bridge.get_merged_config()
                
                # Benchmark cached access
                result = benchmark(bridge.get_merged_config)
            
            assert isinstance(result, FigRegistryKedroConfig)


# =============================================================================
# MODULE-LEVEL TEST CONFIGURATION
# =============================================================================

# Configure pytest markers for test organization
pytestmark = [
    pytest.mark.kedro_plugin,
    pytest.mark.unit
]

# Test module metadata
TEST_MODULE_VERSION = "1.0.0"
TEST_COVERAGE_TARGET = 90.0  # Minimum 90% coverage required
CRITICAL_PATH_COVERAGE_TARGET = 100.0  # 100% for configuration merge operations

# Performance requirements from Section 6.6.4.3
PERFORMANCE_REQUIREMENTS = {
    "config_bridge_merge_time_ms": 50,
    "hook_initialization_time_ms": 25,
    "dataset_save_overhead_ms": 200
}

# Security testing constants from Section 6.6.8.1
SECURITY_VALIDATION_PATTERNS = [
    "!!python", "!!map", "!!set", "__import__",
    "../", "..\\", "%2e%2e%2f", "/etc/", "/root/"
]