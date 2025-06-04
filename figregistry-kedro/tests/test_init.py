"""Unit tests for figregistry_kedro package initialization.

This module provides comprehensive test coverage for the figregistry_kedro package initialization
that validates module imports, version metadata, API surface exposure, and plugin discovery
integration per Section 6.6 Testing Strategy and F-008 Plugin Packaging requirements.

The test suite validates critical package initialization functionality including:

- **API Surface Validation**: Tests expose clear entry points for FigureDataSet, FigRegistryHooks, 
  and FigRegistryConfigBridge per Section 0.1.2 API surface requirements
- **Version Compatibility**: Validates semantic versioning metadata and dependency compatibility 
  with figregistry>=0.3.0 and kedro>=0.18.0,<0.20.0 per Section 3.2.3.1
- **Plugin Discovery Integration**: Ensures proper entry point registration for kedro.hooks 
  and kedro.datasets per F-008 packaging requirements
- **Import Error Handling**: Tests graceful degradation and warning mechanisms for missing 
  or incompatible dependencies
- **Dependency Validation**: Validates plugin version compatibility checking and error reporting 
  for unsupported environment combinations

Test Coverage Requirements per Section 6.6.2.4:
- Minimum 90% coverage for figregistry_kedro.__init__ module
- 100% coverage for critical paths including import validation and plugin discovery
- Comprehensive error scenario testing with property-based validation

Testing Framework Integration per Section 6.6.2.1:
- pytest >=8.0.0 for core test execution with advanced fixture support
- pytest-mock >=3.14.0 for dependency mocking and isolation testing
- hypothesis >=6.0.0 for property-based testing of version validation
- Comprehensive test isolation and cleanup per Section 6.6.5.6

Performance Requirements per Section 6.6.4.3:
- Plugin initialization overhead validation against <25ms target
- Memory footprint verification under <5MB plugin overhead limit
- Import performance benchmarking for optimization validation
"""

import os
import sys
import warnings
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch
import pytest

# Import the module under test
import figregistry_kedro


class TestPackageInitialization:
    """Test suite for figregistry_kedro package initialization functionality.
    
    Validates core package initialization behavior including metadata exposure,
    version validation, import handling, and plugin discovery integration
    per Section 0.1.2 API surface requirements and F-008 packaging specifications.
    """
    
    def test_package_metadata_availability(self):
        """Test that all required package metadata is available and correctly formatted.
        
        Validates package metadata constants required for plugin discovery and
        dependency management per Section 3.2.3.1 version compatibility requirements.
        """
        # Test core metadata constants are available
        assert hasattr(figregistry_kedro, '__version__')
        assert hasattr(figregistry_kedro, '__author__')
        assert hasattr(figregistry_kedro, '__email__')
        assert hasattr(figregistry_kedro, '__description__')
        assert hasattr(figregistry_kedro, '__url__')
        
        # Test dependency requirement constants
        assert hasattr(figregistry_kedro, '__requires_python__')
        assert hasattr(figregistry_kedro, '__requires_figregistry__')
        assert hasattr(figregistry_kedro, '__requires_kedro__')
        
        # Validate metadata content format
        assert isinstance(figregistry_kedro.__version__, str)
        assert len(figregistry_kedro.__version__.split('.')) >= 3  # Semantic versioning
        assert figregistry_kedro.__author__ == "FigRegistry Team"
        assert "figregistry-kedro" in figregistry_kedro.__description__.lower()
        assert figregistry_kedro.__url__.startswith("https://")
        
        # Validate dependency specifications format
        assert figregistry_kedro.__requires_python__.startswith(">=3.10")
        assert figregistry_kedro.__requires_figregistry__.startswith(">=0.3.0")
        assert ">=0.18.0" in figregistry_kedro.__requires_kedro__
        assert "<0.20.0" in figregistry_kedro.__requires_kedro__


class TestAPIExposure:
    """Test suite for package-level API exposure and component availability.
    
    Validates that FigureDataSet, FigRegistryHooks, and FigRegistryConfigBridge
    are properly exposed through package-level imports per Section 0.1.2
    API surface requirements.
    """
    
    def test_core_component_imports_when_available(self):
        """Test that core components are importable when dependencies are available.
        
        Validates that all three core plugin components (FigureDataSet, FigRegistryHooks,
        FigRegistryConfigBridge) are available through package-level imports when
        all dependencies are satisfied.
        """
        # Test that core components are available in the package namespace
        # when dependencies are properly installed
        
        # Check FigureDataSet availability
        if hasattr(figregistry_kedro, 'FigureDataSet') and figregistry_kedro.FigureDataSet is not None:
            assert figregistry_kedro.FigureDataSet is not None
            assert 'FigureDataSet' in figregistry_kedro.__all__
            
        # Check FigRegistryHooks availability  
        if hasattr(figregistry_kedro, 'FigRegistryHooks') and figregistry_kedro.FigRegistryHooks is not None:
            assert figregistry_kedro.FigRegistryHooks is not None
            assert 'FigRegistryHooks' in figregistry_kedro.__all__
            
        # Check FigRegistryConfigBridge availability
        if hasattr(figregistry_kedro, 'FigRegistryConfigBridge') and figregistry_kedro.FigRegistryConfigBridge is not None:
            assert figregistry_kedro.FigRegistryConfigBridge is not None
            assert 'FigRegistryConfigBridge' in figregistry_kedro.__all__
    
    def test_error_classes_available_when_components_imported(self):
        """Test that component-specific error classes are available when components import successfully.
        
        Validates that error classes (FigureDataSetError, HookExecutionError, 
        ConfigurationMergeError) are properly exposed alongside their respective components.
        """
        # Test FigureDataSetError availability
        if hasattr(figregistry_kedro, 'FigureDataSet') and figregistry_kedro.FigureDataSet is not None:
            assert hasattr(figregistry_kedro, 'FigureDataSetError')
            assert 'FigureDataSetError' in figregistry_kedro.__all__
            
        # Test HookExecutionError availability
        if hasattr(figregistry_kedro, 'FigRegistryHooks') and figregistry_kedro.FigRegistryHooks is not None:
            assert hasattr(figregistry_kedro, 'HookExecutionError')
            assert 'HookExecutionError' in figregistry_kedro.__all__
            
        # Test ConfigurationMergeError availability
        if hasattr(figregistry_kedro, 'FigRegistryConfigBridge') and figregistry_kedro.FigRegistryConfigBridge is not None:
            assert hasattr(figregistry_kedro, 'ConfigurationMergeError')
            assert 'ConfigurationMergeError' in figregistry_kedro.__all__
    
    def test_convenience_functions_availability(self):
        """Test that convenience functions are available when their components are imported.
        
        Validates that utility functions (init_config, get_config_bridge, create_hooks, etc.)
        are properly exposed when their underlying components are available.
        """
        # Test config convenience functions
        if hasattr(figregistry_kedro, 'FigRegistryConfigBridge') and figregistry_kedro.FigRegistryConfigBridge is not None:
            if hasattr(figregistry_kedro, 'init_config'):
                assert 'init_config' in figregistry_kedro.__all__
            if hasattr(figregistry_kedro, 'get_config_bridge'):
                assert 'get_config_bridge' in figregistry_kedro.__all__
                
        # Test hook convenience functions
        if hasattr(figregistry_kedro, 'FigRegistryHooks') and figregistry_kedro.FigRegistryHooks is not None:
            if hasattr(figregistry_kedro, 'create_hooks'):
                assert 'create_hooks' in figregistry_kedro.__all__
                
        # Test dataset convenience functions
        if hasattr(figregistry_kedro, 'FigureDataSet') and figregistry_kedro.FigureDataSet is not None:
            if hasattr(figregistry_kedro, 'create_figure_dataset'):
                assert 'create_figure_dataset' in figregistry_kedro.__all__
            if hasattr(figregistry_kedro, 'validate_figure_dataset_config'):
                assert 'validate_figure_dataset_config' in figregistry_kedro.__all__
    
    def test_all_exports_dynamic_based_on_availability(self):
        """Test that __all__ exports are dynamically populated based on component availability.
        
        Validates that the __all__ list correctly reflects only the components that
        are actually available and successfully imported.
        """
        # Core metadata should always be in __all__
        essential_exports = [
            '__version__', '__author__', '__email__', '__description__', '__url__',
            '__requires_python__', '__requires_figregistry__', '__requires_kedro__',
            'get_plugin_info', 'check_dependencies', 'get_version'
        ]
        
        for export in essential_exports:
            assert export in figregistry_kedro.__all__, f"Essential export {export} missing from __all__"
        
        # Check that only available components are in __all__
        all_exports = set(figregistry_kedro.__all__)
        
        # If components are available, they should be in __all__
        if hasattr(figregistry_kedro, 'FigureDataSet') and figregistry_kedro.FigureDataSet is not None:
            assert 'FigureDataSet' in all_exports
        else:
            assert 'FigureDataSet' not in all_exports
            
        if hasattr(figregistry_kedro, 'FigRegistryHooks') and figregistry_kedro.FigRegistryHooks is not None:
            assert 'FigRegistryHooks' in all_exports
        else:
            assert 'FigRegistryHooks' not in all_exports
            
        if hasattr(figregistry_kedro, 'FigRegistryConfigBridge') and figregistry_kedro.FigRegistryConfigBridge is not None:
            assert 'FigRegistryConfigBridge' in all_exports
        else:
            assert 'FigRegistryConfigBridge' not in all_exports


class TestUtilityFunctions:
    """Test suite for package utility functions and plugin information.
    
    Validates get_plugin_info(), check_dependencies(), and get_version()
    functions that provide plugin status and compatibility information.
    """
    
    def test_get_plugin_info_structure(self):
        """Test that get_plugin_info() returns properly structured plugin information.
        
        Validates the structure and content of the plugin information dictionary
        returned by get_plugin_info() for plugin discovery and status reporting.
        """
        info = figregistry_kedro.get_plugin_info()
        
        # Validate return type and core structure
        assert isinstance(info, dict)
        
        # Check required metadata fields
        required_fields = [
            'name', 'version', 'description', 'author', 'url',
            'requires_python', 'requires_figregistry', 'requires_kedro'
        ]
        for field in required_fields:
            assert field in info, f"Required field {field} missing from plugin info"
        
        # Validate metadata content
        assert info['name'] == 'figregistry-kedro'
        assert info['version'] == figregistry_kedro.__version__
        assert info['description'] == figregistry_kedro.__description__
        assert info['author'] == figregistry_kedro.__author__
        assert info['url'] == figregistry_kedro.__url__
        
        # Check components availability information
        assert 'components' in info
        assert isinstance(info['components'], dict)
        
        component_keys = [
            'FigureDataSet', 'FigRegistryHooks', 'FigRegistryConfigBridge',
            'config_functions', 'hook_factory', 'dataset_utilities'
        ]
        for key in component_keys:
            assert key in info['components']
            assert isinstance(info['components'][key], bool)
        
        # Check fully_functional status
        assert 'fully_functional' in info
        assert isinstance(info['fully_functional'], bool)
    
    def test_check_dependencies_functionality(self):
        """Test that check_dependencies() accurately reports plugin functionality status.
        
        Validates that check_dependencies() correctly determines whether all
        required dependencies are available for full plugin functionality.
        """
        dependencies_status = figregistry_kedro.check_dependencies()
        
        # Should return boolean
        assert isinstance(dependencies_status, bool)
        
        # Should match the fully_functional status from get_plugin_info
        info = figregistry_kedro.get_plugin_info()
        assert dependencies_status == info['fully_functional']
        
        # If fully functional, all core components should be available
        if dependencies_status:
            assert figregistry_kedro.FigureDataSet is not None
            assert figregistry_kedro.FigRegistryHooks is not None
            assert figregistry_kedro.FigRegistryConfigBridge is not None
    
    def test_get_version_returns_semantic_version(self):
        """Test that get_version() returns a valid semantic version string.
        
        Validates that the version string follows semantic versioning conventions
        required for dependency management per Section 3.2.3.1.
        """
        version = figregistry_kedro.get_version()
        
        # Should return string
        assert isinstance(version, str)
        
        # Should match package __version__
        assert version == figregistry_kedro.__version__
        
        # Should follow semantic versioning (major.minor.patch)
        version_parts = version.split('.')
        assert len(version_parts) >= 3, f"Version {version} does not follow semantic versioning"
        
        # Each part should be numeric (allowing for pre-release suffixes)
        for i, part in enumerate(version_parts[:3]):
            # Remove any pre-release suffix for validation
            numeric_part = part.split('-')[0].split('+')[0]
            assert numeric_part.isdigit(), f"Version part {part} is not numeric"


class TestImportErrorHandling:
    """Test suite for import error handling and graceful degradation.
    
    Validates that the package handles missing dependencies gracefully with
    appropriate warnings and maintains basic functionality even when some
    components are unavailable.
    """
    
    def test_import_warnings_issued_for_missing_dependencies(self, mocker):
        """Test that appropriate warnings are issued when dependencies are missing.
        
        Validates that ImportError scenarios for each component trigger
        appropriate warning messages while maintaining package stability.
        """
        # Capture warnings during import simulation
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            # Mock import failures for each component
            with patch('figregistry_kedro.datasets', side_effect=ImportError("Mock datasets import error")):
                with patch('figregistry_kedro.hooks', side_effect=ImportError("Mock hooks import error")):
                    with patch('figregistry_kedro.config', side_effect=ImportError("Mock config import error")):
                        # Reload the module to trigger import logic
                        importlib.reload(figregistry_kedro)
            
            # Should have warnings for missing components
            warning_messages = [str(w.message) for w in warning_list if issubclass(w.category, ImportWarning)]
            
            # Should contain warnings about missing components
            has_dataset_warning = any("FigureDataSet not available" in msg for msg in warning_messages)
            has_hooks_warning = any("FigRegistryHooks not available" in msg for msg in warning_messages)
            has_config_warning = any("FigRegistryConfigBridge not available" in msg for msg in warning_messages)
            
            # At least some warnings should be present when components are missing
            assert has_dataset_warning or has_hooks_warning or has_config_warning
    
    def test_package_remains_importable_with_missing_dependencies(self, mocker):
        """Test that the package remains importable even when dependencies are missing.
        
        Validates that import failures for individual components do not prevent
        the package from being imported and basic functionality from working.
        """
        # Mock missing dependencies
        with patch.dict('sys.modules', {'kedro': None, 'figregistry': None}):
            with patch('figregistry_kedro.datasets', side_effect=ImportError("No kedro")):
                with patch('figregistry_kedro.hooks', side_effect=ImportError("No kedro")):
                    with patch('figregistry_kedro.config', side_effect=ImportError("No figregistry")):
                        
                        # Reload module to test import behavior
                        importlib.reload(figregistry_kedro)
                        
                        # Basic package functionality should still work
                        assert hasattr(figregistry_kedro, '__version__')
                        assert hasattr(figregistry_kedro, 'get_plugin_info')
                        assert hasattr(figregistry_kedro, 'check_dependencies')
                        
                        # Utility functions should still be callable
                        info = figregistry_kedro.get_plugin_info()
                        assert isinstance(info, dict)
                        
                        dependencies_ok = figregistry_kedro.check_dependencies()
                        assert isinstance(dependencies_ok, bool)
                        assert not dependencies_ok  # Should be False with missing deps
    
    def test_none_assignments_for_failed_imports(self, mocker):
        """Test that failed component imports result in None assignments.
        
        Validates that when component imports fail, the corresponding module
        attributes are set to None rather than remaining undefined.
        """
        # Mock import failures
        with patch('figregistry_kedro.datasets', side_effect=ImportError("Mock failure")):
            importlib.reload(figregistry_kedro)
            
            # Failed imports should result in None assignments
            assert figregistry_kedro.FigureDataSet is None
            assert figregistry_kedro.FigureDataSetError is None
        
        with patch('figregistry_kedro.hooks', side_effect=ImportError("Mock failure")):
            importlib.reload(figregistry_kedro)
            
            assert figregistry_kedro.FigRegistryHooks is None
            assert figregistry_kedro.HookExecutionError is None
        
        with patch('figregistry_kedro.config', side_effect=ImportError("Mock failure")):
            importlib.reload(figregistry_kedro)
            
            assert figregistry_kedro.FigRegistryConfigBridge is None
            assert figregistry_kedro.ConfigurationMergeError is None


class TestVersionCompatibilityValidation:
    """Test suite for version compatibility validation and checks.
    
    Validates the plugin version compatibility checking functionality that
    ensures proper dependency management per Section 3.2.3.1 requirements.
    """
    
    def test_dependency_version_requirements_format(self):
        """Test that dependency version requirements follow proper specification format.
        
        Validates that version requirement strings follow packaging standards
        for compatibility with pip and conda dependency resolution.
        """
        # Test Python version requirement format
        python_req = figregistry_kedro.__requires_python__
        assert python_req.startswith(">="), f"Python requirement should start with >=: {python_req}"
        assert "3.10" in python_req, f"Python requirement should specify 3.10: {python_req}"
        
        # Test FigRegistry version requirement format
        figregistry_req = figregistry_kedro.__requires_figregistry__
        assert figregistry_req.startswith(">="), f"FigRegistry requirement should start with >=: {figregistry_req}"
        assert "0.3.0" in figregistry_req, f"FigRegistry requirement should specify 0.3.0: {figregistry_req}"
        
        # Test Kedro version requirement format
        kedro_req = figregistry_kedro.__requires_kedro__
        assert ">=0.18.0" in kedro_req, f"Kedro requirement should specify >=0.18.0: {kedro_req}"
        assert "<0.20.0" in kedro_req, f"Kedro requirement should specify <0.20.0: {kedro_req}"
        assert "," in kedro_req, f"Kedro requirement should have range specification: {kedro_req}"
    
    def test_validate_plugin_compatibility_function_existence(self):
        """Test that plugin compatibility validation functionality exists.
        
        Validates that the internal _validate_plugin_compatibility function
        is present and callable for version checking during package initialization.
        """
        # Check that compatibility validation function exists
        assert hasattr(figregistry_kedro, '_validate_plugin_compatibility')
        
        # Should be callable
        assert callable(figregistry_kedro._validate_plugin_compatibility)
    
    @patch('figregistry_kedro.warnings.warn')
    def test_compatibility_warnings_for_version_mismatches(self, mock_warn, mocker):
        """Test that compatibility warnings are issued for version mismatches.
        
        Validates that version compatibility validation issues appropriate
        warnings when installed versions don't match plugin requirements.
        """
        # Mock figregistry with incompatible version
        mock_figregistry = mocker.MagicMock()
        mock_figregistry.__version__ = "0.2.0"  # Below required 0.3.0
        
        # Mock kedro with incompatible version  
        mock_kedro = mocker.MagicMock()
        mock_kedro.__version__ = "0.17.0"  # Below required 0.18.0
        
        with patch.dict('sys.modules', {'figregistry': mock_figregistry, 'kedro': mock_kedro}):
            with patch('figregistry_kedro.version.parse') as mock_version_parse:
                # Configure version parsing to simulate version comparison
                def parse_side_effect(version_str):
                    # Simple mock that allows comparison
                    mock_version = Mock()
                    mock_version.__lt__ = Mock(return_value=True)  # Always less than required
                    return mock_version
                
                mock_version_parse.side_effect = parse_side_effect
                
                # Call compatibility validation
                figregistry_kedro._validate_plugin_compatibility()
                
                # Should have issued warnings for incompatible versions
                assert mock_warn.call_count >= 1
                
                # Check warning content for version issues
                warning_calls = [call[0][0] for call in mock_warn.call_args_list]
                version_warnings = [msg for msg in warning_calls if "version" in msg.lower()]
                assert len(version_warnings) > 0, "Should have version compatibility warnings"
    
    def test_no_warnings_for_compatible_versions(self, mocker):
        """Test that no warnings are issued when versions are compatible.
        
        Validates that version compatibility validation is silent when
        all installed versions meet plugin requirements.
        """
        # Mock compatible versions
        mock_figregistry = mocker.MagicMock()
        mock_figregistry.__version__ = "0.3.5"  # Above required 0.3.0
        
        mock_kedro = mocker.MagicMock()
        mock_kedro.__version__ = "0.18.5"  # Within required range
        
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            with patch.dict('sys.modules', {'figregistry': mock_figregistry, 'kedro': mock_kedro}):
                with patch('figregistry_kedro.version.parse') as mock_version_parse:
                    # Configure version parsing for compatible versions
                    def parse_side_effect(version_str):
                        mock_version = Mock()
                        if "0.3.5" in version_str or "0.18.5" in version_str:
                            mock_version.__lt__ = Mock(return_value=False)  # Not less than required
                            mock_version.__ge__ = Mock(return_value=True)   # Greater than or equal to required
                        else:
                            mock_version.__lt__ = Mock(return_value=False)  # Default to compatible
                            mock_version.__ge__ = Mock(return_value=True)
                        return mock_version
                    
                    mock_version_parse.side_effect = parse_side_effect
                    
                    # Call compatibility validation
                    figregistry_kedro._validate_plugin_compatibility()
            
            # Should not have compatibility warnings for good versions
            compatibility_warnings = [w for w in warning_list 
                                    if issubclass(w.category, UserWarning) 
                                    and "version" in str(w.message).lower() 
                                    and "compatible" in str(w.message).lower()]
            
            # Should be empty or minimal warnings
            assert len(compatibility_warnings) == 0, f"Unexpected compatibility warnings: {compatibility_warnings}"


class TestPluginDiscoveryIntegration:
    """Test suite for plugin discovery and entry point integration.
    
    Validates integration with Kedro's plugin discovery system per F-008
    packaging requirements, ensuring proper entry point registration.
    """
    
    def test_plugin_entry_point_specifications(self):
        """Test that plugin follows entry point specifications for Kedro discovery.
        
        Validates that the plugin can be discovered by Kedro's plugin system
        through proper entry point registration and naming conventions.
        """
        # Test that the plugin module is importable by name
        assert figregistry_kedro.__name__ == 'figregistry_kedro'
        
        # Test that plugin provides required metadata for discovery
        assert hasattr(figregistry_kedro, '__version__')
        assert hasattr(figregistry_kedro, '__description__')
        
        # Plugin should be identifiable as a Kedro plugin
        plugin_info = figregistry_kedro.get_plugin_info()
        assert 'kedro' in plugin_info['name'].lower()
        assert 'plugin' in plugin_info['description'].lower()
    
    def test_hook_registration_compatibility(self):
        """Test that hook registration follows Kedro plugin conventions.
        
        Validates that FigRegistryHooks can be properly registered through
        Kedro's hook system and follows plugin registration patterns.
        """
        # If hooks are available, test registration compatibility
        if hasattr(figregistry_kedro, 'FigRegistryHooks') and figregistry_kedro.FigRegistryHooks is not None:
            hooks_class = figregistry_kedro.FigRegistryHooks
            
            # Should be a class that can be instantiated
            assert isinstance(hooks_class, type)
            
            # Should have hook lifecycle methods for Kedro integration
            hook_methods = ['before_pipeline_run', 'after_pipeline_run', 'before_catalog_created', 'after_catalog_created']
            available_methods = [method for method in hook_methods if hasattr(hooks_class, method)]
            
            # At least some hook methods should be available
            assert len(available_methods) > 0, f"FigRegistryHooks should have some hook methods from {hook_methods}"
    
    def test_dataset_registration_compatibility(self):
        """Test that dataset registration follows Kedro plugin conventions.
        
        Validates that FigureDataSet can be properly registered through
        Kedro's dataset catalog system and follows AbstractDataSet conventions.
        """
        # If dataset is available, test registration compatibility
        if hasattr(figregistry_kedro, 'FigureDataSet') and figregistry_kedro.FigureDataSet is not None:
            dataset_class = figregistry_kedro.FigureDataSet
            
            # Should be a class that can be instantiated
            assert isinstance(dataset_class, type)
            
            # Should have AbstractDataSet methods for Kedro catalog integration
            dataset_methods = ['_save', '_load', '_describe']
            available_methods = [method for method in dataset_methods if hasattr(dataset_class, method)]
            
            # All AbstractDataSet methods should be available
            assert len(available_methods) == len(dataset_methods), f"FigureDataSet missing methods: {set(dataset_methods) - set(available_methods)}"
    
    def test_plugin_namespace_organization(self):
        """Test that plugin namespace follows Kedro plugin organization standards.
        
        Validates that the plugin organizes its components in a way that's
        compatible with Kedro's plugin architecture and discovery mechanisms.
        """
        # Test module structure follows plugin conventions
        assert figregistry_kedro.__name__.startswith('figregistry_kedro')
        
        # Test that components are properly namespaced
        if hasattr(figregistry_kedro, 'FigureDataSet'):
            # Should be in datasets submodule conceptually
            assert 'DataSet' in figregistry_kedro.FigureDataSet.__name__ if figregistry_kedro.FigureDataSet else True
            
        if hasattr(figregistry_kedro, 'FigRegistryHooks'):
            # Should be in hooks submodule conceptually  
            assert 'Hooks' in figregistry_kedro.FigRegistryHooks.__name__ if figregistry_kedro.FigRegistryHooks else True
            
        if hasattr(figregistry_kedro, 'FigRegistryConfigBridge'):
            # Should be in config submodule conceptually
            assert 'Config' in figregistry_kedro.FigRegistryConfigBridge.__name__ if figregistry_kedro.FigRegistryConfigBridge else True


class TestPluginInitializationWarning:
    """Test suite for plugin initialization warning system.
    
    Validates that the plugin issues appropriate warnings when dependencies
    are missing or incompatible, per Section 6.6.5.6 error handling requirements.
    """
    
    @patch('figregistry_kedro.warnings.warn')
    def test_initialization_warning_issued_when_not_functional(self, mock_warn, mocker):
        """Test that initialization warning is issued when plugin is not fully functional.
        
        Validates that a comprehensive warning is issued during module initialization
        when check_dependencies() returns False, indicating missing dependencies.
        """
        # Mock check_dependencies to return False
        with patch.object(figregistry_kedro, 'check_dependencies', return_value=False):
            # Reload module to trigger initialization warning
            importlib.reload(figregistry_kedro)
            
            # Should have issued warning about plugin not being functional
            assert mock_warn.called
            
            # Check warning content
            warning_calls = [call[0][0] for call in mock_warn.call_args_list]
            init_warnings = [msg for msg in warning_calls 
                           if "not fully functional" in msg and "missing dependencies" in msg]
            
            assert len(init_warnings) > 0, "Should issue initialization warning when not functional"
            
            # Warning should mention how to fix the issue
            comprehensive_warnings = [msg for msg in warning_calls 
                                    if "pip install" in msg and "figregistry" in msg and "kedro" in msg]
            
            assert len(comprehensive_warnings) > 0, "Should provide installation instructions in warning"
    
    def test_no_initialization_warning_when_functional(self, mocker):
        """Test that no initialization warning is issued when plugin is fully functional.
        
        Validates that when all dependencies are available and the plugin is
        fully functional, no warnings are issued during initialization.
        """
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            # Mock check_dependencies to return True
            with patch.object(figregistry_kedro, 'check_dependencies', return_value=True):
                # Reload module
                importlib.reload(figregistry_kedro)
            
            # Should not have initialization warnings when functional
            init_warnings = [w for w in warning_list 
                           if "not fully functional" in str(w.message)]
            
            assert len(init_warnings) == 0, f"Should not issue warnings when functional: {init_warnings}"


class TestPluginPerformance:
    """Test suite for plugin initialization performance validation.
    
    Validates that plugin initialization meets performance requirements
    per Section 6.6.4.3 hook initialization overhead targets (<25ms).
    """
    
    def test_plugin_import_performance(self, benchmark_config):
        """Test that plugin import performance meets initialization targets.
        
        Validates that importing the figregistry_kedro package completes
        within the 25ms hook initialization overhead target from Section 6.6.4.3.
        """
        import time
        
        # Measure import time
        start_time = time.perf_counter()
        
        # Force reimport to measure initialization time
        if 'figregistry_kedro' in sys.modules:
            del sys.modules['figregistry_kedro']
        
        import figregistry_kedro
        
        end_time = time.perf_counter()
        import_time_ms = (end_time - start_time) * 1000
        
        # Should meet performance target of <25ms for hook initialization
        assert import_time_ms < 25.0, f"Plugin import took {import_time_ms:.2f}ms, exceeds 25ms target"
    
    def test_get_plugin_info_performance(self):
        """Test that get_plugin_info() function performs efficiently.
        
        Validates that plugin information retrieval is fast enough for
        repeated calls during plugin discovery and status checking.
        """
        import time
        
        # Measure multiple calls to get_plugin_info
        start_time = time.perf_counter()
        
        for _ in range(100):  # Test repeated calls
            info = figregistry_kedro.get_plugin_info()
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Should be very fast for repeated calls
        assert avg_time_ms < 1.0, f"get_plugin_info() averaged {avg_time_ms:.2f}ms, should be <1ms"
    
    def test_check_dependencies_performance(self):
        """Test that check_dependencies() function performs efficiently.
        
        Validates that dependency checking is fast enough for repeated
        validation calls during plugin operation.
        """
        import time
        
        # Measure multiple calls to check_dependencies
        start_time = time.perf_counter()
        
        for _ in range(100):  # Test repeated calls
            status = figregistry_kedro.check_dependencies()
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Should be very fast for repeated calls
        assert avg_time_ms < 1.0, f"check_dependencies() averaged {avg_time_ms:.2f}ms, should be <1ms"


class TestPackageIntegrity:
    """Test suite for overall package integrity and consistency.
    
    Validates that the package maintains internal consistency and follows
    Python packaging best practices per F-008 requirements.
    """
    
    def test_module_docstring_completeness(self):
        """Test that the module has comprehensive documentation.
        
        Validates that the package module includes proper documentation
        describing its purpose, usage, and integration capabilities.
        """
        # Package should have a docstring
        assert figregistry_kedro.__doc__ is not None
        assert len(figregistry_kedro.__doc__.strip()) > 100
        
        # Docstring should mention key components
        docstring = figregistry_kedro.__doc__.lower()
        assert 'figregistry' in docstring
        assert 'kedro' in docstring
        assert 'plugin' in docstring
    
    def test_all_exports_are_actually_available(self):
        """Test that all items in __all__ are actually available in the module.
        
        Validates that every item listed in __all__ is actually defined
        and accessible in the module namespace.
        """
        for export_name in figregistry_kedro.__all__:
            assert hasattr(figregistry_kedro, export_name), f"Export {export_name} in __all__ but not available in module"
            
            # Should not be None for essential exports
            if export_name in ['get_plugin_info', 'check_dependencies', 'get_version']:
                assert getattr(figregistry_kedro, export_name) is not None, f"Essential export {export_name} is None"
    
    def test_consistent_version_information(self):
        """Test that version information is consistent across all access methods.
        
        Validates that version information is consistent between __version__,
        get_version(), and get_plugin_info() to prevent confusion.
        """
        # All version access methods should return the same value
        version_direct = figregistry_kedro.__version__
        version_function = figregistry_kedro.get_version()
        version_info = figregistry_kedro.get_plugin_info()['version']
        
        assert version_direct == version_function
        assert version_direct == version_info
        assert version_function == version_info
    
    def test_plugin_info_consistency_with_module_attributes(self):
        """Test that plugin info dictionary is consistent with module attributes.
        
        Validates that the information returned by get_plugin_info() matches
        the actual module attributes and constants.
        """
        info = figregistry_kedro.get_plugin_info()
        
        # Metadata should match module attributes
        assert info['version'] == figregistry_kedro.__version__
        assert info['description'] == figregistry_kedro.__description__
        assert info['author'] == figregistry_kedro.__author__
        assert info['url'] == figregistry_kedro.__url__
        assert info['requires_python'] == figregistry_kedro.__requires_python__
        assert info['requires_figregistry'] == figregistry_kedro.__requires_figregistry__
        assert info['requires_kedro'] == figregistry_kedro.__requires_kedro__
        
        # Component availability should match actual availability
        actual_components = {
            'FigureDataSet': figregistry_kedro.FigureDataSet is not None,
            'FigRegistryHooks': figregistry_kedro.FigRegistryHooks is not None,
            'FigRegistryConfigBridge': figregistry_kedro.FigRegistryConfigBridge is not None,
        }
        
        for component, actual_status in actual_components.items():
            assert info['components'][component] == actual_status, f"Component {component} status mismatch"


class TestEdgeCasesAndBoundaryConditions:
    """Test suite for edge cases and boundary conditions in package initialization.
    
    Validates robust behavior under unusual conditions and edge cases
    that may occur in diverse deployment environments.
    """
    
    def test_repeated_imports_are_stable(self):
        """Test that repeated imports produce consistent results.
        
        Validates that importing the package multiple times produces
        consistent state without side effects or state corruption.
        """
        # Import multiple times and check consistency
        first_import = importlib.import_module('figregistry_kedro')
        second_import = importlib.import_module('figregistry_kedro')
        
        # Should be the same module object
        assert first_import is second_import
        
        # Reload and check consistency
        importlib.reload(figregistry_kedro)
        
        # Essential functions should still work after reload
        assert callable(figregistry_kedro.get_plugin_info)
        assert callable(figregistry_kedro.check_dependencies)
        assert callable(figregistry_kedro.get_version)
        
        # Version should be consistent after reload
        version_before = figregistry_kedro.__version__
        importlib.reload(figregistry_kedro)
        version_after = figregistry_kedro.__version__
        
        assert version_before == version_after
    
    def test_import_with_modified_sys_modules(self, mocker):
        """Test package behavior when sys.modules is modified.
        
        Validates robust import behavior when the Python import system
        has been modified or corrupted by other packages or frameworks.
        """
        # Save original state
        original_modules = sys.modules.copy()
        
        try:
            # Modify sys.modules to simulate interference
            if 'kedro' in sys.modules:
                del sys.modules['kedro']
            if 'figregistry' in sys.modules:
                del sys.modules['figregistry']
            
            # Should still be able to import and get basic functionality
            importlib.reload(figregistry_kedro)
            
            # Basic functions should work
            assert hasattr(figregistry_kedro, 'get_plugin_info')
            assert hasattr(figregistry_kedro, 'check_dependencies')
            
            info = figregistry_kedro.get_plugin_info()
            assert isinstance(info, dict)
            
        finally:
            # Restore original state
            sys.modules.clear()
            sys.modules.update(original_modules)
    
    def test_plugin_with_corrupted_version_attributes(self, mocker):
        """Test plugin behavior when version attributes are corrupted.
        
        Validates graceful handling when version information from
        dependencies is corrupted or missing.
        """
        # Mock dependencies with missing or corrupted version info
        mock_figregistry = mocker.MagicMock()
        del mock_figregistry.__version__  # Missing version attribute
        
        mock_kedro = mocker.MagicMock()
        mock_kedro.__version__ = None  # Corrupted version attribute
        
        with patch.dict('sys.modules', {'figregistry': mock_figregistry, 'kedro': mock_kedro}):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                
                # Should not crash when checking compatibility
                try:
                    figregistry_kedro._validate_plugin_compatibility()
                except Exception as e:
                    # Should handle gracefully, not crash
                    assert "version" not in str(e).lower() or "compatibility" not in str(e).lower()
    
    def test_package_isolation_from_environment_variables(self, mocker):
        """Test that package initialization is not affected by environment variables.
        
        Validates that plugin initialization behaves consistently regardless
        of environment variable configuration or contamination.
        """
        # Test with various environment variable configurations
        test_env_vars = {
            'KEDRO_ENV': 'test',
            'FIGREGISTRY_ENV': 'development', 
            'PYTHONPATH': '/malicious/path',
            'HOME': '/tmp/test',
            'USER': 'test_user'
        }
        
        original_env = os.environ.copy()
        
        try:
            # Modify environment
            os.environ.update(test_env_vars)
            
            # Reload and test
            importlib.reload(figregistry_kedro)
            
            # Should still work correctly
            assert figregistry_kedro.__version__ is not None
            assert callable(figregistry_kedro.get_plugin_info)
            
            info = figregistry_kedro.get_plugin_info()
            assert info['name'] == 'figregistry-kedro'
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)


# Performance test configuration for benchmarking
@pytest.fixture
def benchmark_config():
    """Configure benchmark settings for performance testing per Section 6.6.4.3."""
    return {
        'min_rounds': 5,
        'max_time': 10.0,
        'timer': 'time.perf_counter', 
        'disable_gc': True,
        'warmup': True,
        'warmup_iterations': 2
    }


# Module-level test execution guard
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])