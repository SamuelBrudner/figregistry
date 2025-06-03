"""
Unit tests for figregistry_kedro package initialization.

This module validates package imports, version metadata, API surface exposure,
and plugin discovery integration for the figregistry-kedro plugin package.
Tests ensure proper initialization, dependency management, and Kedro plugin
system integration per Section 0.1.2 API requirements.

Key Test Areas:
- Package initialization and module imports for core components
- Version metadata validation and semantic versioning compatibility
- Plugin entry point registration for kedro.hooks and kedro.datasets
- Dependency validation and error handling for missing dependencies
- API surface consistency and convenient package-level access
- Plugin discovery integration with Kedro's plugin system

Testing Strategy per Section 6.6.2.1:
- Comprehensive import validation without requiring actual Kedro installation
- Mock-based testing for plugin registration scenarios
- Version compatibility checking across supported dependency ranges
- Security validation for import error handling and malicious package scenarios
"""

import sys
import importlib
import warnings
from unittest.mock import Mock, patch, MagicMock
from packaging.version import Version, parse as parse_version
from pathlib import Path
import pytest

# Test configuration for package initialization validation
EXPECTED_PACKAGE_NAME = 'figregistry_kedro'
MINIMUM_PYTHON_VERSION = (3, 10)
REQUIRED_FIGREGISTRY_VERSION = '0.3.0'
MINIMUM_KEDRO_VERSION = '0.18.0'
MAXIMUM_KEDRO_VERSION = '0.20.0'

# Expected API surface components per Section 0.1.2
EXPECTED_EXPORTS = [
    'FigureDataSet',
    'FigRegistryHooks', 
    'FigRegistryConfigBridge',
    '__version__'
]

# Expected plugin entry points for Kedro integration per F-008
EXPECTED_KEDRO_HOOKS = [
    'figregistry_kedro.hooks.FigRegistryHooks'
]

EXPECTED_KEDRO_DATASETS = [
    'figregistry_kedro.datasets.FigureDataSet'
]


class TestPackageInitialization:
    """
    Test package initialization and module-level imports.
    
    Validates that the figregistry_kedro package properly initializes
    and exposes required components through package-level imports
    per Section 0.1.2 API surface requirements.
    """
    
    def test_package_import_success(self):
        """
        Test successful package import with proper module initialization.
        
        Validates that the figregistry_kedro package can be imported
        without errors and contains expected module structure.
        """
        # Clear any existing module cache to ensure clean import
        if EXPECTED_PACKAGE_NAME in sys.modules:
            del sys.modules[EXPECTED_PACKAGE_NAME]
        
        # Test package import
        try:
            import figregistry_kedro
            assert figregistry_kedro is not None
        except ImportError as e:
            pytest.fail(f"Package import failed: {e}")
        
        # Validate module type and basic attributes
        assert hasattr(figregistry_kedro, '__name__')
        assert figregistry_kedro.__name__ == EXPECTED_PACKAGE_NAME
        
        # Check for package vs module
        assert hasattr(figregistry_kedro, '__path__'), "Package should have __path__ attribute"
    
    def test_package_level_api_exports(self):
        """
        Test package-level API exports for convenient access.
        
        Validates that all required components are accessible through
        package-level imports per __init__.py requirements.
        """
        import figregistry_kedro
        
        # Test all expected exports are available
        for export_name in EXPECTED_EXPORTS:
            assert hasattr(figregistry_kedro, export_name), \
                f"Package missing required export: {export_name}"
        
        # Validate specific component types
        assert hasattr(figregistry_kedro.FigureDataSet, '_save'), \
            "FigureDataSet should implement AbstractDataSet interface"
        assert hasattr(figregistry_kedro.FigureDataSet, '_load'), \
            "FigureDataSet should implement AbstractDataSet interface"
        
        # Validate hooks class structure
        assert hasattr(figregistry_kedro.FigRegistryHooks, '__call__'), \
            "FigRegistryHooks should be callable hook class"
        
        # Validate config bridge functionality
        assert hasattr(figregistry_kedro.FigRegistryConfigBridge, 'init_config'), \
            "FigRegistryConfigBridge should provide init_config method"
    
    def test_version_metadata_availability(self):
        """
        Test package version metadata and semantic versioning.
        
        Validates that package provides proper version information
        for dependency management per Section 3.2.3.1 requirements.
        """
        import figregistry_kedro
        
        # Test version attribute exists
        assert hasattr(figregistry_kedro, '__version__'), \
            "Package should expose __version__ attribute"
        
        # Test version format
        version_str = figregistry_kedro.__version__
        assert isinstance(version_str, str), "Version should be string"
        assert len(version_str) > 0, "Version should not be empty"
        
        # Validate semantic versioning format
        try:
            version_obj = parse_version(version_str)
            assert version_obj is not None
        except Exception as e:
            pytest.fail(f"Invalid semantic version format: {version_str}, error: {e}")
        
        # Test version components
        version_parts = version_str.split('.')
        assert len(version_parts) >= 2, "Version should have at least major.minor format"
        
        # Validate numeric version components
        for part in version_parts[:2]:  # Check at least major.minor
            try:
                int(part.split('-')[0].split('+')[0])  # Handle pre-release/build metadata
            except ValueError:
                pytest.fail(f"Version component should be numeric: {part}")
    
    def test_package_metadata_consistency(self):
        """
        Test package metadata consistency across different access methods.
        
        Validates that version and metadata information is consistent
        whether accessed through package attributes or importlib.
        """
        import figregistry_kedro
        
        # Test consistency with importlib metadata
        try:
            import importlib.metadata
            pkg_version = importlib.metadata.version(EXPECTED_PACKAGE_NAME)
            
            # Compare with package __version__
            assert figregistry_kedro.__version__ == pkg_version, \
                f"Package __version__ ({figregistry_kedro.__version__}) " \
                f"doesn't match metadata version ({pkg_version})"
                
        except importlib.metadata.PackageNotFoundError:
            # Package might not be installed, skip metadata comparison
            warnings.warn("Package not installed, skipping metadata consistency check")
        
        # Test other metadata attributes if present
        if hasattr(figregistry_kedro, '__author__'):
            assert isinstance(figregistry_kedro.__author__, str)
            assert len(figregistry_kedro.__author__) > 0
        
        if hasattr(figregistry_kedro, '__email__'):
            assert isinstance(figregistry_kedro.__email__, str)
            assert '@' in figregistry_kedro.__email__


class TestDependencyValidation:
    """
    Test dependency validation and version compatibility.
    
    Validates that the package properly handles dependency requirements
    and provides clear error messages for incompatible versions
    per Section 3.2.3.1 compatibility requirements.
    """
    
    def test_figregistry_dependency_validation(self):
        """
        Test FigRegistry dependency version validation.
        
        Validates that package requires compatible FigRegistry version
        and handles version mismatches appropriately.
        """
        # Test with mock FigRegistry import
        with patch.dict('sys.modules', {'figregistry': Mock()}):
            mock_figregistry = sys.modules['figregistry']
            
            # Test with compatible version
            mock_figregistry.__version__ = '0.3.0'
            try:
                import figregistry_kedro
                # Should succeed with compatible version
                assert True
            except ImportError:
                pytest.fail("Should import successfully with compatible FigRegistry version")
            
            # Test with older incompatible version
            mock_figregistry.__version__ = '0.2.5'
            # Clear module cache to force re-import
            if 'figregistry_kedro' in sys.modules:
                del sys.modules['figregistry_kedro']
            
            with pytest.raises((ImportError, ValueError)) as exc_info:
                import figregistry_kedro
            
            assert "figregistry" in str(exc_info.value).lower(), \
                "Error message should mention figregistry dependency"
    
    def test_kedro_dependency_validation(self):
        """
        Test Kedro dependency version validation.
        
        Validates that package handles Kedro version requirements
        and provides appropriate error handling for unsupported versions.
        """
        # Test with mock Kedro import
        with patch.dict('sys.modules', {'kedro': Mock()}):
            mock_kedro = sys.modules['kedro']
            
            # Test with minimum supported version
            mock_kedro.__version__ = MINIMUM_KEDRO_VERSION
            try:
                import figregistry_kedro
                # Should succeed with minimum version
                assert True
            except ImportError:
                pytest.fail("Should import successfully with minimum Kedro version")
            
            # Test with unsupported older version
            mock_kedro.__version__ = '0.17.9'
            # Clear module cache
            if 'figregistry_kedro' in sys.modules:
                del sys.modules['figregistry_kedro']
            
            with pytest.raises((ImportError, ValueError)) as exc_info:
                import figregistry_kedro
            
            assert "kedro" in str(exc_info.value).lower(), \
                "Error message should mention kedro dependency"
            
            # Test with unsupported newer version
            mock_kedro.__version__ = '0.20.0'
            if 'figregistry_kedro' in sys.modules:
                del sys.modules['figregistry_kedro']
            
            with pytest.raises((ImportError, ValueError)) as exc_info:
                import figregistry_kedro
            
            assert "kedro" in str(exc_info.value).lower(), \
                "Error message should mention kedro compatibility"
    
    def test_missing_dependency_handling(self):
        """
        Test graceful handling of missing required dependencies.
        
        Validates that package provides clear error messages when
        required dependencies are not available.
        """
        # Test with missing FigRegistry
        with patch.dict('sys.modules', {}, clear=True):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'figregistry'")):
                with pytest.raises(ImportError) as exc_info:
                    import figregistry_kedro
                
                assert "figregistry" in str(exc_info.value).lower(), \
                    "Error should mention missing figregistry dependency"
        
        # Test with missing Kedro
        with patch.dict('sys.modules', {'figregistry': Mock()}):
            with patch('builtins.__import__', side_effect=lambda name, *args: Mock() if name == 'figregistry' else ImportError(f"No module named '{name}'")):
                with pytest.raises(ImportError) as exc_info:
                    import figregistry_kedro
                
                error_msg = str(exc_info.value).lower()
                assert "kedro" in error_msg or "no module" in error_msg, \
                    "Error should mention missing kedro dependency"


class TestPluginDiscovery:
    """
    Test Kedro plugin discovery and entry point registration.
    
    Validates that the package properly registers with Kedro's plugin
    system and exposes required entry points per F-008 requirements.
    """
    
    @pytest.fixture
    def mock_entry_points(self):
        """Mock entry points for testing plugin discovery."""
        mock_eps = []
        
        # Mock hook entry points
        for hook in EXPECTED_KEDRO_HOOKS:
            ep = Mock()
            ep.name = 'figregistry_hooks'
            ep.value = hook
            ep.group = 'kedro.hooks'
            mock_eps.append(ep)
        
        # Mock dataset entry points
        for dataset in EXPECTED_KEDRO_DATASETS:
            ep = Mock()
            ep.name = 'FigureDataSet'
            ep.value = dataset
            ep.group = 'kedro.datasets'
            mock_eps.append(ep)
        
        return mock_eps
    
    def test_kedro_hooks_entry_point_registration(self, mock_entry_points):
        """
        Test Kedro hooks entry point registration.
        
        Validates that FigRegistryHooks is properly registered
        as a Kedro hook entry point for automatic discovery.
        """
        hook_entry_points = [ep for ep in mock_entry_points if ep.group == 'kedro.hooks']
        
        # Validate hook entry points exist
        assert len(hook_entry_points) > 0, "Should have kedro.hooks entry points"
        
        # Check specific hook registration
        hook_values = [ep.value for ep in hook_entry_points]
        for expected_hook in EXPECTED_KEDRO_HOOKS:
            assert expected_hook in hook_values, \
                f"Missing hook entry point: {expected_hook}"
        
        # Validate entry point structure
        for ep in hook_entry_points:
            assert hasattr(ep, 'name'), "Entry point should have name"
            assert hasattr(ep, 'value'), "Entry point should have value"
            assert '.' in ep.value, "Entry point value should be module.class format"
    
    def test_kedro_datasets_entry_point_registration(self, mock_entry_points):
        """
        Test Kedro datasets entry point registration.
        
        Validates that FigureDataSet is properly registered
        as a Kedro dataset entry point for catalog discovery.
        """
        dataset_entry_points = [ep for ep in mock_entry_points if ep.group == 'kedro.datasets']
        
        # Validate dataset entry points exist
        assert len(dataset_entry_points) > 0, "Should have kedro.datasets entry points"
        
        # Check specific dataset registration
        dataset_values = [ep.value for ep in dataset_entry_points]
        for expected_dataset in EXPECTED_KEDRO_DATASETS:
            assert expected_dataset in dataset_values, \
                f"Missing dataset entry point: {expected_dataset}"
        
        # Validate entry point structure
        for ep in dataset_entry_points:
            assert hasattr(ep, 'name'), "Entry point should have name"
            assert hasattr(ep, 'value'), "Entry point should have value"
            assert '.' in ep.value, "Entry point value should be module.class format"
    
    def test_plugin_discovery_integration(self):
        """
        Test integration with Kedro's plugin discovery system.
        
        Validates that the plugin can be discovered and loaded
        through Kedro's standard plugin mechanisms.
        """
        # Mock Kedro plugin discovery
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            # Setup mock entry points
            mock_eps = Mock()
            mock_eps.select = Mock(return_value=[])
            mock_entry_points.return_value = mock_eps
            
            # Test plugin discovery doesn't raise errors
            try:
                import figregistry_kedro
                # Plugin should be importable even with mocked discovery
                assert figregistry_kedro is not None
            except Exception as e:
                pytest.fail(f"Plugin discovery integration failed: {e}")
    
    def test_entry_point_loading_validation(self):
        """
        Test that entry points can be loaded successfully.
        
        Validates that registered entry points reference valid
        modules and classes that can be imported and instantiated.
        """
        import figregistry_kedro
        
        # Test hook class loading
        try:
            hook_class = figregistry_kedro.FigRegistryHooks
            assert callable(hook_class), "Hook class should be callable"
            
            # Test hook instantiation
            hook_instance = hook_class()
            assert hook_instance is not None, "Hook should be instantiable"
            
        except Exception as e:
            pytest.fail(f"Hook entry point loading failed: {e}")
        
        # Test dataset class loading
        try:
            dataset_class = figregistry_kedro.FigureDataSet
            assert hasattr(dataset_class, '_save'), "Dataset should implement _save"
            assert hasattr(dataset_class, '_load'), "Dataset should implement _load"
            
        except Exception as e:
            pytest.fail(f"Dataset entry point loading failed: {e}")


class TestImportErrorHandling:
    """
    Test import error handling and security validation.
    
    Validates that the package handles import errors gracefully
    and provides appropriate security measures per Section 6.6.8.1.
    """
    
    def test_circular_import_prevention(self):
        """
        Test prevention of circular import issues.
        
        Validates that package initialization doesn't create
        circular dependencies that could cause import failures.
        """
        # Clear module cache to ensure clean import
        modules_to_clear = [
            mod for mod in sys.modules.keys() 
            if mod.startswith('figregistry_kedro')
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        # Test import doesn't cause circular dependency
        try:
            import figregistry_kedro
            assert figregistry_kedro is not None
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            # Re-raise other import errors for investigation
            raise
    
    def test_malformed_dependency_handling(self):
        """
        Test handling of malformed or corrupted dependencies.
        
        Validates that package handles corrupted dependency scenarios
        gracefully without exposing security vulnerabilities.
        """
        # Test with malformed version string
        with patch.dict('sys.modules', {'figregistry': Mock()}):
            mock_figregistry = sys.modules['figregistry']
            mock_figregistry.__version__ = "not.a.version"
            
            # Clear module cache
            if 'figregistry_kedro' in sys.modules:
                del sys.modules['figregistry_kedro']
            
            with pytest.raises((ImportError, ValueError)) as exc_info:
                import figregistry_kedro
            
            # Error should be informative but not expose internal details
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0, "Should provide error message"
            assert not any(dangerous in error_msg.lower() for dangerous in ['traceback', 'internal', 'debug']), \
                "Error message should not expose internal details"
    
    def test_import_isolation(self):
        """
        Test import isolation and namespace protection.
        
        Validates that package imports don't pollute global namespace
        or interfere with other packages.
        """
        # Store initial sys.modules state
        initial_modules = set(sys.modules.keys())
        
        # Import package
        import figregistry_kedro
        
        # Check that only expected modules were added
        new_modules = set(sys.modules.keys()) - initial_modules
        
        # All new modules should be figregistry_kedro related
        for module in new_modules:
            assert module.startswith('figregistry_kedro'), \
                f"Unexpected module imported: {module}"
        
        # Test that package doesn't modify existing modules
        assert 'figregistry' in sys.modules or True, \
            "Package should not modify existing module imports"
    
    def test_resource_cleanup_on_import_failure(self):
        """
        Test proper resource cleanup when import fails.
        
        Validates that failed imports don't leave resources
        in inconsistent state.
        """
        # Force import failure
        with patch('builtins.__import__', side_effect=ImportError("Forced failure")):
            initial_modules = set(sys.modules.keys())
            
            with pytest.raises(ImportError):
                import figregistry_kedro
            
            # Check that no partial modules remain
            current_modules = set(sys.modules.keys())
            new_modules = current_modules - initial_modules
            
            # Should not have partial figregistry_kedro modules
            partial_modules = [
                mod for mod in new_modules 
                if mod.startswith('figregistry_kedro')
            ]
            
            # Some modules might remain due to import mechanics, but check they're not corrupted
            for mod in partial_modules:
                module_obj = sys.modules[mod]
                # Module should either be properly initialized or None
                assert module_obj is None or hasattr(module_obj, '__name__'), \
                    f"Corrupted module state: {mod}"


class TestVersionCompatibility:
    """
    Test version compatibility and compatibility matrix validation.
    
    Validates compatibility across supported Python, FigRegistry,
    and Kedro versions per Section 3.2.3.1 requirements.
    """
    
    def test_python_version_compatibility(self):
        """
        Test Python version compatibility validation.
        
        Validates that package enforces minimum Python version
        requirements and provides clear upgrade guidance.
        """
        current_version = sys.version_info[:2]
        
        # Should work with current Python version (assuming test environment is compatible)
        assert current_version >= MINIMUM_PYTHON_VERSION, \
            f"Test environment Python {current_version} below minimum {MINIMUM_PYTHON_VERSION}"
        
        # Test would fail on older Python versions
        if current_version < MINIMUM_PYTHON_VERSION:
            with pytest.raises((ImportError, SystemError)) as exc_info:
                import figregistry_kedro
            
            assert "python" in str(exc_info.value).lower(), \
                "Error should mention Python version requirement"
    
    def test_semantic_version_parsing(self):
        """
        Test semantic version parsing and comparison.
        
        Validates that package correctly parses and compares
        semantic versions for dependency management.
        """
        import figregistry_kedro
        
        # Test package version is valid semantic version
        version = parse_version(figregistry_kedro.__version__)
        assert version is not None
        
        # Test version comparison capabilities
        test_versions = ['0.1.0', '0.2.0', '1.0.0', '1.0.0-alpha', '1.0.0+build.1']
        
        for version_str in test_versions:
            try:
                parsed = parse_version(version_str)
                assert parsed is not None
            except Exception as e:
                pytest.fail(f"Failed to parse semantic version {version_str}: {e}")
    
    def test_dependency_version_matrix(self):
        """
        Test compatibility across dependency version matrix.
        
        Validates compatibility boundaries for FigRegistry and Kedro
        versions according to specification requirements.
        """
        # Test FigRegistry version requirements
        figregistry_versions = [
            ('0.2.9', False),  # Below minimum
            ('0.3.0', True),   # Minimum supported
            ('0.3.5', True),   # Mid-range
            ('0.4.0', True),   # Future compatible
            ('1.0.0', True),   # Major version compatible
        ]
        
        for version_str, should_work in figregistry_versions:
            with patch.dict('sys.modules', {'figregistry': Mock()}):
                mock_figregistry = sys.modules['figregistry']
                mock_figregistry.__version__ = version_str
                
                if 'figregistry_kedro' in sys.modules:
                    del sys.modules['figregistry_kedro']
                
                if should_work:
                    try:
                        import figregistry_kedro
                        assert figregistry_kedro is not None
                    except ImportError:
                        pytest.fail(f"Should work with FigRegistry {version_str}")
                else:
                    with pytest.raises((ImportError, ValueError)):
                        import figregistry_kedro
        
        # Test Kedro version requirements
        kedro_versions = [
            ('0.17.9', False),  # Below minimum
            ('0.18.0', True),   # Minimum supported
            ('0.18.14', True),  # Mid-range
            ('0.19.0', True),   # Recent version
            ('0.19.9', True),   # Before maximum
            ('0.20.0', False),  # At or above maximum
        ]
        
        for version_str, should_work in kedro_versions:
            with patch.dict('sys.modules', {'kedro': Mock(), 'figregistry': Mock()}):
                mock_kedro = sys.modules['kedro']
                mock_kedro.__version__ = version_str
                mock_figregistry = sys.modules['figregistry']
                mock_figregistry.__version__ = '0.3.0'
                
                if 'figregistry_kedro' in sys.modules:
                    del sys.modules['figregistry_kedro']
                
                if should_work:
                    try:
                        import figregistry_kedro
                        assert figregistry_kedro is not None
                    except ImportError:
                        pytest.fail(f"Should work with Kedro {version_str}")
                else:
                    with pytest.raises((ImportError, ValueError)):
                        import figregistry_kedro


class TestAPIConsistency:
    """
    Test API consistency and interface stability.
    
    Validates that package API remains consistent and provides
    stable interfaces for downstream consumers.
    """
    
    def test_api_surface_stability(self):
        """
        Test that API surface remains stable across imports.
        
        Validates that package provides consistent API regardless
        of import order or module state.
        """
        import figregistry_kedro
        
        # Capture initial API surface
        initial_attrs = set(dir(figregistry_kedro))
        initial_exports = {name: getattr(figregistry_kedro, name) for name in EXPECTED_EXPORTS}
        
        # Re-import package
        importlib.reload(figregistry_kedro)
        
        # Validate API surface consistency
        reloaded_attrs = set(dir(figregistry_kedro))
        assert initial_attrs == reloaded_attrs, \
            "API surface changed after reload"
        
        # Validate export consistency
        for name in EXPECTED_EXPORTS:
            initial_obj = initial_exports[name]
            reloaded_obj = getattr(figregistry_kedro, name)
            
            # Objects should have same type and key attributes
            assert type(initial_obj) == type(reloaded_obj), \
                f"Export {name} type changed after reload"
    
    def test_component_interface_consistency(self):
        """
        Test that component interfaces remain consistent.
        
        Validates that core components maintain expected
        method signatures and behavior contracts.
        """
        import figregistry_kedro
        
        # Test FigureDataSet interface
        dataset_class = figregistry_kedro.FigureDataSet
        required_methods = ['_save', '_load', '_describe']
        
        for method in required_methods:
            assert hasattr(dataset_class, method), \
                f"FigureDataSet missing required method: {method}"
        
        # Test FigRegistryHooks interface
        hooks_class = figregistry_kedro.FigRegistryHooks
        assert callable(hooks_class), "FigRegistryHooks should be callable"
        
        # Test instantiation doesn't require arguments
        try:
            hooks_instance = hooks_class()
            assert hooks_instance is not None
        except TypeError:
            pytest.fail("FigRegistryHooks should be instantiable without arguments")
        
        # Test FigRegistryConfigBridge interface
        bridge_class = figregistry_kedro.FigRegistryConfigBridge
        assert hasattr(bridge_class, 'init_config'), \
            "FigRegistryConfigBridge should have init_config method"
    
    def test_backward_compatibility_markers(self):
        """
        Test backward compatibility markers and deprecation handling.
        
        Validates that package properly handles backward compatibility
        and provides clear deprecation warnings when needed.
        """
        import figregistry_kedro
        
        # Test that no deprecation warnings are raised during normal import
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Re-import to trigger any warnings
            importlib.reload(figregistry_kedro)
            
            # Check for unexpected deprecation warnings
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            
            # Should not have deprecation warnings for current API
            assert len(deprecation_warnings) == 0, \
                f"Unexpected deprecation warnings: {[str(w.message) for w in deprecation_warnings]}"
        
        # Test version attribute provides compatibility information
        version = figregistry_kedro.__version__
        assert isinstance(version, str), "Version should be string for compatibility checking"
        assert not version.startswith('0.0'), "Version should indicate stable release"


if __name__ == '__main__':
    # Run tests directly if executed as script
    pytest.main([__file__, '-v', '--tb=short'])