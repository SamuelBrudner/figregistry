"""
Core Kedro Testing Fixtures for FigRegistry-Kedro Integration

This module provides comprehensive Kedro component mocking and project scaffolding fixtures
that enable isolated testing of figregistry-kedro plugin components without requiring full
Kedro pipeline overhead. The fixtures support end-to-end plugin testing scenarios while
maintaining clean separation between test environments and realistic Kedro integration.

Key Capabilities:
- Mock ProjectContext, DataCatalog, and ConfigLoader simulation per Section 6.6.2.3
- TestKedro integration for kedro-pytest framework compatibility per Section 6.6.2.1
- Temporary project creation via kedro new for realistic plugin testing per Section 6.6.7.2
- Comprehensive hook registration and lifecycle testing per Section 6.6.3.8
- Isolated plugin component testing without full pipeline execution overhead
- FigRegistryConfigBridge testing with mock configuration merging scenarios

Testing Framework Integration:
The fixtures integrate with pytest-mock for Kedro component simulation, providing isolated
test environments that validate plugin functionality across the Kedro version matrix
(>=0.18.0,<0.20.0) while maintaining compatibility with kedro-pytest testing framework
for enhanced integration testing capabilities and project scaffolding support.
"""

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union, Tuple
from unittest.mock import MagicMock, Mock, patch
from contextlib import contextmanager

import pytest
import yaml
import json

# Kedro imports with compatibility handling
try:
    from kedro.framework.project import configure_project
    from kedro.framework.context import KedroContext
    from kedro.config import ConfigLoader, MissingConfigException
    from kedro.io import DataCatalog, AbstractDataSet
    from kedro.pipeline import Pipeline
    from kedro.runner import AbstractRunner
    from kedro.framework.hooks import hook_impl
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    KEDRO_AVAILABLE = True
except ImportError:
    # Mock Kedro classes for environments where Kedro isn't available
    KedroContext = type('KedroContext', (), {})
    ConfigLoader = type('ConfigLoader', (), {})
    DataCatalog = type('DataCatalog', (), {})
    AbstractDataSet = type('AbstractDataSet', (), {})
    Pipeline = type('Pipeline', (), {})
    AbstractRunner = type('AbstractRunner', (), {})
    KedroSession = type('KedroSession', (), {})
    KEDRO_AVAILABLE = False


# =============================================================================
# Core Kedro Component Mocking Fixtures
# =============================================================================

@pytest.fixture
def minimal_kedro_context(mocker) -> Mock:
    """
    Provides minimal ProjectContext simulation with ConfigLoader and DataCatalog mocks per Section 6.6.2.6.
    
    Creates a comprehensive mock of Kedro's ProjectContext that includes realistic
    ConfigLoader and DataCatalog behavior for plugin testing without requiring
    full Kedro project setup. This fixture enables isolated testing of plugin
    components that depend on Kedro context.
    
    Args:
        mocker: pytest-mock fixture for creating mock objects
        
    Returns:
        Mock ProjectContext with configured ConfigLoader and DataCatalog
        
    Example:
        def test_figregistry_hook_initialization(minimal_kedro_context):
            context = minimal_kedro_context
            hooks = FigRegistryHooks()
            hooks.after_config_loaded(context, context.config_loader, "conf")
            assert hooks.is_initialized
    """
    # Create mock ConfigLoader with realistic behavior
    mock_config_loader = mocker.Mock(spec=ConfigLoader)
    
    # Configure ConfigLoader responses for different configuration sections
    config_responses = {
        "figregistry": {
            "styles": {
                "control": {
                    "color": "#1f77b4",
                    "marker": "o",
                    "linestyle": "-",
                    "linewidth": 2.0,
                    "label": "Control"
                },
                "treatment": {
                    "color": "#ff7f0e", 
                    "marker": "s",
                    "linestyle": "--",
                    "linewidth": 2.0,
                    "label": "Treatment"
                }
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "naming": {"template": "{name}_{condition}_{ts}"},
                "formats": {"defaults": {"exploratory": ["png"]}}
            },
            "defaults": {
                "figure": {"figsize": [10, 8], "dpi": 150},
                "line": {"linewidth": 2.0, "alpha": 0.8}
            }
        },
        "parameters": {
            "experiment_condition": "control",
            "experiment_phase": "testing", 
            "model_type": "linear_regression",
            "plot_settings": {
                "figure_size": [8, 6],
                "dpi": 150
            }
        },
        "catalog": {
            "test_figure": {
                "type": "figregistry_kedro.datasets.FigureDataSet",
                "filepath": "data/08_reporting/test_figure.png",
                "condition_param": "experiment_condition",
                "purpose": "exploratory"
            }
        }
    }
    
    def mock_get_config(key: str, *args, **kwargs) -> Dict[str, Any]:
        """Mock ConfigLoader.get() method with realistic responses."""
        return config_responses.get(key, {})
    
    mock_config_loader.get.side_effect = mock_get_config
    
    # Create mock DataCatalog with FigureDataSet support
    mock_catalog = mocker.Mock(spec=DataCatalog)
    mock_catalog._datasets = {}
    mock_catalog.list.return_value = ["test_figure", "parameters"]
    
    def mock_catalog_save(name: str, data: Any) -> None:
        """Mock DataCatalog.save() method."""
        mock_catalog._datasets[name] = data
    
    def mock_catalog_load(name: str) -> Any:
        """Mock DataCatalog.load() method."""
        if name in mock_catalog._datasets:
            return mock_catalog._datasets[name]
        elif name == "parameters":
            return config_responses["parameters"]
        else:
            raise KeyError(f"Dataset '{name}' not found in catalog")
    
    mock_catalog.save = mock_catalog_save
    mock_catalog.load = mock_catalog_load
    mock_catalog.exists.return_value = True
    
    # Create comprehensive ProjectContext mock
    mock_context = mocker.Mock(spec=KedroContext)
    mock_context.config_loader = mock_config_loader
    mock_context.catalog = mock_catalog
    mock_context.project_name = "test_project"
    mock_context.project_version = "0.1"
    mock_context.project_path = Path("/tmp/test_project")
    
    # Add additional context properties commonly used in hooks
    mock_context.env = "base"
    mock_context.config_loader.base_env = "base"
    mock_context.config_loader.default_run_env = "base"
    
    return mock_context


@pytest.fixture
def test_catalog_with_figregistry(mocker) -> Mock:
    """
    DataCatalog with FigureDataSet entries for integration testing per Section 6.6.3.7.
    
    Provides a pre-configured DataCatalog mock that includes realistic FigureDataSet
    entries for comprehensive integration testing of the figregistry-kedro plugin.
    This fixture enables testing of catalog interactions, versioning, and dataset
    parameter resolution without requiring full Kedro project infrastructure.
    
    Args:
        mocker: pytest-mock fixture for creating mock objects
        
    Returns:
        Mock DataCatalog with configured FigureDataSet entries
        
    Example:
        def test_catalog_figuredataset_integration(test_catalog_with_figregistry):
            catalog = test_catalog_with_figregistry
            catalog.save("exploratory_plot", matplotlib_figure)
            assert "exploratory_plot" in catalog._datasets
    """
    mock_catalog = mocker.Mock(spec=DataCatalog)
    
    # Configure realistic FigureDataSet catalog entries
    figuredataset_entries = {
        "exploratory_plot": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/exploratory/plot_{run_id}.png",
            "condition_param": "experiment_type",
            "purpose": "exploratory",
            "save_args": {
                "dpi": 200,
                "bbox_inches": "tight"
            }
        },
        "presentation_chart": {
            "type": "figregistry_kedro.datasets.FigureDataSet", 
            "filepath": "data/08_reporting/presentation/chart_{condition}_{timestamp}.pdf",
            "condition_param": "analysis_stage",
            "purpose": "presentation",
            "style_params": {
                "publication_ready": True,
                "high_dpi": True
            },
            "save_args": {
                "dpi": 300,
                "format": "pdf"
            }
        },
        "publication_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/publication/figure_{experiment}_{version}.svg",
            "condition_param": "publication_condition",
            "purpose": "publication",
            "versioned": True,
            "save_args": {
                "dpi": 300,
                "format": "svg",
                "bbox_inches": "tight"
            }
        },
        "debug_plot": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/01_raw/debug/debug_plot.png",
            "condition_param": "debug_mode",
            "purpose": "debug",
            "save_args": {
                "dpi": 100
            }
        }
    }
    
    # Mock catalog behavior with dataset storage
    mock_catalog._datasets = {}
    mock_catalog._catalog_config = figuredataset_entries
    mock_catalog.list.return_value = list(figuredataset_entries.keys())
    
    def mock_save(name: str, data: Any) -> None:
        """Mock save operation for FigureDataSet entries."""
        if name in figuredataset_entries:
            # Simulate FigureDataSet save behavior
            mock_catalog._datasets[name] = {
                "data": data,
                "config": figuredataset_entries[name],
                "saved_timestamp": time.time()
            }
        else:
            mock_catalog._datasets[name] = data
    
    def mock_load(name: str) -> Any:
        """Mock load operation with error handling."""
        if name in mock_catalog._datasets:
            saved_data = mock_catalog._datasets[name]
            if isinstance(saved_data, dict) and "data" in saved_data:
                return saved_data["data"]
            return saved_data
        else:
            raise KeyError(f"Dataset '{name}' not found")
    
    def mock_exists(name: str) -> bool:
        """Mock exists check for datasets."""
        return name in figuredataset_entries or name in mock_catalog._datasets
    
    def mock_release(name: str) -> None:
        """Mock dataset release operation."""
        if name in mock_catalog._datasets:
            del mock_catalog._datasets[name]
    
    mock_catalog.save = mock_save
    mock_catalog.load = mock_load
    mock_catalog.exists = mock_exists
    mock_catalog.release = mock_release
    
    # Add catalog configuration access methods
    mock_catalog.get_dataset_config = lambda name: figuredataset_entries.get(name, {})
    mock_catalog.get_figuredataset_entries = lambda: {
        k: v for k, v in figuredataset_entries.items() 
        if v.get("type") == "figregistry_kedro.datasets.FigureDataSet"
    }
    
    return mock_catalog


@pytest.fixture
def mock_kedro_session(tmp_path, mocker) -> Generator[Dict[str, Any], None, None]:
    """
    Complete Kedro session simulation with temporary project setup per Section 6.6.2.6.
    
    Provides a comprehensive mock of Kedro's session management with temporary
    project structure creation. This fixture enables end-to-end testing of
    plugin components within realistic Kedro session contexts while maintaining
    complete isolation and cleanup.
    
    Args:
        tmp_path: pytest temporary directory fixture
        mocker: pytest-mock fixture for creating mock objects
        
    Yields:
        Dictionary containing session context, project paths, and management utilities
        
    Example:
        def test_complete_plugin_workflow(mock_kedro_session):
            session_data = mock_kedro_session
            session = session_data["session"]
            context = session.load_context()
            # Test complete plugin workflow
    """
    # Create temporary project structure
    project_name = "test_figregistry_project"
    project_path = tmp_path / project_name
    
    # Basic Kedro project directory structure
    (project_path / "conf" / "base").mkdir(parents=True, exist_ok=True)
    (project_path / "conf" / "local").mkdir(parents=True, exist_ok=True)
    (project_path / "src" / project_name).mkdir(parents=True, exist_ok=True)
    (project_path / "data" / "01_raw").mkdir(parents=True, exist_ok=True)
    (project_path / "data" / "08_reporting").mkdir(parents=True, exist_ok=True)
    
    # Create basic configuration files
    base_catalog = {
        "test_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/test_figure.png",
            "condition_param": "experiment_condition"
        }
    }
    
    figregistry_config = {
        "styles": {
            "control": {"color": "#1f77b4", "marker": "o"},
            "treatment": {"color": "#ff7f0e", "marker": "s"}
        },
        "outputs": {
            "base_path": "data/08_reporting",
            "naming": {"template": "{name}_{condition}_{ts}"}
        }
    }
    
    parameters_config = {
        "experiment_condition": "control",
        "experiment_phase": "testing"
    }
    
    # Write configuration files
    with open(project_path / "conf" / "base" / "catalog.yml", "w") as f:
        yaml.dump(base_catalog, f)
    
    with open(project_path / "conf" / "base" / "figregistry.yml", "w") as f:
        yaml.dump(figregistry_config, f)
    
    with open(project_path / "conf" / "base" / "parameters.yml", "w") as f:
        yaml.dump(parameters_config, f)
    
    # Create mock session
    mock_session = mocker.Mock(spec=KedroSession)
    
    # Create comprehensive context mock
    mock_context = mocker.Mock(spec=KedroContext)
    mock_context.project_name = project_name
    mock_context.project_path = project_path
    mock_context.project_version = "0.1"
    mock_context.env = "base"
    
    # Mock ConfigLoader with file-based responses
    mock_config_loader = mocker.Mock(spec=ConfigLoader)
    
    def mock_get_config(key: str, *args, **kwargs) -> Dict[str, Any]:
        """Load configuration from actual files."""
        config_file_map = {
            "catalog": project_path / "conf" / "base" / "catalog.yml",
            "figregistry": project_path / "conf" / "base" / "figregistry.yml",
            "parameters": project_path / "conf" / "base" / "parameters.yml"
        }
        
        if key in config_file_map and config_file_map[key].exists():
            with open(config_file_map[key], "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    mock_config_loader.get.side_effect = mock_get_config
    mock_context.config_loader = mock_config_loader
    
    # Mock DataCatalog
    mock_catalog = mocker.Mock(spec=DataCatalog)
    mock_catalog._datasets = {}
    mock_catalog.list.return_value = ["test_figure"]
    
    def mock_catalog_save(name: str, data: Any) -> None:
        mock_catalog._datasets[name] = data
    
    def mock_catalog_load(name: str) -> Any:
        if name in mock_catalog._datasets:
            return mock_catalog._datasets[name]
        raise KeyError(f"Dataset '{name}' not found")
    
    mock_catalog.save = mock_catalog_save
    mock_catalog.load = mock_catalog_load
    mock_catalog.exists = lambda name: name in mock_catalog._datasets or name == "test_figure"
    
    mock_context.catalog = mock_catalog
    
    # Configure session to return context
    mock_session.load_context.return_value = mock_context
    
    def mock_run(*args, **kwargs):
        """Mock session run method."""
        return {"pipeline_output": "success"}
    
    mock_session.run = mock_run
    
    session_data = {
        "session": mock_session,
        "context": mock_context,
        "project_path": project_path,
        "project_name": project_name,
        "config_loader": mock_config_loader,
        "catalog": mock_catalog,
        "cleanup_paths": [project_path]
    }
    
    try:
        yield session_data
    finally:
        # Cleanup temporary project structure
        if project_path.exists():
            shutil.rmtree(project_path, ignore_errors=True)


@pytest.fixture
def figregistry_config_bridge(minimal_kedro_context, mocker):
    """
    Pre-configured FigRegistryConfigBridge for testing configuration merging scenarios.
    
    Provides a ready-to-use FigRegistryConfigBridge instance with mock Kedro context
    and realistic configuration scenarios for testing configuration merging,
    validation, and precedence rules without requiring full system setup.
    
    Args:
        minimal_kedro_context: Mock ProjectContext fixture
        mocker: pytest-mock fixture for creating additional mocks
        
    Returns:
        Mock FigRegistryConfigBridge instance with test configuration
        
    Example:
        def test_config_bridge_merging(figregistry_config_bridge):
            bridge = figregistry_config_bridge
            merged_config = bridge.get_merged_config()
            assert "styles" in merged_config
    """
    # Import FigRegistryConfigBridge (with fallback for testing environments)
    try:
        from figregistry_kedro.config import FigRegistryConfigBridge
        ConfigBridgeClass = FigRegistryConfigBridge
    except ImportError:
        # Create mock class for testing environments without the actual plugin
        ConfigBridgeClass = type('FigRegistryConfigBridge', (), {})
    
    # Create mock bridge instance
    mock_bridge = mocker.Mock(spec=ConfigBridgeClass)
    
    # Mock configuration data for testing
    mock_figregistry_config = {
        "styles": {
            "baseline": {"color": "#1f77b4", "marker": "o", "linewidth": 2.0},
            "treatment": {"color": "#ff7f0e", "marker": "s", "linewidth": 2.0}
        },
        "outputs": {
            "base_path": "data/08_reporting",
            "naming": {"template": "{name}_{condition}_{ts}"}
        },
        "defaults": {
            "figure": {"figsize": [10, 8], "dpi": 150},
            "line": {"linewidth": 2.0, "alpha": 0.8}
        }
    }
    
    mock_kedro_overrides = {
        "figregistry": {
            "outputs": {
                "base_path": "data/08_reporting/kedro_output"  # Override from Kedro
            },
            "defaults": {
                "figure": {"dpi": 300}  # Override DPI
            }
        }
    }
    
    mock_merged_config = copy.deepcopy(mock_figregistry_config)
    mock_merged_config["outputs"]["base_path"] = "data/08_reporting/kedro_output"
    mock_merged_config["defaults"]["figure"]["dpi"] = 300
    
    # Configure bridge behavior
    mock_bridge.kedro_context = minimal_kedro_context
    mock_bridge.figregistry_config = mock_figregistry_config
    mock_bridge.kedro_overrides = mock_kedro_overrides
    mock_bridge.merged_config = mock_merged_config
    
    # Mock bridge methods
    mock_bridge.get_merged_config.return_value = mock_merged_config
    mock_bridge.get_figregistry_config.return_value = mock_figregistry_config
    mock_bridge.get_kedro_overrides.return_value = mock_kedro_overrides
    mock_bridge.resolve_condition_parameters.return_value = {"experiment_condition": "control"}
    
    def mock_merge_configurations(figregistry_config, kedro_config):
        """Mock configuration merging logic."""
        merged = copy.deepcopy(figregistry_config)
        if "figregistry" in kedro_config:
            for section, values in kedro_config["figregistry"].items():
                if section in merged:
                    if isinstance(merged[section], dict) and isinstance(values, dict):
                        merged[section].update(values)
                    else:
                        merged[section] = values
                else:
                    merged[section] = values
        return merged
    
    mock_bridge.merge_configurations.side_effect = mock_merge_configurations
    
    return mock_bridge


@pytest.fixture
def mock_hook_manager(mocker):
    """
    Mock hook manager for hook registration and lifecycle testing per Section 6.6.3.8.
    
    Provides a comprehensive mock of Kedro's hook management system for testing
    FigRegistryHooks registration, lifecycle integration, and hook execution
    sequencing without requiring full Kedro framework initialization.
    
    Args:
        mocker: pytest-mock fixture for creating mock objects
        
    Returns:
        Mock hook manager with realistic hook registration and execution behavior
        
    Example:
        def test_hook_registration(mock_hook_manager):
            hook_manager = mock_hook_manager
            hooks = FigRegistryHooks()
            hook_manager.register(hooks)
            assert hooks in hook_manager.registered_hooks
    """
    # Create mock hook manager
    mock_manager = mocker.Mock()
    
    # Mock registered hooks storage
    mock_manager.registered_hooks = []
    mock_manager.hook_impls = {}
    
    def mock_register(hook_instance):
        """Mock hook registration."""
        mock_manager.registered_hooks.append(hook_instance)
        
        # Register individual hook implementations
        hook_methods = [
            attr for attr in dir(hook_instance) 
            if attr.startswith('before_') or attr.startswith('after_')
        ]
        
        for method_name in hook_methods:
            if method_name not in mock_manager.hook_impls:
                mock_manager.hook_impls[method_name] = []
            mock_manager.hook_impls[method_name].append(getattr(hook_instance, method_name))
    
    def mock_call_hook(hook_name: str, *args, **kwargs):
        """Mock hook execution."""
        if hook_name in mock_manager.hook_impls:
            results = []
            for hook_impl in mock_manager.hook_impls[hook_name]:
                try:
                    result = hook_impl(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results
        return []
    
    def mock_is_registered(hook_instance) -> bool:
        """Check if hook is registered."""
        return hook_instance in mock_manager.registered_hooks
    
    mock_manager.register = mock_register
    mock_manager.call_hook = mock_call_hook
    mock_manager.is_registered = mock_is_registered
    
    # Mock hook lifecycle methods
    mock_manager.before_pipeline_run = lambda *args, **kwargs: mock_call_hook("before_pipeline_run", *args, **kwargs)
    mock_manager.after_pipeline_run = lambda *args, **kwargs: mock_call_hook("after_pipeline_run", *args, **kwargs)
    mock_manager.before_catalog_created = lambda *args, **kwargs: mock_call_hook("before_catalog_created", *args, **kwargs)
    mock_manager.after_catalog_created = lambda *args, **kwargs: mock_call_hook("after_catalog_created", *args, **kwargs)
    
    return mock_manager


# =============================================================================
# TestKedro Integration Fixtures
# =============================================================================

@pytest.fixture
def test_kedro_integration(tmp_path):
    """
    TestKedro integration for kedro-pytest framework compatibility per Section 6.6.2.1.
    
    Provides integration with the kedro-pytest testing framework for enhanced
    plugin testing capabilities. This fixture enables use of TestKedro utilities
    while maintaining compatibility with custom figregistry-kedro testing patterns.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        TestKedro instance with figregistry-kedro plugin support
        
    Example:
        def test_kedro_pytest_integration(test_kedro_integration):
            test_kedro = test_kedro_integration
            test_kedro.create_kedro_project("test_project")
            assert test_kedro.project_path.exists()
    """
    # Check for kedro-pytest availability
    try:
        from kedro_pytest import TestKedro
        KEDRO_PYTEST_AVAILABLE = True
    except ImportError:
        KEDRO_PYTEST_AVAILABLE = False
        # Create mock TestKedro for environments without kedro-pytest
        TestKedro = type('TestKedro', (), {})
    
    if KEDRO_PYTEST_AVAILABLE:
        # Create TestKedro instance with figregistry-kedro support
        test_kedro = TestKedro(
            project_name="figregistry_test_project",
            project_path=tmp_path,
            enable_hooks=True,
            catalog_fixtures=True
        )
        
        # Add figregistry-kedro specific configuration
        test_kedro.figregistry_config = {
            "styles": {
                "test": {"color": "#FF0000", "marker": "o"}
            },
            "outputs": {
                "base_path": "data/08_reporting"
            }
        }
        
        # Add helper methods for figregistry-kedro testing
        def setup_figregistry_integration():
            """Setup figregistry-kedro plugin integration."""
            if hasattr(test_kedro, 'project_path') and test_kedro.project_path:
                figregistry_config_path = test_kedro.project_path / "conf" / "base" / "figregistry.yml"
                figregistry_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(figregistry_config_path, "w") as f:
                    yaml.dump(test_kedro.figregistry_config, f)
        
        test_kedro.setup_figregistry_integration = setup_figregistry_integration
        
        return test_kedro
    else:
        # Return mock TestKedro for testing environments
        mock_test_kedro = TestKedro()
        mock_test_kedro.project_path = tmp_path / "mock_project"
        mock_test_kedro.project_name = "mock_figregistry_project"
        mock_test_kedro.setup_figregistry_integration = lambda: None
        
        return mock_test_kedro


# =============================================================================
# Project Scaffolding and Temporary Project Fixtures  
# =============================================================================

@pytest.fixture
def kedro_project_scaffold(tmp_path) -> Generator[Dict[str, Any], None, None]:
    """
    Temporary Kedro project creation using kedro new per Section 6.6.7.2.
    
    Creates isolated Kedro projects using the official kedro new command for
    realistic plugin testing scenarios. Provides complete project scaffolding
    with automated cleanup and state management for test independence.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Yields:
        Dictionary containing project context and scaffolding utilities
        
    Example:
        def test_realistic_plugin_integration(kedro_project_scaffold):
            scaffold = kedro_project_scaffold
            project = scaffold["create_project"]("test_project")
            # Test plugin in realistic Kedro project environment
    """
    scaffold_data = {
        "base_path": tmp_path,
        "created_projects": [],
        "active_projects": {},
        "cleanup_registry": []
    }
    
    def create_project(
        name: str,
        starter: str = "spaceflights",
        install_plugin: bool = True,
        enable_hooks: bool = True
    ) -> Dict[str, Any]:
        """
        Create new Kedro project with optional figregistry-kedro integration.
        
        Args:
            name: Project name
            starter: Kedro starter template (default: "spaceflights")
            install_plugin: Whether to install figregistry-kedro plugin
            enable_hooks: Whether to enable FigRegistryHooks
            
        Returns:
            Project context dictionary with paths and utilities
        """
        project_path = tmp_path / name
        scaffold_data["created_projects"].append(project_path)
        
        # Store original working directory
        original_cwd = os.getcwd()
        
        try:
            # Change to temporary directory for project creation
            os.chdir(tmp_path)
            
            # Create Kedro project using kedro new
            cmd = [
                sys.executable, "-m", "kedro", "new",
                "--starter", starter,
                "--name", name,
                "--verbose"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                # Fallback to manual project structure creation
                _create_manual_project_structure(project_path, name)
            
            # Install figregistry-kedro plugin if requested
            if install_plugin:
                _install_plugin_in_project(project_path)
            
            # Setup figregistry configuration
            _setup_figregistry_config(project_path)
            
            # Register hooks if enabled
            if enable_hooks:
                _register_figregistry_hooks(project_path)
            
            # Setup catalog with FigureDataSet entries
            _setup_catalog_with_figuredataset(project_path)
            
            project_context = {
                "name": name,
                "path": project_path,
                "starter": starter,
                "plugin_installed": install_plugin,
                "hooks_enabled": enable_hooks,
                "run_pipeline": lambda: _run_kedro_pipeline(project_path),
                "validate_plugin": lambda: _validate_plugin_installation(project_path),
                "cleanup": lambda: _cleanup_project(project_path)
            }
            
            scaffold_data["active_projects"][name] = project_context
            return project_context
        
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def cleanup_project(name: str):
        """Clean up specific project."""
        if name in scaffold_data["active_projects"]:
            project_context = scaffold_data["active_projects"][name]
            _cleanup_project(project_context["path"])
            del scaffold_data["active_projects"][name]
    
    def cleanup_all_projects():
        """Clean up all created projects."""
        for project_path in scaffold_data["created_projects"]:
            if project_path.exists():
                _cleanup_project(project_path)
        scaffold_data["created_projects"].clear()
        scaffold_data["active_projects"].clear()
    
    scaffold_data.update({
        "create_project": create_project,
        "cleanup_project": cleanup_project,
        "cleanup_all_projects": cleanup_all_projects,
        "list_projects": lambda: list(scaffold_data["active_projects"].keys()),
        "get_project": lambda name: scaffold_data["active_projects"].get(name)
    })
    
    try:
        yield scaffold_data
    finally:
        # Comprehensive cleanup
        cleanup_all_projects()


@pytest.fixture
def minimal_project_context(tmp_path) -> Generator[Dict[str, Any], None, None]:
    """
    Minimal Kedro project context for lightweight plugin testing.
    
    Provides a lightweight Kedro project structure with essential components
    for plugin testing without the overhead of full project scaffolding.
    Ideal for unit tests and focused component validation.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Yields:
        Dictionary containing minimal project context and configuration
        
    Example:
        def test_minimal_plugin_component(minimal_project_context):
            context = minimal_project_context
            # Test plugin component in minimal environment
    """
    project_name = "minimal_test_project"
    project_path = tmp_path / project_name
    
    # Create minimal directory structure
    (project_path / "conf" / "base").mkdir(parents=True, exist_ok=True)
    (project_path / "conf" / "local").mkdir(parents=True, exist_ok=True)
    (project_path / "src" / project_name).mkdir(parents=True, exist_ok=True)
    (project_path / "data" / "08_reporting").mkdir(parents=True, exist_ok=True)
    
    # Create minimal configuration files
    minimal_catalog = {
        "test_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/test_figure.png",
            "condition_param": "test_condition"
        }
    }
    
    minimal_figregistry_config = {
        "styles": {
            "test": {"color": "#FF0000", "marker": "o", "linewidth": 2.0}
        },
        "outputs": {
            "base_path": "data/08_reporting",
            "naming": {"template": "{name}_{condition}"}
        }
    }
    
    minimal_parameters = {
        "test_condition": "test",
        "debug_mode": True
    }
    
    # Write configuration files
    with open(project_path / "conf" / "base" / "catalog.yml", "w") as f:
        yaml.dump(minimal_catalog, f)
    
    with open(project_path / "conf" / "base" / "figregistry.yml", "w") as f:
        yaml.dump(minimal_figregistry_config, f)
        
    with open(project_path / "conf" / "base" / "parameters.yml", "w") as f:
        yaml.dump(minimal_parameters, f)
    
    # Create minimal settings.py for hook registration
    settings_content = '''"""Project settings for minimal test."""

try:
    from figregistry_kedro.hooks import FigRegistryHooks
    HOOKS = (FigRegistryHooks(),)
except ImportError:
    HOOKS = ()
'''
    
    settings_path = project_path / "src" / project_name / "settings.py"
    with open(settings_path, "w") as f:
        f.write(settings_content)
    
    project_context = {
        "name": project_name,
        "path": project_path,
        "conf_path": project_path / "conf",
        "src_path": project_path / "src",
        "data_path": project_path / "data",
        "catalog_config": minimal_catalog,
        "figregistry_config": minimal_figregistry_config,
        "parameters": minimal_parameters,
        "load_config": lambda section: _load_config_section(project_path, section),
        "update_config": lambda section, config: _update_config_section(project_path, section, config)
    }
    
    try:
        yield project_context
    finally:
        # Cleanup minimal project
        if project_path.exists():
            shutil.rmtree(project_path, ignore_errors=True)


# =============================================================================
# Helper Functions for Project Management
# =============================================================================

def _create_manual_project_structure(project_path: Path, name: str):
    """Create minimal Kedro project structure manually."""
    # Create directory structure
    (project_path / "conf" / "base").mkdir(parents=True, exist_ok=True)
    (project_path / "conf" / "local").mkdir(parents=True, exist_ok=True)
    (project_path / "src" / name).mkdir(parents=True, exist_ok=True)
    (project_path / "data" / "01_raw").mkdir(parents=True, exist_ok=True)
    (project_path / "data" / "08_reporting").mkdir(parents=True, exist_ok=True)
    (project_path / "logs").mkdir(parents=True, exist_ok=True)
    
    # Create minimal pyproject.toml
    pyproject_content = f'''[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1"
dependencies = ["kedro>=0.18.0,<0.20.0"]
'''
    with open(project_path / "pyproject.toml", "w") as f:
        f.write(pyproject_content)


def _install_plugin_in_project(project_path: Path):
    """Install figregistry-kedro plugin in project."""
    try:
        # Check if running in development mode
        plugin_src_path = project_path.parent.parent.parent / "src" / "figregistry_kedro"
        if plugin_src_path.exists():
            # Development installation
            cmd = [sys.executable, "-m", "pip", "install", "-e", str(plugin_src_path.parent)]
        else:
            # Package installation
            cmd = [sys.executable, "-m", "pip", "install", "figregistry-kedro"]
        
        subprocess.run(cmd, capture_output=True, text=True, cwd=project_path, timeout=60)
    except Exception:
        # Installation failed, continue with testing
        pass


def _setup_figregistry_config(project_path: Path):
    """Setup figregistry configuration for testing."""
    figregistry_config = {
        "styles": {
            "control": {
                "color": "#1f77b4",
                "marker": "o",
                "linestyle": "-",
                "linewidth": 2.0,
                "label": "Control"
            },
            "treatment": {
                "color": "#ff7f0e",
                "marker": "s",
                "linestyle": "--", 
                "linewidth": 2.0,
                "label": "Treatment"
            },
            "exploratory_*": {
                "color": "#d62728",
                "marker": "x",
                "linestyle": ":",
                "linewidth": 1.5,
                "label": "Exploratory"
            }
        },
        "palettes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        },
        "outputs": {
            "base_path": "data/08_reporting",
            "naming": {"template": "{name}_{condition}_{ts}"},
            "formats": {"defaults": {"exploratory": ["png"]}}
        },
        "defaults": {
            "figure": {"figsize": [10, 8], "dpi": 150},
            "line": {"linewidth": 2.0, "alpha": 0.8}
        }
    }
    
    config_path = project_path / "conf" / "base" / "figregistry.yml"
    with open(config_path, "w") as f:
        yaml.dump(figregistry_config, f)


def _register_figregistry_hooks(project_path: Path):
    """Register FigRegistry hooks in project settings."""
    src_dir = project_path / "src"
    project_name = None
    
    # Find project package name
    for item in src_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            project_name = item.name
            break
    
    if not project_name:
        return
    
    settings_path = src_dir / project_name / "settings.py"
    
    settings_content = '''"""Project settings."""

try:
    from figregistry_kedro.hooks import FigRegistryHooks
    HOOKS = (FigRegistryHooks(),)
except ImportError:
    HOOKS = ()
'''
    
    with open(settings_path, "w") as f:
        f.write(settings_content)


def _setup_catalog_with_figuredataset(project_path: Path):
    """Setup catalog with FigureDataSet entries."""
    catalog_config = {
        "test_figure": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/test_figure.png",
            "condition_param": "experiment_condition",
            "save_args": {"dpi": 300, "bbox_inches": "tight"}
        },
        "exploratory_plot": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/exploratory/plot.png",
            "condition_param": "analysis_type",
            "purpose": "exploratory"
        }
    }
    
    catalog_path = project_path / "conf" / "base" / "catalog.yml"
    with open(catalog_path, "w") as f:
        yaml.dump(catalog_config, f)


def _run_kedro_pipeline(project_path: Path) -> Dict[str, Any]:
    """Execute kedro pipeline and return results."""
    original_cwd = os.getcwd()
    try:
        os.chdir(project_path)
        
        cmd = [sys.executable, "-m", "kedro", "run"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Pipeline execution timed out",
            "success": False
        }
    finally:
        os.chdir(original_cwd)


def _validate_plugin_installation(project_path: Path) -> bool:
    """Validate that figregistry-kedro plugin is properly installed."""
    try:
        # Check if plugin can be imported
        result = subprocess.run(
            [sys.executable, "-c", "import figregistry_kedro"],
            capture_output=True,
            text=True,
            cwd=project_path,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _cleanup_project(project_path: Path):
    """Comprehensive cleanup of Kedro project."""
    if not project_path.exists():
        return
    
    try:
        # Clean up data directories
        data_dir = project_path / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)
        
        # Clean up logs
        logs_dir = project_path / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir, ignore_errors=True)
        
        # Clean up .kedro cache
        kedro_dir = project_path / ".kedro"
        if kedro_dir.exists():
            shutil.rmtree(kedro_dir, ignore_errors=True)
        
        # Remove entire project directory
        shutil.rmtree(project_path, ignore_errors=True)
    except Exception:
        # Ignore cleanup errors
        pass


def _load_config_section(project_path: Path, section: str) -> Dict[str, Any]:
    """Load configuration section from project."""
    config_file = project_path / "conf" / "base" / f"{section}.yml"
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _update_config_section(project_path: Path, section: str, config: Dict[str, Any]):
    """Update configuration section in project."""
    config_file = project_path / "conf" / "base" / f"{section}.yml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(config, f)


# =============================================================================
# Context Managers for Test Environment Management
# =============================================================================

@contextmanager
def kedro_project_context(project_path: Path):
    """
    Context manager for temporary Kedro project environment setup.
    
    Provides a context manager that temporarily configures the environment
    for Kedro project execution, including working directory changes and
    environment variable setup, with automatic restoration on exit.
    
    Args:
        project_path: Path to Kedro project directory
        
    Yields:
        Path object for the project directory
        
    Example:
        with kedro_project_context(project_path) as proj_path:
            # Execute code in Kedro project context
            result = run_kedro_command()
    """
    original_cwd = os.getcwd()
    original_env = os.environ.copy()
    
    try:
        # Change to project directory
        os.chdir(project_path)
        
        # Set Kedro environment variables if needed
        os.environ['KEDRO_ENV'] = os.environ.get('KEDRO_ENV', 'base')
        
        yield project_path
    finally:
        # Restore original environment
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)


@contextmanager 
def isolated_kedro_environment():
    """
    Context manager for completely isolated Kedro testing environment.
    
    Provides complete isolation from any existing Kedro configuration or
    environment settings, ensuring tests run in clean environments without
    interference from user or system Kedro installations.
    
    Yields:
        None (context provides isolation only)
        
    Example:
        with isolated_kedro_environment():
            # Test runs in completely isolated environment
            test_plugin_functionality()
    """
    # Store original environment
    original_env = os.environ.copy()
    original_cwd = os.getcwd()
    
    # Environment variables to isolate
    kedro_env_vars = [
        'KEDRO_ENV',
        'KEDRO_PROJECT_PATH',
        'KEDRO_CONFIG_PATH'
    ]
    
    try:
        # Clear Kedro-specific environment variables
        for var in kedro_env_vars:
            if var in os.environ:
                del os.environ[var]
        
        yield
    finally:
        # Restore original environment
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)


# Export all fixtures for use in tests
__all__ = [
    "minimal_kedro_context",
    "test_catalog_with_figregistry", 
    "mock_kedro_session",
    "figregistry_config_bridge",
    "mock_hook_manager",
    "test_kedro_integration",
    "kedro_project_scaffold",
    "minimal_project_context",
    "kedro_project_context",
    "isolated_kedro_environment"
]