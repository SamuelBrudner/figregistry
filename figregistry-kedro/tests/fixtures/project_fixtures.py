"""
Temporary Kedro project creation fixtures providing realistic project scaffolding 
for end-to-end plugin testing.

This module implements comprehensive project fixture infrastructure that creates
isolated Kedro projects via `kedro new` command, configures plugin integration,
and provides automated cleanup management for integration testing scenarios.

Fixtures support basic and advanced pipeline examples with automated project
teardown, multi-environment configuration testing, and migration scenario validation.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple
import pytest
import yaml
from unittest.mock import patch

# Kedro imports for project management
try:
    from kedro.framework.startup import bootstrap_project
    from kedro.framework.session import KedroSession
    from kedro.framework.context import KedroContext
    from kedro.config import ConfigLoader
    from kedro.io import DataCatalog
    KEDRO_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments without Kedro
    KEDRO_AVAILABLE = False
    bootstrap_project = None
    KedroSession = None
    KedroContext = None
    ConfigLoader = None
    DataCatalog = None

# Local dependencies - will be available as this is part of the test suite
from figregistry_kedro.tests.fixtures.config_fixtures import (
    base_figregistry_config,
    local_override_config,
    merged_config_scenarios
)


class ProjectCleanupError(Exception):
    """Exception raised when project cleanup fails."""
    pass


class KedroProjectManager:
    """
    Manages temporary Kedro project lifecycle for testing.
    
    Provides centralized project creation, configuration, plugin installation,
    and cleanup operations with comprehensive error handling and state tracking.
    """
    
    def __init__(self, tmp_path: Path):
        """Initialize project manager with temporary directory."""
        self.tmp_path = tmp_path
        self.project_path: Optional[Path] = None
        self.project_name: Optional[str] = None
        self.created_projects: List[Path] = []
        self._original_cwd = Path.cwd()
        
    def create_project(
        self, 
        project_name: str = "test_figregistry_project",
        starter: str = "spaceflights",
        tools: Optional[List[str]] = None,
        example: bool = True
    ) -> Path:
        """
        Create a new Kedro project using kedro new command.
        
        Args:
            project_name: Name for the created project
            starter: Kedro starter template to use (default: spaceflights)
            tools: Additional tools to include in project setup
            example: Whether to include example pipeline code
            
        Returns:
            Path to the created project directory
            
        Raises:
            RuntimeError: If project creation fails
        """
        if not KEDRO_AVAILABLE:
            pytest.skip("Kedro not available for project creation testing")
            
        self.project_name = project_name
        self.project_path = self.tmp_path / project_name
        
        # Prepare kedro new command arguments
        cmd_args = [
            "kedro", "new",
            "--starter", starter,
            "--directory", str(self.tmp_path),
            "--name", project_name,
        ]
        
        # Add tools if specified
        if tools:
            cmd_args.extend(["--tools", ",".join(tools)])
            
        # Add example flag if requested
        if example:
            cmd_args.append("--example")
            
        try:
            # Execute kedro new command
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                cwd=self.tmp_path,
                timeout=120  # 2-minute timeout for project creation
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"Kedro project creation failed: {result.stderr}"
                )
                
            # Verify project was created successfully
            if not self.project_path.exists():
                raise RuntimeError(
                    f"Project directory not found after creation: {self.project_path}"
                )
                
            # Track created project for cleanup
            self.created_projects.append(self.project_path)
            
            return self.project_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Kedro project creation timed out after 2 minutes")
        except Exception as e:
            raise RuntimeError(f"Project creation failed: {str(e)}") from e
    
    def install_plugin(self, development_mode: bool = True) -> None:
        """
        Install figregistry-kedro plugin in the project environment.
        
        Args:
            development_mode: If True, install in development mode
        """
        if not self.project_path:
            raise RuntimeError("No project created to install plugin")
            
        # Get the figregistry-kedro package path (parent of tests directory)
        plugin_path = Path(__file__).parent.parent.parent
        
        cmd_args = ["pip", "install"]
        if development_mode:
            cmd_args.extend(["-e", str(plugin_path)])
        else:
            cmd_args.append(str(plugin_path))
            
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"Plugin installation failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Plugin installation timed out")
        except Exception as e:
            raise RuntimeError(f"Plugin installation failed: {str(e)}") from e
    
    def configure_project(
        self,
        figregistry_config: Optional[Dict[str, Any]] = None,
        local_overrides: Optional[Dict[str, Any]] = None,
        catalog_entries: Optional[Dict[str, Any]] = None,
        hooks_config: bool = True
    ) -> None:
        """
        Configure the project with FigRegistry settings and catalog entries.
        
        Args:
            figregistry_config: Base FigRegistry configuration
            local_overrides: Local environment overrides
            catalog_entries: Custom catalog entries to add
            hooks_config: Whether to register FigRegistryHooks
        """
        if not self.project_path:
            raise RuntimeError("No project created to configure")
            
        # Create conf directories if they don't exist
        conf_base = self.project_path / "conf" / "base"
        conf_local = self.project_path / "conf" / "local"
        conf_base.mkdir(parents=True, exist_ok=True)
        conf_local.mkdir(parents=True, exist_ok=True)
        
        # Write base FigRegistry configuration
        if figregistry_config:
            figregistry_yml = conf_base / "figregistry.yml"
            with open(figregistry_yml, 'w') as f:
                yaml.dump(figregistry_config, f, default_flow_style=False)
                
        # Write local overrides if provided
        if local_overrides:
            local_figregistry_yml = conf_local / "figregistry.yml"
            with open(local_figregistry_yml, 'w') as f:
                yaml.dump(local_overrides, f, default_flow_style=False)
                
        # Update catalog with FigureDataSet entries
        if catalog_entries:
            catalog_yml = conf_base / "catalog.yml"
            
            # Load existing catalog or create new one
            existing_catalog = {}
            if catalog_yml.exists():
                with open(catalog_yml, 'r') as f:
                    existing_catalog = yaml.safe_load(f) or {}
                    
            # Merge in new entries
            existing_catalog.update(catalog_entries)
            
            with open(catalog_yml, 'w') as f:
                yaml.dump(existing_catalog, f, default_flow_style=False)
                
        # Configure hooks registration
        if hooks_config:
            self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register FigRegistryHooks in project settings."""
        # Find the source package directory
        src_dirs = list(self.project_path.glob("src/*"))
        if not src_dirs:
            raise RuntimeError("No source package found in project")
            
        package_dir = src_dirs[0]
        settings_py = package_dir / "settings.py"
        
        # Read existing settings
        settings_content = ""
        if settings_py.exists():
            with open(settings_py, 'r') as f:
                settings_content = f.read()
        else:
            # Create basic settings file
            settings_content = '''"""Project settings."""
from kedro.config import ConfigLoader, TemplatedConfigLoader

'''
        
        # Add FigRegistry hooks import and registration
        hooks_import = "from figregistry_kedro.hooks import FigRegistryHooks\n"
        hooks_config = "\nHOOKS = (FigRegistryHooks(),)\n"
        
        if "from figregistry_kedro.hooks import FigRegistryHooks" not in settings_content:
            settings_content = hooks_import + settings_content
            
        if "HOOKS = " not in settings_content:
            settings_content += hooks_config
            
        with open(settings_py, 'w') as f:
            f.write(settings_content)
    
    def cleanup_project(self, force: bool = False) -> None:
        """
        Clean up created project with comprehensive state reset.
        
        Args:
            force: If True, ignore errors during cleanup
        """
        errors = []
        
        # Change back to original directory
        try:
            os.chdir(self._original_cwd)
        except Exception as e:
            errors.append(f"Failed to change directory: {e}")
            
        # Remove all created projects
        for project_path in self.created_projects:
            try:
                if project_path.exists():
                    shutil.rmtree(project_path)
            except Exception as e:
                errors.append(f"Failed to remove {project_path}: {e}")
                
        # Clear tracking variables
        self.created_projects.clear()
        self.project_path = None
        self.project_name = None
        
        # Raise errors unless forced cleanup
        if errors and not force:
            raise ProjectCleanupError(f"Cleanup errors: {'; '.join(errors)}")
    
    def execute_pipeline(
        self, 
        pipeline_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Execute a Kedro pipeline in the project.
        
        Args:
            pipeline_name: Specific pipeline to run (None for default)
            parameters: Runtime parameters for the pipeline
            
        Returns:
            Tuple of (success, output/error message)
        """
        if not self.project_path:
            raise RuntimeError("No project created to execute pipeline")
            
        cmd_args = ["kedro", "run"]
        if pipeline_name:
            cmd_args.extend(["--pipeline", pipeline_name])
            
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=300  # 5-minute timeout for pipeline execution
            )
            
            return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "Pipeline execution timed out after 5 minutes"
        except Exception as e:
            return False, f"Pipeline execution failed: {str(e)}"


@pytest.fixture
def minimal_test_pipeline(tmp_path) -> Generator[KedroProjectManager, None, None]:
    """
    Create isolated Kedro project via kedro new for basic plugin testing.
    
    This fixture implements Section 6.6.7.2 requirements for temporary project
    scaffolding using the spaceflights starter template with minimal configuration
    suitable for basic FigRegistry plugin validation.
    
    Yields:
        KedroProjectManager: Configured project manager with basic setup
    """
    manager = KedroProjectManager(tmp_path)
    
    try:
        # Create basic project with spaceflights template
        project_path = manager.create_project(
            project_name="minimal_test_project",
            starter="spaceflights",
            example=True
        )
        
        # Install plugin in development mode
        manager.install_plugin(development_mode=True)
        
        # Configure with basic FigRegistry settings
        basic_config = {
            'styles': {
                'exploratory': {
                    'figure.figsize': [10, 6],
                    'axes.labelsize': 12,
                    'axes.titlesize': 14
                },
                'presentation': {
                    'figure.figsize': [12, 8],
                    'axes.labelsize': 14,
                    'axes.titlesize': 16,
                    'font.size': 12
                }
            },
            'output': {
                'path_aliases': {
                    'expl': 'data/08_reporting/figures/exploratory',
                    'pres': 'data/08_reporting/figures/presentation'
                }
            }
        }
        
        # Basic catalog entries with FigureDataSet
        catalog_entries = {
            'exploratory_plot': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/figures/exploratory/plot.png',
                'purpose': 'exploratory',
                'condition_param': 'experiment_type',
                'style_params': {'condition': 'exploratory'}
            },
            'presentation_chart': {
                'type': 'figregistry_kedro.datasets.FigureDataSet', 
                'filepath': 'data/08_reporting/figures/presentation/chart.png',
                'purpose': 'presentation',
                'condition_param': 'experiment_type',
                'style_params': {'condition': 'presentation'}
            }
        }
        
        manager.configure_project(
            figregistry_config=basic_config,
            catalog_entries=catalog_entries,
            hooks_config=True
        )
        
        yield manager
        
    finally:
        # Ensure cleanup even on test failures
        manager.cleanup_project(force=True)


@pytest.fixture
def temp_kedro_project(tmp_path) -> Generator[KedroProjectManager, None, None]:
    """
    Create temporary Kedro project with automated cleanup and state management.
    
    Provides a clean project environment for each test with comprehensive
    state tracking and guaranteed cleanup to prevent cross-test contamination.
    
    Yields:
        KedroProjectManager: Clean project manager for test-specific configuration
    """
    manager = KedroProjectManager(tmp_path)
    
    try:
        yield manager
    finally:
        # Comprehensive cleanup with error suppression for test isolation
        manager.cleanup_project(force=True)


@pytest.fixture
def example_project_fixtures(tmp_path) -> Generator[Dict[str, KedroProjectManager], None, None]:
    """
    Create basic and advanced pipeline scenario projects per Section 6.6.4.5.
    
    Provides pre-configured projects representing different complexity levels:
    - basic: Simple single-pipeline with basic FigRegistry integration
    - advanced: Multi-pipeline with environment-specific configurations
    
    Yields:
        Dict mapping scenario names to configured KedroProjectManager instances
    """
    managers = {}
    
    try:
        # Basic example project
        basic_manager = KedroProjectManager(tmp_path / "basic")
        basic_path = basic_manager.create_project(
            project_name="basic_example",
            starter="spaceflights",
            example=True
        )
        basic_manager.install_plugin()
        
        # Basic configuration
        basic_config = {
            'styles': {
                'default': {
                    'figure.figsize': [8, 6],
                    'axes.labelsize': 10
                }
            },
            'output': {
                'path_aliases': {'figures': 'data/08_reporting/figures'}
            }
        }
        
        basic_catalog = {
            'simple_plot': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/figures/simple_plot.png',
                'purpose': 'exploratory'
            }
        }
        
        basic_manager.configure_project(
            figregistry_config=basic_config,
            catalog_entries=basic_catalog
        )
        
        managers['basic'] = basic_manager
        
        # Advanced example project
        advanced_manager = KedroProjectManager(tmp_path / "advanced")
        advanced_path = advanced_manager.create_project(
            project_name="advanced_example", 
            starter="spaceflights",
            tools=["viz"],
            example=True
        )
        advanced_manager.install_plugin()
        
        # Advanced multi-environment configuration
        advanced_base_config = {
            'styles': {
                'exploratory': {
                    'figure.figsize': [10, 6],
                    'axes.labelsize': 12,
                    'font.family': 'sans-serif'
                },
                'production': {
                    'figure.figsize': [12, 9],
                    'axes.labelsize': 14,
                    'font.family': 'serif',
                    'figure.dpi': 300
                }
            },
            'output': {
                'path_aliases': {
                    'expl': 'data/08_reporting/figures/exploratory',
                    'prod': 'data/08_reporting/figures/production'
                },
                'versioning': True
            }
        }
        
        advanced_local_overrides = {
            'styles': {
                'exploratory': {
                    'figure.dpi': 150  # Higher DPI for local testing
                }
            }
        }
        
        advanced_catalog = {
            'training_metrics_plot': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/figures/exploratory/training_metrics.png',
                'purpose': 'exploratory',
                'condition_param': 'environment',
                'versioned': True
            },
            'production_dashboard': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/figures/production/dashboard.png',
                'purpose': 'production',
                'condition_param': 'environment',
                'versioned': True
            }
        }
        
        advanced_manager.configure_project(
            figregistry_config=advanced_base_config,
            local_overrides=advanced_local_overrides,
            catalog_entries=advanced_catalog
        )
        
        managers['advanced'] = advanced_manager
        
        yield managers
        
    finally:
        # Clean up all managers
        for manager in managers.values():
            manager.cleanup_project(force=True)


@pytest.fixture 
def project_scaffolding_fixture(tmp_path) -> Generator[callable, None, None]:
    """
    Configurable project template creation with plugin installation.
    
    Provides a callable factory for creating projects with different
    configurations and templates based on test requirements.
    
    Yields:
        Callable that creates and configures KedroProjectManager instances
    """
    created_managers = []
    
    def create_scaffolded_project(
        template: str = "spaceflights",
        config_scenario: str = "basic",
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> KedroProjectManager:
        """
        Create a scaffolded project with specified configuration.
        
        Args:
            template: Kedro starter template name
            config_scenario: Configuration scenario (basic, advanced, minimal)
            tools: Additional Kedro tools to include
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured KedroProjectManager instance
        """
        project_name = f"scaffolded_{template}_{config_scenario}"
        manager = KedroProjectManager(tmp_path / project_name)
        
        # Create project
        manager.create_project(
            project_name=project_name,
            starter=template,
            tools=tools or [],
            example=True
        )
        
        # Install plugin
        manager.install_plugin()
        
        # Apply configuration scenario
        if config_scenario == "basic":
            config = {
                'styles': {'default': {'figure.figsize': [8, 6]}},
                'output': {'path_aliases': {'out': 'data/08_reporting'}}
            }
            catalog = {
                'test_figure': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': 'data/08_reporting/test_figure.png'
                }
            }
        elif config_scenario == "advanced":
            config = {
                'styles': {
                    'dev': {'figure.figsize': [10, 6]},
                    'prod': {'figure.figsize': [12, 9], 'figure.dpi': 300}
                },
                'output': {'versioning': True}
            }
            catalog = {
                'dev_plot': {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': 'data/08_reporting/dev_plot.png',
                    'condition_param': 'env',
                    'versioned': True
                }
            }
        else:  # minimal
            config = {'styles': {'default': {}}}
            catalog = {}
            
        manager.configure_project(
            figregistry_config=config,
            catalog_entries=catalog
        )
        
        created_managers.append(manager)
        return manager
    
    try:
        yield create_scaffolded_project
    finally:
        # Clean up all created managers
        for manager in created_managers:
            manager.cleanup_project(force=True)


@pytest.fixture
def project_cleanup_fixture():
    """
    Comprehensive rollback and state reset per Section 6.6.7.5.
    
    Provides cleanup utilities for handling failed tests and ensuring
    proper state reset between test executions.
    
    Yields:
        Cleanup utilities for project state management
    """
    cleanup_paths = []
    original_cwd = Path.cwd()
    
    def register_cleanup_path(path: Path) -> None:
        """Register a path for cleanup."""
        cleanup_paths.append(path)
    
    def cleanup_all(force: bool = True) -> List[str]:
        """
        Clean up all registered paths and reset state.
        
        Args:
            force: Whether to ignore cleanup errors
            
        Returns:
            List of cleanup error messages (empty if successful)
        """
        errors = []
        
        # Reset working directory
        try:
            os.chdir(original_cwd)
        except Exception as e:
            if not force:
                errors.append(f"Failed to reset working directory: {e}")
                
        # Remove all registered paths
        for path in cleanup_paths:
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
            except Exception as e:
                if not force:
                    errors.append(f"Failed to remove {path}: {e}")
                    
        # Clear registry
        cleanup_paths.clear()
        
        return errors
    
    cleanup_utils = {
        'register': register_cleanup_path,
        'cleanup_all': cleanup_all,
        'paths': cleanup_paths
    }
    
    try:
        yield cleanup_utils
    finally:
        # Final cleanup with error suppression
        cleanup_all(force=True)


@pytest.fixture
def multi_environment_project_fixture(tmp_path) -> Generator[KedroProjectManager, None, None]:
    """
    Multi-environment configuration testing with conf/base and conf/local scenarios.
    
    Creates a project configured for environment-specific configuration testing,
    validating proper precedence rules and configuration merging behavior.
    
    Yields:
        KedroProjectManager configured for multi-environment testing
    """
    manager = KedroProjectManager(tmp_path)
    
    try:
        # Create project
        manager.create_project(
            project_name="multi_env_project",
            starter="spaceflights"
        )
        manager.install_plugin()
        
        # Base configuration (shared across environments)
        base_config = {
            'styles': {
                'exploratory': {
                    'figure.figsize': [8, 6],
                    'axes.labelsize': 10,
                    'font.family': 'sans-serif'
                },
                'production': {
                    'figure.figsize': [12, 8],
                    'axes.labelsize': 12,
                    'font.family': 'serif'
                }
            },
            'output': {
                'path_aliases': {
                    'expl': 'data/08_reporting/exploratory',
                    'prod': 'data/08_reporting/production'
                }
            }
        }
        
        # Local overrides (environment-specific)
        local_overrides = {
            'styles': {
                'exploratory': {
                    'figure.dpi': 150,  # Higher DPI for local development
                    'axes.grid': True   # Grid enabled for local testing
                },
                'production': {
                    'figure.dpi': 300,  # Publication quality
                    'savefig.bbox': 'tight'
                }
            },
            'output': {
                'versioning': True  # Enable versioning in local environment
            }
        }
        
        # Catalog with environment-aware entries
        catalog_entries = {
            'env_exploratory_plot': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/exploratory/env_plot.png',
                'purpose': 'exploratory',
                'condition_param': 'environment'
            },
            'env_production_chart': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/production/env_chart.png',
                'purpose': 'production', 
                'condition_param': 'environment'
            }
        }
        
        manager.configure_project(
            figregistry_config=base_config,
            local_overrides=local_overrides,
            catalog_entries=catalog_entries
        )
        
        yield manager
        
    finally:
        manager.cleanup_project(force=True)


@pytest.fixture
def migration_project_fixture(tmp_path) -> Generator[Tuple[KedroProjectManager, KedroProjectManager], None, None]:
    """
    Migration testing fixture for conversion from manual plt.savefig() to automated workflows.
    
    Creates a before/after project pair demonstrating migration from manual figure
    management to automated FigRegistry plugin integration.
    
    Yields:
        Tuple of (before_manager, after_manager) for migration scenario testing
    """
    before_manager = KedroProjectManager(tmp_path / "before")
    after_manager = KedroProjectManager(tmp_path / "after")
    
    try:
        # Create "before" project - manual figure management
        before_path = before_manager.create_project(
            project_name="manual_figures_project",
            starter="spaceflights"
        )
        
        # Configure before project with manual plt.savefig() pattern
        manual_catalog = {
            'manual_plot_data': {
                'type': 'pandas.CSVDataSet',
                'filepath': 'data/03_primary/manual_plot_data.csv'
            }
            # Note: No FigureDataSet entries - figures saved manually in nodes
        }
        
        before_manager.configure_project(
            catalog_entries=manual_catalog,
            hooks_config=False  # No FigRegistry hooks
        )
        
        # Create sample pipeline node file with manual saving
        src_dirs = list(before_path.glob("src/*"))
        if src_dirs:
            pipelines_dir = src_dirs[0] / "pipelines" / "data_visualization"
            pipelines_dir.mkdir(parents=True, exist_ok=True)
            
            manual_node_code = '''"""Manual figure generation node."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def create_manual_plot(data: pd.DataFrame) -> None:
    """Create plot with manual plt.savefig() call."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create sample plot
    ax.plot(data.index, data.values if hasattr(data, 'values') else range(len(data)))
    ax.set_title("Manual Plot Example")
    ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    
    # Manual figure saving with hardcoded styling
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    
    # Create output directory manually
    output_dir = Path("data/08_reporting/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with manual path management
    plt.savefig(output_dir / "manual_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
'''
            
            with open(pipelines_dir / "nodes.py", 'w') as f:
                f.write(manual_node_code)
        
        # Create "after" project - automated FigRegistry management
        after_path = after_manager.create_project(
            project_name="automated_figures_project",
            starter="spaceflights"
        )
        after_manager.install_plugin()
        
        # Configure after project with FigRegistry automation
        automated_config = {
            'styles': {
                'default': {
                    'figure.figsize': [10, 6],
                    'figure.facecolor': 'white',
                    'axes.titlesize': 14,
                    'axes.labelsize': 12,
                    'savefig.dpi': 300,
                    'savefig.bbox': 'tight'
                }
            },
            'output': {
                'path_aliases': {
                    'figures': 'data/08_reporting/figures'
                }
            }
        }
        
        automated_catalog = {
            'manual_plot_data': {
                'type': 'pandas.CSVDataSet',
                'filepath': 'data/03_primary/manual_plot_data.csv'
            },
            'automated_plot': {
                'type': 'figregistry_kedro.datasets.FigureDataSet',
                'filepath': 'data/08_reporting/figures/automated_plot.png',
                'purpose': 'default',
                'style_params': {'condition': 'default'}
            }
        }
        
        after_manager.configure_project(
            figregistry_config=automated_config,
            catalog_entries=automated_catalog
        )
        
        # Create automated pipeline node - returns figure object
        src_dirs = list(after_path.glob("src/*"))
        if src_dirs:
            pipelines_dir = src_dirs[0] / "pipelines" / "data_visualization"
            pipelines_dir.mkdir(parents=True, exist_ok=True)
            
            automated_node_code = '''"""Automated figure generation node."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

def create_automated_plot(data: pd.DataFrame) -> Figure:
    """Create plot returning figure object for FigRegistry automation."""
    fig, ax = plt.subplots()  # No manual sizing - handled by FigRegistry
    
    # Create sample plot
    ax.plot(data.index, data.values if hasattr(data, 'values') else range(len(data)))
    ax.set_title("Automated Plot Example")
    ax.set_xlabel("Index") 
    ax.set_ylabel("Values")
    
    # No manual styling or saving - handled by FigRegistry
    return fig
'''
            
            with open(pipelines_dir / "nodes.py", 'w') as f:
                f.write(automated_node_code)
        
        yield before_manager, after_manager
        
    finally:
        before_manager.cleanup_project(force=True)
        after_manager.cleanup_project(force=True)


# Utility functions for project fixture validation

def validate_project_structure(project_path: Path) -> bool:
    """
    Validate that a Kedro project has proper structure.
    
    Args:
        project_path: Path to the Kedro project
        
    Returns:
        True if project structure is valid
    """
    required_paths = [
        project_path / "conf",
        project_path / "conf" / "base",
        project_path / "data",
        project_path / "src"
    ]
    
    return all(path.exists() for path in required_paths)


def validate_plugin_installation(project_path: Path) -> bool:
    """
    Validate that figregistry-kedro plugin is properly installed.
    
    Args:
        project_path: Path to the Kedro project
        
    Returns:
        True if plugin is available
    """
    try:
        result = subprocess.run(
            ["python", "-c", "import figregistry_kedro; print('OK')"],
            capture_output=True,
            text=True,
            cwd=project_path,
            timeout=10
        )
        return result.returncode == 0 and "OK" in result.stdout
    except:
        return False


def validate_figregistry_config(project_path: Path) -> bool:
    """
    Validate that FigRegistry configuration exists and is valid.
    
    Args:
        project_path: Path to the Kedro project
        
    Returns:
        True if configuration is valid
    """
    config_path = project_path / "conf" / "base" / "figregistry.yml"
    if not config_path.exists():
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return isinstance(config, dict) and 'styles' in config
    except:
        return False


# Performance benchmarking utilities for project fixtures

def benchmark_project_creation(tmp_path: Path, iterations: int = 3) -> Dict[str, float]:
    """
    Benchmark project creation performance for optimization.
    
    Args:
        tmp_path: Temporary directory for testing
        iterations: Number of iterations to average
        
    Returns:
        Dictionary with timing metrics
    """
    import time
    
    timings = {
        'project_creation': [],
        'plugin_installation': [],
        'configuration': [],
        'total': []
    }
    
    for i in range(iterations):
        manager = KedroProjectManager(tmp_path / f"benchmark_{i}")
        
        try:
            # Time project creation
            start = time.time()
            manager.create_project(f"benchmark_project_{i}")
            project_time = time.time() - start
            timings['project_creation'].append(project_time)
            
            # Time plugin installation
            start = time.time()
            manager.install_plugin()
            install_time = time.time() - start
            timings['plugin_installation'].append(install_time)
            
            # Time configuration
            start = time.time()
            manager.configure_project(
                figregistry_config={'styles': {'default': {}}},
                catalog_entries={'test': {'type': 'pandas.CSVDataSet', 'filepath': 'test.csv'}}
            )
            config_time = time.time() - start
            timings['configuration'].append(config_time)
            
            timings['total'].append(project_time + install_time + config_time)
            
        finally:
            manager.cleanup_project(force=True)
    
    # Calculate averages
    return {
        key: sum(values) / len(values) 
        for key, values in timings.items()
    }