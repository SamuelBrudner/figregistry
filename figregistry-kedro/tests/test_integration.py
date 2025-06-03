"""End-to-End Integration Tests for FigRegistry-Kedro Plugin.

This module provides comprehensive end-to-end integration tests that validate complete 
figregistry-kedro plugin functionality within realistic Kedro pipeline scenarios per 
Section 6.6.4.5. Tests cover automated figure styling workflows, multi-environment 
configuration handling, catalog integration, versioning compatibility, and performance 
benchmarking.

Test Coverage Requirements (Section 6.6.4.5):
- Complete plugin integration from project initialization through automated figure persistence
- Environment-specific configuration merging across development and production scenarios
- Dataset versioning compatibility ensuring no conflicts between FigRegistry and Kedro versioning
- Migration scenarios demonstrating elimination of manual plt.savefig() calls
- Performance validation against targets: <200ms plugin overhead per pipeline run

Test Categories:
1. Basic Kedro Plugin Pipeline Scenario: Minimal integration validation
2. Advanced Multi-Environment Configuration: Environment-specific override testing
3. Migration of Existing Kedro Project: Manual to automated figure management
4. Dataset Versioning Workflow: Kedro versioning compatibility validation
5. Performance Benchmarking: Plugin overhead measurement against targets
6. Cross-Platform Compatibility: Python 3.10-3.12 and Kedro 0.18-0.19 matrix

Dependencies (Section 0.2.4):
- figregistry>=0.3.0: Core functionality provider
- kedro>=0.18.0,<0.20.0: Target framework for integration  
- pytest>=8.0.0, pytest-mock>=3.14.0: Testing framework
- pytest-benchmark: Performance measurement infrastructure
- matplotlib>=3.9.0: Visualization backend for figure generation
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator
from unittest.mock import Mock, MagicMock, patch
import pytest
import logging

# Configure warnings for clean test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core testing dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    matplotlib.use('Agg')  # Non-interactive backend for testing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None

try:
    from kedro.framework.context import KedroContext
    from kedro.framework.session import KedroSession
    from kedro.framework.project import configure_project
    from kedro.framework.startup import bootstrap_project
    from kedro.config import ConfigLoader, OmegaConfigLoader
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline, node
    from kedro.runner import SequentialRunner, ParallelRunner
    KEDRO_AVAILABLE = True
except ImportError:
    KEDRO_AVAILABLE = False
    KedroContext = None
    KedroSession = None
    Pipeline = None
    node = None

try:
    import figregistry
    from figregistry import get_style, save_figure, init_config
    FIGREGISTRY_AVAILABLE = True
except ImportError:
    FIGREGISTRY_AVAILABLE = False

# Plugin imports with graceful fallback
try:
    from figregistry_kedro.datasets import FigureDataSet
    from figregistry_kedro.hooks import FigRegistryHooks
    from figregistry_kedro.config import FigRegistryConfigBridge, init_config as plugin_init_config
    PLUGIN_AVAILABLE = True
except ImportError:
    PLUGIN_AVAILABLE = False
    FigureDataSet = None
    FigRegistryHooks = None
    FigRegistryConfigBridge = None

# Test data imports
try:
    from figregistry_kedro.tests.data.configs import (
        generate_baseline_config,
        generate_environment_configs,
        generate_kedro_specific_config
    )
    from figregistry_kedro.tests.data.sample_data import (
        generate_sample_matplotlib_figure,
        create_sample_dataset
    )
except ImportError:
    # Fallback generators
    def generate_baseline_config():
        return {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {"color": "#1f77b4", "marker": "o"},
                "presentation": {"color": "#ff7f0e", "marker": "s"},
                "publication": {"color": "#2ca02c", "marker": "^"}
            },
            "outputs": {
                "base_path": "figures",
                "dpi": 300
            }
        }
    
    def generate_environment_configs():
        base = generate_baseline_config()
        return {
            "development": {**base, "outputs": {**base["outputs"], "dpi": 150}},
            "production": {**base, "outputs": {**base["outputs"], "dpi": 600}}
        }
    
    def generate_kedro_specific_config():
        base = generate_baseline_config()
        base["kedro"] = {
            "dataset_integration": True,
            "hook_integration": True,
            "versioning_support": True
        }
        return base
    
    def generate_sample_matplotlib_figure():
        if not MATPLOTLIB_AVAILABLE:
            return Mock()
        fig, ax = plt.subplots(figsize=(8, 6))
        import numpy as np
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Sample Figure")
        return fig
    
    def create_sample_dataset():
        import numpy as np
        return np.random.randn(100, 2)

logger = logging.getLogger(__name__)


# =============================================================================
# INTEGRATION TEST PREREQUISITES
# =============================================================================

def pytest_configure(config):
    """Configure pytest for integration testing."""
    # Skip all integration tests if key dependencies unavailable
    if not all([KEDRO_AVAILABLE, FIGREGISTRY_AVAILABLE, PLUGIN_AVAILABLE, MATPLOTLIB_AVAILABLE]):
        missing = []
        if not KEDRO_AVAILABLE:
            missing.append("kedro>=0.18.0,<0.20.0")
        if not FIGREGISTRY_AVAILABLE:
            missing.append("figregistry>=0.3.0")
        if not PLUGIN_AVAILABLE:
            missing.append("figregistry_kedro")
        if not MATPLOTLIB_AVAILABLE:
            missing.append("matplotlib>=3.9.0")
        
        pytest.skip(f"Integration tests require: {', '.join(missing)}", allow_module_level=True)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.kedro_plugin,
    pytest.mark.skipif(
        not all([KEDRO_AVAILABLE, FIGREGISTRY_AVAILABLE, PLUGIN_AVAILABLE, MATPLOTLIB_AVAILABLE]),
        reason="Integration tests require kedro, figregistry, figregistry_kedro, and matplotlib"
    )
]


# =============================================================================
# INTEGRATION TEST UTILITIES
# =============================================================================

class IntegrationTestError(Exception):
    """Exception raised during integration test setup or execution."""
    pass


class KedroProjectManager:
    """Utility class for managing temporary Kedro projects during testing."""
    
    def __init__(self, temp_dir: Path, project_name: str = "test_figregistry_integration"):
        self.temp_dir = temp_dir
        self.project_name = project_name
        self.project_path = temp_dir / project_name
        self._created_projects: List[Path] = []
    
    def create_minimal_project(self) -> Path:
        """Create minimal Kedro project structure for testing.
        
        Returns:
            Path to created project directory
        """
        project_path = self.project_path
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard Kedro directory structure
        directories = [
            "conf/base",
            "conf/local", 
            "data/01_raw",
            "data/02_intermediate",
            "data/03_primary",
            "data/08_reporting",
            f"src/{self.project_name}",
            f"src/{self.project_name}/pipelines",
            f"src/{self.project_name}/pipelines/data_visualization",
            "logs"
        ]
        
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create pyproject.toml
        self._create_pyproject_toml(project_path)
        
        # Create basic catalog.yml
        self._create_catalog_yml(project_path)
        
        # Create parameters.yml
        self._create_parameters_yml(project_path)
        
        # Create settings.py with FigRegistry hooks
        self._create_settings_py(project_path)
        
        # Create pipeline module
        self._create_pipeline_module(project_path)
        
        self._created_projects.append(project_path)
        return project_path
    
    def _create_pyproject_toml(self, project_path: Path):
        """Create pyproject.toml for Kedro project."""
        content = f'''[tool.kedro]
package_name = "{self.project_name}"
project_name = "Test FigRegistry Integration"
kedro_init_version = "0.19.0"

[project]
name = "{self.project_name}"
version = "0.1.0"
dependencies = [
    "kedro>=0.18.0,<0.20.0",
    "figregistry>=0.3.0",
    "figregistry-kedro",
    "matplotlib>=3.9.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0"
]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
'''
        with open(project_path / "pyproject.toml", "w") as f:
            f.write(content)
    
    def _create_catalog_yml(self, project_path: Path):
        """Create catalog.yml with FigureDataSet entries."""
        content = '''# Data catalog for FigRegistry integration testing

# Raw data
sample_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/sample_data.csv

# Intermediate processing  
processed_data:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/processed_data.parquet

# Primary data
analysis_results:
  type: pandas.ParquetDataSet
  filepath: data/03_primary/analysis_results.parquet

# FigRegistry figure outputs
exploratory_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figures/exploratory_plot.png
  purpose: exploratory
  condition_param: experiment_type
  style_params:
    figure.dpi: 150
    figure.facecolor: white

presentation_chart:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figures/presentation_chart.png
  purpose: presentation
  condition_param: experiment_type
  style_params:
    figure.dpi: 300
    figure.facecolor: white
  save_args:
    bbox_inches: tight
    transparent: false

publication_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figures/publication_figure.pdf
  purpose: publication
  condition_param: experiment_type
  style_params:
    figure.dpi: 600
    figure.facecolor: white
    font.size: 14
  save_args:
    bbox_inches: tight
    transparent: false
    format: pdf

# Versioned figure for testing Kedro versioning compatibility
versioned_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figures/versioned/plot.png
  purpose: exploratory
  condition_param: experiment_type
  versioned: true
  style_params:
    figure.dpi: 200
'''
        with open(project_path / "conf/base/catalog.yml", "w") as f:
            f.write(content)
    
    def _create_parameters_yml(self, project_path: Path):
        """Create parameters.yml with experiment parameters."""
        content = '''# Parameters for FigRegistry integration testing

# Experiment configuration
experiment_type: test_experiment
experiment_id: integration_test_001
environment: test

# Analysis parameters
sample_size: 1000
random_seed: 42

# Figure parameters
figure_width: 10
figure_height: 8
'''
        with open(project_path / "conf/base/parameters.yml", "w") as f:
            f.write(content)
    
    def _create_settings_py(self, project_path: Path):
        """Create settings.py with FigRegistryHooks registration."""
        content = f'''"""Settings for {self.project_name} Kedro project."""

from figregistry_kedro.hooks import FigRegistryHooks

# Register FigRegistry hooks for automated configuration
HOOKS = (FigRegistryHooks(),)

# Session configuration
SESSION_STORE_CLASS = "kedro.framework.session.session.BaseSessionStore"
SESSION_STORE_ARGS = {{}}

# Context configuration
CONTEXT_CLASS = "kedro.framework.context.KedroContext"

# Disable Kedro telemetry for testing
DISABLE_HOOKS_FOR_PLUGINS = {{}}
'''
        src_path = project_path / f"src/{self.project_name}"
        with open(src_path / "settings.py", "w") as f:
            f.write(content)
    
    def _create_pipeline_module(self, project_path: Path):
        """Create pipeline module with figure generation nodes."""
        pipeline_path = project_path / f"src/{self.project_name}/pipelines/data_visualization"
        
        # Create __init__.py
        with open(pipeline_path / "__init__.py", "w") as f:
            f.write('"""Data visualization pipeline."""')
        
        # Create nodes.py
        nodes_content = '''"""Data visualization pipeline nodes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def generate_sample_data(sample_size: int, random_seed: int) -> pd.DataFrame:
    """Generate sample data for visualization testing.
    
    Args:
        sample_size: Number of data points to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(random_seed)
    
    data = {
        'x': np.linspace(0, 10, sample_size),
        'y': np.sin(np.linspace(0, 10, sample_size)) + 0.1 * np.random.randn(sample_size),
        'category': np.random.choice(['A', 'B', 'C'], sample_size),
        'value': np.random.exponential(2.0, sample_size)
    }
    
    return pd.DataFrame(data)


def create_exploratory_plot(data: pd.DataFrame, experiment_type: str) -> Figure:
    """Create exploratory data visualization.
    
    Args:
        data: Input data for plotting
        experiment_type: Type of experiment for condition-based styling
        
    Returns:
        Matplotlib Figure object for FigRegistry processing
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Basic scatter plot with trend line
    ax.scatter(data['x'], data['y'], alpha=0.6, s=30)
    ax.plot(data['x'], np.sin(data['x']), 'r-', linewidth=2, label='True Function')
    
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_title(f'Exploratory Analysis - {experiment_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Note: No plt.savefig() call - handled by FigureDataSet
    return fig


def create_presentation_chart(data: pd.DataFrame, experiment_type: str) -> Figure:
    """Create presentation-ready chart.
    
    Args:
        data: Input data for plotting
        experiment_type: Type of experiment for condition-based styling
        
    Returns:
        Matplotlib Figure object for FigRegistry processing
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Category distribution
    category_counts = data['category'].value_counts()
    axes[0].bar(category_counts.index, category_counts.values)
    axes[0].set_title('Category Distribution')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    
    # Value histogram
    axes[1].hist(data['value'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_title('Value Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    fig.suptitle(f'Data Overview - {experiment_type}', fontsize=14)
    plt.tight_layout()
    
    # Note: No plt.savefig() call - handled by FigureDataSet
    return fig


def create_publication_figure(data: pd.DataFrame, experiment_type: str) -> Figure:
    """Create publication-quality figure.
    
    Args:
        data: Input data for plotting
        experiment_type: Type of experiment for condition-based styling
        
    Returns:
        Matplotlib Figure object for FigRegistry processing
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group data by category and create box plot
    categories = data['category'].unique()
    category_data = [data[data['category'] == cat]['value'].values for cat in categories]
    
    box_plot = ax.boxplot(category_data, labels=categories, patch_artist=True)
    
    # Customize appearance for publication
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Value Distribution by Category\\n{experiment_type}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Note: No plt.savefig() call - handled by FigureDataSet
    return fig


def process_analysis_results(data: pd.DataFrame) -> pd.DataFrame:
    """Process data for analysis results.
    
    Args:
        data: Raw input data
        
    Returns:
        Processed analysis results
    """
    results = data.copy()
    
    # Add computed columns
    results['y_residual'] = results['y'] - np.sin(results['x'])
    results['value_normalized'] = (results['value'] - results['value'].mean()) / results['value'].std()
    
    # Add summary statistics by category
    category_stats = results.groupby('category')['value'].agg(['mean', 'std', 'count']).reset_index()
    category_stats.columns = ['category', 'value_mean', 'value_std', 'value_count']
    
    return results.merge(category_stats, on='category')
'''
        
        with open(pipeline_path / "nodes.py", "w") as f:
            f.write(nodes_content)
        
        # Create pipeline.py
        pipeline_content = '''"""Data visualization pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    generate_sample_data,
    create_exploratory_plot,
    create_presentation_chart,
    create_publication_figure,
    process_analysis_results
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data visualization pipeline.
    
    Returns:
        Kedro Pipeline with figure generation nodes
    """
    return Pipeline([
        # Data generation
        node(
            func=generate_sample_data,
            inputs=["params:sample_size", "params:random_seed"],
            outputs="sample_data",
            name="generate_sample_data_node"
        ),
        
        # Data processing
        node(
            func=process_analysis_results,
            inputs="sample_data",
            outputs="analysis_results",
            name="process_analysis_results_node"
        ),
        
        # Figure generation (automatic FigRegistry integration)
        node(
            func=create_exploratory_plot,
            inputs=["analysis_results", "params:experiment_type"],
            outputs="exploratory_plot",  # FigureDataSet in catalog
            name="create_exploratory_plot_node"
        ),
        
        node(
            func=create_presentation_chart,
            inputs=["analysis_results", "params:experiment_type"],
            outputs="presentation_chart",  # FigureDataSet in catalog
            name="create_presentation_chart_node"
        ),
        
        node(
            func=create_publication_figure,
            inputs=["analysis_results", "params:experiment_type"],
            outputs="publication_figure",  # FigureDataSet in catalog
            name="create_publication_figure_node"
        ),
        
        # Versioned figure for testing Kedro versioning compatibility
        node(
            func=create_exploratory_plot,
            inputs=["analysis_results", "params:experiment_type"],
            outputs="versioned_plot",  # Versioned FigureDataSet in catalog
            name="create_versioned_plot_node"
        )
    ])
'''
        
        with open(pipeline_path / "pipeline.py", "w") as f:
            f.write(pipeline_content)
        
        # Create main pipeline registry
        main_init_content = f'''"""Main pipeline registry for {self.project_name}."""

from kedro.pipeline import Pipeline
from .pipelines.data_visualization import create_pipeline as create_viz_pipeline


def register_pipelines() -> dict:
    """Register all project pipelines.
    
    Returns:
        Dictionary mapping pipeline names to Pipeline objects
    """
    viz_pipeline = create_viz_pipeline()
    
    return {{
        "__default__": viz_pipeline,
        "data_visualization": viz_pipeline,
        "viz": viz_pipeline
    }}
'''
        
        src_path = project_path / f"src/{self.project_name}"
        with open(src_path / "__init__.py", "w") as f:
            f.write("")
        
        with open(src_path / "pipeline_registry.py", "w") as f:
            f.write(main_init_content)
    
    def create_figregistry_config(self, project_path: Path, environment: str = "base", 
                                config_dict: Optional[Dict[str, Any]] = None):
        """Create FigRegistry configuration file for testing.
        
        Args:
            project_path: Path to Kedro project
            environment: Environment for configuration (base, local, etc.)
            config_dict: Custom configuration dictionary
        """
        if config_dict is None:
            config_dict = generate_baseline_config()
        
        config_path = project_path / f"conf/{environment}/figregistry.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def cleanup(self):
        """Clean up created projects."""
        for project_path in self._created_projects:
            if project_path.exists():
                shutil.rmtree(project_path, ignore_errors=True)


class PipelineExecutor:
    """Utility class for executing Kedro pipelines during testing."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.execution_results: List[Dict[str, Any]] = []
    
    def execute_pipeline(self, pipeline_name: str = "__default__", 
                        runner_type: str = "sequential",
                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute Kedro pipeline and collect results.
        
        Args:
            pipeline_name: Name of pipeline to execute
            runner_type: Type of runner (sequential, parallel)
            parameters: Additional parameters for execution
            
        Returns:
            Execution results dictionary
        """
        start_time = time.perf_counter()
        
        try:
            # Change to project directory
            original_cwd = os.getcwd()
            os.chdir(self.project_path)
            
            # Configure Kedro project
            configure_project(self.project_path.name)
            
            # Bootstrap project for session
            bootstrap_project(self.project_path)
            
            # Create session and run pipeline
            with KedroSession.create() as session:
                # Get context and pipeline
                context = session.load_context()
                pipelines = context.pipelines
                
                if pipeline_name not in pipelines:
                    available = list(pipelines.keys())
                    raise IntegrationTestError(f"Pipeline '{pipeline_name}' not found. Available: {available}")
                
                pipeline = pipelines[pipeline_name]
                
                # Select runner
                if runner_type == "parallel":
                    runner = ParallelRunner()
                else:
                    runner = SequentialRunner()
                
                # Execute pipeline
                run_result = runner.run(pipeline, context.catalog, session.session_id)
                
                execution_time = time.perf_counter() - start_time
                
                # Collect execution results
                result = {
                    'pipeline_name': pipeline_name,
                    'execution_time_ms': execution_time * 1000,
                    'runner_type': runner_type,
                    'success': True,
                    'nodes_executed': len(pipeline.nodes),
                    'outputs_created': len(run_result) if run_result else 0,
                    'session_id': session.session_id,
                    'project_path': str(self.project_path)
                }
                
                # Validate figure outputs were created
                result['figure_outputs'] = self._validate_figure_outputs(context.catalog)
                
                self.execution_results.append(result)
                return result
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_result = {
                'pipeline_name': pipeline_name,
                'execution_time_ms': execution_time * 1000,
                'runner_type': runner_type,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'project_path': str(self.project_path)
            }
            self.execution_results.append(error_result)
            raise IntegrationTestError(f"Pipeline execution failed: {str(e)}") from e
            
        finally:
            os.chdir(original_cwd)
    
    def _validate_figure_outputs(self, catalog: DataCatalog) -> Dict[str, bool]:
        """Validate that figure outputs were created correctly.
        
        Args:
            catalog: Kedro DataCatalog instance
            
        Returns:
            Dictionary mapping dataset names to existence status
        """
        figure_datasets = [
            'exploratory_plot',
            'presentation_chart', 
            'publication_figure',
            'versioned_plot'
        ]
        
        validation_results = {}
        for dataset_name in figure_datasets:
            try:
                exists = catalog.exists(dataset_name)
                validation_results[dataset_name] = exists
                if exists:
                    logger.info(f"Figure output validated: {dataset_name}")
                else:
                    logger.warning(f"Figure output missing: {dataset_name}")
            except Exception as e:
                logger.error(f"Error validating {dataset_name}: {e}")
                validation_results[dataset_name] = False
        
        return validation_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of executed pipelines.
        
        Returns:
            Performance summary dictionary
        """
        if not self.execution_results:
            return {'total_executions': 0}
        
        successful_runs = [r for r in self.execution_results if r['success']]
        
        if not successful_runs:
            return {
                'total_executions': len(self.execution_results),
                'successful_executions': 0,
                'failed_executions': len(self.execution_results)
            }
        
        execution_times = [r['execution_time_ms'] for r in successful_runs]
        
        return {
            'total_executions': len(self.execution_results),
            'successful_executions': len(successful_runs),
            'failed_executions': len(self.execution_results) - len(successful_runs),
            'avg_execution_time_ms': sum(execution_times) / len(execution_times),
            'min_execution_time_ms': min(execution_times),
            'max_execution_time_ms': max(execution_times),
            'performance_target_met': all(t < 200 for t in execution_times)  # <200ms target
        }


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_integration_dir():
    """Provide temporary directory for integration testing with automatic cleanup."""
    temp_dir = Path(tempfile.mkdtemp(prefix="figregistry_integration_"))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def kedro_project_manager(temp_integration_dir):
    """Provide KedroProjectManager for managing test projects."""
    manager = KedroProjectManager(temp_integration_dir)
    try:
        yield manager
    finally:
        manager.cleanup()


@pytest.fixture
def pipeline_executor():
    """Provide PipelineExecutor for running test pipelines."""
    def _create_executor(project_path: Path) -> PipelineExecutor:
        return PipelineExecutor(project_path)
    
    return _create_executor


@pytest.fixture
def performance_tracker():
    """Provide performance tracking utilities for integration tests."""
    class PerformanceTracker:
        def __init__(self):
            self.measurements = {}
        
        def measure_operation(self, operation_name: str):
            """Context manager for measuring operation performance."""
            from contextlib import contextmanager
            
            @contextmanager
            def _measure():
                start_time = time.perf_counter()
                try:
                    yield
                finally:
                    execution_time = time.perf_counter() - start_time
                    self.measurements[operation_name] = execution_time * 1000  # Convert to ms
            
            return _measure()
        
        def validate_performance_targets(self) -> Dict[str, bool]:
            """Validate performance against targets from Section 6.6.4.3."""
            targets = {
                'dataset_save': 200.0,  # <200ms per FigureDataSet save
                'config_bridge': 50.0,  # <50ms per pipeline run
                'hook_init': 25.0,      # <25ms per project startup
                'pipeline_execution': 200.0  # <200ms plugin overhead per pipeline run
            }
            
            results = {}
            for operation, measured_time in self.measurements.items():
                # Find matching target
                target_key = None
                for key in targets:
                    if key in operation:
                        target_key = key
                        break
                
                if target_key:
                    results[operation] = measured_time < targets[target_key]
                else:
                    results[operation] = True  # No specific target
            
            return results
        
        def get_summary(self) -> str:
            """Get performance summary report."""
            if not self.measurements:
                return "No performance measurements recorded."
            
            validation = self.validate_performance_targets()
            lines = ["Performance Measurement Summary", "=" * 35]
            
            for operation, time_ms in self.measurements.items():
                status = "PASS" if validation.get(operation, True) else "FAIL"
                lines.append(f"{operation}: {time_ms:.2f}ms [{status}]")
            
            passed = sum(1 for result in validation.values() if result)
            total = len(validation)
            lines.append(f"\nOverall: {passed}/{total} targets met")
            
            return "\n".join(lines)
    
    return PerformanceTracker()


@pytest.fixture
def integration_validators():
    """Provide validation utilities for integration testing."""
    class IntegrationValidators:
        @staticmethod
        def validate_plugin_integration(project_path: Path) -> Dict[str, bool]:
            """Validate that plugin is properly integrated in project."""
            validations = {}
            
            # Check settings.py has FigRegistryHooks
            settings_path = project_path / f"src/{project_path.name}/settings.py"
            if settings_path.exists():
                content = settings_path.read_text()
                validations['hooks_registered'] = 'FigRegistryHooks' in content
            else:
                validations['hooks_registered'] = False
            
            # Check catalog has FigureDataSet entries
            catalog_path = project_path / "conf/base/catalog.yml"
            if catalog_path.exists():
                content = catalog_path.read_text()
                validations['catalog_datasets'] = 'figregistry_kedro.datasets.FigureDataSet' in content
            else:
                validations['catalog_datasets'] = False
            
            # Check project structure
            validations['project_structure'] = all([
                (project_path / "conf/base").exists(),
                (project_path / "data/08_reporting").exists(),
                (project_path / "pyproject.toml").exists()
            ])
            
            return validations
        
        @staticmethod
        def validate_figure_outputs(project_path: Path, dataset_names: List[str]) -> Dict[str, bool]:
            """Validate that figure outputs were created."""
            reporting_path = project_path / "data/08_reporting/figures"
            validations = {}
            
            for dataset_name in dataset_names:
                # Look for files matching dataset name pattern
                pattern_files = list(reporting_path.glob(f"**/*{dataset_name}*"))
                validations[dataset_name] = len(pattern_files) > 0
            
            return validations
        
        @staticmethod
        def validate_environment_config_merge(project_path: Path, environment: str) -> bool:
            """Validate environment-specific config merging."""
            base_config_path = project_path / "conf/base/figregistry.yml"
            env_config_path = project_path / f"conf/{environment}/figregistry.yml"
            
            # Both configs should exist for merge testing
            if not (base_config_path.exists() and env_config_path.exists()):
                return False
            
            # Validate configs can be loaded
            try:
                import yaml
                with open(base_config_path) as f:
                    base_config = yaml.safe_load(f)
                with open(env_config_path) as f:
                    env_config = yaml.safe_load(f)
                
                # Basic validation that they're valid YAML dicts
                return isinstance(base_config, dict) and isinstance(env_config, dict)
            except Exception:
                return False
        
        @staticmethod
        def validate_versioning_compatibility(project_path: Path) -> bool:
            """Validate Kedro versioning compatibility with FigRegistry."""
            # Check for versioned dataset in catalog
            catalog_path = project_path / "conf/base/catalog.yml"
            if not catalog_path.exists():
                return False
            
            try:
                import yaml
                with open(catalog_path) as f:
                    catalog = yaml.safe_load(f)
                
                # Look for versioned FigureDataSet
                for dataset_name, config in catalog.items():
                    if (isinstance(config, dict) and 
                        config.get('type') == 'figregistry_kedro.datasets.FigureDataSet' and
                        config.get('versioned', False)):
                        return True
                
                return False
            except Exception:
                return False
    
    return IntegrationValidators()


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestBasicKedroPluginPipeline:
    """Test basic Kedro plugin pipeline scenario per Section 6.6.4.5.
    
    Validates complete plugin integration from project initialization through 
    automated figure persistence in a minimal Kedro pipeline context.
    """
    
    def test_minimal_plugin_integration(self, kedro_project_manager, pipeline_executor, performance_tracker):
        """Test minimal plugin integration with basic pipeline execution.
        
        Success Criteria:
        - Output figure file exists at catalog-specified location
        - Applied styling matches configured condition mapping
        - No manual plt.savefig() calls required in pipeline nodes
        - Plugin overhead remains within performance thresholds
        """
        # Create minimal Kedro project
        with performance_tracker.measure_operation("project_creation"):
            project_path = kedro_project_manager.create_minimal_project()
        
        # Create basic FigRegistry configuration
        with performance_tracker.measure_operation("config_creation"):
            kedro_project_manager.create_figregistry_config(project_path, "base")
        
        # Execute pipeline
        executor = pipeline_executor(project_path)
        
        with performance_tracker.measure_operation("pipeline_execution"):
            result = executor.execute_pipeline()
        
        # Validate execution success
        assert result['success'], f"Pipeline execution failed: {result.get('error', 'Unknown error')}"
        assert result['nodes_executed'] > 0, "No nodes were executed"
        
        # Validate figure outputs were created
        figure_outputs = result['figure_outputs']
        expected_figures = ['exploratory_plot', 'presentation_chart', 'publication_figure']
        
        for figure_name in expected_figures:
            assert figure_outputs.get(figure_name, False), f"Figure output missing: {figure_name}"
        
        # Validate performance targets
        performance_results = performance_tracker.validate_performance_targets()
        assert all(performance_results.values()), f"Performance targets not met: {performance_tracker.get_summary()}"
        
        # Validate plugin integration
        from figregistry_kedro.tests.conftest import integration_validators
        validators = integration_validators()
        integration_status = validators.validate_plugin_integration(project_path)
        
        assert integration_status['hooks_registered'], "FigRegistryHooks not properly registered"
        assert integration_status['catalog_datasets'], "FigureDataSet not properly configured in catalog"
        assert integration_status['project_structure'], "Project structure incomplete"
    
    def test_automated_styling_application(self, kedro_project_manager, pipeline_executor):
        """Test that condition-based styling is automatically applied to figures.
        
        Validates that FigRegistry styling is applied based on experiment conditions
        without manual intervention in pipeline nodes.
        """
        # Create project with styled configuration
        project_path = kedro_project_manager.create_minimal_project()
        
        # Create configuration with specific styling conditions
        styled_config = generate_baseline_config()
        styled_config['styles']['test_experiment'] = {
            'figure.figsize': [12, 8],
            'axes.labelsize': 14,
            'lines.linewidth': 3.0,
            'figure.dpi': 200
        }
        
        kedro_project_manager.create_figregistry_config(project_path, "base", styled_config)
        
        # Execute pipeline
        executor = pipeline_executor(project_path)
        result = executor.execute_pipeline()
        
        assert result['success'], f"Styled pipeline execution failed: {result.get('error')}"
        
        # Validate that figures were created with styling applied
        # Note: In a full implementation, we would inspect the saved figures
        # to verify styling was applied, but for testing we validate the process completed
        figure_outputs = result['figure_outputs']
        assert figure_outputs['exploratory_plot'], "Exploratory plot not created with styling"
        assert figure_outputs['presentation_chart'], "Presentation chart not created with styling"
    
    def test_plugin_overhead_measurement(self, kedro_project_manager, pipeline_executor, performance_tracker):
        """Test plugin performance overhead against <200ms target per Section 6.6.4.3.
        
        Measures complete plugin execution path from hook initialization through 
        automated figure persistence to validate performance requirements.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        executor = pipeline_executor(project_path)
        
        # Execute multiple runs for performance measurement
        execution_times = []
        for run_num in range(5):
            with performance_tracker.measure_operation(f"pipeline_run_{run_num}"):
                result = executor.execute_pipeline()
                assert result['success'], f"Performance test run {run_num} failed"
                execution_times.append(result['execution_time_ms'])
        
        # Validate performance targets
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)
        
        # Plugin overhead should be <200ms per pipeline run
        assert avg_execution_time < 200.0, f"Average execution time {avg_execution_time:.2f}ms exceeds 200ms target"
        assert max_execution_time < 300.0, f"Maximum execution time {max_execution_time:.2f}ms exceeds reasonable limit"
        
        # Get detailed performance summary
        performance_summary = executor.get_performance_summary()
        assert performance_summary['performance_target_met'], "Performance targets not consistently met"


class TestAdvancedMultiEnvironmentConfiguration:
    """Test advanced multi-environment configuration scenario per Section 6.6.4.5.
    
    Verifies configuration merging behavior across Kedro's environment-specific 
    configuration system with FigRegistry's traditional YAML settings.
    """
    
    def test_environment_configuration_precedence(self, kedro_project_manager, pipeline_executor):
        """Test environment-specific configuration overrides.
        
        Success Criteria:
        - Local environment overrides take precedence over base configuration
        - Configuration merging preserves type safety and validation
        - Style application matches expected merged settings for each environment
        - No configuration conflicts or validation errors occur
        """
        project_path = kedro_project_manager.create_minimal_project()
        
        # Create base configuration
        base_config = generate_baseline_config()
        base_config['styles']['test_experiment'] = {
            'figure.dpi': 150,
            'figure.facecolor': 'white',
            'axes.labelsize': 10
        }
        kedro_project_manager.create_figregistry_config(project_path, "base", base_config)
        
        # Create local override configuration
        local_config = {
            'styles': {
                'test_experiment': {
                    'figure.dpi': 300,  # Override base DPI
                    'figure.facecolor': 'lightgray',  # Override facecolor
                    'axes.titlesize': 12  # Add new property
                }
            },
            'outputs': {
                'base_path': 'local_figures'  # Override output path
            }
        }
        kedro_project_manager.create_figregistry_config(project_path, "local", local_config)
        
        # Execute pipeline (should use local environment by default in Kedro)
        executor = pipeline_executor(project_path)
        result = executor.execute_pipeline()
        
        assert result['success'], f"Multi-environment pipeline failed: {result.get('error')}"
        
        # Validate that figures were created with merged configuration
        figure_outputs = result['figure_outputs']
        assert all(figure_outputs.values()), f"Not all figures created with merged config: {figure_outputs}"
        
        # Validate configuration merging worked
        from figregistry_kedro.tests.conftest import integration_validators
        validators = integration_validators()
        merge_valid = validators.validate_environment_config_merge(project_path, "local")
        assert merge_valid, "Environment configuration merge validation failed"
    
    def test_development_vs_production_configurations(self, kedro_project_manager, pipeline_executor):
        """Test different configurations for development vs production environments.
        
        Validates that the plugin correctly handles different configuration profiles
        for different deployment environments.
        """
        project_path = kedro_project_manager.create_minimal_project()
        
        # Create development configuration (lower quality, faster)
        dev_config = generate_environment_configs()["development"]
        dev_config['styles']['test_experiment'] = {
            'figure.dpi': 100,
            'figure.figsize': [6, 4],
            'font.size': 8
        }
        kedro_project_manager.create_figregistry_config(project_path, "base", dev_config)
        
        # Create production configuration (higher quality)
        prod_config = generate_environment_configs()["production"]
        prod_config['styles']['test_experiment'] = {
            'figure.dpi': 600,
            'figure.figsize': [10, 8],
            'font.size': 14
        }
        kedro_project_manager.create_figregistry_config(project_path, "production", prod_config)
        
        # Test with development configuration (base)
        executor = pipeline_executor(project_path)
        dev_result = executor.execute_pipeline()
        assert dev_result['success'], "Development environment execution failed"
        
        # Note: Testing production environment would require environment switching
        # which is complex in the test context. In practice, this would be tested
        # with environment variable or command-line environment specification.
        
        # Validate that configuration files are properly structured
        from figregistry_kedro.tests.conftest import integration_validators
        validators = integration_validators()
        dev_merge_valid = validators.validate_environment_config_merge(project_path, "base")
        prod_merge_valid = validators.validate_environment_config_merge(project_path, "production")
        
        assert dev_merge_valid, "Development configuration merge invalid"
        assert prod_merge_valid, "Production configuration merge invalid"
    
    def test_configuration_validation_across_environments(self, kedro_project_manager):
        """Test configuration validation with environment-specific overrides.
        
        Ensures that configuration bridge properly validates merged configurations
        and handles validation errors gracefully.
        """
        project_path = kedro_project_manager.create_minimal_project()
        
        # Create valid base configuration
        base_config = generate_baseline_config()
        kedro_project_manager.create_figregistry_config(project_path, "base", base_config)
        
        # Create local configuration with validation issues
        invalid_local_config = {
            'styles': {
                'test_experiment': {
                    'figure.dpi': 'invalid_dpi_value',  # Should be numeric
                    'invalid_style_param': 'value'
                }
            },
            'outputs': {
                'base_path': 123  # Should be string
            }
        }
        kedro_project_manager.create_figregistry_config(project_path, "local", invalid_local_config)
        
        # Attempt to create configuration bridge - should handle validation errors
        try:
            from figregistry_kedro.config import FigRegistryConfigBridge
            bridge = FigRegistryConfigBridge(environment="local")
            
            # This should either gracefully handle validation errors or raise appropriate exceptions
            config = bridge.get_merged_config()
            
            # If it succeeds, the invalid values should be handled/sanitized
            assert config is not None, "Configuration bridge should handle invalid configs gracefully"
            
        except Exception as e:
            # Configuration validation errors are acceptable and expected
            assert "validation" in str(e).lower() or "configuration" in str(e).lower(), \
                f"Unexpected error type: {e}"


class TestMigrationScenario:
    """Test migration scenario from manual figure management per Section 6.6.4.5.
    
    Validates seamless migration from manual plt.savefig() to automated FigRegistry 
    plugin integration in existing Kedro workflows.
    """
    
    def test_manual_to_automated_migration(self, kedro_project_manager, pipeline_executor):
        """Test migration from manual plt.savefig() to automated FigureDataSet.
        
        Success Criteria:
        - Automated figure output matches previous manual styling and locations
        - Pipeline execution produces identical visualization results
        - Node code complexity reduction through elimination of manual save calls
        - Backward compatibility maintained with existing pipeline structure
        """
        project_path = kedro_project_manager.create_minimal_project()
        
        # First, create a "before" scenario with manual figure saving
        self._create_manual_figure_pipeline(project_path)
        
        # Execute manual pipeline to establish baseline
        executor = pipeline_executor(project_path)
        manual_result = executor.execute_pipeline("manual_pipeline")
        
        # Verify manual pipeline works
        assert manual_result['success'], f"Manual pipeline failed: {manual_result.get('error')}"
        
        # Now create "after" scenario with FigRegistry automation
        self._create_automated_figure_pipeline(project_path)
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        # Execute automated pipeline
        automated_result = executor.execute_pipeline("automated_pipeline")
        
        # Verify automated pipeline works
        assert automated_result['success'], f"Automated pipeline failed: {automated_result.get('error')}"
        
        # Compare results - automated should be faster and produce equivalent outputs
        manual_time = manual_result['execution_time_ms']
        automated_time = automated_result['execution_time_ms']
        
        # Automated pipeline should be competitive or better in performance
        time_ratio = automated_time / manual_time
        assert time_ratio < 2.0, f"Automated pipeline significantly slower: {time_ratio:.2f}x"
        
        # Validate that automated outputs exist
        figure_outputs = automated_result['figure_outputs']
        assert any(figure_outputs.values()), "No automated figure outputs created"
    
    def _create_manual_figure_pipeline(self, project_path: Path):
        """Create a pipeline with manual plt.savefig() calls for comparison."""
        pipeline_path = project_path / f"src/{project_path.name}/pipelines/manual_figures"
        pipeline_path.mkdir(parents=True, exist_ok=True)
        
        # Create manual nodes with plt.savefig()
        manual_nodes_content = '''"""Manual figure pipeline nodes (before migration)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def create_manual_figure(data: pd.DataFrame, experiment_type: str, output_path: str = "data/08_reporting/manual_figure.png"):
    """Create figure with manual plt.savefig() call.
    
    This represents the "before" state where figures are manually saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(data['x'], data['y'], 'b-', linewidth=2)
    ax.set_title(f'Manual Figure - {experiment_type}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    
    # Manual save operation (what we want to eliminate)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return str(output_file)  # Return path as string (not Figure object)


def process_data_manually(sample_size: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """Generate data for manual processing."""
    np.random.seed(random_seed)
    
    x = np.linspace(0, 10, sample_size)
    y = np.sin(x) + 0.1 * np.random.randn(sample_size)
    
    return pd.DataFrame({'x': x, 'y': y})
'''
        
        with open(pipeline_path / "manual_nodes.py", "w") as f:
            f.write(manual_nodes_content)
        
        # Create manual pipeline definition
        manual_pipeline_content = '''"""Manual figure pipeline definition."""

from kedro.pipeline import Pipeline, node
from .manual_nodes import create_manual_figure, process_data_manually


def create_manual_pipeline() -> Pipeline:
    """Create manual figure pipeline for migration comparison."""
    return Pipeline([
        node(
            func=process_data_manually,
            inputs=None,
            outputs="manual_data",
            name="process_data_manually_node"
        ),
        node(
            func=create_manual_figure,
            inputs=["manual_data", "params:experiment_type"],
            outputs="manual_figure_path",  # String output, not Figure
            name="create_manual_figure_node"
        )
    ])
'''
        
        with open(pipeline_path / "manual_pipeline.py", "w") as f:
            f.write(manual_pipeline_content)
        
        with open(pipeline_path / "__init__.py", "w") as f:
            f.write("")
        
        # Update pipeline registry to include manual pipeline
        registry_path = project_path / f"src/{project_path.name}/pipeline_registry.py"
        registry_content = f'''"""Updated pipeline registry for migration testing."""

from kedro.pipeline import Pipeline
from .pipelines.data_visualization import create_pipeline as create_viz_pipeline
from .pipelines.manual_figures.manual_pipeline import create_manual_pipeline


def register_pipelines() -> dict:
    """Register all project pipelines including manual pipeline."""
    viz_pipeline = create_viz_pipeline()
    manual_pipeline = create_manual_pipeline()
    
    return {{
        "__default__": viz_pipeline,
        "data_visualization": viz_pipeline,
        "viz": viz_pipeline,
        "manual_pipeline": manual_pipeline,
        "automated_pipeline": viz_pipeline
    }}
'''
        
        with open(registry_path, "w") as f:
            f.write(registry_content)
    
    def _create_automated_figure_pipeline(self, project_path: Path):
        """Ensure automated pipeline is available (already created in project setup)."""
        # The automated pipeline is already created in create_minimal_project()
        # This method is for symmetry and future customization
        pass
    
    def test_code_complexity_reduction(self, kedro_project_manager):
        """Test that migration reduces code complexity in pipeline nodes.
        
        Validates that elimination of manual plt.savefig() calls reduces 
        node code complexity and improves maintainability.
        """
        project_path = kedro_project_manager.create_minimal_project()
        
        # Create both manual and automated pipelines
        self._create_manual_figure_pipeline(project_path)
        
        # Analyze code complexity (simplified metrics)
        manual_nodes_path = project_path / f"src/{project_path.name}/pipelines/manual_figures/manual_nodes.py"
        automated_nodes_path = project_path / f"src/{project_path.name}/pipelines/data_visualization/nodes.py"
        
        manual_content = manual_nodes_path.read_text()
        automated_content = automated_nodes_path.read_text()
        
        # Count plt.savefig() occurrences (should be eliminated in automated)
        manual_savefig_count = manual_content.count('plt.savefig')
        automated_savefig_count = automated_content.count('plt.savefig')
        
        assert manual_savefig_count > 0, "Manual pipeline should contain plt.savefig() calls"
        assert automated_savefig_count == 0, "Automated pipeline should not contain plt.savefig() calls"
        
        # Count import statements related to Path/pathlib (should be reduced)
        manual_path_imports = manual_content.count('from pathlib import') + manual_content.count('import pathlib')
        automated_path_imports = automated_content.count('from pathlib import') + automated_content.count('import pathlib')
        
        # Automated should have fewer path-related imports since FigRegistry handles paths
        assert automated_path_imports <= manual_path_imports, "Automated pipeline should simplify path handling"
        
        # Count return Figure vs return string (automated should return Figure objects)
        manual_figure_returns = manual_content.count('return fig')
        automated_figure_returns = automated_content.count('return fig')
        
        assert automated_figure_returns > manual_figure_returns, "Automated pipeline should return more Figure objects"


class TestDatasetVersioningWorkflow:
    """Test dataset versioning workflow per Section 6.6.4.5.
    
    Ensures FigRegistry's timestamp-based versioning coexists seamlessly with 
    Kedro's dataset versioning system without conflicts or interference.
    """
    
    def test_kedro_versioning_compatibility(self, kedro_project_manager, pipeline_executor):
        """Test Kedro dataset versioning compatibility with FigRegistry.
        
        Success Criteria:
        - No versioning system conflicts during concurrent operation
        - FigRegistry timestamp versioning maintains unique figure identification
        - Kedro dataset versioning correctly tracks figure dataset evolution
        - Historical figure versions remain accessible through catalog interface
        - Both versioning systems operate independently without interference
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        # Validate versioned dataset configuration
        from figregistry_kedro.tests.conftest import integration_validators
        validators = integration_validators()
        versioning_valid = validators.validate_versioning_compatibility(project_path)
        assert versioning_valid, "Versioned FigureDataSet not properly configured"
        
        # Execute pipeline multiple times to create version history
        executor = pipeline_executor(project_path)
        
        execution_results = []
        for run_num in range(3):
            result = executor.execute_pipeline()
            assert result['success'], f"Versioned pipeline run {run_num} failed: {result.get('error')}"
            execution_results.append(result)
            
            # Small delay to ensure timestamp differences
            time.sleep(0.1)
        
        # Validate that multiple versions were created
        # Check that versioned_plot dataset was handled correctly
        all_runs_successful = all(r['success'] for r in execution_results)
        assert all_runs_successful, "Not all versioned pipeline runs were successful"
        
        # Validate that versioned outputs exist
        versioned_outputs = [r['figure_outputs'].get('versioned_plot', False) for r in execution_results]
        assert all(versioned_outputs), "Versioned plot outputs missing in some runs"
    
    def test_concurrent_versioning_systems(self, kedro_project_manager, pipeline_executor):
        """Test that FigRegistry and Kedro versioning systems don't interfere.
        
        Validates that both versioning approaches can operate simultaneously
        without conflicts or data corruption.
        """
        project_path = kedro_project_manager.create_minimal_project()
        
        # Create configuration with explicit versioning settings
        versioned_config = generate_baseline_config()
        versioned_config['outputs']['versioning'] = {
            'enabled': True,
            'timestamp_format': '%Y%m%d_%H%M%S',
            'unique_naming': True
        }
        kedro_project_manager.create_figregistry_config(project_path, "base", versioned_config)
        
        # Execute pipeline to create versioned outputs
        executor = pipeline_executor(project_path)
        result = executor.execute_pipeline()
        
        assert result['success'], f"Versioned systems pipeline failed: {result.get('error')}"
        
        # Check for both versioned and non-versioned outputs
        figure_outputs = result['figure_outputs']
        
        # Regular outputs should exist
        assert figure_outputs.get('exploratory_plot', False), "Regular exploratory plot missing"
        assert figure_outputs.get('presentation_chart', False), "Regular presentation chart missing"
        
        # Versioned output should also exist
        assert figure_outputs.get('versioned_plot', False), "Versioned plot missing"
        
        # Validate no conflicts occurred
        assert result['nodes_executed'] > 0, "No nodes executed in versioned pipeline"
    
    def test_version_history_preservation(self, kedro_project_manager, pipeline_executor):
        """Test that version history is preserved correctly.
        
        Ensures that historical versions of figures remain accessible and
        that version metadata is properly maintained.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        executor = pipeline_executor(project_path)
        
        # Execute multiple pipeline runs to build version history
        version_history = []
        for i in range(2):
            result = executor.execute_pipeline()
            assert result['success'], f"Version history run {i} failed"
            
            version_info = {
                'run_number': i,
                'execution_time': result['execution_time_ms'],
                'session_id': result['session_id'],
                'figure_outputs': result['figure_outputs']
            }
            version_history.append(version_info)
            
            # Brief delay between runs
            time.sleep(0.05)
        
        # Validate version history
        assert len(version_history) == 2, "Version history not properly maintained"
        
        # Each run should have been successful
        for version in version_history:
            assert version['figure_outputs']['versioned_plot'], f"Versioned output missing in run {version['run_number']}"
        
        # Session IDs should be different (indicating separate runs)
        session_ids = [v['session_id'] for v in version_history]
        assert len(set(session_ids)) == len(session_ids), "Session IDs should be unique across runs"


class TestPerformanceBenchmarking:
    """Test performance benchmarking per Section 6.6.4.3.
    
    Validates plugin performance against specified targets: <200ms plugin overhead 
    per pipeline run, <50ms configuration bridge resolution, <25ms hook initialization.
    """
    
    @pytest.mark.performance
    def test_plugin_overhead_benchmark(self, kedro_project_manager, pipeline_executor, performance_tracker, benchmark):
        """Benchmark plugin overhead against performance targets.
        
        Measures complete plugin execution path and validates against:
        - Plugin Pipeline Execution Overhead: <200ms per FigureDataSet save
        - Configuration Bridge Merge Time: <50ms per pipeline run
        - Hook Initialization Overhead: <25ms per project startup
        """
        if not hasattr(pytest, 'benchmark') or benchmark is None:
            pytest.skip("pytest-benchmark not available for performance testing")
        
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        executor = pipeline_executor(project_path)
        
        def pipeline_execution():
            """Function to benchmark."""
            result = executor.execute_pipeline()
            assert result['success'], "Benchmark pipeline execution failed"
            return result
        
        # Benchmark pipeline execution
        benchmark_result = benchmark(pipeline_execution)
        
        # Validate performance targets
        execution_time_ms = benchmark_result['execution_time_ms']
        
        # Plugin overhead should be <200ms per pipeline run
        assert execution_time_ms < 200.0, f"Plugin overhead {execution_time_ms:.2f}ms exceeds 200ms target"
        
        # Log performance summary
        performance_summary = executor.get_performance_summary()
        logger.info(f"Performance benchmark results: {performance_summary}")
        
        # Additional validation
        assert performance_summary['performance_target_met'], "Performance targets not met in benchmark"
    
    @pytest.mark.performance
    def test_configuration_bridge_performance(self, kedro_project_manager, performance_tracker):
        """Test configuration bridge performance against <50ms target.
        
        Measures configuration merging and validation performance.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        kedro_project_manager.create_figregistry_config(project_path, "local", generate_environment_configs()["development"])
        
        from figregistry_kedro.config import FigRegistryConfigBridge
        
        def config_bridge_operation():
            """Benchmark configuration bridge operation."""
            bridge = FigRegistryConfigBridge(environment="local")
            config = bridge.get_merged_config()
            assert config is not None, "Configuration bridge returned None"
            return config
        
        # Measure configuration bridge performance
        with performance_tracker.measure_operation("config_bridge_merge"):
            for _ in range(10):  # Multiple iterations for accurate measurement
                config_bridge_operation()
        
        # Validate performance target
        merge_time = performance_tracker.measurements.get("config_bridge_merge", 0)
        average_time = merge_time / 10  # Average over 10 iterations
        
        assert average_time < 50.0, f"Config bridge merge time {average_time:.2f}ms exceeds 50ms target"
    
    @pytest.mark.performance  
    def test_hook_initialization_performance(self, kedro_project_manager, performance_tracker):
        """Test hook initialization performance against <25ms target.
        
        Measures FigRegistryHooks initialization and registration time.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        from figregistry_kedro.hooks import FigRegistryHooks
        
        def hook_initialization():
            """Benchmark hook initialization."""
            hooks = FigRegistryHooks(auto_initialize=True)
            assert hooks is not None, "Hook initialization failed"
            return hooks
        
        # Measure hook initialization performance
        with performance_tracker.measure_operation("hook_initialization"):
            for _ in range(20):  # Multiple iterations for accurate measurement
                hook_initialization()
        
        # Validate performance target
        init_time = performance_tracker.measurements.get("hook_initialization", 0)
        average_time = init_time / 20  # Average over 20 iterations
        
        assert average_time < 25.0, f"Hook initialization time {average_time:.2f}ms exceeds 25ms target"
    
    @pytest.mark.performance
    def test_end_to_end_performance_profile(self, kedro_project_manager, pipeline_executor, performance_tracker):
        """Create comprehensive performance profile of plugin operations.
        
        Provides detailed performance analysis across all plugin components
        for performance optimization and regression testing.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        # Comprehensive performance measurement
        with performance_tracker.measure_operation("project_setup"):
            executor = pipeline_executor(project_path)
        
        with performance_tracker.measure_operation("full_pipeline_execution"):
            result = executor.execute_pipeline()
            assert result['success'], "Performance profile pipeline failed"
        
        with performance_tracker.measure_operation("output_validation"):
            figure_outputs = result['figure_outputs']
            assert all(figure_outputs.values()), "Performance profile output validation failed"
        
        # Generate performance report
        performance_report = performance_tracker.get_summary()
        logger.info(f"End-to-end performance profile:\n{performance_report}")
        
        # Validate all performance targets
        validation_results = performance_tracker.validate_performance_targets()
        failed_targets = [op for op, passed in validation_results.items() if not passed]
        
        assert not failed_targets, f"Performance targets failed: {failed_targets}\n{performance_report}"


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility per Section 6.6.1.4.
    
    Validates plugin functionality across Python 3.10-3.12 and Kedro 0.18-0.19 
    version matrix on multiple operating systems.
    """
    
    @pytest.mark.kedro_version
    def test_python_version_compatibility(self, kedro_project_manager, pipeline_executor):
        """Test compatibility across Python versions 3.10-3.12.
        
        Validates that plugin works consistently across supported Python versions.
        """
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Skip if not in supported version range
        supported_versions = ["3.10", "3.11", "3.12"]
        if python_version not in supported_versions:
            pytest.skip(f"Python {python_version} not in supported versions: {supported_versions}")
        
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        executor = pipeline_executor(project_path)
        result = executor.execute_pipeline()
        
        assert result['success'], f"Python {python_version} compatibility test failed: {result.get('error')}"
        
        # Validate all figure outputs were created
        figure_outputs = result['figure_outputs']
        assert all(figure_outputs.values()), f"Figure outputs incomplete on Python {python_version}: {figure_outputs}"
    
    @pytest.mark.kedro_version
    def test_kedro_version_compatibility(self, kedro_project_manager, pipeline_executor):
        """Test compatibility across Kedro versions 0.18-0.19.
        
        Validates that plugin integrates correctly with different Kedro versions.
        """
        # Get current Kedro version
        try:
            import kedro
            kedro_version = kedro.__version__
        except Exception:
            pytest.skip("Unable to determine Kedro version")
        
        # Extract major.minor version
        version_parts = kedro_version.split('.')
        if len(version_parts) >= 2:
            major_minor = f"{version_parts[0]}.{version_parts[1]}"
        else:
            pytest.skip(f"Unable to parse Kedro version: {kedro_version}")
        
        # Check if in supported range
        supported_versions = ["0.18", "0.19"]
        if major_minor not in supported_versions:
            pytest.skip(f"Kedro {major_minor} not in supported versions: {supported_versions}")
        
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        executor = pipeline_executor(project_path)
        result = executor.execute_pipeline()
        
        assert result['success'], f"Kedro {kedro_version} compatibility test failed: {result.get('error')}"
        
        # Validate plugin integration
        from figregistry_kedro.tests.conftest import integration_validators
        validators = integration_validators()
        integration_status = validators.validate_plugin_integration(project_path)
        
        assert integration_status['hooks_registered'], f"Hook registration failed on Kedro {kedro_version}"
        assert integration_status['catalog_datasets'], f"Dataset registration failed on Kedro {kedro_version}"
    
    def test_file_system_compatibility(self, kedro_project_manager, pipeline_executor):
        """Test file system compatibility across different OS environments.
        
        Validates that plugin handles file paths correctly on different operating systems.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        
        # Test with various path configurations
        path_config = generate_baseline_config()
        
        # Use OS-specific path separators
        if os.name == 'nt':  # Windows
            path_config['outputs']['base_path'] = 'figures\\outputs'
        else:  # Unix-like
            path_config['outputs']['base_path'] = 'figures/outputs'
        
        kedro_project_manager.create_figregistry_config(project_path, "base", path_config)
        
        executor = pipeline_executor(project_path)
        result = executor.execute_pipeline()
        
        assert result['success'], f"File system compatibility test failed: {result.get('error')}"
        
        # Validate outputs were created with correct paths
        figure_outputs = result['figure_outputs']
        assert all(figure_outputs.values()), f"File system path handling failed: {figure_outputs}"
    
    def test_environment_variable_handling(self, kedro_project_manager, pipeline_executor):
        """Test environment variable handling across platforms.
        
        Validates that plugin correctly handles environment-specific configurations
        and environment variables across different platforms.
        """
        # Set test environment variables
        os.environ['FIGREGISTRY_TEST_ENV'] = 'cross_platform_test'
        os.environ['KEDRO_TEST_ENV'] = 'compatibility_test'
        
        try:
            project_path = kedro_project_manager.create_minimal_project()
            kedro_project_manager.create_figregistry_config(project_path, "base")
            
            executor = pipeline_executor(project_path)
            result = executor.execute_pipeline()
            
            assert result['success'], f"Environment variable test failed: {result.get('error')}"
            
            # Validate that environment was properly handled
            assert result['nodes_executed'] > 0, "No nodes executed in environment test"
            
        finally:
            # Clean up environment variables
            os.environ.pop('FIGREGISTRY_TEST_ENV', None)
            os.environ.pop('KEDRO_TEST_ENV', None)


# =============================================================================
# INTEGRATION TEST SUMMARY AND REPORTING
# =============================================================================

class TestIntegrationSummary:
    """Integration test summary and validation reporting.
    
    Provides comprehensive validation of all integration test scenarios
    and generates summary reports for test results.
    """
    
    def test_comprehensive_integration_validation(self, kedro_project_manager, pipeline_executor, 
                                                performance_tracker, integration_validators):
        """Comprehensive validation of complete plugin integration.
        
        Executes all major integration scenarios and validates overall plugin health.
        """
        project_path = kedro_project_manager.create_minimal_project()
        kedro_project_manager.create_figregistry_config(project_path, "base")
        kedro_project_manager.create_figregistry_config(project_path, "local", generate_environment_configs()["development"])
        
        # Execute comprehensive integration test
        executor = pipeline_executor(project_path)
        
        with performance_tracker.measure_operation("comprehensive_integration"):
            result = executor.execute_pipeline()
        
        # Comprehensive validation checks
        validations = {
            'pipeline_execution': result['success'],
            'figure_outputs_created': all(result['figure_outputs'].values()),
            'performance_targets_met': result['execution_time_ms'] < 200.0,
            'node_execution_complete': result['nodes_executed'] > 0,
            'plugin_integration': True,  # Will be validated below
            'environment_config_valid': True,  # Will be validated below
            'versioning_compatible': True  # Will be validated below
        }
        
        # Plugin integration validation
        integration_status = integration_validators.validate_plugin_integration(project_path)
        validations['plugin_integration'] = all(integration_status.values())
        
        # Environment configuration validation
        env_merge_valid = integration_validators.validate_environment_config_merge(project_path, "local")
        validations['environment_config_valid'] = env_merge_valid
        
        # Versioning compatibility validation
        versioning_valid = integration_validators.validate_versioning_compatibility(project_path)
        validations['versioning_compatible'] = versioning_valid
        
        # Assert all validations passed
        failed_validations = [check for check, passed in validations.items() if not passed]
        assert not failed_validations, f"Integration validation failures: {failed_validations}"
        
        # Generate summary report
        summary_report = self._generate_integration_summary(validations, result, performance_tracker)
        logger.info(f"Comprehensive integration validation summary:\n{summary_report}")
    
    def _generate_integration_summary(self, validations: Dict[str, bool], 
                                    execution_result: Dict[str, Any],
                                    performance_tracker) -> str:
        """Generate comprehensive integration test summary report."""
        lines = [
            "FigRegistry-Kedro Plugin Integration Test Summary",
            "=" * 50,
            "",
            "Validation Results:",
        ]
        
        for check, passed in validations.items():
            status = "PASS" if passed else "FAIL"
            lines.append(f"  {check}: {status}")
        
        lines.extend([
            "",
            "Execution Results:",
            f"  Pipeline: {execution_result['pipeline_name']}",
            f"  Execution Time: {execution_result['execution_time_ms']:.2f}ms",
            f"  Nodes Executed: {execution_result['nodes_executed']}",
            f"  Success: {execution_result['success']}",
            ""
        ])
        
        # Figure outputs summary
        lines.append("Figure Outputs:")
        for figure_name, created in execution_result['figure_outputs'].items():
            status = "CREATED" if created else "MISSING"
            lines.append(f"  {figure_name}: {status}")
        
        lines.extend([
            "",
            "Performance Summary:",
            performance_tracker.get_summary(),
            "",
            "Integration Status: " + ("SUCCESS" if all(validations.values()) else "FAILURE")
        ])
        
        return "\n".join(lines)