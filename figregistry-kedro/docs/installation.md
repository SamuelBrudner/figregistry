# Installation Guide

This comprehensive guide covers installation and setup of the figregistry-kedro plugin, providing automated figure styling and management within Kedro machine learning pipelines. The plugin bridges FigRegistry's condition-based visualization system with Kedro's data catalog architecture, enabling seamless figure management in data science workflows.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
- [Environment Setup](#environment-setup)
- [Kedro Project Integration](#kedro-project-integration)
- [Development Installation](#development-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Upgrade Instructions](#upgrade-instructions)

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum (8GB recommended for complex pipelines)
- **Storage**: 2GB available disk space for dependencies

### Framework Compatibility

The figregistry-kedro plugin maintains strict compatibility requirements to ensure reliable integration:

| Component | Version Range | Purpose |
|-----------|---------------|---------|
| **Python** | ≥3.10 | Advanced type hinting and performance optimizations |
| **FigRegistry** | ≥0.3.0 | Core visualization configuration and styling system |
| **Kedro** | ≥0.18.0,<0.20.0 | Data pipeline framework with stable AbstractDataSet interface |
| **Matplotlib** | ≥3.9.0 | Visualization backend with rcParams integration |
| **Pydantic** | ≥2.9.0 | Configuration validation and type safety |

### Platform-Specific Notes

#### Windows
- Windows 10 or Windows 11 required
- PowerShell 5.1+ or PowerShell Core recommended
- Visual Studio Build Tools may be required for some scientific dependencies

#### macOS
- macOS 10.15 (Catalina) or later
- Xcode Command Line Tools recommended: `xcode-select --install`
- Homebrew package manager recommended for system dependencies

#### Linux
- Ubuntu 18.04+, CentOS 8+, or equivalent distributions
- GCC compiler and development headers: `sudo apt-get install build-essential`
- Python development headers: `sudo apt-get install python3-dev`

## Quick Start

For users who want to get started immediately with default settings:

```bash
# Install the plugin
pip install figregistry-kedro

# Create a new Kedro project with FigRegistry integration
kedro new --starter=pandas-iris my_kedro_project
cd my_kedro_project

# Install the plugin in your project
pip install figregistry-kedro

# Configure hooks in settings.py
echo "from figregistry_kedro.hooks import FigRegistryHooks
HOOKS = (FigRegistryHooks(),)" >> src/my_kedro_project/settings.py

# Run the pipeline with automated figure management
kedro run
```

This quick start creates a functional Kedro project with FigRegistry integration in under 5 minutes.

## Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest installation method uses pip to install from the Python Package Index:

```bash
# Install the latest stable release
pip install figregistry-kedro

# Install a specific version
pip install figregistry-kedro==0.1.0

# Install with specific dependency versions
pip install "figregistry-kedro[test]" "kedro>=0.18.14"

# Upgrade to latest version
pip install --upgrade figregistry-kedro
```

**Benefits:**
- Latest stable releases
- Automatic dependency resolution
- Fastest installation method
- Compatible with virtual environments

**When to use:**
- Production deployments
- Standard data science workflows
- First-time users

### Method 2: Conda Installation

Install from conda-forge for comprehensive dependency management:

```bash
# Install from conda-forge (recommended for scientific computing)
conda install -c conda-forge figregistry-kedro

# Create a new environment with the plugin
conda create -n kedro-figregistry -c conda-forge python=3.11 figregistry-kedro
conda activate kedro-figregistry

# Install in existing environment
conda install -c conda-forge figregistry-kedro

# Install with specific versions
conda install -c conda-forge "figregistry-kedro>=0.1.0" "kedro>=0.18.0,<0.20.0"
```

**Benefits:**
- Optimized scientific computing stack
- Cross-platform binary packages
- Robust dependency conflict resolution
- Integration with Anaconda ecosystem

**When to use:**
- Scientific computing environments
- Complex dependency requirements
- Cross-platform deployment
- Anaconda/Miniconda users

### Method 3: Development Installation

Install from source for development, testing, or accessing unreleased features:

```bash
# Clone the repository
git clone https://github.com/figregistry/figregistry-kedro.git
cd figregistry-kedro

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[dev,test,docs]"
```

**Benefits:**
- Access to latest development features
- Ability to modify source code
- Full development toolchain
- Contribution-ready setup

**When to use:**
- Plugin development
- Contributing to the project
- Testing unreleased features
- Custom modifications

## Environment Setup

### Option 1: Virtual Environment (Python venv)

Create an isolated Python environment for your Kedro projects:

```bash
# Create a new virtual environment
python -m venv kedro-figregistry-env

# Activate the environment
# Windows:
kedro-figregistry-env\Scripts\activate
# macOS/Linux:
source kedro-figregistry-env/bin/activate

# Install the plugin
pip install figregistry-kedro

# Install Kedro and common data science packages
pip install kedro pandas numpy matplotlib seaborn scikit-learn

# Deactivate when done
deactivate
```

### Option 2: Conda Environment

Leverage Conda's comprehensive package management:

```bash
# Create environment from scratch
conda create -n kedro-figregistry python=3.11
conda activate kedro-figregistry
conda install -c conda-forge figregistry-kedro kedro pandas numpy matplotlib

# Create from environment file (if provided)
conda env create -f environment.yml
conda activate kedro-figregistry

# Export your environment for sharing
conda env export > environment.yml
```

### Option 3: Poetry Environment

For modern Python dependency management:

```bash
# Initialize a new project with Poetry
poetry init
poetry add figregistry-kedro kedro

# Install dependencies
poetry install

# Activate the environment
poetry shell

# Run commands within the environment
poetry run kedro run
```

### Option 4: Docker Environment

Containerized setup for reproducible deployments:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install figregistry-kedro kedro pandas matplotlib

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Default command
CMD ["kedro", "run"]
```

```bash
# Build and run the container
docker build -t kedro-figregistry .
docker run -v $(pwd):/app kedro-figregistry
```

## Kedro Project Integration

### Step 1: Hook Registration

Configure FigRegistry hooks in your Kedro project's settings:

```python
# src/your_project/settings.py

from figregistry_kedro.hooks import FigRegistryHooks

# Basic hook registration
HOOKS = (FigRegistryHooks(),)

# Advanced configuration
HOOKS = (
    FigRegistryHooks(
        auto_initialize=True,
        enable_performance_monitoring=False,
        fallback_on_error=True,
        max_initialization_time=0.005  # 5ms maximum
    ),
)

# Integration with existing hooks
from kedro_datasets.spark import SparkDataSet

HOOKS = (
    FigRegistryHooks(),
    # ... other project hooks
)
```

### Step 2: Configuration Setup

Create FigRegistry configuration files in your Kedro project:

```bash
# Create base configuration directory
mkdir -p conf/base

# Create environment-specific directories
mkdir -p conf/local conf/staging conf/production
```

**Base Configuration** (`conf/base/figregistry.yml`):

```yaml
# Base FigRegistry configuration
styles:
  exploratory:
    figure.figsize: [10, 6]
    figure.dpi: 100
    axes.grid: true
    axes.spines.top: false
    axes.spines.right: false
  
  presentation:
    figure.figsize: [12, 8]
    figure.dpi: 200
    font.size: 14
    axes.labelsize: 16
  
  publication:
    figure.figsize: [8, 6]
    figure.dpi: 300
    font.family: serif
    axes.labelsize: 12

palettes:
  default:
    colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

outputs:
  base_path: data/08_reporting
  create_subdirs: true
  timestamp_format: "%Y%m%d_%H%M%S"

defaults:
  purpose: exploratory
  format: png
  bbox_inches: tight

# Kedro-specific settings
kedro:
  catalog_integration:
    auto_register_datasets: true
    versioning_enabled: true
  hook_settings:
    auto_initialize: true
    enable_performance_monitoring: false
    fallback_on_error: true
```

### Step 3: Data Catalog Configuration

Add FigureDataSet entries to your Kedro data catalog:

```yaml
# conf/base/catalog.yml

# Basic figure dataset
analysis_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/analysis_{default_run_id}.png
  purpose: exploratory

# Advanced configuration with styling
presentation_charts:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/presentation/{default_run_id}_results.pdf
  purpose: presentation
  style_params:
    figure.dpi: 300
    font.family: sans-serif
  save_args:
    bbox_inches: tight
    transparent: false

# Versioned datasets
publication_figures:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/publication/figure_{version}.pdf
  purpose: publication
  versioned: true
  condition_param: experiment_type
  save_args:
    dpi: 600
    format: pdf
```

### Step 4: Pipeline Integration

Update your pipeline nodes to use FigRegistry-managed figures:

```python
# src/your_project/pipelines/visualization/nodes.py

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

def create_analysis_plot(data: pd.DataFrame) -> Figure:
    """Create exploratory analysis plot.
    
    Returns matplotlib Figure that will be automatically styled
    and saved through FigRegistry integration.
    """
    fig, ax = plt.subplots()
    
    # Create your plot
    ax.scatter(data['x'], data['y'])
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_title('Data Analysis')
    
    # No manual styling or saving required
    # FigRegistry will apply appropriate styling based on 'purpose'
    return fig

def create_presentation_chart(processed_data: pd.DataFrame) -> Figure:
    """Create presentation-ready chart.
    
    Styling will be automatically applied based on 'presentation' purpose.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot
    ax1.plot(processed_data['time'], processed_data['metric1'], 'b-', linewidth=2)
    ax1.set_title('Metric 1 Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Metric 1')
    
    # Right subplot
    ax2.bar(processed_data['category'], processed_data['count'])
    ax2.set_title('Count by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    return fig
```

```python
# src/your_project/pipelines/visualization/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_analysis_plot, create_presentation_chart

def create_visualization_pipeline(**kwargs) -> Pipeline:
    """Create visualization pipeline with FigRegistry integration."""
    return pipeline([
        node(
            func=create_analysis_plot,
            inputs="processed_data",
            outputs="analysis_plot",  # Matches catalog entry
            name="create_analysis_plot"
        ),
        node(
            func=create_presentation_chart,
            inputs="processed_data",
            outputs="presentation_charts",  # Matches catalog entry
            name="create_presentation_chart"
        ),
    ])
```

## Development Installation

### Setting Up Development Environment

For plugin development or contributing to the project:

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/your-username/figregistry-kedro.git
cd figregistry-kedro

# 3. Create development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 4. Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# 5. Install pre-commit hooks
pre-commit install

# 6. Verify installation
python -m pytest tests/ -v
python -c "import figregistry_kedro; print('Installation successful!')"
```

### Development Dependencies

The development installation includes comprehensive tooling:

```python
# Development tools (automatically installed with [dev])
black>=23.0.0           # Code formatting
isort>=5.12.0           # Import sorting
mypy>=1.0.0             # Type checking
ruff>=0.1.0             # Fast linting
pre-commit>=3.0.0       # Git hooks

# Testing framework (automatically installed with [test])
pytest>=7.0.0           # Test runner
pytest-cov>=4.0.0       # Coverage measurement
pytest-mock>=3.14.0     # Mocking utilities
pytest-xdist>=3.0.0     # Parallel testing
hypothesis>=6.0.0       # Property-based testing

# Documentation tools (automatically installed with [docs])
mkdocs>=1.4.0           # Documentation generator
mkdocs-material>=9.0.0  # Material theme
mkdocstrings>=0.20.0    # API documentation
```

### Running Tests

Comprehensive test execution for development validation:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=figregistry_kedro --cov-report=html

# Run specific test modules
pytest tests/test_datasets.py -v
pytest tests/test_hooks.py -v
pytest tests/test_config.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Run tests in parallel
pytest -n auto

# Run performance tests
pytest tests/test_performance.py --benchmark-only
```

### Code Quality Checks

Ensure code quality before committing:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/figregistry_kedro/

# Linting
ruff check src/ tests/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Verification

### Installation Verification

Verify successful installation with these quick checks:

```python
# Basic import test
python -c "import figregistry_kedro; print(f'Version: {figregistry_kedro.__version__}')"

# Component availability test
python -c """
from figregistry_kedro.datasets import FigureDataSet
from figregistry_kedro.hooks import FigRegistryHooks
from figregistry_kedro.config import FigRegistryConfigBridge
print('All components imported successfully!')
"""

# Kedro integration test
python -c """
import kedro
print(f'Kedro version: {kedro.__version__}')
from kedro.framework.hooks import _NullPluginManager
from figregistry_kedro.hooks import FigRegistryHooks
hooks = FigRegistryHooks()
print('Kedro integration verified!')
"""
```

### Functional Testing

Test plugin functionality with a minimal example:

```python
# test_plugin_functionality.py
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

def test_basic_functionality():
    """Test basic FigRegistry-Kedro integration."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title('Test Figure')
        
        # Test FigureDataSet
        from figregistry_kedro.datasets import FigureDataSet
        
        dataset = FigureDataSet(
            filepath=str(tmpdir / "test_figure.png"),
            purpose="exploratory"
        )
        
        # Save figure
        dataset.save(fig)
        
        # Verify file was created
        assert (tmpdir / "test_figure.png").exists()
        print("✓ Basic functionality test passed!")

if __name__ == "__main__":
    test_basic_functionality()
```

```bash
# Run the test
python test_plugin_functionality.py
```

### Performance Verification

Validate performance meets specifications:

```python
# test_performance.py
import time
import matplotlib.pyplot as plt
from figregistry_kedro.datasets import FigureDataSet
from figregistry_kedro.config import FigRegistryConfigBridge

def test_performance():
    """Verify performance requirements."""
    
    # Test configuration initialization time
    start_time = time.time()
    bridge = FigRegistryConfigBridge()
    config = bridge.get_merged_config()
    init_time = time.time() - start_time
    
    assert init_time < 0.010, f"Config initialization took {init_time:.3f}s, exceeds 10ms limit"
    print(f"✓ Configuration initialization: {init_time*1000:.1f}ms")
    
    # Test figure styling time
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    
    start_time = time.time()
    # Style application would happen during save
    style_time = time.time() - start_time
    
    print(f"✓ Style application: {style_time*1000:.1f}ms")
    print("✓ Performance requirements verified!")

if __name__ == "__main__":
    test_performance()
```

## Troubleshooting

### Common Installation Issues

#### Issue: ImportError - No module named 'figregistry_kedro'

**Symptoms:**
```
ImportError: No module named 'figregistry_kedro'
```

**Solutions:**
1. Verify installation: `pip list | grep figregistry-kedro`
2. Check Python environment: `which python` and `which pip`
3. Reinstall package: `pip uninstall figregistry-kedro && pip install figregistry-kedro`
4. Virtual environment issues: Ensure you've activated the correct environment

#### Issue: Kedro compatibility errors

**Symptoms:**
```
TypeError: 'AbstractDataSet' object has no attribute 'load'
kedro.framework.hooks.manager.PluginManager.hook.X not found
```

**Solutions:**
1. Check Kedro version: `kedro --version`
2. Verify compatibility: Kedro >=0.18.0,<0.20.0 required
3. Update Kedro: `pip install "kedro>=0.18.14,<0.20.0"`
4. Clean installation: Remove and reinstall both packages

#### Issue: Matplotlib backend issues

**Symptoms:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
TclError: no display name and no $DISPLAY environment variable
```

**Solutions:**
1. Set matplotlib backend: `export MPLBACKEND=Agg`
2. Install GUI backend: `pip install tkinter` (Linux: `sudo apt-get install python3-tk`)
3. Use headless backend in code:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   ```

#### Issue: Permission errors during installation

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
1. Use user installation: `pip install --user figregistry-kedro`
2. Use virtual environment instead of system Python
3. On Linux/macOS with system Python: `sudo pip install figregistry-kedro` (not recommended)

### Configuration Issues

#### Issue: FigRegistry configuration not found

**Symptoms:**
```
WARNING:figregistry_kedro.config:Failed to load FigRegistry config from Kedro
```

**Solutions:**
1. Create configuration file: `conf/base/figregistry.yml`
2. Check file permissions and YAML syntax
3. Validate configuration with yamllint
4. Use standalone config: Create `figregistry.yaml` in project root

#### Issue: Hook registration failures

**Symptoms:**
```
ERROR:kedro.framework.session:Hook 'FigRegistryHooks' could not be loaded
```

**Solutions:**
1. Verify settings.py import:
   ```python
   from figregistry_kedro.hooks import FigRegistryHooks
   HOOKS = (FigRegistryHooks(),)
   ```
2. Check for syntax errors in settings.py
3. Ensure plugin is installed in the correct environment
4. Restart Kedro session/kernel

### Runtime Issues

#### Issue: Figures not being styled

**Symptoms:**
- Figures saved without FigRegistry styling
- Default matplotlib appearance retained

**Solutions:**
1. Verify catalog configuration uses `figregistry_kedro.FigureDataSet`
2. Check purpose parameter is set correctly
3. Validate FigRegistry configuration syntax
4. Enable debug logging:
   ```python
   import logging
   logging.getLogger('figregistry_kedro').setLevel(logging.DEBUG)
   ```

#### Issue: Performance degradation

**Symptoms:**
- Slow figure saving
- Increased pipeline execution time

**Solutions:**
1. Disable performance monitoring in production:
   ```python
   HOOKS = (FigRegistryHooks(enable_performance_monitoring=False),)
   ```
2. Check configuration caching is enabled
3. Reduce figure DPI for exploratory outputs
4. Profile with: `kedro run --profile`

### Environment-Specific Issues

#### Windows-Specific Issues

**Path separator problems:**
```python
# Use pathlib for cross-platform paths
from pathlib import Path
filepath = Path("data") / "08_reporting" / "figure.png"
```

**PowerShell execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS-Specific Issues

**SSL certificate errors:**
```bash
# Update certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Homebrew Python conflicts:**
```bash
# Use pyenv for Python version management
brew install pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

#### Linux-Specific Issues

**Missing system dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev python3-pip build-essential

# CentOS/RHEL
sudo yum install python3-devel python3-pip gcc gcc-c++
```

### Getting Help

If troubleshooting doesn't resolve your issue:

1. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check GitHub Issues:** [figregistry-kedro/issues](https://github.com/figregistry/figregistry-kedro/issues)

3. **Create detailed bug reports** including:
   - Operating system and version
   - Python version (`python --version`)
   - Package versions (`pip list`)
   - Complete error traceback
   - Minimal reproduction example

4. **Community support:**
   - Kedro Discord community
   - FigRegistry GitHub discussions
   - Stack Overflow with tags: `kedro`, `figregistry`

## Upgrade Instructions

### Upgrading the Plugin

#### Standard Upgrade

```bash
# Upgrade to latest version
pip install --upgrade figregistry-kedro

# Upgrade with dependency updates
pip install --upgrade figregistry-kedro figregistry kedro

# Check new version
python -c "import figregistry_kedro; print(figregistry_kedro.__version__)"
```

#### Upgrading with Conda

```bash
# Update from conda-forge
conda update -c conda-forge figregistry-kedro

# Update entire environment
conda update --all
```

### Version-Specific Upgrade Notes

#### Upgrading from 0.1.x to 0.2.x

**Breaking Changes:**
- Configuration schema updated for new styling options
- Hook interface changes for improved performance

**Migration Steps:**
1. Update configuration files to new schema
2. Review hook registration in settings.py
3. Test all figure outputs for styling consistency

#### Kedro Version Compatibility

When upgrading Kedro itself:

```bash
# Check current compatibility
pip show figregistry-kedro | grep "Requires:"

# Safe Kedro upgrade within supported range
pip install "kedro>=0.18.14,<0.20.0"

# Verify compatibility
kedro --version
python -c "from figregistry_kedro.hooks import FigRegistryHooks; print('Compatible!')"
```

### Pre-Upgrade Checklist

Before upgrading in production environments:

1. **Backup current environment:**
   ```bash
   pip freeze > requirements_backup.txt
   conda env export > environment_backup.yml
   ```

2. **Test in development:**
   ```bash
   # Create test environment
   python -m venv test_upgrade
   source test_upgrade/bin/activate
   pip install figregistry-kedro==<new_version>
   ```

3. **Validate configuration compatibility:**
   ```bash
   kedro run --dry-run  # Test pipeline without execution
   ```

4. **Review changelog:** Check release notes for breaking changes

5. **Update documentation:** Ensure team is aware of changes

### Rollback Procedure

If upgrade causes issues:

```bash
# Rollback to previous version
pip install figregistry-kedro==<previous_version>

# Restore from backup
pip install -r requirements_backup.txt

# Or restore conda environment
conda env remove -n current_env
conda env create -f environment_backup.yml
```

---

This installation guide provides comprehensive coverage of all installation methods, environment setup options, and troubleshooting guidance for the figregistry-kedro plugin. For additional support, consult the [Configuration Guide](configuration.md) for detailed setup options and the [API Reference](api/) for complete functionality documentation.