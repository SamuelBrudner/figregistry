# Installation Guide

This guide provides comprehensive installation instructions for the figregistry-kedro plugin, covering multiple installation methods, environment setup, and integration with existing Kedro projects.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Development Environment Setup](#development-environment-setup)
- [Kedro Project Integration](#kedro-project-integration)
- [Verification and Testing](#verification-and-testing)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Quick Start

For users who want to get started immediately:

```bash
# Install the plugin
pip install figregistry-kedro

# Verify installation
python -c "import figregistry_kedro; print('Installation successful!')"

# Add to existing Kedro project
cd your-kedro-project
kedro registry list  # Should show FigureDataSet available
```

## System Requirements

### Python Version Compatibility

The figregistry-kedro plugin requires Python 3.10 or later for optimal compatibility:

| Python Version | Support Status | Notes |
|---------------|----------------|-------|
| **3.10** | ✅ Recommended | Full feature support with optimal performance |
| **3.11** | ✅ Supported | Complete compatibility with enhanced type safety |
| **3.12** | ✅ Supported | Latest features and performance improvements |
| **3.13** | ✅ Supported | Forward compatibility for future-proofing |
| 3.9 | ❌ Not Supported | Limited type annotation support |
| 3.8 | ❌ Not Supported | Missing required language features |

### Framework Compatibility Matrix

The plugin is designed to work seamlessly across different versions of its core dependencies:

| Framework | Required Version | Tested Versions | Notes |
|-----------|------------------|-----------------|-------|
| **FigRegistry** | >=0.3.0 | 0.3.0, 0.3.1+ | Core visualization management system |
| **Kedro** | >=0.18.0,<0.20.0 | 0.18.0-0.19.x | Pipeline orchestration framework |
| **Matplotlib** | >=3.9.0 | 3.9.0+ | Visualization backend |
| **Pydantic** | >=2.9.0 | 2.9.0+ | Configuration validation |
| **PyYAML** | >=6.0.1 | 6.0.1+ | Configuration parsing |

### System Dependencies

Ensure your system meets these basic requirements:

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+, CentOS 8+)
- **Memory**: Minimum 4GB RAM (8GB recommended for development)
- **Storage**: 2GB available space for dependencies and build artifacts
- **Network**: Internet connection for package downloads

## Installation Methods

### Method 1: pip Installation (Recommended)

The pip installation method provides the most straightforward setup for most users.

#### Standard Installation

```bash
# Install from PyPI
pip install figregistry-kedro

# Upgrade to latest version
pip install --upgrade figregistry-kedro

# Install specific version
pip install figregistry-kedro==0.1.0
```

#### Installation with Optional Dependencies

```bash
# Install with development tools
pip install "figregistry-kedro[dev]"

# Install with testing framework only
pip install "figregistry-kedro[test]"

# Install with documentation tools
pip install "figregistry-kedro[docs]"

# Install all optional dependencies
pip install "figregistry-kedro[dev,test,docs]"
```

#### Virtual Environment Installation

For isolated environments (recommended for development):

```bash
# Create virtual environment
python -m venv figregistry-kedro-env

# Activate virtual environment
# On Windows:
figregistry-kedro-env\Scripts\activate
# On macOS/Linux:
source figregistry-kedro-env/bin/activate

# Install plugin
pip install figregistry-kedro

# Verify installation
python -c "import figregistry_kedro; print(figregistry_kedro.__version__)"
```

### Method 2: conda Installation

The conda installation method is ideal for scientific computing environments and provides robust dependency management.

#### conda-forge Installation

```bash
# Install from conda-forge channel
conda install -c conda-forge figregistry-kedro

# Create environment with figregistry-kedro
conda create -n figregistry-kedro -c conda-forge python=3.11 figregistry-kedro

# Activate environment
conda activate figregistry-kedro
```

#### Environment File Installation

Create a comprehensive environment configuration:

```yaml
# environment.yml
name: figregistry-kedro
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - figregistry-kedro
  - jupyter  # Optional: for notebook development
  - pytest  # Optional: for testing
```

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate figregistry-kedro
```

### Method 3: Development Installation

For contributors and advanced users who need the latest features or want to modify the plugin.

#### Clone and Install from Source

```bash
# Clone repository
git clone https://github.com/figregistry/figregistry-kedro.git
cd figregistry-kedro

# Create development environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify development installation
pytest tests/ --cov=figregistry_kedro
```

#### Development Environment with conda

```bash
# Clone repository
git clone https://github.com/figregistry/figregistry-kedro.git
cd figregistry-kedro

# Create conda development environment
conda env create -f environment-dev.yml
conda activate figregistry-kedro-dev

# Install in development mode
pip install -e ".[dev]"

# Setup development tools
pre-commit install
```

## Development Environment Setup

### Multi-Environment Development Strategy

The figregistry-kedro plugin supports a tri-environment approach for comprehensive development workflows:

#### Production Environment

Minimal runtime dependencies for deployment:

```yaml
# environment-prod.yml
name: figregistry-prod
channels:
  - conda-forge
dependencies:
  - python=3.11
  - figregistry>=0.3.0
  - kedro>=0.18.0,<0.20.0
  - matplotlib>=3.9.0
  - pydantic>=2.9.0
  - pyyaml>=6.0.1
  - numpy>=1.24.0
  - pandas>=2.0.0
```

#### Development Environment

Complete development toolchain with testing and quality tools:

```yaml
# environment-dev.yml
name: figregistry-dev
channels:
  - conda-forge
dependencies:
  - python=3.11
  - figregistry>=0.3.0
  - kedro>=0.18.0,<0.20.0
  - kedro-datasets>=1.0.0
  - matplotlib>=3.9.0
  - pydantic>=2.9.0
  - pyyaml>=6.0.1
  - numpy>=1.24.0
  - pandas>=2.0.0
  # Development tools
  - pytest>=7.0.0
  - pytest-cov>=4.0.0
  - pytest-mock>=3.14.0
  - hypothesis>=6.0.0
  - black>=23.0.0
  - isort>=5.12.0
  - mypy>=1.0.0
  - pre-commit>=3.0.0
  # Documentation
  - mkdocs>=1.4.0
  - mkdocs-material>=9.0.0
  - mkdocstrings>=0.20.0
  # Utilities
  - jupyter>=1.0.0
  - ipykernel>=6.0.0
```

#### Plugin Testing Environment

Specialized environment for plugin development and testing:

```yaml
# environment-plugin.yml
name: figregistry-kedro-plugin
channels:
  - conda-forge
dependencies:
  - python=3.11
  - figregistry>=0.3.0
  - kedro>=0.18.0,<0.20.0
  - kedro-datasets>=1.0.0
  # Plugin-specific testing
  - pytest>=7.0.0
  - pytest-mock>=3.14.0
  - hypothesis>=6.0.0
  # Code quality
  - black>=23.0.0
  - isort>=5.12.0
  - mypy>=1.0.0
  # Build tools
  - build>=0.7.0
  - setuptools-scm>=6.2
```

### Automated Environment Setup

Use the provided setup scripts for streamlined environment creation:

```bash
# Linux/macOS setup script
curl -sSL https://raw.githubusercontent.com/figregistry/figregistry-kedro/main/scripts/setup_env.sh | bash

# Windows PowerShell setup
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/figregistry/figregistry-kedro/main/scripts/setup_env.ps1" | Invoke-Expression
```

Or manually create environments:

```bash
# Development environment setup
conda env create -f environment-dev.yml
conda activate figregistry-dev

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import figregistry_kedro; print('Development environment ready!')"
pytest --version
kedro --version
```

## Kedro Project Integration

### New Kedro Project Setup

Create a new Kedro project with figregistry-kedro integration:

```bash
# Create new Kedro project
kedro new --starter=pandas-iris

# Navigate to project directory
cd new-kedro-project

# Install figregistry-kedro
pip install figregistry-kedro

# Initialize FigRegistry configuration
kedro registry describe  # Should show FigureDataSet available
```

### Existing Kedro Project Integration

Integrate the plugin into an existing Kedro project:

#### Step 1: Install the Plugin

```bash
# Navigate to your existing Kedro project
cd your-existing-kedro-project

# Install the plugin
pip install figregistry-kedro

# Verify Kedro recognizes the plugin
kedro registry list | grep -i figure
```

#### Step 2: Register Hooks

Add FigRegistry hooks to your project settings:

```python
# src/your_project/settings.py
"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

from figregistry_kedro.hooks import FigRegistryHooks

# Register FigRegistry hooks
HOOKS = (FigRegistryHooks(),)

# If you have existing hooks, add FigRegistryHooks to the tuple:
# from your_project.hooks import ProjectHooks
# HOOKS = (ProjectHooks(), FigRegistryHooks())
```

#### Step 3: Create FigRegistry Configuration

Add a FigRegistry configuration file to your project:

```yaml
# conf/base/figregistry.yml
figregistry_version: ">=0.3.0"

# Basic condition-based styling
condition_styles:
  training:
    color: "#2E86AB"
    marker: "o"
    linestyle: "-"
  
  validation:
    color: "#A23B72"
    marker: "s"
    linestyle: "--"
  
  test:
    color: "#F18F01"
    marker: "^"
    linestyle: ":"

# Output configuration
paths:
  base_dir: "data/08_reporting"

# Kedro integration settings
kedro:
  dataset_defaults:
    purpose: "expl"
    dpi: 300
    format: "png"
  
  hooks:
    auto_init: true
    log_operations: true
```

#### Step 4: Update Data Catalog

Modify your data catalog to use FigureDataSet:

```yaml
# conf/base/catalog.yml

# Example: Replace matplotlib figure outputs with FigureDataSet
model_accuracy_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/model_accuracy.png
  purpose: expl
  condition_param: "model_type"

training_history_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/training_history.pdf
  purpose: pub
  condition_param: "experiment_id"
  style_params:
    dpi: 300
    bbox_inches: "tight"
  versioned: true
```

#### Step 5: Update Pipeline Nodes

Modify your pipeline nodes to return matplotlib figures instead of saving them manually:

```python
# Before: Manual figure saving
def create_accuracy_plot(model_results):
    fig, ax = plt.subplots()
    ax.plot(model_results['epochs'], model_results['accuracy'])
    ax.set_title('Model Accuracy')
    plt.savefig('data/08_reporting/accuracy.png')  # Remove this line
    plt.close()
    return None

# After: Return figure for FigureDataSet
def create_accuracy_plot(model_results):
    fig, ax = plt.subplots()
    ax.plot(model_results['epochs'], model_results['accuracy'])
    ax.set_title('Model Accuracy')
    return fig  # Return figure instead of saving manually
```

### Migration from Manual Figure Management

For projects currently using manual plt.savefig() calls:

#### Migration Checklist

- [ ] Install figregistry-kedro plugin
- [ ] Register FigRegistryHooks in settings.py
- [ ] Create conf/base/figregistry.yml configuration
- [ ] Update catalog entries to use FigureDataSet
- [ ] Modify pipeline nodes to return figures instead of saving them
- [ ] Remove manual plt.savefig() calls
- [ ] Test pipeline execution with new configuration

#### Automated Migration Script

Use the provided migration script for large projects:

```bash
# Download migration script
curl -O https://raw.githubusercontent.com/figregistry/figregistry-kedro/main/scripts/migrate_project.py

# Run migration (creates backup first)
python migrate_project.py --project-dir . --backup

# Review changes
git diff

# Test migrated project
kedro run
```

## Verification and Testing

### Installation Verification

Verify your installation with these commands:

```bash
# Check plugin installation
python -c "import figregistry_kedro; print(f'figregistry-kedro {figregistry_kedro.__version__} installed successfully')"

# Verify Kedro recognizes the plugin
kedro registry list | grep -E "(FigureDataSet|figregistry)"

# Check dependencies
python -c "import figregistry, kedro, matplotlib, pydantic; print('All dependencies available')"

# Validate hook registration (if already configured)
python -c "from your_project.settings import HOOKS; print(f'Hooks registered: {len(HOOKS)}')"
```

### Functionality Testing

Test the plugin functionality with a simple example:

```python
# test_figregistry_kedro.py
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from figregistry_kedro.datasets import FigureDataSet

def test_basic_functionality():
    """Test basic FigureDataSet functionality."""
    
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title('Test Figure')
    
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Test FigureDataSet save operation
        dataset = FigureDataSet(filepath=tmp_path, purpose='expl')
        dataset.save(fig)
        
        # Verify file was created
        assert Path(tmp_path).exists()
        print("✅ FigureDataSet save test passed")
        
        # Test load operation (returns path for verification)
        loaded_path = dataset.load()
        assert Path(loaded_path).exists()
        print("✅ FigureDataSet load test passed")
        
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        plt.close(fig)

if __name__ == "__main__":
    test_basic_functionality()
    print("🎉 All tests passed! Installation verified.")
```

Run the test:

```bash
python test_figregistry_kedro.py
```

### Integration Testing

Test the plugin within a Kedro project context:

```bash
# Create test project
kedro new --starter=pandas-iris --directory=test-figregistry-kedro

# Navigate to test project
cd test-figregistry-kedro

# Install plugin
pip install figregistry-kedro

# Add minimal configuration
cat > conf/base/figregistry.yml << EOF
figregistry_version: ">=0.3.0"
condition_styles:
  setosa:
    color: "#FF6B6B"
  versicolor:
    color: "#4ECDC4"
  virginica:
    color: "#45B7D1"
paths:
  base_dir: "data/08_reporting"
kedro:
  dataset_defaults:
    purpose: "expl"
EOF

# Register hooks
cat > src/test_figregistry_kedro/settings.py << EOF
from figregistry_kedro.hooks import FigRegistryHooks
HOOKS = (FigRegistryHooks(),)
EOF

# Add test figure output to catalog
cat >> conf/base/catalog.yml << EOF

iris_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/iris_analysis.png
  purpose: expl
  condition_param: "species"
EOF

# Test pipeline execution
kedro run

# Verify output
ls -la data/08_reporting/
```

## Troubleshooting

### Common Installation Issues

#### Issue 1: ImportError with FigRegistry

**Problem**: `ImportError: No module named 'figregistry'`

**Solution**:
```bash
# Ensure figregistry is installed (should be automatic)
pip install figregistry>=0.3.0

# If using conda, install from conda-forge
conda install -c conda-forge figregistry

# Verify installation
python -c "import figregistry; print(figregistry.__version__)"
```

#### Issue 2: Kedro Version Compatibility

**Problem**: `kedro.framework.cli.utils.KedroCliError: kedro version incompatible`

**Solution**:
```bash
# Check current Kedro version
kedro --version

# Upgrade to compatible version
pip install "kedro>=0.18.0,<0.20.0"

# If using conda
conda install -c conda-forge "kedro>=0.18.0,<0.20.0"

# Verify compatibility
python -c "import kedro; print(kedro.__version__)"
```

#### Issue 3: Pydantic Validation Errors

**Problem**: `pydantic.ValidationError` during configuration loading

**Solution**:
```bash
# Update to compatible Pydantic version
pip install "pydantic>=2.9.0"

# Check configuration syntax
python -c "
import yaml
with open('conf/base/figregistry.yml') as f:
    config = yaml.safe_load(f)
    print('Configuration syntax valid')
"

# Validate with FigRegistry
python -c "
import figregistry
figregistry.init_config('conf/base/figregistry.yml')
print('FigRegistry configuration valid')
"
```

#### Issue 4: matplotlib Backend Issues

**Problem**: Figures not displaying or saving correctly

**Solution**:
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Set non-interactive backend for headless environments
export MPLBACKEND=Agg

# Or set in Python code
python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print('Backend set successfully')
"
```

### Environment-Specific Issues

#### Docker/Container Environments

```dockerfile
# Dockerfile additions for figregistry-kedro
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set matplotlib backend for headless environment
ENV MPLBACKEND=Agg

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install figregistry-kedro
RUN pip install figregistry-kedro

# Verify installation
RUN python -c "import figregistry_kedro; print('Installation verified')"
```

#### Windows-Specific Issues

```powershell
# Windows PowerShell troubleshooting

# Check Python version
python --version

# Install with specific versions if needed
pip install "figregistry-kedro" --force-reinstall

# Windows path issues - use forward slashes in config
# In figregistry.yml:
# paths:
#   base_dir: "data/08_reporting"  # Use forward slashes

# Test installation
python -c "import figregistry_kedro; print('Windows installation successful')"
```

#### macOS-Specific Issues

```bash
# macOS troubleshooting

# Ensure Xcode command line tools are installed
xcode-select --install

# Install via Homebrew Python if system Python issues
brew install python@3.11
/opt/homebrew/bin/python3.11 -m pip install figregistry-kedro

# Fix potential SSL certificate issues
/Applications/Python\ 3.11/Install\ Certificates.command
```

### Performance Optimization

#### Large Project Performance

For projects with many figures or large datasets:

```python
# src/your_project/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

# Optimize hook configuration
HOOKS = (
    FigRegistryHooks(
        cache_config=True,        # Enable configuration caching
        lazy_loading=True,        # Load configurations on-demand
        batch_operations=True     # Batch figure operations
    ),
)
```

#### Memory Management

For memory-constrained environments:

```yaml
# conf/base/figregistry.yml
kedro:
  dataset_defaults:
    dpi: 150  # Lower DPI for reduced memory usage
    format: "png"  # Use PNG instead of PDF for smaller files
  
  performance:
    figure_cache_size: 10  # Limit figure cache
    auto_cleanup: true     # Automatic memory cleanup
```

### Getting Additional Help

If you encounter issues not covered in this guide:

1. **Check the logs**: Enable debug logging to see detailed operations
   ```bash
   kedro run --log-level=DEBUG
   ```

2. **Validate your environment**: Use the verification commands in the [Verification and Testing](#verification-and-testing) section

3. **Review configuration**: Ensure your `figregistry.yml` follows the schema in the [Configuration Guide](configuration.md)

4. **Check compatibility**: Verify all dependencies meet the version requirements in the [System Requirements](#system-requirements) section

5. **Search existing issues**: Check the [GitHub Issues](https://github.com/figregistry/figregistry-kedro/issues) for similar problems

6. **Submit a bug report**: If you find a new issue, please report it with:
   - Your operating system and Python version
   - Complete error messages and stack traces
   - Minimal reproduction example
   - Your configuration files (with sensitive data removed)

## Advanced Configuration

### Custom Installation Locations

For enterprise environments with custom package locations:

```bash
# Install from custom PyPI server
pip install --index-url https://your-pypi-server.com/simple/ figregistry-kedro

# Install from local wheel
pip install ./dist/figregistry_kedro-*.whl

# Install from Git repository
pip install git+https://github.com/figregistry/figregistry-kedro.git@main
```

### Offline Installation

For air-gapped environments:

```bash
# Download packages
pip download figregistry-kedro --dest ./packages/

# Install offline
pip install --find-links ./packages/ --no-index figregistry-kedro
```

### Integration with Package Managers

#### Poetry Integration

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
figregistry-kedro = "^0.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^23.0.0"
```

#### pipenv Integration

```toml
# Pipfile
[packages]
figregistry-kedro = "*"

[dev-packages]
pytest = "*"
black = "*"

[requires]
python_version = "3.11"
```

This installation guide provides comprehensive coverage of all installation scenarios and troubleshooting steps needed to successfully deploy figregistry-kedro in any environment.