# FigRegistry-Kedro Migration Guide

## Overview

This comprehensive guide walks you through the process of migrating an existing Kedro project from manual matplotlib figure management to automated figregistry-kedro integration. The migration eliminates scattered `plt.savefig()` calls, centralizes styling configuration, and introduces automatic condition-based styling for consistent scientific visualization workflows.

### Migration Benefits

- **Eliminate Manual Figure Management**: Remove all `plt.savefig()` calls from pipeline nodes
- **Centralized Styling Configuration**: Move styling logic from code to configuration files
- **Automatic Condition-Based Styling**: Apply different styles based on experimental conditions
- **Improved Maintainability**: Reduce code duplication and improve separation of concerns
- **Enhanced Versioning**: Integrate with Kedro's versioning system for reproducible experiments

## Prerequisites

Before starting the migration, ensure you have:

- An existing Kedro project (version 0.18.0 or higher)
- Python 3.10 or higher
- Matplotlib-based visualization nodes in your pipeline
- Basic familiarity with Kedro's catalog and configuration system

## Migration Steps

### Step 1: Install figregistry-kedro

Update your project's `pyproject.toml` to include the figregistry-kedro dependency:

```toml
# Before: Traditional dependencies
[project]
dependencies = [
    "kedro>=0.18.0,<0.20.0",
    "matplotlib>=3.9.0",
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    # ... other dependencies
]

# After: Add figregistry-kedro integration
[project]
dependencies = [
    "kedro>=0.18.0,<0.20.0",
    "figregistry-kedro>=0.1.0",  # <-- Add this dependency
    "matplotlib>=3.9.0",
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    # ... other dependencies
]
```

Install the updated dependencies:

```bash
pip install -e .
# or if using conda
conda env update --file environment.yml
```

### Step 2: Register FigRegistry Hooks

Update your project's `src/<project_name>/settings.py` to register the FigRegistry lifecycle hooks:

```python
# Before: Standard Kedro settings
"""Project settings."""
from kedro.config import TemplatedConfigLoader

CONFIG_LOADER_CLASS = TemplatedConfigLoader

# After: Add FigRegistry hooks
"""Project settings."""
from kedro.config import TemplatedConfigLoader
from figregistry_kedro.hooks import FigRegistryHooks  # <-- Import hooks

CONFIG_LOADER_CLASS = TemplatedConfigLoader

# Register FigRegistry hooks for lifecycle management
HOOKS = (
    FigRegistryHooks(
        auto_initialize=True,
        enable_performance_monitoring=False,
        fallback_on_error=True
    ),
)
```

### Step 3: Create FigRegistry Configuration

Create a new configuration file `conf/base/figregistry.yml` with your styling definitions:

```yaml
# conf/base/figregistry.yml
# Condition-based styling configuration

styles:
  # Exploratory analysis styling
  exploratory:
    figure.figsize: [10, 6]
    figure.dpi: 100
    axes.grid: true
    axes.labelsize: 12
    axes.titlesize: 14
    legend.fontsize: 10
    
  # Presentation-ready styling
  presentation:
    figure.figsize: [12, 8]
    figure.dpi: 150
    axes.grid: false
    axes.labelsize: 14
    axes.titlesize: 16
    font.weight: bold
    legend.fontsize: 12
    
  # Publication-quality styling
  publication:
    figure.figsize: [8, 6]
    figure.dpi: 300
    axes.grid: false
    axes.labelsize: 11
    axes.titlesize: 12
    font.family: serif
    legend.fontsize: 9

palettes:
  default:
    primary: "#2E86AB"
    secondary: "#A23B72"
    accent: "#F18F01"

outputs:
  base_path: "data/08_reporting"
  naming_convention: "{purpose}_{condition}_{timestamp}"
  
defaults:
  purpose: exploratory
  format: png
  bbox_inches: tight
```

### Step 4: Update Data Catalog

Modify your `conf/base/catalog.yml` to use FigureDataSet instead of standard file outputs:

```yaml
# Before: Manual figure management (remove these entries)
# training_plot:
#   type: matplotlib.MatplotlibWriter
#   filepath: data/08_reporting/training_plot.png

# validation_metrics:
#   type: matplotlib.MatplotlibWriter  
#   filepath: data/08_reporting/validation_metrics.png

# After: FigRegistry automated figure management
training_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/training_plot.png
  purpose: exploratory
  condition_param: experiment_condition
  style_params:
    figure.facecolor: white
    axes.spines.top: false
    axes.spines.right: false
  save_args:
    bbox_inches: tight
    transparent: false

validation_metrics:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/validation_metrics.png
  purpose: presentation
  condition_param: model_type
  style_params:
    figure.dpi: 200
  save_args:
    bbox_inches: tight
    format: png

# For publication-ready figures
publication_results:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/publication_results.pdf
  purpose: publication
  condition_param: analysis_type
  save_args:
    bbox_inches: tight
    format: pdf
    transparent: true
```

### Step 5: Update Pipeline Parameters

Add condition parameters to your `conf/base/parameters.yml` for dynamic styling:

```yaml
# Add these parameters for condition-based styling
experiment_condition: "baseline_model"
model_type: "random_forest"
analysis_type: "performance_comparison"

# Your existing parameters remain unchanged
model_options:
  n_estimators: 100
  max_depth: 10
  random_state: 42

# ... other existing parameters
```

### Step 6: Refactor Pipeline Nodes

Update your pipeline nodes to remove manual figure management and return matplotlib figure objects:

#### Before: Manual Figure Management

```python
# src/<project_name>/pipelines/data_science/nodes.py
import matplotlib.pyplot as plt
import seaborn as sns

def create_training_plot(training_data: pd.DataFrame) -> None:
    """Create training visualization with manual styling."""
    # Manual styling configuration
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 6), dpi=100)
    plt.rcParams.update({
        'axes.grid': True,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10
    })
    
    # Create plot
    plt.plot(training_data['epoch'], training_data['loss'], label='Training Loss')
    plt.plot(training_data['epoch'], training_data['val_loss'], label='Validation Loss')
    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Manual save with hardcoded path
    plt.savefig('data/08_reporting/training_plot.png', 
                bbox_inches='tight', dpi=100)
    plt.close()

def create_validation_metrics(results: pd.DataFrame) -> None:
    """Create validation metrics with manual styling."""
    # Duplicate styling configuration
    plt.figure(figsize=(12, 8), dpi=150)
    plt.rcParams.update({
        'axes.grid': False,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'font.weight': 'bold'
    })
    
    # Create plot
    sns.barplot(data=results, x='metric', y='value')
    plt.title('Model Performance Metrics')
    plt.xticks(rotation=45)
    
    # Manual save with hardcoded path
    plt.savefig('data/08_reporting/validation_metrics.png',
                bbox_inches='tight', dpi=150)
    plt.close()
```

#### After: Automated Figure Management

```python
# src/<project_name>/pipelines/data_science/nodes.py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

def create_training_plot(training_data: pd.DataFrame) -> Figure:
    """Create training visualization with automated styling."""
    # Create figure without manual styling - FigRegistry handles this
    fig, ax = plt.subplots()
    
    # Focus on data visualization logic only
    ax.plot(training_data['epoch'], training_data['loss'], label='Training Loss')
    ax.plot(training_data['epoch'], training_data['val_loss'], label='Validation Loss')
    ax.set_title('Model Training Progress')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Return figure object - no manual save required
    return fig

def create_validation_metrics(results: pd.DataFrame) -> Figure:
    """Create validation metrics with automated styling."""
    # Create figure without manual styling
    fig, ax = plt.subplots()
    
    # Focus on data visualization logic only
    sns.barplot(data=results, x='metric', y='value', ax=ax)
    ax.set_title('Model Performance Metrics')
    ax.tick_params(axis='x', rotation=45)
    
    # Return figure object - no manual save required
    return fig

def create_publication_results(analysis_data: pd.DataFrame) -> Figure:
    """Create publication-ready results visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Performance comparison
    analysis_data.plot(kind='bar', ax=ax1)
    ax1.set_title('Performance Comparison')
    ax1.set_ylabel('Accuracy Score')
    
    # Right subplot: Feature importance
    feature_importance = analysis_data.groupby('feature')['importance'].mean()
    feature_importance.plot(kind='barh', ax=ax2)
    ax2.set_title('Feature Importance')
    ax2.set_xlabel('Importance Score')
    
    plt.tight_layout()
    return fig
```

### Step 7: Update Pipeline Definition

Update your pipeline to use the new node signatures:

```python
# src/<project_name>/pipelines/data_science/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import create_training_plot, create_validation_metrics, create_publication_results

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_training_plot,
            inputs="training_data",
            outputs="training_plot",  # Now outputs to FigureDataSet
            name="create_training_plot_node",
        ),
        node(
            func=create_validation_metrics,
            inputs="model_results",
            outputs="validation_metrics",  # Now outputs to FigureDataSet
            name="create_validation_metrics_node",
        ),
        node(
            func=create_publication_results,
            inputs="analysis_data",
            outputs="publication_results",  # Now outputs to FigureDataSet
            name="create_publication_results_node",
        ),
    ])
```

## Validation Steps

### Step 1: Verify Installation

Check that figregistry-kedro is properly installed:

```bash
python -c "import figregistry_kedro; print('✓ FigRegistry-Kedro installed successfully')"
```

### Step 2: Validate Configuration

Test configuration loading:

```python
# Run this in a Python shell within your project directory
from figregistry_kedro.config import init_config
from kedro.config import ConfigLoader

config_loader = ConfigLoader("conf")
config = init_config(config_loader, "base")
if config:
    print("✓ Configuration loaded successfully")
else:
    print("✗ Configuration loading failed")
```

### Step 3: Test Pipeline Execution

Run your pipeline to ensure figures are generated correctly:

```bash
kedro run --pipeline=data_science
```

Check that figures are created in the expected locations with proper styling applied.

### Step 4: Verify Hook Registration

Confirm hooks are registered by checking the logs for initialization messages:

```bash
kedro run --pipeline=data_science --log-level=DEBUG
```

Look for log messages like:
- `"Initializing FigRegistry configuration for environment: base"`
- `"FigRegistry initialization completed successfully"`

## Migration Best Practices

### 1. Gradual Migration Strategy

Migrate one pipeline at a time rather than all at once:

```python
# Migrate high-impact visualizations first
priority_pipelines = [
    "data_science",      # Core analysis figures
    "reporting",         # Customer-facing reports
    "model_evaluation"   # Performance metrics
]

# Keep existing manual figures during transition
legacy_pipelines = [
    "exploratory_analysis",  # Migrate later
    "data_quality_checks"    # Migrate last
]
```

### 2. Condition Parameter Strategy

Design meaningful condition parameters that reflect your experimental structure:

```yaml
# Good: Semantic condition parameters
parameters:
  experiment_condition: "baseline_vs_treatment"
  model_architecture: "transformer_large"
  dataset_version: "v2.1_cleaned"

# Avoid: Generic or unclear conditions
parameters:
  condition: "test1"
  type: "A"
```

### 3. Style Organization

Organize styles by output purpose rather than individual plots:

```yaml
# Good: Purpose-based organization
styles:
  exploratory:     # For quick analysis
    figure.dpi: 100
    figure.figsize: [10, 6]
  
  presentation:    # For stakeholder meetings
    figure.dpi: 150
    figure.figsize: [12, 8]
    font.size: 14
  
  publication:     # For papers/reports
    figure.dpi: 300
    figure.figsize: [8, 6]
    font.family: serif

# Avoid: Plot-specific styles
styles:
  training_plot_style:
    # Too specific
  validation_chart_format:
    # Hard to reuse
```

### 4. Error Handling

Configure appropriate fallback behavior for production systems:

```python
# In settings.py
HOOKS = (
    FigRegistryHooks(
        auto_initialize=True,
        fallback_on_error=True,  # Continue if FigRegistry fails
        enable_performance_monitoring=False  # Disable in production
    ),
)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "FigRegistry not initialized" Error

**Symptoms**: Pipeline fails with configuration not found errors

**Solution**: Verify hook registration and configuration file paths

```python
# Check hook registration in settings.py
from figregistry_kedro.hooks import FigRegistryHooks
print("✓ Hooks imported successfully")

# Verify configuration file exists
from pathlib import Path
config_path = Path("conf/base/figregistry.yml")
if config_path.exists():
    print("✓ Configuration file found")
else:
    print("✗ Configuration file missing")
```

#### Issue: Styling Not Applied

**Symptoms**: Figures generated but don't show expected styling

**Solution**: Check condition parameter resolution

```python
# Add debug logging to verify condition resolution
import logging
logging.getLogger("figregistry_kedro").setLevel(logging.DEBUG)

# Run pipeline and check logs for:
# "Resolved condition 'experiment_condition' = 'baseline_model'"
```

#### Issue: Performance Degradation

**Symptoms**: Pipeline runs significantly slower after migration

**Solution**: Enable performance monitoring and optimize configuration

```python
# Enable performance monitoring
HOOKS = (
    FigRegistryHooks(
        enable_performance_monitoring=True,
        max_initialization_time=0.005  # 5ms limit
    ),
)

# Check cache statistics
from figregistry_kedro.datasets import FigureDataSet
stats = FigureDataSet.get_performance_stats()
print(f"Average save time: {stats['avg_save_time']*1000:.2f}ms")
```

#### Issue: Import Errors

**Symptoms**: Cannot import figregistry_kedro components

**Solution**: Verify installation and dependencies

```bash
# Reinstall with verbose output
pip install --upgrade --force-reinstall figregistry-kedro

# Check dependency versions
pip show figregistry-kedro
pip show kedro
pip show figregistry
```

### Environment-Specific Configuration

For multi-environment projects, create environment-specific overrides:

```yaml
# conf/local/figregistry.yml (for development)
styles:
  exploratory:
    figure.dpi: 72  # Lower resolution for faster development

# conf/production/figregistry.yml (for production)
styles:
  presentation:
    figure.dpi: 300  # High resolution for production outputs
  publication:
    figure.dpi: 600  # Ultra-high resolution for publications
```

## Advanced Migration Scenarios

### Migrating Complex Subplots

For complex matplotlib figures with subplots:

```python
# Before: Manual subplot management
def create_complex_dashboard():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    # ... complex plotting logic ...
    plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

# After: Return figure for automated management
def create_complex_dashboard() -> Figure:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ... same plotting logic ...
    return fig  # FigRegistry handles sizing and saving
```

### Migrating Animation or Interactive Plots

For animated or interactive visualizations:

```yaml
# Catalog configuration for animations
animated_results:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/animated_results.gif
  purpose: presentation
  save_args:
    format: gif
    writer: pillow
    fps: 2
```

### Integrating with Existing Tools

For projects using other visualization libraries:

```python
# Plotly integration example
import plotly.graph_objects as go
from plotly.io import to_image
import matplotlib.pyplot as plt

def create_plotly_figure() -> Figure:
    """Convert Plotly figure to matplotlib for FigRegistry."""
    # Create Plotly figure
    plotly_fig = go.Figure()
    plotly_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    
    # Convert to matplotlib via image
    img_bytes = to_image(plotly_fig, format="png")
    
    # Create matplotlib figure with image
    fig, ax = plt.subplots()
    ax.imshow(plt.imread(io.BytesIO(img_bytes)))
    ax.axis('off')
    
    return fig
```

## Migration Checklist

Use this checklist to ensure complete migration:

### Pre-Migration
- [ ] Backup existing project
- [ ] Document current figure output locations
- [ ] Identify all nodes with `plt.savefig()` calls
- [ ] Review existing styling patterns
- [ ] Test current pipeline functionality

### Migration Implementation
- [ ] Install figregistry-kedro dependency
- [ ] Register FigRegistryHooks in settings.py
- [ ] Create conf/base/figregistry.yml configuration
- [ ] Update catalog.yml with FigureDataSet entries
- [ ] Add condition parameters to parameters.yml
- [ ] Refactor pipeline nodes to return Figure objects
- [ ] Remove manual plt.savefig() calls
- [ ] Update pipeline definitions

### Post-Migration Validation
- [ ] Verify all tests pass
- [ ] Check figure output quality and styling
- [ ] Validate performance impact (<5% overhead)
- [ ] Test multi-environment configuration
- [ ] Verify version control integration
- [ ] Update documentation
- [ ] Train team on new workflow

## Conclusion

The migration to figregistry-kedro transforms your Kedro project from manual figure management to an automated, configuration-driven approach. This results in:

- **50-80% reduction** in visualization-related code
- **Centralized styling** management across all pipelines
- **Automatic consistency** across experimental conditions
- **Improved maintainability** through separation of concerns
- **Enhanced reproducibility** with version-controlled styling

The migration process typically takes 1-2 days for small projects and up to a week for large, complex pipelines. The long-term benefits in maintainability and consistency make this investment worthwhile for any data science project with significant visualization requirements.

For additional support and advanced use cases, refer to the [figregistry-kedro documentation](https://figregistry-kedro.readthedocs.io) or the project's [GitHub repository](https://github.com/blitzy-public-samples/figregistry-kedro).