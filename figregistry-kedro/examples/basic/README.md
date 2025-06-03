# FigRegistry-Kedro Basic Integration Example

This example demonstrates the seamless integration of FigRegistry's automated figure styling and management capabilities within Kedro data pipelines. Through this minimal project, you'll experience zero-touch figure management that eliminates manual `plt.savefig()` calls while providing consistent, publication-ready visualizations across your entire pipeline.

## Overview

The FigRegistry-Kedro integration transforms how data scientists and ML engineers handle visualization in their pipelines by automatically applying condition-based styling and managing figure outputs through Kedro's catalog system. This example showcases the core integration features that enable automated figure management without requiring any changes to your existing pipeline nodes.

### What This Example Demonstrates

- **Zero-Touch Figure Management**: Pipeline nodes output raw matplotlib figures; the catalog handles all styling and persistence automatically
- **Configuration Bridge**: Seamless merging of FigRegistry's YAML configuration with Kedro's environment-specific settings
- **Condition-Based Styling**: Automatic style application based on experimental conditions from pipeline parameters
- **Lifecycle Integration**: FigRegistry initialization through Kedro hooks without manual configuration management
- **Versioned Output**: Integration with Kedro's versioning system for experiment tracking and reproducibility

## Key Integration Features

### F-005: FigureDataSet Integration
- **Custom Kedro Dataset**: `figregistry_kedro.datasets.FigureDataSet` automatically applies styling during catalog save operations
- **Specialized Parameters**: `purpose`, `condition_param`, and `style_params` enable fine-grained control over figure styling
- **Seamless Interception**: Figure objects are automatically styled without modifying node implementations

### F-006: Lifecycle Hooks
- **Automatic Initialization**: `FigRegistryHooks` register during Kedro startup for transparent integration
- **Context Management**: Configuration state maintained throughout pipeline execution
- **Non-Invasive**: No changes required to existing pipeline code or node functions

### F-007: Configuration Bridge
- **Unified Configuration**: `FigRegistryConfigBridge` merges Kedro and FigRegistry configurations
- **Environment Support**: Works with Kedro's environment-specific configuration patterns
- **Precedence Rules**: Clear hierarchy for configuration conflict resolution

## Installation and Setup

### Prerequisites

- Python 3.10+ 
- Kedro 0.18.0+
- FigRegistry 0.3.0+

### Installation

```bash
# Install the figregistry-kedro plugin
pip install figregistry-kedro

# Or install from source for development
pip install -e "figregistry-kedro[dev]"

# Navigate to the basic example
cd figregistry-kedro/examples/basic
```

### Development Setup

For development and testing:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Project Structure

This basic example follows Kedro project conventions while showcasing FigRegistry integration:

```
figregistry-kedro/examples/basic/
├── README.md                          # This documentation
├── pyproject.toml                     # Project dependencies and configuration
├── .kedro.yml                         # Kedro project metadata
│
├── conf/                              # Configuration directory
│   └── base/                          # Base environment configuration
│       ├── catalog.yml                # Data catalog with FigureDataSet entries
│       ├── figregistry.yml            # FigRegistry configuration for Kedro
│       ├── parameters.yml             # Pipeline parameters for condition resolution
│       └── logging.yml                # Logging configuration
│
├── src/                               # Source code directory
│   └── figregistry_kedro_basic_example/
│       ├── __init__.py                # Package initialization
│       ├── settings.py                # Kedro settings with FigRegistry hooks
│       └── pipelines/                 # Pipeline definitions
│           └── data_visualization/    # Example visualization pipeline
│               ├── __init__.py
│               ├── nodes.py           # Pipeline nodes generating figures
│               └── pipeline.py        # Pipeline definition
│
└── data/                              # Data directory (Kedro convention)
    ├── 01_raw/                        # Raw data inputs
    ├── 02_intermediate/               # Intermediate data
    ├── 03_primary/                    # Primary data
    └── 08_reporting/                  # Output figures (auto-generated)
```

## Configuration Deep Dive

### Kedro Data Catalog (`conf/base/catalog.yml`)

The catalog configuration demonstrates FigureDataSet usage with specialized parameters:

```yaml
# Example FigureDataSet configuration
training_performance_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/training_performance.png
  purpose: "training_analysis"
  condition_param: "params:experiment.condition"
  style_params:
    figsize: [10, 6]
    dpi: 300
  versioned: true

validation_metrics_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/validation_metrics.png
  purpose: "validation_analysis" 
  condition_param: "params:experiment.condition"
  versioned: true
```

**Key Parameters:**
- `purpose`: Links to FigRegistry style configuration sections
- `condition_param`: Resolves experimental conditions from pipeline parameters
- `style_params`: Additional matplotlib parameters for fine-tuning
- `versioned`: Enables Kedro's built-in versioning for experiment tracking

### FigRegistry Configuration (`conf/base/figregistry.yml`)

This file demonstrates the configuration bridge between FigRegistry and Kedro:

```yaml
# FigRegistry configuration adapted for Kedro integration
style:
  training_analysis:
    baseline:
      color: "#2E86AB"
      marker: "o"
      linestyle: "-"
      linewidth: 2
    experimental:
      color: "#A23B72"
      marker: "s" 
      linestyle: "--"
      linewidth: 2
      
  validation_analysis:
    baseline:
      color: "#F18F01"
      marker: "^"
      linestyle: "-"
    experimental:
      color: "#C73E1D"
      marker: "v"
      linestyle: "-."

output:
  base_path: "data/08_reporting"
  timestamp: true
  slug_separator: "_"

palette:
  primary: ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
  secondary: ["#5E9DAF", "#B86B87", "#F3A332", "#D15A40"]
```

### Pipeline Parameters (`conf/base/parameters.yml`)

Parameters drive condition-based styling through the `condition_param` mechanism:

```yaml
# Experimental conditions for styling resolution
experiment:
  condition: "baseline"  # This value drives style selection
  model_type: "random_forest"
  train_size: 0.8
  random_state: 42

# Data processing parameters
data_processing:
  n_samples: 1000
  noise_level: 0.1
  feature_count: 10

# Visualization parameters  
visualization:
  show_confidence_intervals: true
  include_baseline_comparison: true
  performance_metrics: ["accuracy", "precision", "recall", "f1"]
```

## Running the Example

### Basic Execution

Run the complete pipeline to see FigRegistry integration in action:

```bash
# Execute the full pipeline
kedro run

# Run specific pipeline
kedro run --pipeline data_visualization

# Run with different experimental condition
kedro run --params experiment.condition=experimental
```

### Step-by-Step Execution

1. **Hook Registration**: FigRegistryHooks automatically register at startup
2. **Configuration Loading**: ConfigBridge merges FigRegistry and Kedro configurations
3. **Pipeline Execution**: Nodes generate matplotlib figures without manual saving
4. **Automatic Styling**: FigureDataSet applies condition-based styles during catalog saves
5. **Versioned Output**: Figures saved with Kedro's versioning and FigRegistry's naming

### Expected Output

After running `kedro run`, you'll find automatically styled figures in:

```
data/08_reporting/
├── training_performance_[timestamp].png
├── validation_metrics_[timestamp].png
└── [additional pipeline outputs...]
```

Each figure will be automatically styled according to:
- The `condition` parameter value (`baseline` or `experimental`)
- The `purpose` mapping in your FigRegistry configuration
- Any additional `style_params` specified in the catalog

## Understanding the Integration

### Zero-Touch Figure Management

Traditional approach (what you **don't** need to do):
```python
# ❌ Manual approach - no longer needed
def create_training_plot(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['x'], data['y'], color='blue', marker='o')
    plt.savefig('training_performance.png', dpi=300, bbox_inches='tight')
    return fig
```

FigRegistry-Kedro approach (what you **do**):
```python
# ✅ Automated approach - just return the figure
def create_training_plot(data):
    fig, ax = plt.subplots()  # FigRegistry handles sizing
    ax.plot(data['x'], data['y'])  # FigRegistry handles styling
    return fig  # Catalog handles saving with versioning
```

### Condition-Based Styling Flow

1. **Parameter Resolution**: `condition_param: "params:experiment.condition"` extracts `"baseline"` from parameters
2. **Style Lookup**: FigRegistry finds `style.training_analysis.baseline` configuration  
3. **Style Application**: Matplotlib properties automatically applied to figure
4. **Automatic Saving**: Output manager handles file naming, versioning, and persistence

### Configuration Precedence

The configuration bridge follows this precedence hierarchy:
1. **Catalog-level `style_params`**: Highest priority for figure-specific overrides
2. **FigRegistry condition styles**: Condition-based styling from `figregistry.yml`
3. **FigRegistry defaults**: Base styling defaults and palette definitions
4. **Matplotlib defaults**: Fallback to standard matplotlib styling

## Advanced Usage Patterns

### Multiple Experimental Conditions

Modify parameters to test different styling conditions:

```bash
# Test different experimental conditions
kedro run --params experiment.condition=baseline
kedro run --params experiment.condition=experimental

# Compare outputs to see automatic styling differences
ls -la data/08_reporting/
```

### Environment-Specific Configuration

Leverage Kedro's environment system for different deployment scenarios:

```bash
# Development environment with debug styling
kedro run --env local

# Production environment with publication-ready styling  
kedro run --env production
```

### Pipeline Parameter Overrides

Dynamically control styling through runtime parameters:

```bash
# Override specific parameters
kedro run --params experiment.condition=experimental,visualization.show_confidence_intervals=false
```

## Troubleshooting

### Common Issues and Solutions

#### FigRegistry Configuration Not Loading
**Problem**: Figures appear unstyled or use matplotlib defaults
**Solution**: 
```bash
# Check configuration files exist
ls -la conf/base/figregistry.yml

# Verify hook registration in logs  
kedro run --log-level DEBUG | grep -i figregistry
```

#### Style Not Applied
**Problem**: Condition-based styling not working
**Solution**:
- Verify `condition_param` points to valid parameter path
- Check that condition value exists in `figregistry.yml` style mapping
- Ensure parameter value matches style condition key exactly

#### Dataset Configuration Errors
**Problem**: FigureDataSet not recognized
**Solution**:
```bash
# Verify figregistry-kedro installation
pip list | grep figregistry-kedro

# Check catalog configuration syntax
kedro catalog list | grep FigureDataSet
```

#### Version Compatibility Issues
**Problem**: Plugin not working with Kedro version
**Solution**:
```bash
# Check version compatibility
pip show kedro figregistry figregistry-kedro

# Upgrade to compatible versions
pip install "kedro>=0.18.0,<0.20.0" "figregistry>=0.3.0"
```

### Debug Mode

Enable detailed logging to troubleshoot integration issues:

```bash
# Run with debug logging
PYTHONPATH=src kedro run --log-level DEBUG

# Filter FigRegistry-specific logs
kedro run --log-level DEBUG 2>&1 | grep -E "(figregistry|FigRegistry)"
```

### Configuration Validation

Validate your configuration setup:

```python
# Test configuration loading manually
from figregistry_kedro.config import FigRegistryConfigBridge
from kedro.config import ConfigLoader

config_loader = ConfigLoader("conf")
bridge = FigRegistryConfigBridge(config_loader)
merged_config = bridge.get_merged_config()
print(merged_config)
```

## Next Steps

### Extending the Example

1. **Add Custom Conditions**: Extend `figregistry.yml` with additional experimental conditions
2. **Multiple Purposes**: Create different visualization types with distinct styling purposes
3. **Advanced Styling**: Leverage FigRegistry's full styling capabilities (palettes, fonts, layouts)
4. **Pipeline Integration**: Incorporate into existing Kedro pipelines for immediate productivity gains

### Migration to Production

1. **Environment Configuration**: Set up environment-specific FigRegistry configurations
2. **CI/CD Integration**: Include figure validation in your testing pipelines  
3. **Team Collaboration**: Share FigRegistry configurations across team members
4. **Version Control**: Track both data and visualization changes through Kedro's versioning

### Advanced Features

Explore additional FigRegistry-Kedro capabilities:
- **Multi-environment styling**: Different styles for development vs. production
- **Complex condition hierarchies**: Nested experimental conditions
- **Custom dataset parameters**: Fine-grained control over figure properties
- **Integration with MLflow**: Experiment tracking with visualization artifacts

## Support and Contributing

### Getting Help

- **Documentation**: [FigRegistry-Kedro Documentation](https://figregistry-kedro.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/figregistry/figregistry-kedro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/figregistry/figregistry-kedro/discussions)

### Contributing

We welcome contributions! See our [Contributing Guide](../../CONTRIBUTING.md) for details on:
- Setting up development environment
- Running tests and linting
- Submitting pull requests
- Code style and conventions

### Community

Join the FigRegistry community:
- **Kedro Plugin Registry**: Listed in the official Kedro plugin directory
- **Blog Posts**: Tutorial content and best practices
- **Conference Talks**: Presentations on scientific visualization automation

---

**Ready to automate your pipeline visualizations?** Start with this basic example, then adapt the patterns to your own Kedro projects for immediate productivity gains and consistent, publication-ready figures.