# FigRegistry-Kedro Integration: Basic Example

[![Kedro](https://img.shields.io/badge/kedro-0.18.0+-green)](https://kedro.org)
[![FigRegistry](https://img.shields.io/badge/figregistry-0.3.0+-blue)](https://github.com/figregistry/figregistry)
[![Python](https://img.shields.io/badge/python-3.10+-yellow)](https://python.org)

> **Complete demonstration of automated figure styling and versioning within Kedro pipelines**

This example showcases the seamless integration between [Kedro](https://kedro.org) and [FigRegistry](https://github.com/figregistry/figregistry), demonstrating how to eliminate manual `plt.savefig()` calls while enabling automated condition-based styling, versioning, and organized figure management in data science workflows.

## 🎯 What You'll Learn

By running this example, you'll understand:

- **Zero-Touch Figure Management**: How FigRegistry eliminates manual figure saving code from pipeline nodes
- **Automated Condition-Based Styling**: Dynamic figure styling based on experimental parameters
- **Kedro-FigRegistry Configuration Bridge**: Seamless integration between both configuration systems
- **Enterprise-Grade Versioning**: Automated figure versioning integrated with Kedro's catalog system
- **Production-Ready Workflows**: Professional figure management without compromising development velocity

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ installed
- Virtual environment tool (`conda`, `venv`, or `virtualenv`)
- Git (for cloning the repository)

### Installation

1. **Navigate to the example directory**:
   ```bash
   cd figregistry-kedro/examples/basic
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using conda (recommended)
   conda create -n figregistry-kedro-basic python=3.10
   conda activate figregistry-kedro-basic
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the project and dependencies**:
   ```bash
   pip install -e .
   ```

4. **Run the example pipeline**:
   ```bash
   kedro run
   ```

5. **View the generated figures**:
   ```bash
   # Explore the automated outputs
   ls -la data/08_reporting/
   
   # View specific examples
   open data/08_reporting/exploratory/data_exploration.png
   open data/08_reporting/presentation/validation_results.pdf
   ```

That's it! You've just experienced automated figure styling in action. 🎉

## 📊 What Just Happened?

When you ran `kedro run`, the following automated process occurred:

1. **Pipeline Execution**: Kedro nodes created matplotlib figures as normal Python objects
2. **Automatic Styling**: FigRegistry applied condition-based styling based on `experiment_condition` parameter
3. **Intelligent Output**: Figures were automatically saved with appropriate quality, format, and naming
4. **Zero Manual Code**: No `plt.savefig()` calls were needed in any pipeline nodes

### Key Integration Points

- **FigureDataSet**: Custom Kedro dataset that bridges matplotlib figures with FigRegistry styling
- **Configuration Bridge**: Seamless merging of Kedro parameters with FigRegistry configuration
- **Lifecycle Hooks**: Automatic FigRegistry initialization at pipeline startup

## 🏗️ Project Structure

```
figregistry-kedro-basic-example/
├── conf/base/
│   ├── catalog.yml          # Dataset definitions with FigRegistry integration
│   ├── figregistry.yml      # FigRegistry styling configuration
│   └── parameters.yml       # Experimental parameters for condition-based styling
├── src/figregistry_kedro_basic_example/
│   ├── settings.py          # Kedro settings with FigRegistry hooks
│   └── pipelines/
│       └── data_visualization/
│           ├── pipeline.py  # Pipeline definition
│           └── nodes.py     # Pipeline nodes that create figures
├── data/08_reporting/       # Generated figures (created after running)
├── pyproject.toml          # Project dependencies and configuration
└── README.md               # This file
```

## ⚙️ Configuration Explained

### 1. Catalog Configuration (`conf/base/catalog.yml`)

The catalog showcases three levels of FigRegistry integration:

#### Basic Integration
```yaml
exploratory_data_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/exploratory/data_exploration.png
  purpose: exploratory                    # Maps to styling purpose
  condition_param: experiment_condition   # Dynamic styling parameter
```

#### Advanced Styling
```yaml
validation_results:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/presentation/validation_results.pdf
  purpose: presentation
  condition_param: experiment_condition
  style_params:
    # Custom overrides for specific requirements
    figure.dpi: 300
    font.size: 14
    axes.titlesize: 16
  save_args:
    format: pdf                          # Vector format for presentations
    bbox_inches: tight
  versioned: true                        # Enable Kedro versioning
```

#### Publication Quality
```yaml
manuscript_figure_1:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/publication/manuscript_figure_1.eps
  purpose: publication
  condition_param: experiment_condition
  style_params:
    font.family: serif                   # Academic publication standards
    font.size: 10
    figure.figsize: [7, 5]              # Single-column width
  save_args:
    format: eps                          # Vector format for journals
    dpi: 600                            # High resolution for print
```

### 2. FigRegistry Configuration (`conf/base/figregistry.yml`)

Defines condition-based styling that automatically applies based on experimental parameters:

```yaml
styles:
  treatment_group_a:
    color: "#FF6B6B"           # Vibrant red for treatment group A
    marker: "o"                # Circle markers
    linestyle: "-"             # Solid lines
    linewidth: 2.5
    label: "Treatment Group A"
    
  treatment_group_b:
    color: "#4ECDC4"           # Teal for treatment group B
    marker: "s"                # Square markers
    linestyle: "-"
    linewidth: 2.5
    label: "Treatment Group B"
    
  control_group:
    color: "#45B7D1"           # Blue for control group
    marker: "^"                # Triangle markers
    linestyle: "--"            # Dashed lines for reference
    linewidth: 2.0
    label: "Control Group"
```

### 3. Lifecycle Hooks (`src/.../settings.py`)

Enables automatic FigRegistry initialization:

```python
from figregistry_kedro.hooks import FigRegistryHooks

# Register hooks for automatic initialization
HOOKS = (FigRegistryHooks(),)
```

## 🔧 Running the Example

### Standard Execution

```bash
# Run the complete pipeline
kedro run

# Run specific pipeline
kedro run --pipeline data_visualization

# Run with different parameters
kedro run --params experiment_condition:treatment_group_b
```

### Development Mode

```bash
# Run with verbose logging
kedro run --log-level DEBUG

# Run specific nodes
kedro run --node exploratory_analysis

# Run with parameter overrides
kedro run --params experiment_phase:training,model_type:random_forest
```

### Exploring Outputs

After running the pipeline, explore the generated figures:

```bash
# View directory structure
tree data/08_reporting/

# Check exploratory outputs (PNG format)
ls data/08_reporting/exploratory/

# Check presentation outputs (PDF format)
ls data/08_reporting/presentation/

# Check publication outputs (vector formats)
ls data/08_reporting/publication/
```

## 📈 Understanding the Results

### Automatic Styling Examples

The pipeline demonstrates condition-based styling through several scenarios:

1. **Exploratory Analysis** (`experiment_condition: treatment_group_a`):
   - **Result**: Red color (#FF6B6B), circle markers, solid lines
   - **Format**: PNG at 150 DPI for fast iteration
   - **Usage**: Quick data exploration and development

2. **Presentation Figures** (same condition):
   - **Result**: Same styling with enhanced presentation formatting
   - **Format**: PDF at 300 DPI with larger fonts
   - **Usage**: Stakeholder meetings and presentations

3. **Publication Quality** (same condition):
   - **Result**: Same styling with publication-specific formatting
   - **Format**: EPS at 600 DPI with serif fonts
   - **Usage**: Academic manuscripts and journals

### Performance Metrics

The integration achieves these performance targets:

- **Style Resolution**: <1ms per figure (cached lookups)
- **Save Overhead**: <5% compared to manual `plt.savefig()`
- **Pipeline Startup**: <100ms for configuration initialization
- **Memory Usage**: Minimal overhead with efficient caching

## 💡 Key Features Demonstrated

### 1. Zero-Touch Figure Management

**Before (Manual Approach)**:
```python
def create_plot(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(data['x'], data['y'], color='red', marker='o')
    ax.set_title('Results')
    plt.savefig('output.png', dpi=150, bbox_inches='tight')
    plt.close()
    return None  # Nothing returned to catalog
```

**After (FigRegistry Integration)**:
```python
def create_plot(data):
    fig, ax = plt.subplots()  # Figure size handled by FigRegistry
    ax.plot(data['x'], data['y'])  # Styling handled automatically
    ax.set_title('Results')
    return fig  # Return figure to catalog for automatic styling
```

### 2. Condition-Based Styling

Parameters automatically drive figure appearance:

```yaml
# In parameters.yml
experiment_condition: treatment_group_a

# Results in automatic application of:
# - Red color (#FF6B6B)
# - Circle markers
# - Solid lines
# - "Treatment Group A" labels
```

### 3. Purpose-Driven Output Quality

The same data generates different output formats based on intended use:

- **Exploratory**: PNG, 150 DPI, fast iteration
- **Presentation**: PDF, 300 DPI, enhanced fonts
- **Publication**: EPS, 600 DPI, serif fonts, precise dimensions

### 4. Versioning Integration

Kedro's versioning system works seamlessly with FigRegistry:

```yaml
validation_results:
  type: figregistry_kedro.FigureDataSet
  versioned: true  # Automatic version tracking
```

Results in timestamped figure outputs:
```
data/08_reporting/presentation/validation_results/
├── 2024-01-15T14.30.25.123Z/
│   └── validation_results.pdf
└── 2024-01-15T15.45.12.456Z/
    └── validation_results.pdf
```

## 🔍 Configuration Bridge Details

The FigRegistry-Kedro configuration bridge enables seamless integration between both systems:

### Precedence Rules

1. **Highest Priority**: Dataset-specific `style_params` in catalog.yml
2. **Medium Priority**: Kedro parameters.yml values
3. **Lowest Priority**: FigRegistry figregistry.yml defaults

### Parameter Resolution

```python
# In parameters.yml
experiment_condition: treatment_group_a

# In catalog.yml
condition_param: experiment_condition

# Results in style lookup:
figregistry.get_style("treatment_group_a")
```

### Environment-Specific Configuration

```yaml
# In conf/local/parameters.yml (overrides base)
experiment_condition: treatment_group_b

# Automatically resolves to different styling
# without changing pipeline code
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. "FigRegistry not initialized" Error

**Problem**: FigRegistry hooks not properly registered.

**Solution**:
```python
# Check src/figregistry_kedro_basic_example/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)  # Ensure this line exists
```

#### 2. "Condition not found" Warnings

**Problem**: Parameter referenced in `condition_param` doesn't exist.

**Solution**:
```yaml
# Check conf/base/parameters.yml contains:
experiment_condition: treatment_group_a

# Or check condition exists in figregistry.yml:
styles:
  treatment_group_a:  # Must match parameter value
    color: "#FF6B6B"
```

#### 3. Figures Not Styled

**Problem**: FigureDataSet not applied correctly.

**Solution**:
```yaml
# Ensure catalog.yml uses correct dataset type:
my_figure:
  type: figregistry_kedro.FigureDataSet  # Not matplotlib.MatplotlibWriter
  filepath: data/08_reporting/my_figure.png
  purpose: exploratory
  condition_param: experiment_condition
```

#### 4. Import Errors

**Problem**: figregistry-kedro package not properly installed.

**Solution**:
```bash
# Reinstall in development mode
pip install -e .

# Or check installation
pip list | grep figregistry
```

### Debugging Steps

1. **Enable Debug Logging**:
   ```bash
   kedro run --log-level DEBUG
   ```

2. **Check Configuration Merge**:
   ```python
   # In a Python console within the project
   from kedro.framework.startup import bootstrap_project
   from kedro.framework.session import KedroSession
   
   bootstrap_project(".")
   with KedroSession.create() as session:
       context = session.load_context()
       print(context.config_loader["figregistry"])
   ```

3. **Verify Hook Registration**:
   ```bash
   kedro info
   # Should show FigRegistryHooks in the hooks section
   ```

4. **Test FigRegistry Directly**:
   ```python
   import figregistry
   figregistry.init_config()
   style = figregistry.get_style("treatment_group_a")
   print(style)
   ```

### Performance Monitoring

Check if the integration meets performance targets:

```python
import time
from figregistry_kedro.datasets import FigureDataSet

# Time a save operation
start = time.time()
# ... create and save figure ...
overhead = time.time() - start
print(f"Save overhead: {overhead*1000:.2f}ms")
# Target: <50ms overhead
```

## 🎓 Educational Value

This example demonstrates several key software engineering principles:

### 1. Separation of Concerns
- **Pipeline Logic**: Focuses on data processing and figure creation
- **Styling Logic**: Handled separately by FigRegistry configuration
- **Output Management**: Automated by the dataset integration

### 2. Configuration-Driven Development
- Visual styling controlled by configuration files
- No hardcoded styling in source code
- Easy to modify appearance without code changes

### 3. Plugin Architecture
- FigRegistry remains independent from Kedro
- Integration through well-defined interfaces
- Non-invasive enhancement of existing systems

### 4. Performance Optimization
- Caching for repeated style lookups
- Lazy loading of configuration
- Minimal overhead for production usage

## 🚀 Next Steps

After exploring this basic example, consider:

### 1. Advanced Example
Explore the [advanced example](../advanced/README.md) featuring:
- Multi-environment configurations
- Complex experimental designs
- Production deployment patterns
- Performance optimization techniques

### 2. Migration Guide
See the [migration example](../migration/README.md) to understand:
- Converting existing Kedro projects
- Before/after comparison
- Step-by-step migration process

### 3. Custom Integration
Build your own integration:
- Create custom condition parameters
- Design project-specific styling rules
- Implement advanced configuration patterns

### 4. Production Deployment
Scale to production environments:
- CI/CD integration
- Multi-environment configuration management
- Performance monitoring and optimization

## 📚 Additional Resources

### Documentation
- [FigRegistry Documentation](https://figregistry.readthedocs.io/)
- [Kedro Documentation](https://kedro.readthedocs.io/)
- [FigRegistry-Kedro Plugin API Reference](../../docs/api/)

### Examples and Tutorials
- [Advanced Integration Example](../advanced/)
- [Migration from Manual Approach](../migration/)
- [Custom Dataset Development](../../docs/custom-datasets.md)

### Community
- [GitHub Issues](https://github.com/figregistry/figregistry-kedro/issues)
- [Discussion Forum](https://github.com/figregistry/figregistry-kedro/discussions)
- [Kedro Community Slack](https://kedro.org/community)

## 🤝 Contributing

Found an issue or want to improve this example?

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/figregistry/figregistry-kedro.git
cd figregistry-kedro/examples/basic

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run code quality checks
pre-commit run --all-files
```

## 📄 License

This example is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

---

**Happy visualizing with FigRegistry and Kedro!** 🎨📊

*For questions or support, please visit our [GitHub repository](https://github.com/figregistry/figregistry-kedro) or [documentation site](https://figregistry-kedro.readthedocs.io/).*