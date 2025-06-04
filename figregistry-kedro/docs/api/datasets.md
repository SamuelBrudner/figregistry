# FigureDataSet API Reference

## Overview

The `FigureDataSet` class is a custom Kedro `AbstractDataSet` implementation that bridges matplotlib figure objects with FigRegistry's automated styling and versioning system. This dataset enables zero-touch figure management within Kedro pipelines by automatically applying condition-based styling and handling file persistence while maintaining full compatibility with Kedro's catalog system and versioning capabilities.

### Key Features

- **Automated Styling**: Applies FigRegistry's condition-based styling automatically during figure persistence
- **Kedro Integration**: Full compliance with Kedro's `AbstractDataSet` interface and catalog system  
- **Versioning Support**: Compatible with Kedro's dataset versioning for experiment tracking
- **Performance Optimized**: <5% overhead compared to manual matplotlib save operations
- **Thread Safety**: Supports parallel execution with Kedro's concurrent runners
- **Zero Configuration**: Works out-of-the-box with minimal catalog configuration

### Quick Start

```yaml
# catalog.yml
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/analysis_results.png
  purpose: presentation
  condition_param: experiment_condition
```

```python
# In your Kedro node
import matplotlib.pyplot as plt

def create_analysis_plot(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['x'], data['y'])
    ax.set_title('Analysis Results')
    return fig  # FigureDataSet automatically applies styling and saves
```

## Class Reference

### FigureDataSet

```python
class FigureDataSet(AbstractDataset[Figure, Figure])
```

Custom Kedro AbstractDataSet implementation for matplotlib figures with automated FigRegistry styling and versioning integration.

#### Inheritance

- `kedro.io.AbstractDataset[matplotlib.figure.Figure, matplotlib.figure.Figure]`

#### Constructor

```python
def __init__(
    self,
    filepath: str,
    purpose: str = "exploratory", 
    condition_param: Optional[str] = None,
    style_params: Optional[Dict[str, Any]] = None,
    format_kwargs: Optional[Dict[str, Any]] = None,
    load_version: Optional[str] = None,
    save_version: Optional[str] = None,
    versioned: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    enable_caching: bool = True,
    **kwargs
)
```

Initialize FigureDataSet with configuration parameters.

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | **Required** | File path for figure output (required by Kedro convention) |
| `purpose` | `str` | `"exploratory"` | Output categorization: `"exploratory"`, `"presentation"`, or `"publication"` |
| `condition_param` | `Optional[str]` | `None` | Parameter name for dynamic condition resolution from pipeline context |
| `style_params` | `Optional[Dict[str, Any]]` | `None` | Dataset-specific styling overrides for FigRegistry styling |
| `format_kwargs` | `Optional[Dict[str, Any]]` | `None` | Additional arguments passed to matplotlib `savefig()` function |
| `load_version` | `Optional[str]` | `None` | Version string for load operations (Kedro versioning) |
| `save_version` | `Optional[str]` | `None` | Version string for save operations (Kedro versioning) |
| `versioned` | `bool` | `False` | Enable Kedro versioning for this dataset |
| `metadata` | `Optional[Dict[str, Any]]` | `None` | Additional metadata for dataset description |
| `enable_caching` | `bool` | `True` | Enable style resolution caching for performance optimization |
| `**kwargs` | | | Additional parameters passed to parent `AbstractDataset` |

##### Raises

- `FigureDatasetError`: When parameter validation fails
- `ValueError`: When required parameters are missing or invalid

##### Example

```python
# Basic configuration
dataset = FigureDataSet(
    filepath="data/08_reporting/results.png",
    purpose="presentation"
)

# Advanced configuration with versioning
dataset = FigureDataSet(
    filepath="data/08_reporting/model_performance.png",
    purpose="publication",
    condition_param="model_type",
    style_params={"color": "#2E86AB", "linewidth": 2.5},
    format_kwargs={"dpi": 300, "bbox_inches": "tight"},
    versioned=True,
    metadata={"experiment_id": "exp_001"}
)
```

## Core Methods

### _save()

```python
def _save(self, data: Figure) -> None
```

Save matplotlib figure with automated FigRegistry styling.

This method implements the core dataset functionality by intercepting matplotlib figure save operations and automatically applying FigRegistry's condition-based styling before persistence. The implementation maintains compatibility with Kedro's versioning system while providing performance optimization through intelligent caching.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `matplotlib.figure.Figure` | matplotlib Figure object to save with styling |

#### Raises

- `FigureDatasetError`: When save operation fails
- `ValueError`: When figure object is invalid

#### Implementation Details

1. **Figure Validation**: Validates input as matplotlib Figure object
2. **Style Application**: Applies FigRegistry condition-based styling automatically
3. **Path Resolution**: Resolves save path with Kedro versioning support
4. **Directory Creation**: Ensures output directory exists with proper permissions
5. **Format Optimization**: Applies purpose-specific save parameters
6. **Performance Tracking**: Monitors save operation timing for optimization

#### Performance

- Target: <5% overhead compared to manual `matplotlib.pyplot.savefig()` calls
- Typical save time: 10-50ms depending on figure complexity and format
- Memory overhead: Minimal through efficient figure handling

### _load()

```python
def _load(self) -> Figure
```

Load matplotlib figure from saved file.

Note: Loading figures from disk is primarily for debugging and inspection purposes, as matplotlib figures are typically generated during pipeline execution rather than loaded from storage.

#### Returns

| Type | Description |
|------|-------------|
| `matplotlib.figure.Figure` | matplotlib Figure object loaded from file |

#### Raises

- `FigureDatasetError`: When load operation fails  
- `FileNotFoundError`: When figure file does not exist

#### Usage Note

Figure loading recreates a basic matplotlib figure containing the saved image data. This is primarily useful for:

- Debugging saved figure content
- Creating composite figures from saved components
- Pipeline testing and validation workflows

### _describe()

```python
def _describe(self) -> Dict[str, Any]
```

Return description dictionary for dataset introspection.

#### Returns

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Dictionary containing dataset configuration and metadata |

#### Return Value Structure

```python
{
    "filepath": str,                    # Figure output path
    "purpose": str,                     # Output purpose category
    "condition_param": Optional[str],   # Condition parameter name
    "style_params": Dict[str, Any],     # Dataset-specific styling overrides
    "format_kwargs": Dict[str, Any],    # Matplotlib savefig arguments
    "versioned": bool,                  # Kedro versioning enabled
    "load_version": Optional[str],      # Load version identifier
    "save_version": Optional[str],      # Save version identifier
    "protocol": str,                    # File system protocol
    "metadata": Dict[str, Any],         # Additional metadata
    "enable_caching": bool,             # Style caching enabled
    "figregistry_available": bool,      # FigRegistry installation status
    "config_bridge_initialized": bool   # Configuration bridge status
}
```

### _exists()

```python
def _exists(self) -> bool
```

Check if figure file exists at the dataset path.

#### Returns

| Type | Description |
|------|-------------|
| `bool` | `True` if figure file exists, `False` otherwise |

## Configuration Parameters

### Core Parameters

#### filepath

**Type**: `str`  
**Required**: Yes  
**Description**: File path for figure output following Kedro's file path conventions.

```yaml
# Absolute path
filepath: /path/to/project/data/08_reporting/figure.png

# Relative path (recommended)
filepath: data/08_reporting/analysis_results.png

# With environment-specific paths
filepath: ${base_path}/figures/experiment_${experiment_id}.png
```

#### purpose

**Type**: `str`  
**Default**: `"exploratory"`  
**Valid Values**: `"exploratory"`, `"presentation"`, `"publication"`  
**Description**: Output categorization that determines default styling and format parameters.

| Purpose | Use Case | Default DPI | Default Format Options |
|---------|----------|-------------|------------------------|
| `exploratory` | Initial data analysis, debugging | 150 | Standard quality, fast generation |
| `presentation` | Meeting slides, reports | 200 | High quality, optimized for screens |
| `publication` | Academic papers, publications | 300 | Maximum quality, print-ready |

```yaml
# Basic purpose configuration
my_exploratory_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/exploration.png
  purpose: exploratory

my_publication_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/paper_figure_1.png
  purpose: publication
```

#### condition_param

**Type**: `Optional[str]`  
**Default**: `None`  
**Description**: Parameter name for dynamic condition resolution from pipeline context. When specified, the dataset will resolve this parameter from Kedro's runtime context to determine appropriate FigRegistry styling.

```yaml
# Dynamic styling based on experimental condition
experiment_results:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/results_${condition}.png
  purpose: presentation
  condition_param: experiment_condition

# Model comparison with dynamic styling
model_performance:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/model_${model_type}_performance.png
  purpose: publication
  condition_param: model_type
```

**Pipeline Parameter Resolution**:
```python
# In parameters.yml
experiment_condition: "treatment_group_a"
model_type: "random_forest"

# The dataset automatically resolves these values for styling
```

#### style_params

**Type**: `Optional[Dict[str, Any]]`  
**Default**: `None`  
**Description**: Dataset-specific styling overrides that take precedence over FigRegistry's condition-based styling.

```yaml
# Override specific styling properties
custom_styled_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/custom_plot.png
  purpose: presentation
  style_params:
    color: "#2E86AB"
    linewidth: 2.5
    marker: "o"
    alpha: 0.8
    grid: true

# Complex styling overrides
detailed_analysis:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/detailed_analysis.png
  purpose: publication
  style_params:
    color: "#1A1A1A"
    linewidth: 3.0
    marker: "s"
    markersize: 8
    linestyle: "--"
    figsize: [12, 8]
    facecolor: "white"
```

**Supported Style Properties**:
- `color`: Line/marker color (hex, named colors, RGB tuples)
- `linewidth`: Line width in points  
- `linestyle`: Line style (`"-"`, `"--"`, `"-."`, `":"`)
- `marker`: Marker style (`"o"`, `"s"`, `"^"`, `"*"`, etc.)
- `markersize`: Marker size in points
- `alpha`: Transparency (0.0 to 1.0)
- `figsize`: Figure size as `[width, height]` in inches
- `dpi`: Figure resolution (dots per inch)
- `facecolor`: Figure background color
- `grid`: Enable/disable grid lines (`true`/`false`)

### Format Parameters

#### format_kwargs

**Type**: `Optional[Dict[str, Any]]`  
**Default**: `None`  
**Description**: Additional arguments passed directly to matplotlib's `savefig()` function for fine-grained format control.

```yaml
# High-quality publication figure
publication_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/publication_quality.pdf
  purpose: publication
  format_kwargs:
    dpi: 300
    bbox_inches: tight
    pad_inches: 0.1
    transparent: false
    facecolor: white
    edgecolor: none

# Presentation figure with transparency
presentation_slide:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/slide_figure.png
  purpose: presentation
  format_kwargs:
    dpi: 200
    bbox_inches: tight
    transparent: true
    facecolor: none
```

**Common Format Arguments**:
- `dpi`: Resolution in dots per inch (72, 150, 200, 300)
- `bbox_inches`: Bounding box behavior (`"tight"` or specific measurements)
- `pad_inches`: Padding around figure when `bbox_inches="tight"`
- `transparent`: Enable transparent background (`true`/`false`)
- `facecolor`: Figure background color
- `edgecolor`: Figure edge color
- `quality`: JPEG quality (1-100, JPEG format only)
- `optimize`: PNG optimization (`true`/`false`, PNG format only)

### Versioning Parameters

#### versioned

**Type**: `bool`  
**Default**: `False`  
**Description**: Enable Kedro's dataset versioning for experiment tracking and reproducibility.

```yaml
# Enable versioning for experiment tracking
versioned_results:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/experiment_results.png
  versioned: true
  purpose: presentation

# Versioned publication figures
paper_figures:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/paper_figure_{figure_id}.pdf
  versioned: true
  purpose: publication
  format_kwargs:
    dpi: 300
    bbox_inches: tight
```

#### load_version / save_version

**Type**: `Optional[str]`  
**Default**: `None`  
**Description**: Specific version identifiers for load and save operations when versioning is enabled.

```yaml
# Load specific version, save new version
analysis_comparison:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/comparison.png
  versioned: true
  load_version: "2024-01-15T10.30.00.000Z"
  # save_version auto-generated if not specified
```

**Version Format**: Kedro uses ISO 8601 timestamp format: `YYYY-MM-DDTHH.MM.SS.sssZ`

### Performance Parameters

#### enable_caching

**Type**: `bool`  
**Default**: `True`  
**Description**: Enable style resolution caching for improved performance in workflows with repeated styling operations.

```yaml
# Disable caching for debugging
debug_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/debug_plot.png
  enable_caching: false
  purpose: exploratory

# Enable caching for production (default)
production_dashboard:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/dashboard.png
  enable_caching: true
  purpose: presentation
```

**Caching Benefits**:
- Reduces style resolution time for repeated conditions
- Improves performance in large pipeline workflows  
- Thread-safe cache implementation for parallel execution
- Automatic cache size management to prevent memory issues

## Usage Examples

### Basic Figure Pipeline

```python
# nodes.py
import matplotlib.pyplot as plt
import pandas as pd

def create_sales_trend_plot(sales_data: pd.DataFrame) -> plt.Figure:
    """Create sales trend visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sales_data['date'], sales_data['revenue'])
    ax.set_title('Sales Trend Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue ($)')
    
    return fig

def create_regional_comparison(regional_data: pd.DataFrame) -> plt.Figure:
    """Create regional performance comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    regions = regional_data['region'].unique()
    for region in regions:
        region_data = regional_data[regional_data['region'] == region]
        ax.plot(region_data['month'], region_data['sales'], 
                label=region, marker='o')
    
    ax.set_title('Regional Sales Comparison')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales Volume')
    ax.legend()
    
    return fig
```

```yaml
# catalog.yml
sales_trend_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/sales_trend.png
  purpose: presentation
  condition_param: analysis_type

regional_comparison_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/regional_comparison.png
  purpose: publication
  style_params:
    color: "#2E86AB"
    linewidth: 2.0
  format_kwargs:
    dpi: 300
    bbox_inches: tight
```

```python
# pipeline.py
from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_sales_trend_plot,
            inputs="sales_data",
            outputs="sales_trend_figure",
            name="create_sales_trend"
        ),
        node(
            func=create_regional_comparison,
            inputs="regional_sales_data", 
            outputs="regional_comparison_figure",
            name="create_regional_comparison"
        )
    ])
```

### Multi-Environment Configuration

```yaml
# conf/base/catalog.yml
experiment_results:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/experiment_results.png
  purpose: ${purpose_level}
  condition_param: experiment_condition
  versioned: true

model_performance:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/model_performance.png
  purpose: publication
  format_kwargs:
    dpi: ${output_dpi}
    bbox_inches: tight
```

```yaml
# conf/local/parameters.yml
purpose_level: exploratory
output_dpi: 150
experiment_condition: local_testing

# conf/production/parameters.yml  
purpose_level: publication
output_dpi: 300
experiment_condition: production_run
```

### Advanced Styling with Conditions

```python
# Create conditional styling based on model performance
def create_model_evaluation_plot(
    model_results: pd.DataFrame, 
    performance_threshold: float
) -> plt.Figure:
    """Create model evaluation with conditional styling."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot will be automatically styled based on performance level
    ax.scatter(model_results['precision'], model_results['recall'])
    ax.set_title('Model Performance Evaluation')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    
    # Add performance threshold line
    ax.axhline(y=performance_threshold, linestyle='--', alpha=0.7)
    ax.axvline(x=performance_threshold, linestyle='--', alpha=0.7)
    
    return fig
```

```yaml
# catalog.yml with conditional styling
model_evaluation_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/model_evaluation_${model_performance_level}.png
  purpose: presentation
  condition_param: model_performance_level
  style_params:
    alpha: 0.7
    marker: "o"
  format_kwargs:
    dpi: 200
    bbox_inches: tight
```

```yaml
# parameters.yml
model_performance_level: high_performance  # or 'low_performance', 'medium_performance'
```

### Integration with FigRegistry Configuration

```yaml
# conf/base/figregistry.yml
styles:
  high_performance:
    color: "#2E8B57"  # Green for good performance
    marker: "o"
    linewidth: 2.5
    alpha: 0.8
    
  medium_performance:
    color: "#FFB84D"  # Orange for medium performance  
    marker: "s"
    linewidth: 2.0
    alpha: 0.7
    
  low_performance:
    color: "#E74C3C"  # Red for poor performance
    marker: "^"
    linewidth: 1.5
    alpha: 0.6

defaults:
  figure:
    figsize: [10, 6]
    dpi: 150
  fallback_style:
    color: "#95A5A6"
    marker: "o"
    linewidth: 1.5
```

## Error Handling

### Exception Types

#### FigureDatasetError

Custom exception for FigureDataSet-specific errors with detailed error context.

```python
class FigureDatasetError(DatasetError):
    """Custom exception for FigureDataSet-specific errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
```

**Common Error Scenarios**:

```python
# Parameter validation errors
try:
    dataset = FigureDataSet(
        filepath="",  # Invalid empty filepath
        purpose="invalid_purpose"  # Invalid purpose
    )
except FigureDatasetError as e:
    print(f"Configuration error: {e}")
    print(f"Error details: {e.details}")

# Figure save errors  
try:
    dataset._save(invalid_figure_object)
except FigureDatasetError as e:
    print(f"Save operation failed: {e}")
    print(f"Error context: {e.details}")
```

### Parameter Validation

The dataset performs comprehensive parameter validation during initialization:

#### Filepath Validation

```python
# Valid filepath examples
FigureDataSet(filepath="data/08_reporting/figure.png")         # ✅ Valid
FigureDataSet(filepath="relative/path/figure.pdf")            # ✅ Valid  
FigureDataSet(filepath="/absolute/path/figure.svg")           # ✅ Valid

# Invalid filepath examples
FigureDataSet(filepath="")                                    # ❌ Empty string
FigureDataSet(filepath=None)                                  # ❌ None value
FigureDataSet(filepath=123)                                   # ❌ Non-string type
```

#### Purpose Validation

```python
# Valid purpose values
FigureDataSet(filepath="figure.png", purpose="exploratory")   # ✅ Valid
FigureDataSet(filepath="figure.png", purpose="presentation")  # ✅ Valid
FigureDataSet(filepath="figure.png", purpose="publication")   # ✅ Valid

# Invalid purpose values  
FigureDataSet(filepath="figure.png", purpose="invalid")       # ❌ Invalid value
FigureDataSet(filepath="figure.png", purpose="")              # ❌ Empty string
FigureDataSet(filepath="figure.png", purpose=None)            # ❌ None value
```

#### Condition Parameter Validation

```python
# Valid condition parameter examples
FigureDataSet(filepath="figure.png", condition_param="experiment_condition")  # ✅ Valid identifier
FigureDataSet(filepath="figure.png", condition_param="model_type")            # ✅ Valid identifier
FigureDataSet(filepath="figure.png", condition_param=None)                    # ✅ Valid (optional)

# Invalid condition parameter examples
FigureDataSet(filepath="figure.png", condition_param="")                      # ❌ Empty string
FigureDataSet(filepath="figure.png", condition_param="123invalid")            # ❌ Invalid identifier
FigureDataSet(filepath="figure.png", condition_param="param-with-dashes")     # ❌ Invalid identifier
```

#### Style Parameters Validation

```python
# Valid style parameters
FigureDataSet(
    filepath="figure.png",
    style_params={"color": "#2E86AB", "linewidth": 2.0}       # ✅ Valid dictionary
)

FigureDataSet(
    filepath="figure.png", 
    style_params=None                                          # ✅ Valid (optional)
)

# Invalid style parameters
FigureDataSet(
    filepath="figure.png",
    style_params="invalid"                                     # ❌ Non-dictionary type
)

FigureDataSet(
    filepath="figure.png",
    style_params={123: "value"}                                # ❌ Non-string keys
)
```

### Runtime Error Handling

#### Figure Object Validation

```python
def handle_save_operation():
    dataset = FigureDataSet(filepath="data/08_reporting/test.png")
    
    try:
        # Valid figure object
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        dataset._save(fig)  # ✅ Succeeds
        
    except FigureDatasetError as e:
        print(f"Save failed: {e}")
        
    try:
        # Invalid figure object
        dataset._save("not_a_figure")  # ❌ Raises FigureDatasetError
        
    except FigureDatasetError as e:
        print(f"Invalid figure object: {e}")
        print(f"Expected type: matplotlib.figure.Figure")
        print(f"Received type: {e.details.get('provided_type')}")
```

#### File System Error Handling

```python
def handle_filesystem_errors():
    dataset = FigureDataSet(
        filepath="/readonly/directory/figure.png"  # Read-only directory
    )
    
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        dataset._save(fig)
        
    except FigureDatasetError as e:
        if "Failed to create output directory" in str(e):
            print("Directory creation failed - check permissions")
            print(f"Target path: {e.details.get('save_path')}")
        elif "Matplotlib savefig operation failed" in str(e):
            print("File save operation failed") 
            print(f"Save arguments: {e.details.get('save_kwargs')}")
```

#### Configuration Bridge Errors

```python
def handle_configuration_errors():
    try:
        dataset = FigureDataSet(
            filepath="data/08_reporting/figure.png",
            condition_param="nonexistent_parameter"
        )
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        dataset._save(fig)
        
    except FigureDatasetError as e:
        if "Failed to resolve condition parameter" in str(e):
            print("Condition parameter not found in pipeline context")
            print("Available parameters:", list(e.details.get('available_params', [])))
```

### Error Recovery Strategies

#### Graceful Fallback

The dataset implements graceful fallback behavior when FigRegistry features are unavailable:

```python
# If FigRegistry is not installed or configuration fails
dataset = FigureDataSet(filepath="data/08_reporting/figure.png")

# Still saves figure with basic matplotlib functionality
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
dataset._save(fig)  # Succeeds without styling
```

#### Performance Degradation Handling

```python
# Monitor for performance issues
dataset = FigureDataSet(
    filepath="data/08_reporting/figure.png",
    enable_caching=True
)

# Check performance metrics
metrics = dataset.get_performance_metrics()
if metrics['save_operations']['average_ms'] > 100:
    print("Warning: Save operations taking longer than expected")
    print(f"Average save time: {metrics['save_operations']['average_ms']:.2f}ms")
    print("Consider optimizing figure complexity or output format")
```

## Kedro Integration Patterns

### Catalog Configuration

#### Basic Catalog Entry

```yaml
# catalog.yml
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/my_figure.png
```

#### Advanced Catalog Configuration

```yaml
# Complex configuration with all options
analysis_dashboard:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/dashboard_${environment}.png
  purpose: presentation
  condition_param: analysis_phase
  style_params:
    color: "#2E86AB"
    linewidth: 2.0
    alpha: 0.8
  format_kwargs:
    dpi: 200
    bbox_inches: tight
    facecolor: white
  versioned: true
  metadata:
    description: "Main analysis dashboard"
    created_by: "data_team"
    tags: ["dashboard", "presentation"]
```

#### Parameterized Catalog Entries

```yaml
# Use Kedro parameters in catalog configuration
results_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/${figure_name}_${experiment_id}.png
  purpose: ${output_purpose}
  condition_param: ${condition_parameter}
  format_kwargs:
    dpi: ${output_resolution}
    bbox_inches: tight
```

### Pipeline Node Integration

#### Simple Node Function

```python
# Standard node function returning matplotlib figure
def create_scatter_plot(data: pd.DataFrame) -> plt.Figure:
    """Create scatter plot - automatically styled by FigureDataSet."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['x'], data['y'])
    ax.set_title('Data Scatter Plot')
    return fig  # FigureDataSet handles styling and saving
```

#### Multiple Figure Outputs

```python
def create_analysis_figures(data: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
    """Create multiple figures for different analysis views."""
    
    # Time series plot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data['date'], data['value'])
    ax1.set_title('Time Series Analysis')
    
    # Distribution plot  
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(data['value'], bins=30)
    ax2.set_title('Value Distribution')
    
    return fig1, fig2
```

```yaml
# catalog.yml for multiple outputs
timeseries_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/timeseries.png
  purpose: presentation

distribution_plot:
  type: figregistry_kedro.datasets.FigureDataSet  
  filepath: data/08_reporting/distribution.png
  purpose: exploratory
```

```python
# pipeline.py
node(
    func=create_analysis_figures,
    inputs="processed_data",
    outputs=["timeseries_plot", "distribution_plot"],
    name="create_analysis_figures"
)
```

#### Conditional Figure Generation

```python
def create_conditional_plot(
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Optional[plt.Figure]:
    """Create figure only if certain conditions are met."""
    
    if parameters.get('create_plots', False):
        fig, ax = plt.subplots()
        ax.plot(data['x'], data['y'])
        ax.set_title(f"Plot for {parameters.get('experiment_name', 'Unknown')}")
        return fig
    
    return None  # FigureDataSet handles None gracefully
```

### Versioning Integration

#### Automatic Versioning

```yaml
# Enable automatic versioning
experiment_results:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/experiment_results.png
  versioned: true
  purpose: publication
```

Kedro automatically creates versioned files:
```
data/08_reporting/experiment_results.png/
├── 2024-01-15T10.30.00.000Z/
│   └── experiment_results.png
├── 2024-01-15T11.45.30.000Z/
│   └── experiment_results.png
└── 2024-01-15T14.20.15.000Z/
    └── experiment_results.png
```

#### Loading Specific Versions

```yaml
# Load specific version for comparison
baseline_comparison:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/baseline_results.png  
  versioned: true
  load_version: "2024-01-15T10.30.00.000Z"
```

#### Version-Aware Pipeline Nodes

```python
def compare_with_baseline(
    current_results: pd.DataFrame,
    baseline_figure: plt.Figure  # Loaded from specific version
) -> plt.Figure:
    """Create comparison plot with baseline version."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Current results
    ax1.plot(current_results['x'], current_results['y'])
    ax1.set_title('Current Results')
    
    # Display baseline figure (simplified)
    ax2.imshow(baseline_figure.canvas.get_renderer().buffer_rgba())
    ax2.set_title('Baseline Comparison')
    ax2.axis('off')
    
    return fig
```

### Environment-Specific Configuration

#### Multi-Environment Setup

```yaml
# conf/base/catalog.yml - Base configuration
reports_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/reports.png
  purpose: ${report_purpose}
  format_kwargs:
    dpi: ${report_dpi}
    bbox_inches: tight
```

```yaml
# conf/local/parameters.yml - Development environment
report_purpose: exploratory
report_dpi: 150

# conf/staging/parameters.yml - Testing environment  
report_purpose: presentation
report_dpi: 200

# conf/production/parameters.yml - Production environment
report_purpose: publication
report_dpi: 300
```

#### Environment-Specific Styling

```yaml
# conf/base/figregistry.yml
styles:
  development:
    color: "#A8E6CF"  # Light green for dev
    alpha: 0.7
    linewidth: 1.5
  
  staging:
    color: "#FFB6C1"  # Light pink for staging
    alpha: 0.8
    linewidth: 2.0
    
  production:
    color: "#1A1A1A"  # Black for production
    alpha: 1.0
    linewidth: 2.5
```

```yaml
# Environment-specific condition parameters
# conf/local/parameters.yml
environment_condition: development

# conf/staging/parameters.yml
environment_condition: staging

# conf/production/parameters.yml  
environment_condition: production
```

### Parallel Execution

#### Thread-Safe Operations

FigureDataSet is designed for safe use with Kedro's parallel runners:

```python
# kedro run --parallel  # Works safely with FigureDataSet

# Or with specific runner
# kedro run --runner=ThreadRunner
# kedro run --runner=ParallelRunner
```

#### Concurrent Figure Generation

```python
# Multiple nodes can safely generate figures concurrently
def create_plot_a(data_a: pd.DataFrame) -> plt.Figure:
    """Node A - runs concurrently with Node B."""
    fig, ax = plt.subplots()
    ax.plot(data_a['x'], data_a['y'])
    return fig

def create_plot_b(data_b: pd.DataFrame) -> plt.Figure:
    """Node B - runs concurrently with Node A."""
    fig, ax = plt.subplots()
    ax.scatter(data_b['x'], data_b['y'])
    return fig
```

```yaml
# Both datasets can be saved concurrently
plot_a:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/analysis_a.png
  enable_caching: true  # Thread-safe caching

plot_b:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/analysis_b.png  
  enable_caching: true  # Independent cache for each dataset
```

## Performance Optimization

### Performance Targets

The FigureDataSet maintains strict performance targets to ensure minimal impact on pipeline execution:

| Operation | Target Time | Typical Range | Performance Notes |
|-----------|-------------|---------------|-------------------|
| Style Resolution | <1ms | 0.1-0.8ms | With caching enabled |
| Figure Save | <100ms | 10-50ms | Depends on format and complexity |
| Configuration Merge | <10ms | 2-8ms | Per pipeline initialization |
| Memory Overhead | <5% | 1-3% | Compared to manual matplotlib |

### Optimization Strategies

#### Style Caching

```python
# Enable caching for repeated style resolution (default)
dataset = FigureDataSet(
    filepath="data/08_reporting/figure.png",
    enable_caching=True  # Recommended for production
)

# Disable caching for debugging only
debug_dataset = FigureDataSet(
    filepath="data/08_reporting/debug.png", 
    enable_caching=False  # Use only for troubleshooting
)
```

#### Format Optimization

```yaml
# Optimized for different use cases
exploratory_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/exploration.png
  purpose: exploratory
  format_kwargs:
    dpi: 100        # Lower DPI for faster generation
    optimize: true  # PNG optimization

presentation_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/presentation.png
  purpose: presentation
  format_kwargs:
    dpi: 200
    bbox_inches: tight

publication_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/publication.pdf
  purpose: publication
  format_kwargs:
    dpi: 300
    bbox_inches: tight
    pad_inches: 0.1
```

#### Memory Management

```python
def create_large_figure(large_dataset: pd.DataFrame) -> plt.Figure:
    """Efficiently handle large figures."""
    
    # Use matplotlib's memory-efficient practices
    plt.ioff()  # Turn off interactive mode
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Process data in chunks for very large datasets
    chunk_size = 10000
    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset.iloc[i:i+chunk_size]
        ax.plot(chunk['x'], chunk['y'], alpha=0.5)
    
    ax.set_title('Large Dataset Visualization')
    
    # FigureDataSet automatically handles memory cleanup
    return fig
```

### Performance Monitoring

#### Built-in Metrics

```python
# Access performance metrics
dataset = FigureDataSet(filepath="data/08_reporting/figure.png")

# After some operations...
metrics = dataset.get_performance_metrics()

print(f"Save operations: {metrics['save_operations']['count']}")
print(f"Average save time: {metrics['save_operations']['average_ms']:.2f}ms")
print(f"Cache hit rate: {metrics['cache_performance']['hit_rate']:.2%}")
```

#### Performance Monitoring in Pipelines

```python
def monitor_figure_performance():
    """Monitor figure generation performance across pipeline."""
    
    from figregistry_kedro.datasets import get_performance_summary
    
    # Get global performance metrics
    summary = get_performance_summary()
    
    print("Figure Generation Performance Summary:")
    print(f"Total saves: {len(summary['save_times'])}")
    print(f"Average save time: {sum(summary['save_times'])/len(summary['save_times']):.2f}ms")
    print(f"Cache hits: {summary['cache_hits']}")
    print(f"Cache misses: {summary['cache_misses']}")
    
    # Identify slow operations
    slow_operations = [t for t in summary['save_times'] if t > 100]
    if slow_operations:
        print(f"Warning: {len(slow_operations)} operations exceeded 100ms target")
```

#### Optimization Recommendations

Based on performance metrics, consider these optimizations:

1. **High Cache Miss Rate**: Review condition parameters and styling consistency
2. **Slow Save Operations**: Reduce figure complexity or lower DPI for exploratory plots  
3. **High Memory Usage**: Use vectorized operations and efficient data structures
4. **Concurrent Bottlenecks**: Verify thread-safe usage patterns

### Concurrent Execution Guidelines

#### Thread Safety

```python
# Safe concurrent usage
import concurrent.futures
import matplotlib.pyplot as plt

def create_figure_safely(data, output_path):
    """Thread-safe figure creation."""
    
    # Each thread gets its own dataset instance
    dataset = FigureDataSet(filepath=output_path)
    
    # matplotlib operations are thread-safe when using separate figure objects
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    
    # FigureDataSet handles thread-safe saving
    dataset._save(fig)
    plt.close(fig)  # Clean up memory

# Create multiple figures concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i, data_chunk in enumerate(data_chunks):
        future = executor.submit(
            create_figure_safely, 
            data_chunk, 
            f"data/08_reporting/chunk_{i}.png"
        )
        futures.append(future)
    
    # Wait for completion
    concurrent.futures.wait(futures)
```

#### Kedro Parallel Runner Compatibility

```bash
# All parallel runners supported
kedro run --runner=SequentialRunner    # Single-threaded (default)
kedro run --runner=ParallelRunner      # Multi-process  
kedro run --runner=ThreadRunner        # Multi-threaded
```

```python
# Pipeline configuration for parallel execution
def create_pipeline() -> Pipeline:
    return Pipeline([
        # These nodes can run in parallel
        node(
            func=create_sales_plot,
            inputs="sales_data",
            outputs="sales_figure",
            name="sales_plot",
            tags=["figures", "parallel"]
        ),
        node(
            func=create_revenue_plot, 
            inputs="revenue_data",
            outputs="revenue_figure",
            name="revenue_plot",
            tags=["figures", "parallel"]
        ),
        # This node depends on both figures
        node(
            func=create_summary_report,
            inputs=["sales_figure", "revenue_figure"],
            outputs="summary_report",
            name="summary_report"
        )
    ])
```

## Version Compatibility

### Kedro Version Support

| Kedro Version | Support Status | Notes |
|---------------|----------------|--------|
| 0.18.x | ✅ Full Support | Recommended minimum version |
| 0.19.x | ✅ Full Support | Latest stable features |
| 0.20.x | ⚠️ Under Development | Future compatibility planned |
| <0.18.0 | ❌ Not Supported | AbstractDataSet interface changes |

### Python Version Requirements

| Python Version | Support Status | Notes |
|----------------|----------------|--------|
| 3.10 | ✅ Full Support | Recommended minimum |
| 3.11 | ✅ Full Support | Optimal performance |
| 3.12 | ✅ Full Support | Latest stable release |
| <3.10 | ❌ Not Supported | Missing required language features |

### Dependency Compatibility

#### Required Dependencies

```toml
# Minimum required versions
kedro = ">=0.18.0,<0.20.0"
figregistry = ">=0.3.0"
matplotlib = ">=3.9.0" 
pydantic = ">=2.9.0"
pyyaml = ">=6.0.1"
```

#### Optional Dependencies

```toml
# Optional enhancements
kedro-viz = ">=4.0.0"          # Visualization in Kedro-Viz
kedro-mlflow = ">=0.11.0"      # MLflow experiment tracking
pytest-mock = ">=3.10.0"      # Testing framework
```

### Migration Guide

#### From Manual Figure Saving

```python
# Before: Manual matplotlib saving
def create_plot_old(data):
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    plt.savefig('data/08_reporting/plot.png', dpi=150, bbox_inches='tight')
    plt.close()

# After: FigureDataSet automation
def create_plot_new(data):
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y']) 
    return fig  # FigureDataSet handles save, styling, and cleanup
```

```yaml
# Add to catalog.yml
plot_output:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/plot.png
  purpose: presentation
  format_kwargs:
    dpi: 150
    bbox_inches: tight
```

#### From Basic Kedro Datasets

```yaml
# Before: Using PickleDataSet or other dataset types
my_plot:
  type: pickle.PickleDataSet
  filepath: data/06_models/plot_object.pkl

# After: Direct figure handling
my_plot:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/plot.png
  purpose: presentation
```

#### Updating Existing Pipelines

1. **Update Dependencies**:
   ```bash
   pip install figregistry-kedro
   ```

2. **Update Catalog Entries**:
   ```yaml
   # Replace matplotlib-related datasets
   figure_outputs:
     type: figregistry_kedro.datasets.FigureDataSet
     filepath: data/08_reporting/analysis.png
   ```

3. **Simplify Node Functions**:
   ```python
   # Remove manual savefig calls, return Figure objects directly
   def create_analysis_plot(data):
       fig, ax = plt.subplots()
       ax.plot(data['x'], data['y'])
       return fig  # Let FigureDataSet handle the rest
   ```

4. **Optional: Add FigRegistry Configuration**:
   ```yaml
   # conf/base/figregistry.yml
   styles:
     analysis_condition:
       color: "#2E86AB"
       linewidth: 2.0
   ```

### Compatibility Testing

The package includes comprehensive compatibility tests across supported versions:

```python
# Example compatibility test structure
@pytest.mark.parametrize("kedro_version", ["0.18.0", "0.18.14", "0.19.0"])
@pytest.mark.parametrize("python_version", ["3.10", "3.11", "3.12"])
def test_basic_functionality(kedro_version, python_version):
    """Test basic FigureDataSet functionality across versions."""
    dataset = FigureDataSet(filepath="test_figure.png")
    
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    
    # Should work across all supported versions
    dataset._save(fig)
    assert dataset._exists()
```

## Troubleshooting

### Common Issues

#### Import Errors

```python
# Problem: FigureDataSet not found
from figregistry_kedro.datasets import FigureDataSet
# ImportError: No module named 'figregistry_kedro'

# Solution: Install the package
# pip install figregistry-kedro
```

```python
# Problem: AbstractDataSet import error  
# kedro.io.core.DataSetError: Cannot import AbstractDataSet

# Solution: Check Kedro version compatibility
# pip install "kedro>=0.18.0,<0.20.0"
```

#### Configuration Issues

```yaml
# Problem: Invalid purpose value
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figure.png
  purpose: invalid_purpose  # ❌ Not a valid purpose

# Solution: Use valid purpose values
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figure.png  
  purpose: presentation  # ✅ Valid: exploratory, presentation, publication
```

```yaml
# Problem: Invalid condition parameter
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figure.png
  condition_param: "param-with-dashes"  # ❌ Invalid Python identifier

# Solution: Use valid Python identifiers
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figure.png
  condition_param: "param_with_underscores"  # ✅ Valid identifier
```

#### Runtime Errors

```python
# Problem: Figure object validation error
def create_invalid_plot():
    return "not_a_figure"  # ❌ Returns string instead of Figure

# Solution: Return matplotlib Figure object
def create_valid_plot():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig  # ✅ Returns Figure object
```

```python
# Problem: File permission errors
# FigureDatasetError: Failed to create output directory

# Solution: Check file system permissions
import os
output_dir = "data/08_reporting"
if not os.access(output_dir, os.W_OK):
    print(f"No write permission for {output_dir}")
    # Fix permissions or change output directory
```

#### Performance Issues

```python
# Problem: Slow save operations
# Warning: Figure save time 150.25ms exceeds performance target

# Solution 1: Reduce figure complexity
def optimized_plot(data):
    # Limit data points for large datasets
    if len(data) > 10000:
        data = data.sample(n=10000)
    
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    return fig

# Solution 2: Adjust format parameters
my_figure:
  type: figregistry_kedro.datasets.FigureDataSet
  filepath: data/08_reporting/figure.png
  format_kwargs:
    dpi: 150  # Reduce from 300 for faster saves
    optimize: false  # Disable PNG optimization for speed
```

### Debugging Techniques

#### Enable Debug Logging

```python
import logging

# Enable debug logging for FigureDataSet
logging.getLogger('figregistry_kedro.datasets').setLevel(logging.DEBUG)

# Enable debug logging for the full package
logging.getLogger('figregistry_kedro').setLevel(logging.DEBUG)
```

#### Performance Profiling

```python
def profile_figure_generation():
    """Profile figure generation performance."""
    import time
    
    dataset = FigureDataSet(
        filepath="data/08_reporting/profile_test.png",
        enable_caching=True
    )
    
    start_time = time.time()
    
    # Create test figure
    fig, ax = plt.subplots()
    ax.plot(range(1000), [x**2 for x in range(1000)])
    
    figure_creation_time = time.time() - start_time
    
    # Save with FigureDataSet
    save_start = time.time()
    dataset._save(fig)
    save_time = time.time() - save_start
    
    print(f"Figure creation: {figure_creation_time*1000:.2f}ms")
    print(f"Dataset save: {save_time*1000:.2f}ms") 
    print(f"Total time: {(figure_creation_time + save_time)*1000:.2f}ms")
    
    # Get detailed metrics
    metrics = dataset.get_performance_metrics()
    print(f"Cache hit rate: {metrics['cache_performance']['hit_rate']:.2%}")
```

#### Configuration Validation

```python
def validate_dataset_config():
    """Validate FigureDataSet configuration."""
    from figregistry_kedro.datasets import validate_figure_dataset_config
    
    config = {
        "type": "figregistry_kedro.datasets.FigureDataSet",
        "filepath": "data/08_reporting/test.png",
        "purpose": "presentation",
        "condition_param": "experiment_condition"
    }
    
    try:
        is_valid = validate_figure_dataset_config(config)
        print(f"Configuration is valid: {is_valid}")
    except Exception as e:
        print(f"Configuration validation failed: {e}")
```

#### Testing Dataset Functionality

```python
def test_dataset_functionality():
    """Test basic dataset operations."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, "test_figure.png")
        
        # Create dataset
        dataset = FigureDataSet(filepath=test_path)
        
        # Test description
        description = dataset._describe()
        print("Dataset description:", description)
        
        # Test save operation
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        try:
            dataset._save(fig)
            print("Save operation: SUCCESS")
        except Exception as e:
            print(f"Save operation: FAILED - {e}")
        
        # Test existence check
        exists = dataset._exists()
        print(f"File exists after save: {exists}")
        
        # Test load operation
        try:
            loaded_fig = dataset._load()
            print("Load operation: SUCCESS")
            print(f"Loaded figure type: {type(loaded_fig)}")
        except Exception as e:
            print(f"Load operation: FAILED - {e}")
```

### Support and Community

#### Getting Help

1. **Documentation**: Check the comprehensive documentation in `docs/`
2. **Examples**: Review example projects in `examples/`
3. **GitHub Issues**: Report bugs and request features
4. **Kedro Community**: Engage with the broader Kedro ecosystem

#### Contributing

1. **Bug Reports**: Include configuration, error messages, and environment details
2. **Feature Requests**: Describe use cases and expected behavior
3. **Pull Requests**: Follow the contribution guidelines in `CONTRIBUTING.md`

#### Best Practices

1. **Configuration Management**: Use environment-specific parameters
2. **Performance Monitoring**: Regularly check performance metrics
3. **Testing**: Include figure generation in your test suites
4. **Version Control**: Track both code and configuration changes
5. **Documentation**: Document custom styling and configuration patterns

---

*This API reference covers FigureDataSet v1.0.0+. For older versions or migration guidance, see the [Migration Guide](#migration-guide) section.*