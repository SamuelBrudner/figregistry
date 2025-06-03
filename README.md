# FigRegistry

A configuration-driven figure generation and management system for scientific visualization, supporting both Python (Matplotlib) and MATLAB.

## Features

- **Unified Configuration**: Define figure styles, layouts, and output paths in YAML
- **Cross-Platform**: Consistent behavior between Python and MATLAB
- **Condition-Based Styling**: Map experimental conditions to visual styles
- **Versioned Outputs**: Automatic timestamping and naming conventions
- **Flexible Outputs**: Support for different output purposes (exploratory, presentation, publication)

## Installation

### Core Package (Python)
```bash
pip install -e .
```

### Kedro Plugin
For automated figure styling and management in Kedro data pipelines:
```bash
pip install figregistry-kedro
```

### MATLAB
Add the `matlab` directory to your MATLAB path.

## Quick Start

1. Create a `figregistry.yaml` file in your project root:
   ```yaml
   figregistry_version: ">=0.3"
   style:
     rcparams:
       font.family: sans-serif
       font.size: 9
   layout:
     width_cm: 8.9
   condition_styles:
     control:
       color: "#A0A0A0"
       marker: "^"
   ```

2. In Python:
   ```python
   from figregistry import save_figure, get_style
   
   # Apply styles based on condition
   style = get_style('control')
   plt.plot(x, y, **style)
   save_figure('my_plot')
   ```

3. In MATLAB:
   ```matlab
   style = figregistry.getStyle('control');
   plot(x, y, 'Color', style.color, 'Marker', style.marker);
   figregistry.saveFigure('my_plot');
   ```

## Kedro Integration

The `figregistry-kedro` plugin extends FigRegistry's capabilities into Kedro machine learning pipelines, providing automated figure styling, versioning, and management within data science workflows.

### Plugin Components

The plugin provides three core components for seamless integration:

- **FigureDataSet**: Custom Kedro dataset that automatically applies FigRegistry styling to matplotlib figures during catalog save operations
- **FigRegistryHooks**: Lifecycle hooks for initializing FigRegistry configuration at pipeline startup and managing context throughout execution
- **FigRegistryConfigBridge**: Configuration translation layer that merges Kedro project configurations with FigRegistry settings

### Quick Start with Kedro

1. Install the plugin:
   ```bash
   pip install figregistry-kedro
   ```

2. Configure your Kedro data catalog (`conf/base/catalog.yml`):
   ```yaml
   model_performance_plot:
     type: figregistry_kedro.FigureDataSet
     filepath: data/08_reporting/model_performance.png
     condition_param: model_type
     style_params:
       purpose: publication
   ```

3. Create figures in your Kedro nodes:
   ```python
   def plot_model_performance(model_metrics: pd.DataFrame) -> plt.Figure:
       fig, ax = plt.subplots()
       ax.plot(model_metrics['epoch'], model_metrics['accuracy'])
       return fig  # FigRegistry styling applied automatically
   ```

4. Register hooks in your `settings.py`:
   ```python
   from figregistry_kedro.hooks import hooks

   HOOKS = hooks
   ```

### Configuration Integration

The plugin supports both standalone `figregistry.yaml` files and Kedro-integrated configuration in `conf/base/figregistry.yml`:

```yaml
figregistry_version: ">=0.3"
style:
  rcparams:
    font.family: sans-serif
    font.size: 9
layout:
  width_cm: 8.9
condition_styles:
  random_forest:
    color: "#2E8B57"
    marker: "o"
  xgboost:
    color: "#4682B4" 
    marker: "s"
```

### Plugin Documentation

For comprehensive documentation including advanced configuration, deployment patterns, and migration guides, see the [figregistry-kedro documentation](figregistry-kedro/docs/).

**Requirements**: `kedro>=0.18.0,<0.20.0` and `figregistry>=0.3.0`

## Documentation

See the [documentation](docs/index.md) for detailed usage and configuration options.

## Development

### Setup

1. Create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate figregistry
   pre-commit install
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

### Plugin Development

For developing the Kedro plugin:

1. Install additional dependencies:
   ```bash
   pip install kedro>=0.18.0,<0.20.0 kedro-datasets pytest-mock
   ```

2. Run plugin tests:
   ```bash
   cd figregistry-kedro
   pytest
   ```

### Testing

Run core tests:
```bash
pytest
```

Run all tests including plugin:
```bash
pytest --cov=figregistry --cov=figregistry_kedro
```

## License

MIT