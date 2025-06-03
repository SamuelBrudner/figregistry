# FigRegistry

A configuration-driven figure generation and management system for scientific visualization, supporting both Python (Matplotlib) and MATLAB.

## Features

- **Unified Configuration**: Define figure styles, layouts, and output paths in YAML
- **Cross-Platform**: Consistent behavior between Python and MATLAB
- **Condition-Based Styling**: Map experimental conditions to visual styles
- **Versioned Outputs**: Automatic timestamping and naming conventions
- **Flexible Outputs**: Support for different output purposes (exploratory, presentation, publication)

## Kedro Integration

The `figregistry-kedro` plugin extends FigRegistry's configuration-driven visualization capabilities into Kedro machine learning pipelines, enabling automated figure styling, versioning, and management within data science workflows.

### Key Plugin Components

#### FigureDataSet
Custom Kedro dataset that automatically applies FigRegistry styling to matplotlib figures during catalog save operations:
- Seamless integration with Kedro's data catalog and versioning
- Automatic condition-based styling based on pipeline parameters
- Elimination of manual `plt.savefig()` calls in node functions
- Support for all FigRegistry output formats (PNG, PDF, SVG)

#### FigRegistryHooks
Lifecycle hooks for configuration initialization and context management:
- Automatic FigRegistry configuration loading at pipeline startup
- Non-invasive integration that preserves Kedro's execution model
- Thread-safe operation for parallel pipeline execution
- Transparent context management requiring no code changes

#### FigRegistryConfigBridge
Configuration translation layer between Kedro and FigRegistry:
- Seamless merging of Kedro project configurations with traditional `figregistry.yaml` settings
- Support for environment-specific configuration overrides (dev, staging, production)
- Unified configuration source eliminating duplicate configuration management
- Automatic validation and type safety across both systems

### Kedro Plugin Quick Start

1. Install the plugin:
   ```bash
   pip install figregistry-kedro
   ```

2. Configure your Kedro catalog (`conf/base/catalog.yml`):
   ```yaml
   exploratory_plot:
     type: figregistry_kedro.FigureDataSet
     filepath: data/08_reporting/exploratory_plot
     condition_param: experimental_condition
     style_params:
       purpose: "exploratory"
   
   publication_figure:
     type: figregistry_kedro.FigureDataSet
     filepath: data/08_reporting/publication_figure
     condition_param: treatment_group
     style_params:
       purpose: "publication"
       dpi: 300
   ```

3. Create pipeline nodes that return matplotlib figures:
   ```python
   def create_scatter_plot(data: pd.DataFrame) -> plt.Figure:
       fig, ax = plt.subplots()
       ax.scatter(data['x'], data['y'])
       ax.set_title('Data Analysis Results')
       return fig
   ```

4. Register hooks in your project settings (`src/your_project/settings.py`):
   ```python
   from figregistry_kedro.hooks import hooks
   
   HOOKS = (hooks,)
   ```

### Compatibility
- **Kedro**: `>=0.18.0,<0.20.0`
- **FigRegistry**: `>=0.3.0`
- **Python**: `>=3.10`

For comprehensive documentation, examples, and advanced configuration options, see the [figregistry-kedro documentation](figregistry-kedro/docs/).

## Installation

### Python
```bash
pip install -e .
```

### Kedro Plugin
For automated figure styling and management in Kedro workflows:
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

3. For Kedro plugin development, also install:
   ```bash
   pip install "kedro>=0.18.0,<0.20.0"
   pip install -e "./figregistry-kedro[dev]"
   ```

### Testing

Run tests:
```bash
pytest
```

## License

MIT