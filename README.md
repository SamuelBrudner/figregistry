# FigRegistry

A configuration-driven figure generation and management system for scientific visualization, supporting both Python (Matplotlib) and MATLAB.

## Features

- **Unified Configuration**: Define figure styles, layouts, and output paths in YAML
- **Cross-Platform**: Consistent behavior between Python and MATLAB
- **Condition-Based Styling**: Map experimental conditions to visual styles
- **Versioned Outputs**: Automatic timestamping and naming conventions
- **Flexible Outputs**: Support for different output purposes (exploratory, presentation, publication)

## Installation

### Python
```bash
pip install -e .
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

### Testing

Run tests:
```bash
pytest
```

## License

MIT
