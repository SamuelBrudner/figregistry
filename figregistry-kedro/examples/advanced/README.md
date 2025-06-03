# FigRegistry-Kedro Advanced Example

This example demonstrates the full capabilities of the `figregistry-kedro` plugin in a production-ready, multi-environment Kedro project. It showcases sophisticated configuration management, complex styling conditions, advanced pipeline implementations, and enterprise deployment patterns.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Multi-Environment Configuration](#multi-environment-configuration)
- [Setup and Installation](#setup-and-installation)
- [Pipeline Implementations](#pipeline-implementations)
- [FigRegistry Integration Features](#figregistry-integration-features)
- [Environment-Specific Configurations](#environment-specific-configurations)
- [Production Deployment](#production-deployment)
- [Advanced Usage Patterns](#advanced-usage-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

The advanced example demonstrates a realistic machine learning workflow with multiple data processing stages, sophisticated visualization requirements, and production-ready configuration management. This project illustrates:

- **Multi-environment configuration management**: Development, staging, and production configurations with proper overrides
- **FigRegistryConfigBridge functionality**: Seamless integration between Kedro's ConfigLoader and FigRegistry's YAML-based configuration
- **Advanced FigureDataSet configurations**: Complex styling parameters, condition-based styling, and versioning integration
- **Enterprise deployment patterns**: CI/CD integration, environment-specific settings, and scalable configuration management
- **Sophisticated pipeline implementations**: Multiple experimental scenarios with condition-based figure outputs

### Key Features Demonstrated

- **F-005**: Kedro FigureDataSet Integration with advanced parameters
- **F-006**: Lifecycle Hooks for seamless initialization and context management
- **F-007**: Configuration Bridge with multi-environment support
- **Environment-specific overrides**: Development, staging, and production configurations
- **Complex condition mapping**: Multiple experimental scenarios with automated styling
- **Versioning integration**: Kedro's native versioning with FigRegistry's timestamp management
- **Parallel execution support**: Thread-safe operation in concurrent pipeline scenarios

## Project Structure

```
figregistry-kedro/examples/advanced/
├── README.md                                    # This comprehensive documentation
├── .kedro.yml                                  # Kedro project configuration
├── pyproject.toml                              # Advanced dependency management
├── conf/                                       # Multi-environment configuration
│   ├── base/                                   # Base configuration layer
│   │   ├── catalog.yml                         # Data catalog with FigureDataSet
│   │   ├── parameters.yml                      # Pipeline parameters
│   │   └── figregistry.yml                     # Base FigRegistry configuration
│   ├── local/                                  # Local development overrides
│   │   └── figregistry.yml                     # Local FigRegistry overrides
│   ├── staging/                                # Staging environment
│   │   └── figregistry.yml                     # Staging-specific settings
│   └── production/                             # Production environment
│       └── figregistry.yml                     # Production-optimized settings
├── src/figregistry_kedro_advanced_example/     # Source code
│   ├── __init__.py
│   ├── settings.py                             # Kedro settings with FigRegistry hooks
│   └── pipelines/                              # Pipeline implementations
│       ├── training/                           # Model training pipeline
│       ├── inference/                          # Model inference pipeline
│       └── reporting/                          # Visualization and reporting
└── data/                                       # Data storage layers
    ├── 01_raw/                                 # Raw input data
    ├── 02_intermediate/                        # Processed intermediate data
    ├── 03_primary/                             # Cleaned primary datasets
    └── 08_reporting/                           # Generated figures and reports
```

## Multi-Environment Configuration

The advanced example implements a sophisticated configuration hierarchy that demonstrates the FigRegistryConfigBridge's ability to merge Kedro environment configurations with FigRegistry's YAML-based settings.

### Configuration Hierarchy

1. **Base Configuration** (`conf/base/figregistry.yml`): Core settings shared across all environments
2. **Environment Overrides**: Specific configurations for development, staging, and production
3. **Local Overrides** (`conf/local/figregistry.yml`): Developer-specific settings (git-ignored)

### Base Configuration Structure

```yaml
# conf/base/figregistry.yml
figregistry_version: ">=0.3.0"

# Output configuration for different environments
paths:
  exploratory: "data/08_reporting/exploratory"
  presentation: "data/08_reporting/presentation"
  publication: "data/08_reporting/publication"

# Base naming convention
naming:
  timestamp_format: "{ts}_{name}"
  include_timestamp: true

# Environment-agnostic style definitions
style:
  figure_size: [10, 6]
  dpi: 300
  font_family: "Arial"

# Experimental condition styles
condition_styles:
  baseline:
    color: "#1f77b4"
    marker: "o"
    linestyle: "-"
    label: "Baseline Model"
  
  experimental:
    color: "#ff7f0e"
    marker: "s"
    linestyle: "--"
    label: "Experimental Model"
  
  production:
    color: "#2ca02c"
    marker: "^"
    linestyle: "-."
    label: "Production Model"
  
  # Advanced condition patterns
  model_v*:
    color: "#d62728"
    marker: "D"
    linestyle: ":"
    label: "Model Version {version}"

# Layout settings for complex figures
layout:
  subplot_spacing: 0.3
  margin:
    top: 0.95
    bottom: 0.15
    left: 0.15
    right: 0.95

# Kedro-specific configuration section
kedro:
  # Map Kedro data layers to FigRegistry purposes
  layer_purpose_mapping:
    "08_reporting": "publication"
    "07_model_output": "presentation" 
    "exploratory": "exploratory"
  
  # Default condition parameter for pipeline context
  default_condition_param: "model_type"
```

### Environment-Specific Overrides

#### Development Environment (`conf/local/figregistry.yml`)

```yaml
# Local development overrides - faster iteration
figregistry_version: ">=0.3.0"

paths:
  exploratory: "data/08_reporting/dev"
  presentation: "data/08_reporting/dev"
  publication: "data/08_reporting/dev"

naming:
  timestamp_format: "dev_{ts}_{name}"
  include_timestamp: true

style:
  dpi: 150  # Lower DPI for faster rendering
  figure_size: [8, 5]  # Smaller figures for development

# Development-specific condition styles
condition_styles:
  debug:
    color: "#e377c2"
    marker: "x"
    linestyle: "-"
    label: "Debug Mode"
```

#### Production Environment (`conf/production/figregistry.yml`)

```yaml
# Production environment - optimized for quality and performance
figregistry_version: ">=0.3.0"

paths:
  exploratory: "/opt/app/outputs/exploratory"
  presentation: "/opt/app/outputs/presentation"
  publication: "/opt/app/outputs/publication"

naming:
  timestamp_format: "prod_{ts}_{name}"
  include_timestamp: true

style:
  dpi: 600  # High DPI for production quality
  figure_size: [12, 8]  # Larger figures for presentations
  font_family: "Times New Roman"  # Professional font

layout:
  margin:
    top: 0.98
    bottom: 0.12
    left: 0.12
    right: 0.98

# Production logging and metadata
metadata:
  include_git_hash: true
  include_environment_info: true
  include_dependency_versions: true
```

## Setup and Installation

### Prerequisites

- Python 3.10+ (3.10, 3.11, 3.12 supported)
- Kedro 0.18.0+ (compatible up to 0.19.x)
- figregistry 0.3.0+
- figregistry-kedro plugin

### Quick Start

1. **Clone and navigate to the advanced example**:
   ```bash
   cd figregistry-kedro/examples/advanced
   ```

2. **Create and activate a dedicated environment**:
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate figregistry-kedro-advanced
   
   # Or using pip in a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install the project in development mode**:
   ```bash
   pip install -e .
   ```

4. **Verify the installation**:
   ```bash
   kedro info
   kedro registry list  # Should show figregistry-kedro components
   ```

### Advanced Installation Options

#### Environment-Specific Setup

**Development Environment**:
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Verify hook registration
python -c "from figregistry_kedro.hooks import FigRegistryHooks; print('Hooks available')"
```

**Production Environment**:
```bash
# Production installation with minimal dependencies
pip install --no-dev .

# Set production environment
export KEDRO_ENV=production

# Verify production configuration
kedro catalog list | grep figregistry
```

#### Docker Setup

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

ENV KEDRO_ENV=production
CMD ["kedro", "run"]
```

## Pipeline Implementations

The advanced example includes three sophisticated pipelines that demonstrate various figregistry-kedro integration patterns.

### Training Pipeline

**Location**: `src/figregistry_kedro_advanced_example/pipelines/training/`

**Purpose**: Model training with comprehensive visualization of training metrics, validation curves, and model performance analysis.

**Key Features**:
- Multi-algorithm comparison with condition-based styling
- Training progress visualization with automatic styling
- Model performance metrics with publication-ready figures
- Hyperparameter optimization visualization

**Example Node Implementation**:
```python
def create_training_plots(
    training_history: Dict[str, List[float]], 
    model_type: str
) -> plt.Figure:
    """
    Create training progress visualization.
    
    The figure will be automatically styled by FigureDataSet
    based on the model_type condition parameter.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(training_history['loss'], label='Training Loss')
    ax1.plot(training_history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curves  
    ax2.plot(training_history['accuracy'], label='Training Accuracy')
    ax2.plot(training_history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig
```

**Catalog Configuration**:
```yaml
# Training pipeline figures in catalog.yml
training_loss_curves:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/training_loss_curves.png
  purpose: presentation
  condition_param: model_type
  style_params:
    title_prefix: "Training Progress:"
    include_timestamp: true
  versioned: true

model_comparison_plot:
  type: figregistry_kedro.FigureDataSet  
  filepath: data/08_reporting/model_comparison.png
  purpose: publication
  condition_param: comparison_type
  style_params:
    high_dpi: true
    include_metadata: true
  versioned: true
```

### Inference Pipeline

**Location**: `src/figregistry_kedro_advanced_example/pipelines/inference/`

**Purpose**: Model inference with prediction visualization, confidence intervals, and error analysis.

**Key Features**:
- Prediction vs. actual scatter plots
- Confidence interval visualization
- Error distribution analysis
- Feature importance plots

**Advanced Styling Example**:
```python
def create_prediction_analysis(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    model_version: str
) -> plt.Figure:
    """
    Create comprehensive prediction analysis visualization.
    
    Demonstrates complex styling with model version conditions.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prediction vs Actual scatter
    ax1.scatter(actuals, predictions, alpha=0.6)
    ax1.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Prediction Accuracy')
    
    # Residuals plot
    residuals = predictions - actuals
    ax2.scatter(predictions, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Analysis')
    
    # Error distribution
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    
    # QQ plot for normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Normal Q-Q Plot')
    
    plt.tight_layout()
    return fig
```

### Reporting Pipeline

**Location**: `src/figregistry_kedro_advanced_example/pipelines/reporting/`

**Purpose**: Final report generation with executive summaries, detailed analysis, and publication-ready figures.

**Key Features**:
- Executive dashboard creation
- Multi-model comparison reports
- Statistical significance testing visualization
- Automated report generation with styled figures

## FigRegistry Integration Features

### FigureDataSet Advanced Configuration

The advanced example demonstrates sophisticated FigureDataSet configurations that leverage all plugin capabilities:

```yaml
# Advanced FigureDataSet configurations
executive_dashboard:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/executive_dashboard.png
  purpose: presentation
  condition_param: dashboard_type
  style_params:
    # Complex styling parameters
    title_suffix: " - Executive Summary"
    watermark: "CONFIDENTIAL"
    include_timestamp: true
    custom_dpi: 300
    color_scheme: "corporate"
  versioned: true
  metadata:
    # Additional metadata for tracking
    created_by: "${oc.env:USER}"
    environment: "${oc.env:KEDRO_ENV,local}"
    model_version: "${model_version}"

# Multi-format output example  
publication_figure:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/publication_ready
  purpose: publication
  condition_param: figure_type
  save_args:
    # Multiple format outputs
    formats: ["png", "pdf", "svg"]
    dpi: 600
    bbox_inches: "tight"
    facecolor: "white"
    edgecolor: "none"
  style_params:
    publication_ready: true
    include_caption: true
    font_size_base: 12
  versioned: true
```

### Lifecycle Hooks Implementation

The advanced example includes comprehensive hook registration and configuration:

```python
# src/figregistry_kedro_advanced_example/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

# Hook registration with custom configuration
HOOKS = (FigRegistryHooks(),)

# Advanced hook configuration
FIGREGISTRY_HOOK_CONFIG = {
    "config_merge_strategy": "environment_override",
    "validation_level": "strict",
    "error_handling": "raise_on_config_error",
    "performance_logging": True,
    "context_isolation": True
}
```

### Configuration Bridge Functionality

The FigRegistryConfigBridge enables seamless integration between Kedro's environment management and FigRegistry's configuration system:

**Merge Strategy Example**:
```python
# Automatic configuration merging
# 1. Load base configuration from conf/base/figregistry.yml
# 2. Apply environment-specific overrides from conf/{env}/figregistry.yml  
# 3. Apply local overrides from conf/local/figregistry.yml
# 4. Merge with runtime parameters from Kedro context
# 5. Validate final configuration against FigRegistry schema

# The bridge handles complex merging scenarios:
base_config = {
    "style": {"dpi": 300, "figure_size": [10, 6]},
    "paths": {"publication": "data/08_reporting/pub"}
}

environment_override = {
    "style": {"dpi": 600},  # Override DPI for production
    "paths": {"publication": "/opt/app/outputs/pub"}  # Production path
}

# Result: Merged configuration with environment precedence
final_config = {
    "style": {"dpi": 600, "figure_size": [10, 6]},  # DPI overridden
    "paths": {"publication": "/opt/app/outputs/pub"}  # Path overridden
}
```

## Environment-Specific Configurations

### Development Environment Features

**Optimized for Speed and Iteration**:
- Lower DPI settings for faster rendering
- Simplified output paths for easy access
- Debug-specific styling conditions
- Comprehensive logging for troubleshooting

**Configuration Highlights**:
```yaml
# conf/local/figregistry.yml (development)
style:
  dpi: 150  # Fast rendering
  interactive_backend: true
  debug_annotations: true

paths:
  all_outputs: "data/08_reporting/dev"  # Single output location

condition_styles:
  debug:
    color: "#e377c2"
    alpha: 0.8
    debug_markers: true
```

### Staging Environment Features

**Production Validation Environment**:
- Production-like settings with monitoring
- Comprehensive validation and testing
- Performance benchmarking
- Error tracking and reporting

**Configuration Highlights**:
```yaml
# conf/staging/figregistry.yml
style:
  dpi: 450  # Near-production quality
  validation_mode: true

monitoring:
  performance_tracking: true
  error_reporting: true
  quality_metrics: true

paths:
  with_validation: true
  backup_enabled: true
```

### Production Environment Features

**Optimized for Quality and Reliability**:
- Maximum quality settings
- Robust error handling
- Comprehensive logging and monitoring
- Automated backup and versioning

**Configuration Highlights**:
```yaml
# conf/production/figregistry.yml
style:
  dpi: 600  # Maximum quality
  professional_formatting: true
  quality_assurance: true

reliability:
  backup_figures: true
  error_recovery: true
  performance_monitoring: true
  
security:
  watermark_enabled: true
  metadata_sanitization: true
```

## Production Deployment

### CI/CD Integration

The advanced example includes comprehensive CI/CD integration patterns suitable for enterprise deployment:

**GitHub Actions Workflow Example**:
```yaml
# .github/workflows/advanced-example.yml
name: Advanced Example CI/CD

on:
  push:
    branches: [main, develop]
    paths: ['examples/advanced/**']
  pull_request:
    paths: ['examples/advanced/**']

jobs:
  test-advanced-example:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
        kedro-version: [0.18.0, 0.19.0]
        environment: [local, staging]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        cd examples/advanced
        pip install kedro==${{ matrix.kedro-version }}
        pip install -e .
    
    - name: Run pipeline tests
      env:
        KEDRO_ENV: ${{ matrix.environment }}
      run: |
        cd examples/advanced
        kedro test
        kedro run --env ${{ matrix.environment }}
    
    - name: Validate figure outputs
      run: |
        cd examples/advanced
        python scripts/validate_figures.py
        
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      if: matrix.environment == 'staging'
      with:
        name: generated-figures-${{ matrix.python-version }}
        path: examples/advanced/data/08_reporting/
```

### Container Deployment

**Production Dockerfile**:
```dockerfile
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN pip install -e .

# Create non-root user
RUN groupadd -r kedro && useradd -r -g kedro kedro
RUN chown -R kedro:kedro /app
USER kedro

# Production configuration
ENV KEDRO_ENV=production
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD kedro catalog list > /dev/null || exit 1

CMD ["kedro", "run"]
```

### Infrastructure as Code

**Kubernetes Deployment Example**:
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: figregistry-kedro-advanced
spec:
  replicas: 1
  selector:
    matchLabels:
      app: figregistry-kedro-advanced
  template:
    metadata:
      labels:
        app: figregistry-kedro-advanced
    spec:
      containers:
      - name: kedro-app
        image: figregistry-kedro-advanced:latest
        env:
        - name: KEDRO_ENV
          value: "production"
        - name: FIGREGISTRY_OUTPUT_PATH
          value: "/mnt/outputs"
        volumeMounts:
        - name: output-storage
          mountPath: /mnt/outputs
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi" 
            cpu: "2"
      volumes:
      - name: output-storage
        persistentVolumeClaim:
          claimName: figregistry-outputs-pvc
```

## Advanced Usage Patterns

### Parallel Pipeline Execution

The advanced example demonstrates thread-safe operation in parallel execution scenarios:

```python
# Parallel-safe pipeline execution
def run_parallel_analysis():
    """
    Demonstrate parallel pipeline execution with figregistry-kedro.
    Each thread gets isolated FigRegistry context.
    """
    from concurrent.futures import ThreadPoolExecutor
    from kedro.runner import ThreadRunner
    
    # Configure thread-safe execution
    runner = ThreadRunner(max_workers=4)
    
    # Each pipeline run gets isolated configuration context
    pipeline_runs = [
        {"model_type": "baseline", "data_slice": "train"},
        {"model_type": "experimental", "data_slice": "validation"},
        {"model_type": "production", "data_slice": "test"}
    ]
    
    results = []
    for run_params in pipeline_runs:
        # Each run gets unique styling based on parameters
        result = runner.run(
            pipeline=create_analysis_pipeline(),
            catalog=load_catalog(),
            run_id=f"analysis_{run_params['model_type']}_{run_params['data_slice']}"
        )
        results.append(result)
    
    return results
```

### Dynamic Configuration Management

Advanced configuration patterns for runtime customization:

```python
# Dynamic configuration example
def configure_runtime_styling(experiment_config: Dict) -> None:
    """
    Demonstrate runtime configuration updates for experiment-specific styling.
    """
    from figregistry_kedro.config import FigRegistryConfigBridge
    
    # Load base configuration
    bridge = FigRegistryConfigBridge()
    base_config = bridge.load_config()
    
    # Apply experiment-specific overrides
    experiment_styles = {
        f"experiment_{experiment_config['id']}": {
            "color": experiment_config["primary_color"],
            "marker": experiment_config["marker_style"],
            "label": f"Experiment {experiment_config['name']}"
        }
    }
    
    # Merge configurations
    updated_config = bridge.merge_configurations(
        base_config,
        {"condition_styles": experiment_styles}
    )
    
    # Apply to current context
    bridge.apply_config(updated_config)
```

### Custom Hook Implementation

Advanced hook customization for specialized requirements:

```python
# Custom hook implementation
from figregistry_kedro.hooks import FigRegistryHooks
from kedro.framework.hooks import hook_impl

class CustomFigRegistryHooks(FigRegistryHooks):
    """Extended hooks with custom functionality."""
    
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """Custom initialization with performance monitoring."""
        # Call parent implementation
        super().before_pipeline_run(run_params, pipeline, catalog)
        
        # Add performance monitoring
        self._start_performance_monitoring()
        
        # Custom configuration validation
        self._validate_production_requirements(run_params)
    
    @hook_impl  
    def after_dataset_saved(self, dataset_name, data, filepath):
        """Custom post-save processing for figures."""
        if isinstance(data, plt.Figure):
            # Generate thumbnail
            self._create_thumbnail(filepath)
            
            # Log figure metrics
            self._log_figure_metrics(dataset_name, data)
            
            # Update figure registry
            self._update_figure_registry(dataset_name, filepath)
    
    def _validate_production_requirements(self, run_params):
        """Ensure production environment meets quality requirements."""
        if run_params.get("env") == "production":
            # Validate DPI settings
            # Check output permissions
            # Verify backup configurations
            pass
```

## Troubleshooting

### Common Issues and Solutions

#### Configuration Issues

**Problem**: Configuration merge conflicts between environments
```
ConfigurationError: Conflicting values in configuration merge
```

**Solution**:
```python
# Check configuration precedence
kedro catalog describe figregistry_config

# Validate specific environment configuration
export KEDRO_ENV=staging
python -c "from figregistry_kedro.config import FigRegistryConfigBridge; print(FigRegistryConfigBridge().validate_config())"

# Debug configuration merge process
export FIGREGISTRY_DEBUG=true
kedro run --pipeline preprocessing
```

#### Hook Registration Issues

**Problem**: FigRegistryHooks not being executed
```
WARNING: FigRegistry hooks not found in execution context
```

**Solution**:
```python
# Verify hook registration in settings.py
# src/figregistry_kedro_advanced_example/settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = (FigRegistryHooks(),)

# Test hook registration
kedro info | grep -i hook

# Verify hook execution
export KEDRO_LOGGING_CONFIG=logging.yml
kedro run --pipeline training | grep -i figregistry
```

#### FigureDataSet Issues

**Problem**: Figures not being styled correctly
```
StyleWarning: Condition 'model_v2' not found in condition_styles
```

**Solution**:
```yaml
# Add missing condition styles to figregistry.yml
condition_styles:
  # Existing conditions...
  
  # Add wildcard pattern for model versions
  "model_v*":
    color: "#d62728" 
    marker: "D"
    linestyle: ":"
    label: "Model Version {version}"
  
  # Or add specific condition
  model_v2:
    color: "#9467bd"
    marker: "v"
    linestyle: "-"
    label: "Model V2"
```

#### Performance Issues

**Problem**: Slow figure generation in production
```
PerformanceWarning: Figure save time exceeded 5 seconds
```

**Solution**:
```yaml
# Optimize production configuration
# conf/production/figregistry.yml
style:
  dpi: 300  # Reduce from 600 if needed
  optimize_rendering: true
  parallel_save: true

performance:
  max_save_time: 10  # seconds
  enable_caching: true
  compression_level: 6
```

### Debug Mode

Enable comprehensive debugging for troubleshooting:

```bash
# Enable debug mode
export FIGREGISTRY_DEBUG=true
export KEDRO_LOGGING_CONFIG=conf/logging.yml

# Run with verbose output
kedro run --pipeline training --verbose

# Check specific component logs
python -c "
from figregistry_kedro.datasets import FigureDataSet
from figregistry_kedro.hooks import FigRegistryHooks
print('Components loaded successfully')
"
```

### Performance Monitoring

Monitor performance in production environments:

```python
# Performance monitoring example
import time
from functools import wraps

def monitor_figure_performance(func):
    """Decorator to monitor figure generation performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        if duration > 5:  # 5 second threshold
            print(f"Performance warning: {func.__name__} took {duration:.2f}s")
        
        return result
    return wrapper

# Apply to figure generation functions
@monitor_figure_performance
def create_analysis_plot(data):
    # Your plotting code here
    pass
```

## Additional Resources

- **Core FigRegistry Documentation**: Main package documentation and API reference
- **Kedro Documentation**: Official Kedro framework documentation
- **Plugin API Reference**: Detailed API documentation for figregistry-kedro
- **Community Examples**: Additional examples and community contributions
- **Performance Optimization Guide**: Best practices for production deployments
- **Migration Guide**: Converting existing Kedro projects to use figregistry-kedro

## Support and Contributing

For issues, feature requests, or contributions:

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Community support and discussions
- **Pull Requests**: Contribute improvements and new features
- **Documentation**: Help improve documentation and examples

This advanced example demonstrates the full potential of the figregistry-kedro integration for enterprise-scale machine learning workflows with sophisticated visualization requirements.