# Advanced FigRegistry-Kedro Integration Example

## Overview

This comprehensive example demonstrates sophisticated **multi-environment** FigRegistry integration with Kedro ML pipelines, showcasing enterprise-grade configuration management, advanced styling patterns, and production-ready deployment workflows. The example implements complex experimental scenarios with condition-based styling, sophisticated pipeline implementations, and comprehensive environment-specific configuration management.

## 🎯 Key Features Demonstrated

### Core Integration Features (F-005, F-006, F-007)

- **Advanced FigureDataSet Configuration**: Complex dataset configurations with sophisticated parameter resolution, style overrides, and condition-based styling
- **Lifecycle Hook Integration**: Automated FigRegistry initialization through Kedro hooks with context management
- **Configuration Bridge**: Seamless merging of Kedro and FigRegistry configurations with environment-specific overrides
- **Multi-Environment Support**: Development, staging, and production configuration patterns with optimized settings

### Enterprise-Grade Capabilities

- **Sophisticated Styling Patterns**: Hierarchical condition-based styling with wildcard matching and palette inheritance
- **Production Deployment Patterns**: CI/CD integration, environment management, and enterprise configuration practices
- **Advanced Pipeline Integration**: Complex experimental workflows with automated figure versioning and management
- **Performance Optimization**: Caching strategies, concurrent execution support, and optimized rendering configurations

## 📁 Project Structure

```
figregistry-kedro/examples/advanced/
├── .kedro.yml                          # Enterprise Kedro project configuration
├── pyproject.toml                      # Comprehensive dependencies and tooling
├── README.md                           # This documentation
├── conf/                               # Multi-environment configuration
│   ├── base/                          # Base configurations
│   │   ├── catalog.yml                # Advanced FigureDataSet catalog
│   │   ├── figregistry.yml            # Enterprise FigRegistry configuration
│   │   ├── parameters.yml             # Pipeline parameters
│   │   └── ...                        # Other Kedro configurations
│   ├── local/                         # Development environment overrides
│   │   ├── figregistry.yml            # Local development optimizations
│   │   └── ...                        # Development-specific settings
│   ├── staging/                       # Staging environment configuration
│   │   └── figregistry.yml            # Staging validation settings
│   └── production/                    # Production environment configuration
│       └── figregistry.yml            # Production deployment settings
├── src/figregistry_kedro_advanced_example/  # Source code
│   ├── __init__.py                    # Package initialization
│   ├── settings.py                    # Kedro project settings
│   ├── pipeline_registry.py          # Pipeline registration
│   └── pipelines/                     # Multiple sophisticated pipelines
│       ├── training/                  # ML model training workflows
│       ├── inference/                 # Production inference pipelines
│       └── reporting/                 # Publication and presentation outputs
├── data/                              # Data storage with organized structure
│   ├── 01_raw/                        # Raw data inputs
│   ├── 02_intermediate/               # Processed data
│   ├── 03_primary/                    # Modeling data
│   └── 08_reporting/                  # Output figures and reports
│       └── figures/                   # FigRegistry-managed figures
│           ├── training/              # Training pipeline outputs
│           ├── inference/             # Inference pipeline outputs
│           ├── reporting/             # Publication-ready figures
│           └── environments/          # Environment-specific outputs
└── tests/                             # Comprehensive test suite
    ├── test_integration.py            # End-to-end integration tests
    ├── test_pipelines.py             # Pipeline functionality tests
    └── test_config.py                # Configuration validation tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ with conda or virtualenv
- Git for version control
- Modern IDE (VS Code, PyCharm, etc.)

### Installation and Setup

1. **Clone and Navigate to Example**:
   ```bash
   git clone https://github.com/figregistry/figregistry-kedro.git
   cd figregistry-kedro/examples/advanced
   ```

2. **Create Isolated Environment**:
   ```bash
   # Option A: Conda (recommended)
   conda create -n figregistry-kedro-advanced python=3.10
   conda activate figregistry-kedro-advanced
   
   # Option B: Virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   # Development installation with all features
   pip install -e ".[all]"
   
   # Or minimal installation
   pip install -e ".[dev]"
   ```

4. **Verify Installation**:
   ```bash
   kedro info
   kedro registry list
   ```

### Running the Example

#### Basic Execution

```bash
# Run complete pipeline with FigRegistry integration
kedro run

# Run specific pipeline
kedro run --pipeline=training
kedro run --pipeline=inference  
kedro run --pipeline=reporting
```

#### Environment-Specific Execution

```bash
# Development environment (local configuration)
kedro run --env=local

# Staging environment validation
kedro run --env=staging

# Production deployment
kedro run --env=production
```

#### Advanced Pipeline Operations

```bash
# Run with specific parameters
kedro run --params="model_type:random_forest,confidence_level:high"

# Run with versioning enabled
kedro run --pipeline=training --env=production

# Parallel execution (where supported)
kedro run --runner=ParallelRunner --max-workers=4
```

## 🔧 Advanced Configuration

### FigRegistryConfigBridge (F-007)

The configuration bridge seamlessly merges Kedro's environment-specific configurations with FigRegistry settings:

#### Configuration Hierarchy

1. **Base Configuration** (`conf/base/figregistry.yml`):
   - Enterprise-grade styling patterns
   - Sophisticated condition mappings
   - Multi-purpose output configurations
   - Advanced palette definitions

2. **Environment Overrides** (`conf/{env}/figregistry.yml`):
   - **Local**: Development-optimized settings with debugging features
   - **Staging**: Validation and testing configurations
   - **Production**: High-quality, performance-optimized settings

#### Key Configuration Features

```yaml
# Example from conf/base/figregistry.yml
kedro_integration:
  dataset_mappings:
    training_figures:
      purpose: "exploration"
      condition_param: "model_type"
      style_params:
        default_condition: "training.model.baseline"
        environment_condition: "training.{environment}"
        
  environment_overrides:
    development:
      defaults.style.figure.dpi: 72      # Fast rendering
    production:
      defaults.style.figure.dpi: 300     # High quality
```

### Advanced FigureDataSet Configuration (F-005)

The data catalog demonstrates sophisticated FigureDataSet usage with complex parameter resolution:

#### Basic FigureDataSet Configuration

```yaml
training_model_performance_plots:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/figures/training/model_performance_comparison.png
  purpose: presentation
  condition_param: model_type
  style_params:
    figure.figsize: [14, 10]
    figure.dpi: 200
    axes.spines.top: false
    axes.spines.right: false
  save_args:
    dpi: 300
    bbox_inches: tight
    facecolor: white
  versioned: true
```

#### Advanced Parameter Resolution

```yaml
# Hierarchical condition resolution
hierarchical_condition_example:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/figures/advanced/hierarchical_condition_example.png
  purpose: presentation
  condition_param: experimental_scenarios.algorithm_comparison_study.conditions.baseline_comparison.condition_id
  style_params:
    figure.figsize: [14, 10]
    figure.dpi: 200
  versioned: true
```

#### Multi-Purpose Outputs

The catalog supports different output purposes with automatic styling:

- **Exploration** (`purpose: exploration`): Development and analysis workflows
- **Presentation** (`purpose: presentation`): Internal meetings and stakeholder demos  
- **Publication** (`purpose: publication`): External reports and academic papers

### Lifecycle Hooks Integration (F-006)

Automated FigRegistry initialization through Kedro hooks:

```python
# In settings.py
from figregistry_kedro.hooks import FigRegistryHooks

HOOKS = [FigRegistryHooks()]
```

The hooks provide:
- **Automatic Configuration**: FigRegistry initialization at pipeline startup
- **Context Management**: Shared configuration state across pipeline execution
- **Environment Integration**: Seamless integration with Kedro's environment system

## 🎨 Sophisticated Styling Patterns

### Condition-Based Styling (F-002)

#### Basic Experimental Conditions

```yaml
styles:
  "experiment.control":
    color: "palettes.extended_scientific.control"
    marker: "o"
    linestyle: "-"
    label: "Control Group"
    
  "experiment.treatment.group_a":
    color: "palettes.extended_scientific.treatment_a"
    marker: "^"
    linestyle: "-"
    label: "Treatment Group A"
```

#### Wildcard Pattern Matching

```yaml
styles:
  # All training conditions inherit base styling
  "training.*":
    linewidth: 2.0
    alpha: 0.85
    markeredgewidth: 1.0
    
  # Environment-specific wildcards
  "*.production":
    alpha: 1.0
    linestyle: "-"
    linewidth: 2.5
    
  "*.development":
    alpha: 0.7
    linestyle: ":"
    linewidth: 1.5
```

#### Statistical Significance Styling

```yaml
styles:
  "result.significant.p001":
    color: "palettes.extended_scientific.positive"
    marker: "***"
    linewidth: 3.0
    label: "p < 0.001 (***)"
    
  "result.not_significant":
    color: "palettes.extended_scientific.neutral"
    marker: "o"
    linestyle: ":"
    label: "Not Significant"
```

### Advanced Palette Management

#### Multi-Purpose Palettes

```yaml
palettes:
  # Scientific publication palette
  publication:
    primary: "#2E86AB"
    secondary: "#A23B72"
    accent: "#F18F01"
    background: "#FFFFFF"
    
  # High-contrast presentation palette
  presentation:
    primary: "#1B365D"
    secondary: "#F4A261"
    accent: "#E76F51"
    background: "#F8F9FA"
    
  # Colorblind-accessible palette
  accessible:
    primary: "#1f77b4"    # Protanopia safe
    secondary: "#ff7f0e"  # Deuteranopia safe  
    accent: "#2ca02c"     # Tritanopia safe
```

## 🔬 Pipeline Implementations

### Training Pipeline

**Purpose**: ML model development with automated performance visualization

**Key Figures Generated**:
- Data quality analysis plots
- Model performance comparisons  
- Hyperparameter optimization results
- Cross-validation analysis
- Feature importance visualizations
- Model convergence tracking

**Configuration Example**:
```python
# In pipeline nodes
def create_model_performance_plot(model_results, params):
    """Generate model performance comparison figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot model performance metrics
    for model_name, results in model_results.items():
        ax.plot(results['epochs'], results['loss'], label=model_name)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Model Training Performance')
    ax.legend()
    
    # FigRegistry handles styling and saving automatically through catalog
    return fig
```

### Inference Pipeline

**Purpose**: Production model deployment with quality monitoring

**Key Figures Generated**:
- Prediction quality analysis
- Batch vs real-time comparisons
- Model drift detection plots
- Performance metrics tracking
- A/B testing statistical analysis

**Advanced Features**:
- Real-time monitoring dashboards
- Confidence level visualization
- Performance threshold alerts
- Statistical significance testing

### Reporting Pipeline

**Purpose**: Publication and presentation-ready outputs

**Key Figures Generated**:
- Executive summary dashboards
- Technical deep-dive analysis  
- Performance benchmarking
- Business impact analysis
- Multi-algorithm comparison studies

**Output Formats**:
- **PNG**: Web and presentation use
- **PDF**: Publications and reports
- **SVG**: Interactive applications

## 🌍 Multi-Environment Deployment

### Development Environment (`--env=local`)

**Optimizations**:
- Fast rendering (72 DPI)
- Enhanced debugging features
- Hot-reload configuration support
- Verbose logging and validation
- Bright colors for easy identification

**Configuration Highlights**:
```yaml
# conf/local/figregistry.yml
defaults:
  figure:
    dpi: 100                    # Fast rendering
    edgecolor: "black"          # Visible borders for debugging
    
debugging:
  condition_resolution:
    enabled: true               # Debug condition resolution
    log_attempts: true          # Log resolution attempts
```

### Staging Environment (`--env=staging`)

**Features**:
- Validation-focused settings
- Balanced quality (200 DPI)
- Performance monitoring
- Quality assurance checks
- Pre-production testing

### Production Environment (`--env=production`)

**Optimizations**:
- Maximum quality (300 DPI)
- Performance optimization
- Minimal overhead
- Production monitoring
- Enterprise-grade outputs

**Configuration Highlights**:
```yaml
# conf/production/figregistry.yml
defaults:
  style:
    figure.dpi: 300             # High quality
    
performance:
  targets:
    save_overhead_percent: 2.0  # Minimal overhead
    
outputs:
  formats:
    pdf:
      dpi: 600                  # Publication quality
```

## 📊 Advanced Use Cases

### Experimental Design Support

#### A/B Testing Analysis
```yaml
# Catalog configuration for A/B testing
ab_testing_analysis:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/figures/experiments/ab_testing_results.png
  purpose: publication
  condition_param: experimental_group
  style_params:
    figure.figsize: [14, 8]
    errorbar.capsize: 6
  versioned: true
```

#### Multi-Algorithm Comparison
```yaml
# Complex condition resolution for algorithm studies
algorithm_comparison:
  type: figregistry_kedro.FigureDataSet
  condition_param: experimental_scenarios.algorithm_comparison_study.algorithm_type
  style_params:
    figure.figsize: [16, 12]
    axes.titlesize: 16
  versioned: true
```

### Publication-Ready Outputs

#### Journal Standards
```yaml
# Publication-quality configuration
journal_figures:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/figures/publication/journal_quality_figures.pdf
  purpose: publication
  style_params:
    figure.figsize: [8, 6]      # Standard column width
    font.family: ["Times New Roman", "serif"]
    font.size: 10
    axes.prop_cycle: cycler('color', ['black', 'gray', 'darkgray'])
  save_args:
    dpi: 600                    # High resolution for print
    format: pdf
```

### Performance Optimization

#### Concurrent Pipeline Execution
```yaml
# Thread-safe configuration
concurrent_pipeline_plots:
  type: figregistry_kedro.FigureDataSet
  style_params:
    figure.figsize: [10, 6]
  versioned: true
```

#### Cache-Optimized Configurations
```yaml
# Performance-optimized settings
kedro:
  performance:
    enable_caching: true
    cache_ttl_seconds: 300
    enable_concurrent_access: true
```

## 🔍 Debugging and Development

### Configuration Validation

```bash
# Validate configuration merging
kedro config-loader list

# Debug condition resolution
kedro run --pipeline=training --env=local --params="debug_mode:true"
```

### Hot Reload Development

The local environment supports configuration hot reload for rapid iteration:

1. Modify `conf/local/figregistry.yml`
2. Re-run pipeline nodes
3. See immediate styling changes
4. Enhanced logging shows configuration changes

### Testing and Validation

```bash
# Run integration tests
pytest tests/test_integration.py -v

# Test configuration bridge
pytest tests/test_config.py::test_config_bridge -v

# Validate all environments
kedro run --env=local --pipeline=training
kedro run --env=staging --pipeline=training  
kedro run --env=production --pipeline=training
```

## 📈 Performance Considerations

### Optimization Strategies

1. **Style Caching**: Configured style lookup caching with TTL
2. **Concurrent Access**: Thread-safe operations for parallel execution
3. **Format Optimization**: Environment-specific output format selection
4. **Memory Management**: Efficient figure object handling

### Performance Targets

| Environment | DPI | Overhead Target | Cache Strategy |
|-------------|-----|-----------------|----------------|
| Development | 72-150 | <10% | Disabled for hot reload |
| Staging | 200 | <5% | Moderate caching |
| Production | 300 | <2% | Aggressive caching |

### Monitoring and Metrics

```yaml
# Performance monitoring configuration
performance:
  monitoring:
    enabled: true
    log_slow_operations: true
    timing_threshold_ms: 10
    
  targets:
    config_load_ms: 50
    style_lookup_ms: 1.0
    save_overhead_percent: 2.0
```

## 🚀 Production Deployment

### CI/CD Integration

#### GitHub Actions Example
```yaml
name: Advanced Example CI/CD
on: [push, pull_request]

jobs:
  test-multi-environment:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
        kedro-version: [0.18.0, 0.19.0]
        environment: [local, staging, production]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        cd examples/advanced
        pip install -e ".[dev]"
        
    - name: Test environment
      run: |
        cd examples/advanced
        kedro run --env=${{ matrix.environment }} --pipeline=training
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY examples/advanced/ .

RUN pip install -e ".[all]"

# Production environment
ENV KEDRO_ENV=production

CMD ["kedro", "run"]
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: figregistry-kedro-advanced
spec:
  replicas: 3
  selector:
    matchLabels:
      app: figregistry-kedro-advanced
  template:
    metadata:
      labels:
        app: figregistry-kedro-advanced
    spec:
      containers:
      - name: pipeline
        image: figregistry-kedro-advanced:latest
        env:
        - name: KEDRO_ENV
          value: "production"
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Configuration Bridge Issues

**Problem**: Configuration not merging correctly
```bash
# Debug configuration loading
kedro config-loader list --env=local

# Validate merge results
python -c "
from kedro.config import OmegaConfigLoader
loader = OmegaConfigLoader('conf')
config = loader['figregistry']
print(config)
"
```

**Solution**: Check configuration hierarchy and validation

#### Condition Resolution Problems

**Problem**: Conditions not resolving to expected styles
```python
# Add debugging to pipeline parameters
parameters = {
    "debug_mode": True,
    "condition_resolution_debug": True,
    "model_type": "random_forest"
}
```

**Solution**: Enable verbose condition resolution logging

#### Performance Issues

**Problem**: Slow figure saving operations
```yaml
# Optimize configuration
kedro:
  performance:
    enable_caching: true
    style_cache_size: 1000
    optimize_rendering: true
```

**Solution**: Enable caching and performance optimization

### Debug Mode Features

Enable comprehensive debugging:
```yaml
# conf/local/figregistry.yml
debugging:
  condition_resolution:
    enabled: true
    log_attempts: true
    log_fallbacks: true
    
  style_application:
    enabled: true
    log_style_lookup: true
    log_style_application: true
```

## 📚 Additional Resources

### Documentation Links

- [FigRegistry Core Documentation](https://figregistry.readthedocs.io/)
- [Kedro Framework Documentation](https://kedro.readthedocs.io/)
- [FigRegistry-Kedro Plugin API Reference](../../../docs/api/)

### Example Variations

- [Basic Example](../basic/): Simple single-environment setup
- [Migration Example](../migration/): Converting existing Kedro projects

### Community and Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/figregistry/figregistry-kedro/issues)
- **Discussions**: [Community Q&A and examples](https://github.com/figregistry/figregistry-kedro/discussions)
- **Stack Overflow**: Tag questions with `figregistry` and `kedro`

## 📄 License

This example is part of the figregistry-kedro project and is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

## 🤝 Contributing

Contributions to improve this advanced example are welcome! Please see our [Contributing Guidelines](../../CONTRIBUTING.md) for details on:

- Adding new pipeline patterns
- Improving configuration examples
- Enhancing documentation
- Adding test coverage
- Performance optimizations

---

*This advanced example demonstrates the full capabilities of FigRegistry-Kedro integration in enterprise environments. For simpler use cases, see the [basic example](../basic/). For migration guidance, see the [migration example](../migration/).*