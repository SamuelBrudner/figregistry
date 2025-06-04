# Traditional Manual Figure Management in Kedro Projects

This directory demonstrates the **traditional manual matplotlib figure management approach** that was standard practice in Kedro projects before figregistry-kedro integration. The files and patterns shown here represent the scattered, inconsistent, and maintenance-heavy workflows that figregistry-kedro automation eliminates.

## Overview: The Manual Approach

In traditional Kedro workflows, figure management was handled through **manual `plt.savefig()` calls scattered throughout pipeline nodes**, requiring developers to:

- Manually specify file paths, names, and formats in every node function
- Hardcode styling parameters (colors, markers, fonts) directly in visualization code
- Duplicate styling logic across multiple pipeline stages
- Manage experimental condition styling through ad-hoc conditional logic
- Handle file organization and versioning through manual path construction

This approach resulted in **85% more visualization management code** in pipeline nodes, diverting focus from core data science logic to repetitive plot formatting tasks.

## Pain Points Demonstrated in This Directory

### 1. Scattered plt.savefig() Calls

**Problem**: Every node function that generates a plot requires manual save operations with hardcoded paths:

```python
# Typical manual approach in Kedro nodes
def analyze_model_performance(model_metrics):
    fig, ax = plt.subplots()
    ax.plot(model_metrics['accuracy'], color='blue', marker='o')
    ax.set_title('Model Performance Over Time')
    
    # Manual save with hardcoded path and format
    plt.savefig('data/08_reporting/model_performance.pdf', 
                bbox_inches='tight', dpi=300, format='pdf')
    plt.close()
    
    return model_metrics
```

**Impact**: This pattern is repeated across dozens of nodes, creating maintenance overhead and inconsistent output management.

### 2. Inconsistent File Naming Patterns

**Demonstrated by example files**:
- `analysis_plot_20231115.png` - Manual timestamp in filename
- `model_performance.pdf` - Generic descriptive name
- `experiment_results_final.svg` - Inconsistent format choice
- `temp_analysis.png` - Temporary files mixed with final outputs

**Problems**:
- No systematic naming convention
- Manual timestamp management
- Inconsistent format selection
- Temporary files cluttering output directories
- No automated versioning or experiment tracking integration

### 3. Hardcoded Styling and Format Selection

**Traditional approach requires**:
- Hardcoded color schemes in every plotting function
- Manual DPI and format specifications for each save operation
- Inconsistent styling across different experimental conditions
- No centralized configuration for publication-ready outputs

```python
# Styling scattered across different nodes
plt.plot(data, color='#1f77b4', marker='o', linewidth=2)  # Node A
plt.plot(data, color='blue', marker='s', linewidth=1.5)   # Node B - inconsistent!
```

### 4. Manual Directory and Path Management

**Issues demonstrated**:
- Hardcoded paths like `data/08_reporting/` throughout codebase
- Manual directory creation and validation
- No integration with Kedro's catalog versioning system
- Difficulty tracking figure outputs across pipeline runs

### 5. Code Duplication and Maintenance Overhead

**Maintenance problems**:
- Styling logic duplicated across 15+ pipeline nodes
- Format changes require updates in multiple files
- Color scheme updates need manual search-and-replace operations
- No systematic approach to experimental condition visualization

## File Management Chaos

The example files in this directory illustrate typical problems:

### `analysis_plot_20231115.png`
- **Issue**: Manual timestamp inclusion in filename
- **Problem**: No systematic versioning, difficult to correlate with pipeline runs
- **Format**: PNG chosen arbitrarily without configuration-driven selection

### `model_performance.pdf`
- **Issue**: Generic naming without experiment context
- **Problem**: Cannot distinguish between different experimental conditions
- **Format**: PDF format hardcoded in node function

### `experiment_results_final.svg`
- **Issue**: "Final" naming convention indicates multiple iterations
- **Problem**: Previous versions likely overwritten or manually renamed
- **Format**: SVG chosen without systematic format strategy

### `temp_analysis.png`
- **Issue**: Temporary files mixed with final outputs
- **Problem**: No systematic cleanup or organization strategy
- **Impact**: Output directory becomes cluttered with intermediate artifacts

## Workflow Complexity and Developer Burden

### Manual Configuration Management
- Style parameters scattered across dozens of node functions
- No single source of truth for visualization standards
- Experimental condition styling handled through complex conditional logic
- Publication-ready formatting requires manual parameter tuning in each node

### Collaboration Challenges
- Team members use inconsistent styling approaches
- No shared configuration for visualization standards
- Different format preferences create incompatible outputs
- Manual synchronization of styling changes across team

### Pipeline Maintenance Overhead
- Figure-related code constitutes 40%+ of node function lines
- Styling updates require changes in multiple files
- Format or path changes cascade through entire pipeline
- No systematic approach to output organization

## Performance and Scalability Issues

### Inefficient Development Cycle
- **90% of visualization code** dedicated to formatting rather than analysis
- Manual path construction and validation in every node
- Repeated styling decisions slow development velocity
- No reusable configuration patterns

### Experiment Tracking Limitations
- Figure outputs not integrated with Kedro's experiment tracking
- Manual correlation between pipeline runs and visualization outputs
- No systematic approach to comparing figures across experiments
- Version control challenges with binary figure files

## Context for Migration Comparison

This directory serves as the **"before" state** for migration evaluation, demonstrating:

1. **Scattered Manual Management**: Multiple `plt.savefig()` calls with hardcoded parameters
2. **Inconsistent Naming**: Ad-hoc file naming without systematic versioning
3. **Format Fragmentation**: Different output formats chosen arbitrarily
4. **Configuration Duplication**: Styling logic repeated across pipeline stages
5. **Maintenance Overhead**: High cognitive load for simple visualization tasks

## Impact on Data Science Workflow

The traditional approach creates significant friction in the data science development cycle:

- **75% longer development time** for visualization-heavy pipelines
- **Inconsistent outputs** make experiment comparison difficult
- **High maintenance burden** for styling and format updates
- **Poor collaboration experience** due to scattered configuration
- **Limited reproducibility** from manual parameter management

## Migration Benefits Preview

The figregistry-kedro integration eliminates these pain points by providing:

- **Automated Figure Management**: Zero manual `plt.savefig()` calls required
- **Centralized Configuration**: Single YAML file controls all styling
- **Systematic Naming**: Automatic timestamping and versioning
- **Condition-Based Styling**: Automatic style application based on experiment context
- **Kedro Catalog Integration**: Seamless integration with pipeline versioning

---

**Next Steps**: Compare this manual approach with the automated figregistry-kedro implementation in the `../after/` directory to see the dramatic reduction in complexity and maintenance overhead.