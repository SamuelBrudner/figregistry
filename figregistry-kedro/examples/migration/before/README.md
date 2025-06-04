# Traditional Manual Figure Management in Kedro

This example demonstrates the traditional approach to matplotlib figure management in Kedro projects **before** integrating `figregistry-kedro`. This serves as the "before" state in our migration comparison, showcasing the scattered manual processes, code duplication, and maintenance overhead that `figregistry-kedro` eliminates.

## Overview: The Manual Approach Problems

This traditional Kedro project exemplifies the common pain points that data science teams face when managing matplotlib visualizations in pipeline workflows:

### 🚨 Core Problems Demonstrated

1. **Scattered `plt.savefig()` Calls**: Manual save operations distributed throughout pipeline nodes
2. **Hardcoded Styling**: Inconsistent visual formatting copied across multiple functions  
3. **Manual File Management**: Fragmented path construction and naming conventions
4. **Code Duplication**: Repeated styling logic across different pipeline stages
5. **Maintenance Overhead**: Style changes require hunting through multiple files
6. **Inconsistent Outputs**: No systematic approach to experimental condition visualization

### 📁 Project Structure

```
kedro-manual-example/
├── src/kedro_manual_example/
│   ├── __init__.py              # Basic package initialization
│   ├── settings.py              # Standard Kedro settings (no FigRegistry)
│   ├── pipeline_registry.py     # Traditional pipeline registration
│   └── nodes.py                 # Nodes with manual figure management
├── conf/base/
│   ├── catalog.yml             # Basic dataset definitions
│   └── parameters.yml          # Pipeline parameters (no style config)
├── data/                       # Standard Kedro data structure
├── .kedro.yml                  # Project configuration
├── pyproject.toml              # Dependencies (no figregistry-kedro)
└── README.md                   # This documentation
```

## 🔍 Manual Figure Management Anti-Patterns

### Problem 1: Scattered Save Operations

In the traditional approach, each pipeline node that generates a plot must handle its own saving logic:

```python
def create_scatter_plot(df: pd.DataFrame) -> None:
    """Node that creates scatter plot with manual save."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hardcoded styling repeated in every node
    ax.scatter(df['x'], df['y'], 
              color='blue',           # Hardcoded color
              alpha=0.7,             # Hardcoded transparency
              s=50)                  # Hardcoded size
    
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_title('Data Scatter Plot')
    
    # Manual save with hardcoded path
    output_path = 'data/08_reporting/scatter_plot_20241204_143022.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### Problem 2: Duplicated Styling Logic

Every visualization node contains similar styling code with subtle inconsistencies:

```python
def create_line_chart(df: pd.DataFrame) -> None:
    """Another node with duplicated styling."""
    fig, ax = plt.subplots(figsize=(12, 8))  # Different figure size!
    
    # Similar but slightly different styling
    ax.plot(df['time'], df['value'],
           color='red',              # Different color scheme
           linewidth=2,             # Manual line width
           alpha=0.8)               # Different alpha value
    
    # Repeated axis configuration
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Analysis')
    
    # Different save path pattern
    output_path = 'data/08_reporting/timeseries_chart_manual.png'
    plt.savefig(output_path, dpi=150)  # Different DPI!
    plt.close()
```

### Problem 3: Manual File Path Management

Each node constructs output paths manually, leading to inconsistencies:

```python
def create_histogram(df: pd.DataFrame) -> None:
    """Node with manual path construction."""
    fig, ax = plt.subplots()
    
    ax.hist(df['values'], bins=30, 
           color='green',            # Yet another color scheme
           alpha=0.6,               # Another alpha value
           edgecolor='black')
    
    # Manual timestamp and path construction
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'histogram_{timestamp}.png'
    output_dir = 'data/08_reporting/plots'
    
    # Directory creation logic in every node
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=300, format='png')
    plt.close()
```

### Problem 4: No Systematic Condition Handling

Experimental conditions are handled inconsistently across nodes:

```python
def create_condition_plot(df: pd.DataFrame, condition: str) -> None:
    """Node with manual condition styling."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Manual condition-to-style mapping (repeated everywhere)
    if condition == 'experimental':
        color = 'red'
        marker = 'o'
        linestyle = '-'
    elif condition == 'control':
        color = 'blue'
        marker = 's'
        linestyle = '--'
    else:
        # Default case handling
        color = 'gray'
        marker = '.'
        linestyle = ':'
    
    ax.plot(df['x'], df['y'], 
           color=color, marker=marker, linestyle=linestyle)
    
    # Manual path with condition
    output_path = f'data/08_reporting/condition_{condition}_plot.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
```

## 🚧 Maintenance Overhead Issues

### Style Update Nightmare

When you need to change the default color scheme:

1. **❌ Hunt through multiple files** to find all hardcoded color values
2. **❌ Update each node individually** with new styling parameters  
3. **❌ Test each pipeline node** to ensure changes work correctly
4. **❌ Risk introducing inconsistencies** between different visualizations
5. **❌ Coordinate team updates** when multiple developers modify different nodes

### File Organization Chaos

- **Inconsistent naming**: `scatter_plot_20241204.png`, `timeseries_chart_manual.png`, `histogram_20241204_143022.png`
- **Mixed directory structures**: Some files in `data/08_reporting/`, others in `data/08_reporting/plots/`
- **Manual versioning**: Timestamp patterns vary across different nodes
- **No systematic organization**: Experimental conditions scattered across filenames

### Configuration Management Problems

- **No central configuration**: Styling parameters hardcoded in each function
- **Environment inconsistencies**: No support for different styling in dev/staging/prod
- **Parameter drift**: Subtle differences accumulate over time
- **Difficult testing**: Hard to validate styling consistency across nodes

## 🏃‍♂️ Running This Traditional Example

### Prerequisites

```bash
# Install basic dependencies (no figregistry-kedro)
pip install kedro>=0.18.0 matplotlib pandas numpy
```

### Setup and Execution

```bash
# Clone and navigate to the project
cd figregistry-kedro/examples/migration/before/

# Install project dependencies
pip install -e .

# Run the pipeline with manual figure management
kedro run

# View scattered output files
ls -la data/08_reporting/
# Output shows inconsistent naming and organization:
# scatter_plot_20241204_143022.png
# timeseries_chart_manual.png  
# plots/histogram_20241204_143055.png
# condition_experimental_plot.png
```

### Manual Configuration

Since there's no automated configuration system, styling changes require:

```bash
# 1. Open each node file individually
vim src/kedro_manual_example/nodes.py

# 2. Find and replace hardcoded values manually
# Search for: color='blue'
# Replace with: color='navy'

# 3. Repeat for every styling parameter across all nodes
# 4. Test each node individually to verify changes
```

## 📊 Impact Analysis: Before vs After

### Code Metrics - Traditional Approach

| Metric | Value | Issue |
|--------|-------|-------|
| **Lines of styling code** | ~15-25 per node | Massive duplication |
| **Manual save calls** | 1 per visualization node | Scattered throughout codebase |
| **Hardcoded parameters** | 8-12 per node | No central configuration |
| **File path constructions** | Manual in each node | Inconsistent patterns |
| **Style update effort** | 30+ minutes per change | Hunt-and-replace across files |
| **Testing complexity** | Test every node individually | No systematic validation |

### Developer Experience Issues

- **❌ Context switching**: Jump between multiple files for simple style changes
- **❌ Copy-paste errors**: Introduce bugs when duplicating styling logic
- **❌ Merge conflicts**: Team members modifying the same hardcoded values
- **❌ Documentation burden**: Must document styling in each node
- **❌ Onboarding complexity**: New developers must learn each node's conventions

### Operational Problems

- **❌ Inconsistent outputs**: Visualizations look different across pipeline stages
- **❌ Manual file cleanup**: No systematic organization of output files
- **❌ Version tracking**: Difficult to associate figures with experimental runs
- **❌ Environment drift**: Styling differs between development and production
- **❌ Scalability issues**: Adding new visualizations requires extensive boilerplate

## 🎯 What FigRegistry-Kedro Eliminates

This traditional approach demonstrates exactly why `figregistry-kedro` was created. The automated solution addresses every pain point shown here:

### ✅ Automated Figure Management
- **Single configuration file** instead of scattered hardcoded values
- **Automatic save operations** through `FigureDataSet` integration
- **Systematic file organization** with configurable naming patterns
- **Environment-specific styling** through Kedro's configuration system

### ✅ Zero Code Duplication  
- **Condition-based styling** automatically applied based on parameters
- **Centralized style definitions** in `figregistry.yaml`
- **Consistent application** across all pipeline visualizations
- **DRY principle compliance** eliminates repeated styling logic

### ✅ Effortless Maintenance
- **One-line style changes** update all figures simultaneously
- **Version-controlled configuration** with Git-friendly YAML
- **Team collaboration** through shared configuration standards
- **Automated testing** validates styling consistency

## 🔗 Next Steps: Migration Path

After experiencing these traditional approach limitations, see the "after" example to understand how `figregistry-kedro` transforms this workflow:

1. **[View the After Example](../after/README.md)**: See the same pipeline with automated figure management
2. **[Migration Guide](../README.md)**: Step-by-step conversion instructions
3. **[FigRegistry Documentation](../../docs/README.md)**: Complete integration guide

---

**Remember**: This example intentionally demonstrates problematic patterns to highlight the value of automated figure management. In production, `figregistry-kedro` eliminates these issues while maintaining the same pipeline logic and improving consistency, maintainability, and developer experience.