# Traditional Manual Figure Management in Kedro

## Overview

This example demonstrates the **traditional approach** to figure management in Kedro projects, showcasing the manual matplotlib workflows that existed before the introduction of `figregistry-kedro`. This serves as the "before" state in our migration example, highlighting the pain points and maintenance overhead that automated figure management eliminates.

> ⚠️ **Warning**: This example intentionally shows problematic practices that have been superseded by the automated `figregistry-kedro` approach. These patterns should **not** be used in production code.

## Problems Demonstrated

### 1. Scattered `plt.savefig()` Calls

In traditional Kedro projects, figure saving logic is scattered throughout individual node functions, leading to:

- **Code duplication**: Every node that creates figures includes manual save logic
- **Inconsistent naming**: Each developer implements their own file naming conventions
- **Hardcoded paths**: File paths are embedded directly in node functions
- **No centralized configuration**: Styling and output settings distributed across multiple files

```python
# Typical problematic pattern found in nodes.py
def create_data_distribution_plot(data: pd.DataFrame) -> None:
    """Create distribution plot with manual figure management."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hardcoded styling scattered throughout function
    ax.hist(data['value'], bins=30, color='blue', alpha=0.7)
    ax.set_title('Data Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Manual file management with hardcoded paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_distribution_{timestamp}.png"
    output_path = os.path.join("data", "08_reporting", filename)
    
    # Manual directory creation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Scattered plt.savefig() calls throughout codebase
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 2. Hardcoded Styling Without Systematic Management

Each visualization function contains embedded styling code that:

- **Cannot be centrally managed**: Style changes require editing multiple files
- **Lacks consistency**: Different nodes use different color schemes and styling
- **No condition-based logic**: Cannot automatically adapt styling based on experimental conditions
- **Difficult to maintain**: Style updates require touching every visualization function

```python
# Problematic styling patterns
def create_comparison_plot(baseline: pd.DataFrame, experimental: pd.DataFrame) -> None:
    """Comparison plot with hardcoded styling."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hardcoded colors and styling - repeated across functions
    ax.scatter(baseline['x'], baseline['y'], 
              color='#1f77b4', alpha=0.6, s=50, label='Baseline')
    ax.scatter(experimental['x'], experimental['y'], 
              color='#ff7f0e', alpha=0.6, s=50, label='Experimental')
    
    # Repeated styling code in every function
    ax.set_title('Baseline vs Experimental Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Variable', fontsize=14)
    ax.set_ylabel('Y Variable', fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Manual file naming and saving
    plt.savefig("data/08_reporting/comparison_analysis.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
```

### 3. Manual File Path Management and Configuration

Traditional approaches require manual management of:

- **File naming conventions**: Inconsistent across different developers and projects
- **Directory structures**: Manual creation and path management
- **Versioning**: No systematic approach to figure versioning
- **Output formats**: Hardcoded format selection in each function

```python
# Manual path management anti-patterns
def save_analysis_figure(fig: plt.Figure, analysis_type: str, condition: str) -> None:
    """Manual figure saving with problematic path management."""
    
    # Hardcoded base directory
    base_dir = "data/08_reporting/figures"
    
    # Manual timestamp generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inconsistent naming conventions
    if condition == "high_dose":
        filename = f"{analysis_type}_highdose_{timestamp}.png"
    elif condition == "low_dose":
        filename = f"{analysis_type}_lowdose_{timestamp}.png"
    else:
        filename = f"{analysis_type}_control_{timestamp}.png"
    
    # Manual directory creation
    full_path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Manual format and quality settings
    fig.savefig(full_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
```

### 4. Code Duplication and Maintenance Overhead

The traditional approach leads to significant maintenance burden:

```python
# Repeated styling code across multiple functions
COMMON_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Repeated in multiple files
COMMON_FIGURE_SIZE = (10, 6)  # Duplicated constants
COMMON_DPI = 300  # Scattered throughout codebase

def style_scatter_plot(ax, title):
    """Repeated styling function - duplicated across modules."""
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    # ... more repeated styling code

def style_line_plot(ax, title):
    """Similar styling function - slight variations cause inconsistency."""
    ax.set_title(title, fontsize=16, fontweight='bold')  # Different fontsize!
    ax.grid(True, alpha=0.2)  # Different alpha!
    ax.legend(fontsize=12)  # Different fontsize!
    # ... inconsistent styling patterns
```

### 5. No Systematic Condition-Based Styling

Traditional approaches cannot systematically handle experimental conditions:

```python
def plot_experimental_results(data: pd.DataFrame, condition: str) -> None:
    """Manual condition handling without systematic style management."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Manual condition-based styling - error-prone and unmaintainable
    if condition == "control":
        color = '#808080'
        marker = 'o'
        linestyle = '-'
    elif condition == "treatment_a":
        color = '#ff4444'
        marker = 's'
        linestyle = '--'
    elif condition == "treatment_b":
        color = '#4444ff'
        marker = '^'
        linestyle = '-.'
    else:
        # Fallback styling - often forgotten or inconsistent
        color = 'black'
        marker = '.'
        linestyle = ':'
    
    # Plot with manually resolved styling
    ax.plot(data['x'], data['y'], color=color, marker=marker, 
            linestyle=linestyle, label=condition)
    
    # Manual saving with condition-specific naming
    output_file = f"results_{condition}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(f"data/08_reporting/{output_file}")
    plt.close()
```

## Project Structure

```
kedro_manual_example/
├── README.md                     # This documentation
├── pyproject.toml               # Basic Kedro dependencies without figregistry-kedro
├── .kedro.yml                   # Standard Kedro project configuration
├── conf/
│   └── base/
│       ├── catalog.yml          # Basic catalog without FigureDataSet
│       └── parameters.yml       # Manual styling parameters
├── data/
│   ├── 01_raw/                 # Raw data
│   ├── 02_intermediate/        # Processed data
│   ├── 03_primary/            # Primary datasets
│   └── 08_reporting/          # Manual figure outputs
└── src/kedro_manual_example/
    ├── __init__.py
    ├── settings.py             # Standard Kedro settings
    ├── pipeline_registry.py    # Basic pipeline registration
    └── nodes.py               # Nodes with manual figure management
```

## Pain Points Highlighted

### Maintenance Overhead

1. **Style Updates**: Changing plot styling requires editing multiple node functions
2. **Inconsistent Output**: Different developers create figures with different conventions
3. **File Management**: Manual directory creation and naming leads to organizational chaos
4. **Code Duplication**: Repeated figure styling and saving logic across the codebase
5. **Error-Prone**: Manual condition handling leads to styling inconsistencies and bugs

### Development Inefficiency

1. **Time Waste**: Developers spend significant time on figure management instead of analysis
2. **Debugging Difficulty**: Figure-related bugs scattered across multiple files
3. **Testing Complexity**: Manual figure operations difficult to test systematically
4. **Collaboration Issues**: Team members use different styling and naming conventions

### Production Risks

1. **Inconsistent Outputs**: Figures vary in quality and styling across pipeline runs
2. **Missing Files**: Manual path management can lead to missing output directories
3. **Version Conflicts**: No systematic versioning leads to figure overwrites
4. **Configuration Drift**: Styling parameters drift apart across different parts of the pipeline

## Running the Traditional Example

### Prerequisites

```bash
# Install basic dependencies
pip install kedro[pandas] matplotlib pandas numpy scipy
```

### Setup Instructions

1. **Initialize the project:**
```bash
cd figregistry-kedro/examples/migration/before
pip install -e .
```

2. **Run the traditional pipeline:**
```bash
kedro run
```

3. **Observe the problems:**
   - Check multiple files for scattered styling code
   - Notice inconsistent figure outputs in `data/08_reporting/`
   - See hardcoded paths and manual timestamp generation
   - Find repeated styling logic across different nodes

### Expected Traditional Workflow

1. **Manual Setup**: Developers manually configure styling in each function
2. **Scattered Execution**: plt.savefig() calls embedded throughout pipeline nodes
3. **Manual Organization**: Developers manually create directory structures
4. **Inconsistent Output**: Each function applies different styling approaches
5. **Manual Maintenance**: Style changes require editing multiple files

## Migration Benefits

After experiencing the traditional approach, compare with the `figregistry-kedro` automated alternative:

### ❌ Before (Traditional Approach)
- Scattered `plt.savefig()` calls in every node function
- Hardcoded styling parameters throughout the codebase
- Manual file path construction and directory management
- Inconsistent experimental condition handling
- Code duplication across visualization functions
- No centralized configuration management
- Manual versioning and timestamp generation

### ✅ After (figregistry-kedro Integration)
- **Zero manual saves**: Automatic figure persistence through catalog
- **Centralized styling**: Configuration-driven styling through `figregistry.yaml`
- **Automated organization**: Systematic file naming and directory structure
- **Condition-based styling**: Automatic style application based on experimental conditions
- **DRY principle**: Single configuration source eliminates code duplication
- **Lifecycle integration**: Automated initialization and context management
- **Version management**: Integrated with Kedro's catalog versioning system

## Key Learnings

This example demonstrates why manual figure management becomes unsustainable:

1. **Technical Debt**: Manual approaches accumulate maintenance overhead rapidly
2. **Consistency Issues**: Without central coordination, outputs become inconsistent
3. **Developer Productivity**: Significant time waste on figure management tasks
4. **Quality Risks**: Manual processes are error-prone and difficult to standardize
5. **Collaboration Friction**: Teams struggle with different styling and naming conventions

## Next Steps

To see how `figregistry-kedro` solves these problems:

1. **Review the automated example**: See `../after/` for the same analysis with figregistry-kedro
2. **Compare implementations**: Notice the elimination of manual figure management
3. **Study configuration**: Understand centralized styling through YAML configuration
4. **Test the improvements**: Experience automated figure generation and management

---

> 💡 **Migration Tip**: When converting from manual to automated figure management, start by identifying all `plt.savefig()` calls in your codebase - these represent opportunities for automation through figregistry-kedro integration.