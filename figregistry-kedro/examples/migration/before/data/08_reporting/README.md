# Traditional Manual Figure Management: The "Before" State

This directory demonstrates the traditional manual matplotlib figure management approach used in typical Kedro projects before figregistry-kedro integration. The scattered figure files in this directory illustrate the pain points and maintenance overhead that automated figure management eliminates.

## Overview of Manual Figure Management Problems

### 📁 Inconsistent File Naming Patterns

The files in this directory showcase the typical naming chaos found in manual figure management:

- **`analysis_plot_20231115.png`** - Manual timestamp embedding in filenames
  - Developer had to manually format and insert the date
  - Creates inconsistent timestamp formats across different developers
  - No standardized naming convention leads to confusion

- **`model_performance.pdf`** - Generic descriptive names without context
  - No timestamp or version information
  - Difficult to distinguish between different experimental runs
  - Cannot track when the figure was generated

- **`experiment_results_final.svg`** - Manual versioning attempts
  - "final" naming indicates poor version control practices
  - Unclear what makes this version "final" vs other iterations
  - Manual versioning leads to files like "final_v2", "final_REALLY_final"

- **`temp_analysis.png`** - Temporary files left behind
  - Development artifacts forgotten in production directories
  - No cleanup mechanism for temporary visualizations
  - Clutters output directories with outdated figures

### 🔧 Manual Format Selection Problems

Each file uses a different format (PNG, PDF, SVG) based on individual developer decisions:

- **Inconsistent quality**: Different formats have different resolution and quality characteristics
- **Purpose mismatch**: Presentation figures in PNG format, publication figures without vector graphics
- **Storage inefficiency**: No consideration of file size vs quality trade-offs
- **Integration complexity**: Different formats require different handling in downstream processes

### 💻 Code-Level Problems (What You Don't See Here)

These scattered files are symptoms of deeper problems in the pipeline code:

#### Scattered `plt.savefig()` Calls
```python
# Typical manual approach spread across multiple node functions
def analyze_data(df):
    fig, ax = plt.subplots()
    ax.plot(df['x'], df['y'], color='blue', marker='o')  # Hardcoded style
    plt.savefig('data/08_reporting/analysis_plot_20231115.png', dpi=300, bbox_inches='tight')
    plt.close()
    return some_data

def create_model_report(model_results):
    plt.figure(figsize=(10, 6))  # Hardcoded figure size
    plt.bar(model_results.index, model_results.values, color='red')  # Hardcoded style
    plt.savefig('data/08_reporting/model_performance.pdf')  # Different format, different path
    plt.close()
    return report_data
```

#### Hardcoded Styling Without Consistency
- Color schemes vary between functions and developers
- Font sizes and figure dimensions chosen arbitrarily
- No consideration for experimental conditions or purposes
- Styling parameters scattered throughout codebase

#### Manual Path Management
- File paths hardcoded in each function
- No centralized output directory management
- Risk of path conflicts and overwrites
- Difficult to reorganize output structure

### 📊 Maintenance Overhead

#### Developer Time Waste
- **15-20 minutes per figure** spent on manual styling decisions
- **Repeated styling code** across multiple nodes (estimated 90% code duplication)
- **Manual file organization** and cleanup tasks
- **Debugging path issues** and format compatibility problems

#### Quality and Consistency Issues
- **Visual inconsistency** across experiment outputs
- **Missing figures** due to hardcoded paths that don't exist
- **Overwritten results** from naming conflicts
- **Lost experimental context** when figures lack proper metadata

#### Collaboration Problems
- **Style conflicts** when multiple developers work on visualization code
- **Inconsistent quality standards** across team members
- **Difficulty reproducing** specific experimental visualizations
- **Knowledge silos** where each developer has their own styling approach

### 🔄 Technical Debt Accumulation

#### Performance Issues
- **Redundant styling calculations** performed in every function
- **Inefficient file I/O** with repeated directory creation attempts
- **Memory leaks** from figures not properly closed
- **Slow pipeline execution** due to styling overhead

#### Maintenance Complexity
- **Style updates require code changes** across multiple files
- **Format migrations** need manual code updates in every visualization node
- **Path restructuring** requires hunting down hardcoded references
- **Testing complexity** due to scattered figure generation logic

## Migration Benefits Preview

The figregistry-kedro integration eliminates ALL of these problems by providing:

✅ **Automated consistent naming** with configurable timestamp formats  
✅ **Centralized style management** through YAML configuration  
✅ **Purpose-driven format selection** based on usage context  
✅ **Zero manual plt.savefig() calls** in pipeline code  
✅ **Condition-based styling** that adapts to experimental parameters  
✅ **Integrated versioning** through Kedro's catalog system  
✅ **Clean code separation** between analysis logic and visualization styling  

## Next Steps

To see how figregistry-kedro transforms this chaotic manual approach into a clean, automated workflow, examine the corresponding files in the `../after/` directory. The transformation demonstrates:

1. **Code simplification**: Node functions focus purely on analysis logic
2. **Configuration centralization**: All styling managed through `figregistry.yml`
3. **Automated organization**: Consistent naming and directory structure
4. **Quality improvements**: Publication-ready figures with zero manual intervention

The difference is dramatic - what once required scattered manual effort across dozens of files becomes a single configuration file and clean dataset declarations in the Kedro catalog.

---

**Note**: This directory represents the "before" state in a migration example. The actual figure files here are examples of the output chaos typical in manual matplotlib management workflows. For the clean, automated approach, see `../after/data/08_reporting/` where figregistry-kedro manages everything automatically.