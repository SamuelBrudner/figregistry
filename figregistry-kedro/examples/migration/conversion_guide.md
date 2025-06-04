# FigRegistry-Kedro Migration Guide: From Manual to Automated Figure Management

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Migration Benefits](#migration-benefits)
4. [Step-by-Step Conversion Process](#step-by-step-conversion-process)
5. [Before/After Code Examples](#beforeafter-code-examples)
6. [Configuration Setup](#configuration-setup)
7. [Validation and Testing](#validation-and-testing)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Overview

This guide provides comprehensive instructions for migrating existing Kedro projects from manual matplotlib figure management to automated figregistry-kedro integration. The migration transforms scattered `plt.savefig()` calls and hardcoded styling into a centralized, condition-based automation system that eliminates code duplication while ensuring consistent, publication-quality visualizations.

### What This Migration Accomplishes

- **Eliminates manual `plt.savefig()` calls** throughout pipeline nodes (Target: 90% code reduction)
- **Automates condition-based styling** through catalog configuration (F-005)
- **Centralizes figure management** via FigRegistry's configuration system (F-007)
- **Integrates seamlessly** with Kedro's versioning and experiment tracking (F-005-RQ-002)
- **Maintains backward compatibility** with existing pipeline logic

### Core Components Involved

- **FigureDataSet**: Custom Kedro dataset for automated figure styling and persistence
- **FigRegistryHooks**: Lifecycle hooks for automatic configuration initialization
- **FigRegistryConfigBridge**: Configuration translation between Kedro and FigRegistry
- **figregistry.yml**: Centralized styling and output configuration

## Prerequisites

### Required Dependencies

```bash
# Install figregistry-kedro with all dependencies
pip install figregistry-kedro

# Verify installation
python -c "import figregistry_kedro; print('✅ Installation successful')"
```

### Kedro Version Compatibility

- **Kedro**: `>=0.18.0,<0.20.0`
- **FigRegistry**: `>=0.3.0`
- **Python**: `>=3.10`

### Project Requirements

- Existing Kedro project with visualization pipeline nodes
- Pipeline nodes that currently use matplotlib for figure creation
- Write access to project configuration directories (`conf/`)

## Migration Benefits

### Quantified Improvements

| Metric | Before (Manual) | After (Automated) | Improvement |
|--------|----------------|-------------------|-------------|
| Lines of styling code | ~20-50 per figure | 0 per figure | 90%+ reduction |
| Configuration management | Scattered across nodes | Centralized in catalog | Single source |
| Style consistency | Manual, error-prone | Automatic, guaranteed | 100% consistent |
| File organization | Manual path management | Automatic organization | Zero maintenance |
| Experimental variations | Hardcoded conditions | Dynamic resolution | Infinite flexibility |

### Workflow Benefits

- **Zero manual intervention** for figure styling and persistence
- **Automatic versioning** integration with Kedro experiments
- **Environment-specific styling** through configuration hierarchy
- **Publication-ready outputs** with consistent quality standards
- **Reduced maintenance overhead** for visualization updates

## Step-by-Step Conversion Process

### Step 1: Install and Verify figregistry-kedro

```bash
# Install the integration package
pip install figregistry-kedro

# Verify all components are available
python -c "
from figregistry_kedro.datasets import FigureDataSet
from figregistry_kedro.hooks import FigRegistryHooks
from figregistry_kedro.config import FigRegistryConfigBridge
print('✅ All components imported successfully')
"
```

### Step 2: Register FigRegistryHooks in settings.py

**Location**: `src/{project_name}/settings.py`

**BEFORE (Manual Approach)**:
```python
# settings.py - Traditional configuration
HOOKS = []  # No automated figure management
```

**AFTER (Automated Approach)**:
```python
# settings.py - FigRegistry integration enabled
from figregistry_kedro.hooks import FigRegistryHooks

# Register FigRegistryHooks for automated lifecycle management (F-006)
HOOKS = (
    FigRegistryHooks(
        enable_performance_monitoring=True,    # Track integration performance
        fallback_on_errors=True,              # Graceful error handling
        strict_validation=True,               # Validate merged configurations
        config_cache_enabled=True            # Enable performance caching
    ),
)

# Enhanced configuration patterns for FigRegistry integration (F-007)
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "catalog": ["catalog*.yml", "catalog*.yaml"],
        "parameters": ["parameters*.yml", "parameters*.yaml"],
        "credentials": ["credentials*.yml", "credentials*.yaml"],
        # NEW: Enable FigRegistry configuration discovery
        "figregistry": ["figregistry*.yml", "figregistry*.yaml"]
    }
}
```

### Step 3: Create FigRegistry Configuration

**Location**: `conf/base/figregistry.yml`

Create a comprehensive FigRegistry configuration that defines your styling rules:

```yaml
# figregistry.yml - Centralized styling configuration
figregistry_version: ">=0.3.0"

metadata:
  config_version: "1.0.0"
  description: "Automated figure styling for Kedro pipeline"
  environment: "development"

# Condition-based style mappings (F-002)
styles:
  # Exploratory analysis styling
  exploratory:
    color: "#2E86AB"
    marker: "o"
    linestyle: "-"
    linewidth: 2.0
    alpha: 0.8
    label: "Exploratory Analysis"

  # Presentation-quality styling
  presentation:
    color: "#E74C3C"
    marker: "s"
    linestyle: "-"
    linewidth: 2.5
    alpha: 0.9
    label: "Presentation"

  # Publication-ready styling
  publication:
    color: "#2C3E50"
    marker: "D"
    linestyle: "-"
    linewidth: 2.8
    alpha: 1.0
    label: "Publication"

  # Model-specific styling
  random_forest:
    color: "#27AE60"
    marker: "^"
    linestyle: "-"
    linewidth: 2.3
    alpha: 0.85
    label: "Random Forest"

  linear_model:
    color: "#8E44AD"
    marker: "v"
    linestyle: "--"
    linewidth: 2.1
    alpha: 0.8
    label: "Linear Model"

# Default styling fallbacks
defaults:
  figure:
    figsize: [10, 8]
    dpi: 150
  line:
    color: "#34495E"
    linewidth: 2.0
  fallback_style:
    color: "#95A5A6"
    marker: "o"
    linestyle: "-"
    linewidth: 1.5
    alpha: 0.7
    label: "Default"

# Output management configuration (F-004)
outputs:
  base_path: "data/08_reporting"
  naming:
    template: "{name}_{condition}_{ts}"
  directories:
    exploratory: "exploratory"
    presentation: "presentation"
    publication: "publication"
```

### Step 4: Update Data Catalog Configuration

**Location**: `conf/base/catalog.yml`

Transform your catalog to use FigureDataSet for all matplotlib outputs:

**BEFORE (Manual Dataset Entries)**:
```yaml
# Traditional approach - no automated figure management
training_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/training_data.csv

# No figure datasets - manual plt.savefig() calls in nodes
```

**AFTER (Automated FigureDataSet Entries)**:
```yaml
# Data pipeline remains unchanged
training_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/training_data.csv

# NEW: Automated figure management through FigureDataSet (F-005)
model_performance_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/model_performance_${params:experiment_type}.png
  purpose: presentation                        # Maps to styling purpose
  condition_param: model_type                  # Resolves from parameters.yml
  style_params:                               # Dataset-specific overrides
    figure.figsize: [12, 8]
    figure.dpi: 300
  save_args:
    bbox_inches: tight
    facecolor: white
  versioned: true                             # Kedro versioning integration

feature_importance_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/feature_importance_${params:model_type}.png
  purpose: exploratory
  condition_param: analysis_phase
  style_params:
    figure.figsize: [10, 12]                  # Vertical layout for feature names
  versioned: true

validation_results_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/validation_results.pdf
  purpose: publication                        # Publication-quality output
  condition_param: validation_method
  style_params:
    figure.dpi: 300                          # High DPI for publication
  save_args:
    format: pdf                              # Vector format
    bbox_inches: tight
  versioned: true
```

### Step 5: Update Pipeline Nodes

Transform your pipeline nodes from manual figure management to automated processing:

**BEFORE (Manual Approach)**:
```python
# nodes.py - Manual matplotlib management with problems
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_model_performance_plot(model_metrics: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> None:
    """PROBLEMATIC: Manual figure management with hardcoded styling."""
    
    # PROBLEM: Hardcoded styling parameters
    if parameters.get("model_type") == "random_forest":
        color = "#27AE60"
        marker = "s"
        linewidth = 2.5
    elif parameters.get("model_type") == "linear_model":
        color = "#8E44AD"
        marker = "^"
        linewidth = 2.0
    else:
        color = "gray"  # PROBLEM: Inconsistent fallback
        marker = "o"
        linewidth = 1.5
    
    # PROBLEM: Manual figure creation with styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # PROBLEM: Manual styling application
    ax1.plot(model_metrics['train_scores'], color=color, 
            marker=marker, linewidth=linewidth, alpha=0.8)
    ax1.set_title("Training Performance", fontsize=14, color=color)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(model_metrics['val_scores'], color=color,
            marker=marker, linewidth=linewidth, alpha=0.8)
    ax2.set_title("Validation Performance", fontsize=14, color=color)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # PROBLEM: Manual file path construction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = parameters.get("model_type", "unknown")
    output_dir = "data/08_reporting/presentation"
    
    # PROBLEM: Manual directory creation
    os.makedirs(output_dir, exist_ok=True)
    filename = f"model_performance_{model_type}_{timestamp}.png"
    full_path = os.path.join(output_dir, filename)
    
    # PROBLEM: Manual plt.savefig() call with hardcoded parameters
    plt.savefig(full_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved plot to: {full_path}")
    return None  # Returns nothing - figure handled manually
```

**AFTER (Automated Approach)**:
```python
# nodes.py - Automated figure management (clean and simple)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any

def create_model_performance_plot(model_metrics: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> Figure:
    """SOLUTION: Clean node focused on visualization logic only."""
    
    # Clean figure creation - no manual styling needed
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Pure visualization logic - styling handled by FigureDataSet
    ax1.plot(model_metrics['train_scores'])
    ax1.set_title("Training Performance")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.grid(True)
    
    ax2.plot(model_metrics['val_scores'])
    ax2.set_title("Validation Performance")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Return figure object - FigureDataSet handles everything else
    return fig  # ✅ FigureDataSet applies styling and saves automatically
```

### Step 6: Update Parameters Configuration

**Location**: `conf/base/parameters.yml`

Ensure your parameters support condition-based styling:

```yaml
# parameters.yml - Enable condition resolution for automated styling

# Model configuration for condition_param resolution
model_configuration:
  model_type: "random_forest"              # Used by condition_param: model_type
  training_variant: "optimized"            # Used for training-specific styling
  complexity_level: "moderate"             # Used for complexity-based styling

# Analysis configuration
analysis_configuration:
  analysis_phase: "validation"             # Used for phase-specific styling
  output_target: "stakeholder"             # Used for audience-specific styling

# Dataset configuration
dataset_configuration:
  dataset_variant: "real_world"            # Used for data-specific styling
  data_quality: "high_quality"             # Used for quality-based styling
  sample_size_category: "medium_sample"    # Used for sample-dependent styling

# Execution context
execution_environment:
  environment_type: "production"           # Used for environment-specific styling
  resource_level: "standard"               # Used for resource-aware styling

# Visualization context
visualization_context:
  audience_type: "technical"               # Used for audience-specific outputs
  presentation_medium: "screen"            # Used for medium-specific optimization
  accessibility_level: "colorblind_safe"   # Used for accessibility requirements
  quality_requirement: "review"            # Used for quality-specific styling

# Experiment tracking
experiment_tracking:
  experiment_id: "exp_001"                 # Used for experiment-specific outputs
  experiment_condition: "baseline"         # Primary condition identifier
```

## Before/After Code Examples

### Example 1: Training Metrics Visualization

**BEFORE - Manual Approach (Problematic)**:
```python
def visualize_training_metrics(metrics: Dict, params: Dict) -> None:
    """Problems: Manual styling, hardcoded paths, code duplication."""
    
    # Manual styling logic (repeated across functions)
    if params.get("experiment_type") == "baseline":
        colors = ["#3498DB", "#E74C3C"]
        linewidth = 2.0
    elif params.get("experiment_type") == "optimized":
        colors = ["#27AE60", "#8E44AD"]
        linewidth = 2.5
    else:
        colors = ["gray", "lightgray"]
        linewidth = 1.5
    
    # Manual figure creation and styling
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['epochs'], metrics['train_loss'], 
           color=colors[0], linewidth=linewidth, label='Training Loss')
    ax.plot(metrics['epochs'], metrics['val_loss'],
           color=colors[1], linewidth=linewidth, label='Validation Loss')
    
    # Manual styling application
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Manual file management
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/08_reporting/training_metrics_{timestamp}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Manual save with hardcoded parameters
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

**AFTER - Automated Approach (Clean)**:
```python
def visualize_training_metrics(metrics: Dict, params: Dict) -> Figure:
    """Solution: Clean visualization logic, automated management."""
    
    # Simple figure creation - no styling needed
    fig, ax = plt.subplots()  # Size handled by FigureDataSet
    
    # Pure visualization logic
    ax.plot(metrics['epochs'], metrics['train_loss'], label='Training Loss')
    ax.plot(metrics['epochs'], metrics['val_loss'], label='Validation Loss')
    
    # Basic labeling - styling applied automatically
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True)
    
    # Return figure - FigureDataSet handles styling and saving
    return fig
```

**Catalog Configuration**:
```yaml
training_metrics_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/training_metrics.png
  purpose: validation
  condition_param: experiment_type  # Automatically resolves styling
  versioned: true
```

### Example 2: Feature Importance Analysis

**BEFORE - Manual Approach**:
```python
def create_feature_importance_plot(importance_scores: pd.DataFrame,
                                  parameters: Dict[str, Any]) -> None:
    """Manual feature importance with repetitive styling code."""
    
    # Manual color scheme selection
    model_type = parameters.get("model_type", "unknown")
    if model_type == "random_forest":
        bar_color = "#2ECC71"
        title_color = "#27AE60"
    elif model_type == "gradient_boosting":
        bar_color = "#3498DB"
        title_color = "#2980B9"
    else:
        bar_color = "#95A5A6"
        title_color = "#7F8C8D"
    
    # Manual figure configuration
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Manual plotting with styling
    bars = ax.barh(importance_scores['feature'], 
                   importance_scores['importance'],
                   color=bar_color, alpha=0.8, edgecolor='black')
    
    # Manual formatting
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_type.title()}', 
                fontsize=14, color=title_color, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Manual file handling
    purpose = parameters.get("purpose", "analysis")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feature_importance_{model_type}_{purpose}_{timestamp}.png"
    
    output_dir = "data/08_reporting/analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

**AFTER - Automated Approach**:
```python
def create_feature_importance_plot(importance_scores: pd.DataFrame,
                                  parameters: Dict[str, Any]) -> Figure:
    """Clean feature importance focused on analysis logic."""
    
    # Simple figure creation
    fig, ax = plt.subplots()
    
    # Core visualization logic only
    ax.barh(importance_scores['feature'], 
            importance_scores['importance'])
    
    # Basic labeling
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Analysis')
    ax.grid(True, axis='x')
    
    # Return for automated processing
    return fig
```

**Catalog Configuration**:
```yaml
feature_importance_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/feature_importance_${params:model_type}.png
  purpose: exploratory
  condition_param: model_type  # Automatic model-specific styling
  style_params:
    figure.figsize: [10, 8]    # Vertical layout for feature names
  versioned: true
```

## Configuration Setup

### Complete Configuration Examples

#### 1. Enhanced figregistry.yml

```yaml
# Complete figregistry.yml for production use
figregistry_version: ">=0.3.0"

metadata:
  config_version: "1.0.0"
  description: "Production figure styling configuration"
  environment: "production"
  last_updated: "2024-01-01"

# Comprehensive style definitions
styles:
  # Purpose-based styling
  exploratory:
    color: "#2E86AB"
    marker: "o"
    linestyle: "-"
    linewidth: 2.0
    alpha: 0.8
    markersize: 8

  presentation:
    color: "#E74C3C"
    marker: "s"
    linestyle: "-"
    linewidth: 2.5
    alpha: 0.9
    markersize: 9

  publication:
    color: "#2C3E50"
    marker: "D"
    linestyle: "-"
    linewidth: 3.0
    alpha: 1.0
    markersize: 10

  # Model-specific styling
  random_forest:
    color: "#27AE60"
    marker: "^"
    linestyle: "-"
    linewidth: 2.3
    alpha: 0.85
    label: "Random Forest"

  linear_model:
    color: "#8E44AD"
    marker: "v"
    linestyle: "--"
    linewidth: 2.1
    alpha: 0.8
    label: "Linear Model"

  neural_network:
    color: "#F39C12"
    marker: "*"
    linestyle: "-"
    linewidth: 2.4
    alpha: 0.9
    label: "Neural Network"

  # Quality-based styling
  high_quality:
    color: "#1ABC9C"
    marker: "P"
    linestyle: "-"
    linewidth: 2.2
    alpha: 0.9
    label: "High Quality"

  medium_quality:
    color: "#F1C40F"
    marker: "X"
    linestyle: "-"
    linewidth: 2.0
    alpha: 0.8
    label: "Medium Quality"

# Advanced defaults
defaults:
  figure:
    figsize: [10, 8]
    dpi: 150
    facecolor: "white"
  
  axes:
    grid: true
    grid_alpha: 0.3
    titlesize: 14
    labelsize: 12
  
  line:
    color: "#34495E"
    linewidth: 2.0
    alpha: 0.8
  
  legend:
    loc: "best"
    fontsize: 11
    framealpha: 0.9
  
  text:
    fontsize: 11
    color: "#2C3E50"

# Output configuration
outputs:
  base_path: "data/08_reporting"
  naming:
    template: "{name}_{condition}_{ts}"
    timestamp_format: "%Y%m%d_%H%M%S"
  
  directories:
    exploratory: "exploratory"
    presentation: "presentation"
    publication: "publication"
    validation: "validation"
    analysis: "analysis"

# Performance optimization
performance:
  cache_enabled: true
  max_cache_size: 1000
  style_resolution_timeout_ms: 10
```

#### 2. Advanced Catalog Configuration

```yaml
# Complete catalog.yml with multiple FigureDataSet examples

# Training and validation visualizations
training_loss_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/training/loss_curves_${params:experiment_id}.png
  purpose: validation
  condition_param: model_configuration.model_type
  style_params:
    figure.figsize: [12, 6]
    line.linewidth: 2.5
  save_args:
    dpi: 300
    bbox_inches: tight
  versioned: true

model_comparison_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/comparison/model_comparison.pdf
  purpose: publication
  condition_param: analysis_configuration.comparison_type
  style_params:
    figure.figsize: [14, 10]
    figure.dpi: 300
  save_args:
    format: pdf
    bbox_inches: tight
  versioned: true

# Feature analysis
feature_correlation_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/features/correlation_matrix.png
  purpose: exploratory
  condition_param: dataset_configuration.data_quality
  style_params:
    figure.figsize: [12, 12]
  versioned: true

feature_importance_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/features/importance_${params:model_type}.png
  purpose: presentation
  condition_param: model_configuration.model_type
  style_params:
    figure.figsize: [10, 12]  # Vertical for feature names
  versioned: true

# Performance evaluation
confusion_matrix_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/evaluation/confusion_matrix.png
  purpose: validation
  condition_param: dataset_configuration.data_quality
  style_params:
    figure.figsize: [8, 8]
  versioned: true

roc_curve_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/evaluation/roc_curves.png
  purpose: publication
  condition_param: model_configuration.model_type
  style_params:
    figure.figsize: [10, 8]
    figure.dpi: 300
  save_args:
    dpi: 300
    bbox_inches: tight
  versioned: true

# Diagnostic plots
residual_analysis_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/diagnostics/residuals.png
  purpose: exploratory
  condition_param: dataset_configuration.sample_size_category
  style_params:
    figure.figsize: [12, 8]
    scatter.alpha: 0.6
  versioned: true

learning_curve_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/diagnostics/learning_curves.png
  purpose: validation
  condition_param: model_configuration.training_variant
  style_params:
    figure.figsize: [14, 6]
    line.linewidth: 2.5
  versioned: true
```

## Validation and Testing

### Step 1: Test Hook Registration

Create a validation script to verify hook registration:

```python
# test_migration.py - Migration validation script
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def test_hook_registration():
    """Test that FigRegistryHooks are properly registered."""
    try:
        from {project_name}.settings import HOOKS
        
        if not HOOKS:
            print("❌ No hooks registered")
            return False
        
        # Check for FigRegistryHooks
        hook_types = [type(hook).__name__ for hook in HOOKS]
        if "FigRegistryHooks" not in hook_types:
            print(f"❌ FigRegistryHooks not found. Found: {hook_types}")
            return False
        
        print("✅ FigRegistryHooks registered successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_figregistry_config():
    """Test that figregistry.yml is valid and loadable."""
    try:
        config_path = project_root / "conf" / "base" / "figregistry.yml"
        
        if not config_path.exists():
            print(f"❌ figregistry.yml not found at {config_path}")
            return False
        
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ["styles", "defaults", "outputs"]
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            print(f"❌ Missing required sections: {missing_sections}")
            return False
        
        print("✅ figregistry.yml configuration valid")
        print(f"   Styles defined: {len(config['styles'])}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_figuredataset_catalog():
    """Test that catalog contains FigureDataSet entries."""
    try:
        catalog_path = project_root / "conf" / "base" / "catalog.yml"
        
        if not catalog_path.exists():
            print(f"❌ catalog.yml not found at {catalog_path}")
            return False
        
        import yaml
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)
        
        # Find FigureDataSet entries
        figuredatasets = []
        for name, config in catalog.items():
            if isinstance(config, dict) and "figregistry_kedro.FigureDataSet" in str(config.get("type", "")):
                figuredatasets.append(name)
        
        if not figuredatasets:
            print("❌ No FigureDataSet entries found in catalog")
            return False
        
        print(f"✅ Found {len(figuredatasets)} FigureDataSet entries:")
        for ds in figuredatasets[:5]:  # Show first 5
            print(f"   - {ds}")
        
        return True
        
    except Exception as e:
        print(f"❌ Catalog validation error: {e}")
        return False

def run_migration_validation():
    """Run complete migration validation."""
    print("🔍 Validating figregistry-kedro migration...")
    print("=" * 50)
    
    tests = [
        ("Hook Registration", test_hook_registration),
        ("FigRegistry Configuration", test_figregistry_config),
        ("Catalog Integration", test_figuredataset_catalog)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("🎉 MIGRATION VALIDATION PASSED!")
        print("Your project is ready for automated figure management.")
        return True
    else:
        print("⚠️  MIGRATION VALIDATION FAILED")
        print("Please address the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = run_migration_validation()
    sys.exit(0 if success else 1)
```

### Step 2: Test Pipeline Execution

Run a test pipeline to verify the integration:

```bash
# Test the pipeline with figregistry-kedro integration
kedro run --pipeline=data_visualization

# Check that figures were created automatically
ls -la data/08_reporting/

# Verify automated styling was applied
kedro run --pipeline=data_visualization --params="model_type:linear_model"
```

### Step 3: Performance Validation

Create a performance test to ensure minimal overhead:

```python
# performance_test.py - Measure integration overhead
import time
import matplotlib.pyplot as plt
import numpy as np
from figregistry_kedro.datasets import FigureDataSet

def test_save_performance():
    """Test FigureDataSet save performance vs manual saves."""
    
    # Create test figure
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 1000)
    ax.plot(x, np.sin(x))
    
    # Test manual matplotlib save
    start_time = time.time()
    fig.savefig("test_manual.png", dpi=300, bbox_inches='tight')
    manual_time = (time.time() - start_time) * 1000
    
    # Test FigureDataSet save
    dataset = FigureDataSet(
        filepath="test_figregistry.png",
        purpose="exploratory",
        condition_param="test_condition"
    )
    
    start_time = time.time()
    dataset._save(fig)
    figregistry_time = (time.time() - start_time) * 1000
    
    # Calculate overhead
    overhead_percent = ((figregistry_time - manual_time) / manual_time) * 100
    
    print(f"Manual save time: {manual_time:.2f}ms")
    print(f"FigureDataSet save time: {figregistry_time:.2f}ms")
    print(f"Overhead: {overhead_percent:.1f}%")
    
    # Validate overhead is within acceptable range (<5%)
    if overhead_percent < 5.0:
        print("✅ Performance overhead within acceptable range")
        return True
    else:
        print("❌ Performance overhead exceeds 5% target")
        return False

if __name__ == "__main__":
    test_save_performance()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Hook Registration Issues

**Problem**: `FigRegistryHooks not found` error

**Solution**:
```python
# Verify installation
pip list | grep figregistry-kedro

# If not installed:
pip install figregistry-kedro

# Check import in settings.py
from figregistry_kedro.hooks import FigRegistryHooks
print("✅ Import successful")
```

#### 2. Configuration Loading Errors

**Problem**: `figregistry.yml not found` or loading errors

**Solution**:
```bash
# Verify file location
ls -la conf/base/figregistry.yml

# Validate YAML syntax
python -c "
import yaml
with open('conf/base/figregistry.yml') as f:
    config = yaml.safe_load(f)
print('✅ YAML syntax valid')
"
```

#### 3. Condition Resolution Issues

**Problem**: Styles not applied correctly

**Solutions**:

**Check parameter availability**:
```yaml
# In parameters.yml - ensure condition parameters exist
model_configuration:
  model_type: "random_forest"  # Must match condition_param usage
```

**Verify condition_param configuration**:
```yaml
# In catalog.yml - check condition_param path
my_plot:
  type: figregistry_kedro.FigureDataSet
  condition_param: model_configuration.model_type  # Nested parameter path
```

**Add fallback styling**:
```yaml
# In figregistry.yml - always include fallback
defaults:
  fallback_style:
    color: "#95A5A6"
    marker: "o"
    linestyle: "-"
```

#### 4. Versioning Integration Issues

**Problem**: Kedro versioning not working with FigureDataSet

**Solution**:
```yaml
# Ensure versioned: true in catalog entry
my_versioned_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/plot.png
  versioned: true  # ← Critical for Kedro versioning
  purpose: validation
```

#### 5. Performance Issues

**Problem**: Slow figure saves or style resolution

**Solutions**:

**Enable caching**:
```python
# In settings.py
HOOKS = (
    FigRegistryHooks(
        config_cache_enabled=True,  # Enable configuration caching
        cache_size=1000            # Increase cache size
    ),
)
```

**Optimize style configuration**:
```yaml
# In figregistry.yml - avoid complex nested configurations
styles:
  simple_style:  # Keep styles simple and focused
    color: "#2E86AB"
    linewidth: 2.0
```

### Debugging Tools

#### Enable Debug Logging

```python
# Add to pipeline nodes for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# FigRegistry-specific logging
logger = logging.getLogger("figregistry_kedro")
logger.setLevel(logging.DEBUG)
```

#### Configuration Inspection

```python
# Inspect merged configuration
from figregistry_kedro.config import get_merged_config

config = get_merged_config(environment="local")
print("Available styles:", list(config.get("styles", {}).keys()))
print("Output configuration:", config.get("outputs", {}))
```

#### Hook State Monitoring

```python
# Check hook state
from figregistry_kedro.hooks import get_global_hook_state

state = get_global_hook_state()
print("Hook state:", state)
```

## Best Practices

### 1. Configuration Organization

#### Structured Style Definitions
```yaml
# Organize styles by category for maintainability
styles:
  # Purpose-based styles
  exploratory:
    color: "#2E86AB"
    # ... other properties
    
  # Model-specific styles
  random_forest:
    color: "#27AE60"
    # ... other properties
    
  # Quality-based styles
  high_quality:
    color: "#1ABC9C"
    # ... other properties
```

#### Environment-Specific Overrides
```yaml
# conf/local/figregistry.yml - Development overrides
styles:
  exploratory:
    figure:
      dpi: 150  # Lower DPI for faster development iteration

# conf/production/figregistry.yml - Production overrides  
styles:
  publication:
    figure:
      dpi: 300  # High DPI for production quality
```

### 2. Parameter Design

#### Clear Parameter Hierarchy
```yaml
# parameters.yml - Structured for condition resolution
model_configuration:
  model_type: "random_forest"
  training_variant: "optimized"
  
analysis_configuration:
  analysis_phase: "validation"
  output_target: "stakeholder"
  
execution_environment:
  environment_type: "production"
  resource_level: "standard"
```

#### Consistent Naming Conventions
```yaml
# Use consistent naming patterns for condition parameters
dataset_configuration:
  dataset_variant: "real_world"        # Use descriptive variants
  data_quality: "high_quality"         # Use quality indicators
  sample_size_category: "large_sample" # Use category classifications
```

### 3. Catalog Design

#### Logical Output Organization
```yaml
# Group related visualizations
training_metrics_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/training/metrics.png
  
validation_results_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/validation/results.png
  
final_presentation_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/presentation/summary.png
```

#### Template-Based Naming
```yaml
# Use parameter substitution for dynamic naming
model_comparison_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/comparison/${params:model_type}_vs_baseline.png
  condition_param: model_configuration.model_type
```

### 4. Node Implementation

#### Clean Separation of Concerns
```python
def create_visualization(data: pd.DataFrame, parameters: Dict) -> Figure:
    """Keep nodes focused on visualization logic only."""
    
    # ✅ Good: Focus on data visualization
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    ax.set_title("Analysis Results")
    
    # ✅ Good: Return figure for automated processing
    return fig
    
    # ❌ Avoid: Manual styling, saving, or path management
    # plt.savefig("output.png")  # Don't do this
```

#### Parameterized Visualization Logic
```python
def create_model_comparison(results: Dict, parameters: Dict) -> Figure:
    """Use parameters for visualization logic, not styling."""
    
    fig, ax = plt.subplots()
    
    # ✅ Good: Use parameters for data selection and layout
    models_to_compare = parameters.get("models_to_compare", ["model_a", "model_b"])
    
    for model in models_to_compare:
        if model in results:
            ax.plot(results[model]['metrics'], label=model)
    
    ax.legend()
    return fig
```

### 5. Testing Strategy

#### Unit Testing for Nodes
```python
# test_nodes.py
import pytest
from matplotlib.figure import Figure

def test_create_visualization():
    """Test that node returns a valid Figure object."""
    # Arrange
    test_data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    test_params = {"analysis_type": "exploration"}
    
    # Act
    result = create_visualization(test_data, test_params)
    
    # Assert
    assert isinstance(result, Figure)
    assert len(result.get_axes()) > 0
```

#### Integration Testing
```python
# test_integration.py
def test_figuredataset_integration():
    """Test that FigureDataSet processes figures correctly."""
    # Create test dataset
    dataset = FigureDataSet(
        filepath="test_output.png",
        purpose="validation",
        condition_param="test_condition"
    )
    
    # Create test figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    
    # Test save operation
    dataset._save(fig)
    
    # Verify output exists
    assert Path("test_output.png").exists()
```

### 6. Performance Optimization

#### Efficient Style Caching
```python
# Enable caching for better performance
HOOKS = (
    FigRegistryHooks(
        config_cache_enabled=True,
        max_cache_size=1000,
        performance_target_ms=10.0
    ),
)
```

#### Lazy Loading Patterns
```yaml
# Use versioning strategically
development_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/dev_plot.png
  versioned: false  # Disable versioning for development speed

production_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/prod_plot.png
  versioned: true   # Enable versioning for production tracking
```

### 7. Documentation and Maintenance

#### Configuration Documentation
```yaml
# Document your styling choices
styles:
  publication:
    # Publication-quality styling for peer review
    # - High DPI for crisp text
    # - Conservative colors for accessibility
    # - Larger fonts for readability
    color: "#2C3E50"
    marker: "D"
    linewidth: 3.0
```

#### Migration Tracking
```python
# Keep migration notes in settings.py
"""
Migration Notes:
- Converted 15 manual plt.savefig() calls to FigureDataSet automation
- Eliminated ~500 lines of repetitive styling code
- Added condition-based styling for 8 experimental conditions
- Performance overhead: <2% compared to manual approach
"""
```

---

## Summary

This migration guide provides a comprehensive approach to transforming Kedro projects from manual matplotlib figure management to automated figregistry-kedro integration. The key transformation involves:

1. **Installing figregistry-kedro** and registering FigRegistryHooks
2. **Creating centralized configuration** in figregistry.yml  
3. **Updating the data catalog** to use FigureDataSet entries
4. **Modifying pipeline nodes** to return Figure objects instead of manual saves
5. **Validating the migration** through comprehensive testing

The result is a significant reduction in visualization-related code (target: 90%), consistent styling across all experimental conditions, and seamless integration with Kedro's versioning and experiment tracking capabilities.

For additional support and advanced configuration options, refer to the comprehensive examples in the `figregistry-kedro/examples/` directory.