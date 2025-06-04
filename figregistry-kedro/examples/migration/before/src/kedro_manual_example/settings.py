"""
Traditional Kedro Project Settings - Manual Figure Management Example

This module demonstrates traditional Kedro project configuration before
figregistry-kedro integration. This "before" state shows how data science
teams managed figure generation manually through scattered plt.savefig() calls,
hardcoded styling parameters, and manual file path management throughout
pipeline nodes.

Manual Configuration Challenges Demonstrated:
- Code duplication across nodes for consistent styling
- Hardcoded file paths and naming conventions
- Manual matplotlib rcParams configuration in each node
- Inconsistent styling parameters between experimental conditions
- Manual figure versioning and organization overhead
- No centralized configuration for team-wide consistency

This example serves as the baseline for migration comparison, highlighting
the maintenance overhead and scattered configuration that figregistry-kedro
integration eliminates through automated lifecycle management and
condition-based styling.

Traditional Workflow Requirements:
- Manual plt.savefig() calls in every pipeline node
- Hardcoded style dictionaries scattered across functions
- Manual file path construction and directory management
- Individual node responsibility for figure styling consistency
- Manual configuration of experimental condition mappings

Comparison with figregistry-kedro Integration:
Before (This File):
    - Standard Kedro HOOKS configuration without lifecycle integration
    - Manual figure management requiring developer intervention
    - Scattered styling code with high maintenance overhead
    - No centralized configuration for visualization parameters

After (figregistry-kedro):
    - Automated FigRegistryHooks registration for lifecycle integration
    - Zero-touch figure management through FigureDataSet interception
    - Centralized YAML-based configuration for consistent styling
    - Condition-based styling with no manual intervention required

Usage:
    This traditional configuration requires manual figure management:
    
    # In pipeline nodes (manual approach):
    def create_scatter_plot(data, params):
        fig, ax = plt.subplots(figsize=(10, 8))
        # Manual styling configuration
        ax.scatter(data['x'], data['y'], 
                  color='blue' if params['condition'] == 'control' else 'red',
                  marker='o', s=50, alpha=0.7)
        ax.set_xlabel('X Values', fontsize=12, fontfamily='Arial')
        ax.set_ylabel('Y Values', fontsize=12, fontfamily='Arial')
        ax.set_title('Scatter Plot Analysis', fontsize=14, fontweight='bold')
        
        # Manual file path construction and saving
        output_dir = f"data/08_reporting/{params['purpose']}"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"scatter_plot_{params['condition']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath

Migration Benefits:
    After adopting figregistry-kedro, the same functionality becomes:
    
    def create_scatter_plot(data):
        fig, ax = plt.subplots()  # No manual sizing needed
        ax.scatter(data['x'], data['y'])  # No manual styling needed
        ax.set_xlabel('X Values')  # Consistent fonts applied automatically
        ax.set_ylabel('Y Values')
        ax.set_title('Scatter Plot Analysis')
        return fig  # FigureDataSet handles saving automatically
"""

from typing import Any, Dict, Iterable, Optional

# Traditional Kedro imports - no figregistry-kedro dependencies
# This demonstrates the baseline state before lifecycle integration

# Kedro framework version compatibility for traditional workflows
# Standard Kedro requirements without figregistry-kedro enhancements
KEDRO_VERSION_REQUIRED = ">=0.18.0,<0.20.0"

# Project metadata for traditional Kedro project identification
PROJECT_NAME = "kedro_manual_example"
PROJECT_VERSION = "0.1.0"

# Traditional Kedro Hooks Configuration
# 
# In traditional Kedro projects, hooks were used for generic pipeline
# lifecycle management without specialized figure handling capabilities.
# This configuration demonstrates standard Kedro patterns before
# figregistry-kedro automated integration became available.
#
# Note: NO FigRegistryHooks registration - manual figure management required
# This highlights the absence of automated configuration initialization,
# context management, and lifecycle integration that figregistry-kedro provides.
HOOKS: Iterable[Any] = (
    # Traditional hooks might include:
    # - Custom logging hooks for pipeline monitoring
    # - Data validation hooks for input/output checking  
    # - Performance monitoring hooks for execution tracking
    # - Custom error handling hooks for failure management
    #
    # Example of traditional custom hooks (commented out for baseline):
    # CustomLoggingHook(),
    # DataValidationHook(),
    # PerformanceMonitoringHook(),
    
    # No FigRegistryHooks - demonstrates manual configuration requirement
    # Before figregistry-kedro: teams had to implement figure management
    # logic manually within each pipeline node function
)

# Traditional Session Store Configuration
# 
# Standard Kedro session store configuration without specialized
# support for automated figure styling context or configuration caching
# that figregistry-kedro hooks provide.
SESSION_STORE_CLASS = "kedro.io.MemoryDataSet"

# Session store arguments for traditional workflow patterns
SESSION_STORE_ARGS: Dict[str, Any] = {
    # Standard session configuration without FigRegistry context support
    # Manual figure management means no centralized styling context
    # Each node must handle styling configuration independently
}

# Traditional Data Catalog Configuration
#
# Standard Kedro catalog configuration demonstrating the limitations
# of manual figure management before figregistry-kedro integration.
# Notice the absence of FigureDataSet configuration and automated
# figure versioning capabilities.
CATALOG_CONFIG: Dict[str, Any] = {
    # Basic catalog versioning for data assets
    # No automated figure versioning - manual file management required
    "versioned": True,
    
    # Standard caching for data operations
    # No specialized figure dataset caching or styling context
    "enable_cache": True,
}

# Traditional Configuration Loader Settings
#
# Standard Kedro configuration patterns without FigRegistryConfigBridge
# integration. This demonstrates the manual configuration management
# burden that teams faced before automated figure styling became available.
CONFIG_LOADER_CLASS = "kedro.config.ConfigLoader"
CONFIG_LOADER_ARGS: Dict[str, Any] = {
    # Traditional configuration patterns - no figregistry.yml integration
    "config_patterns": {
        # Standard Kedro configuration files
        "catalog": ["catalog*.yml", "catalog*.yaml"],
        "parameters": ["parameters*.yml", "parameters*.yaml"],
        "credentials": ["credentials*.yml", "credentials*.yaml"],
        
        # Note: No figregistry configuration pattern
        # Manual styling requires hardcoded parameters in each node
        # or scattered configuration files without centralized management
    },
    
    # Standard environment configuration without FigRegistry integration
    "base_env": "base",
    "default_run_env": "local",
}

# Traditional Pipeline Discovery Configuration
PIPELINES_MODULE = f"{PROJECT_NAME}.pipeline_registry"

# Manual Figure Management Configuration Examples
#
# These configurations demonstrate the scattered approach teams used
# before figregistry-kedro automation became available. Notice the
# maintenance overhead and lack of centralized styling management.

# Example: Manual Styling Configuration (scattered across nodes)
MANUAL_FIGURE_STYLING = {
    # Teams had to maintain these configurations manually in each node
    # or through custom parameter files without automated condition mapping
    "default_style": {
        "figure.figsize": [10, 8],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "Arial",
    },
    
    # Manual condition-based styling required custom logic in each node
    "condition_styles": {
        "control": {
            "color": "#1f77b4",  # Blue
            "marker": "o",
            "linestyle": "-",
            "alpha": 0.7,
        },
        "treatment": {
            "color": "#ff7f0e",  # Orange
            "marker": "s",
            "linestyle": "--",
            "alpha": 0.7,
        },
    },
    
    # Manual output directory configuration
    "output_directories": {
        "exploratory": "data/08_reporting/exploratory",
        "presentation": "data/08_reporting/presentation",
        "publication": "data/08_reporting/publication",
    },
}

# Example: Manual File Management Configuration
MANUAL_FILE_MANAGEMENT = {
    # Teams had to implement custom file naming and versioning logic
    "naming_convention": "{purpose}_{condition}_{timestamp}",
    "timestamp_format": "%Y%m%d_%H%M%S",
    "supported_formats": ["png", "pdf", "svg"],
    "default_dpi": 300,
    "default_bbox_inches": "tight",
}

# Traditional Error Handling Configuration
#
# Without automated lifecycle integration, teams implemented custom
# error handling for figure generation failures throughout pipeline nodes.
MANUAL_ERROR_HANDLING = {
    "on_styling_error": "continue",  # No centralized error management
    "on_save_error": "log_and_continue",  # Manual error recovery
    "fallback_styling": "default",  # No condition-based fallback
    "max_retry_attempts": 0,  # No automated retry logic
}

# Documentation: Traditional Workflow Challenges
#
# This section documents the specific challenges that teams faced
# with manual figure management before figregistry-kedro integration.

# Challenge 1: Code Duplication
# 
# Every pipeline node that generated figures required duplicate code for:
# - matplotlib rcParams configuration
# - Styling parameter application
# - File path construction and directory creation
# - Error handling and logging
# - Format-specific save operations
#
# Example of duplicated code across nodes:
"""
def node_1_visualization(data, params):
    # Duplicate styling setup in every node
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['axes.titlesize'] = 14
    # ... more rcParams configuration
    
    fig, ax = plt.subplots()
    # ... plotting logic
    
    # Duplicate file management in every node
    output_dir = f"data/08_reporting/{params['purpose']}"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"plot1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def node_2_visualization(data, params):
    # Same styling setup duplicated
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['axes.titlesize'] = 14
    # ... identical rcParams configuration
    
    fig, ax = plt.subplots()
    # ... different plotting logic
    
    # Same file management duplicated
    output_dir = f"data/08_reporting/{params['purpose']}"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"plot2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
"""

# Challenge 2: Inconsistent Styling
#
# Without centralized configuration management, teams struggled with:
# - Different styling parameters across nodes and developers
# - Inconsistent condition-based styling implementations
# - Manual synchronization of style changes across multiple files
# - No validation for styling consistency

# Challenge 3: Maintenance Overhead
#
# Manual figure management created ongoing maintenance burden:
# - Updates required changes across multiple pipeline nodes
# - Testing required manual verification of styling consistency
# - New experimental conditions needed manual code updates
# - File organization required manual directory management

# Challenge 4: Development Friction
#
# The manual approach created development friction:
# - New team members needed training on styling conventions
# - Figure styling changes required touching multiple files
# - No separation between visualization logic and styling concerns
# - Difficult to maintain consistency across team members

# Migration Path Documentation
#
# For teams migrating from this traditional approach to figregistry-kedro:
#
# Step 1: Install figregistry-kedro package
# Step 2: Create figregistry.yml configuration file
# Step 3: Register FigRegistryHooks in settings.py
# Step 4: Replace manual plt.savefig() with FigureDataSet catalog entries
# Step 5: Remove manual styling code from pipeline nodes
# Step 6: Update catalog.yml with FigureDataSet configurations
#
# Expected Benefits:
# - 90% reduction in styling-related code lines per Section 0.1.1
# - Elimination of code duplication across pipeline nodes
# - Centralized configuration management through YAML
# - Automated condition-based styling application
# - Consistent styling across all team members and environments

# Performance Comparison
#
# Traditional Approach Performance Characteristics:
# - Manual rcParams configuration overhead in each node
# - Repeated file system operations for directory creation
# - No configuration caching between nodes
# - Manual error handling increases execution time
#
# After Migration to figregistry-kedro:
# - One-time configuration initialization per pipeline run
# - Cached styling configuration across all nodes
# - Optimized file operations through FigureDataSet
# - Reduced error handling overhead through automated fallbacks

# Summary: Traditional Configuration Limitations
#
# This traditional settings.py demonstrates the limitations of manual
# figure management that existed before figregistry-kedro integration:
#
# ❌ No automated FigRegistry initialization
# ❌ No lifecycle integration for configuration management
# ❌ No centralized styling context for pipeline nodes
# ❌ No condition-based styling automation
# ❌ No figure versioning integration with Kedro catalog
# ❌ No performance optimization for repeated styling operations
# ❌ High maintenance overhead for styling consistency
# ❌ Code duplication across pipeline nodes
# ❌ Manual error handling and fallback management
# ❌ No team-wide configuration standardization
#
# The migration to figregistry-kedro addresses all these limitations
# through automated lifecycle integration, centralized configuration
# management, and zero-touch figure styling capabilities.