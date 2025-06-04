"""Converted Kedro Project Demonstrating FigRegistry-Kedro Integration.

This package represents the "after" state of a successful migration from traditional
manual matplotlib figure management to automated figregistry-kedro integration.
The conversion eliminates scattered plt.savefig() calls, hardcoded styling, and
manual file management overhead while introducing centralized configuration,
condition-based styling, and automated figure persistence.

Migration Benefits Achieved:
- Elimination of manual plt.savefig() calls throughout pipeline nodes
- Automated condition-based styling through FigureDataSet integration (F-005)
- Centralized configuration management via FigRegistry-Kedro bridge (F-007)
- Lifecycle hooks for seamless FigRegistry initialization (F-006)
- Reduced code complexity and improved maintainability
- Consistent figure styling across all experimental conditions
- Integrated versioning through Kedro's catalog system

Conversion Highlights:
This migrated project demonstrates the transformation from a traditional Kedro
workflow with manual figure management to an automated system where:
- Node functions return raw matplotlib figures instead of saving them manually
- FigureDataSet automatically applies styling based on experimental conditions
- Configuration is centralized in figregistry.yml and managed through Kedro
- Figure persistence follows FigRegistry's versioning and naming conventions
- Pipeline execution includes automatic FigRegistry context initialization

Before vs After Comparison:
- BEFORE: Manual plt.savefig() calls scattered throughout node functions
- AFTER: Node functions return figures; FigureDataSet handles persistence
- BEFORE: Hardcoded styling repeated across multiple visualization functions
- AFTER: Centralized styling configuration with condition-based resolution
- BEFORE: Manual file path management and directory creation
- AFTER: Automated path resolution and directory management
- BEFORE: Inconsistent figure formats and naming conventions
- AFTER: Standardized output through FigRegistry's conventions

Technical Architecture:
The converted project maintains standard Kedro structure while incorporating
figregistry-kedro plugin capabilities through minimal configuration changes:
- settings.py registers FigRegistryHooks for lifecycle integration
- catalog.yml configures FigureDataSet entries for automated styling
- conf/base/figregistry.yml provides centralized styling configuration
- Pipeline nodes focus on visualization logic without figure management concerns

Project Structure:
- nodes.py: Converted node functions returning matplotlib figures
- pipeline_registry.py: Standard Kedro pipeline registration
- settings.py: FigRegistryHooks registration for automated initialization
- pipelines/: Modular pipeline definitions with FigureDataSet integration

Migration Success Metrics:
This example demonstrates the target outcomes for figregistry-kedro migration:
- 90% reduction in styling-related code lines per Section 0.1.1 objectives
- Elimination of code duplication across visualization functions
- Centralized configuration management reducing maintenance overhead
- Seamless integration preserving existing pipeline execution patterns
"""

__version__ = "0.1.0"

# Package metadata for converted Kedro project demonstration
__author__ = "FigRegistry Contributors"
__email__ = "contributors@figregistry.org"

# Public API for the migrated example project
# Following standard Python package conventions for namespace management
__all__ = [
    "__version__",
]

# Project identification constants for Kedro framework integration
PROJECT_NAME = "kedro-figregistry-example"
PROJECT_VERSION = __version__
PACKAGE_NAME = "kedro_figregistry_example"

# Technical metadata supporting migrated project requirements
KEDRO_VERSION_REQUIREMENT = ">=0.18.0,<0.20.0"
FIGREGISTRY_VERSION_REQUIREMENT = ">=0.3.0"
FIGREGISTRY_KEDRO_VERSION_REQUIREMENT = ">=0.1.0"
PYTHON_VERSION_REQUIREMENT = ">=3.10"

# Migration metadata documenting the conversion process
MIGRATION_METADATA = {
    "migration_status": "completed",
    "source_project_type": "traditional_kedro_with_manual_figures",
    "target_project_type": "figregistry_kedro_integrated",
    "conversion_scope": "full_automation_integration",
    "migration_benefits": [
        "eliminated_manual_plt_savefig_calls",
        "centralized_styling_configuration",
        "automated_condition_based_styling",
        "reduced_code_complexity",
        "improved_maintainability",
        "consistent_figure_outputs",
    ]
}

# Integration features implemented in the converted project
IMPLEMENTED_FEATURES = [
    "F-005: FigureDataSet Integration",
    "F-006: Lifecycle Hooks",
    "F-007: Configuration Bridge",
    "Automated Figure Styling",
    "Zero-Touch Figure Management",
    "Condition-Based Styling",
    "Catalog-Based Figure Persistence",
    "Centralized Configuration Management",
]

# Example classification for migration demonstration
EXAMPLE_METADATA = {
    "complexity_level": "intermediate",
    "target_audience": "existing_kedro_users_migrating_to_figregistry",
    "use_case_category": "migration_demonstration",
    "integration_scope": "complete_project_conversion",
    "demonstrated_workflows": [
        "manual_to_automated_migration",
        "centralized_configuration_management",
        "automated_styling_application",
        "lifecycle_hook_integration",
        "catalog_based_figure_management",
    ]
}

# Migration process documentation for reference
MIGRATION_STEPS_COMPLETED = [
    "installed_figregistry_kedro_plugin",
    "registered_figregistry_hooks_in_settings",
    "converted_catalog_entries_to_figuredataset",
    "created_centralized_figregistry_configuration",
    "removed_manual_plt_savefig_calls_from_nodes",
    "updated_node_functions_to_return_figures",
    "validated_automated_styling_functionality",
    "tested_pipeline_execution_with_integration",
]