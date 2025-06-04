"""
Pipeline registry for the migrated Kedro project demonstrating figregistry-kedro integration.

This module provides the standard Kedro create_pipeline() function following conventional
patterns while supporting pipelines that leverage FigureDataSet for automated figure
styling and persistence. The registry demonstrates the 'after' state of migration where
figregistry-kedro is seamlessly integrated with minimal structural changes to existing
pipeline organization.

Migration Benefits Demonstrated:
- Standard pipeline registry patterns remain unchanged during figregistry-kedro adoption
- Pipeline organization and discovery work exactly as before migration
- Automated figure management is enabled through catalog configuration, not code changes
- Zero impact on pipeline registry structure or registration conventions

Key Integration Features:
- Supports pipelines with FigureDataSet outputs for automated styling (F-005)
- Enables condition-based styling through pipeline parameter resolution (F-005-RQ-004)
- Compatible with Kedro's versioning and experiment tracking (F-005-RQ-002)
- Demonstrates converted project state per Section 0.2.1 implementation plan

Usage:
    This registry is used by Kedro's standard pipeline discovery mechanism.
    The create_pipeline() function is automatically called during project
    initialization to register all available pipelines.

    Standard Kedro Commands:
        kedro run                                    # Run all pipelines
        kedro run --pipeline=data_visualization      # Run specific pipeline
        kedro run --pipeline=dv                      # Run using short alias
        kedro run --params="experiment_condition:production"  # With parameters

Example Integration:
    After migration, the same pipeline registry patterns work unchanged, but
    now figures are automatically styled and managed through catalog configuration:

    Before Migration (manual figure management):
        - Pipeline nodes contained plt.savefig() calls
        - Styling code scattered throughout node functions
        - Manual file management and naming
        - Inconsistent figure formatting

    After Migration (automated figure management):
        - Pipeline nodes return raw matplotlib figures
        - All styling handled automatically by FigureDataSet
        - Consistent formatting based on experimental conditions
        - Zero manual figure management overhead
"""

from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines.data_visualization import (
    create_pipeline as create_data_visualization_pipeline,
    create_exploratory_pipeline,
    create_presentation_pipeline,
    create_publication_pipeline,
    create_complete_demonstration_pipeline,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the main project pipeline with figregistry-kedro automation support.
    
    This function follows standard Kedro conventions while demonstrating
    the seamless integration of figregistry-kedro for automated figure
    management. The pipeline structure and registration patterns remain
    unchanged from pre-migration state, showing minimal impact of adoption.
    
    Migration Demonstration:
    The create_pipeline() function signature and behavior are identical
    to pre-migration implementations, demonstrating that figregistry-kedro
    integration requires no changes to pipeline registry patterns or
    project organization structures.
    
    Key Automation Features Supported:
    - FigureDataSet intercepts pipeline figure outputs automatically (F-005-RQ-001)
    - Condition-based styling via catalog configuration (F-005-RQ-004) 
    - Automated versioning integration with Kedro (F-005-RQ-002)
    - Context injection for experimental condition resolution (F-005-RQ-004)
    - Complete elimination of manual plt.savefig() calls (Section 0.1.1)
    
    Pipeline Organization:
    The main pipeline combines all visualization workflows to demonstrate
    comprehensive figregistry-kedro automation across different output
    purposes and experimental conditions:
    
    1. Exploratory Analysis:
       - Rapid iteration styling for data exploration
       - Automatic application of 'exploratory' purpose styling
       - Supports fast feedback loops in research workflows
    
    2. Presentation Outputs:
       - Professional formatting for stakeholder reports
       - Consistent branding and styling across all presentations
       - Automatic resolution of 'presentation' purpose styling
    
    3. Publication Quality:
       - Academic formatting for peer review and publication
       - Compliance with journal styling requirements
       - High-resolution outputs with publication standards
    
    Configuration Integration:
    All figure styling is managed through catalog configuration rather than
    code changes, demonstrating the non-invasive nature of figregistry-kedro:
    
    ```yaml
    # conf/base/catalog.yml - Example automated figure configuration
    exploratory_scatter_plot:
      type: figregistry_kedro.FigureDataSet
      filepath: data/08_reporting/exploratory_scatter_analysis.png
      purpose: exploratory
      condition_param: experiment_condition
      style_params:
        figure.dpi: 150
        figure.facecolor: white
      save_args:
        bbox_inches: tight
        transparent: false
    ```
    
    Args:
        **kwargs: Arbitrary keyword arguments passed from Kedro framework.
                 These are preserved to maintain compatibility with existing
                 pipeline creation patterns and custom parameters.
    
    Returns:
        Pipeline: Complete project pipeline with automated figure management.
                 This pipeline includes all visualization workflows with
                 FigureDataSet automation enabled through catalog configuration.
    
    Example Usage:
        # Standard Kedro pipeline execution - no changes needed
        kedro run
        
        # Run with experimental condition parameters for styling automation
        kedro run --params="experiment_condition:production,visualization_params.theme:dark"
        
        # Run specific pipeline subset
        kedro run --pipeline=data_visualization
    """
    # Create the main data visualization pipeline with figregistry-kedro automation
    # This demonstrates the standard pipeline creation pattern unchanged by migration
    return create_data_visualization_pipeline(**kwargs)


def create_exploratory_analysis_pipeline(**kwargs) -> Pipeline:
    """
    Create exploratory analysis pipeline optimized for rapid iteration.
    
    This pipeline demonstrates automated styling specifically configured
    for exploratory data analysis workflows, showing how different styling
    purposes can be achieved through catalog configuration alone.
    
    Exploratory Analysis Benefits:
    - Fast iteration with consistent styling across all exploratory outputs
    - Automatic application of 'exploratory' purpose styling rules
    - Reduced overhead for research and investigation workflows
    - Maintains visual consistency during rapid analysis cycles
    
    Returns:
        Pipeline: Exploratory analysis pipeline with automated styling
    """
    return create_exploratory_pipeline(**kwargs)


def create_presentation_ready_pipeline(**kwargs) -> Pipeline:
    """
    Create presentation-ready pipeline for stakeholder communications.
    
    This pipeline shows how the same visualization logic produces 
    professional presentation outputs through automated styling based
    on 'presentation' purpose configuration in the catalog.
    
    Presentation Benefits:
    - Professional formatting automatically applied to all outputs
    - Consistent branding and styling across stakeholder materials
    - High-quality graphics suitable for executive presentations
    - Eliminates manual formatting for presentation preparation
    
    Returns:
        Pipeline: Presentation-ready pipeline with automated styling
    """
    return create_presentation_pipeline(**kwargs)


def create_publication_quality_pipeline(**kwargs) -> Pipeline:
    """
    Create publication-quality pipeline for academic outputs.
    
    This pipeline demonstrates the highest level of automated styling
    for academic publication requirements, showing how figregistry-kedro
    can automatically apply journal-specific formatting rules.
    
    Publication Benefits:
    - Automatic compliance with academic publication standards
    - High-resolution outputs suitable for peer review
    - Consistent formatting across all manuscript figures
    - Reproducible figure generation for review processes
    
    Returns:
        Pipeline: Publication-quality pipeline with automated styling
    """
    return create_publication_pipeline(**kwargs)


def create_demonstration_pipeline(**kwargs) -> Pipeline:
    """
    Create comprehensive demonstration pipeline showing all automation features.
    
    This meta-pipeline combines all automation capabilities to provide a
    complete demonstration of figregistry-kedro integration benefits within
    a single workflow execution.
    
    Comprehensive Features:
    - Multiple output purposes (exploratory, presentation, publication)
    - Condition-based styling through parameter resolution
    - Automated versioning integration with Kedro
    - Complete elimination of manual figure management
    - Full separation of visualization logic and styling concerns
    
    Returns:
        Pipeline: Complete demonstration of all automation features
    """
    return create_complete_demonstration_pipeline(**kwargs)


# Standard Kedro pipeline registry following conventional patterns
# This registry demonstrates that figregistry-kedro integration requires
# no changes to existing pipeline discovery and registration mechanisms
PIPELINE_REGISTRY = {
    # Main pipeline combining all visualization workflows
    "__default__": create_pipeline,
    
    # Primary data visualization pipeline with figregistry-kedro automation
    "data_visualization": create_pipeline,
    "dv": create_pipeline,  # Short alias for command-line convenience
    
    # Specialized pipeline variants for different output purposes
    "exploratory": create_exploratory_analysis_pipeline,
    "presentation": create_presentation_ready_pipeline, 
    "publication": create_publication_quality_pipeline,
    
    # Comprehensive demonstration pipeline
    "demo": create_demonstration_pipeline,
    "demonstration": create_demonstration_pipeline,
    "complete": create_demonstration_pipeline,
}


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Register all project pipelines for Kedro discovery.
    
    This function provides the standard pipeline registry that Kedro uses
    to discover and execute pipelines. The implementation follows conventional
    Kedro patterns while supporting figregistry-kedro automation features.
    
    Migration Compatibility:
    The register_pipelines() function signature and return type are identical
    to pre-migration implementations, demonstrating that adopting figregistry-kedro
    requires no changes to existing pipeline registration patterns or project
    organization structures.
    
    Registry Features:
    - Standard "__default__" pipeline for default execution
    - Named pipelines for targeted execution via --pipeline parameter
    - Short aliases for command-line convenience
    - Specialized variants demonstrating different automation features
    
    Automation Integration:
    All registered pipelines support figregistry-kedro automation through:
    - FigureDataSet configurations in the catalog
    - Automatic style application based on experimental conditions
    - Context injection for parameter-based styling resolution
    - Seamless integration with Kedro's versioning system
    
    Returns:
        Dict[str, Pipeline]: Mapping of pipeline names to Pipeline objects.
                           Standard Kedro registry format supporting automated
                           figure management through catalog configuration.
    
    Example Usage:
        # Execute default pipeline with automation
        kedro run
        
        # Execute specific pipeline with experimental conditions
        kedro run --pipeline=data_visualization --params="experiment_condition:production"
        
        # Execute exploratory analysis subset
        kedro run --pipeline=exploratory
        
        # Execute presentation-ready outputs
        kedro run --pipeline=presentation
        
        # Execute complete demonstration
        kedro run --pipeline=demo --params="experiment_condition:demo"
    """
    # Build registry using standard Kedro patterns
    # This approach ensures compatibility with existing Kedro tooling
    # while enabling figregistry-kedro automation features
    registry = {}
    
    # Register each pipeline with proper instantiation
    for name, pipeline_func in PIPELINE_REGISTRY.items():
        registry[name] = pipeline_func()
    
    return registry


# Export for clean imports and standard Kedro discovery
# This maintains compatibility with existing import patterns
__all__ = [
    "create_pipeline",
    "register_pipelines", 
    "create_exploratory_analysis_pipeline",
    "create_presentation_ready_pipeline",
    "create_publication_quality_pipeline", 
    "create_demonstration_pipeline",
]