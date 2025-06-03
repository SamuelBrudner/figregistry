"""Kedro pipeline registry for the figregistry-kedro converted project.

This module provides the standard create_pipeline() function following conventional Kedro 
patterns enhanced with figregistry-kedro integration. It registers pipelines that leverage 
automated figure styling and versioning through FigureDataSet, demonstrating the converted 
state where manual plt.savefig() calls have been eliminated in favor of catalog-driven 
figure management with zero-touch condition-based styling.

This serves as the "after" state in the migration example, showcasing how figregistry-kedro 
integration requires minimal structural changes to existing pipeline registries while 
providing comprehensive automated figure management capabilities through the catalog system.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from kedro_figregistry_example.pipelines.data_visualization import (
    create_pipeline as create_data_viz_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines with figregistry-kedro integration support.
    
    This function demonstrates the converted pipeline registration pattern after 
    figregistry-kedro integration. The structure remains identical to traditional 
    Kedro patterns, showcasing the minimal migration overhead while enabling 
    automated figure styling and lifecycle management through the plugin system.
    
    The registered pipelines now showcase:
    - Automated figure styling through FigureDataSet catalog configuration
    - Zero manual plt.savefig() calls - all handled by catalog persistence
    - Condition-based styling via catalog parameters (purpose, condition_param)
    - Integrated versioning with Kedro's catalog versioning system
    - Systematic styling consistency across all pipeline figure outputs
    - Automated lifecycle management through FigRegistryHooks integration
    
    This converted approach eliminates manual styling overhead while providing 
    enterprise-grade figure management automation with minimal code changes from 
    the traditional manual approach shown in the migration "before" example.
    
    Key Integration Benefits Demonstrated:
    - Pipeline nodes focus purely on analysis logic without figure management
    - Catalog configuration drives all styling decisions through figregistry.yaml
    - Automatic application of condition-based styling per experimental parameters
    - Seamless versioning integration with pipeline execution tracking
    - Environment-specific styling support through Kedro's configuration layers
    - Zero code duplication for styling logic across different visualizations
    
    Returns:
        Dict[str, Pipeline]: Dictionary mapping pipeline names to Pipeline objects.
            Contains the following pipelines with automated figure management:
            - "data_visualization": Converted visualization pipeline demonstrating 
              automated figure styling through FigureDataSet catalog integration
              with condition-based styling and zero manual plt.savefig() calls
            - "__default__": Default pipeline combining all registered pipelines
              for standard Kedro execution with comprehensive figure automation
              
    Example:
        >>> # Migration from manual to automated approach
        >>> pipelines = register_pipelines()
        >>> viz_pipeline = pipelines["data_visualization"]
        >>> # Pipeline nodes output raw matplotlib figures
        >>> # FigureDataSet automatically applies styling and saves
        >>> # No manual configuration or plt.savefig() calls required
        >>> # Condition-based styling applied via catalog parameters
        
    Configuration Integration:
        The pipeline registry operates seamlessly with figregistry-kedro's 
        configuration bridge, which merges:
        - Traditional figregistry.yaml styling definitions
        - Kedro project configuration from conf/base/figregistry.yml  
        - Environment-specific overrides for development/staging/production
        - Catalog-level parameters for dataset-specific styling control
        
    Lifecycle Integration:
        FigRegistryHooks automatically initialize during pipeline startup:
        - before_pipeline_run: Establishes merged configuration context
        - Dataset save operations: Apply automatic styling and persistence
        - Error handling: Graceful fallback with diagnostic information
        - after_pipeline_run: Clean up resources and log summary statistics
    """
    
    # Create the converted data visualization pipeline with figregistry-kedro integration
    # This pipeline demonstrates the "after" state benefits:
    # - Automated FigureDataSet styling through catalog configuration
    # - Zero manual plt.savefig() calls in pipeline node implementations
    # - Condition-based styling through pipeline parameter resolution
    # - Integrated versioning with Kedro's catalog versioning system
    # - Systematic styling consistency without code duplication
    # - Environment-aware configuration through Kedro's config management
    data_visualization_pipeline = create_data_viz_pipeline()
    
    return {
        "data_visualization": data_visualization_pipeline,
        "__default__": data_visualization_pipeline,
    }


# Alternative function name for backwards compatibility with older Kedro versions
# This maintains standard Kedro conventions while enabling figregistry-kedro automation
# Demonstrates that migration preserves existing patterns and naming conventions
create_pipeline = register_pipelines