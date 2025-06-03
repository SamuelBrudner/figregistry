"""Traditional Kedro pipeline registry demonstrating manual figure management approaches.

This module provides the standard create_pipeline() function following conventional Kedro 
patterns without figregistry-kedro integration. It registers pipelines that use manual 
figure management with scattered plt.savefig() calls, hardcoded styling, and fragmented 
configuration, showing the baseline pipeline structure before automated figure styling 
and lifecycle integration capabilities.

This serves as the "before" state in the migration example, demonstrating the manual 
overhead and code duplication that figregistry-kedro eliminates through automated 
condition-based styling and integrated lifecycle management.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from kedro_manual_example.pipelines.data_visualization import create_pipeline as create_data_viz_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines following traditional Kedro conventions.
    
    This function demonstrates the standard pipeline registration pattern used in 
    Kedro projects before figregistry-kedro integration. It registers individual 
    pipelines that handle figure management manually through plt.savefig() calls 
    scattered throughout pipeline nodes.
    
    The registered pipelines showcase:
    - Manual figure styling with hardcoded parameters
    - Scattered plt.savefig() calls in individual node functions  
    - Fragmented styling configuration across multiple files
    - Manual file path management and naming conventions
    - Code duplication patterns for styling across different nodes
    - Lack of systematic condition-based styling management
    
    This traditional approach requires significant manual intervention and maintenance,
    which figregistry-kedro eliminates through automated styling and lifecycle hooks.
    
    Returns:
        Dict[str, Pipeline]: Dictionary mapping pipeline names to Pipeline objects.
            Contains the following pipelines:
            - "data_visualization": Main visualization pipeline demonstrating manual 
              figure management patterns with hardcoded styling and scattered save calls
            - "__default__": Default pipeline combining all registered pipelines
              for standard Kedro execution without automated figure management
              
    Example:
        >>> pipelines = register_pipelines()
        >>> viz_pipeline = pipelines["data_visualization"]
        >>> # Pipeline contains nodes with manual plt.savefig() calls
        >>> # No automated styling or condition-based resolution
        >>> # Requires manual configuration of colors, markers, etc.
    """
    
    # Create the data visualization pipeline using traditional manual approaches
    # This pipeline demonstrates the problematic patterns that figregistry-kedro solves:
    # - Manual plt.savefig() calls in each node function
    # - Hardcoded styling parameters throughout the codebase  
    # - Fragmented configuration requiring updates in multiple locations
    # - Code duplication for styling logic across different visualizations
    # - Manual file path construction and naming management
    data_visualization_pipeline = create_data_viz_pipeline()
    
    return {
        "data_visualization": data_visualization_pipeline,
        "__default__": data_visualization_pipeline,
    }


# Alternative function name for backwards compatibility with older Kedro versions
# This demonstrates the traditional pipeline registry patterns that existed before
# figregistry-kedro provided automated lifecycle integration
create_pipeline = register_pipelines