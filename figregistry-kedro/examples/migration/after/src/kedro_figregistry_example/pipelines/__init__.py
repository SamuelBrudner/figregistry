"""Pipeline registry for the migrated Kedro project with figregistry-kedro integration.

This module demonstrates the 'after' state of migration where pipelines leverage
figregistry-kedro for automated figure styling and management. The pipeline
structure remains unchanged, enabling seamless migration from manual plt.savefig()
calls to automated catalog-driven figure management.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from .data_visualization import create_pipeline as create_data_visualization_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines for Kedro discovery.
    
    This function provides the pipeline registry that Kedro uses to discover
    and execute pipelines. In the migration example, this demonstrates how
    existing pipeline registration patterns remain unchanged when adopting
    figregistry-kedro integration.
    
    Returns:
        Dict[str, Pipeline]: Mapping of pipeline names to Pipeline objects.
            The "__default__" pipeline combines all individual pipelines,
            while named pipelines can be executed independently via
            `kedro run --pipeline <name>`.
    """
    # Individual pipelines for modular execution
    data_viz_pipeline = create_data_visualization_pipeline()
    
    # Combined pipeline registry following Kedro conventions
    return {
        "__default__": data_viz_pipeline,
        "data_visualization": data_viz_pipeline,
        "dv": data_viz_pipeline,  # Short alias for command-line convenience
    }


# Re-export for clean imports
__all__ = ["register_pipelines"]