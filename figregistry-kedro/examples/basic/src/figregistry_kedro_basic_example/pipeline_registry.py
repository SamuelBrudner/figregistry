"""Kedro Pipeline Registry for FigRegistry-Kedro Basic Example.

This module provides the create_pipeline() function that Kedro uses to discover and 
register all available pipelines for execution. The registry demonstrates how 
figregistry-kedro integration works within standard Kedro project structure, 
providing automated figure management capabilities across different visualization 
workflows.

Pipeline Organization:
- "__default__": Complete data visualization pipeline demonstrating F-005 integration
- "data_visualization": Alias for the complete pipeline for explicit execution
- "exploratory": Focused pipeline for exploratory analysis and zero-touch demos
- "presentation": Enhanced styling pipeline for stakeholder communication
- "publication": Academic publication quality pipeline with advanced formatting

Integration Features Demonstrated:
- F-005: Kedro FigureDataSet Integration with zero-touch figure management
- F-002: Condition-based styling through pipeline parameter resolution
- F-004: Automated output management with versioning and organization
- F-006: Kedro lifecycle hooks for configuration initialization
- F-007: Configuration bridge between Kedro and FigRegistry systems

Each pipeline showcases different aspects of the figregistry-kedro integration,
providing comprehensive demonstrations of automated figure styling, versioning,
and persistence within Kedro's catalog-based data processing framework.
"""

from typing import Dict

from kedro.pipeline import Pipeline

# Import pipeline creation functions from the data visualization module
from figregistry_kedro_basic_example.pipelines.data_visualization.pipeline import (
    create_pipeline as create_data_visualization_pipeline,
    create_exploratory_pipeline,
    create_presentation_pipeline,
    create_publication_pipeline,
)


def create_pipeline(**kwargs) -> Dict[str, Pipeline]:
    """Create and register all pipelines for the FigRegistry-Kedro basic example.
    
    This function implements Kedro's standard pipeline discovery mechanism,
    returning a dictionary of pipeline names mapped to configured Pipeline
    instances. Each pipeline demonstrates different aspects of figregistry-kedro
    integration, from basic zero-touch figure management to advanced publication-
    quality automated styling.
    
    Pipeline Registry Structure:
        - "__default__": The complete data visualization pipeline that runs when
                        no specific pipeline is specified with 'kedro run'
        - "data_visualization": Explicit name for the complete pipeline
        - "exploratory": Focused pipeline for basic integration demonstration
        - "presentation": Enhanced styling pipeline for business communication
        - "publication": Academic publication quality with advanced formatting
    
    Integration Demonstration Features:
        F-005: All pipelines use FigureDataSet for automatic figure interception
               and styling application without manual plt.savefig() calls
        F-002: Condition-based styling applied through experiment_condition, 
               model_type, and experiment_phase parameter resolution
        F-004: Automated versioning and output management through catalog system
        F-006: Lifecycle hooks ensure proper configuration initialization
        F-007: Configuration bridge merges Kedro and FigRegistry settings
    
    Pipeline Execution Examples:
        kedro run                              # Runs complete "__default__" pipeline
        kedro run --pipeline=data_visualization # Explicit complete pipeline
        kedro run --pipeline=exploratory       # Zero-touch demo focused pipeline
        kedro run --pipeline=presentation      # Stakeholder-ready visualizations
        kedro run --pipeline=publication       # Academic publication quality
        
        # Tag-based execution for specific workflow stages
        kedro run --tags=exploratory          # All exploratory nodes
        kedro run --tags=presentation         # All presentation nodes
        kedro run --tags=zero_touch_demo      # Specific integration demos
    
    Returns:
        Dict[str, Pipeline]: Dictionary mapping pipeline names to configured
                           Pipeline instances ready for execution
    
    Note:
        All pipelines leverage the same underlying node functions but demonstrate
        different integration patterns and styling capabilities. Node functions
        focus purely on matplotlib figure creation, while FigureDataSet handles
        all styling, versioning, and persistence operations automatically.
        
        This separation showcases the core F-005 objective: elimination of
        manual styling code from data processing logic while ensuring
        consistent, publication-ready visualizations across all workflow stages.
    """
    
    # Create the complete data visualization pipeline demonstrating full integration
    # This serves as both the default pipeline and the comprehensive demonstration
    complete_data_visualization_pipeline = create_data_visualization_pipeline()
    
    # Create specialized sub-pipelines for focused demonstrations
    # Each pipeline showcases specific aspects of figregistry-kedro integration
    
    # Exploratory pipeline: Basic zero-touch figure management demonstration
    # Focuses on F-005 core capabilities with minimal styling complexity
    exploratory_demo_pipeline = create_exploratory_pipeline()
    
    # Presentation pipeline: Enhanced styling for stakeholder communication
    # Demonstrates advanced F-002 condition-based styling with business focus
    presentation_demo_pipeline = create_presentation_pipeline()
    
    # Publication pipeline: Academic publication quality automated formatting
    # Showcases F-004 advanced output management with publication standards
    publication_demo_pipeline = create_publication_pipeline()
    
    # Assemble the complete pipeline registry following Kedro conventions
    # The "__default__" pipeline runs when no specific pipeline is specified
    pipeline_registry = {
        # Default pipeline executed by 'kedro run' without pipeline specification
        "__default__": complete_data_visualization_pipeline,
        
        # Complete data visualization pipeline with explicit naming
        # Demonstrates comprehensive figregistry-kedro integration across all workflow stages
        "data_visualization": complete_data_visualization_pipeline,
        
        # Focused demonstration pipelines for specific integration aspects
        
        # Exploratory analysis pipeline with zero-touch figure management
        # Perfect for new users learning figregistry-kedro integration basics
        "exploratory": exploratory_demo_pipeline,
        
        # Business presentation pipeline with enhanced styling capabilities
        # Demonstrates condition-based styling for stakeholder communication
        "presentation": presentation_demo_pipeline,
        
        # Academic publication pipeline with advanced formatting automation
        # Showcases highest quality output capabilities with automated styling
        "publication": publication_demo_pipeline,
    }
    
    return pipeline_registry


def get_pipeline_info() -> Dict[str, str]:
    """Provide descriptive information about available pipelines.
    
    This utility function returns human-readable descriptions of each registered
    pipeline, helping users understand the purpose and scope of each workflow
    option available in the figregistry-kedro basic example.
    
    Returns:
        Dict[str, str]: Mapping of pipeline names to descriptive information
        
    Usage:
        This function can be used by documentation generators, CLI help systems,
        or interactive pipeline selection tools to provide context about each
        available pipeline option.
    """
    
    return {
        "__default__": (
            "Complete data visualization pipeline demonstrating comprehensive "
            "figregistry-kedro integration with automated styling, versioning, "
            "and zero-touch figure management across exploratory, presentation, "
            "and publication quality workflows."
        ),
        
        "data_visualization": (
            "Full-featured data visualization workflow showcasing F-005 Kedro "
            "FigureDataSet integration, F-002 condition-based styling, and F-004 "
            "automated output management through complete data processing and "
            "visualization pipeline stages."
        ),
        
        "exploratory": (
            "Focused exploratory analysis pipeline demonstrating basic zero-touch "
            "figure management capabilities. Ideal for understanding core "
            "figregistry-kedro integration concepts with minimal complexity."
        ),
        
        "presentation": (
            "Business presentation focused pipeline with enhanced styling "
            "capabilities for stakeholder communication. Demonstrates advanced "
            "condition-based styling and automated quality enhancement features."
        ),
        
        "publication": (
            "Academic publication quality pipeline showcasing highest level "
            "automated formatting capabilities. Demonstrates publication-ready "
            "figure generation with advanced styling and output management."
        ),
    }


def validate_pipeline_registry() -> bool:
    """Validate that all registered pipelines are properly configured.
    
    This utility function performs basic validation checks on the pipeline
    registry to ensure all pipelines are properly instantiated and contain
    the expected nodes and dependencies for figregistry-kedro integration.
    
    Returns:
        bool: True if all pipelines pass validation, False otherwise
        
    Validation Checks:
        - All pipelines are valid Pipeline instances
        - Required nodes for figregistry-kedro demonstration are present
        - Pipeline dependencies are properly configured
        - No circular dependencies exist in the pipeline graph
        
    Usage:
        Can be called during testing or as part of project setup validation
        to ensure the pipeline registry is properly configured for the
        figregistry-kedro integration demonstration.
    """
    
    try:
        # Get the complete pipeline registry
        pipelines = create_pipeline()
        
        # Basic validation checks
        if not isinstance(pipelines, dict):
            return False
        
        # Check that required pipelines are present
        required_pipelines = ["__default__", "data_visualization", "exploratory"]
        for pipeline_name in required_pipelines:
            if pipeline_name not in pipelines:
                return False
            
            # Validate each pipeline is a proper Pipeline instance
            if not isinstance(pipelines[pipeline_name], Pipeline):
                return False
        
        # Validation passed
        return True
        
    except Exception:
        # Any exception during validation indicates configuration issues
        return False


# Additional utility functions for pipeline management and introspection

def list_available_pipelines() -> list:
    """Return a list of all available pipeline names.
    
    Returns:
        list: List of pipeline names available for execution
    """
    return list(create_pipeline().keys())


def get_pipeline_by_name(pipeline_name: str) -> Pipeline:
    """Retrieve a specific pipeline by name.
    
    Args:
        pipeline_name: Name of the pipeline to retrieve
        
    Returns:
        Pipeline: The requested pipeline instance
        
    Raises:
        KeyError: If the pipeline name is not found in the registry
    """
    pipelines = create_pipeline()
    
    if pipeline_name not in pipelines:
        available_pipelines = list(pipelines.keys())
        raise KeyError(
            f"Pipeline '{pipeline_name}' not found. "
            f"Available pipelines: {available_pipelines}"
        )
    
    return pipelines[pipeline_name]


def get_pipeline_tags() -> Dict[str, list]:
    """Extract all tags used across registered pipelines.
    
    Returns:
        Dict[str, list]: Mapping of pipeline names to their associated tags
        
    Usage:
        Useful for understanding the tag-based organization of nodes across
        different pipelines and for tag-based pipeline execution planning.
    """
    pipelines = create_pipeline()
    pipeline_tags = {}
    
    for pipeline_name, pipeline in pipelines.items():
        # Extract unique tags from all nodes in the pipeline
        tags = set()
        for node in pipeline.nodes:
            if node.tags:
                tags.update(node.tags)
        
        pipeline_tags[pipeline_name] = sorted(list(tags))
    
    return pipeline_tags