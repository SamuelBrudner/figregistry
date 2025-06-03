"""Pipeline Registry - FigRegistry Kedro Basic Example.

This module provides the create_pipeline() function that registers all available
pipelines for the basic example project, demonstrating automated figure styling
and management through figregistry-kedro integration.

The basic example showcases the core value proposition of zero-touch figure
management within Kedro workflows, where:
- Pipeline nodes create raw matplotlib figures without styling concerns
- FigureDataSet automatically applies condition-based styling during catalog save operations
- Different output purposes (exploratory, presentation, publication) receive appropriate styling
- Versioning and file management are handled seamlessly through Kedro's catalog system

Pipeline Architecture:
The basic example implements a single comprehensive pipeline that demonstrates
the end-to-end workflow from data generation through publication-ready
visualization output, showcasing all major integration features:

1. Data Processing Stage: Generate and prepare synthetic data for visualization
2. Model Training Stage: Train machine learning model with comprehensive metrics
3. Exploratory Analysis Stage: Create data exploration figures with zero-touch styling
4. Presentation Stage: Generate stakeholder-ready visualizations with enhanced formatting
5. Publication Stage: Produce manuscript-quality figures with academic styling

Key Integration Features Demonstrated:
- F-005: Automated figure management through FigureDataSet integration
- F-002: Condition-based styling through pipeline parameter resolution
- F-004: Purpose-driven output quality (exploratory/presentation/publication)
- F-005-RQ-001: Zero-touch figure processing eliminating manual plt.savefig() calls
- F-005-RQ-004: Context injection for conditional styling based on experimental parameters

Educational Value:
This pipeline serves as the primary learning resource for teams adopting
figregistry-kedro integration, providing clear examples of:
- How to structure pipeline nodes for automated figure management
- Configuration patterns for different visualization purposes
- Best practices for condition-based styling in data science workflows
- Integration patterns that preserve both Kedro and FigRegistry design philosophies
"""

import logging
from typing import Dict, Any

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

# Import all node functions from the nodes module
from figregistry_kedro_basic_example.nodes import (
    # Data processing nodes
    generate_synthetic_data,
    prepare_training_data,
    train_model,
    
    # Exploratory visualization nodes
    create_exploratory_data_plot,
    create_training_progress_plot,
    create_algorithm_comparison,
    
    # Presentation visualization nodes
    create_validation_results,
    create_performance_dashboard,
    create_treatment_comparison,
    
    # Publication visualization nodes
    create_manuscript_figure_1,
    create_supplementary_analysis,
    create_final_combined_results,
    
    # Utility and demonstration nodes
    create_simple_example,
    create_feature_showcase,
    validate_node_outputs
)

# Configure logging
logger = logging.getLogger(__name__)


def create_pipeline(**kwargs: Any) -> Dict[str, Pipeline]:
    """Create and register all pipelines for the basic figregistry-kedro example.
    
    This function demonstrates the standard Kedro pipeline registry pattern while
    showcasing comprehensive figregistry-kedro integration across multiple workflow
    stages. The returned dictionary contains pipelines that highlight different
    aspects of automated figure management and condition-based styling.
    
    Pipeline Organization:
    The basic example implements a modular pipeline architecture that can be
    executed as a complete workflow or individual components:
    
    - **data_processing**: Core data generation and model training
    - **data_visualization**: Complete visualization workflow demonstrating integration
    - **exploratory_analysis**: Data exploration with zero-touch styling
    - **presentation_ready**: Stakeholder visualizations with enhanced formatting
    - **publication_quality**: Academic manuscript figures with publication styling
    - **feature_demonstrations**: Comprehensive integration capability showcase
    
    Args:
        **kwargs: Additional pipeline configuration parameters (unused in basic example)
        
    Returns:
        Dictionary mapping pipeline names to Pipeline objects for Kedro execution
        
    Integration Features:
    Each pipeline component demonstrates specific figregistry-kedro capabilities:
    
    1. **Automated Styling**: All visualization nodes return raw matplotlib figures
       without manual styling, relying on FigureDataSet for automatic style application
    
    2. **Condition-Based Processing**: Pipeline parameters drive styling decisions
       through condition_param resolution in catalog configuration
    
    3. **Purpose-Driven Quality**: Different output purposes automatically receive
       appropriate formatting (exploratory: PNG/150dpi, presentation: PDF/300dpi,
       publication: EPS/600dpi with serif fonts)
    
    4. **Zero-Touch Workflow**: Complete elimination of manual plt.savefig() calls
       and file management, with all persistence handled by Kedro catalog integration
    
    5. **Versioning Integration**: Seamless compatibility with Kedro's built-in
       versioning system for experiment tracking and reproducibility
    
    Example Usage:
        # Execute complete workflow with all integration features
        kedro run --pipeline=data_visualization
        
        # Run specific stages for focused demonstration
        kedro run --pipeline=exploratory_analysis
        kedro run --pipeline=publication_quality
        
        # Execute with custom parameters for condition-based styling
        kedro run --pipeline=data_visualization --params experiment_condition:treatment_group_a
    
    Configuration Requirements:
    The pipelines require proper catalog configuration with FigureDataSet entries:
    
    ```yaml
    # conf/base/catalog.yml
    exploratory_data_plot:
      type: figregistry_kedro.datasets.FigureDataSet
      purpose: exploratory
      condition_param: experiment_condition
      
    manuscript_figure_1:
      type: figregistry_kedro.datasets.FigureDataSet
      purpose: publication
      condition_param: analysis_stage
    ```
    
    Parameter Configuration:
    Pipeline behavior is controlled through parameters.yml:
    
    ```yaml
    # conf/base/parameters.yml
    experiment_condition: "treatment_group_a"  # Controls styling themes
    experiment_phase: "training"               # Workflow stage styling
    model_type: "linear_regression"           # Algorithm-specific styling
    analysis_stage: "final_results"          # Publication phase styling
    ```
    """
    logger.info("Creating pipeline registry for figregistry-kedro basic example")
    
    # ==========================================================================
    # Data Processing Pipeline
    # ==========================================================================
    
    data_processing_pipeline = pipeline([
        node(
            func=generate_synthetic_data,
            inputs="params:all",
            outputs="raw_synthetic_data",
            name="generate_synthetic_data_node",
            tags=["data_generation", "processing"]
        ),
        node(
            func=prepare_training_data,
            inputs=["raw_synthetic_data", "params:all"],
            outputs="training_dataset",
            name="prepare_training_data_node",
            tags=["data_preparation", "processing"]
        ),
        node(
            func=train_model,
            inputs=["training_dataset", "params:all"],
            outputs="model_results",
            name="train_model_node",
            tags=["model_training", "processing"]
        ),
        node(
            func=validate_node_outputs,
            inputs="model_results",
            outputs="validation_status",
            name="validate_outputs_node",
            tags=["validation", "processing"]
        )
    ])
    
    # ==========================================================================
    # Exploratory Analysis Pipeline - Zero-Touch Figure Management
    # ==========================================================================
    
    exploratory_analysis_pipeline = pipeline([
        node(
            func=create_exploratory_data_plot,
            inputs=["raw_synthetic_data", "params:all"],
            outputs="exploratory_data_plot",  # FigureDataSet with purpose: exploratory
            name="create_exploratory_plot_node",
            tags=["exploratory", "visualization", "zero_touch"]
        ),
        node(
            func=create_training_progress_plot,
            inputs=["model_results", "params:all"],
            outputs="training_progress_plot",  # FigureDataSet with experiment_phase styling
            name="create_training_progress_node",
            tags=["exploratory", "visualization", "model_analysis"]
        ),
        node(
            func=create_algorithm_comparison,
            inputs=["model_results", "params:all"],
            outputs="algorithm_comparison_plot",  # FigureDataSet with model_type styling
            name="create_algorithm_comparison_node",
            tags=["exploratory", "visualization", "comparison"]
        ),
        node(
            func=create_simple_example,
            inputs=["raw_synthetic_data", "params:all"],
            outputs="simple_example_plot",  # Basic demonstration figure
            name="create_simple_example_node",
            tags=["exploratory", "visualization", "documentation"]
        )
    ])
    
    # ==========================================================================
    # Presentation-Ready Pipeline - Enhanced Quality and Formatting
    # ==========================================================================
    
    presentation_ready_pipeline = pipeline([
        node(
            func=create_validation_results,
            inputs=["model_results", "params:all"],
            outputs="validation_results_plot",  # FigureDataSet with purpose: presentation
            name="create_validation_results_node",
            tags=["presentation", "visualization", "stakeholder"]
        ),
        node(
            func=create_performance_dashboard,
            inputs=["model_results", "training_dataset", "params:all"],
            outputs="performance_dashboard_plot",  # FigureDataSet with analysis_stage styling
            name="create_performance_dashboard_node",
            tags=["presentation", "visualization", "dashboard"]
        ),
        node(
            func=create_treatment_comparison,
            inputs=["training_dataset", "model_results", "params:all"],
            outputs="treatment_comparison_plot",  # FigureDataSet with experiment_condition styling
            name="create_treatment_comparison_node",
            tags=["presentation", "visualization", "statistical"]
        )
    ])
    
    # ==========================================================================
    # Publication Quality Pipeline - Academic Manuscript Standards
    # ==========================================================================
    
    publication_quality_pipeline = pipeline([
        node(
            func=create_manuscript_figure_1,
            inputs=["model_results", "training_dataset", "params:all"],
            outputs="manuscript_figure_1",  # FigureDataSet with purpose: publication
            name="create_manuscript_figure_1_node",
            tags=["publication", "visualization", "manuscript"]
        ),
        node(
            func=create_supplementary_analysis,
            inputs=["model_results", "training_dataset", "params:all"],
            outputs="supplementary_analysis_plot",  # FigureDataSet with experiment_phase styling
            name="create_supplementary_analysis_node",
            tags=["publication", "visualization", "supplementary"]
        ),
        node(
            func=create_final_combined_results,
            inputs=["model_results", "training_dataset", "params:all"],
            outputs="final_combined_results_plot",  # FigureDataSet with analysis_stage styling
            name="create_final_results_node",
            tags=["publication", "visualization", "synthesis"]
        )
    ])
    
    # ==========================================================================
    # Feature Demonstration Pipeline - Integration Showcase
    # ==========================================================================
    
    feature_demonstrations_pipeline = pipeline([
        node(
            func=create_feature_showcase,
            inputs=["model_results", "training_dataset", "params:all"],
            outputs="feature_showcase_plot",  # Comprehensive integration demonstration
            name="create_feature_showcase_node",
            tags=["demonstration", "visualization", "showcase"]
        )
    ])
    
    # ==========================================================================
    # Complete Data Visualization Pipeline - End-to-End Workflow
    # ==========================================================================
    
    # Combine all visualization pipelines into comprehensive workflow
    complete_visualization_pipeline = (
        exploratory_analysis_pipeline +
        presentation_ready_pipeline +
        publication_quality_pipeline +
        feature_demonstrations_pipeline
    )
    
    # Add data processing dependencies to create end-to-end workflow
    data_visualization_pipeline = data_processing_pipeline + complete_visualization_pipeline
    
    # ==========================================================================
    # Pipeline Registry Dictionary
    # ==========================================================================
    
    # Create comprehensive pipeline registry demonstrating different execution patterns
    pipeline_registry = {
        # Complete end-to-end workflow demonstrating all integration features
        "__default__": data_visualization_pipeline,
        "data_visualization": data_visualization_pipeline,
        
        # Individual component pipelines for focused demonstration
        "data_processing": data_processing_pipeline,
        "exploratory_analysis": data_processing_pipeline + exploratory_analysis_pipeline,
        "presentation_ready": data_processing_pipeline + presentation_ready_pipeline,
        "publication_quality": data_processing_pipeline + publication_quality_pipeline,
        "feature_demonstrations": data_processing_pipeline + feature_demonstrations_pipeline,
        
        # Visualization-only pipelines (assuming data already exists)
        "viz_exploratory": exploratory_analysis_pipeline,
        "viz_presentation": presentation_ready_pipeline,
        "viz_publication": publication_quality_pipeline,
        "viz_showcase": feature_demonstrations_pipeline,
        "viz_all": complete_visualization_pipeline
    }
    
    # Log pipeline registration summary
    logger.info(f"Registered {len(pipeline_registry)} pipelines for basic example:")
    for pipeline_name, pipeline_obj in pipeline_registry.items():
        node_count = len(pipeline_obj.nodes)
        logger.info(f"  - {pipeline_name}: {node_count} nodes")
    
    logger.info("All pipelines demonstrate zero-touch figure management with automated styling")
    logger.info("FigureDataSet integration eliminates manual plt.savefig() calls")
    logger.info("Condition-based styling applied through pipeline parameter resolution")
    
    return pipeline_registry


def create_pipelines(**kwargs: Any) -> Dict[str, Pipeline]:
    """Alternative pipeline creation function for compatibility.
    
    Some Kedro configurations may expect a create_pipelines() function instead
    of create_pipeline(). This function provides compatibility while maintaining
    the same functionality.
    
    Args:
        **kwargs: Pipeline configuration parameters
        
    Returns:
        Dictionary of registered pipelines
    """
    logger.info("Creating pipelines via create_pipelines() compatibility function")
    return create_pipeline(**kwargs)


# =============================================================================
# Pipeline Configuration and Documentation
# =============================================================================

def get_pipeline_descriptions() -> Dict[str, str]:
    """Get descriptions of all available pipelines for documentation and CLI help.
    
    Returns:
        Dictionary mapping pipeline names to their descriptions
    """
    return {
        "__default__": "Complete figregistry-kedro integration demonstration with all features",
        "data_visualization": "End-to-end workflow from data generation to publication figures",
        "data_processing": "Data generation, preparation, and model training only", 
        "exploratory_analysis": "Data processing + exploratory visualizations with zero-touch styling",
        "presentation_ready": "Data processing + presentation-quality stakeholder visualizations",
        "publication_quality": "Data processing + manuscript-ready academic figures",
        "feature_demonstrations": "Data processing + comprehensive integration feature showcase",
        "viz_exploratory": "Exploratory visualizations only (requires existing data)",
        "viz_presentation": "Presentation visualizations only (requires existing data)",
        "viz_publication": "Publication visualizations only (requires existing data)",
        "viz_showcase": "Feature showcase visualization only (requires existing data)",
        "viz_all": "All visualization pipelines only (requires existing data)"
    }


def get_integration_features() -> Dict[str, str]:
    """Get documentation of integration features demonstrated by the pipelines.
    
    Returns:
        Dictionary mapping feature codes to descriptions
    """
    return {
        "F-005": "Automated figure management through FigureDataSet integration",
        "F-002": "Condition-based styling through pipeline parameter resolution", 
        "F-004": "Purpose-driven output quality (exploratory/presentation/publication)",
        "F-005-RQ-001": "Zero-touch figure processing eliminating manual plt.savefig() calls",
        "F-005-RQ-004": "Context injection for conditional styling based on experimental parameters",
        "F-006": "Lifecycle hooks for configuration initialization and context management",
        "F-007": "Configuration bridge between Kedro and FigRegistry systems"
    }


# Export functions for Kedro discovery
__all__ = [
    "create_pipeline",
    "create_pipelines", 
    "get_pipeline_descriptions",
    "get_integration_features"
]