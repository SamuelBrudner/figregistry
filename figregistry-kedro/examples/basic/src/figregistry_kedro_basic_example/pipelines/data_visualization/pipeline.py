"""Data Visualization Pipeline - FigRegistry-Kedro Integration Demo.

This pipeline demonstrates the complete figregistry-kedro integration workflow,
showcasing automated figure styling, condition-based visualization, and zero-touch
figure management within Kedro's catalog-based data processing framework.

Key Integration Features Demonstrated:
- F-005: Automated figure styling and persistence via FigureDataSet integration
- F-002: Condition-based styling through pipeline parameter resolution
- F-005-RQ-001: Zero-touch figure processing eliminating manual plt.savefig() calls
- F-005-RQ-002: Kedro versioning integration for figure outputs
- F-005-RQ-004: Context injection for conditional styling via pipeline parameters
- F-004: Multiple output purposes (exploratory, presentation, publication)

Pipeline Architecture:
1. Data Generation and Processing: Creates synthetic datasets and trains models
2. Exploratory Visualization: Raw data analysis with automated styling
3. Presentation Visualizations: Stakeholder-ready figures with enhanced quality
4. Publication Figures: Academic publication quality with automated formatting

All figure outputs use FigureDataSet for automatic styling application based on
experimental conditions resolved from pipeline parameters, demonstrating the
target 90% reduction in styling code while ensuring consistent, publication-ready
visualizations across all workflow stages.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
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


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data visualization pipeline demonstrating figregistry-kedro integration.
    
    This pipeline serves as the primary demonstration of F-005 requirements, implementing
    a comprehensive workflow that showcases automated figure styling, condition-based
    visualization, and seamless catalog integration through FigureDataSet.
    
    Pipeline Design Principles:
    - Nodes focus purely on data processing and plot creation logic
    - Raw matplotlib Figure objects returned to catalog without styling
    - FigureDataSet automatically applies FigRegistry styling during save operations
    - Condition parameters resolved from pipeline context (parameters.yml)
    - Output quality and format automatically adjusted based on purpose configuration
    
    Returns:
        Pipeline: Configured Kedro pipeline with connected nodes demonstrating
                 zero-touch figure management and automated styling workflows
    
    Integration Features:
        - Elimination of manual plt.savefig() calls per F-005 objectives
        - Condition-based styling via experiment_condition parameter resolution
        - Automatic style application based on experimental conditions per F-002
        - Different output purposes (exploratory, presentation, publication) per F-004
        - Kedro versioning integration for reproducible figure outputs per F-005-RQ-002
    """
    
    # =============================================================================
    # Data Processing Workflow
    # =============================================================================
    
    data_processing_pipeline = pipeline([
        # Generate synthetic dataset for visualization demonstrations
        node(
            func=generate_synthetic_data,
            inputs="params:data_generation",
            outputs="raw_synthetic_data",
            name="generate_synthetic_data_node",
            tags=["data_generation", "synthetic"]
        ),
        
        # Prepare training data with feature engineering
        node(
            func=prepare_training_data,
            inputs=["raw_synthetic_data", "params:data_processing"],
            outputs="processed_training_data",
            name="prepare_training_data_node",
            tags=["data_processing", "feature_engineering"]
        ),
        
        # Train machine learning model for visualization demonstrations
        node(
            func=train_model,
            inputs=["processed_training_data", "params:model_training"],
            outputs="trained_model_results",
            name="train_model_node",
            tags=["model_training", "machine_learning"]
        ),
        
        # Validate model outputs for FigRegistry integration
        node(
            func=validate_node_outputs,
            inputs="trained_model_results",
            outputs=None,
            name="validate_model_outputs_node",
            tags=["validation", "quality_assurance"]
        )
    ])
    
    # =============================================================================
    # Exploratory Visualization Workflow - Zero-Touch Figure Management
    # =============================================================================
    
    exploratory_pipeline = pipeline([
        # Create exploratory data visualization demonstrating automated styling
        # Note: No manual styling or save operations - handled by FigureDataSet
        node(
            func=create_exploratory_data_plot,
            inputs=["raw_synthetic_data", "params:visualization"],
            outputs="exploratory_data_figure",  # Intercepted by FigureDataSet
            name="create_exploratory_data_plot_node",
            tags=["exploratory", "visualization", "zero_touch_demo"]
        ),
        
        # Training progress visualization with experiment_phase conditioning
        node(
            func=create_training_progress_plot,
            inputs=["trained_model_results", "params:visualization"],
            outputs="training_progress_figure",  # Automatic styling via condition_param
            name="create_training_progress_plot_node",
            tags=["training", "visualization", "progress_monitoring"]
        ),
        
        # Algorithm comparison with model_type condition styling
        node(
            func=create_algorithm_comparison,
            inputs=["trained_model_results", "params:visualization"],
            outputs="algorithm_comparison_figure",  # model_type parameter drives styling
            name="create_algorithm_comparison_node",
            tags=["algorithm", "comparison", "performance_analysis"]
        ),
        
        # Simple example for documentation and tutorials
        node(
            func=create_simple_example,
            inputs=["raw_synthetic_data", "params:visualization"],
            outputs="simple_example_figure",  # Minimal styling demonstration
            name="create_simple_example_node",
            tags=["simple", "documentation", "tutorial"]
        )
    ])
    
    # =============================================================================
    # Presentation-Quality Visualization Workflow
    # =============================================================================
    
    presentation_pipeline = pipeline([
        # Validation results with presentation-quality automated styling
        # FigureDataSet applies enhanced styling for purpose: 'presentation'
        node(
            func=create_validation_results,
            inputs=["trained_model_results", "params:visualization"],
            outputs="validation_results_figure",  # Enhanced presentation styling
            name="create_validation_results_node",
            tags=["presentation", "validation", "stakeholder_ready"]
        ),
        
        # Comprehensive performance dashboard for executive presentation
        node(
            func=create_performance_dashboard,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="performance_dashboard_figure",  # Dashboard-quality styling
            name="create_performance_dashboard_node",
            tags=["dashboard", "performance", "executive_summary"]
        ),
        
        # Treatment group comparison with condition-based styling
        node(
            func=create_treatment_comparison,
            inputs=["processed_training_data", "trained_model_results", "params:visualization"],
            outputs="treatment_comparison_figure",  # Treatment-specific styling
            name="create_treatment_comparison_node",
            tags=["treatment", "comparison", "experimental_analysis"]
        )
    ])
    
    # =============================================================================
    # Publication-Quality Visualization Workflow
    # =============================================================================
    
    publication_pipeline = pipeline([
        # Primary manuscript figure with publication-quality formatting
        # FigureDataSet applies academic publication styling automatically
        node(
            func=create_manuscript_figure_1,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="manuscript_figure_1",  # Publication-quality formatting
            name="create_manuscript_figure_1_node",
            tags=["publication", "manuscript", "academic"]
        ),
        
        # Supplementary analysis with experiment_phase styling
        node(
            func=create_supplementary_analysis,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="supplementary_analysis_figure",  # Phase-specific styling
            name="create_supplementary_analysis_node",
            tags=["supplementary", "detailed_analysis", "academic"]
        ),
        
        # Final combined results for manuscript conclusion
        node(
            func=create_final_combined_results,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="final_combined_results_figure",  # Comprehensive publication styling
            name="create_final_combined_results_node",
            tags=["final_results", "comprehensive", "conclusion"]
        )
    ])
    
    # =============================================================================
    # Feature Showcase and Integration Demonstration
    # =============================================================================
    
    showcase_pipeline = pipeline([
        # Comprehensive feature showcase demonstrating all integration capabilities
        node(
            func=create_feature_showcase,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="feature_showcase_figure",  # All features demonstration
            name="create_feature_showcase_node",
            tags=["showcase", "integration", "comprehensive_demo"]
        )
    ])
    
    # =============================================================================
    # Complete Pipeline Assembly
    # =============================================================================
    
    # Combine all sub-pipelines into the complete data visualization workflow
    complete_pipeline = (
        data_processing_pipeline +
        exploratory_pipeline +
        presentation_pipeline +
        publication_pipeline +
        showcase_pipeline
    )
    
    return complete_pipeline


def create_exploratory_pipeline(**kwargs) -> Pipeline:
    """Create pipeline focused on exploratory analysis and zero-touch demonstrations.
    
    This sub-pipeline isolates the exploratory visualization workflow for focused
    testing and demonstration of basic figregistry-kedro integration features.
    
    Returns:
        Pipeline: Exploratory visualization pipeline with automated styling
    """
    return pipeline([
        node(
            func=generate_synthetic_data,
            inputs="params:data_generation",
            outputs="raw_synthetic_data",
            name="generate_data_for_exploration",
            tags=["data_generation"]
        ),
        
        node(
            func=create_exploratory_data_plot,
            inputs=["raw_synthetic_data", "params:visualization"],
            outputs="exploratory_figure",
            name="create_exploratory_figure",
            tags=["exploratory", "zero_touch"]
        ),
        
        node(
            func=create_simple_example,
            inputs=["raw_synthetic_data", "params:visualization"],
            outputs="simple_figure",
            name="create_simple_figure",
            tags=["simple", "demo"]
        )
    ])


def create_presentation_pipeline(**kwargs) -> Pipeline:
    """Create pipeline focused on presentation-quality visualizations.
    
    This sub-pipeline demonstrates enhanced styling capabilities for stakeholder
    communication and business presentation scenarios.
    
    Returns:
        Pipeline: Presentation-focused pipeline with enhanced styling
    """
    return pipeline([
        node(
            func=prepare_training_data,
            inputs=["raw_synthetic_data", "params:data_processing"],
            outputs="training_data",
            name="prepare_data_for_presentation",
            tags=["data_processing"]
        ),
        
        node(
            func=train_model,
            inputs=["training_data", "params:model_training"],
            outputs="model_results",
            name="train_model_for_presentation",
            tags=["model_training"]
        ),
        
        node(
            func=create_validation_results,
            inputs=["model_results", "params:visualization"],
            outputs="validation_figure",
            name="create_validation_figure",
            tags=["presentation", "validation"]
        ),
        
        node(
            func=create_performance_dashboard,
            inputs=["model_results", "training_data", "params:visualization"],
            outputs="dashboard_figure",
            name="create_dashboard_figure",
            tags=["dashboard", "performance"]
        )
    ])


def create_publication_pipeline(**kwargs) -> Pipeline:
    """Create pipeline focused on publication-quality academic figures.
    
    This sub-pipeline demonstrates the highest level of automated figure quality
    through FigRegistry's publication purpose configuration.
    
    Returns:
        Pipeline: Publication-focused pipeline with academic formatting
    """
    return pipeline([
        node(
            func=create_manuscript_figure_1,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="manuscript_main_figure",
            name="create_manuscript_main_figure",
            tags=["publication", "manuscript"]
        ),
        
        node(
            func=create_supplementary_analysis,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="manuscript_supplementary_figure",
            name="create_manuscript_supplementary_figure",
            tags=["publication", "supplementary"]
        ),
        
        node(
            func=create_final_combined_results,
            inputs=["trained_model_results", "processed_training_data", "params:visualization"],
            outputs="manuscript_final_figure",
            name="create_manuscript_final_figure",
            tags=["publication", "final"]
        )
    ])