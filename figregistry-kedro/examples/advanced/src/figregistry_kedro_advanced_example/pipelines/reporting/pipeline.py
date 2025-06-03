"""Advanced Kedro reporting pipeline with sophisticated figregistry-kedro integration.

This module demonstrates the most advanced automated figure styling patterns for enterprise 
reporting workflows per F-005 feature requirements. The pipeline showcases sophisticated 
condition-based styling for different audiences (technical, executive, publication), 
output formats (presentation, publication, exploration), and complex experimental 
scenarios with automated figure management that transforms raw analysis results into 
polished, presentation-ready visualizations without manual intervention.

Key Features Demonstrated:
- Enterprise-grade reporting visualizations with zero-touch figure management per Section 0.1.1
- Sophisticated condition-based styling for multiple audiences per F-002 and F-005-RQ-004
- Advanced experimental condition resolution with wildcard matching per F-002-RQ-002
- Complex catalog output configurations demonstrating enterprise-grade purpose categorization
- Integration with Kedro's versioning system for publication-ready outputs per F-005-RQ-002
- Elimination of manual plt.savefig() calls through automated FigureDataSet integration
- Most sophisticated experimental condition mapping for reporting scenarios
- Production-ready patterns suitable for enterprise decision-making workflows

This pipeline represents the pinnacle of figregistry-kedro integration sophistication,
providing automated figure styling and versioning that meets enterprise publication
standards across all reporting contexts.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from ...nodes import (
    create_training_loss_visualization,
    create_inference_results_analysis, 
    create_ab_test_comparison_report,
    create_model_performance_comparison
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the most sophisticated reporting pipeline with advanced figregistry-kedro integration.
    
    This pipeline demonstrates enterprise-grade automated figure styling across multiple 
    reporting scenarios with complex experimental conditions, audience-specific styling,
    and production-ready visualization management. The pipeline transforms raw analysis 
    results into polished, presentation-ready reports through zero-touch figure management.
    
    The pipeline showcases:
    - Advanced condition-based styling for different audiences and output formats
    - Sophisticated experimental condition resolution with multi-variable parameters
    - Complex FigureDataSet configurations with enterprise-grade purpose categorization
    - Integration with Kedro versioning for publication-ready figure outputs
    - Elimination of manual figure management through automated styling and versioning
    - Most sophisticated experimental condition handling per F-002-RQ-002 requirements
    
    Returns:
        Pipeline: Sophisticated reporting pipeline with automated figure management
            demonstrating the most advanced figregistry-kedro integration patterns
            
    Note:
        All nodes in this pipeline output raw matplotlib figures that are automatically
        styled and saved through FigureDataSet catalog entries. No manual plt.savefig()
        calls are required, demonstrating complete automation per Section 0.1.1 objectives.
        
        The pipeline demonstrates sophisticated experimental condition resolution through
        catalog parameters that map to complex styling conditions for different audiences,
        output formats, and publication requirements.
    """
    
    # Advanced Technical Deep-Dive Reporting Subpipeline
    # Demonstrates sophisticated condition-based styling for technical audiences
    # with complex experimental condition resolution and advanced output categorization
    technical_reporting_pipeline = pipeline(
        [
            # Training Analysis with Advanced Condition Resolution
            # Demonstrates complex experimental condition mapping with multi-variable parameters
            node(
                func=create_training_loss_visualization,
                inputs=[
                    "preprocessed_training_history",  # Model training metrics data
                    "preprocessed_validation_history",  # Validation performance data
                    "model_architecture_config",  # Model configuration parameters
                    "params:reporting_experiment_conditions"  # Advanced experimental parameters
                ],
                outputs="technical_training_analysis_figure",  # -> FigureDataSet with technical styling
                name="generate_technical_training_analysis",
                tags=["reporting", "technical", "training_analysis", "advanced_styling"]
            ),
            
            # Advanced Inference Analysis with Sophisticated Error Assessment
            # Shows complex condition-based styling for technical performance evaluation
            node(
                func=create_inference_results_analysis,
                inputs=[
                    "model_predictions_dataset",  # Prediction results with confidence scores
                    "ground_truth_labels_dataset",  # True labels for evaluation
                    "deployed_model_metadata",  # Model version and performance metadata
                    "params:reporting_experiment_conditions"  # Multi-variable experimental conditions
                ],
                outputs="technical_inference_analysis_figure",  # -> FigureDataSet with advanced technical styling
                name="generate_technical_inference_analysis", 
                tags=["reporting", "technical", "inference_analysis", "error_assessment"]
            ),
            
            # Model Performance Comparison with Enterprise-Grade Analysis
            # Demonstrates the most sophisticated experimental condition handling
            node(
                func=create_model_performance_comparison,
                inputs=[
                    "model_comparison_results",  # Comprehensive model evaluation data
                    "benchmark_performance_data",  # Industry/internal benchmarks
                    "model_selection_criteria_config",  # Evaluation criteria and weights
                    "params:reporting_experiment_conditions"  # Complex condition parameters
                ],
                outputs="technical_model_comparison_figure",  # -> FigureDataSet with sophisticated styling
                name="generate_technical_model_comparison",
                tags=["reporting", "technical", "model_comparison", "enterprise_analysis"]
            )
        ],
        namespace="technical_reporting",
        parameters={
            "reporting_experiment_conditions": "params:technical_reporting_conditions"
        }
    )
    
    # Executive Summary Reporting Subpipeline  
    # Demonstrates audience-specific styling for executive presentations
    # with sophisticated condition resolution for business intelligence reporting
    executive_reporting_pipeline = pipeline(
        [
            # Executive A/B Test Impact Analysis
            # Shows advanced condition-based styling for executive decision-making
            node(
                func=create_ab_test_comparison_report,
                inputs=[
                    "ab_test_control_results",  # Control group performance metrics
                    "ab_test_treatment_results",  # Treatment group performance metrics  
                    "ab_test_experiment_config",  # A/B test configuration and parameters
                    "params:reporting_experiment_conditions"  # Executive-focused condition parameters
                ],
                outputs="executive_ab_test_report_figure",  # -> FigureDataSet with executive styling
                name="generate_executive_ab_test_report",
                tags=["reporting", "executive", "ab_testing", "business_impact"]
            ),
            
            # Executive Performance Dashboard
            # Demonstrates complex experimental condition mapping for executive audiences
            node(
                func=create_model_performance_comparison,
                inputs=[
                    "executive_performance_summary",  # High-level performance metrics
                    "competitive_benchmark_data",  # Market comparison data
                    "executive_criteria_config",  # Executive-focused evaluation criteria
                    "params:reporting_experiment_conditions"  # Executive condition parameters
                ],
                outputs="executive_performance_dashboard_figure",  # -> FigureDataSet with business styling
                name="generate_executive_performance_dashboard",
                tags=["reporting", "executive", "dashboard", "performance_summary"]
            )
        ],
        namespace="executive_reporting", 
        parameters={
            "reporting_experiment_conditions": "params:executive_reporting_conditions"
        }
    )
    
    # Publication-Ready Research Reporting Subpipeline
    # Demonstrates the most sophisticated styling for peer-reviewed publications
    # with advanced experimental condition resolution and publication-quality output
    publication_reporting_pipeline = pipeline(
        [
            # Publication Training Results with Advanced Statistical Analysis
            # Shows sophisticated condition-based styling for academic publications
            node(
                func=create_training_loss_visualization,
                inputs=[
                    "publication_training_dataset",  # Cleaned training data for publication
                    "publication_validation_dataset",  # Validation data with statistical analysis
                    "publication_model_config",  # Publication-ready model configuration
                    "params:reporting_experiment_conditions"  # Publication condition parameters
                ],
                outputs="publication_training_results_figure",  # -> FigureDataSet with publication styling
                name="generate_publication_training_results",
                tags=["reporting", "publication", "training_results", "statistical_analysis"]
            ),
            
            # Publication Inference Analysis with Comprehensive Error Assessment
            # Demonstrates advanced condition resolution for peer-reviewed research
            node(
                func=create_inference_results_analysis,
                inputs=[
                    "publication_inference_results",  # Publication-quality inference data
                    "publication_ground_truth",  # Verified ground truth for publication
                    "publication_model_metadata",  # Comprehensive model documentation
                    "params:reporting_experiment_conditions"  # Publication-specific conditions
                ],
                outputs="publication_inference_analysis_figure",  # -> FigureDataSet with academic styling
                name="generate_publication_inference_analysis",
                tags=["reporting", "publication", "inference_analysis", "peer_review"]
            ),
            
            # Publication Comparative Analysis with Statistical Rigor
            # Shows the most sophisticated experimental condition handling for research
            node(
                func=create_model_performance_comparison,
                inputs=[
                    "publication_comparison_data",  # Comprehensive comparison dataset
                    "academic_benchmark_standards",  # Academic performance benchmarks
                    "publication_evaluation_criteria",  # Rigorous evaluation methodology
                    "params:reporting_experiment_conditions"  # Research-grade condition parameters
                ],
                outputs="publication_comparative_analysis_figure",  # -> FigureDataSet with research styling
                name="generate_publication_comparative_analysis",
                tags=["reporting", "publication", "comparative_analysis", "research_quality"]
            )
        ],
        namespace="publication_reporting",
        parameters={
            "reporting_experiment_conditions": "params:publication_reporting_conditions"
        }
    )
    
    # Multi-Audience Cross-Format Reporting Subpipeline
    # Demonstrates the most advanced experimental condition resolution patterns
    # with sophisticated wildcard matching and partial condition resolution per F-002-RQ-002
    cross_format_reporting_pipeline = pipeline(
        [
            # Cross-Audience Training Analysis with Advanced Condition Mapping
            # Shows sophisticated experimental condition wildcard matching
            node(
                func=create_training_loss_visualization,
                inputs=[
                    "preprocessed_training_history",  # Shared training data
                    "preprocessed_validation_history",  # Shared validation data
                    "model_architecture_config",  # Shared model configuration
                    "params:reporting_experiment_conditions"  # Cross-format condition parameters
                ],
                outputs="cross_format_training_analysis_figure",  # -> FigureDataSet with dynamic styling
                name="generate_cross_format_training_analysis",
                tags=["reporting", "cross_format", "training_analysis", "wildcard_conditions"]
            ),
            
            # Multi-Format A/B Test Analysis with Complex Condition Resolution
            # Demonstrates advanced experimental condition partial matching patterns
            node(
                func=create_ab_test_comparison_report,
                inputs=[
                    "ab_test_control_results",  # Control group data
                    "ab_test_treatment_results",  # Treatment group data
                    "ab_test_experiment_config",  # Experiment configuration
                    "params:reporting_experiment_conditions"  # Multi-format condition parameters
                ],
                outputs="cross_format_ab_test_report_figure",  # -> FigureDataSet with adaptive styling
                name="generate_cross_format_ab_test_report",
                tags=["reporting", "cross_format", "ab_testing", "adaptive_styling"]
            ),
            
            # Advanced Model Comparison with Sophisticated Condition Hierarchy
            # Shows the most complex experimental condition resolution patterns
            node(
                func=create_model_performance_comparison,
                inputs=[
                    "model_comparison_results",  # Comprehensive model data
                    "benchmark_performance_data",  # Benchmark comparison data
                    "model_selection_criteria_config",  # Selection criteria
                    "params:reporting_experiment_conditions"  # Hierarchical condition parameters
                ],
                outputs="cross_format_model_comparison_figure",  # -> FigureDataSet with hierarchical styling
                name="generate_cross_format_model_comparison",
                tags=["reporting", "cross_format", "model_comparison", "hierarchical_conditions"]
            )
        ],
        namespace="cross_format_reporting",
        parameters={
            "reporting_experiment_conditions": "params:cross_format_reporting_conditions"
        }
    )
    
    # Combine all reporting subpipelines into sophisticated enterprise reporting workflow
    # This demonstrates the most advanced figregistry-kedro integration patterns with
    # complex experimental condition resolution, audience-specific styling, and
    # automated figure management across multiple reporting contexts
    return (
        technical_reporting_pipeline +
        executive_reporting_pipeline + 
        publication_reporting_pipeline +
        cross_format_reporting_pipeline
    )