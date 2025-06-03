"""Advanced Inference Pipeline for FigRegistry-Kedro Integration.

This pipeline demonstrates sophisticated ML inference workflows with comprehensive 
automated figure management through figregistry-kedro integration. It showcases 
advanced condition-based styling, complex experimental parameter resolution, and 
enterprise-grade inference visualization patterns suitable for production ML workflows.

Key Features Demonstrated:
- Advanced automated figure styling for ML inference workflows per F-005 requirements
- Elimination of manual plt.savefig() calls through FigureDataSet integration per Section 0.1.1
- Sophisticated condition-based styling for inference scenarios per F-002 and F-005-RQ-004
- Inference-specific visualizations including prediction vs actual plots, error analysis, 
  model performance comparisons, and evaluation metrics per proposed folder details
- Integration with Kedro's versioning system for inference figure outputs per F-005-RQ-002
- Zero-touch figure management in ML inference workflows per F-005 technical specifications
- Complex experimental condition handling and style resolution per F-002-RQ-002 requirements
- Multiple output purposes (exploratory inference plots, presentation inference summaries, 
  publication inference figures) per F-004 requirements

The pipeline creates a complete inference workflow that automatically generates and styles
sophisticated visualizations without any manual figure management code, demonstrating
the power of figregistry-kedro integration for enterprise ML inference pipelines.
"""

import logging
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

# Import node functions from the shared nodes module
from figregistry_kedro_advanced_example.nodes import (
    create_inference_results_analysis,
    create_model_performance_comparison,
    create_ab_test_comparison_report
)

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """Create comprehensive advanced inference pipeline with sophisticated figregistry-kedro integration.
    
    This pipeline demonstrates enterprise-grade ML inference workflows with automated
    figure management through figregistry-kedro's FigureDataSet integration. The pipeline
    showcases advanced condition-based styling, complex experimental parameter resolution,
    and sophisticated inference visualization patterns suitable for production environments.
    
    Pipeline Architecture:
    - Prediction Analysis: Comprehensive inference results visualization with statistical analysis
    - Model Comparison: Multi-algorithm performance evaluation with production readiness assessment  
    - A/B Testing: Advanced experimental comparison with business impact analysis
    - Cross-Validation: Inference performance evaluation across different data splits
    - Error Analysis: Detailed error pattern investigation with confidence assessment
    
    Advanced Features:
    - Complex experimental condition resolution with multi-variable parameters
    - Sophisticated inference-specific styling conditions (datasets, model versions, metrics)
    - Production-grade error analysis and performance profiling visualizations
    - Enterprise-level A/B testing workflows with statistical significance analysis
    - Automated figure versioning integrated with Kedro's experiment tracking
    - Zero-touch figure management eliminating all manual plt.savefig() operations
    
    Configuration Integration:
    The pipeline relies on FigureDataSet catalog entries with advanced figregistry-kedro
    parameters for automated styling and output management:
    
    ```yaml
    inference_prediction_analysis:
      type: figregistry_kedro.FigureDataSet
      filepath: data/08_reporting/inference/prediction_analysis_{inference_dataset}_{model_version}.png
      purpose: presentation
      condition_param: inference_condition
      style_params:
        figure.dpi: 300
        figure.facecolor: white
    ```
    
    Experimental Conditions:
    The pipeline demonstrates sophisticated condition resolution patterns:
    - inference_condition: Dynamic condition based on dataset, model version, and evaluation metrics
    - Model-specific styling: Different visual treatments for different algorithm types
    - Environment-aware outputs: Development vs production inference styling
    - Performance-based styling: Visual differentiation based on model performance tiers
    
    Args:
        **kwargs: Pipeline configuration parameters, including:
            - inference_datasets: List of datasets for inference evaluation
            - model_versions: List of model versions to compare
            - evaluation_metrics: Primary metrics for performance assessment
            - experimental_conditions: Complex condition mapping for styling
            - environment: Deployment environment (development, staging, production)
    
    Returns:
        Pipeline: Complete inference pipeline with automated figure management demonstrating
            advanced figregistry-kedro integration patterns per F-005 requirements.
    
    Note:
        This pipeline demonstrates zero-touch figure management per Section 0.1.1 objectives.
        All figure outputs are automatically styled and versioned through FigureDataSet
        integration without any manual styling or save operations in the node functions.
        The pipeline showcases the elimination of manual plt.savefig() calls while providing
        sophisticated, production-ready inference visualizations.
    """
    logger.info("Creating advanced inference pipeline with figregistry-kedro integration")
    
    # Define the comprehensive inference workflow nodes
    inference_nodes = [
        # Primary inference results analysis with sophisticated statistical overlays
        node(
            func=create_inference_results_analysis,
            inputs=[
                "inference_predictions",           # Model predictions with confidence scores
                "inference_ground_truth",         # True labels for evaluation  
                "inference_model_metadata",       # Model information and performance metrics
                "params:inference_experiment"     # Experimental parameters for condition resolution
            ],
            outputs="inference_prediction_analysis",  # Automatically styled through FigureDataSet
            name="generate_inference_prediction_analysis",
            tags=["inference", "analysis", "primary_evaluation"]
        ),
        
        # Advanced model performance comparison across multiple algorithms
        node(
            func=create_model_performance_comparison,
            inputs=[
                "model_comparison_results",       # Performance results from multiple models
                "inference_benchmark_data",      # Baseline/benchmark performance data
                "params:model_comparison_config", # Comparison configuration and evaluation criteria
                "params:inference_experiment"    # Experimental parameters for advanced condition resolution
            ],
            outputs="inference_model_performance_comparison", # Advanced styling via condition resolution
            name="generate_model_performance_comparison", 
            tags=["inference", "model_comparison", "performance_evaluation"]
        ),
        
        # Sophisticated A/B testing analysis for inference algorithm comparison
        node(
            func=create_ab_test_comparison_report,
            inputs=[
                "inference_control_results",     # Control group inference results
                "inference_treatment_results",   # Treatment group inference results  
                "params:ab_test_config",         # A/B test configuration and parameters
                "params:inference_experiment"   # Complex experimental condition mapping
            ],
            outputs="inference_ab_test_report",  # Enterprise-grade A/B test visualization
            name="generate_inference_ab_test_analysis",
            tags=["inference", "ab_testing", "statistical_analysis"]
        ),
        
        # Cross-validation inference performance analysis
        node(
            func=create_inference_results_analysis,
            inputs=[
                "cv_inference_predictions",      # Cross-validation predictions
                "cv_inference_ground_truth",     # Cross-validation ground truth
                "cv_model_metadata",             # Cross-validation model metadata
                "params:cv_inference_experiment" # Cross-validation specific experimental parameters
            ],
            outputs="inference_cross_validation_analysis", # CV-specific styling conditions
            name="generate_cross_validation_inference_analysis",
            tags=["inference", "cross_validation", "model_validation"]
        ),
        
        # Advanced error analysis and confidence assessment
        node(
            func=create_inference_results_analysis,
            inputs=[
                "error_analysis_predictions",    # Predictions for detailed error analysis
                "error_analysis_ground_truth",   # Ground truth for error pattern investigation  
                "error_analysis_metadata",       # Error analysis specific metadata
                "params:error_analysis_experiment" # Error analysis experimental conditions
            ],
            outputs="inference_error_analysis",  # Error-focused styling with confidence visualization
            name="generate_inference_error_analysis",
            tags=["inference", "error_analysis", "confidence_assessment"]
        ),
        
        # Production readiness assessment visualization
        node(
            func=create_model_performance_comparison,
            inputs=[
                "production_readiness_results",  # Production readiness assessment data
                "production_benchmark_data",     # Production baseline performance
                "params:production_assessment_config", # Production assessment criteria
                "params:production_experiment"   # Production environment experimental parameters
            ],
            outputs="inference_production_readiness_report", # Production-specific styling
            name="generate_production_readiness_assessment",
            tags=["inference", "production_readiness", "deployment_evaluation"]
        ),
        
        # Advanced inference performance profiling across different data segments
        node(
            func=create_inference_results_analysis,
            inputs=[
                "segmented_inference_predictions", # Predictions across different data segments
                "segmented_inference_ground_truth", # Segmented ground truth data
                "segmented_model_metadata",       # Segment-specific model metadata
                "params:segmented_inference_experiment" # Segmentation experimental parameters  
            ],
            outputs="inference_segmentation_analysis", # Segment-aware styling conditions
            name="generate_segmented_inference_analysis", 
            tags=["inference", "segmentation", "performance_profiling"]
        ),
        
        # Temporal inference performance analysis for model drift detection
        node(
            func=create_inference_results_analysis,
            inputs=[
                "temporal_inference_predictions", # Time-series inference predictions
                "temporal_inference_ground_truth", # Time-series ground truth
                "temporal_model_metadata",        # Temporal analysis metadata
                "params:temporal_inference_experiment" # Temporal analysis experimental conditions
            ],
            outputs="inference_temporal_drift_analysis", # Temporal-specific styling with drift visualization
            name="generate_temporal_inference_analysis",
            tags=["inference", "temporal_analysis", "drift_detection"]
        ),
        
        # Multi-environment inference comparison (dev/staging/prod)
        node(
            func=create_ab_test_comparison_report,
            inputs=[
                "dev_environment_results",       # Development environment inference results
                "prod_environment_results",      # Production environment inference results
                "params:environment_comparison_config", # Environment comparison configuration
                "params:environment_experiment"  # Multi-environment experimental parameters
            ],
            outputs="inference_environment_comparison", # Environment-aware styling conditions
            name="generate_environment_inference_comparison",
            tags=["inference", "environment_comparison", "deployment_validation"]
        ),
        
        # Advanced ensemble inference analysis
        node(
            func=create_model_performance_comparison,
            inputs=[
                "ensemble_inference_results",    # Ensemble model inference results
                "individual_model_results",      # Individual model comparison data
                "params:ensemble_comparison_config", # Ensemble evaluation configuration
                "params:ensemble_experiment"     # Ensemble-specific experimental parameters
            ],
            outputs="inference_ensemble_analysis", # Ensemble-specific styling with model contribution visualization
            name="generate_ensemble_inference_analysis",
            tags=["inference", "ensemble_analysis", "model_aggregation"]
        )
    ]
    
    # Create modular pipeline with comprehensive tagging for advanced workflow management
    inference_pipeline = pipeline(
        pipe=Pipeline(inference_nodes),
        inputs={
            # Primary inference data inputs
            "inference_predictions",
            "inference_ground_truth", 
            "inference_model_metadata",
            
            # Model comparison inputs
            "model_comparison_results",
            "inference_benchmark_data",
            
            # A/B testing inputs
            "inference_control_results",
            "inference_treatment_results",
            
            # Cross-validation inputs
            "cv_inference_predictions",
            "cv_inference_ground_truth",
            "cv_model_metadata",
            
            # Error analysis inputs
            "error_analysis_predictions",
            "error_analysis_ground_truth",
            "error_analysis_metadata",
            
            # Production readiness inputs
            "production_readiness_results",
            "production_benchmark_data",
            
            # Segmentation analysis inputs
            "segmented_inference_predictions",
            "segmented_inference_ground_truth", 
            "segmented_model_metadata",
            
            # Temporal analysis inputs
            "temporal_inference_predictions",
            "temporal_inference_ground_truth",
            "temporal_model_metadata",
            
            # Environment comparison inputs
            "dev_environment_results",
            "prod_environment_results",
            
            # Ensemble analysis inputs
            "ensemble_inference_results",
            "individual_model_results",
            
            # Parameter inputs for sophisticated experimental condition resolution
            "params:inference_experiment",
            "params:model_comparison_config",
            "params:ab_test_config", 
            "params:cv_inference_experiment",
            "params:error_analysis_experiment",
            "params:production_assessment_config",
            "params:production_experiment",
            "params:segmented_inference_experiment",
            "params:temporal_inference_experiment",
            "params:environment_comparison_config",
            "params:environment_experiment",
            "params:ensemble_comparison_config",
            "params:ensemble_experiment"
        },
        outputs={
            # Primary analysis outputs - all automatically styled via FigureDataSet
            "inference_prediction_analysis",           # F-005-RQ-001: FigureDataSet intercepts and styles
            "inference_model_performance_comparison",   # F-005-RQ-004: Context injection for conditional styling
            "inference_ab_test_report",                # F-002-RQ-002: Wildcard and partial matching demonstration
            "inference_cross_validation_analysis",     # F-005-RQ-002: Kedro versioning integration
            "inference_error_analysis",                # Advanced error analysis with confidence visualization
            "inference_production_readiness_report",   # Production-grade readiness assessment
            "inference_segmentation_analysis",         # Performance profiling across data segments
            "inference_temporal_drift_analysis",       # Temporal analysis with drift detection
            "inference_environment_comparison",        # Multi-environment inference comparison
            "inference_ensemble_analysis"              # Ensemble model analysis with contribution visualization
        },
        namespace="inference",
        parameters={
            # Sophisticated experimental parameter configuration for advanced condition resolution
            "inference_experiment": {
                "treatment": "inference_analysis",      # F-002 condition-based styling treatment
                "environment": "production",            # Environment-specific styling conditions
                "model_version": "v2.1.0",             # Model version for condition resolution
                "evaluation_strategy": "holdout",       # Evaluation strategy parameter
                "data_split": "test",                   # Data split for condition mapping
                "experiment_id": "inference_eval_001", # Unique experiment identifier
                "performance_tier": "high",            # Performance-based styling condition
                "confidence_threshold": 0.85,          # Confidence-based visualization parameters
                "primary_metric": "accuracy",          # Primary evaluation metric
                "visualization_mode": "comprehensive"   # Comprehensive analysis mode
            },
            
            # Model comparison configuration for sophisticated algorithm evaluation
            "model_comparison_config": {
                "evaluation_strategy": "cross_validation",
                "primary_metric": "f1_score",
                "secondary_metrics": ["precision", "recall", "auc_roc"],
                "ranking_criteria": {
                    "accuracy": 0.4,
                    "speed": 0.3, 
                    "interpretability": 0.3
                },
                "evaluation_criteria": ["accuracy", "speed", "interpretability", "robustness"]
            },
            
            # A/B testing configuration for statistical analysis
            "ab_test_config": {
                "experiment_name": "inference_algorithm_comparison",
                "primary_metric": "prediction_accuracy",
                "secondary_metrics": ["inference_time", "confidence_calibration"],
                "significance_threshold": 0.05,
                "revenue_per_conversion": 150
            },
            
            # Cross-validation experimental parameters
            "cv_inference_experiment": {
                "treatment": "cross_validation_inference",
                "environment": "validation",
                "cv_folds": 5,
                "stratification": "enabled",
                "evaluation_mode": "robust"
            },
            
            # Error analysis experimental configuration  
            "error_analysis_experiment": {
                "treatment": "error_pattern_analysis",
                "environment": "analysis",
                "confidence_analysis": "enabled",
                "error_categorization": "detailed",
                "visualization_depth": "comprehensive"
            },
            
            # Production assessment configuration
            "production_assessment_config": {
                "readiness_factors": ["performance", "stability", "scalability", "interpretability", "compliance"],
                "deployment_criteria": {
                    "min_accuracy": 0.85,
                    "max_latency": 100,
                    "stability_threshold": 0.95
                }
            },
            
            # Production environment experimental parameters
            "production_experiment": {
                "treatment": "production_readiness",
                "environment": "production",
                "assessment_mode": "comprehensive",
                "compliance_level": "enterprise"
            },
            
            # Segmented analysis experimental parameters
            "segmented_inference_experiment": {
                "treatment": "segmentation_analysis", 
                "environment": "analysis",
                "segmentation_strategy": "demographic_behavioral",
                "segment_count": 8,
                "performance_profiling": "enabled"
            },
            
            # Temporal analysis experimental parameters
            "temporal_inference_experiment": {
                "treatment": "temporal_drift_analysis",
                "environment": "monitoring",
                "drift_detection": "enabled",
                "temporal_granularity": "daily",
                "alert_thresholds": {
                    "performance_drop": 0.05,
                    "drift_score": 0.1
                }
            },
            
            # Environment comparison configuration
            "environment_comparison_config": {
                "comparison_type": "dev_vs_prod",
                "evaluation_metrics": ["accuracy", "latency", "throughput"],
                "environment_factors": ["data_quality", "infrastructure", "load_patterns"]
            },
            
            # Environment experimental parameters
            "environment_experiment": {
                "treatment": "environment_validation",
                "comparison_mode": "comprehensive",
                "deployment_validation": "enabled"
            },
            
            # Ensemble comparison configuration
            "ensemble_comparison_config": {
                "ensemble_methods": ["voting", "stacking", "blending"],
                "individual_models": ["random_forest", "gradient_boosting", "neural_network"],
                "contribution_analysis": "enabled",
                "optimization_strategy": "performance_weighted"
            },
            
            # Ensemble experimental parameters
            "ensemble_experiment": {
                "treatment": "ensemble_optimization",
                "environment": "production",
                "ensemble_strategy": "adaptive_weighted",
                "model_contribution_analysis": "detailed"
            }
        }
    )
    
    logger.info(
        f"Created advanced inference pipeline with {len(inference_nodes)} nodes "
        f"demonstrating sophisticated figregistry-kedro integration patterns"
    )
    
    # Log advanced integration features being demonstrated
    logger.info(
        "Pipeline demonstrates: "
        "F-005 FigureDataSet integration, "
        "F-002 condition-based styling, "
        "F-005-RQ-001 automatic figure interception, "
        "F-005-RQ-002 Kedro versioning integration, " 
        "F-005-RQ-004 context injection for conditional styling, "
        "F-002-RQ-002 wildcard and partial matching conditions"
    )
    
    return inference_pipeline


# Additional utility functions for advanced pipeline configuration

def create_inference_pipeline_with_custom_conditions(
    custom_conditions: dict,
    experimental_parameters: dict,
    **kwargs
) -> Pipeline:
    """Create inference pipeline with custom experimental conditions.
    
    This function demonstrates advanced customization patterns for figregistry-kedro
    integration, allowing sophisticated experimental condition resolution and 
    parameter mapping for complex ML inference workflows.
    
    Args:
        custom_conditions: Custom condition mapping for FigRegistry styling
        experimental_parameters: Advanced experimental parameter configuration
        **kwargs: Additional pipeline configuration parameters
        
    Returns:
        Pipeline: Customized inference pipeline with advanced condition resolution
    """
    # Create base pipeline
    base_pipeline = create_pipeline(**kwargs)
    
    # Apply custom experimental conditions to parameters
    custom_parameters = {}
    for param_key, param_value in experimental_parameters.items():
        if isinstance(param_value, dict):
            # Merge custom conditions into parameter dictionaries
            updated_param = {**param_value}
            for condition_key, condition_value in custom_conditions.items():
                if condition_key in updated_param:
                    updated_param[condition_key] = condition_value
            custom_parameters[param_key] = updated_param
        else:
            custom_parameters[param_key] = param_value
    
    # Create customized pipeline with advanced condition resolution
    customized_pipeline = modular_pipeline(
        pipe=base_pipeline,
        namespace="custom_inference",
        parameters=custom_parameters
    )
    
    logger.info(
        f"Created customized inference pipeline with {len(custom_conditions)} "
        f"custom conditions and {len(experimental_parameters)} experimental parameters"
    )
    
    return customized_pipeline


def get_inference_pipeline_metadata() -> dict:
    """Get comprehensive metadata about the inference pipeline configuration.
    
    This function provides detailed information about the pipeline's advanced
    figregistry-kedro integration patterns, experimental condition resolution,
    and sophisticated visualization capabilities.
    
    Returns:
        dict: Comprehensive pipeline metadata including:
            - Feature demonstrations (F-005, F-002, etc.)
            - Experimental condition patterns
            - Advanced styling capabilities
            - Integration requirements
            - Performance characteristics
    """
    return {
        "pipeline_name": "advanced_inference_pipeline",
        "figregistry_kedro_version": ">=0.1.0",
        "feature_demonstrations": {
            "F-005": "Kedro FigureDataSet Integration",
            "F-005-RQ-001": "Automatic figure interception and styling",
            "F-005-RQ-002": "Kedro versioning system integration", 
            "F-005-RQ-004": "Context injection for conditional styling",
            "F-002": "Condition-based styling with experimental parameters",
            "F-002-RQ-002": "Wildcard and partial matching for conditions",
            "F-004": "Multiple output purposes (exploratory, presentation, publication)"
        },
        "experimental_conditions": {
            "inference_analysis": "Primary inference workflow styling",
            "cross_validation_inference": "Cross-validation specific styling",
            "error_pattern_analysis": "Error analysis focused styling",
            "production_readiness": "Production assessment styling",
            "segmentation_analysis": "Data segment analysis styling",
            "temporal_drift_analysis": "Temporal analysis with drift detection",
            "environment_validation": "Multi-environment comparison styling",
            "ensemble_optimization": "Ensemble model analysis styling"
        },
        "advanced_capabilities": {
            "zero_touch_figure_management": "Complete elimination of manual plt.savefig() calls",
            "sophisticated_condition_resolution": "Multi-variable experimental parameter mapping",
            "enterprise_grade_visualizations": "Production-ready inference analysis patterns",
            "automated_versioning": "Seamless integration with Kedro experiment tracking",
            "complex_statistical_analysis": "Advanced statistical overlays and significance testing",
            "performance_profiling": "Comprehensive model performance evaluation",
            "production_readiness_assessment": "Enterprise deployment evaluation"
        },
        "integration_requirements": {
            "kedro": ">=0.18.0,<0.20.0",
            "figregistry": ">=0.3.0", 
            "matplotlib": ">=3.9.0",
            "figregistry_kedro": ">=0.1.0"
        },
        "performance_characteristics": {
            "figure_styling_overhead": "<5% compared to manual saves per F-005-RQ specifications",
            "condition_resolution_time": "<1ms per F-002-RQ requirements",
            "pipeline_initialization_overhead": "<50ms per F-006-RQ specifications"
        },
        "node_count": 10,
        "output_count": 10,
        "parameter_sets": 12,
        "experimental_conditions_count": 8,
        "tags": [
            "inference", "analysis", "primary_evaluation", "model_comparison",
            "performance_evaluation", "ab_testing", "statistical_analysis", 
            "cross_validation", "model_validation", "error_analysis",
            "confidence_assessment", "production_readiness", "deployment_evaluation",
            "segmentation", "performance_profiling", "temporal_analysis", 
            "drift_detection", "environment_comparison", "deployment_validation",
            "ensemble_analysis", "model_aggregation"
        ]
    }