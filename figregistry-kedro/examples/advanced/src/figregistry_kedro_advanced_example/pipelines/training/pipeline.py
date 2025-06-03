"""Advanced ML Training Pipeline with Sophisticated FigRegistry Integration.

This module implements a comprehensive Kedro pipeline demonstrating sophisticated automated 
figure management for machine learning training workflows. The pipeline showcases advanced 
figregistry-kedro integration patterns that eliminate manual figure styling while providing 
enterprise-grade visualization automation for complex experimental conditions.

Key Advanced Features Demonstrated:
- Complex experimental condition resolution for training scenarios per F-002 requirements
- Zero-touch figure management through FigureDataSet integration per F-005 specifications  
- Advanced condition-based styling for multiple training variables per F-002-RQ-002
- Sophisticated training visualizations (loss curves, accuracy plots, hyperparameter analysis)
- Integration with Kedro's versioning system for reproducible training artifacts per F-005-RQ-002
- Multi-variable experimental condition mapping with wildcard support per F-002 requirements
- Production-ready training workflow patterns suitable for enterprise ML operations

The pipeline demonstrates elimination of manual plt.savefig() calls throughout training nodes,
enabling data scientists to focus on model development while ensuring consistent, publication-ready
training visualizations across all experimental conditions per Section 0.1.1 objectives.
"""

import logging
from typing import Any, Dict, List, Optional

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

# Import sophisticated training node functions
from ...nodes import (
    create_training_loss_visualization,
    create_model_performance_comparison,
    create_ab_test_comparison_report,
    create_inference_results_analysis
)

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """Create sophisticated ML training pipeline with advanced figregistry-kedro integration.
    
    This function implements a comprehensive training pipeline that demonstrates sophisticated
    automated figure management through figregistry-kedro integration. The pipeline showcases
    advanced experimental condition resolution, zero-touch figure styling, and enterprise-grade
    visualization automation for complex ML training workflows.
    
    The pipeline eliminates manual plt.savefig() calls from all training nodes, instead relying
    on FigureDataSet to automatically apply condition-based styling and manage figure persistence
    based on sophisticated experimental conditions including model architecture, hyperparameter
    settings, treatment groups, and environment configurations per F-005 requirements.
    
    Args:
        **kwargs: Additional pipeline configuration parameters
        
    Returns:
        Pipeline: Complete training pipeline with sophisticated figregistry-kedro integration
            demonstrating advanced automated figure management for ML training workflows
            
    Note:
        The pipeline demonstrates F-005 feature requirements through:
        - FigureDataSet integration for automated styling per F-005-RQ-001
        - Kedro versioning compatibility per F-005-RQ-002  
        - Advanced condition-based styling per F-005-RQ-004
        - Complex experimental condition resolution per F-002-RQ-002
        
        All catalog entries use figregistry_kedro.FigureDataSet with sophisticated
        purpose categorization, condition parameter resolution, and style parameter
        overrides that demonstrate production-ready automated figure management.
    """
    logger.info("Creating advanced ML training pipeline with sophisticated figregistry-kedro integration")
    
    # Define sophisticated training pipeline with advanced automated figure management
    training_pipeline = pipeline([
        
        # Advanced Training Loss Analysis with Complex Experimental Conditions
        node(
            func=create_training_loss_visualization,
            inputs=[
                "training_history",           # Training metrics DataFrame with epoch-level data
                "validation_history",        # Validation metrics DataFrame with performance tracking
                "model_config",              # Model architecture and hyperparameter configuration
                "params:experiment_params"   # Advanced experimental condition parameters
            ],
            outputs="training_loss_visualization",  # FigureDataSet with advanced condition-based styling
            name="generate_training_loss_visualization",
            tags=["training", "visualization", "loss_analysis", "figregistry_integration"]
        ),
        
        # Sophisticated Model Architecture Comparison with Multi-Variable Conditions
        node(
            func=create_model_performance_comparison,
            inputs=[
                "model_comparison_results",   # List of model performance dictionaries
                "benchmark_baseline_data",   # Benchmark performance DataFrame for comparison
                "params:comparison_config",  # Comparison configuration with evaluation criteria
                "params:experiment_params"   # Experimental parameters for advanced condition resolution
            ],
            outputs="model_performance_comparison",  # FigureDataSet with sophisticated styling
            name="generate_model_performance_comparison", 
            tags=["training", "model_comparison", "performance_analysis", "figregistry_integration"]
        ),
        
        # Advanced Hyperparameter Optimization Analysis with Treatment Groups
        node(
            func=create_ab_test_comparison_report,
            inputs=[
                "hyperparameter_control_results",    # Control group hyperparameter results
                "hyperparameter_treatment_results",  # Treatment group hyperparameter results  
                "params:hyperparameter_experiment_config", # A/B test configuration for hyperparameters
                "params:experiment_params"           # Advanced experimental condition parameters
            ],
            outputs="hyperparameter_optimization_analysis", # FigureDataSet with treatment-based styling
            name="generate_hyperparameter_optimization_analysis",
            tags=["training", "hyperparameter_optimization", "ab_testing", "figregistry_integration"]
        ),
        
        # Sophisticated Training Progress Monitoring with Experimental Condition Mapping
        node(
            func=create_training_loss_visualization,
            inputs=[
                "epoch_training_metrics",    # Detailed epoch-level training metrics
                "epoch_validation_metrics",  # Detailed epoch-level validation metrics
                "params:advanced_model_config", # Advanced model configuration with complex parameters
                "params:experiment_params"   # Experimental parameters for condition resolution
            ],
            outputs="training_progress_monitoring", # FigureDataSet with progress-specific styling
            name="generate_training_progress_monitoring",
            tags=["training", "progress_monitoring", "real_time_analysis", "figregistry_integration"]
        ),
        
        # Advanced Cross-Validation Analysis with Multiple Experimental Variables
        node(
            func=create_inference_results_analysis,
            inputs=[
                "cross_validation_predictions", # Cross-validation prediction results
                "cross_validation_ground_truth", # Cross-validation ground truth labels
                "params:cross_validation_metadata", # Cross-validation experimental metadata
                "params:experiment_params"      # Advanced experimental condition parameters
            ],
            outputs="cross_validation_analysis", # FigureDataSet with cross-validation styling
            name="generate_cross_validation_analysis",
            tags=["training", "cross_validation", "model_evaluation", "figregistry_integration"]
        ),
        
        # Sophisticated Learning Curve Analysis with Complex Condition Resolution
        node(
            func=create_model_performance_comparison,
            inputs=[
                "learning_curve_results",    # Learning curve performance data across sample sizes
                "learning_curve_baselines",  # Baseline learning curve data for comparison
                "params:learning_curve_config", # Learning curve analysis configuration
                "params:experiment_params"   # Experimental parameters for advanced styling
            ],
            outputs="learning_curve_analysis",   # FigureDataSet with learning curve styling
            name="generate_learning_curve_analysis",
            tags=["training", "learning_curves", "sample_efficiency", "figregistry_integration"]
        ),
        
        # Advanced Feature Importance Analysis with Treatment-Based Styling
        node(
            func=create_inference_results_analysis,
            inputs=[
                "feature_importance_scores",  # Feature importance analysis results
                "feature_importance_metadata", # Feature importance experimental metadata
                "params:feature_analysis_config", # Feature analysis configuration
                "params:experiment_params"    # Advanced experimental condition parameters
            ],
            outputs="feature_importance_analysis", # FigureDataSet with feature-based styling
            name="generate_feature_importance_analysis",
            tags=["training", "feature_analysis", "interpretability", "figregistry_integration"]
        ),
        
        # Sophisticated Training Convergence Analysis with Multi-Variable Conditions
        node(
            func=create_training_loss_visualization,
            inputs=[
                "convergence_training_data",  # Training convergence analysis data
                "convergence_validation_data", # Validation convergence analysis data
                "params:convergence_model_config", # Convergence analysis model configuration
                "params:experiment_params"    # Experimental parameters for condition resolution
            ],
            outputs="training_convergence_analysis", # FigureDataSet with convergence styling
            name="generate_training_convergence_analysis",
            tags=["training", "convergence_analysis", "optimization_analysis", "figregistry_integration"]
        ),
        
        # Advanced Ensemble Training Analysis with Complex Experimental Design
        node(
            func=create_model_performance_comparison,
            inputs=[
                "ensemble_training_results",  # Ensemble model training results
                "ensemble_baseline_data",     # Ensemble baseline comparison data
                "params:ensemble_config",     # Ensemble training configuration
                "params:experiment_params"    # Advanced experimental condition parameters
            ],
            outputs="ensemble_training_analysis", # FigureDataSet with ensemble-specific styling
            name="generate_ensemble_training_analysis",
            tags=["training", "ensemble_methods", "advanced_modeling", "figregistry_integration"]
        ),
        
        # Sophisticated Training Resource Utilization Analysis with Environment Conditions
        node(
            func=create_ab_test_comparison_report,
            inputs=[
                "resource_control_metrics",   # Control group resource utilization metrics
                "resource_treatment_metrics", # Treatment group resource utilization metrics
                "params:resource_experiment_config", # Resource utilization experiment configuration
                "params:experiment_params"    # Advanced experimental condition parameters
            ],
            outputs="training_resource_analysis", # FigureDataSet with resource-based styling
            name="generate_training_resource_analysis",
            tags=["training", "resource_optimization", "efficiency_analysis", "figregistry_integration"]
        )
        
    ], namespace="training")
    
    # Log advanced pipeline configuration details
    pipeline_nodes = len(training_pipeline.nodes)
    figregistry_outputs = sum(1 for node in training_pipeline.nodes 
                             if any("visualization" in output or "analysis" in output 
                                   for output in node.outputs))
    
    logger.info(
        f"Created sophisticated training pipeline: {pipeline_nodes} nodes, "
        f"{figregistry_outputs} FigureDataSet outputs with advanced figregistry-kedro integration"
    )
    
    # Add comprehensive experimental condition metadata for advanced styling
    pipeline_metadata = {
        "pipeline_type": "ml_training",
        "figregistry_integration_level": "advanced", 
        "experimental_conditions": [
            "model_architecture",      # Primary condition for architecture-based styling
            "treatment_group",         # Treatment/control experimental design
            "environment",             # Development/staging/production environments
            "hyperparameter_set",      # Hyperparameter configuration conditions
            "optimization_strategy",   # Optimizer and learning rate conditions
            "data_augmentation",       # Data augmentation experimental conditions
            "regularization_method",   # Regularization technique conditions
            "ensemble_strategy"        # Ensemble method experimental conditions
        ],
        "advanced_features": [
            "zero_touch_figure_management",     # Eliminates manual plt.savefig() calls
            "condition_based_styling",          # Automatic styling based on experimental conditions
            "kedro_versioning_integration",     # Seamless integration with Kedro versioning
            "multi_variable_condition_resolution", # Complex condition parameter mapping
            "enterprise_grade_visualization",   # Production-ready visualization patterns
            "automated_style_inheritance"       # Hierarchical style resolution
        ],
        "catalog_integration": {
            "dataset_type": "figregistry_kedro.FigureDataSet",
            "automated_styling": True,
            "condition_resolution": "advanced",
            "versioning_support": True,
            "purpose_categorization": ["exploratory", "presentation", "publication"],
            "style_override_capability": True
        }
    }
    
    # Store metadata for pipeline introspection and documentation
    training_pipeline.metadata = pipeline_metadata
    
    logger.info(
        f"Advanced training pipeline configured with sophisticated experimental conditions: "
        f"{', '.join(pipeline_metadata['experimental_conditions'])}"
    )
    
    return training_pipeline


def create_training_comparison_pipeline(**kwargs) -> Pipeline:
    """Create advanced training comparison pipeline for A/B testing training strategies.
    
    This supplementary pipeline demonstrates sophisticated A/B testing patterns for training
    strategies, showcasing advanced experimental design with complex condition resolution
    and automated figure management through figregistry-kedro integration.
    
    Args:
        **kwargs: Additional pipeline configuration parameters
        
    Returns:
        Pipeline: Training comparison pipeline with advanced A/B testing patterns
        
    Note:
        Demonstrates advanced F-002-RQ-002 wildcard and partial matching capabilities
        with complex experimental condition hierarchies for training strategy comparison.
    """
    logger.info("Creating advanced training comparison pipeline for A/B testing training strategies")
    
    comparison_pipeline = pipeline([
        
        # Advanced Training Strategy A/B Testing with Complex Conditions
        node(
            func=create_ab_test_comparison_report,
            inputs=[
                "training_strategy_control",   # Control training strategy results
                "training_strategy_treatment", # Treatment training strategy results
                "params:training_strategy_experiment", # Training strategy experiment configuration
                "params:experiment_params"     # Advanced experimental condition parameters
            ],
            outputs="training_strategy_comparison", # FigureDataSet with strategy-based styling
            name="generate_training_strategy_comparison",
            tags=["training_comparison", "strategy_testing", "advanced_ab_testing", "figregistry_integration"]
        ),
        
        # Sophisticated Optimizer Comparison with Multi-Variable Analysis
        node(
            func=create_model_performance_comparison,
            inputs=[
                "optimizer_comparison_data",   # Multi-optimizer performance comparison
                "optimizer_baseline_metrics",  # Baseline optimizer performance data
                "params:optimizer_comparison_config", # Optimizer comparison configuration
                "params:experiment_params"     # Advanced experimental condition parameters
            ],
            outputs="optimizer_performance_comparison", # FigureDataSet with optimizer-based styling
            name="generate_optimizer_performance_comparison",
            tags=["training_comparison", "optimizer_analysis", "performance_comparison", "figregistry_integration"]
        ),
        
        # Advanced Learning Rate Schedule Comparison with Treatment Groups
        node(
            func=create_ab_test_comparison_report,
            inputs=[
                "learning_rate_control_results",   # Control learning rate schedule results
                "learning_rate_treatment_results", # Treatment learning rate schedule results
                "params:learning_rate_experiment", # Learning rate experiment configuration
                "params:experiment_params"         # Advanced experimental condition parameters
            ],
            outputs="learning_rate_schedule_comparison", # FigureDataSet with learning rate styling
            name="generate_learning_rate_schedule_comparison",
            tags=["training_comparison", "learning_rate_analysis", "schedule_optimization", "figregistry_integration"]
        )
        
    ], namespace="training_comparison")
    
    logger.info(f"Created training comparison pipeline with {len(comparison_pipeline.nodes)} advanced A/B testing nodes")
    
    return comparison_pipeline


def create_training_monitoring_pipeline(**kwargs) -> Pipeline:
    """Create real-time training monitoring pipeline with advanced visualization automation.
    
    This pipeline demonstrates real-time training monitoring patterns with sophisticated
    condition-based styling for live training visualization, showcasing advanced figregistry-kedro
    integration for continuous training workflows.
    
    Args:
        **kwargs: Additional pipeline configuration parameters
        
    Returns:
        Pipeline: Real-time training monitoring pipeline with automated visualization
        
    Note:
        Demonstrates advanced real-time figure management with dynamic condition resolution
        suitable for production training monitoring systems per F-005 requirements.
    """
    logger.info("Creating real-time training monitoring pipeline with advanced visualization automation")
    
    monitoring_pipeline = pipeline([
        
        # Real-Time Training Loss Monitoring with Dynamic Conditions
        node(
            func=create_training_loss_visualization,
            inputs=[
                "real_time_training_metrics",  # Live training metrics stream
                "real_time_validation_metrics", # Live validation metrics stream
                "params:monitoring_model_config", # Real-time monitoring configuration
                "params:experiment_params"      # Advanced experimental condition parameters
            ],
            outputs="real_time_training_monitor", # FigureDataSet with real-time styling
            name="generate_real_time_training_monitor",
            tags=["training_monitoring", "real_time_analysis", "live_visualization", "figregistry_integration"]
        ),
        
        # Advanced Training Anomaly Detection with Experimental Conditions
        node(
            func=create_inference_results_analysis,
            inputs=[
                "training_anomaly_predictions", # Training anomaly detection results
                "training_anomaly_ground_truth", # Training anomaly ground truth labels
                "params:anomaly_detection_metadata", # Anomaly detection experimental metadata
                "params:experiment_params"       # Advanced experimental condition parameters
            ],
            outputs="training_anomaly_analysis", # FigureDataSet with anomaly-based styling
            name="generate_training_anomaly_analysis",
            tags=["training_monitoring", "anomaly_detection", "quality_assurance", "figregistry_integration"]
        ),
        
        # Sophisticated Training Health Monitoring with Multi-Variable Conditions
        node(
            func=create_model_performance_comparison,
            inputs=[
                "training_health_metrics",     # Training health monitoring data
                "training_health_baselines",   # Training health baseline data
                "params:health_monitoring_config", # Health monitoring configuration
                "params:experiment_params"      # Advanced experimental condition parameters
            ],
            outputs="training_health_analysis", # FigureDataSet with health-based styling
            name="generate_training_health_analysis",
            tags=["training_monitoring", "health_monitoring", "system_diagnostics", "figregistry_integration"]
        )
        
    ], namespace="training_monitoring")
    
    logger.info(f"Created training monitoring pipeline with {len(monitoring_pipeline.nodes)} real-time visualization nodes")
    
    return monitoring_pipeline


def create_complete_training_pipeline(**kwargs) -> Pipeline:
    """Create complete comprehensive training pipeline combining all advanced training workflows.
    
    This master function combines all training pipeline components to create a comprehensive
    ML training workflow that demonstrates the full scope of figregistry-kedro integration
    capabilities with sophisticated experimental condition handling and automated figure management.
    
    Args:
        **kwargs: Additional pipeline configuration parameters
        
    Returns:
        Pipeline: Complete comprehensive training pipeline demonstrating advanced
            figregistry-kedro integration across all training workflow aspects
            
    Note:
        This pipeline serves as the primary demonstration of F-005 feature capabilities,
        showcasing enterprise-grade automated figure management for complex ML training
        workflows with zero-touch styling and sophisticated experimental condition resolution.
        
        The complete pipeline eliminates all manual plt.savefig() calls from training
        workflows while providing consistent, publication-ready visualizations across
        all experimental conditions per Section 0.1.1 primary objectives.
    """
    logger.info("Creating complete comprehensive training pipeline with full figregistry-kedro integration")
    
    # Combine all training pipeline components for comprehensive demonstration
    complete_pipeline = (
        create_pipeline(**kwargs) +
        create_training_comparison_pipeline(**kwargs) + 
        create_training_monitoring_pipeline(**kwargs)
    )
    
    # Add comprehensive metadata for the complete training pipeline
    complete_metadata = {
        "pipeline_scope": "comprehensive_ml_training",
        "figregistry_integration": "complete_advanced_demonstration",
        "total_nodes": len(complete_pipeline.nodes),
        "figregistry_datasets": sum(1 for node in complete_pipeline.nodes 
                                   if any("visualization" in output or "analysis" in output or "monitor" in output
                                         for output in node.outputs)),
        "experimental_design_patterns": [
            "single_variable_conditions",      # Basic condition-based styling
            "multi_variable_conditions",       # Complex condition combinations
            "hierarchical_conditions",         # Nested condition resolution
            "wildcard_pattern_matching",       # F-002-RQ-002 wildcard support
            "treatment_group_analysis",        # A/B testing patterns
            "real_time_condition_resolution",  # Dynamic condition updating
            "environment_specific_styling"     # Environment-based styling
        ],
        "advanced_features_demonstrated": [
            "zero_touch_figure_management",           # Core F-005 objective
            "automated_condition_based_styling",      # F-002 requirements
            "kedro_versioning_integration",           # F-005-RQ-002 compliance
            "sophisticated_experimental_design",     # F-002-RQ-002 patterns
            "enterprise_grade_visualization",        # Production-ready patterns
            "comprehensive_error_handling",          # Robust error management
            "performance_optimized_styling",         # <5% overhead per F-005 specs
            "thread_safe_pipeline_execution"        # Concurrent pipeline support
        ],
        "catalog_integration_capabilities": {
            "purpose_categories": ["exploratory", "presentation", "publication"],
            "condition_parameter_resolution": "advanced_multi_variable",
            "style_parameter_overrides": "comprehensive_customization",
            "versioning_support": "full_kedro_compatibility",
            "automated_directory_management": "intelligent_path_resolution",
            "format_support": ["PNG", "PDF", "SVG", "EPS"],
            "performance_optimization": "sub_5_percent_overhead"
        }
    }
    
    # Store comprehensive metadata
    complete_pipeline.metadata = complete_metadata
    
    # Log comprehensive pipeline statistics
    total_nodes = complete_metadata["total_nodes"]
    figregistry_outputs = complete_metadata["figregistry_datasets"]
    coverage_percentage = (figregistry_outputs / total_nodes) * 100 if total_nodes > 0 else 0
    
    logger.info(
        f"Created complete training pipeline: {total_nodes} total nodes, "
        f"{figregistry_outputs} FigureDataSet outputs ({coverage_percentage:.1f}% automation coverage)"
    )
    
    logger.info(
        f"Advanced features demonstrated: {', '.join(complete_metadata['advanced_features_demonstrated'])}"
    )
    
    return complete_pipeline