"""
Advanced Inference Pipeline for FigRegistry-Kedro Integration

This module implements a sophisticated inference pipeline that demonstrates enterprise-grade
automated figure management through figregistry-kedro integration. The pipeline showcases
complex experimental condition handling, advanced ML inference visualizations, and seamless
catalog integration with zero manual figure management overhead.

Key Capabilities Demonstrated:
- Advanced automated figure styling for ML inference workflows (F-005)
- Elimination of manual plt.savefig() calls through FigureDataSet integration (Section 0.1.1)
- Sophisticated condition-based styling for inference scenarios (F-002, F-005-RQ-004)
- Inference-specific visualizations: prediction vs actual, error analysis, performance comparisons
- Integration with Kedro's versioning system for inference figure outputs (F-005-RQ-002)
- Zero-touch figure management in ML inference workflows (F-005 specifications)
- Complex experimental condition handling and style resolution (F-002-RQ-002)
- Multiple output purposes: exploratory, presentation, publication inference visualizations (F-004)

The pipeline creates a comprehensive inference workflow that processes model predictions,
generates performance analyses, and produces publication-ready visualizations without
any manual styling or save operations. All figure styling and persistence is handled
automatically by FigureDataSet based on experimental conditions and output purposes.

Architecture:
- Nodes generate matplotlib figures as outputs (no manual styling)
- FigureDataSet intercepts outputs during catalog save operations
- Configuration bridge provides merged configuration context
- Style manager applies condition-based styling before persistence
- Output manager handles file operations with FigRegistry conventions
- Versioning system tracks inference results across experimental runs

This demonstrates the pinnacle of automated visualization workflows in production
ML systems, enabling data scientists to focus on inference logic while ensuring
consistent, publication-ready visualizations across all pipeline outputs.
"""

import logging
from typing import Any, Dict, List, Optional

# Kedro pipeline imports
from kedro.pipeline import Pipeline, node

# Import advanced node functions for inference workflows
from ...nodes import (
    # Advanced inference analysis functions
    create_model_inference_analysis,
    create_ab_testing_analysis,
    create_executive_performance_dashboard,
    create_statistical_analysis_report,
    
    # Utility functions and classes
    ExperimentalConfiguration,
    validate_experimental_configuration,
    prepare_condition_hierarchy,
    calculate_enterprise_performance_metrics
)

# Configure module logger
logger = logging.getLogger(__name__)

# Enterprise-grade inference pipeline constants
INFERENCE_PIPELINE_VERSION = "2.1.0"
SUPPORTED_MODEL_TYPES = ["classification", "regression", "ensemble", "deep_learning"]
ADVANCED_CONDITION_PATTERNS = [
    "model_type", "dataset_name", "evaluation_metric", "deployment_environment",
    "performance_tier", "confidence_threshold", "prediction_horizon"
]


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create sophisticated inference pipeline demonstrating advanced figregistry-kedro integration.
    
    This function constructs a comprehensive inference Pipeline that showcases enterprise-grade
    automated figure management capabilities. The pipeline demonstrates elimination of manual
    plt.savefig() calls, sophisticated condition-based styling, and seamless integration with
    Kedro's versioning system for production ML workflows.
    
    Key Features Demonstrated:
    - Advanced condition-based styling for inference scenarios (F-002, F-005-RQ-004)
    - Automated FigureDataSet styling eliminating manual figure management (Section 0.1.1)
    - Complex experimental condition mapping and wildcard pattern matching (F-002-RQ-002) 
    - Integration with Kedro versioning for inference figure outputs (F-005-RQ-002)
    - Multiple output purposes: exploratory, presentation, publication (F-004)
    - Zero-touch figure management in ML inference workflows (F-005 specifications)
    - Sophisticated experimental condition handling and style resolution
    - Enterprise-grade inference visualization patterns with automated styling
    
    Pipeline Architecture:
    1. Model Inference Analysis Node:
       - Processes model predictions and ground truth data
       - Generates comprehensive performance visualizations
       - Demonstrates inference-specific condition styling
       
    2. Prediction Error Analysis Node:
       - Analyzes prediction errors and residual patterns
       - Creates error distribution and bias visualizations
       - Shows advanced error analysis condition mapping
       
    3. Model Performance Comparison Node:
       - Compares multiple model versions or configurations
       - Generates A/B testing analysis for model performance
       - Demonstrates complex experimental condition hierarchies
       
    4. Executive Inference Dashboard Node:
       - Creates high-level performance summaries for stakeholders
       - Shows publication-ready executive reporting capabilities
       - Demonstrates sophisticated business context styling
       
    5. Statistical Inference Report Node:
       - Generates comprehensive statistical analysis of inference results
       - Creates publication-ready statistical reporting
       - Shows academic/professional styling automation
    
    Condition-Based Styling Examples:
    - model_type=classification → categorical color schemes and confusion matrices
    - model_type=regression → continuous color maps and scatter plots
    - evaluation_metric=accuracy → performance tier color coding
    - deployment_environment=production → professional presentation styling
    - performance_tier=excellent → emphasis and highlighting patterns
    - dataset_name=customer_churn → domain-specific color palettes
    - confidence_threshold=high → alpha transparency adjustments
    
    Output Catalog Demonstrations:
    - Exploratory inference plots: Quick analysis with relaxed styling
    - Presentation inference summaries: Business-ready visualizations
    - Publication inference figures: Academic/professional publication quality
    - Versioned inference outputs: Experiment tracking and reproducibility
    
    Args:
        **kwargs: Additional pipeline configuration parameters
    
    Returns:
        Pipeline: Sophisticated inference pipeline with advanced figregistry-kedro integration
        
    Note:
        All figure outputs use FigureDataSet which automatically applies condition-based
        styling and handles persistence without any manual plt.savefig() calls in node
        functions. This demonstrates the complete elimination of manual figure management
        overhead in production ML inference workflows.
    """
    try:
        logger.info(f"Creating advanced inference pipeline v{INFERENCE_PIPELINE_VERSION}")
        
        # Extract pipeline configuration parameters
        pipeline_config = kwargs.get("pipeline_config", {})
        enable_advanced_features = pipeline_config.get("enable_advanced_features", True)
        experimental_mode = pipeline_config.get("experimental_mode", False)
        
        # Define comprehensive inference pipeline nodes with sophisticated condition mapping
        inference_nodes = [
            
            # 1. Core Model Inference Analysis
            # Demonstrates F-005-RQ-001: FigureDataSet intercepts pipeline outputs and styles
            # Showcases F-002 condition-based styling for inference scenarios
            node(
                func=create_model_inference_analysis,
                inputs=[
                    "inference_results",           # Model predictions with confidence scores
                    "ground_truth_data",          # True labels/values for evaluation  
                    "model_configuration",        # Model metadata and hyperparameters
                    "deployment_context"          # Production deployment conditions
                ],
                outputs="inference_analysis_exploratory",  # Exploratory purpose output
                name="create_inference_analysis_exploratory",
                tags=["inference", "analysis", "exploratory", "model_performance"]
            ),
            
            # 2. Advanced Prediction vs Actual Analysis
            # Demonstrates sophisticated condition resolution for regression vs classification
            # Shows F-005-RQ-004: Support context injection for conditional styling
            node(
                func=create_model_inference_analysis,
                inputs=[
                    "inference_results",
                    "ground_truth_data", 
                    "model_configuration",
                    "prediction_analysis_context"  # Advanced prediction-specific context
                ],
                outputs="prediction_accuracy_presentation",  # Presentation purpose output
                name="create_prediction_accuracy_analysis",
                tags=["inference", "prediction", "presentation", "accuracy_analysis"]
            ),
            
            # 3. Model Performance Comparison with A/B Testing
            # Demonstrates complex experimental condition hierarchies (F-002-RQ-002)
            # Shows advanced condition mapping with wildcard and partial matching
            node(
                func=create_ab_testing_analysis,
                inputs=[
                    "model_comparison_data",      # Multi-model performance results
                    "ab_testing_configuration",   # ExperimentalConfiguration object
                    "statistical_test_results",   # Hypothesis testing outcomes
                    "business_impact_metrics"     # Business KPIs and impact analysis
                ],
                outputs="model_comparison_presentation",  # Advanced presentation output
                name="create_model_comparison_analysis", 
                tags=["inference", "comparison", "ab_testing", "presentation"]
            ),
            
            # 4. Error Analysis and Residual Examination
            # Demonstrates inference-specific visualization patterns with automated styling
            # Shows F-005-RQ-002: Kedro versioning compatibility for inference outputs
            node(
                func=create_model_inference_analysis,
                inputs=[
                    "prediction_errors",          # Calculated prediction errors
                    "residual_analysis_data",     # Residual patterns and distributions
                    "error_analysis_config",      # Error analysis configuration
                    "model_diagnostics_context"   # Model diagnostic conditions
                ],
                outputs="error_analysis_publication",  # Publication quality output
                name="create_error_analysis_publication",
                tags=["inference", "error_analysis", "publication", "diagnostics"]
            ),
            
            # 5. Executive Inference Performance Dashboard
            # Demonstrates enterprise-grade visualization with business context styling
            # Shows sophisticated condition-based styling for executive audiences
            node(
                func=create_executive_performance_dashboard,
                inputs=[
                    "inference_kpi_data",         # Key performance indicators
                    "inference_trend_analysis",   # Trend analysis and forecasting
                    "comparative_model_metrics",  # Cross-model performance comparison
                    "executive_dashboard_context" # Executive presentation context
                ],
                outputs="executive_inference_dashboard",  # Executive presentation output
                name="create_executive_inference_dashboard",
                tags=["inference", "executive", "dashboard", "presentation", "kpi"]
            ),
            
            # 6. Statistical Inference Report Generation
            # Demonstrates publication-ready statistical analysis with automated formatting
            # Shows F-004 multiple output purposes with sophisticated styling adaptation
            node(
                func=create_statistical_analysis_report,
                inputs=[
                    "statistical_inference_results", # Comprehensive statistical analysis
                    "inference_dataset_metadata",    # Dataset characteristics and metadata
                    "statistical_analysis_config",   # Statistical methods configuration
                    "publication_context"            # Academic/professional publication context
                ],
                outputs="statistical_inference_report",  # Publication quality output
                name="create_statistical_inference_report",
                tags=["inference", "statistics", "publication", "academic"]
            ),
            
            # 7. Real-time Inference Monitoring Dashboard
            # Demonstrates advanced condition handling for production monitoring
            # Shows sophisticated experimental condition resolution patterns
            node(
                func=create_executive_performance_dashboard,
                inputs=[
                    "realtime_inference_metrics", # Real-time performance monitoring
                    "monitoring_trend_data",      # Time-series monitoring trends
                    "alert_threshold_metrics",    # Alerting and threshold analysis
                    "monitoring_dashboard_context" # Production monitoring context
                ],
                outputs="realtime_monitoring_exploratory",  # Operational monitoring output
                name="create_realtime_monitoring_dashboard",
                tags=["inference", "monitoring", "realtime", "exploratory", "operations"]
            ),
            
            # 8. Model Confidence and Uncertainty Analysis
            # Demonstrates advanced uncertainty visualization with condition-based styling
            # Shows sophisticated confidence interval and uncertainty quantification
            node(
                func=create_model_inference_analysis,
                inputs=[
                    "confidence_analysis_data",   # Model confidence scores and distributions
                    "uncertainty_quantification", # Uncertainty estimates and bounds
                    "confidence_analysis_config", # Confidence analysis configuration
                    "uncertainty_context"         # Uncertainty analysis conditions
                ],
                outputs="confidence_analysis_presentation",  # Advanced presentation output
                name="create_confidence_analysis",
                tags=["inference", "confidence", "uncertainty", "presentation"]
            ),
            
            # 9. Feature Importance and Explainability Analysis
            # Demonstrates inference explainability visualization with automated styling
            # Shows advanced condition mapping for different model interpretation methods
            node(
                func=create_statistical_analysis_report,
                inputs=[
                    "feature_importance_data",    # Feature importance scores and rankings
                    "explainability_metadata",   # Model explainability method metadata
                    "interpretation_config",     # Interpretation analysis configuration
                    "explainability_context"     # Model interpretation conditions
                ],
                outputs="feature_importance_publication",  # Publication quality output
                name="create_feature_importance_analysis",
                tags=["inference", "explainability", "publication", "interpretation"]
            ),
            
            # 10. Deployment Performance and Scalability Analysis
            # Demonstrates production deployment visualization with business context
            # Shows enterprise-grade performance monitoring and scaling analysis
            node(
                func=create_executive_performance_dashboard,
                inputs=[
                    "deployment_performance_data", # Production deployment metrics
                    "scalability_analysis",       # Performance scaling characteristics
                    "resource_utilization_metrics", # System resource usage analysis
                    "deployment_context"          # Production deployment conditions
                ],
                outputs="deployment_analysis_presentation",  # Business presentation output
                name="create_deployment_analysis",
                tags=["inference", "deployment", "scalability", "presentation", "performance"]
            )
        ]
        
        # Add conditional advanced nodes if experimental features are enabled
        if enable_advanced_features and experimental_mode:
            logger.info("Adding experimental advanced inference features")
            
            advanced_nodes = [
                
                # Advanced Multi-Model Ensemble Analysis
                # Demonstrates complex condition hierarchies for ensemble methods
                node(
                    func=create_ab_testing_analysis,
                    inputs=[
                        "ensemble_model_results",     # Multi-model ensemble predictions
                        "ensemble_configuration",     # Ensemble method configuration
                        "ensemble_statistical_tests", # Ensemble performance tests
                        "ensemble_business_context"   # Ensemble business impact context
                    ],
                    outputs="ensemble_analysis_publication",  # Publication quality ensemble analysis
                    name="create_ensemble_analysis",
                    tags=["inference", "ensemble", "advanced", "publication"]
                ),
                
                # Advanced Time Series Inference Analysis
                # Demonstrates temporal condition mapping for time series predictions
                node(
                    func=create_statistical_analysis_report,
                    inputs=[
                        "timeseries_inference_results", # Time series prediction results
                        "temporal_analysis_metadata",   # Temporal analysis characteristics
                        "forecasting_config",          # Forecasting method configuration
                        "temporal_context"             # Time series conditions
                    ],
                    outputs="timeseries_analysis_presentation",  # Advanced temporal analysis
                    name="create_timeseries_analysis",
                    tags=["inference", "timeseries", "forecasting", "presentation", "advanced"]
                )
            ]
            
            inference_nodes.extend(advanced_nodes)
        
        # Create sophisticated inference pipeline with comprehensive node network
        inference_pipeline = Pipeline(
            nodes=inference_nodes,
            tags=["inference", "advanced", "figregistry_integration"]
        )
        
        # Log pipeline creation success with configuration details
        node_count = len(inference_nodes)
        logger.info(
            f"Successfully created advanced inference pipeline with {node_count} nodes. "
            f"Features: advanced={enable_advanced_features}, experimental={experimental_mode}"
        )
        
        # Add pipeline metadata for comprehensive documentation
        pipeline_metadata = {
            "pipeline_version": INFERENCE_PIPELINE_VERSION,
            "node_count": node_count,
            "supported_model_types": SUPPORTED_MODEL_TYPES,
            "condition_patterns": ADVANCED_CONDITION_PATTERNS,
            "features": {
                "advanced_features": enable_advanced_features,
                "experimental_mode": experimental_mode,
                "automated_styling": True,
                "versioning_support": True,
                "condition_based_styling": True,
                "zero_touch_management": True
            },
            "figregistry_integration": {
                "f005_compliance": True,  # FigureDataSet integration
                "f002_compliance": True,  # Condition-based styling
                "f004_compliance": True,  # Multiple output purposes
                "automated_figure_management": True,
                "kedro_versioning_support": True,
                "enterprise_grade_styling": True
            }
        }
        
        logger.debug(f"Pipeline metadata: {pipeline_metadata}")
        
        return inference_pipeline
        
    except Exception as e:
        logger.error(f"Failed to create advanced inference pipeline: {e}")
        raise RuntimeError(
            f"Advanced inference pipeline creation failed: {e}. "
            "Check pipeline configuration and node function availability."
        ) from e


def create_inference_sub_pipeline(
    sub_pipeline_type: str,
    **kwargs
) -> Pipeline:
    """
    Create specialized inference sub-pipeline for specific use cases.
    
    This function demonstrates modular pipeline creation for different inference
    scenarios, showcasing how figregistry-kedro integration adapts to various
    experimental conditions and business requirements.
    
    Args:
        sub_pipeline_type: Type of sub-pipeline to create
                          Options: "model_comparison", "error_analysis", 
                                  "performance_monitoring", "explainability"
        **kwargs: Additional configuration parameters
    
    Returns:
        Pipeline: Specialized inference sub-pipeline
        
    Note:
        Each sub-pipeline demonstrates different aspects of automated figure
        management and condition-based styling for specific inference workflows.
    """
    try:
        logger.info(f"Creating inference sub-pipeline: {sub_pipeline_type}")
        
        if sub_pipeline_type == "model_comparison":
            # Model comparison sub-pipeline with advanced A/B testing
            return Pipeline([
                node(
                    func=create_ab_testing_analysis,
                    inputs=[
                        "model_comparison_data",
                        "comparison_experimental_config",
                        "comparison_statistical_tests",
                        "comparison_business_metrics"
                    ],
                    outputs="model_comparison_comprehensive",
                    name="comprehensive_model_comparison",
                    tags=["comparison", "ab_testing", "comprehensive"]
                )
            ])
            
        elif sub_pipeline_type == "error_analysis":
            # Error analysis sub-pipeline with statistical reporting
            return Pipeline([
                node(
                    func=create_statistical_analysis_report,
                    inputs=[
                        "error_statistical_analysis",
                        "error_dataset_metadata", 
                        "error_statistical_config",
                        "error_publication_context"
                    ],
                    outputs="error_analysis_comprehensive",
                    name="comprehensive_error_analysis",
                    tags=["error_analysis", "statistics", "comprehensive"]
                )
            ])
            
        elif sub_pipeline_type == "performance_monitoring":
            # Performance monitoring sub-pipeline with executive dashboards
            return Pipeline([
                node(
                    func=create_executive_performance_dashboard,
                    inputs=[
                        "monitoring_kpi_data",
                        "monitoring_trend_analysis",
                        "monitoring_comparative_metrics",
                        "monitoring_executive_context"
                    ],
                    outputs="performance_monitoring_comprehensive",
                    name="comprehensive_performance_monitoring",
                    tags=["monitoring", "executive", "comprehensive"]
                )
            ])
            
        elif sub_pipeline_type == "explainability":
            # Model explainability sub-pipeline with interpretation analysis
            return Pipeline([
                node(
                    func=create_model_inference_analysis,
                    inputs=[
                        "explainability_results",
                        "interpretation_ground_truth",
                        "explainability_model_config",
                        "interpretation_context"
                    ],
                    outputs="explainability_comprehensive",
                    name="comprehensive_explainability_analysis",
                    tags=["explainability", "interpretation", "comprehensive"]
                )
            ])
            
        else:
            raise ValueError(
                f"Unknown sub-pipeline type: {sub_pipeline_type}. "
                f"Supported types: model_comparison, error_analysis, "
                f"performance_monitoring, explainability"
            )
            
    except Exception as e:
        logger.error(f"Failed to create inference sub-pipeline '{sub_pipeline_type}': {e}")
        raise RuntimeError(
            f"Inference sub-pipeline creation failed: {e}"
        ) from e


def validate_inference_pipeline_config(config: Dict[str, Any]) -> bool:
    """
    Validate inference pipeline configuration for figregistry-kedro integration.
    
    This function demonstrates comprehensive configuration validation for advanced
    inference pipelines, ensuring proper setup for automated figure management
    and condition-based styling.
    
    Args:
        config: Pipeline configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: When configuration validation fails
    """
    try:
        logger.info("Validating inference pipeline configuration")
        
        # Validate required configuration sections
        required_sections = [
            "model_types", "condition_patterns", "output_purposes",
            "experimental_conditions", "business_context"
        ]
        
        missing_sections = [
            section for section in required_sections 
            if section not in config
        ]
        
        if missing_sections:
            raise ValueError(
                f"Missing required configuration sections: {missing_sections}"
            )
        
        # Validate model types
        supported_types = set(SUPPORTED_MODEL_TYPES)
        configured_types = set(config["model_types"])
        
        if not configured_types.issubset(supported_types):
            invalid_types = configured_types - supported_types
            raise ValueError(
                f"Unsupported model types: {invalid_types}. "
                f"Supported types: {supported_types}"
            )
        
        # Validate condition patterns
        configured_patterns = config["condition_patterns"]
        if not isinstance(configured_patterns, list):
            raise ValueError("condition_patterns must be a list")
        
        # Validate output purposes
        valid_purposes = ["exploratory", "presentation", "publication"]
        configured_purposes = config["output_purposes"]
        
        invalid_purposes = [
            purpose for purpose in configured_purposes
            if purpose not in valid_purposes
        ]
        
        if invalid_purposes:
            raise ValueError(
                f"Invalid output purposes: {invalid_purposes}. "
                f"Valid purposes: {valid_purposes}"
            )
        
        # Validate experimental conditions
        if "experimental_conditions" in config:
            experimental_config = config["experimental_conditions"]
            validate_experimental_configuration(experimental_config)
        
        logger.info("Inference pipeline configuration validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Inference pipeline configuration validation failed: {e}")
        raise ValueError(
            f"Invalid inference pipeline configuration: {e}"
        ) from e


def get_inference_pipeline_catalog_config() -> Dict[str, Any]:
    """
    Get comprehensive catalog configuration for inference pipeline outputs.
    
    This function demonstrates sophisticated catalog configuration for automated
    figure management, showcasing advanced condition-based styling parameters
    and multiple output purposes for enterprise inference workflows.
    
    Returns:
        Dict[str, Any]: Complete catalog configuration for inference pipeline
        
    Note:
        This configuration demonstrates F-005 FigureDataSet integration with
        sophisticated condition mapping and automated styling parameters.
    """
    catalog_config = {
        
        # Exploratory Inference Analysis Output
        # Demonstrates basic condition-based styling for exploratory purposes
        "inference_analysis_exploratory": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/exploratory/inference_analysis_{run_id}.png",
            "purpose": "exploratory",
            "condition_param": "model_type",
            "style_params": {
                "alpha": 0.7,
                "linewidth": 1.5,
                "grid": True
            },
            "format_kwargs": {
                "dpi": 150,
                "bbox_inches": "tight"
            },
            "versioned": True,
            "metadata": {
                "description": "Exploratory model inference analysis with automated styling",
                "tags": ["inference", "exploratory", "model_performance"]
            }
        },
        
        # Presentation Quality Prediction Accuracy Analysis
        # Demonstrates advanced condition resolution for business presentations
        "prediction_accuracy_presentation": {
            "type": "figregistry_kedro.datasets.FigureDataSet", 
            "filepath": "data/08_reporting/inference/presentation/prediction_accuracy_{run_id}.png",
            "purpose": "presentation",
            "condition_param": "evaluation_metric",
            "style_params": {
                "alpha": 0.9,
                "linewidth": 2.0,
                "professional_theme": True
            },
            "format_kwargs": {
                "dpi": 200,
                "bbox_inches": "tight",
                "facecolor": "white"
            },
            "versioned": True,
            "metadata": {
                "description": "Business presentation quality prediction accuracy analysis",
                "tags": ["inference", "presentation", "accuracy", "business"]
            }
        },
        
        # Advanced Model Comparison with A/B Testing
        # Demonstrates complex experimental condition hierarchies
        "model_comparison_presentation": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/presentation/model_comparison_{run_id}.png", 
            "purpose": "presentation",
            "condition_param": "comparison_type",
            "style_params": {
                "comparison_emphasis": True,
                "statistical_annotations": True,
                "business_impact_highlighting": True
            },
            "format_kwargs": {
                "dpi": 200,
                "bbox_inches": "tight"
            },
            "versioned": True,
            "metadata": {
                "description": "Advanced model comparison with A/B testing analysis",
                "tags": ["inference", "comparison", "ab_testing", "presentation"]
            }
        },
        
        # Publication Quality Error Analysis
        # Demonstrates publication-ready formatting with academic styling
        "error_analysis_publication": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/publication/error_analysis_{run_id}.pdf",
            "purpose": "publication", 
            "condition_param": "analysis_type",
            "style_params": {
                "publication_ready": True,
                "academic_formatting": True,
                "statistical_emphasis": True
            },
            "format_kwargs": {
                "dpi": 300,
                "bbox_inches": "tight",
                "format": "pdf"
            },
            "versioned": True,
            "metadata": {
                "description": "Publication quality error analysis for academic papers",
                "tags": ["inference", "error_analysis", "publication", "academic"]
            }
        },
        
        # Executive Inference Dashboard
        # Demonstrates enterprise-grade executive reporting with business styling
        "executive_inference_dashboard": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/executive/dashboard_{run_id}.png",
            "purpose": "presentation",
            "condition_param": "executive_audience",
            "style_params": {
                "executive_theme": True,
                "business_kpi_emphasis": True,
                "professional_branding": True
            },
            "format_kwargs": {
                "dpi": 200,
                "bbox_inches": "tight", 
                "facecolor": "white"
            },
            "versioned": True,
            "metadata": {
                "description": "Executive inference performance dashboard",
                "tags": ["inference", "executive", "dashboard", "business"]
            }
        },
        
        # Statistical Inference Report
        # Demonstrates comprehensive statistical reporting with automated formatting
        "statistical_inference_report": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/publication/statistical_report_{run_id}.pdf",
            "purpose": "publication",
            "condition_param": "statistical_method",
            "style_params": {
                "academic_style": True,
                "statistical_notation": True,
                "publication_formatting": True
            },
            "format_kwargs": {
                "dpi": 300,
                "bbox_inches": "tight",
                "format": "pdf"
            },
            "versioned": True,
            "metadata": {
                "description": "Comprehensive statistical inference analysis report",
                "tags": ["inference", "statistics", "publication", "comprehensive"]
            }
        },
        
        # Real-time Monitoring Dashboard
        # Demonstrates operational monitoring with real-time styling adaptations
        "realtime_monitoring_exploratory": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/monitoring/realtime_{run_id}.png",
            "purpose": "exploratory",
            "condition_param": "monitoring_tier",
            "style_params": {
                "realtime_theme": True,
                "alert_highlighting": True,
                "operational_emphasis": True
            },
            "format_kwargs": {
                "dpi": 150,
                "bbox_inches": "tight"
            },
            "versioned": True,
            "metadata": {
                "description": "Real-time inference monitoring dashboard",
                "tags": ["inference", "monitoring", "realtime", "operations"]
            }
        },
        
        # Advanced Confidence Analysis
        # Demonstrates uncertainty quantification visualization with sophisticated styling
        "confidence_analysis_presentation": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/presentation/confidence_analysis_{run_id}.png",
            "purpose": "presentation",
            "condition_param": "confidence_method",
            "style_params": {
                "uncertainty_emphasis": True,
                "confidence_interval_styling": True,
                "probabilistic_visualization": True
            },
            "format_kwargs": {
                "dpi": 200,
                "bbox_inches": "tight"
            },
            "versioned": True,
            "metadata": {
                "description": "Advanced model confidence and uncertainty analysis",
                "tags": ["inference", "confidence", "uncertainty", "presentation"]
            }
        },
        
        # Feature Importance Publication Analysis
        # Demonstrates model explainability with publication-ready formatting
        "feature_importance_publication": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/publication/feature_importance_{run_id}.pdf",
            "purpose": "publication",
            "condition_param": "interpretation_method",
            "style_params": {
                "explainability_theme": True,
                "feature_ranking_emphasis": True,
                "interpretation_clarity": True
            },
            "format_kwargs": {
                "dpi": 300,
                "bbox_inches": "tight",
                "format": "pdf"
            },
            "versioned": True,
            "metadata": {
                "description": "Publication quality feature importance and explainability analysis",
                "tags": ["inference", "explainability", "publication", "interpretation"]
            }
        },
        
        # Deployment Performance Analysis
        # Demonstrates production deployment visualization with business context
        "deployment_analysis_presentation": {
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": "data/08_reporting/inference/presentation/deployment_analysis_{run_id}.png",
            "purpose": "presentation",
            "condition_param": "deployment_environment",
            "style_params": {
                "production_theme": True,
                "scalability_emphasis": True,
                "performance_highlighting": True
            },
            "format_kwargs": {
                "dpi": 200,
                "bbox_inches": "tight"
            },
            "versioned": True,
            "metadata": {
                "description": "Production deployment performance and scalability analysis",
                "tags": ["inference", "deployment", "performance", "production"]
            }
        }
    }
    
    logger.info(f"Generated catalog configuration for {len(catalog_config)} inference outputs")
    return catalog_config


def get_advanced_condition_mapping() -> Dict[str, Any]:
    """
    Get sophisticated condition mapping configuration for inference styling.
    
    This function demonstrates advanced condition-based styling configuration
    that showcases the full capabilities of figregistry-kedro integration for
    complex experimental scenarios and enterprise workflows.
    
    Returns:
        Dict[str, Any]: Advanced condition mapping configuration
        
    Note:
        This demonstrates F-002-RQ-002 wildcard and partial matching capabilities
        with sophisticated inheritance and style resolution patterns.
    """
    condition_mapping = {
        
        # Model Type Conditions
        # Demonstrates basic condition-based styling for different model architectures
        "model_type": {
            "classification": {
                "color_palette": "categorical_distinct",
                "marker": "o",
                "visualization_emphasis": "confusion_matrix",
                "performance_metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "regression": {
                "color_palette": "continuous_viridis", 
                "marker": "s",
                "visualization_emphasis": "scatter_plots",
                "performance_metrics": ["mae", "rmse", "r2", "mape"]
            },
            "ensemble": {
                "color_palette": "ensemble_gradient",
                "marker": "^",
                "visualization_emphasis": "model_combination",
                "performance_metrics": ["ensemble_accuracy", "diversity_index"]
            },
            "deep_learning": {
                "color_palette": "neural_network_theme",
                "marker": "D",
                "visualization_emphasis": "learning_curves",
                "performance_metrics": ["loss", "validation_loss", "convergence"]
            }
        },
        
        # Evaluation Metric Conditions
        # Demonstrates sophisticated metric-based styling adaptation
        "evaluation_metric": {
            "accuracy": {
                "performance_color_map": "green_to_red_accuracy",
                "threshold_lines": [0.8, 0.9, 0.95],
                "emphasis_style": "performance_tier"
            },
            "precision": {
                "performance_color_map": "blue_precision_scale",
                "threshold_lines": [0.7, 0.85, 0.9],
                "emphasis_style": "precision_focus"
            },
            "recall": {
                "performance_color_map": "orange_recall_scale",
                "threshold_lines": [0.7, 0.8, 0.9],
                "emphasis_style": "recall_sensitivity"
            },
            "f1_score": {
                "performance_color_map": "balanced_f1_scale",
                "threshold_lines": [0.75, 0.85, 0.92],
                "emphasis_style": "balanced_metrics"
            }
        },
        
        # Deployment Environment Conditions
        # Demonstrates environment-specific styling for different deployment contexts
        "deployment_environment": {
            "development": {
                "theme": "casual_development",
                "color_intensity": 0.7,
                "annotation_level": "detailed",
                "grid_style": "relaxed"
            },
            "staging": {
                "theme": "testing_validation",
                "color_intensity": 0.8,
                "annotation_level": "moderate",
                "grid_style": "structured"
            },
            "production": {
                "theme": "professional_production", 
                "color_intensity": 1.0,
                "annotation_level": "essential",
                "grid_style": "minimal_clean"
            },
            "enterprise": {
                "theme": "enterprise_corporate",
                "color_intensity": 0.9,
                "annotation_level": "business_focused",
                "grid_style": "corporate_standard"
            }
        },
        
        # Performance Tier Conditions
        # Demonstrates performance-based visual emphasis and styling
        "performance_tier": {
            "excellent": {
                "status_color": "#2E8B57",  # Sea Green
                "emphasis_level": "high",
                "highlight_style": "success_glow",
                "annotation_style": "achievement"
            },
            "good": {
                "status_color": "#4682B4",  # Steel Blue
                "emphasis_level": "medium",
                "highlight_style": "standard",
                "annotation_style": "positive"
            },
            "average": {
                "status_color": "#DAA520",  # Goldenrod
                "emphasis_level": "medium",
                "highlight_style": "neutral",
                "annotation_style": "informational"
            },
            "needs_attention": {
                "status_color": "#CD5C5C",  # Indian Red
                "emphasis_level": "high",
                "highlight_style": "warning_pulse",
                "annotation_style": "alert"
            }
        },
        
        # Dataset Name Conditions (Domain-Specific Styling)
        # Demonstrates domain-aware styling based on dataset characteristics
        "dataset_name": {
            "customer_churn": {
                "domain_theme": "business_analytics",
                "color_palette": "customer_lifecycle",
                "icon_set": "business_metrics"
            },
            "medical_diagnosis": {
                "domain_theme": "healthcare_clinical",
                "color_palette": "medical_safety",
                "icon_set": "healthcare_symbols"
            },
            "financial_risk": {
                "domain_theme": "financial_analysis",
                "color_palette": "risk_assessment",
                "icon_set": "financial_indicators"
            },
            "manufacturing_quality": {
                "domain_theme": "industrial_process",
                "color_palette": "quality_control",
                "icon_set": "manufacturing_metrics"
            }
        },
        
        # Confidence Threshold Conditions
        # Demonstrates confidence-based visual styling adjustments
        "confidence_threshold": {
            "high": {
                "alpha_multiplier": 1.0,
                "line_thickness_multiplier": 1.2,
                "marker_size_multiplier": 1.1,
                "confidence_emphasis": "strong"
            },
            "medium": {
                "alpha_multiplier": 0.8,
                "line_thickness_multiplier": 1.0,
                "marker_size_multiplier": 1.0,
                "confidence_emphasis": "moderate"
            },
            "low": {
                "alpha_multiplier": 0.6,
                "line_thickness_multiplier": 0.8,
                "marker_size_multiplier": 0.9,
                "confidence_emphasis": "subtle"
            }
        },
        
        # Prediction Horizon Conditions
        # Demonstrates temporal-based styling for time series inference
        "prediction_horizon": {
            "short_term": {
                "temporal_style": "immediate_focus",
                "line_pattern": "solid",
                "uncertainty_bands": "tight",
                "time_emphasis": "current"
            },
            "medium_term": {
                "temporal_style": "planning_horizon",
                "line_pattern": "dashed",
                "uncertainty_bands": "moderate", 
                "time_emphasis": "projected"
            },
            "long_term": {
                "temporal_style": "strategic_forecast",
                "line_pattern": "dotted",
                "uncertainty_bands": "wide",
                "time_emphasis": "strategic"
            }
        },
        
        # Wildcard and Pattern Matching Examples
        # Demonstrates F-002-RQ-002 flexible matching capabilities
        "pattern_matching": {
            "*_classification": {
                "base_style": "classification_family",
                "inheritance": "model_type.classification"
            },
            "*_regression": {
                "base_style": "regression_family", 
                "inheritance": "model_type.regression"
            },
            "production_*": {
                "base_style": "production_family",
                "inheritance": "deployment_environment.production"
            },
            "*_experiment": {
                "base_style": "experimental_family",
                "special_annotations": True,
                "experimental_emphasis": True
            }
        }
    }
    
    logger.info("Generated advanced condition mapping configuration")
    return condition_mapping


# Export pipeline creation function and utilities for advanced usage
__all__ = [
    "create_pipeline",
    "create_inference_sub_pipeline", 
    "validate_inference_pipeline_config",
    "get_inference_pipeline_catalog_config",
    "get_advanced_condition_mapping",
    "INFERENCE_PIPELINE_VERSION",
    "SUPPORTED_MODEL_TYPES",
    "ADVANCED_CONDITION_PATTERNS"
]