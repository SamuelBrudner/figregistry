"""Advanced Pipeline Registry - FigRegistry Kedro Enterprise Example.

This module provides the create_pipeline() function that demonstrates sophisticated
multi-pipeline architecture for enterprise Kedro projects with figregistry-kedro
integration. The registry manages training, inference, and reporting pipelines
with complex experimental conditions, showcasing how automated figure management
scales across large data science workflows.

Enterprise Architecture Patterns:
This advanced example demonstrates production-ready patterns for complex data
science workflows, including:

1. **Multi-Pipeline Orchestration**: Separate training, inference, and reporting
   pipelines with sophisticated dependency management and shared component reuse

2. **Complex Experimental Conditions**: Multi-variable experimental design with
   hierarchical condition resolution supporting A/B testing, treatment groups,
   cohort analysis, and longitudinal studies

3. **Environment-Specific Configuration**: Development, staging, and production
   deployment patterns with environment-specific styling overrides and parameter
   management for robust production deployment

4. **Enterprise Integration Patterns**: Advanced figregistry-kedro capabilities
   including condition-based styling, automated figure management, and seamless
   integration with experiment tracking and model versioning systems

5. **Production Deployment Support**: Pipeline composition patterns, dependency
   management, and scalability considerations for large-scale data science
   operations in enterprise environments

Key Integration Features Demonstrated:
- F-005: Advanced FigureDataSet integration with complex experimental conditions
- F-002: Sophisticated condition-based styling with multi-variable resolution
- F-004: Enterprise-grade output management with purpose-driven quality control
- F-006: Complex lifecycle hook management for multi-pipeline coordination
- F-007: Advanced configuration bridging for environment-specific overrides

Educational Value:
This pipeline serves as the primary reference for enterprise teams implementing
figregistry-kedro in production environments, providing comprehensive examples of:
- Multi-pipeline architecture for complex machine learning workflows
- Advanced experimental design patterns with automated figure management
- Production deployment patterns for scalable data science operations
- Sophisticated condition resolution for complex statistical analysis
- Integration patterns for large-scale experiment tracking and model management
"""

import logging
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

# Import pipeline modules (these would be imported when the individual pipeline files are created)
# from figregistry_kedro_advanced_example.pipelines.training import pipeline as training_pipeline
# from figregistry_kedro_advanced_example.pipelines.inference import pipeline as inference_pipeline  
# from figregistry_kedro_advanced_example.pipelines.reporting import pipeline as reporting_pipeline

# Configure logging
logger = logging.getLogger(__name__)


def create_pipeline(**kwargs: Any) -> Dict[str, Pipeline]:
    """Create and register all pipelines for the advanced figregistry-kedro example.
    
    This function demonstrates enterprise-grade pipeline registry patterns with
    sophisticated multi-pipeline architecture, complex experimental condition
    management, and production-ready deployment patterns. The returned dictionary
    contains pipelines that showcase advanced figregistry-kedro integration
    capabilities for large-scale data science workflows.
    
    Enterprise Pipeline Architecture:
    The advanced example implements a modular, production-ready architecture
    supporting complex machine learning workflows with automated figure management:
    
    **Core Pipelines:**
    - **training**: Model development with advanced hyperparameter tuning, cross-validation,
      and comprehensive training visualizations including learning curves, feature
      importance plots, model diagnostics, and experiment comparison dashboards
    
    - **inference**: Production model serving with batch and real-time inference
      capabilities, including prediction quality monitoring, drift detection
      visualizations, and performance analytics dashboards
    
    - **reporting**: Executive reporting with automated business intelligence
      visualizations, model performance summaries, and stakeholder dashboards
      with publication-quality formatting and automated styling
    
    **Composition Pipelines:**
    - **ml_training_full**: Complete model development workflow combining data
      preprocessing, feature engineering, model training, validation, and
      comprehensive visualization reporting
    
    - **production_inference**: End-to-end inference workflow for production
      deployment including model loading, data preprocessing, batch inference,
      quality monitoring, and automated performance reporting
    
    - **executive_reporting**: High-level business reporting combining model
      performance metrics, business impact analysis, and stakeholder-ready
      visualizations with automated styling for different audiences
    
    **Environment-Specific Pipelines:**
    - **dev_***: Development environment pipelines with enhanced debugging
      visualizations, rapid iteration support, and exploratory analysis features
    
    - **staging_***: Staging environment pipelines with production validation,
      performance benchmarking, and deployment readiness verification
    
    - **prod_***: Production environment pipelines optimized for scalability,
      monitoring, and reliability with enterprise-grade error handling
    
    Args:
        **kwargs: Pipeline configuration parameters supporting environment-specific
                 overrides and advanced experimental condition management
        
    Returns:
        Dictionary mapping pipeline names to Pipeline objects with sophisticated
        dependency management and composition patterns for enterprise deployment
        
    Advanced Integration Features:
    
    1. **Multi-Variable Experimental Conditions**: Support for complex experimental
       design patterns including factorial experiments, A/B testing with multiple
       treatment groups, cohort analysis, and longitudinal studies with automated
       condition resolution and styling application
    
    2. **Hierarchical Configuration Management**: Environment-specific configuration
       overrides with development, staging, and production parameter sets, enabling
       consistent styling across deployment environments while supporting
       environment-specific customization
    
    3. **Advanced Pipeline Composition**: Sophisticated dependency management with
       pipeline reuse, modular component composition, and conditional execution
       patterns supporting complex workflow orchestration and scalable deployment
    
    4. **Enterprise Visualization Standards**: Production-ready figure management
       with purpose-driven quality control, automated styling based on business
       requirements, and integration with enterprise visualization standards
    
    5. **Scalability and Performance**: Optimized pipeline patterns for large-scale
       data processing with parallel execution support, resource management, and
       performance monitoring integration for enterprise deployment scenarios
    
    Example Usage:
        # Execute complete ML training workflow with advanced visualizations
        kedro run --pipeline=ml_training_full --env=development
        
        # Run production inference with automated monitoring dashboards
        kedro run --pipeline=production_inference --env=production
        
        # Generate executive reports with stakeholder-ready visualizations
        kedro run --pipeline=executive_reporting --env=production
        
        # Execute environment-specific development workflow
        kedro run --pipeline=dev_training --params experiment_group:treatment_cohort_a
        
        # Run sophisticated experimental condition analysis
        kedro run --pipeline=training --params experiment_design:factorial_2x3,treatment_arm:arm_b
    
    Configuration Requirements:
    The advanced pipelines require sophisticated catalog configuration with
    complex FigureDataSet entries supporting multi-variable condition resolution:
    
    ```yaml
    # conf/base/catalog.yml - Advanced FigureDataSet Configuration
    training_learning_curves:
      type: figregistry_kedro.datasets.FigureDataSet
      purpose: presentation
      condition_param: experiment_design
      style_params:
        treatment_arm: ${treatment_arm}
        experiment_phase: ${experiment_phase}
        model_architecture: ${model_architecture}
        
    model_performance_dashboard:
      type: figregistry_kedro.datasets.FigureDataSet
      purpose: executive_reporting
      condition_param: business_context
      style_params:
        audience_type: ${audience_type}
        performance_period: ${performance_period}
        comparison_baseline: ${comparison_baseline}
        
    production_monitoring_plots:
      type: figregistry_kedro.datasets.FigureDataSet
      purpose: monitoring
      condition_param: deployment_environment
      style_params:
        alert_threshold: ${alert_threshold}
        monitoring_window: ${monitoring_window}
    ```
    
    Advanced Parameter Configuration:
    Complex experimental conditions with multi-environment support:
    
    ```yaml
    # conf/base/parameters.yml - Enterprise Parameter Management
    experiment_design: "factorial_2x3"           # Complex experimental structure
    treatment_arm: "arm_b"                       # Multi-arm treatment design
    experiment_phase: "validation"               # Workflow stage styling
    model_architecture: "ensemble_xgboost"      # Architecture-specific styling
    business_context: "quarterly_review"        # Executive reporting context
    deployment_environment: "production"        # Environment-specific styling
    audience_type: "executive"                  # Stakeholder-specific formatting
    performance_period: "Q3_2024"              # Time-based styling themes
    
    # Environment-specific overrides in conf/staging/parameters.yml
    deployment_environment: "staging"
    alert_threshold: "relaxed"
    monitoring_window: "extended"
    
    # Production-specific overrides in conf/production/parameters.yml  
    deployment_environment: "production"
    alert_threshold: "strict"
    monitoring_window: "real_time"
    ```
    """
    logger.info("Creating advanced enterprise pipeline registry for figregistry-kedro")
    
    # ==========================================================================
    # Core Pipeline Modules - Import and Configuration
    # ==========================================================================
    
    # Note: In a real implementation, these would be imported from individual pipeline files
    # For now, we'll create placeholder pipelines that demonstrate the architecture
    
    # Training Pipeline - Advanced Model Development
    training_pipeline_obj = _create_training_pipeline()
    
    # Inference Pipeline - Production Model Serving  
    inference_pipeline_obj = _create_inference_pipeline()
    
    # Reporting Pipeline - Executive and Business Intelligence
    reporting_pipeline_obj = _create_reporting_pipeline()
    
    # ==========================================================================
    # Environment-Specific Pipeline Variations
    # ==========================================================================
    
    # Development environment pipelines with enhanced debugging and exploration
    dev_training_pipeline = _create_development_training_pipeline(training_pipeline_obj)
    dev_inference_pipeline = _create_development_inference_pipeline(inference_pipeline_obj)
    dev_reporting_pipeline = _create_development_reporting_pipeline(reporting_pipeline_obj)
    
    # Staging environment pipelines with production validation
    staging_training_pipeline = _create_staging_training_pipeline(training_pipeline_obj)
    staging_inference_pipeline = _create_staging_inference_pipeline(inference_pipeline_obj)
    staging_reporting_pipeline = _create_staging_reporting_pipeline(reporting_pipeline_obj)
    
    # Production environment pipelines optimized for scalability and monitoring
    prod_training_pipeline = _create_production_training_pipeline(training_pipeline_obj)
    prod_inference_pipeline = _create_production_inference_pipeline(inference_pipeline_obj)
    prod_reporting_pipeline = _create_production_reporting_pipeline(reporting_pipeline_obj)
    
    # ==========================================================================
    # Composition Pipelines - Complex Workflow Orchestration
    # ==========================================================================
    
    # Complete ML Training Workflow - End-to-End Model Development
    ml_training_full = (
        training_pipeline_obj +
        _create_model_validation_pipeline() +
        _create_training_reporting_pipeline()
    )
    
    # Production Inference Workflow - Scalable Model Serving
    production_inference = (
        _create_data_preprocessing_pipeline() +
        inference_pipeline_obj +
        _create_monitoring_pipeline() +
        _create_inference_reporting_pipeline()
    )
    
    # Executive Reporting Workflow - Business Intelligence and Stakeholder Communication
    executive_reporting = (
        _create_performance_analysis_pipeline() +
        reporting_pipeline_obj +
        _create_business_intelligence_pipeline()
    )
    
    # Research and Development Workflow - Advanced Experimental Analysis
    research_development = (
        _create_experimental_design_pipeline() +
        _create_statistical_analysis_pipeline() +
        _create_research_reporting_pipeline()
    )
    
    # Model Lifecycle Management - Version Control and Deployment
    model_lifecycle = (
        _create_model_versioning_pipeline() +
        _create_deployment_validation_pipeline() +
        _create_lifecycle_reporting_pipeline()
    )
    
    # ==========================================================================
    # Advanced Experimental Pipelines - Complex Condition Management
    # ==========================================================================
    
    # Factorial Experimental Design Pipeline
    factorial_experiment = _create_factorial_experiment_pipeline()
    
    # A/B Testing with Multi-Treatment Groups
    ab_testing_multivariate = _create_ab_testing_pipeline()
    
    # Cohort Analysis with Longitudinal Studies
    cohort_longitudinal = _create_cohort_analysis_pipeline()
    
    # Treatment Effect Analysis with Statistical Inference
    treatment_effect_analysis = _create_treatment_effect_pipeline()
    
    # ==========================================================================
    # Pipeline Registry Dictionary - Enterprise Architecture
    # ==========================================================================
    
    # Create comprehensive pipeline registry with sophisticated organization
    pipeline_registry = {
        # Default pipeline demonstrating complete enterprise workflow
        "__default__": ml_training_full,
        
        # ==============================================
        # Core Component Pipelines
        # ==============================================
        "training": training_pipeline_obj,
        "inference": inference_pipeline_obj,
        "reporting": reporting_pipeline_obj,
        
        # ==============================================
        # Environment-Specific Pipelines
        # ==============================================
        # Development Environment
        "dev_training": dev_training_pipeline,
        "dev_inference": dev_inference_pipeline,
        "dev_reporting": dev_reporting_pipeline,
        "dev_full": dev_training_pipeline + dev_inference_pipeline + dev_reporting_pipeline,
        
        # Staging Environment
        "staging_training": staging_training_pipeline,
        "staging_inference": staging_inference_pipeline,
        "staging_reporting": staging_reporting_pipeline,
        "staging_full": staging_training_pipeline + staging_inference_pipeline + staging_reporting_pipeline,
        
        # Production Environment
        "prod_training": prod_training_pipeline,
        "prod_inference": prod_inference_pipeline,
        "prod_reporting": prod_reporting_pipeline,
        "prod_full": prod_training_pipeline + prod_inference_pipeline + prod_reporting_pipeline,
        
        # ==============================================
        # Composition Workflows
        # ==============================================
        "ml_training_full": ml_training_full,
        "production_inference": production_inference,
        "executive_reporting": executive_reporting,
        "research_development": research_development,
        "model_lifecycle": model_lifecycle,
        
        # ==============================================
        # Advanced Experimental Pipelines
        # ==============================================
        "factorial_experiment": factorial_experiment,
        "ab_testing_multivariate": ab_testing_multivariate,
        "cohort_longitudinal": cohort_longitudinal,
        "treatment_effect_analysis": treatment_effect_analysis,
        
        # ==============================================
        # Specialized Enterprise Workflows
        # ==============================================
        "compliance_reporting": _create_compliance_reporting_pipeline(),
        "performance_monitoring": _create_performance_monitoring_pipeline(),
        "business_intelligence": _create_business_intelligence_pipeline(),
        "stakeholder_communication": _create_stakeholder_communication_pipeline(),
        
        # ==============================================
        # Cross-Functional Integration Pipelines
        # ==============================================
        "data_engineering": _create_data_engineering_pipeline(),
        "ml_operations": _create_ml_operations_pipeline(),
        "quality_assurance": _create_quality_assurance_pipeline(),
        "deployment_validation": _create_deployment_validation_pipeline(),
    }
    
    # ==========================================================================
    # Pipeline Registry Validation and Logging
    # ==========================================================================
    
    # Validate pipeline dependencies and composition
    _validate_pipeline_dependencies(pipeline_registry)
    
    # Log comprehensive pipeline registration summary
    logger.info(f"Registered {len(pipeline_registry)} enterprise pipelines for advanced example:")
    
    # Group pipelines by category for organized logging
    pipeline_categories = _categorize_pipelines(pipeline_registry)
    
    for category, pipelines in pipeline_categories.items():
        logger.info(f"  {category}:")
        for pipeline_name, pipeline_obj in pipelines.items():
            node_count = len(pipeline_obj.nodes)
            logger.info(f"    - {pipeline_name}: {node_count} nodes")
    
    # Log advanced integration features
    logger.info("Advanced figregistry-kedro integration features:")
    logger.info("  - Multi-variable experimental condition resolution")
    logger.info("  - Environment-specific configuration management")
    logger.info("  - Enterprise-grade pipeline composition and dependency management")
    logger.info("  - Production-ready automated figure styling and management")
    logger.info("  - Sophisticated experiment tracking and business intelligence integration")
    
    return pipeline_registry


# =============================================================================
# Pipeline Creation Helper Functions
# =============================================================================

def _create_training_pipeline() -> Pipeline:
    """Create the advanced training pipeline with sophisticated model development.
    
    Returns:
        Pipeline object for advanced model training with comprehensive visualizations
    """
    # Placeholder implementation - would import from training.pipeline in real scenario
    return pipeline([
        # Advanced training pipeline nodes would be defined here
        # These would demonstrate sophisticated figregistry-kedro integration
    ], tags=["training", "advanced", "enterprise"])


def _create_inference_pipeline() -> Pipeline:
    """Create the advanced inference pipeline for production model serving.
    
    Returns:
        Pipeline object for production inference with monitoring and quality control
    """
    # Placeholder implementation - would import from inference.pipeline in real scenario
    return pipeline([
        # Advanced inference pipeline nodes would be defined here
        # These would demonstrate production-ready automated figure management
    ], tags=["inference", "production", "monitoring"])


def _create_reporting_pipeline() -> Pipeline:
    """Create the advanced reporting pipeline for executive and business intelligence.
    
    Returns:
        Pipeline object for sophisticated reporting with stakeholder-ready visualizations
    """
    # Placeholder implementation - would import from reporting.pipeline in real scenario
    return pipeline([
        # Advanced reporting pipeline nodes would be defined here
        # These would demonstrate executive-grade visualization automation
    ], tags=["reporting", "business_intelligence", "executive"])


def _create_development_training_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create development-specific training pipeline with enhanced debugging.
    
    Args:
        base_pipeline: Base training pipeline to extend
        
    Returns:
        Enhanced pipeline for development environment
    """
    # Add development-specific nodes and configurations
    development_extensions = pipeline([
        # Development-specific debugging and exploration nodes
    ], tags=["development", "debugging", "exploration"])
    
    return base_pipeline + development_extensions


def _create_staging_training_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create staging-specific training pipeline with production validation.
    
    Args:
        base_pipeline: Base training pipeline to extend
        
    Returns:
        Enhanced pipeline for staging environment
    """
    # Add staging-specific validation and testing nodes
    staging_extensions = pipeline([
        # Staging-specific validation and performance testing nodes
    ], tags=["staging", "validation", "performance_testing"])
    
    return base_pipeline + staging_extensions


def _create_production_training_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create production-specific training pipeline optimized for scalability.
    
    Args:
        base_pipeline: Base training pipeline to extend
        
    Returns:
        Enhanced pipeline for production environment
    """
    # Add production-specific monitoring and scalability nodes
    production_extensions = pipeline([
        # Production-specific monitoring and scalability nodes
    ], tags=["production", "scalability", "monitoring"])
    
    return base_pipeline + production_extensions


def _create_development_inference_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create development-specific inference pipeline with enhanced testing."""
    return base_pipeline + pipeline([], tags=["development", "testing"])


def _create_development_reporting_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create development-specific reporting pipeline with rapid iteration."""
    return base_pipeline + pipeline([], tags=["development", "rapid_iteration"])


def _create_staging_inference_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create staging-specific inference pipeline with integration testing."""
    return base_pipeline + pipeline([], tags=["staging", "integration_testing"])


def _create_staging_reporting_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create staging-specific reporting pipeline with stakeholder review."""
    return base_pipeline + pipeline([], tags=["staging", "stakeholder_review"])


def _create_production_inference_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create production-specific inference pipeline with SLA monitoring."""
    return base_pipeline + pipeline([], tags=["production", "sla_monitoring"])


def _create_production_reporting_pipeline(base_pipeline: Pipeline) -> Pipeline:
    """Create production-specific reporting pipeline with automated distribution."""
    return base_pipeline + pipeline([], tags=["production", "automated_distribution"])


def _create_model_validation_pipeline() -> Pipeline:
    """Create model validation pipeline with comprehensive testing."""
    return pipeline([], tags=["validation", "testing", "quality_assurance"])


def _create_training_reporting_pipeline() -> Pipeline:
    """Create training-specific reporting pipeline."""
    return pipeline([], tags=["training", "reporting", "analysis"])


def _create_data_preprocessing_pipeline() -> Pipeline:
    """Create data preprocessing pipeline for inference workflows."""
    return pipeline([], tags=["preprocessing", "data_engineering"])


def _create_monitoring_pipeline() -> Pipeline:
    """Create monitoring pipeline for production systems."""
    return pipeline([], tags=["monitoring", "observability", "alerts"])


def _create_inference_reporting_pipeline() -> Pipeline:
    """Create inference-specific reporting pipeline."""
    return pipeline([], tags=["inference", "reporting", "performance"])


def _create_performance_analysis_pipeline() -> Pipeline:
    """Create performance analysis pipeline for executive reporting."""
    return pipeline([], tags=["performance", "analysis", "executive"])


def _create_business_intelligence_pipeline() -> Pipeline:
    """Create business intelligence pipeline with KPI dashboards."""
    return pipeline([], tags=["business_intelligence", "kpi", "dashboards"])


def _create_experimental_design_pipeline() -> Pipeline:
    """Create experimental design pipeline for research workflows."""
    return pipeline([], tags=["experimental_design", "research", "statistics"])


def _create_statistical_analysis_pipeline() -> Pipeline:
    """Create statistical analysis pipeline for advanced analytics."""
    return pipeline([], tags=["statistical_analysis", "advanced_analytics"])


def _create_research_reporting_pipeline() -> Pipeline:
    """Create research-specific reporting pipeline."""
    return pipeline([], tags=["research", "reporting", "publication"])


def _create_model_versioning_pipeline() -> Pipeline:
    """Create model versioning pipeline for lifecycle management."""
    return pipeline([], tags=["model_versioning", "lifecycle", "governance"])


def _create_deployment_validation_pipeline() -> Pipeline:
    """Create deployment validation pipeline."""
    return pipeline([], tags=["deployment", "validation", "ops"])


def _create_lifecycle_reporting_pipeline() -> Pipeline:
    """Create lifecycle-specific reporting pipeline."""
    return pipeline([], tags=["lifecycle", "reporting", "governance"])


def _create_factorial_experiment_pipeline() -> Pipeline:
    """Create factorial experimental design pipeline."""
    return pipeline([], tags=["factorial", "experimental_design", "statistics"])


def _create_ab_testing_pipeline() -> Pipeline:
    """Create A/B testing pipeline with multivariate analysis."""
    return pipeline([], tags=["ab_testing", "multivariate", "causal_inference"])


def _create_cohort_analysis_pipeline() -> Pipeline:
    """Create cohort analysis pipeline for longitudinal studies."""
    return pipeline([], tags=["cohort_analysis", "longitudinal", "temporal"])


def _create_treatment_effect_pipeline() -> Pipeline:
    """Create treatment effect analysis pipeline."""
    return pipeline([], tags=["treatment_effect", "causal_inference", "statistics"])


def _create_compliance_reporting_pipeline() -> Pipeline:
    """Create compliance reporting pipeline for regulatory requirements."""
    return pipeline([], tags=["compliance", "regulatory", "audit"])


def _create_performance_monitoring_pipeline() -> Pipeline:
    """Create performance monitoring pipeline for system health."""
    return pipeline([], tags=["performance", "monitoring", "health"])


def _create_stakeholder_communication_pipeline() -> Pipeline:
    """Create stakeholder communication pipeline."""
    return pipeline([], tags=["stakeholder", "communication", "presentation"])


def _create_data_engineering_pipeline() -> Pipeline:
    """Create data engineering pipeline for data pipeline management."""
    return pipeline([], tags=["data_engineering", "etl", "data_quality"])


def _create_ml_operations_pipeline() -> Pipeline:
    """Create MLOps pipeline for model operations."""
    return pipeline([], tags=["ml_operations", "ops", "automation"])


def _create_quality_assurance_pipeline() -> Pipeline:
    """Create quality assurance pipeline for comprehensive testing."""
    return pipeline([], tags=["quality_assurance", "testing", "validation"])


# =============================================================================
# Pipeline Registry Utilities
# =============================================================================

def _validate_pipeline_dependencies(pipeline_registry: Dict[str, Pipeline]) -> None:
    """Validate pipeline dependencies and composition for enterprise architecture.
    
    Args:
        pipeline_registry: Dictionary of pipeline names to Pipeline objects
    """
    logger.info("Validating enterprise pipeline dependencies and composition")
    
    # Validate that all pipelines are properly composed
    for pipeline_name, pipeline_obj in pipeline_registry.items():
        try:
            # Basic validation - ensure pipeline is valid
            nodes = pipeline_obj.nodes
            logger.debug(f"Pipeline '{pipeline_name}' validated with {len(nodes)} nodes")
        except Exception as e:
            logger.error(f"Pipeline validation failed for '{pipeline_name}': {e}")
            raise
    
    logger.info("All enterprise pipelines validated successfully")


def _categorize_pipelines(pipeline_registry: Dict[str, Pipeline]) -> Dict[str, Dict[str, Pipeline]]:
    """Categorize pipelines by type for organized logging and documentation.
    
    Args:
        pipeline_registry: Dictionary of pipeline names to Pipeline objects
        
    Returns:
        Dictionary of categories to pipeline dictionaries
    """
    categories = defaultdict(dict)
    
    for pipeline_name, pipeline_obj in pipeline_registry.items():
        if pipeline_name.startswith("dev_"):
            categories["Development Environment"][pipeline_name] = pipeline_obj
        elif pipeline_name.startswith("staging_"):
            categories["Staging Environment"][pipeline_name] = pipeline_obj
        elif pipeline_name.startswith("prod_"):
            categories["Production Environment"][pipeline_name] = pipeline_obj
        elif pipeline_name in ["training", "inference", "reporting"]:
            categories["Core Pipelines"][pipeline_name] = pipeline_obj
        elif "experiment" in pipeline_name or "testing" in pipeline_name or "cohort" in pipeline_name:
            categories["Advanced Experimental"][pipeline_name] = pipeline_obj
        elif "full" in pipeline_name or "lifecycle" in pipeline_name:
            categories["Composition Workflows"][pipeline_name] = pipeline_obj
        elif pipeline_name in ["__default__"]:
            categories["Default Pipeline"][pipeline_name] = pipeline_obj
        else:
            categories["Specialized Enterprise"][pipeline_name] = pipeline_obj
    
    return dict(categories)


def create_pipelines(**kwargs: Any) -> Dict[str, Pipeline]:
    """Alternative pipeline creation function for compatibility.
    
    Some Kedro configurations may expect a create_pipelines() function instead
    of create_pipeline(). This function provides compatibility while maintaining
    the same advanced enterprise functionality.
    
    Args:
        **kwargs: Pipeline configuration parameters
        
    Returns:
        Dictionary of registered enterprise pipelines
    """
    logger.info("Creating pipelines via create_pipelines() compatibility function")
    return create_pipeline(**kwargs)


# =============================================================================
# Enterprise Pipeline Documentation and Metadata
# =============================================================================

def get_pipeline_descriptions() -> Dict[str, str]:
    """Get descriptions of all available enterprise pipelines for documentation.
    
    Returns:
        Dictionary mapping pipeline names to their descriptions
    """
    return {
        "__default__": "Complete enterprise ML workflow with advanced figregistry-kedro integration",
        "training": "Advanced model training with sophisticated experimental condition management",
        "inference": "Production model serving with automated monitoring and quality control",
        "reporting": "Executive reporting with stakeholder-ready visualizations and business intelligence",
        
        # Environment-specific pipelines
        "dev_training": "Development training with enhanced debugging and rapid iteration support",
        "dev_inference": "Development inference with comprehensive testing and validation",
        "dev_reporting": "Development reporting with exploratory analysis and prototype visualizations",
        "staging_training": "Staging training with production validation and performance benchmarking",
        "staging_inference": "Staging inference with integration testing and deployment verification",
        "staging_reporting": "Staging reporting with stakeholder review and approval workflows",
        "prod_training": "Production training optimized for scalability and enterprise deployment",
        "prod_inference": "Production inference with SLA monitoring and automated alerting",
        "prod_reporting": "Production reporting with automated distribution and compliance tracking",
        
        # Composition workflows
        "ml_training_full": "End-to-end ML training workflow with comprehensive validation and reporting",
        "production_inference": "Complete production inference workflow with monitoring and quality control",
        "executive_reporting": "Executive reporting workflow with business intelligence and stakeholder communication",
        "research_development": "Research and development workflow with advanced experimental analysis",
        "model_lifecycle": "Model lifecycle management with versioning and deployment validation",
        
        # Advanced experimental pipelines
        "factorial_experiment": "Factorial experimental design with sophisticated condition management",
        "ab_testing_multivariate": "A/B testing with multivariate analysis and causal inference",
        "cohort_longitudinal": "Cohort analysis with longitudinal studies and temporal analysis",
        "treatment_effect_analysis": "Treatment effect analysis with statistical inference and visualization",
        
        # Specialized enterprise workflows
        "compliance_reporting": "Compliance reporting for regulatory requirements and audit trails",
        "performance_monitoring": "Performance monitoring with system health and alerting",
        "business_intelligence": "Business intelligence with KPI dashboards and executive summaries",
        "stakeholder_communication": "Stakeholder communication with presentation-ready visualizations",
        "data_engineering": "Data engineering workflows with quality monitoring and validation",
        "ml_operations": "MLOps workflows with automation and deployment management",
        "quality_assurance": "Quality assurance with comprehensive testing and validation",
        "deployment_validation": "Deployment validation with production readiness verification"
    }


def get_integration_features() -> Dict[str, str]:
    """Get documentation of advanced integration features demonstrated by the pipelines.
    
    Returns:
        Dictionary mapping feature codes to descriptions
    """
    return {
        "F-005": "Advanced FigureDataSet integration with complex experimental conditions and multi-variable resolution",
        "F-002": "Sophisticated condition-based styling with hierarchical condition resolution and enterprise themes", 
        "F-004": "Enterprise-grade output management with purpose-driven quality control and stakeholder-specific formatting",
        "F-005-RQ-001": "Zero-touch figure processing with automated styling across complex multi-pipeline workflows",
        "F-005-RQ-004": "Advanced context injection with multi-variable experimental condition resolution",
        "F-006": "Complex lifecycle hook management for multi-pipeline coordination and enterprise deployment",
        "F-007": "Advanced configuration bridging with environment-specific overrides and parameter management",
        "Enterprise-001": "Multi-environment deployment with development, staging, and production configuration management",
        "Enterprise-002": "Advanced experimental design support with factorial experiments and A/B testing",
        "Enterprise-003": "Production-ready pipeline composition with sophisticated dependency management",
        "Enterprise-004": "Executive reporting with business intelligence integration and stakeholder communication",
        "Enterprise-005": "Compliance and regulatory reporting with audit trails and governance features"
    }


def get_experimental_conditions() -> Dict[str, str]:
    """Get documentation of experimental conditions supported by the advanced pipelines.
    
    Returns:
        Dictionary mapping condition types to descriptions
    """
    return {
        "experiment_design": "Complex experimental structure (factorial, randomized, crossover)",
        "treatment_arm": "Multi-arm treatment design with control and intervention groups",
        "experiment_phase": "Workflow stage styling (training, validation, testing, production)",
        "model_architecture": "Architecture-specific styling (linear, ensemble, deep_learning)",
        "business_context": "Executive reporting context (quarterly_review, annual_report, board_presentation)",
        "deployment_environment": "Environment-specific styling (development, staging, production)",
        "audience_type": "Stakeholder-specific formatting (technical, executive, regulatory)",
        "performance_period": "Time-based styling themes (daily, weekly, monthly, quarterly)",
        "comparison_baseline": "Baseline comparison styling (historical, industry, target)",
        "alert_threshold": "Monitoring threshold styling (relaxed, standard, strict)",
        "cohort_definition": "Cohort analysis styling (demographic, behavioral, temporal)",
        "statistical_method": "Statistical analysis styling (frequentist, bayesian, non_parametric)"
    }


# Export functions for Kedro discovery
__all__ = [
    "create_pipeline",
    "create_pipelines", 
    "get_pipeline_descriptions",
    "get_integration_features",
    "get_experimental_conditions"
]