"""Advanced FigRegistry-Kedro Integration Example Project.

This package demonstrates enterprise-grade figregistry-kedro plugin capabilities
through a sophisticated multi-pipeline Kedro project showcasing automated figure
styling, versioning, and management at scale within complex data science workflows.
The example eliminates manual figure management overhead while demonstrating
production-ready deployment patterns and advanced experimental condition handling.

Advanced Features Demonstrated:
- Multi-pipeline architecture with training, inference, and reporting workflows
- Environment-specific configuration management (development, staging, production) 
- Complex experimental condition resolution with multi-variable parameter handling
- Enterprise-grade FigureDataSet integration with sophisticated catalog configurations
- Production-ready FigRegistryHooks deployment across distributed pipeline execution
- Advanced configuration bridge supporting environment-specific overrides (F-007)
- Scalable automated figure styling for high-volume visualization workflows (F-005)
- Sophisticated lifecycle management for complex multi-environment deployments (F-006)

Enterprise Integration Patterns:
This advanced example project follows enterprise Kedro conventions while showcasing
the figregistry-kedro plugin's ability to scale across complex organizational
workflows. The project demonstrates how automated figure styling integrates with
sophisticated experimental designs, multi-environment deployment scenarios, and
production-ready configuration management without requiring modifications to
existing enterprise pipeline logic.

Multi-Environment Architecture:
- conf/base/: Core configuration shared across all environments
- conf/local/: Development environment overrides and local customizations
- conf/staging/: Staging environment configuration for pre-production testing
- conf/production/: Production environment configuration with enterprise settings
- Complex parameter resolution supporting multi-treatment experimental designs
- Advanced catalog configurations with environment-specific output management

Production Deployment Features:
- Thread-safe configuration management for parallel pipeline execution
- Scalable figure generation supporting thousands of visualizations per workflow
- Advanced error handling and recovery patterns for enterprise environments
- Comprehensive logging and monitoring integration for production observability
- Security-compliant configuration management with credential isolation
- Performance optimization patterns for high-throughput visualization workflows

Project Structure:
- pipelines/training/: Machine learning model training with advanced visualizations
- pipelines/inference/: Model inference workflows with automated reporting figures
- pipelines/reporting/: Complex statistical analysis and presentation-ready outputs
- nodes.py: Enterprise-grade node functions demonstrating sophisticated figure creation
- pipeline_registry.py: Advanced pipeline composition and dependency management
- settings.py: Production-ready FigRegistryHooks configuration with environment support
- utils.py: Advanced utility functions for complex experimental condition management

Technical Architecture:
This example demonstrates figregistry-kedro integration patterns suitable for
enterprise data science platforms, showcasing how automated figure management
scales to support complex organizational workflows while maintaining the zero
external dependency philosophy that makes FigRegistry suitable for regulated
environments and air-gapped deployments.

Usage:
This package serves as both a functional enterprise-grade Kedro project and a
comprehensive reference implementation for deploying figregistry-kedro integration
in production data science platforms supporting multiple experimental scenarios,
environment-specific configuration management, and scalable visualization workflows.
"""

__version__ = "0.1.0"

# Package metadata for enterprise Kedro project discovery and plugin integration
__author__ = "FigRegistry Contributors"
__email__ = "contributors@figregistry.org"

# Public API for the advanced example project
# Following standard Python package conventions for enterprise namespace management
__all__ = [
    "__version__",
    "PROJECT_NAME",
    "PROJECT_VERSION", 
    "PACKAGE_NAME",
    "ADVANCED_FEATURES",
    "DEPLOYMENT_ENVIRONMENTS",
    "INTEGRATION_CAPABILITIES",
]

# Project identification constants for enterprise Kedro framework integration
PROJECT_NAME = "figregistry-kedro-advanced-example"
PROJECT_VERSION = __version__
PACKAGE_NAME = "figregistry_kedro_advanced_example"

# Technical metadata supporting F-008 requirements for enterprise plugin packaging
KEDRO_VERSION_REQUIREMENT = ">=0.18.0,<0.20.0"
FIGREGISTRY_VERSION_REQUIREMENT = ">=0.3.0"
PYTHON_VERSION_REQUIREMENT = ">=3.10"

# Advanced integration metadata for enterprise figregistry-kedro plugin demonstration
ADVANCED_FEATURES = [
    "F-005: Advanced FigureDataSet Integration with Complex Catalog Configurations",
    "F-006: Enterprise Lifecycle Hooks with Multi-Environment Support", 
    "F-007: Sophisticated Configuration Bridge with Environment-Specific Overrides",
    "Multi-Pipeline Architecture (Training, Inference, Reporting)",
    "Complex Experimental Condition Resolution",
    "Production-Ready Configuration Management",
    "Scalable Automated Figure Styling",
    "Enterprise-Grade Error Handling and Recovery",
    "Advanced Versioning Integration with Kedro Catalog System",
    "Distributed Pipeline Execution Support",
]

# Multi-environment deployment configuration for enterprise scenarios
DEPLOYMENT_ENVIRONMENTS = {
    "development": {
        "config_overrides": "conf/local/",
        "output_management": "local_filesystem",
        "figure_quality": "draft",
        "performance_mode": "debug",
        "logging_level": "DEBUG",
    },
    "staging": {
        "config_overrides": "conf/staging/",
        "output_management": "shared_storage", 
        "figure_quality": "high",
        "performance_mode": "optimized",
        "logging_level": "INFO",
    },
    "production": {
        "config_overrides": "conf/production/",
        "output_management": "enterprise_storage",
        "figure_quality": "publication",
        "performance_mode": "high_throughput",
        "logging_level": "WARNING",
    },
}

# Integration capabilities matrix for enterprise feature demonstration
INTEGRATION_CAPABILITIES = {
    "pipeline_architecture": {
        "multi_pipeline_support": True,
        "pipeline_composition": "advanced",
        "dependency_management": "sophisticated",
        "parallel_execution": True,
        "distributed_computing": True,
    },
    "configuration_management": {
        "environment_specific_overrides": True,
        "hierarchical_configuration": True,
        "parameter_resolution": "complex",
        "validation_framework": "pydantic",
        "hot_reloading": True,
    },
    "figure_management": {
        "automated_styling": "condition_based",
        "versioning_integration": "kedro_catalog", 
        "output_formats": ["png", "pdf", "svg", "eps"],
        "quality_levels": ["draft", "high", "publication"],
        "batch_processing": True,
    },
    "experimental_design": {
        "condition_complexity": "multi_variable",
        "parameter_sources": ["catalog", "pipeline", "environment"],
        "style_inheritance": "hierarchical",
        "fallback_mechanisms": "comprehensive",
        "validation_rules": "strict",
    },
    "enterprise_features": {
        "security_compliance": True,
        "audit_logging": True,
        "performance_monitoring": True,
        "error_recovery": "automatic",
        "scaling_patterns": "horizontal",
    },
}

# Educational classification for advanced example project categorization
EXAMPLE_METADATA = {
    "complexity_level": "advanced",
    "target_audience": "enterprise_data_science_teams",
    "use_case_category": "production_deployment_patterns",
    "integration_scope": "comprehensive_plugin_ecosystem", 
    "demonstrated_workflows": [
        "multi_pipeline_architecture",
        "environment_specific_configuration",
        "complex_experimental_condition_resolution",
        "enterprise_figure_management",
        "production_deployment_patterns",
        "scalable_visualization_workflows",
        "advanced_error_handling_recovery",
        "sophisticated_lifecycle_integration",
    ],
    "enterprise_considerations": [
        "thread_safe_configuration_management",
        "distributed_pipeline_execution_support",
        "security_compliant_credential_management", 
        "performance_optimized_figure_generation",
        "comprehensive_logging_monitoring_integration",
        "automated_testing_validation_frameworks",
    ],
}

# Pipeline composition metadata for advanced architecture demonstration
PIPELINE_COMPOSITION = {
    "training": {
        "purpose": "machine_learning_model_development",
        "figure_outputs": [
            "loss_curves",
            "validation_metrics", 
            "feature_importance_plots",
            "confusion_matrices",
            "roc_curves",
        ],
        "experimental_conditions": [
            "model_type",
            "hyperparameter_set",
            "dataset_version",
            "training_environment",
        ],
    },
    "inference": {
        "purpose": "model_prediction_workflows",
        "figure_outputs": [
            "prediction_distributions",
            "confidence_intervals",
            "batch_processing_metrics",
            "prediction_accuracy_plots",
        ],
        "experimental_conditions": [
            "model_version",
            "inference_batch_size",
            "prediction_threshold",
            "deployment_environment",
        ],
    },
    "reporting": {
        "purpose": "stakeholder_communication",
        "figure_outputs": [
            "executive_summaries",
            "performance_dashboards",
            "comparative_analyses",
            "trend_visualizations",
        ],
        "experimental_conditions": [
            "reporting_period",
            "audience_type",
            "presentation_format",
            "data_granularity",
        ],
    },
}