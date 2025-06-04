"""
Reporting Pipeline Package for FigRegistry-Kedro Advanced Example

This package provides enterprise-grade reporting pipeline functionality demonstrating
sophisticated figregistry-kedro integration patterns. The package enables Kedro's
framework to discover and load reporting pipeline components through standardized
Python package initialization and export patterns.

Key Features:
- F-008 compliant pipeline package initialization for Kedro discovery
- F-005 advanced FigureDataSet integration examples across multiple reporting scenarios
- Enterprise-grade automated figure management eliminating manual styling intervention
- Multiple pipeline variants for different enterprise reporting use cases

Pipeline Variants Available:
- Main Reporting Pipeline: Comprehensive enterprise reporting with automated styling
- Presentation Pipeline: Stakeholder communication with presentation-optimized formatting
- Publication Pipeline: Academic/professional publication with journal-quality outputs
- Combined Enterprise Pipeline: Master pipeline integrating all reporting scenarios

The package follows Kedro's standard pipeline discovery patterns, enabling seamless
integration with pipeline registries and project configuration through proper namespace
management and import structure.

Technical Integration:
- All pipelines demonstrate F-005-RQ-004 advanced context injection for conditional styling
- F-002 sophisticated condition-based styling with complex experimental conditions
- F-005-RQ-002 integration with Kedro versioning for publication-ready outputs
- Enterprise-grade zero-touch figure management through FigureDataSet automation

Usage:
    from figregistry_kedro_advanced_example.pipelines.reporting import create_pipeline
    
    # Create the main reporting pipeline
    reporting_pipeline = create_pipeline()
    
    # Or create specialized variants
    from figregistry_kedro_advanced_example.pipelines.reporting import (
        create_presentation_pipeline,
        create_publication_pipeline,
        create_combined_enterprise_pipeline
    )
"""

# Import pipeline creation functions from the pipeline module
# This enables Kedro's pipeline discovery mechanism per F-008 requirements
from .pipeline import (
    create_pipeline,
    create_presentation_pipeline,
    create_publication_pipeline,
    create_combined_enterprise_pipeline,
)

# Export pipeline creation functions for Kedro framework discovery
# This follows standard Kedro pipeline package conventions per Section 0.2.1
__all__ = [
    "create_pipeline",
    "create_presentation_pipeline", 
    "create_publication_pipeline",
    "create_combined_enterprise_pipeline",
]

# Package metadata for documentation and discovery
__version__ = "1.0.0"
__description__ = "Advanced enterprise reporting pipeline with sophisticated figregistry-kedro integration"
__author__ = "FigRegistry-Kedro Integration Team"

# Pipeline package identification for Kedro registry
# This enables proper pipeline loading and namespace management
PIPELINE_NAME = "reporting"
PIPELINE_DESCRIPTION = (
    "Enterprise-grade reporting pipeline demonstrating advanced figregistry-kedro "
    "integration with automated figure styling, condition-based formatting, and "
    "sophisticated output management across multiple enterprise reporting scenarios"
)

# Pipeline capability metadata for documentation and discovery
PIPELINE_CAPABILITIES = {
    "automated_styling": True,
    "condition_based_formatting": True,
    "multi_audience_support": True,
    "publication_quality": True,
    "enterprise_integration": True,
    "figregistry_integration": "advanced",
    "supported_outputs": ["executive", "technical", "publication", "presentation"],
    "demonstrates_features": ["F-005", "F-002", "F-008"],
}

# Configuration hints for pipeline usage
PIPELINE_CONFIGURATION_HINTS = {
    "required_catalog_entries": [
        "executive_dashboard_figure",
        "statistical_report_figure", 
        "ab_testing_analysis_figure",
        "model_inference_analysis_figure",
        "hyperparameter_optimization_figure",
        "training_metrics_dashboard_figure",
    ],
    "required_parameters": [
        "executive_reporting_context",
        "statistical_analysis_config",
        "publication_context",
        "ab_testing_experiment_config",
    ],
    "figregistry_features_used": [
        "condition_based_styling",
        "automated_output_management", 
        "purpose_driven_formatting",
        "audience_specific_conditions",
    ],
}