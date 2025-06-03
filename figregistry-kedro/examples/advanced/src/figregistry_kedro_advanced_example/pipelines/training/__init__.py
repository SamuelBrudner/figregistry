"""Advanced ML Training Pipeline Package for FigRegistry-Kedro Integration.

This package provides sophisticated machine learning training pipelines that demonstrate
advanced figregistry-kedro integration capabilities. The package implements comprehensive
automated figure management for complex experimental conditions while eliminating manual
plt.savefig() calls throughout training workflows per Section 0.1.1 primary objectives.

The training pipeline package showcases:
- Zero-touch figure management through FigureDataSet integration per F-005 requirements
- Advanced condition-based styling for complex experimental variables per F-002-RQ-002
- Sophisticated Kedro pipeline discovery and registration per F-008-RQ-004 specifications
- Enterprise-grade ML training visualization automation suitable for production workflows
- Multi-variable experimental condition resolution with wildcard support per F-002 requirements

This package serves as the primary demonstration of F-005 feature capabilities, showcasing
enterprise-grade automated figure management for complex ML training workflows with
sophisticated experimental condition handling per Section 0.2.1 implementation plan.

Package Structure:
    pipeline.py: Core pipeline implementations with advanced figregistry-kedro integration
    __init__.py: Package initialization with Kedro pipeline discovery support per F-008
    
Exported Functions:
    create_pipeline: Main training pipeline with sophisticated condition-based styling
    create_training_comparison_pipeline: A/B testing patterns for training strategies
    create_training_monitoring_pipeline: Real-time training monitoring with automated visualization
    create_complete_training_pipeline: Comprehensive training workflow demonstration
    
Example Usage:
    # Standard Kedro pipeline discovery pattern per F-008-RQ-004
    from figregistry_kedro_advanced_example.pipelines.training import create_pipeline
    
    # Advanced pipeline patterns for sophisticated experimental design
    from figregistry_kedro_advanced_example.pipelines.training import (
        create_training_comparison_pipeline,
        create_training_monitoring_pipeline, 
        create_complete_training_pipeline
    )
    
    # Create pipeline for Kedro project registration
    training_pipeline = create_pipeline()
    
Note:
    This package demonstrates F-005 integration requirements through comprehensive
    automated figure management with <5% performance overhead per F-005 specifications
    and full compatibility with Kedro's versioning system per F-005-RQ-002.
    
    All pipeline outputs use figregistry_kedro.FigureDataSet for zero-touch figure
    management, eliminating manual styling code while ensuring consistent, publication-ready
    visualizations across all experimental conditions per F-005-RQ-001 requirements.
"""

import logging
from typing import Optional

from kedro.pipeline import Pipeline

# Import all pipeline creation functions for advanced example demonstration
from .pipeline import (
    create_pipeline,
    create_training_comparison_pipeline, 
    create_training_monitoring_pipeline,
    create_complete_training_pipeline
)

# Configure package-level logging for training pipeline operations
logger = logging.getLogger(__name__)

# Package metadata for Kedro pipeline discovery per F-008 requirements
__version__ = "0.1.0"
__author__ = "FigRegistry-Kedro Integration Team"
__description__ = "Advanced ML Training Pipeline with Sophisticated FigRegistry Integration"

# Export primary pipeline creation function for Kedro discovery per F-008-RQ-004
__all__ = [
    "create_pipeline",                      # Primary pipeline for Kedro discovery
    "create_training_comparison_pipeline",  # A/B testing training strategies
    "create_training_monitoring_pipeline",  # Real-time training monitoring
    "create_complete_training_pipeline",    # Comprehensive training workflow
]


def get_pipeline_registry() -> dict[str, Pipeline]:
    """Get registry of all available training pipelines for advanced example demonstration.
    
    This function provides a comprehensive registry of all training pipeline variations
    available in the advanced example, supporting sophisticated experimental design
    patterns and demonstrating the full scope of figregistry-kedro integration capabilities.
    
    The registry enables users to selectively execute different aspects of the training
    workflow, from basic condition-based styling to sophisticated multi-variable
    experimental condition resolution with advanced automated figure management.
    
    Returns:
        dict[str, Pipeline]: Registry mapping pipeline names to Pipeline objects
            - "training": Main training pipeline with sophisticated condition-based styling
            - "training_comparison": A/B testing patterns for training strategy comparison
            - "training_monitoring": Real-time training monitoring with automated visualization
            - "training_complete": Comprehensive training workflow with full integration
            
    Example:
        # Access specific training pipeline variations
        registry = get_pipeline_registry()
        main_pipeline = registry["training"]
        ab_testing_pipeline = registry["training_comparison"]
        monitoring_pipeline = registry["training_monitoring"]
        complete_pipeline = registry["training_complete"]
        
    Note:
        All pipelines in the registry demonstrate F-005 feature requirements through
        automated figure management with sophisticated experimental condition handling
        per Section 0.2.1 implementation specifications.
        
        The registry supports CI/CD integration per F-008-RQ-004 requirements, enabling
        automated testing of all training pipeline variations in the advanced example.
    """
    logger.info("Building training pipeline registry for advanced example demonstration")
    
    # Build comprehensive pipeline registry for advanced example
    pipeline_registry = {
        "training": create_pipeline(),
        "training_comparison": create_training_comparison_pipeline(),
        "training_monitoring": create_training_monitoring_pipeline(),
        "training_complete": create_complete_training_pipeline()
    }
    
    # Log registry statistics for advanced example tracking
    total_pipelines = len(pipeline_registry)
    total_nodes = sum(len(pipeline.nodes) for pipeline in pipeline_registry.values())
    
    logger.info(
        f"Training pipeline registry created: {total_pipelines} pipeline variations, "
        f"{total_nodes} total nodes with advanced figregistry-kedro integration"
    )
    
    return pipeline_registry


def validate_pipeline_configuration() -> bool:
    """Validate training pipeline configuration for advanced example requirements.
    
    This function performs comprehensive validation of the training pipeline configuration
    to ensure compliance with F-005 integration requirements and F-008 packaging standards.
    The validation covers pipeline structure, figregistry-kedro integration, and advanced
    experimental condition support per Section 0.2.1 specifications.
    
    Returns:
        bool: True if all training pipelines pass validation requirements
        
    Raises:
        ValueError: If pipeline configuration fails validation requirements
        
    Note:
        This validation function supports F-008-RQ-004 CI/CD integration requirements
        by enabling automated validation of advanced example pipeline configurations
        before deployment and testing.
    """
    logger.info("Validating training pipeline configuration for advanced example requirements")
    
    try:
        # Validate primary training pipeline per F-005 requirements
        training_pipeline = create_pipeline()
        if not training_pipeline.nodes:
            raise ValueError("Primary training pipeline contains no nodes")
            
        # Validate FigureDataSet integration across all pipeline outputs
        figregistry_outputs = []
        for node in training_pipeline.nodes:
            figregistry_outputs.extend([
                output for output in node.outputs 
                if any(keyword in output for keyword in ["visualization", "analysis", "monitor"])
            ])
            
        if not figregistry_outputs:
            raise ValueError("Training pipeline missing figregistry-kedro FigureDataSet outputs")
            
        # Validate advanced pipeline variations for comprehensive demonstration
        comparison_pipeline = create_training_comparison_pipeline()
        monitoring_pipeline = create_training_monitoring_pipeline()
        complete_pipeline = create_complete_training_pipeline()
        
        # Ensure all pipeline variations are properly configured
        pipeline_variations = [
            ("training_comparison", comparison_pipeline),
            ("training_monitoring", monitoring_pipeline), 
            ("training_complete", complete_pipeline)
        ]
        
        for name, pipeline in pipeline_variations:
            if not pipeline.nodes:
                raise ValueError(f"Pipeline variation '{name}' contains no nodes")
                
        logger.info(
            f"Training pipeline validation successful: {len(figregistry_outputs)} FigureDataSet outputs, "
            f"{len(pipeline_variations) + 1} pipeline variations validated"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline validation failed: {e}")
        raise ValueError(f"Training pipeline configuration validation failed: {e}") from e


# Initialize package-level configuration for advanced example per F-008 requirements
logger.info(f"Initialized training pipeline package v{__version__} with advanced figregistry-kedro integration")

# Validate pipeline configuration on package import for early error detection
try:
    _validation_result = validate_pipeline_configuration()
    logger.info("Training pipeline package validation completed successfully")
except Exception as e:
    logger.warning(f"Training pipeline package validation warning: {e}")
    # Allow package import to continue for development flexibility