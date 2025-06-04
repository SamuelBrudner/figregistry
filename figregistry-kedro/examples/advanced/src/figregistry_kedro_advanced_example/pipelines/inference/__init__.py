"""Advanced Inference Pipeline Package for FigRegistry-Kedro Integration.

This package provides the inference pipeline module for the advanced figregistry-kedro
example, demonstrating sophisticated ML inference workflows with comprehensive automated
figure management. The package enables proper discovery and registration of the inference
pipeline by Kedro's framework while showcasing enterprise-grade visualization patterns.

Package Features:
- Pipeline discovery support per F-008 requirements for Kedro framework integration
- Advanced inference pipeline with zero-touch figure management per Section 0.1.1 objectives  
- Sophisticated condition-based styling for ML inference scenarios per F-005-RQ-004
- Enterprise-grade visualization patterns suitable for production ML workflows
- Integration with Kedro's versioning system for inference figure outputs per F-005-RQ-002
- Comprehensive experimental parameter resolution and style management per F-002 requirements

The package exports the create_pipeline function that implements advanced figregistry-kedro
integration patterns, demonstrating the elimination of manual plt.savefig() calls while
providing sophisticated, production-ready inference visualizations through automated
FigureDataSet management.

Exported Functions:
    create_pipeline: Creates the advanced inference pipeline with figregistry-kedro integration
    create_inference_pipeline_with_custom_conditions: Creates customized inference pipeline
    get_inference_pipeline_metadata: Provides comprehensive pipeline metadata and capabilities

Integration Requirements:
    - kedro>=0.18.0,<0.20.0: Core pipeline framework per F-005 dependencies
    - figregistry>=0.3.0: Core visualization configuration system
    - figregistry-kedro>=0.1.0: FigureDataSet and hooks integration per F-005 specifications
    - matplotlib>=3.9.0: Advanced plotting capabilities for inference visualizations

For detailed pipeline configuration and advanced usage patterns, see the pipeline.py module
documentation and the figregistry-kedro integration guide.
"""

# Import the pipeline creation function for Kedro framework discovery per F-008-RQ-001
from .pipeline import (
    create_pipeline,
    create_inference_pipeline_with_custom_conditions,
    get_inference_pipeline_metadata
)

# Package metadata for pipeline discovery and registration per F-008 requirements
__version__ = "0.1.0"
__author__ = "FigRegistry-Kedro Integration Team"
__description__ = "Advanced ML inference pipeline with automated figure management"

# Export main pipeline creation function for Kedro discovery per Section 0.2.1 implementation plan
__all__ = [
    "create_pipeline",
    "create_inference_pipeline_with_custom_conditions", 
    "get_inference_pipeline_metadata"
]