"""Inference Pipeline Package.

This module provides the pipeline initialization for the inference pipeline
in the advanced FigRegistry-Kedro integration example. The inference pipeline
demonstrates sophisticated condition-based styling for ML inference scenarios,
showcasing how FigRegistry can automatically apply different visualization 
styles based on inference parameters and model performance metrics.

The pipeline package follows Kedro conventions for pipeline discovery and
registration, enabling the framework to properly import and execute the
inference pipeline components within the broader ML workflow.

Example Usage:
    The pipeline is automatically discovered by Kedro through this initialization:
    
    ```python
    # In your pipeline_registry.py
    from figregistry_kedro_advanced_example.pipelines.inference import create_pipeline
    
    inference_pipeline = create_pipeline()
    ```

Components:
    - create_pipeline: Factory function that returns configured inference pipeline
    - Pipeline nodes for model inference and results visualization
    - FigRegistry integration for automatic inference plot styling

Integration Features:
    - Condition-based styling for different inference scenarios
    - Automated figure versioning aligned with model versions
    - Performance-based visualization styling (accuracy, precision, recall)
    - Environment-specific output formatting (dev, staging, production)
"""

from .pipeline import create_pipeline

# Export the pipeline creation function for Kedro discovery
__all__ = ["create_pipeline"]