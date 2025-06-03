"""Advanced Reporting Pipeline Package - Enterprise FigRegistry-Kedro Integration.

This package provides the most sophisticated reporting pipeline demonstrating automated
figure styling, advanced condition-based visualization management, and enterprise-grade
zero-touch figure processing within Kedro's catalog-based data processing framework.
The pipeline showcases publication-ready figure generation with multi-audience styling
for technical deep-dives, executive summaries, and peer-reviewed research outputs.

Pipeline Functions:
    create_pipeline(): Advanced reporting pipeline demonstrating the most sophisticated
                      figregistry-kedro integration patterns across multiple audiences,
                      output formats, and complex experimental condition scenarios

Advanced Integration Features Demonstrated:
    - F-005: Enterprise-grade automated figure styling via FigureDataSet integration
    - F-002: Sophisticated condition-based styling with multi-variable parameters
    - F-005-RQ-001: Complete zero-touch figure processing (elimination of manual plt.savefig())
    - F-005-RQ-002: Advanced Kedro versioning integration for publication-ready outputs
    - F-005-RQ-004: Complex context injection for hierarchical conditional styling
    - F-002-RQ-002: Advanced experimental condition resolution with wildcard matching
    - F-004: Multiple sophisticated output purposes (technical analysis, executive reporting, 
             publication materials, cross-format presentations)

Enterprise Reporting Capabilities:
    - Technical Deep-Dive Reports: Sophisticated analysis for engineering audiences
    - Executive Summary Reports: Business-focused visualizations for decision makers  
    - Publication-Ready Figures: Peer-reviewed research quality outputs
    - Cross-Format Reporting: Adaptive styling for multiple presentation contexts
    - Advanced Experimental Condition Mapping: Complex parameter resolution patterns
    - Audience-Specific Styling: Dynamic figure formatting based on target audience
    - Multi-Variable Condition Resolution: Sophisticated experimental parameter handling

Usage:
    The pipeline function is automatically discoverable by Kedro's framework
    and can be registered in pipeline_registry.py for advanced reporting workflows:
    
    ```python
    from figregistry_kedro_advanced_example.pipelines.reporting import (
        create_pipeline as create_reporting_pipeline
    )
    
    def register_pipelines() -> Dict[str, Pipeline]:
        return {
            "reporting": create_reporting_pipeline(),
            "advanced_reporting": create_reporting_pipeline(),
            "__default__": create_reporting_pipeline(),
        }
    ```

Architecture Highlights:
    - Modular Subpipeline Design: Technical, executive, publication, and cross-format
      reporting components with namespace isolation and parameter specialization
    - Sophisticated Condition Resolution: Multi-level experimental parameter mapping
      with wildcard support and partial condition matching per F-002-RQ-002
    - Enterprise-Grade Output Management: Publication-ready figure generation with
      automated versioning, audience-specific styling, and format optimization
    - Advanced Integration Patterns: Demonstrates the pinnacle of figregistry-kedro
      capabilities for complex enterprise reporting requirements

Requirements Fulfilled:
    - F-008: Valid Python package initialization for Kedro pipeline discovery
    - Section 0.2.1: Advanced import patterns for pipeline registry access
    - F-005: Complete Kedro framework integration with sophisticated dataset handling
    - F-002: Advanced condition-based styling with complex parameter resolution
    - Section 0.1.2: Advanced example structure demonstrating enterprise patterns
    - Standard Python package conventions for robust namespace management
"""

from .pipeline import create_pipeline

__all__ = [
    "create_pipeline"
]

# Advanced package metadata for enterprise integration verification
__package_info__ = {
    "name": "reporting",
    "description": "Advanced enterprise reporting pipeline with sophisticated figregistry-kedro integration",
    "version": "0.1.0",
    "sophistication_level": "enterprise",
    "audience_support": [
        "technical_deep_dive",
        "executive_summary", 
        "publication_ready",
        "cross_format_adaptive"
    ],
    "advanced_integration_features": [
        "F-005: Enterprise FigureDataSet integration",
        "F-002: Advanced condition-based styling with multi-variable parameters",
        "F-004: Sophisticated automated output management",
        "F-005-RQ-001: Complete zero-touch figure processing",
        "F-005-RQ-002: Advanced Kedro versioning integration for publications",
        "F-005-RQ-004: Complex context injection for hierarchical styling",
        "F-002-RQ-002: Advanced experimental condition resolution with wildcards"
    ],
    "experimental_condition_complexity": [
        "wildcard_matching",
        "partial_condition_resolution", 
        "hierarchical_parameter_mapping",
        "multi_variable_condition_handling",
        "audience_specific_parameter_injection"
    ],
    "reporting_capabilities": [
        "technical_analysis_reports",
        "executive_decision_dashboards",
        "publication_quality_figures", 
        "cross_format_presentation_materials",
        "enterprise_grade_visualizations"
    ],
    "kedro_compatibility": ">=0.18.0,<0.20.0",
    "figregistry_compatibility": ">=0.3.0",
    "enterprise_features": {
        "automated_styling": True,
        "publication_ready": True,
        "multi_audience_support": True,
        "complex_condition_resolution": True,
        "zero_touch_processing": True,
        "version_integration": True,
        "namespace_isolation": True
    }
}

# Pipeline discovery metadata for Kedro framework integration per F-008 requirements
__kedro_pipeline__ = {
    "pipeline_name": "reporting",
    "create_function": "create_pipeline",
    "description": "Advanced enterprise reporting pipeline with sophisticated figregistry-kedro integration",
    "integration_level": "enterprise",
    "demonstration_focus": "advanced_automated_figure_management"
}