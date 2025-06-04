"""
Advanced Enterprise Reporting Pipeline for FigRegistry-Kedro Integration

This pipeline demonstrates the pinnacle of automated figure management through sophisticated
figregistry-kedro integration patterns. It showcases enterprise-grade reporting workflows
that automatically generate publication-ready visualizations with condition-based styling
for different audiences (technical, executive, publication) and output formats without
any manual styling intervention.

Key Demonstrations:
- F-005 Complete elimination of manual plt.savefig() calls through FigureDataSet automation
- F-002 Sophisticated condition-based styling for enterprise experimental conditions
- F-005-RQ-004 Advanced context injection and parameter resolution for conditional styling
- F-005-RQ-002 Integration with Kedro's versioning system for publication-ready outputs
- Enterprise-grade automated figure management transforming analysis into polished reports

The pipeline implements the most complex figregistry-kedro integration scenarios including:
- Multiple audience-specific styling conditions (technical/executive/publication)
- Dynamic output purpose resolution (exploratory/presentation/publication)
- Advanced experimental condition mapping with wildcard and hierarchical matching
- Automated style inheritance and override patterns for enterprise consistency
- Zero-touch figure generation with complete separation of analysis logic and presentation

Technical Architecture:
- All nodes output matplotlib.Figure objects consumed by FigureDataSet
- Complex experimental conditions resolved through pipeline parameter inheritance
- Sophisticated catalog configuration with purpose categorization and condition mapping
- Advanced style resolution demonstrating F-002-RQ-002 wildcard and partial matching
- Enterprise-grade error handling and performance monitoring integration
"""

import logging
from typing import Dict, List, Any, Optional
from kedro.pipeline import Pipeline, node, pipeline

# Import advanced node functions demonstrating enterprise reporting patterns
from figregistry_kedro_advanced_example.nodes import (
    # Executive reporting nodes
    create_executive_performance_dashboard,
    create_statistical_analysis_report,
    
    # Business intelligence and analysis nodes  
    create_ab_testing_analysis,
    create_model_inference_analysis,
    create_hyperparameter_optimization_analysis,
    
    # Advanced training and technical analysis nodes
    create_advanced_training_metrics_dashboard,
    
    # Utility functions for experimental configuration
    validate_experimental_configuration,
    prepare_condition_hierarchy,
    calculate_enterprise_performance_metrics
)

# Configure pipeline logger
logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create sophisticated enterprise reporting pipeline demonstrating advanced figregistry-kedro integration.
    
    This function constructs the most advanced example of automated figure management within
    Kedro workflows, showcasing elimination of manual styling through FigureDataSet integration
    with sophisticated condition-based styling for enterprise reporting scenarios.
    
    The pipeline demonstrates:
    - F-005 requirements: Complete automated figure styling through FigureDataSet integration
    - F-002 requirements: Sophisticated condition-based styling with complex experimental conditions
    - F-005-RQ-004: Advanced context injection for conditional styling from pipeline parameters
    - F-005-RQ-002: Integration with Kedro versioning for publication-ready figure outputs
    - Enterprise-grade zero-touch figure management eliminating manual plt.savefig() calls
    
    Pipeline Architecture:
    1. Data preparation and experimental condition validation
    2. Executive dashboard generation with audience-specific styling
    3. Technical analysis reporting with publication-ready formatting
    4. Business intelligence visualization with presentation-quality styling
    5. Statistical analysis with academic/professional publication formatting
    6. Advanced experimental analysis with complex condition resolution
    
    All figure outputs are handled automatically by FigureDataSet with:
    - Purpose-driven styling (exploratory, presentation, publication)
    - Audience-specific conditions (technical, executive, publication)
    - Output format optimization (screen, print, publication)
    - Automated versioning and experiment tracking integration
    
    Returns:
        Pipeline: Sophisticated reporting pipeline with automated figure management
        
    Note:
        Catalog configuration must include FigureDataSet entries with appropriate:
        - purpose: "exploratory" | "presentation" | "publication"
        - condition_param: Parameter name for dynamic condition resolution
        - style_params: Dataset-specific styling overrides
        - versioned: true for experiment tracking integration
    """
    logger.info("Creating advanced enterprise reporting pipeline with figregistry-kedro integration")
    
    # Define sophisticated reporting pipeline with automated figure management
    reporting_pipeline = pipeline([
        
        # 1. Executive Performance Dashboard - Publication Quality
        # Demonstrates F-005-RQ-004 with executive audience condition resolution
        node(
            func=create_executive_performance_dashboard,
            inputs=[
                "kpi_performance_data",
                "trend_analysis_results", 
                "comparative_business_metrics",
                "params:executive_reporting_context"
            ],
            outputs="executive_dashboard_figure",
            name="generate_executive_performance_dashboard",
            tags=["reporting", "executive", "publication", "high_priority"],
        ),
        
        # 2. Statistical Analysis Report - Academic Publication Format
        # Demonstrates F-002 with publication-grade condition-based styling
        node(
            func=create_statistical_analysis_report,
            inputs=[
                "comprehensive_statistical_results",
                "dataset_analysis_metadata",
                "params:statistical_analysis_config",
                "params:publication_context"
            ],
            outputs="statistical_report_figure",
            name="generate_statistical_analysis_report", 
            tags=["reporting", "publication", "technical", "peer_review"],
        ),
        
        # 3. A/B Testing Analysis - Business Intelligence Dashboard
        # Demonstrates sophisticated experimental condition mapping per F-002-RQ-002
        node(
            func=validate_experimental_configuration,
            inputs="params:ab_testing_experiment_config",
            outputs="validated_ab_testing_config",
            name="validate_ab_testing_configuration",
            tags=["validation", "experimental_design"],
        ),
        
        node(
            func=create_ab_testing_analysis,
            inputs=[
                "ab_testing_experiment_data",
                "validated_ab_testing_config",
                "statistical_significance_tests",
                "params:business_impact_metrics"
            ],
            outputs="ab_testing_analysis_figure",
            name="generate_ab_testing_analysis",
            tags=["reporting", "business_intelligence", "experimental"],
        ),
        
        # 4. Model Inference Analysis - Technical Deep Dive
        # Demonstrates F-005-RQ-004 with model performance condition resolution
        node(
            func=create_model_inference_analysis,
            inputs=[
                "model_inference_results",
                "model_ground_truth_data",
                "params:production_model_config",
                "params:deployment_environment_context"
            ],
            outputs="model_inference_analysis_figure",
            name="generate_model_inference_analysis",
            tags=["reporting", "technical", "model_performance"],
        ),
        
        # 5. Hyperparameter Optimization Report - Research Documentation
        # Demonstrates F-002 with optimization algorithm condition-based styling
        node(
            func=create_hyperparameter_optimization_analysis,
            inputs=[
                "hyperparameter_optimization_results",
                "optimal_hyperparameter_config",
                "params:optimization_methodology_metadata",
                "params:research_experimental_conditions"
            ],
            outputs="hyperparameter_optimization_figure",
            name="generate_hyperparameter_optimization_report",
            tags=["reporting", "research", "optimization", "technical"],
        ),
        
        # 6. Advanced Training Metrics Dashboard - Development/Operations
        # Demonstrates F-005 with training regime condition resolution
        node(
            func=create_advanced_training_metrics_dashboard,
            inputs=[
                "model_training_history",
                "model_validation_history",
                "params:model_architecture_metadata",
                "params:training_experimental_conditions"
            ],
            outputs="training_metrics_dashboard_figure",
            name="generate_training_metrics_dashboard",
            tags=["reporting", "training", "development", "monitoring"],
        ),
        
        # 7. Executive Summary Generation - Multi-Format Output
        # Demonstrates F-005-RQ-002 with versioning for different output formats
        node(
            func=create_executive_performance_dashboard,
            inputs=[
                "executive_summary_kpi_data",
                "executive_summary_trends",
                "executive_summary_comparisons", 
                "params:board_presentation_context"
            ],
            outputs="board_presentation_figure",
            name="generate_board_presentation_summary",
            tags=["reporting", "executive", "board", "presentation"],
        ),
        
        # 8. Technical Documentation Report - Internal Documentation
        # Demonstrates complex condition hierarchy with technical audience styling
        node(
            func=create_statistical_analysis_report,
            inputs=[
                "internal_technical_analysis",
                "technical_documentation_metadata",
                "params:internal_documentation_config",
                "params:technical_team_context"
            ],
            outputs="technical_documentation_figure",
            name="generate_technical_documentation",
            tags=["reporting", "documentation", "internal", "technical"],
        ),
        
        # 9. Performance Metrics Calculation - Supporting Analysis
        # Demonstrates advanced data processing for condition resolution
        node(
            func=calculate_enterprise_performance_metrics,
            inputs="consolidated_business_data",
            outputs="calculated_enterprise_metrics",
            name="calculate_performance_metrics",
            tags=["analysis", "metrics", "preprocessing"],
        ),
        
        # 10. Condition Hierarchy Preparation - Advanced Configuration
        # Demonstrates F-005-RQ-004 sophisticated parameter resolution setup
        node(
            func=prepare_condition_hierarchy,
            inputs=[
                "validated_ab_testing_config",
                "calculated_enterprise_metrics",
                "params:enterprise_business_context"
            ],
            outputs="enterprise_condition_hierarchy",
            name="prepare_advanced_condition_hierarchy",
            tags=["configuration", "condition_resolution", "advanced"],
        ),
        
        # 11. Multi-Audience Summary Report - Adaptive Styling
        # Demonstrates F-002-RQ-002 with audience-adaptive condition matching
        node(
            func=create_executive_performance_dashboard,
            inputs=[
                "multi_audience_summary_data",
                "adaptive_trend_analysis",
                "audience_specific_metrics",
                "enterprise_condition_hierarchy"
            ],
            outputs="multi_audience_summary_figure", 
            name="generate_multi_audience_summary",
            tags=["reporting", "adaptive", "multi_audience", "advanced"],
        ),
        
    ], namespace="reporting")
    
    logger.info(
        f"Advanced enterprise reporting pipeline created with {len(reporting_pipeline.nodes)} nodes "
        f"demonstrating sophisticated figregistry-kedro integration patterns"
    )
    
    return reporting_pipeline


def create_presentation_pipeline(**kwargs) -> Pipeline:
    """
    Create specialized presentation pipeline for high-stakes stakeholder communication.
    
    This pipeline demonstrates F-005 integration optimized for presentation scenarios
    with emphasis on visual clarity, audience engagement, and professional formatting.
    All outputs use presentation-optimized styling through automated FigureDataSet
    condition resolution.
    
    Returns:
        Pipeline: Presentation-focused pipeline with automated styling for stakeholder communication
    """
    logger.info("Creating specialized presentation pipeline for stakeholder communication")
    
    presentation_pipeline = pipeline([
        
        # High-level executive overview with presentation styling
        node(
            func=create_executive_performance_dashboard,
            inputs=[
                "presentation_kpi_summary",
                "stakeholder_trend_highlights",
                "competitive_position_metrics",
                "params:stakeholder_presentation_context"
            ],
            outputs="stakeholder_presentation_figure",
            name="generate_stakeholder_presentation_dashboard",
            tags=["presentation", "stakeholder", "high_level"],
        ),
        
        # Business case visualization with impact emphasis
        node(
            func=create_ab_testing_analysis,
            inputs=[
                "business_case_experiment_data",
                "validated_ab_testing_config",
                "roi_impact_analysis",
                "params:business_case_presentation_context"
            ],
            outputs="business_case_presentation_figure",
            name="generate_business_case_presentation",
            tags=["presentation", "business_case", "roi"],
        ),
        
        # Technology overview with accessible technical presentation
        node(
            func=create_model_inference_analysis,
            inputs=[
                "technology_showcase_results",
                "technology_validation_data",
                "params:technology_presentation_config",
                "params:accessible_technical_context"
            ],
            outputs="technology_presentation_figure",
            name="generate_technology_presentation",
            tags=["presentation", "technology", "accessible"],
        ),
        
    ], namespace="presentation")
    
    return presentation_pipeline


def create_publication_pipeline(**kwargs) -> Pipeline:
    """
    Create academic/professional publication pipeline with journal-quality outputs.
    
    This pipeline demonstrates F-005 integration optimized for academic and professional
    publication scenarios with strict formatting requirements, statistical rigor, and
    peer-review quality standards through automated FigureDataSet styling.
    
    Returns:
        Pipeline: Publication-quality pipeline with automated academic formatting
    """
    logger.info("Creating publication-quality pipeline for academic and professional publication")
    
    publication_pipeline = pipeline([
        
        # Journal article statistical analysis with peer-review formatting
        node(
            func=create_statistical_analysis_report,
            inputs=[
                "peer_review_statistical_analysis",
                "journal_submission_metadata",
                "params:journal_publication_config",
                "params:peer_review_context"
            ],
            outputs="journal_statistical_figure",
            name="generate_journal_statistical_analysis",
            tags=["publication", "journal", "peer_review", "statistical"],
        ),
        
        # Conference presentation technical deep-dive
        node(
            func=create_hyperparameter_optimization_analysis,
            inputs=[
                "conference_optimization_study",
                "research_methodology_config",
                "params:conference_publication_metadata",
                "params:academic_research_context"
            ],
            outputs="conference_technical_figure",
            name="generate_conference_technical_analysis",
            tags=["publication", "conference", "technical", "research"],
        ),
        
        # White paper business analysis with professional formatting
        node(
            func=create_ab_testing_analysis,
            inputs=[
                "white_paper_experiment_analysis",
                "validated_ab_testing_config",
                "industry_benchmark_comparisons",
                "params:white_paper_publication_context"
            ],
            outputs="white_paper_analysis_figure",
            name="generate_white_paper_analysis",
            tags=["publication", "white_paper", "professional", "industry"],
        ),
        
    ], namespace="publication")
    
    return publication_pipeline


def create_combined_enterprise_pipeline(**kwargs) -> Pipeline:
    """
    Create comprehensive enterprise pipeline combining all reporting scenarios.
    
    This master pipeline demonstrates the full scope of figregistry-kedro integration
    across all enterprise reporting use cases, showcasing the complete elimination of
    manual figure styling through sophisticated automated condition resolution.
    
    Returns:
        Pipeline: Comprehensive enterprise pipeline with complete automated figure management
    """
    logger.info("Creating comprehensive enterprise pipeline with complete figregistry-kedro integration")
    
    # Combine all specialized pipelines for comprehensive enterprise reporting
    comprehensive_pipeline = (
        create_pipeline(**kwargs) +
        create_presentation_pipeline(**kwargs) + 
        create_publication_pipeline(**kwargs)
    )
    
    logger.info(
        f"Comprehensive enterprise pipeline created with {len(comprehensive_pipeline.nodes)} total nodes "
        f"across reporting, presentation, and publication scenarios"
    )
    
    return comprehensive_pipeline


# Export pipeline creation functions for flexible usage
__all__ = [
    "create_pipeline",
    "create_presentation_pipeline", 
    "create_publication_pipeline",
    "create_combined_enterprise_pipeline"
]