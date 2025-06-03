"""
Migration example pipeline demonstrating automated figure management through figregistry-kedro integration.

This pipeline showcases the 'after' state where manual matplotlib figure management has been completely
eliminated through FigureDataSet automation. The pipeline demonstrates:

- Zero plt.savefig() calls in node functions - all handled automatically by FigureDataSet
- Condition-based styling through catalog configuration and pipeline parameters
- Automatic versioning integration with Kedro's catalog system
- Different output purposes (exploratory, presentation) for comprehensive styling demonstration
- Complete separation between visualization logic and figure management concerns

Key Migration Benefits Demonstrated:
1. Elimination of manual figure management overhead
2. Centralized styling configuration through figregistry.yml integration
3. Automatic application of experimental condition-based styling
4. Seamless integration with Kedro's versioning and experiment tracking
5. Reduced code duplication and maintenance burden

Usage:
    This pipeline is designed to be run as part of a Kedro project with figregistry-kedro
    integration enabled. The catalog configuration automatically handles all figure styling
    and persistence based on the configured purpose and condition parameters.

Example Kedro Command:
    kedro run --pipeline=data_visualization --params="experiment_condition:production"
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...nodes import (
    create_exploratory_scatter_plot,
    create_time_series_analysis,
    create_categorical_summary,
    create_comparative_analysis,
    create_statistical_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create data visualization pipeline demonstrating figregistry-kedro automation.
    
    This pipeline represents the 'after' state of migration where all manual figure
    management has been eliminated. Each node outputs raw matplotlib figures that
    are automatically styled and persisted by FigureDataSet based on catalog
    configuration and experimental conditions.
    
    Key Automation Features Demonstrated:
    - FigureDataSet intercepts all figure outputs automatically (F-005-RQ-001)
    - Condition-based styling via pipeline parameter resolution (F-005-RQ-004)
    - Automated versioning through Kedro's catalog system (F-005-RQ-002)
    - Different output purposes for comprehensive styling demonstration (F-002)
    - Complete elimination of manual plt.savefig() calls (Section 0.1.1)
    
    Pipeline Architecture:
    The pipeline flows through several visualization stages, each demonstrating
    different aspects of automated figure management:
    
    1. Exploratory Analysis Stage:
       - Scatter plot analysis with automatic styling based on experimental conditions
       - Time series analysis with rolling statistics and trend detection
       - Demonstrates 'exploratory' purpose for rapid iteration styling
    
    2. Categorical Analysis Stage:
       - Summary statistics and distribution analysis
       - Cross-tabulation and success rate comparisons
       - Shows automated handling of complex multi-panel figures
    
    3. Comparative Analysis Stage:
       - Statistical comparisons between datasets
       - Demonstrates 'presentation' purpose for formal reporting styling
    
    4. Comprehensive Reporting Stage:
       - Statistical summary with publication-quality formatting
       - Demonstrates 'publication' purpose for final output styling
    
    Catalog Integration:
    Each output is configured in the catalog with figregistry-kedro specific
    parameters that enable automated styling:
    
    - purpose: 'exploratory', 'presentation', or 'publication'
    - condition_param: Parameter name for dynamic condition resolution
    - style_params: Dataset-specific styling overrides
    
    Example catalog configuration for one of the outputs:
    ```yaml
    exploratory_scatter_plot:
      type: figregistry_kedro.FigureDataSet
      filepath: data/08_reporting/exploratory_scatter_analysis.png
      purpose: exploratory
      condition_param: experiment_condition
      style_params:
        figure.dpi: 150
        figure.facecolor: white
      save_args:
        bbox_inches: tight
        transparent: false
    ```
    
    Returns:
        Pipeline: Configured Kedro pipeline with automated figure management
    """
    
    # Define the main visualization pipeline with automated figure management
    # All nodes output raw matplotlib figures - no manual styling or saving required
    main_pipeline = pipeline([
        # Stage 1: Exploratory Analysis
        # These nodes demonstrate rapid iteration styling for data exploration
        node(
            func=create_exploratory_scatter_plot,
            inputs=["model_input_table", "params:visualization_params"],
            outputs="exploratory_scatter_plot",
            name="generate_exploratory_scatter_plot",
            tags=["exploration", "correlation_analysis"],
        ),
        
        node(
            func=create_time_series_analysis,
            inputs=["time_series_data", "params:visualization_params"], 
            outputs="time_series_analysis_plot",
            name="generate_time_series_analysis",
            tags=["exploration", "temporal_analysis"],
        ),
        
        # Stage 2: Categorical Analysis
        # Demonstrates automated styling for categorical data visualizations
        node(
            func=create_categorical_summary,
            inputs=["categorical_features", "params:visualization_params"],
            outputs="categorical_summary_plot",
            name="generate_categorical_summary",
            tags=["analysis", "categorical_data"],
        ),
        
        # Stage 3: Comparative Analysis  
        # Shows presentation-quality styling for formal reports
        node(
            func=create_comparative_analysis,
            inputs=["primary_dataset", "secondary_dataset", "params:visualization_params"],
            outputs="comparative_analysis_plot", 
            name="generate_comparative_analysis",
            tags=["comparison", "statistical_testing"],
        ),
        
        # Stage 4: Comprehensive Statistical Summary
        # Demonstrates publication-quality styling for final outputs
        node(
            func=create_statistical_summary,
            inputs=["combined_analysis_data", "params:visualization_params"],
            outputs="statistical_summary_report",
            name="generate_statistical_summary", 
            tags=["reporting", "publication_ready"],
        ),
    ])
    
    # Return the modular pipeline with namespace support
    # This enables easy integration into larger Kedro projects while
    # maintaining clear separation of concerns for figure management
    return modular_pipeline(
        main_pipeline,
        inputs={
            # Input datasets - these would be defined in the catalog
            # as regular Kedro datasets (CSV, Parquet, etc.)
            "model_input_table",
            "time_series_data", 
            "categorical_features",
            "primary_dataset",
            "secondary_dataset", 
            "combined_analysis_data",
        },
        outputs={
            # Output datasets - these are configured in the catalog as
            # figregistry_kedro.FigureDataSet instances with automated styling
            "exploratory_scatter_plot",
            "time_series_analysis_plot",
            "categorical_summary_plot", 
            "comparative_analysis_plot",
            "statistical_summary_report",
        },
        parameters={
            # Pipeline parameters for condition resolution and customization
            "visualization_params": "params:visualization_params",
        },
        namespace="data_visualization",
    )


def create_exploratory_pipeline(**kwargs) -> Pipeline:
    """
    Create a focused exploratory analysis pipeline subset.
    
    This pipeline demonstrates rapid iteration capabilities with automated
    styling optimized for exploratory data analysis. All outputs use the
    'exploratory' purpose for fast iteration styling.
    
    Benefits for Migration:
    - Shows how exploratory analysis becomes more efficient with automation
    - Demonstrates consistent styling across exploratory outputs
    - Eliminates manual figure management overhead in research workflows
    
    Returns:
        Pipeline: Exploratory analysis pipeline with automated styling
    """
    return pipeline([
        node(
            func=create_exploratory_scatter_plot,
            inputs=["model_input_table", "params:visualization_params"],
            outputs="exploratory_scatter_plot_focused",
            name="focused_scatter_analysis",
            tags=["exploration", "focused"],
        ),
        
        node(
            func=create_time_series_analysis, 
            inputs=["time_series_data", "params:visualization_params"],
            outputs="exploratory_time_series_focused",
            name="focused_time_series_analysis",
            tags=["exploration", "focused"],
        ),
    ])


def create_presentation_pipeline(**kwargs) -> Pipeline:
    """
    Create a presentation-focused pipeline for stakeholder reports.
    
    This pipeline demonstrates automated styling optimized for presentation
    purposes, showing how the same visualization logic can produce different
    styled outputs based on catalog configuration alone.
    
    Migration Value:
    - Same node functions, different styling through catalog configuration
    - Eliminates need for separate presentation-specific visualization code
    - Enables easy switching between output purposes without code changes
    
    Returns:
        Pipeline: Presentation-quality pipeline with automated styling
    """
    return pipeline([
        node(
            func=create_comparative_analysis,
            inputs=["primary_dataset", "secondary_dataset", "params:presentation_params"],
            outputs="presentation_comparative_analysis",
            name="presentation_comparative_analysis", 
            tags=["presentation", "stakeholder_ready"],
        ),
        
        node(
            func=create_statistical_summary,
            inputs=["combined_analysis_data", "params:presentation_params"],
            outputs="presentation_summary_report",
            name="presentation_statistical_summary",
            tags=["presentation", "executive_summary"],
        ),
    ])


def create_publication_pipeline(**kwargs) -> Pipeline:
    """
    Create a publication-ready pipeline for academic outputs.
    
    This pipeline demonstrates the highest quality styling automation for
    academic publication requirements, showing how figregistry-kedro can
    automatically apply publication-specific styling rules.
    
    Academic Workflow Benefits:
    - Consistent publication-quality figures across all analyses
    - Automatic compliance with journal formatting requirements
    - Reproducible figure generation for peer review processes
    
    Returns:
        Pipeline: Publication-quality pipeline with automated styling
    """
    return pipeline([
        node(
            func=create_statistical_summary,
            inputs=["publication_ready_data", "params:publication_params"],
            outputs="publication_statistical_analysis",
            name="publication_quality_analysis",
            tags=["publication", "peer_review_ready"],
        ),
    ])


# Pipeline registry for demonstration of different automation approaches
# This shows how the same underlying visualization logic can be configured
# for different output purposes through catalog configuration alone
def create_complete_demonstration_pipeline(**kwargs) -> Pipeline:
    """
    Create a comprehensive demonstration pipeline showing all automation features.
    
    This meta-pipeline combines all demonstration aspects to show the complete
    range of figregistry-kedro automation capabilities within a single workflow.
    
    Comprehensive Demonstration Features:
    - Multiple output purposes (exploratory, presentation, publication)
    - Condition-based styling through parameter resolution
    - Automated versioning integration
    - Zero manual figure management
    - Complete separation of visualization logic and styling concerns
    
    Migration Comparison:
    Before: 50+ lines of manual styling and save code scattered across nodes
    After: 0 lines of manual figure management - fully automated through catalog
    
    Returns:
        Pipeline: Complete demonstration pipeline with all automation features
    """
    # Combine all pipeline variants to demonstrate comprehensive automation
    base_pipeline = create_pipeline(**kwargs)
    exploratory_subset = create_exploratory_pipeline(**kwargs)
    presentation_subset = create_presentation_pipeline(**kwargs) 
    publication_subset = create_publication_pipeline(**kwargs)
    
    # Return combined pipeline demonstrating full automation range
    return (
        base_pipeline +
        exploratory_subset + 
        presentation_subset +
        publication_subset
    )


# Export the primary pipeline for standard Kedro discovery
# This enables 'kedro run --pipeline=data_visualization' execution
__all__ = [
    "create_pipeline",
    "create_exploratory_pipeline", 
    "create_presentation_pipeline",
    "create_publication_pipeline",
    "create_complete_demonstration_pipeline",
]