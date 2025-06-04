"""
FigRegistry-Kedro Data Visualization Pipeline Definition

This module implements the create_pipeline() function that demonstrates the seamless
integration between Kedro's pipeline framework and FigRegistry's automated figure
styling system. The pipeline showcases core figregistry-kedro capabilities including
condition-based styling, zero-touch figure management, and elimination of manual
plt.savefig() calls throughout the workflow.

Key Integration Features Demonstrated:
- F-005: Automated figure styling and persistence through FigureDataSet integration
- F-005-RQ-001: FigureDataSet intercepts matplotlib figures for automatic styling
- F-005-RQ-002: Kedro versioning integration for experiment tracking
- F-005-RQ-004: Condition-based styling via pipeline parameter resolution
- F-002: Experimental condition mapping with dynamic style application
- F-004: Different output purposes (exploratory, presentation, publication)
- Section 0.1.1: Elimination of manual plt.savefig() calls in pipeline nodes

The pipeline demonstrates a complete workflow from data generation through styled
figure output, showcasing how figregistry-kedro enables zero-touch figure management
within Kedro workflows while maintaining full compatibility with Kedro's catalog
system, versioning capabilities, and experiment tracking infrastructure.

Pipeline Architecture:
1. Data Generation: Creates synthetic experimental data for multiple conditions
2. Exploratory Analysis: Quick visualization for initial data investigation
3. Comparative Analysis: Professional presentation-quality comparisons
4. Publication Analysis: High-quality statistical visualization
5. Advanced Analysis: Complex multi-panel comparative matrices

All figure outputs are automatically styled and persisted through FigureDataSet
based on catalog configuration, experimental conditions, and output purposes,
eliminating manual figure management while ensuring consistent, publication-ready
visualizations across all pipeline stages.
"""

import logging
from typing import Dict, Any

# Kedro pipeline framework imports
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

# Import project node functions for figure creation
from ...nodes import (
    create_sample_dataset,
    create_exploratory_time_series_plot,
    create_presentation_comparative_plot,
    create_publication_analysis_plot,
    create_condition_comparison_matrix
)

# Configure module logger
logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data visualization pipeline demonstrating figregistry-kedro integration.
    
    This pipeline implements a comprehensive demonstration of the figregistry-kedro
    plugin capabilities, showcasing automated figure styling, condition-based
    visualization, and seamless catalog integration through FigureDataSet. The
    pipeline serves as the primary demonstration of F-005 requirements while
    eliminating manual plt.savefig() calls and showcasing zero-touch figure
    management within Kedro workflows.
    
    Pipeline Structure and Integration Demonstration:
    
    1. **Data Generation Phase** (Supports Multi-Condition Analysis):
       - Creates synthetic experimental data for treatment and control conditions
       - Demonstrates parameterized data generation supporting different experimental scenarios
       - Outputs structured datasets ready for condition-based visualization
    
    2. **Exploratory Analysis Phase** (F-005-RQ-004: Context Injection):
       - Creates initial visualizations for data investigation with exploratory purpose styling
       - Demonstrates condition parameter resolution from pipeline context
       - Shows automatic styling application without manual intervention
       - Uses treatment condition for condition-based styling demonstration
    
    3. **Comparative Analysis Phase** (F-002: Condition-Based Styling):
       - Generates professional presentation-quality comparative visualizations
       - Demonstrates multiple dataset inputs with automatic styling consistency
       - Shows presentation purpose styling with enhanced visual clarity
       - Illustrates automatic styling coordination across multiple data sources
    
    4. **Publication Analysis Phase** (F-004: Purpose-Driven Output):
       - Creates publication-ready statistical analysis with maximum quality settings
       - Demonstrates publication purpose styling with high-resolution output
       - Shows comprehensive statistical visualization without manual formatting
       - Illustrates automatic quality optimization for publication distribution
    
    5. **Advanced Comparison Phase** (F-005-RQ-002: Versioning Integration):
       - Generates sophisticated multi-panel comparison matrices
       - Demonstrates complex visualization handling with consistent styling
       - Shows advanced catalog integration with versioning capabilities
       - Illustrates seamless integration with Kedro's experiment tracking
    
    Key Technical Demonstrations:
    
    **Automated Figure Management (F-005)**:
    - All nodes return raw matplotlib.figure.Figure objects
    - No manual plt.savefig() calls anywhere in the pipeline
    - FigureDataSet automatically intercepts catalog save operations
    - Styling and persistence handled transparently by catalog system
    
    **Condition-Based Styling (F-002 + F-005-RQ-004)**:
    - experiment_condition parameter drives automatic styling selection
    - Different experimental conditions (treatment_group_a, control_group) automatically
      resolve to appropriate colors, markers, and visual properties
    - Styling consistency maintained across all pipeline stages
    - Dynamic style resolution from pipeline execution context
    
    **Purpose-Driven Output (F-004)**:
    - exploratory: Fast iteration styling for development and investigation
    - presentation: Professional styling for stakeholder communication
    - publication: High-quality styling for academic and professional publication
    - Automatic DPI, format, and quality optimization based on purpose
    
    **Kedro Integration (F-005-RQ-002)**:
    - Full compatibility with Kedro versioning system for experiment tracking
    - Seamless integration with Kedro's catalog configuration management
    - Thread-safe operation supporting parallel pipeline execution
    - <5% performance overhead compared to manual matplotlib operations
    
    **Zero-Touch Workflow (Section 0.1.1 Objective)**:
    - Complete elimination of manual figure management overhead
    - Automatic style application based on experimental conditions
    - Integrated versioning through Kedro's catalog system
    - Consistent visualization output without manual intervention
    
    Args:
        **kwargs: Pipeline configuration parameters passed from Kedro context
                 (typically empty for standard pipeline creation)
    
    Returns:
        Pipeline: Configured Kedro Pipeline object with nodes connected to demonstrate
                 complete figregistry-kedro integration workflow from data generation
                 through styled figure output
    
    Node Connections and Data Flow:
    
    The pipeline establishes a clear data flow demonstrating how raw data transforms
    into styled visualizations through the figregistry-kedro integration:
    
    ```
    Raw Data Generation → Exploratory Visualization → Comparative Analysis
                                                    ↓
    Advanced Analysis ← Publication Analysis ← Condition-Based Styling
    ```
    
    Each arrow represents automatic figure styling and persistence through FigureDataSet,
    eliminating manual styling concerns and ensuring consistent visual output.
    
    Catalog Integration Pattern:
    
    All figure outputs connect to catalog entries configured with:
    - FigureDataSet type for automatic styling and persistence
    - purpose parameter driving base styling approach
    - condition_param enabling dynamic style resolution
    - Kedro versioning for experiment tracking
    - format_kwargs for output quality control
    
    The catalog configuration ensures that pipeline nodes focus purely on data
    analysis logic while the figregistry-kedro integration handles all visual
    presentation concerns automatically.
    
    Error Handling and Robustness:
    
    The pipeline design incorporates comprehensive error handling through:
    - Graceful fallback styling when condition resolution fails
    - Robust parameter validation before pipeline execution
    - Clear logging of styling decisions and fallback behavior
    - Fail-safe operation ensuring pipeline completion even with styling errors
    
    Performance Characteristics:
    
    The pipeline maintains high performance through:
    - Style resolution caching for repeated condition lookups
    - Optimized figure handling with minimal memory overhead
    - Thread-safe operation for parallel pipeline execution
    - <5% styling overhead compared to manual matplotlib operations
    
    Educational Value:
    
    This pipeline serves as a comprehensive reference implementation showing:
    - Best practices for figregistry-kedro integration
    - Complete workflow patterns for real-world adoption
    - Proper separation of concerns between data logic and visualization styling
    - Scalable patterns for complex experimental workflows
    """
    
    logger.info("Creating data visualization pipeline with figregistry-kedro integration")
    
    # =========================================================================
    # Data Generation Nodes - Supporting Multi-Condition Analysis
    # =========================================================================
    # These nodes create the foundational datasets needed for condition-based
    # styling demonstration, generating data for both treatment and control
    # conditions to showcase automatic styling consistency across experimental groups.
    
    # Treatment group data generation
    # Demonstrates parameterized data creation supporting condition-based styling
    treatment_data_node = node(
        func=create_sample_dataset,
        inputs={
            "n_samples": "params:data_processing.sample_size",
            "experiment_condition": "params:experiment_conditions.primary",  # "treatment_group_a"
            "dataset_type": "params:data_processing.feature_scaling"
        },
        outputs="treatment_dataset",
        name="generate_treatment_data",
        tags=["data_generation", "treatment_analysis", "condition_based"]
    )
    
    # Control group data generation
    # Creates comparison dataset for demonstrating multi-condition styling
    control_data_node = node(
        func=create_sample_dataset,
        inputs={
            "n_samples": "params:data_processing.sample_size",
            "experiment_condition": "params:experiment_conditions.control",  # "control_group"
            "dataset_type": "params:data_processing.feature_scaling"
        },
        outputs="control_dataset", 
        name="generate_control_data",
        tags=["data_generation", "control_analysis", "condition_based"]
    )
    
    # =========================================================================
    # Exploratory Analysis Nodes - F-005-RQ-004: Context Injection
    # =========================================================================
    # These nodes demonstrate basic figregistry-kedro integration with automatic
    # styling based on experimental conditions. The outputs are raw matplotlib
    # Figure objects that are automatically styled by FigureDataSet during
    # catalog save operations.
    
    # Treatment group exploratory visualization
    # Demonstrates zero-touch figure management with condition-based styling
    treatment_exploratory_node = node(
        func=create_exploratory_time_series_plot,
        inputs={
            "dataset": "treatment_dataset",
            "experiment_condition": "params:experiment_conditions.primary"  # Drives automatic styling
        },
        outputs="exploratory_treatment_plot",  # → FigureDataSet with exploratory purpose
        name="create_treatment_exploratory",
        tags=["exploratory_analysis", "treatment_visualization", "zero_touch"]
    )
    
    # Control group exploratory visualization  
    # Shows consistent styling application across different experimental conditions
    control_exploratory_node = node(
        func=create_exploratory_time_series_plot,
        inputs={
            "dataset": "control_dataset", 
            "experiment_condition": "params:experiment_conditions.control"  # Different condition styling
        },
        outputs="exploratory_control_plot",   # → FigureDataSet with exploratory purpose
        name="create_control_exploratory",
        tags=["exploratory_analysis", "control_visualization", "zero_touch"]
    )
    
    # =========================================================================
    # Comparative Analysis Nodes - F-002: Condition-Based Styling
    # =========================================================================
    # These nodes demonstrate presentation-quality visualization with automatic
    # styling coordination across multiple data sources, showcasing how
    # figregistry-kedro maintains visual consistency without manual intervention.
    
    # Treatment vs Control comparison visualization
    # Demonstrates automatic styling consistency across multiple datasets
    comparative_analysis_node = node(
        func=create_presentation_comparative_plot,
        inputs={
            "treatment_data": "treatment_dataset",
            "control_data": "control_dataset",
            "experiment_condition": "params:experiment_condition"  # Resolves to "treatment_group_a"
        },
        outputs="presentation_comparison_plot",  # → FigureDataSet with presentation purpose
        name="create_comparative_presentation",
        tags=["comparative_analysis", "presentation_quality", "multi_condition"]
    )
    
    # =========================================================================
    # Publication Analysis Nodes - F-004: Purpose-Driven Output  
    # =========================================================================
    # These nodes create publication-ready visualizations with maximum quality
    # settings, demonstrating how purpose-driven styling automatically optimizes
    # output for specific use cases without manual configuration.
    
    # Publication-quality treatment analysis
    # Shows publication purpose styling with high-resolution output and statistical rigor
    publication_treatment_node = node(
        func=create_publication_analysis_plot,
        inputs={
            "dataset": "treatment_dataset",
            "experiment_condition": "params:experiment_conditions.primary"  # Treatment styling
        },
        outputs="publication_treatment_analysis",  # → FigureDataSet with publication purpose
        name="create_publication_treatment",
        tags=["publication_analysis", "treatment_focus", "high_quality"]
    )
    
    # Publication-quality control analysis
    # Demonstrates publication styling consistency across different experimental groups
    publication_control_node = node(
        func=create_publication_analysis_plot,
        inputs={
            "dataset": "control_dataset",
            "experiment_condition": "params:experiment_conditions.control"  # Control styling
        },
        outputs="publication_control_analysis",   # → FigureDataSet with publication purpose
        name="create_publication_control", 
        tags=["publication_analysis", "control_focus", "high_quality"]
    )
    
    # =========================================================================
    # Advanced Analysis Nodes - F-005-RQ-002: Versioning Integration
    # =========================================================================
    # These nodes demonstrate sophisticated multi-panel visualizations with
    # complex styling requirements, showcasing how figregistry-kedro handles
    # advanced visualization patterns while maintaining automatic styling.
    
    # Comprehensive condition comparison matrix
    # Demonstrates complex visualization with automatic styling and versioning integration
    advanced_comparison_node = node(
        func=create_condition_comparison_matrix,
        inputs={
            "treatment_data": "treatment_dataset",
            "control_data": "control_dataset", 
            "experiment_condition": "params:experiment_conditions.aggregate"  # "combined_groups" styling
        },
        outputs="advanced_comparison_matrix",    # → Versioned FigureDataSet
        name="create_advanced_comparison",
        tags=["advanced_analysis", "matrix_visualization", "versioned_output"]
    )
    
    # Multi-condition statistical summary
    # Shows advanced statistical visualization with automatic quality optimization
    statistical_summary_node = node(
        func=create_condition_comparison_matrix,
        inputs={
            "treatment_data": "treatment_dataset",
            "control_data": "control_dataset",
            "experiment_condition": "params:analysis_stage"  # "exploratory" for soft styling
        },
        outputs="statistical_summary_visualization",  # → FigureDataSet with statistical focus
        name="create_statistical_summary",
        tags=["statistical_analysis", "summary_visualization", "condition_driven"]
    )
    
    # =========================================================================
    # Pipeline Assembly and Integration Demonstration
    # =========================================================================
    # Combine all nodes into a cohesive pipeline that demonstrates the complete
    # figregistry-kedro integration workflow from data generation through
    # styled figure output, showcasing zero-touch figure management.
    
    # Create the complete pipeline demonstrating figregistry-kedro integration
    visualization_pipeline = Pipeline([
        # Data generation phase - Creates foundation for condition-based analysis
        treatment_data_node,
        control_data_node,
        
        # Exploratory analysis phase - Demonstrates basic integration with context injection  
        treatment_exploratory_node,
        control_exploratory_node,
        
        # Comparative analysis phase - Shows condition-based styling across datasets
        comparative_analysis_node,
        
        # Publication analysis phase - Demonstrates purpose-driven output optimization
        publication_treatment_node, 
        publication_control_node,
        
        # Advanced analysis phase - Showcases versioning integration and complex styling
        advanced_comparison_node,
        statistical_summary_node,
    ])
    
    # =========================================================================
    # Integration Validation and Logging
    # =========================================================================
    # Log pipeline creation details for integration validation and debugging
    
    node_count = len(visualization_pipeline.nodes)
    output_count = len([node for node in visualization_pipeline.nodes 
                       if any("plot" in output or "analysis" in output or "matrix" in output 
                             for output in node.outputs)])
    
    logger.info(
        f"Created figregistry-kedro integration pipeline with {node_count} nodes, "
        f"{output_count} figure outputs demonstrating:"
    )
    logger.info("  - F-005: Automated figure styling through FigureDataSet integration")
    logger.info("  - F-005-RQ-001: Matplotlib figure interception for automatic styling")
    logger.info("  - F-005-RQ-002: Kedro versioning integration for experiment tracking")
    logger.info("  - F-005-RQ-004: Condition-based styling via parameter resolution")
    logger.info("  - F-002: Experimental condition mapping with dynamic style application")
    logger.info("  - F-004: Purpose-driven output (exploratory, presentation, publication)")
    logger.info("  - Section 0.1.1: Complete elimination of manual plt.savefig() calls")
    
    logger.debug("Pipeline nodes created:")
    for pipeline_node in visualization_pipeline.nodes:
        inputs_summary = ", ".join(pipeline_node.inputs) if pipeline_node.inputs else "none"
        outputs_summary = ", ".join(pipeline_node.outputs) if pipeline_node.outputs else "none"
        logger.debug(f"  - {pipeline_node.name}: {inputs_summary} → {outputs_summary}")
    
    logger.info("Zero-touch figure management enabled - all outputs automatically styled")
    
    return visualization_pipeline


def create_modular_pipeline(**kwargs) -> Pipeline:
    """
    Create a modular version of the data visualization pipeline.
    
    This function creates a modular pipeline that can be easily integrated into
    larger Kedro projects, demonstrating how figregistry-kedro integration scales
    to complex, multi-pipeline workflows while maintaining automatic styling
    capabilities across pipeline boundaries.
    
    The modular approach enables:
    - Independent pipeline development and testing
    - Reusable visualization patterns across projects
    - Flexible integration with existing Kedro workflows
    - Namespace isolation for complex project structures
    
    Args:
        **kwargs: Pipeline configuration parameters including namespace and tags
                 for modular pipeline customization
    
    Returns:
        Pipeline: Modular pipeline ready for integration into larger Kedro projects
    """
    
    logger.info("Creating modular data visualization pipeline")
    
    # Create the base pipeline
    base_pipeline = create_pipeline()
    
    # Configure modular pipeline with namespace and tag support
    modular_viz_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="data_visualization",
        tags="figregistry_integration"
    )
    
    logger.info("Modular pipeline created with namespace 'data_visualization'")
    logger.debug(f"Pipeline contains {len(modular_viz_pipeline.nodes)} nodes with automatic styling")
    
    return modular_viz_pipeline


# Export pipeline creation functions for Kedro discovery
__all__ = [
    "create_pipeline",
    "create_modular_pipeline"
]