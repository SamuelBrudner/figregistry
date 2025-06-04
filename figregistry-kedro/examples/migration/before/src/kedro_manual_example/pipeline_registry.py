"""Traditional Kedro Pipeline Registry - Manual Figure Management.

This module provides the create_pipeline() function following standard Kedro 
conventions without figregistry-kedro integration. The registry demonstrates 
traditional pipeline organization patterns where each node function manually 
handles figure creation, styling, and persistence through scattered plt.savefig() 
calls and hardcoded parameters.

This represents the "before" state in the migration example, showcasing the 
maintenance challenges and code duplication that automated figure management 
solutions address.

Pipeline Organization (Traditional Manual Approach):
- "__default__": Basic data visualization pipeline with manual figure management
- "data_visualization": Complete pipeline demonstrating manual approaches
- "exploratory": Quick analysis with scattered styling patterns
- "reporting": Dashboard and summary visualizations with manual file handling

Problems Demonstrated:
- Manual plt.savefig() calls scattered throughout pipeline nodes
- Hardcoded styling parameters within individual node functions  
- Manual file path construction and naming conventions
- Code duplication for styling across different visualization functions
- Inconsistent experimental condition handling without systematic management
- No automated configuration management or lifecycle integration

Each pipeline relies entirely on manual intervention for figure styling,
versioning, and output management, highlighting the overhead eliminated
by figregistry-kedro automation.
"""

from typing import Dict

from kedro.pipeline import Pipeline, node

# Import traditional node functions with manual figure management
from kedro_manual_example.nodes import (
    explore_data_distribution,
    analyze_correlations, 
    create_summary_dashboard,
    create_publication_plots,
    quick_diagnostic_plot,
    batch_plot_generator,
)


def create_pipeline(**kwargs) -> Dict[str, Pipeline]:
    """Create pipeline registry using traditional Kedro patterns.
    
    This function demonstrates conventional Kedro pipeline organization
    without figregistry-kedro integration. Each pipeline consists of nodes
    that manually handle all aspects of figure creation, styling, and 
    persistence through individual plt.savefig() calls and hardcoded
    configuration parameters.
    
    Traditional Pipeline Structure:
        - All node functions contain manual figure management code
        - Styling parameters hardcoded within each function
        - File paths and naming managed manually per function
        - No systematic approach to experimental condition styling
        - No automated configuration initialization or lifecycle management
    
    Pipeline Execution Examples:
        kedro run                              # Runs "__default__" with manual management
        kedro run --pipeline=data_visualization # Complete traditional pipeline
        kedro run --pipeline=exploratory       # Quick manual analysis
        kedro run --pipeline=reporting         # Manual dashboard generation
    
    Contrasts with figregistry-kedro Integration:
        - Manual plt.savefig() vs automated FigureDataSet persistence
        - Hardcoded styling vs condition-based automatic styling
        - Scattered configuration vs centralized configuration management
        - Code duplication vs systematic reusable styling patterns
        - Manual versioning vs automated output management
    
    Returns:
        Dict[str, Pipeline]: Dictionary mapping pipeline names to Pipeline
                           instances configured with traditional manual approaches
                           
    Note:
        These pipelines showcase the maintenance overhead and complexity
        that figregistry-kedro integration eliminates through automated
        figure styling, versioning, and persistence capabilities.
    """
    
    # Traditional data visualization pipeline with manual figure management
    # Each node handles its own styling, file paths, and plt.savefig() calls
    data_visualization_pipeline = Pipeline(
        [
            # Exploratory data analysis with manual styling and hardcoded paths
            node(
                func=explore_data_distribution,
                inputs=["processed_data", "params:visualization"],
                outputs=None,  # No catalog output - manual plt.savefig() in function
                name="explore_data_distribution_manual",
                tags=["exploratory", "manual_management", "scattered_styling"]
            ),
            
            # Correlation analysis with different manual styling approach
            node(
                func=analyze_correlations,
                inputs=["processed_data", "params:visualization"],
                outputs=None,  # Manual file saving within function
                name="analyze_correlations_manual", 
                tags=["analysis", "manual_management", "hardcoded_paths"]
            ),
            
            # Summary dashboard with complex manual subplot management
            node(
                func=create_summary_dashboard,
                inputs=["processed_data", "params:visualization"],
                outputs="dashboard_summary",  # Returns data but saves figures manually
                name="create_summary_dashboard_manual",
                tags=["reporting", "manual_management", "mixed_outputs"]
            ),
            
            # Publication plots with manual multi-format saving
            node(
                func=create_publication_plots,
                inputs=["processed_data", "params:visualization"],
                outputs=None,  # Complex manual saving logic in function
                name="create_publication_plots_manual",
                tags=["publication", "manual_management", "format_complexity"]
            ),
        ]
    )
    
    # Quick exploratory pipeline with minimal manual styling
    # Demonstrates inconsistent approach to rapid prototyping
    exploratory_pipeline = Pipeline(
        [
            # Quick exploratory visualization with basic manual styling
            node(
                func=explore_data_distribution,
                inputs=["raw_data", "params:exploration"],
                outputs=None,  # Manual plt.savefig() with hardcoded parameters
                name="quick_exploration_manual",
                tags=["exploratory", "quick_analysis", "minimal_styling"]
            ),
            
            # Rapid diagnostic plot with different manual approach
            node(
                func=quick_diagnostic_plot,
                inputs=["raw_data", "params:exploration.column"],
                outputs=None,  # Simple manual save with basic parameters
                name="diagnostic_plot_manual",
                tags=["diagnostic", "rapid_prototyping", "basic_manual"]
            ),
        ]
    )
    
    # Reporting pipeline with manual dashboard and batch generation
    # Shows complex manual file management across multiple visualizations
    reporting_pipeline = Pipeline(
        [
            # Summary dashboard with manual layout and styling management
            node(
                func=create_summary_dashboard,
                inputs=["aggregated_data", "params:reporting"], 
                outputs="summary_metadata",
                name="reporting_dashboard_manual",
                tags=["reporting", "dashboard", "manual_layout"]
            ),
            
            # Batch plot generation with manual iteration and file management
            node(
                func=batch_plot_generator,
                inputs=["multiple_datasets", "params:batch_output_path"],
                outputs=None,  # Manual batch file creation and management
                name="batch_plots_manual",
                tags=["batch", "manual_iteration", "file_management"]
            ),
        ]
    )
    
    # Complete traditional pipeline combining all manual approaches
    # Demonstrates the full complexity of manual figure management
    complete_manual_pipeline = Pipeline(
        [
            # Data exploration with manual styling configuration
            node(
                func=explore_data_distribution,
                inputs=["processed_data", "params:visualization"],
                outputs=None,
                name="manual_data_exploration",
                tags=["complete", "exploratory", "manual_styling"]
            ),
            
            # Statistical analysis with different manual styling patterns
            node(
                func=analyze_correlations,
                inputs=["processed_data", "params:visualization"],
                outputs=None,
                name="manual_correlation_analysis", 
                tags=["complete", "analysis", "statistical"]
            ),
            
            # Comprehensive dashboard with manual subplot management
            node(
                func=create_summary_dashboard,
                inputs=["processed_data", "params:visualization"],
                outputs="complete_summary",
                name="manual_comprehensive_dashboard",
                tags=["complete", "reporting", "comprehensive"]
            ),
            
            # Publication-quality plots with manual formatting
            node(
                func=create_publication_plots,
                inputs=["processed_data", "params:visualization"],
                outputs=None,
                name="manual_publication_figures",
                tags=["complete", "publication", "high_quality"]
            ),
        ]
    )
    
    # Traditional pipeline registry following standard Kedro conventions
    # No lifecycle hooks, configuration bridges, or automated management
    return {
        # Default pipeline for traditional manual figure management
        "__default__": data_visualization_pipeline,
        
        # Complete data visualization pipeline with manual approaches
        "data_visualization": complete_manual_pipeline,
        
        # Quick exploratory analysis with minimal manual styling
        "exploratory": exploratory_pipeline,
        
        # Reporting pipeline with manual dashboard and batch generation  
        "reporting": reporting_pipeline,
        
        # Individual component pipelines for focused manual demonstrations
        "correlation_only": Pipeline([
            node(
                func=analyze_correlations,
                inputs=["input_data", "params:analysis"],
                outputs=None,
                name="standalone_correlation_manual",
                tags=["standalone", "correlation", "manual"]
            )
        ]),
        
        "dashboard_only": Pipeline([
            node(
                func=create_summary_dashboard,
                inputs=["dashboard_data", "params:dashboard"],
                outputs="dashboard_output",
                name="standalone_dashboard_manual", 
                tags=["standalone", "dashboard", "manual"]
            )
        ]),
    }


def get_traditional_pipeline_info() -> Dict[str, str]:
    """Provide information about traditional manual figure management pipelines.
    
    Returns descriptive information about each registered pipeline, highlighting
    the manual approaches and maintenance challenges that each demonstrates.
    
    Returns:
        Dict[str, str]: Pipeline names mapped to descriptions of manual approaches
    """
    return {
        "__default__": (
            "Traditional data visualization pipeline with manual plt.savefig() "
            "calls, hardcoded styling parameters, and scattered file management "
            "logic throughout individual node functions."
        ),
        
        "data_visualization": (
            "Complete manual figure management workflow demonstrating scattered "
            "styling code, inconsistent file naming patterns, manual path "
            "construction, and code duplication across visualization functions."
        ),
        
        "exploratory": (
            "Quick exploratory analysis pipeline with minimal manual styling "
            "and basic plt.savefig() calls, showing inconsistent approach to "
            "rapid prototyping and analysis visualization."
        ),
        
        "reporting": (
            "Manual reporting pipeline with complex dashboard layout management, "
            "manual batch file generation, and inconsistent styling approaches "
            "across different reporting components."
        ),
        
        "correlation_only": (
            "Focused correlation analysis with manual styling configuration "
            "and hardcoded file management, demonstrating single-purpose "
            "traditional visualization approach."
        ),
        
        "dashboard_only": (
            "Standalone dashboard generation with manual subplot management, "
            "hardcoded layout parameters, and manual file saving logic."
        ),
    }


def get_manual_management_challenges() -> Dict[str, list]:
    """Document the specific challenges demonstrated by manual figure management.
    
    Returns:
        Dict[str, list]: Categories of challenges with specific examples from
                        the traditional pipeline implementation
    """
    return {
        "code_duplication": [
            "Styling parameters repeated across multiple node functions",
            "Manual file path construction logic duplicated",
            "Color scheme definitions scattered throughout codebase",
            "Error handling patterns inconsistent between functions"
        ],
        
        "maintenance_overhead": [
            "Updating styling requires changes in multiple locations",
            "File naming conventions manually managed per function",
            "Experimental condition styling requires manual coordination",
            "Configuration changes need updates across multiple nodes"
        ],
        
        "inconsistency_issues": [
            "Different DPI settings across visualization functions",
            "Varying file format choices without systematic approach",
            "Inconsistent color schemes for experimental conditions",
            "Mixed error handling approaches across pipeline"
        ],
        
        "scalability_problems": [
            "Adding new experimental conditions requires code changes",
            "Batch operations need manual iteration and file management",
            "Publication formatting requires function-specific implementation",
            "Version management handled manually per visualization"
        ],
        
        "integration_difficulties": [
            "No systematic approach to pipeline configuration",
            "Manual coordination between different visualization stages",
            "Scattered configuration management across node functions",
            "No lifecycle integration for consistent setup and teardown"
        ]
    }


def compare_with_automated_approach() -> Dict[str, Dict[str, str]]:
    """Compare traditional manual approach with figregistry-kedro automation.
    
    Returns:
        Dict[str, Dict[str, str]]: Comparison matrix showing before/after states
    """
    return {
        "figure_saving": {
            "manual": "plt.savefig() calls scattered throughout node functions",
            "automated": "FigureDataSet automatically handles persistence with versioning"
        },
        
        "styling_management": {
            "manual": "Hardcoded styling parameters within each function",
            "automated": "Centralized condition-based styling through configuration"
        },
        
        "file_organization": {
            "manual": "Manual path construction and naming in each function",
            "automated": "Systematic output management with automated versioning"
        },
        
        "configuration": {
            "manual": "Configuration scattered across individual node parameters",
            "automated": "Unified configuration through figregistry.yaml and Kedro integration"
        },
        
        "experimental_conditions": {
            "manual": "Manual condition checking and styling in each function",
            "automated": "Automatic style resolution based on pipeline parameters"
        },
        
        "maintenance": {
            "manual": "Updates require changes across multiple functions",
            "automated": "Single configuration update affects all visualizations"
        },
        
        "code_quality": {
            "manual": "Mixed concerns with visualization logic and file management",
            "automated": "Clean separation between data processing and figure management"
        }
    }


def validate_traditional_pipelines() -> bool:
    """Validate that traditional pipelines are properly configured.
    
    Performs basic validation of the manual figure management pipeline
    registry to ensure proper Kedro compatibility without advanced features.
    
    Returns:
        bool: True if pipelines are valid, False otherwise
    """
    try:
        pipelines = create_pipeline()
        
        # Validate basic structure
        if not isinstance(pipelines, dict):
            return False
        
        # Check required traditional pipelines
        required = ["__default__", "data_visualization", "exploratory"]
        for pipeline_name in required:
            if pipeline_name not in pipelines:
                return False
            
            if not isinstance(pipelines[pipeline_name], Pipeline):
                return False
                
        return True
        
    except Exception:
        return False


# Utility functions for traditional pipeline management

def list_manual_pipelines() -> list:
    """List all available traditional manual figure management pipelines."""
    return list(create_pipeline().keys())


def get_pipeline_tags_traditional() -> Dict[str, list]:
    """Extract tags from traditional pipelines showing manual management patterns."""
    pipelines = create_pipeline()
    pipeline_tags = {}
    
    for pipeline_name, pipeline in pipelines.items():
        tags = set()
        for node in pipeline.nodes:
            if node.tags:
                tags.update(node.tags)
        pipeline_tags[pipeline_name] = sorted(list(tags))
    
    return pipeline_tags


def get_manual_approach_summary() -> Dict[str, str]:
    """Provide summary of the traditional manual approach demonstrated."""
    return {
        "approach": "Traditional Manual Figure Management", 
        "framework": "Standard Kedro without automation extensions",
        "figure_handling": "Manual plt.savefig() calls in node functions",
        "styling": "Hardcoded parameters within individual functions",
        "configuration": "Scattered across node parameters and function code",
        "maintenance": "Manual updates required across multiple locations",
        "pain_points": "Code duplication, inconsistent styling, manual file management",
        "migration_benefit": "Elimination of scattered manual code through automation"
    }