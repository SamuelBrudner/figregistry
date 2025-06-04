"""
Advanced Kedro Node Functions for Enterprise FigRegistry-Kedro Integration

This module demonstrates sophisticated figure generation patterns with complex experimental 
conditions, multi-variable analysis, and production-ready visualization workflows. These 
functions showcase enterprise-grade use cases including model training visualizations, 
inference result plotting, statistical analysis reporting, and complex condition-based 
styling scenarios that eliminate manual figure management through automated 
figregistry-kedro integration.

Key Capabilities Demonstrated:
- Advanced matplotlib figure creation without manual styling (F-005 requirements)
- Sophisticated condition-based styling through complex parameter resolution (F-005-RQ-004)  
- Automated FigureDataSet styling for enterprise experimental conditions (F-002 requirements)
- Complex output purposes covering enterprise use cases (F-004 requirements)
- Production-ready figure generation patterns for enterprise deployment scenarios
- Zero-touch figure management workflows eliminating manual plt.savefig() calls
- Multi-treatment studies and A/B testing visualization patterns
- Statistical analysis reporting with automated styling inheritance
- Model performance tracking with condition-aware visualization
- Advanced experimental condition hierarchies and pattern matching

The module serves as a comprehensive reference for implementing enterprise-grade data 
science workflows where figure generation, styling, and persistence are completely 
automated through the figregistry-kedro integration, enabling data scientists to focus 
on analysis logic while ensuring consistent, publication-ready visualizations across 
all pipeline outputs.

Technical Architecture:
- All functions return matplotlib.figure.Figure objects for FigureDataSet consumption
- No manual styling or save operations - delegated to figregistry-kedro automation
- Complex experimental conditions resolved through pipeline parameter inheritance
- Production-ready error handling and performance optimization
- Thread-safe operations supporting parallel pipeline execution
- Comprehensive logging and metrics collection for enterprise monitoring
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import copy

# Core scientific computing imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve, validation_curve

# Matplotlib imports for advanced figure creation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
import seaborn as sns

# Import utilities for advanced condition resolution and data transformation
from .utils import (
    AdvancedConditionResolver,
    MultiEnvironmentConfigManager,
    AdvancedDataTransformer,
    StatisticalAnalysisHelper,
    ExperimentalCondition,
    EnvironmentConfiguration,
    performance_monitor,
    performance_context,
    ConditionResolutionError,
    DataTransformationError,
    StatisticalAnalysisError
)

# Configure module logger
logger = logging.getLogger(__name__)

# Module-level constants for enterprise configuration
ENTERPRISE_FIGURE_DPI = 300
ENTERPRISE_FONT_SIZE = 12
DEFAULT_CONFIDENCE_INTERVAL = 0.95
PERFORMANCE_ANALYSIS_WINDOW = 50
MAX_SUBPLOT_COUNT = 16
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05

# Enterprise-grade styling configurations for different purposes
ENTERPRISE_STYLING_CONFIGS = {
    "training": {
        "purpose": "exploratory",
        "style_emphasis": "performance_tracking",
        "color_palette": "performance_gradient",
        "default_alpha": 0.8
    },
    "inference": {
        "purpose": "presentation", 
        "style_emphasis": "clarity",
        "color_palette": "categorical_distinct",
        "default_alpha": 0.9
    },
    "reporting": {
        "purpose": "publication",
        "style_emphasis": "professional",
        "color_palette": "publication_ready",
        "default_alpha": 1.0
    }
}


@dataclass
class ExperimentalConfiguration:
    """
    Represents complex experimental configuration for advanced condition-based styling.
    
    This class supports sophisticated experimental scenarios including multi-treatment
    studies, A/B testing frameworks, and hierarchical experimental designs that
    leverage figregistry-kedro's automated styling capabilities.
    """
    
    experiment_id: str
    treatment_groups: List[str]
    control_group: str
    experimental_factors: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    statistical_power: float = 0.8
    effect_size_threshold: float = 0.1
    
    def get_condition_hierarchy(self) -> Dict[str, Any]:
        """Generate condition hierarchy for sophisticated pattern matching."""
        return {
            "experiment": {
                "id": self.experiment_id,
                "type": "multi_treatment",
                "power": self.statistical_power
            },
            "treatment": {
                "groups": self.treatment_groups,
                "control": self.control_group
            },
            "factors": self.experimental_factors,
            "baseline": self.baseline_metrics
        }


# Advanced Training Pipeline Nodes

@performance_monitor("training_metrics_visualization", target_ms=50.0)
def create_advanced_training_metrics_dashboard(
    training_history: Dict[str, List[float]],
    validation_history: Dict[str, List[float]], 
    model_metadata: Dict[str, Any],
    experimental_config: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create comprehensive training metrics dashboard with automated styling.
    
    This function demonstrates enterprise-grade training visualization patterns with
    sophisticated condition-based styling applied automatically through FigureDataSet.
    The visualization adapts styling based on model type, optimization algorithm,
    and experimental conditions without any manual styling intervention.
    
    Args:
        training_history: Dictionary of training metric time series
        validation_history: Dictionary of validation metric time series  
        model_metadata: Model configuration and hyperparameter information
        experimental_config: Advanced experimental condition parameters
        
    Returns:
        matplotlib.Figure: Complex training dashboard with automated styling
        
    Note:
        This function demonstrates F-005-RQ-001 by outputting matplotlib figures
        that FigureDataSet automatically intercepts and styles based on:
        - Model architecture type (CNN, RNN, Transformer) → styling inheritance
        - Optimization algorithm → color palette selection  
        - Training regime (fine-tuning, transfer, from-scratch) → visual emphasis
        - Performance tier (high/medium/low) → alpha and marker adjustments
    """
    try:
        with performance_context("training_dashboard_creation"):
            logger.info(f"Creating advanced training metrics dashboard for model: {model_metadata.get('model_name', 'unknown')}")
            
            # Create sophisticated figure layout with GridSpec
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # Extract and validate metrics
            metrics = list(training_history.keys())
            epochs = range(1, len(training_history[metrics[0]]) + 1)
            
            # Primary metrics plot (loss curves)
            ax_loss = fig.add_subplot(gs[0, :2])
            for metric in ['loss', 'val_loss']:
                if metric in training_history:
                    ax_loss.plot(epochs, training_history[metric], label=f"Training {metric.replace('_', ' ').title()}")
                if metric in validation_history:
                    ax_loss.plot(epochs, validation_history[metric], label=f"Validation {metric.replace('_', ' ').title()}")
            
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss Value')
            ax_loss.set_title('Model Training Progress - Loss Evolution')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            # Accuracy/Performance metrics plot
            ax_acc = fig.add_subplot(gs[0, 2])
            accuracy_metrics = [m for m in metrics if 'acc' in m.lower() or 'score' in m.lower()]
            for metric in accuracy_metrics:
                if metric in training_history:
                    ax_acc.plot(epochs, training_history[metric], marker='o', markersize=3, label=f"Train {metric}")
                if metric in validation_history:
                    ax_acc.plot(epochs, validation_history[metric], marker='s', markersize=3, label=f"Val {metric}")
            
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy Score')
            ax_acc.set_title('Model Performance Metrics')
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
            
            # Learning rate schedule visualization
            ax_lr = fig.add_subplot(gs[1, 0])
            if 'learning_rate' in training_history:
                ax_lr.semilogy(epochs, training_history['learning_rate'])
                ax_lr.set_xlabel('Epoch')
                ax_lr.set_ylabel('Learning Rate (log scale)')
                ax_lr.set_title('Learning Rate Schedule')
                ax_lr.grid(True, alpha=0.3)
            else:
                ax_lr.text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                          ha='center', va='center', transform=ax_lr.transAxes)
                ax_lr.set_title('Learning Rate Schedule')
            
            # Gradient norm tracking
            ax_grad = fig.add_subplot(gs[1, 1])
            if 'grad_norm' in training_history:
                ax_grad.plot(epochs, training_history['grad_norm'], alpha=0.7)
                ax_grad.set_xlabel('Epoch')
                ax_grad.set_ylabel('Gradient Norm')
                ax_grad.set_title('Gradient Magnitude Evolution')
                ax_grad.grid(True, alpha=0.3)
            else:
                ax_grad.text(0.5, 0.5, 'Gradient Norm\nTracking Not Available',
                            ha='center', va='center', transform=ax_grad.transAxes)
                ax_grad.set_title('Gradient Magnitude Evolution')
            
            # Model convergence analysis
            ax_conv = fig.add_subplot(gs[1, 2])
            if 'loss' in training_history and len(training_history['loss']) > 5:
                # Calculate loss moving average for convergence analysis
                window_size = min(10, len(training_history['loss']) // 4)
                loss_ma = pd.Series(training_history['loss']).rolling(window=window_size).mean()
                loss_std = pd.Series(training_history['loss']).rolling(window=window_size).std()
                
                ax_conv.fill_between(epochs, loss_ma - loss_std, loss_ma + loss_std, alpha=0.3)
                ax_conv.plot(epochs, loss_ma, linewidth=2, label='Moving Average')
                ax_conv.set_xlabel('Epoch')
                ax_conv.set_ylabel('Loss (Smoothed)')
                ax_conv.set_title('Convergence Analysis')
                ax_conv.legend()
                ax_conv.grid(True, alpha=0.3)
            
            # Training efficiency metrics
            ax_eff = fig.add_subplot(gs[2, 0])
            if 'batch_time' in training_history:
                ax_eff.plot(epochs, training_history['batch_time'], alpha=0.7)
                ax_eff.set_xlabel('Epoch')
                ax_eff.set_ylabel('Batch Time (seconds)')
                ax_eff.set_title('Training Efficiency')
                ax_eff.grid(True, alpha=0.3)
            else:
                ax_eff.text(0.5, 0.5, 'Efficiency Metrics\nNot Available',
                           ha='center', va='center', transform=ax_eff.transAxes)
                ax_eff.set_title('Training Efficiency')
            
            # Memory utilization if available
            ax_mem = fig.add_subplot(gs[2, 1])
            if 'memory_usage' in training_history:
                ax_mem.plot(epochs, training_history['memory_usage'], color='purple', alpha=0.7)
                ax_mem.set_xlabel('Epoch')
                ax_mem.set_ylabel('Memory Usage (MB)')
                ax_mem.set_title('Memory Utilization')
                ax_mem.grid(True, alpha=0.3)
            else:
                ax_mem.text(0.5, 0.5, 'Memory Usage\nNot Tracked',
                           ha='center', va='center', transform=ax_mem.transAxes)
                ax_mem.set_title('Memory Utilization')
            
            # Model metadata summary
            ax_meta = fig.add_subplot(gs[2, 2])
            ax_meta.axis('off')
            
            # Prepare metadata text
            metadata_text = []
            metadata_text.append(f"Model: {model_metadata.get('model_name', 'Unknown')}")
            metadata_text.append(f"Architecture: {model_metadata.get('architecture', 'Unknown')}")
            metadata_text.append(f"Total Parameters: {model_metadata.get('total_params', 'Unknown'):,}")
            metadata_text.append(f"Optimizer: {model_metadata.get('optimizer', 'Unknown')}")
            metadata_text.append(f"Learning Rate: {model_metadata.get('learning_rate', 'Unknown')}")
            metadata_text.append(f"Batch Size: {model_metadata.get('batch_size', 'Unknown')}")
            
            if experimental_config:
                metadata_text.append("--- Experimental Conditions ---")
                for key, value in experimental_config.items():
                    metadata_text.append(f"{key.title()}: {value}")
            
            ax_meta.text(0.05, 0.95, '\n'.join(metadata_text), 
                        transform=ax_meta.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace')
            ax_meta.set_title('Training Configuration')
            
            # Set overall figure title with experimental context
            experiment_info = experimental_config.get('experiment_id', 'Standard Training') if experimental_config else 'Standard Training'
            fig.suptitle(f'Advanced Training Metrics Dashboard - {experiment_info}', 
                        fontsize=16, fontweight='bold')
            
            # Add subtle watermark for enterprise usage
            fig.text(0.99, 0.01, f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='right', va='bottom', fontsize=8, alpha=0.5)
            
            logger.info(f"Successfully created training dashboard with {len(metrics)} metrics tracked")
            
            # Note: No plt.savefig() call - FigureDataSet handles all styling and persistence
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create training metrics dashboard: {e}")
        # Create minimal fallback figure for error handling
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Training Dashboard Creation Failed\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Metrics Dashboard - Error State')
        return fig


@performance_monitor("hyperparameter_optimization_analysis", target_ms=75.0)
def create_hyperparameter_optimization_analysis(
    optimization_results: pd.DataFrame,
    best_config: Dict[str, Any],
    optimization_metadata: Dict[str, Any],
    experimental_conditions: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create comprehensive hyperparameter optimization analysis visualization.
    
    This function demonstrates sophisticated visualization of hyperparameter search
    results with automated condition-based styling. The analysis includes parameter
    importance, optimization convergence, and performance landscape visualization
    with enterprise-grade presentation quality achieved through figregistry-kedro
    automation.
    
    Args:
        optimization_results: DataFrame with hyperparameter combinations and scores
        best_config: Dictionary containing optimal hyperparameter configuration
        optimization_metadata: Metadata about the optimization process
        experimental_conditions: Complex experimental condition parameters
        
    Returns:
        matplotlib.Figure: Advanced hyperparameter analysis with automated styling
        
    Note:
        Demonstrates F-005-RQ-004 through complex condition resolution:
        - Optimization algorithm type → specialized color palettes
        - Search space dimensionality → layout adaptation
        - Performance variance → emphasis adjustments
        - Convergence quality → visual feedback styling
    """
    try:
        with performance_context("hyperparameter_analysis_creation"):
            logger.info(f"Creating hyperparameter optimization analysis for {len(optimization_results)} configurations")
            
            # Create sophisticated figure layout
            fig = plt.figure(figsize=(18, 14))
            gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
            
            # Extract hyperparameter columns and performance metrics
            param_columns = [col for col in optimization_results.columns 
                           if col not in ['score', 'std_score', 'mean_score', 'trial_id', 'duration']]
            score_column = 'score' if 'score' in optimization_results.columns else optimization_results.columns[-1]
            
            # 1. Optimization convergence plot
            ax_convergence = fig.add_subplot(gs[0, :2])
            if 'trial_id' in optimization_results.columns:
                # Plot cumulative best score
                cumulative_best = optimization_results[score_column].cummax()
                ax_convergence.plot(optimization_results['trial_id'], cumulative_best, 
                                  linewidth=2, marker='o', markersize=3, label='Best Score')
                ax_convergence.fill_between(optimization_results['trial_id'], 
                                          optimization_results[score_column].cummin(),
                                          cumulative_best, alpha=0.2, label='Search Range')
            else:
                # Plot by index if trial_id not available
                cumulative_best = optimization_results[score_column].cummax()
                ax_convergence.plot(range(len(optimization_results)), cumulative_best,
                                  linewidth=2, marker='o', markersize=3, label='Best Score')
            
            ax_convergence.set_xlabel('Optimization Trial')
            ax_convergence.set_ylabel('Model Performance Score')
            ax_convergence.set_title('Hyperparameter Optimization Convergence')
            ax_convergence.legend()
            ax_convergence.grid(True, alpha=0.3)
            
            # 2. Parameter importance analysis
            ax_importance = fig.add_subplot(gs[0, 2:])
            if len(param_columns) > 0:
                # Calculate parameter importance through variance analysis
                importance_scores = []
                for param in param_columns:
                    if optimization_results[param].dtype in ['int64', 'float64']:
                        # For numeric parameters, use correlation with score
                        correlation = abs(optimization_results[param].corr(optimization_results[score_column]))
                        importance_scores.append(correlation if not np.isnan(correlation) else 0.0)
                    else:
                        # For categorical parameters, use variance across groups
                        group_means = optimization_results.groupby(param)[score_column].mean()
                        variance = group_means.var()
                        importance_scores.append(variance if not np.isnan(variance) else 0.0)
                
                # Normalize importance scores
                max_importance = max(importance_scores) if importance_scores else 1.0
                normalized_importance = [score / max_importance for score in importance_scores]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(param_columns))
                bars = ax_importance.barh(y_pos, normalized_importance, alpha=0.7)
                ax_importance.set_yticks(y_pos)
                ax_importance.set_yticklabels(param_columns)
                ax_importance.set_xlabel('Relative Importance')
                ax_importance.set_title('Hyperparameter Importance Analysis')
                ax_importance.grid(True, alpha=0.3, axis='x')
                
                # Color bars by importance
                for bar, importance in zip(bars, normalized_importance):
                    bar.set_color(plt.cm.viridis(importance))
            
            # 3. Performance distribution analysis
            ax_dist = fig.add_subplot(gs[1, 0])
            ax_dist.hist(optimization_results[score_column], bins=20, alpha=0.7, density=True)
            ax_dist.axvline(optimization_results[score_column].mean(), color='red', 
                           linestyle='--', label='Mean')
            ax_dist.axvline(optimization_results[score_column].median(), color='orange',
                           linestyle='--', label='Median')
            ax_dist.axvline(optimization_results[score_column].max(), color='green',
                           linestyle='--', label='Best')
            ax_dist.set_xlabel('Performance Score')
            ax_dist.set_ylabel('Density')
            ax_dist.set_title('Score Distribution')
            ax_dist.legend()
            ax_dist.grid(True, alpha=0.3)
            
            # 4. Parameter correlation heatmap
            ax_corr = fig.add_subplot(gs[1, 1:3])
            if len(param_columns) > 1:
                # Select numeric parameters for correlation analysis
                numeric_params = [col for col in param_columns 
                                if optimization_results[col].dtype in ['int64', 'float64']]
                
                if len(numeric_params) >= 2:
                    corr_data = optimization_results[numeric_params + [score_column]].corr()
                    im = ax_corr.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                    ax_corr.set_xticks(range(len(corr_data.columns)))
                    ax_corr.set_yticks(range(len(corr_data.columns)))
                    ax_corr.set_xticklabels(corr_data.columns, rotation=45, ha='right')
                    ax_corr.set_yticklabels(corr_data.columns)
                    ax_corr.set_title('Parameter Correlation Matrix')
                    
                    # Add correlation values to cells
                    for i in range(len(corr_data.columns)):
                        for j in range(len(corr_data.columns)):
                            text = ax_corr.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                              ha='center', va='center', 
                                              color='white' if abs(corr_data.iloc[i, j]) > 0.5 else 'black')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax_corr, shrink=0.8)
                    cbar.set_label('Correlation Coefficient')
                else:
                    ax_corr.text(0.5, 0.5, 'Insufficient Numeric\nParameters for Correlation',
                               ha='center', va='center', transform=ax_corr.transAxes)
            else:
                ax_corr.text(0.5, 0.5, 'Single Parameter\nOptimization',
                           ha='center', va='center', transform=ax_corr.transAxes)
            
            # 5. Best configuration visualization
            ax_best = fig.add_subplot(gs[1, 3])
            ax_best.axis('off')
            
            # Format best configuration text
            best_config_text = ["Best Configuration:"]
            best_config_text.append("-" * 20)
            for param, value in best_config.items():
                if isinstance(value, float):
                    best_config_text.append(f"{param}: {value:.4f}")
                else:
                    best_config_text.append(f"{param}: {value}")
            
            best_score = optimization_results[score_column].max()
            best_config_text.append("-" * 20)
            best_config_text.append(f"Best Score: {best_score:.4f}")
            
            ax_best.text(0.05, 0.95, '\n'.join(best_config_text),
                        transform=ax_best.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace')
            ax_best.set_title('Optimal Configuration')
            
            # 6. Parameter search space visualization (2D projection)
            ax_space = fig.add_subplot(gs[2, :2])
            if len(param_columns) >= 2:
                # Select two most important parameters for 2D visualization
                if len(importance_scores) >= 2:
                    top_param_indices = np.argsort(importance_scores)[-2:]
                    param_x = param_columns[top_param_indices[1]]
                    param_y = param_columns[top_param_indices[0]]
                else:
                    param_x = param_columns[0]
                    param_y = param_columns[1]
                
                # Create scatter plot colored by performance
                scatter = ax_space.scatter(optimization_results[param_x], 
                                         optimization_results[param_y],
                                         c=optimization_results[score_column],
                                         cmap='viridis', alpha=0.7, s=50)
                ax_space.set_xlabel(param_x)
                ax_space.set_ylabel(param_y)
                ax_space.set_title(f'Search Space: {param_x} vs {param_y}')
                
                # Mark best configuration
                best_idx = optimization_results[score_column].idxmax()
                ax_space.scatter(optimization_results.loc[best_idx, param_x],
                               optimization_results.loc[best_idx, param_y],
                               c='red', s=200, marker='*', edgecolors='black',
                               label='Best Config')
                ax_space.legend()
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax_space)
                cbar.set_label('Performance Score')
            else:
                ax_space.text(0.5, 0.5, 'Insufficient Parameters\nfor Space Visualization',
                            ha='center', va='center', transform=ax_space.transAxes)
                ax_space.set_title('Search Space Visualization')
            
            # 7. Optimization efficiency metrics
            ax_efficiency = fig.add_subplot(gs[2, 2:])
            
            # Calculate efficiency metrics
            total_trials = len(optimization_results)
            best_trial_idx = optimization_results[score_column].idxmax()
            efficiency_metrics = []
            
            efficiency_metrics.append(f"Total Trials: {total_trials}")
            efficiency_metrics.append(f"Best Found at Trial: {best_trial_idx + 1}")
            efficiency_metrics.append(f"Search Efficiency: {((best_trial_idx + 1) / total_trials * 100):.1f}%")
            
            if 'duration' in optimization_results.columns:
                total_time = optimization_results['duration'].sum()
                avg_time = optimization_results['duration'].mean()
                efficiency_metrics.append(f"Total Time: {total_time:.2f}s")
                efficiency_metrics.append(f"Avg Trial Time: {avg_time:.2f}s")
            
            # Performance improvement metrics
            score_range = optimization_results[score_column].max() - optimization_results[score_column].min()
            score_std = optimization_results[score_column].std()
            efficiency_metrics.append(f"Score Range: {score_range:.4f}")
            efficiency_metrics.append(f"Score Std Dev: {score_std:.4f}")
            
            # Add experimental conditions if provided
            if experimental_conditions:
                efficiency_metrics.append("-" * 25)
                efficiency_metrics.append("Experimental Conditions:")
                for key, value in experimental_conditions.items():
                    efficiency_metrics.append(f"{key}: {value}")
            
            ax_efficiency.text(0.05, 0.95, '\n'.join(efficiency_metrics),
                             transform=ax_efficiency.transAxes, fontsize=10,
                             verticalalignment='top', fontfamily='monospace')
            ax_efficiency.axis('off')
            ax_efficiency.set_title('Optimization Efficiency Analysis')
            
            # Set overall figure title
            optimization_type = optimization_metadata.get('algorithm', 'Hyperparameter Optimization')
            fig.suptitle(f'Advanced {optimization_type} Analysis', fontsize=16, fontweight='bold')
            
            # Add enterprise metadata
            fig.text(0.99, 0.01, f'Analysis Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='right', va='bottom', fontsize=8, alpha=0.5)
            
            logger.info(f"Successfully created hyperparameter optimization analysis with {total_trials} trials")
            
            # Note: FigureDataSet will apply condition-based styling and handle persistence
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create hyperparameter optimization analysis: {e}")
        # Create minimal fallback figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Hyperparameter Analysis Failed\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Hyperparameter Optimization Analysis - Error State')
        return fig


# Advanced Inference Pipeline Nodes

@performance_monitor("model_inference_analysis", target_ms=60.0)
def create_model_inference_analysis(
    inference_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    model_config: Dict[str, Any],
    deployment_context: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create comprehensive model inference analysis with performance metrics.
    
    This function demonstrates enterprise-grade inference result visualization with
    sophisticated condition-based styling based on model performance characteristics,
    deployment environment, and prediction confidence levels. All styling is handled
    automatically through figregistry-kedro integration.
    
    Args:
        inference_results: DataFrame containing model predictions and confidence scores
        ground_truth: DataFrame containing true labels for evaluation
        model_config: Model configuration and metadata
        deployment_context: Production deployment context and conditions
        
    Returns:
        matplotlib.Figure: Advanced inference analysis with automated styling
        
    Note:
        Demonstrates F-002 condition-based styling through:
        - Model performance tier → color intensity and marker styles
        - Confidence distribution → alpha transparency adjustments  
        - Prediction accuracy → emphasis and highlighting patterns
        - Deployment environment → professional vs exploratory styling
    """
    try:
        with performance_context("inference_analysis_creation"):
            logger.info(f"Creating model inference analysis for {len(inference_results)} predictions")
            
            # Create sophisticated figure layout
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # Determine if this is classification or regression based on data
            is_classification = 'class_probabilities' in inference_results.columns or \
                              'predicted_class' in inference_results.columns
            
            # 1. Prediction confidence distribution
            ax_conf = fig.add_subplot(gs[0, 0])
            if 'confidence' in inference_results.columns:
                confidence_scores = inference_results['confidence']
                ax_conf.hist(confidence_scores, bins=30, alpha=0.7, density=True)
                ax_conf.axvline(confidence_scores.mean(), color='red', linestyle='--', 
                               label=f'Mean: {confidence_scores.mean():.3f}')
                ax_conf.axvline(confidence_scores.median(), color='orange', linestyle='--',
                               label=f'Median: {confidence_scores.median():.3f}')
                ax_conf.set_xlabel('Prediction Confidence')
                ax_conf.set_ylabel('Density')
                ax_conf.set_title('Prediction Confidence Distribution')
                ax_conf.legend()
                ax_conf.grid(True, alpha=0.3)
            else:
                ax_conf.text(0.5, 0.5, 'Confidence Scores\nNot Available',
                           ha='center', va='center', transform=ax_conf.transAxes)
                ax_conf.set_title('Prediction Confidence Distribution')
            
            # 2. Performance metrics overview
            ax_metrics = fig.add_subplot(gs[0, 1])
            
            if is_classification:
                # Classification metrics
                if 'predicted_class' in inference_results.columns and 'true_class' in ground_truth.columns:
                    y_true = ground_truth['true_class']
                    y_pred = inference_results['predicted_class']
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    
                    # Create metrics bar chart
                    metrics_names = ['Accuracy']
                    metrics_values = [accuracy]
                    
                    bars = ax_metrics.bar(metrics_names, metrics_values, alpha=0.7)
                    ax_metrics.set_ylim(0, 1)
                    ax_metrics.set_ylabel('Score')
                    ax_metrics.set_title('Classification Performance')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metrics_values):
                        height = bar.get_height()
                        ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom')
                        
                        # Color bars based on performance
                        if value >= 0.9:
                            bar.set_color('green')
                        elif value >= 0.7:
                            bar.set_color('orange')
                        else:
                            bar.set_color('red')
                else:
                    ax_metrics.text(0.5, 0.5, 'Classification Labels\nNot Available',
                                  ha='center', va='center', transform=ax_metrics.transAxes)
            else:
                # Regression metrics
                if 'predicted_value' in inference_results.columns and 'true_value' in ground_truth.columns:
                    y_true = ground_truth['true_value']
                    y_pred = inference_results['predicted_value']
                    
                    mae = np.mean(np.abs(y_true - y_pred))
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                    
                    metrics_names = ['MAE', 'RMSE', 'R²']
                    metrics_values = [mae, rmse, max(0, r2)]  # Ensure R² is non-negative for visualization
                    
                    bars = ax_metrics.bar(metrics_names, metrics_values, alpha=0.7)
                    ax_metrics.set_ylabel('Error/Score')
                    ax_metrics.set_title('Regression Performance')
                    
                    # Add value labels
                    for bar, value in zip(bars, metrics_values):
                        height = bar.get_height()
                        ax_metrics.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                       f'{value:.3f}', ha='center', va='bottom')
                else:
                    ax_metrics.text(0.5, 0.5, 'Regression Values\nNot Available',
                                  ha='center', va='center', transform=ax_metrics.transAxes)
            
            ax_metrics.grid(True, alpha=0.3)
            
            # 3. Prediction vs actual scatter plot
            ax_scatter = fig.add_subplot(gs[0, 2])
            
            if is_classification:
                # For classification, show confusion matrix as heatmap
                if 'predicted_class' in inference_results.columns and 'true_class' in ground_truth.columns:
                    y_true = ground_truth['true_class']
                    y_pred = inference_results['predicted_class']
                    
                    cm = confusion_matrix(y_true, y_pred)
                    im = ax_scatter.imshow(cm, interpolation='nearest', cmap='Blues')
                    
                    # Add labels
                    classes = sorted(list(set(y_true) | set(y_pred)))
                    tick_marks = np.arange(len(classes))
                    ax_scatter.set_xticks(tick_marks)
                    ax_scatter.set_yticks(tick_marks)
                    ax_scatter.set_xticklabels(classes)
                    ax_scatter.set_yticklabels(classes)
                    ax_scatter.set_xlabel('Predicted')
                    ax_scatter.set_ylabel('Actual')
                    ax_scatter.set_title('Confusion Matrix')
                    
                    # Add text annotations
                    thresh = cm.max() / 2.
                    for i, j in np.ndindex(cm.shape):
                        ax_scatter.text(j, i, format(cm[i, j], 'd'),
                                       ha="center", va="center",
                                       color="white" if cm[i, j] > thresh else "black")
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax_scatter, shrink=0.8)
                else:
                    ax_scatter.text(0.5, 0.5, 'Classification Data\nNot Available',
                                  ha='center', va='center', transform=ax_scatter.transAxes)
            else:
                # For regression, show actual vs predicted scatter
                if 'predicted_value' in inference_results.columns and 'true_value' in ground_truth.columns:
                    y_true = ground_truth['true_value']
                    y_pred = inference_results['predicted_value']
                    
                    # Color points by confidence if available
                    if 'confidence' in inference_results.columns:
                        scatter = ax_scatter.scatter(y_true, y_pred, 
                                                   c=inference_results['confidence'],
                                                   cmap='viridis', alpha=0.6)
                        plt.colorbar(scatter, ax=ax_scatter, label='Confidence')
                    else:
                        ax_scatter.scatter(y_true, y_pred, alpha=0.6)
                    
                    # Add perfect prediction line
                    min_val = min(min(y_true), min(y_pred))
                    max_val = max(max(y_true), max(y_pred))
                    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    ax_scatter.set_xlabel('Actual Values')
                    ax_scatter.set_ylabel('Predicted Values')
                    ax_scatter.set_title('Predicted vs Actual')
                    ax_scatter.grid(True, alpha=0.3)
                else:
                    ax_scatter.text(0.5, 0.5, 'Regression Data\nNot Available',
                                  ha='center', va='center', transform=ax_scatter.transAxes)
            
            # 4. Prediction error analysis
            ax_error = fig.add_subplot(gs[1, :])
            
            if not is_classification and 'predicted_value' in inference_results.columns and 'true_value' in ground_truth.columns:
                y_true = ground_truth['true_value']
                y_pred = inference_results['predicted_value']
                errors = y_pred - y_true
                
                # Create error distribution and residual plot
                ax_error_hist = ax_error
                ax_error_hist.hist(errors, bins=30, alpha=0.7, density=True)
                ax_error_hist.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
                ax_error_hist.axvline(errors.mean(), color='orange', linestyle='--',
                                     label=f'Mean Error: {errors.mean():.3f}')
                ax_error_hist.set_xlabel('Prediction Error (Predicted - Actual)')
                ax_error_hist.set_ylabel('Density')
                ax_error_hist.set_title('Prediction Error Distribution')
                ax_error_hist.legend()
                ax_error_hist.grid(True, alpha=0.3)
            elif is_classification:
                # For classification, show probability calibration if available
                if 'class_probabilities' in inference_results.columns and 'true_class' in ground_truth.columns:
                    # Extract probability for positive class (assuming binary classification)
                    probs = inference_results['class_probabilities'].apply(lambda x: max(eval(x)) if isinstance(x, str) else max(x))
                    ax_error.hist(probs, bins=20, alpha=0.7, density=True)
                    ax_error.set_xlabel('Maximum Class Probability')
                    ax_error.set_ylabel('Density')
                    ax_error.set_title('Class Probability Distribution')
                    ax_error.grid(True, alpha=0.3)
                else:
                    ax_error.text(0.5, 0.5, 'Classification Probabilities\nNot Available',
                                ha='center', va='center', transform=ax_error.transAxes)
                    ax_error.set_title('Class Probability Distribution')
            else:
                ax_error.text(0.5, 0.5, 'Error Analysis\nData Not Available',
                            ha='center', va='center', transform=ax_error.transAxes)
                ax_error.set_title('Prediction Error Analysis')
            
            # 5. Model performance summary
            ax_summary = fig.add_subplot(gs[2, 0])
            ax_summary.axis('off')
            
            summary_text = ["Model Performance Summary:"]
            summary_text.append("-" * 25)
            summary_text.append(f"Model: {model_config.get('model_name', 'Unknown')}")
            summary_text.append(f"Version: {model_config.get('version', 'Unknown')}")
            summary_text.append(f"Total Predictions: {len(inference_results)}")
            
            if 'confidence' in inference_results.columns:
                avg_conf = inference_results['confidence'].mean()
                summary_text.append(f"Avg Confidence: {avg_conf:.3f}")
            
            if deployment_context:
                summary_text.append("")
                summary_text.append("Deployment Context:")
                for key, value in deployment_context.items():
                    summary_text.append(f"{key}: {value}")
            
            ax_summary.text(0.05, 0.95, '\n'.join(summary_text),
                          transform=ax_summary.transAxes, fontsize=10,
                          verticalalignment='top', fontfamily='monospace')
            ax_summary.set_title('Performance Summary')
            
            # 6. Inference latency analysis (if available)
            ax_latency = fig.add_subplot(gs[2, 1])
            if 'inference_time' in inference_results.columns:
                latencies = inference_results['inference_time']
                ax_latency.hist(latencies, bins=20, alpha=0.7)
                ax_latency.axvline(latencies.mean(), color='red', linestyle='--',
                                 label=f'Mean: {latencies.mean():.3f}ms')
                ax_latency.axvline(latencies.quantile(0.95), color='orange', linestyle='--',
                                 label=f'95th %ile: {latencies.quantile(0.95):.3f}ms')
                ax_latency.set_xlabel('Inference Time (ms)')
                ax_latency.set_ylabel('Frequency')
                ax_latency.set_title('Inference Latency Distribution')
                ax_latency.legend()
                ax_latency.grid(True, alpha=0.3)
            else:
                ax_latency.text(0.5, 0.5, 'Latency Data\nNot Available',
                              ha='center', va='center', transform=ax_latency.transAxes)
                ax_latency.set_title('Inference Latency Distribution')
            
            # 7. Feature importance or prediction breakdown (if available)
            ax_features = fig.add_subplot(gs[2, 2])
            if 'feature_importance' in inference_results.columns:
                # Show feature importance for a sample prediction
                sample_importance = eval(inference_results['feature_importance'].iloc[0])
                if isinstance(sample_importance, dict):
                    features = list(sample_importance.keys())[:10]  # Top 10 features
                    importances = [sample_importance[f] for f in features]
                    
                    y_pos = np.arange(len(features))
                    bars = ax_features.barh(y_pos, importances, alpha=0.7)
                    ax_features.set_yticks(y_pos)
                    ax_features.set_yticklabels(features)
                    ax_features.set_xlabel('Feature Importance')
                    ax_features.set_title('Sample Feature Importance')
                    
                    # Color bars by importance
                    max_imp = max(importances)
                    for bar, imp in zip(bars, importances):
                        bar.set_color(plt.cm.viridis(imp / max_imp))
                else:
                    ax_features.text(0.5, 0.5, 'Feature Importance\nFormat Not Supported',
                                   ha='center', va='center', transform=ax_features.transAxes)
            else:
                ax_features.text(0.5, 0.5, 'Feature Importance\nNot Available',
                               ha='center', va='center', transform=ax_features.transAxes)
            ax_features.set_title('Feature Importance Analysis')
            
            # Set overall figure title
            model_name = model_config.get('model_name', 'Model')
            analysis_type = 'Classification' if is_classification else 'Regression'
            fig.suptitle(f'{model_name} Inference Analysis - {analysis_type}', 
                        fontsize=16, fontweight='bold')
            
            # Add enterprise metadata
            fig.text(0.99, 0.01, f'Analysis Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='right', va='bottom', fontsize=8, alpha=0.5)
            
            logger.info(f"Successfully created inference analysis for {model_name}")
            
            # Note: FigureDataSet handles all styling and persistence automatically
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create model inference analysis: {e}")
        # Create minimal fallback figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Inference Analysis Failed\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Inference Analysis - Error State')
        return fig


@performance_monitor("ab_testing_visualization", target_ms=45.0)
def create_ab_testing_analysis(
    experiment_data: pd.DataFrame,
    experiment_config: ExperimentalConfiguration,
    statistical_tests: Dict[str, Any],
    business_metrics: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create comprehensive A/B testing analysis visualization.
    
    This function demonstrates sophisticated A/B testing visualization with automated
    condition-based styling that adapts based on statistical significance, effect
    sizes, and business impact. The visualization provides executive-level insights
    with publication-ready formatting through figregistry-kedro automation.
    
    Args:
        experiment_data: DataFrame containing experimental results and metrics
        experiment_config: ExperimentalConfiguration with complex conditions
        statistical_tests: Dictionary containing statistical test results
        business_metrics: Optional business impact metrics and KPIs
        
    Returns:
        matplotlib.Figure: Advanced A/B testing analysis with automated styling
        
    Note:
        Demonstrates sophisticated F-005-RQ-004 condition resolution:
        - Statistical significance level → color coding and emphasis
        - Effect size magnitude → marker sizes and line weights  
        - Experimental power → confidence interval styling
        - Business impact tier → professional vs exploratory presentation
    """
    try:
        with performance_context("ab_testing_analysis_creation"):
            logger.info(f"Creating A/B testing analysis for experiment: {experiment_config.experiment_id}")
            
            # Create sophisticated figure layout
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
            
            # Extract treatment groups and control
            treatment_groups = experiment_config.treatment_groups
            control_group = experiment_config.control_group
            all_groups = [control_group] + treatment_groups
            
            # Define metrics to analyze
            metric_columns = [col for col in experiment_data.columns 
                            if col not in ['group', 'user_id', 'timestamp', 'session_id']]
            primary_metric = metric_columns[0] if metric_columns else 'conversion_rate'
            
            # 1. Treatment group comparison - Primary metric
            ax_primary = fig.add_subplot(gs[0, :2])
            
            group_data = []
            group_labels = []
            group_means = []
            
            for group in all_groups:
                if group in experiment_data['group'].values:
                    group_values = experiment_data[experiment_data['group'] == group][primary_metric]
                    group_data.append(group_values)
                    group_labels.append(group)
                    group_means.append(group_values.mean())
            
            # Create box plot with individual points
            bp = ax_primary.boxplot(group_data, labels=group_labels, patch_artist=True)
            
            # Color boxes based on performance relative to control
            control_mean = group_means[0] if group_means else 0
            for i, (patch, mean) in enumerate(zip(bp['boxes'], group_means)):
                if i == 0:  # Control group
                    patch.set_facecolor('lightblue')
                else:
                    # Color based on improvement
                    improvement = (mean - control_mean) / control_mean if control_mean != 0 else 0
                    if improvement > 0.05:  # Significant improvement
                        patch.set_facecolor('lightgreen')
                    elif improvement < -0.05:  # Significant degradation
                        patch.set_facecolor('lightcoral')
                    else:
                        patch.set_facecolor('lightyellow')
            
            # Add statistical significance markers
            if 'pairwise_tests' in statistical_tests:
                for i, test_result in enumerate(statistical_tests['pairwise_tests']):
                    if test_result.get('p_value', 1.0) < 0.05:
                        # Add significance marker
                        y_max = max([max(data) for data in group_data])
                        ax_primary.text(i + 1, y_max * 1.1, '*', ha='center', va='center',
                                      fontsize=16, fontweight='bold', color='red')
            
            ax_primary.set_xlabel('Treatment Groups')
            ax_primary.set_ylabel(primary_metric.replace('_', ' ').title())
            ax_primary.set_title(f'Primary Metric Comparison: {primary_metric.replace("_", " ").title()}')
            ax_primary.grid(True, alpha=0.3)
            
            # 2. Statistical significance summary
            ax_stats = fig.add_subplot(gs[0, 2])
            ax_stats.axis('off')
            
            stats_text = ["Statistical Test Results:"]
            stats_text.append("-" * 20)
            
            if 'anova' in statistical_tests:
                anova_result = statistical_tests['anova']
                stats_text.append(f"ANOVA F-stat: {anova_result.get('f_statistic', 'N/A'):.3f}")
                stats_text.append(f"ANOVA p-value: {anova_result.get('p_value', 'N/A'):.4f}")
                stats_text.append("")
            
            if 'pairwise_tests' in statistical_tests:
                stats_text.append("Pairwise Comparisons:")
                for i, test in enumerate(statistical_tests['pairwise_tests']):
                    group_name = treatment_groups[i] if i < len(treatment_groups) else f"Group {i+1}"
                    p_val = test.get('p_value', 1.0)
                    effect_size = test.get('effect_size', 0.0)
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    stats_text.append(f"{group_name}: p={p_val:.4f} {significance}")
                    stats_text.append(f"  Effect size: {effect_size:.3f}")
            
            ax_stats.text(0.05, 0.95, '\n'.join(stats_text),
                         transform=ax_stats.transAxes, fontsize=9,
                         verticalalignment='top', fontfamily='monospace')
            ax_stats.set_title('Statistical Summary')
            
            # 3. Effect size visualization
            ax_effect = fig.add_subplot(gs[1, 0])
            
            if 'pairwise_tests' in statistical_tests:
                effect_sizes = [test.get('effect_size', 0.0) for test in statistical_tests['pairwise_tests']]
                treatment_names = treatment_groups[:len(effect_sizes)]
                
                bars = ax_effect.bar(range(len(effect_sizes)), effect_sizes, alpha=0.7)
                ax_effect.set_xticks(range(len(effect_sizes)))
                ax_effect.set_xticklabels(treatment_names, rotation=45, ha='right')
                ax_effect.set_ylabel('Effect Size (Cohen\'s d)')
                ax_effect.set_title('Treatment Effect Sizes')
                ax_effect.grid(True, alpha=0.3)
                
                # Color bars by effect size magnitude
                for bar, effect in zip(bars, effect_sizes):
                    if abs(effect) >= 0.8:  # Large effect
                        bar.set_color('darkgreen')
                    elif abs(effect) >= 0.5:  # Medium effect
                        bar.set_color('orange')
                    elif abs(effect) >= 0.2:  # Small effect
                        bar.set_color('yellow')
                    else:  # Negligible effect
                        bar.set_color('lightgray')
                
                # Add effect size interpretation lines
                ax_effect.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
                ax_effect.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
                ax_effect.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
                ax_effect.legend()
            else:
                ax_effect.text(0.5, 0.5, 'Effect Size\nData Not Available',
                             ha='center', va='center', transform=ax_effect.transAxes)
            
            # 4. Confidence intervals visualization
            ax_ci = fig.add_subplot(gs[1, 1])
            
            confidence_level = 1 - STATISTICAL_SIGNIFICANCE_THRESHOLD
            means = []
            ci_lower = []
            ci_upper = []
            group_names = []
            
            for group in all_groups:
                if group in experiment_data['group'].values:
                    group_values = experiment_data[experiment_data['group'] == group][primary_metric]
                    mean_val = group_values.mean()
                    se = stats.sem(group_values)
                    ci = stats.t.interval(confidence_level, len(group_values)-1, loc=mean_val, scale=se)
                    
                    means.append(mean_val)
                    ci_lower.append(ci[0])
                    ci_upper.append(ci[1])
                    group_names.append(group)
            
            x_pos = range(len(means))
            ax_ci.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower),
                                              np.array(ci_upper) - np.array(means)],
                          fmt='o', capsize=5, capthick=2, alpha=0.8)
            
            ax_ci.set_xticks(x_pos)
            ax_ci.set_xticklabels(group_names, rotation=45, ha='right')
            ax_ci.set_ylabel(primary_metric.replace('_', ' ').title())
            ax_ci.set_title(f'{int(confidence_level*100)}% Confidence Intervals')
            ax_ci.grid(True, alpha=0.3)
            
            # 5. Time series analysis (if timestamp available)
            ax_time = fig.add_subplot(gs[1, 2])
            
            if 'timestamp' in experiment_data.columns:
                experiment_data['timestamp'] = pd.to_datetime(experiment_data['timestamp'])
                
                # Group by date and treatment
                daily_data = experiment_data.groupby([experiment_data['timestamp'].dt.date, 'group'])[primary_metric].mean().unstack('group')
                
                for group in all_groups:
                    if group in daily_data.columns:
                        ax_time.plot(daily_data.index, daily_data[group], marker='o', label=group, alpha=0.8)
                
                ax_time.set_xlabel('Date')
                ax_time.set_ylabel(f'Daily Avg {primary_metric.replace("_", " ").title()}')
                ax_time.set_title('Metric Evolution Over Time')
                ax_time.legend()
                ax_time.grid(True, alpha=0.3)
                
                # Rotate date labels
                for tick in ax_time.get_xticklabels():
                    tick.set_rotation(45)
            else:
                ax_time.text(0.5, 0.5, 'Timestamp Data\nNot Available',
                           ha='center', va='center', transform=ax_time.transAxes)
                ax_time.set_title('Metric Evolution Over Time')
            
            # 6. Business impact analysis
            ax_business = fig.add_subplot(gs[2, :2])
            
            if business_metrics:
                # Create business metrics comparison
                metric_names = list(business_metrics.keys())
                metric_values = list(business_metrics.values())
                
                if len(metric_names) > 0:
                    # Assume metrics are improvement percentages
                    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in metric_values]
                    bars = ax_business.bar(metric_names, metric_values, color=colors, alpha=0.7)
                    
                    ax_business.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax_business.set_ylabel('Improvement (%)')
                    ax_business.set_title('Business Impact Analysis')
                    ax_business.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metric_values):
                        height = bar.get_height()
                        ax_business.text(bar.get_x() + bar.get_width()/2., 
                                       height + (0.01 if height >= 0 else -0.01),
                                       f'{value:.1f}%', ha='center', 
                                       va='bottom' if height >= 0 else 'top')
                    
                    # Rotate labels if many metrics
                    if len(metric_names) > 5:
                        for tick in ax_business.get_xticklabels():
                            tick.set_rotation(45)
                else:
                    ax_business.text(0.5, 0.5, 'Business Metrics\nNot Provided',
                                   ha='center', va='center', transform=ax_business.transAxes)
            else:
                ax_business.text(0.5, 0.5, 'Business Impact\nData Not Available',
                               ha='center', va='center', transform=ax_business.transAxes)
                ax_business.set_title('Business Impact Analysis')
            
            # 7. Experiment configuration summary
            ax_config = fig.add_subplot(gs[2, 2])
            ax_config.axis('off')
            
            config_text = ["Experiment Configuration:"]
            config_text.append("-" * 20)
            config_text.append(f"ID: {experiment_config.experiment_id}")
            config_text.append(f"Control: {experiment_config.control_group}")
            config_text.append(f"Treatments: {', '.join(experiment_config.treatment_groups)}")
            config_text.append(f"Statistical Power: {experiment_config.statistical_power:.2f}")
            config_text.append(f"Min Effect Size: {experiment_config.effect_size_threshold:.3f}")
            config_text.append("")
            
            # Add sample sizes
            config_text.append("Sample Sizes:")
            for group in all_groups:
                if group in experiment_data['group'].values:
                    sample_size = len(experiment_data[experiment_data['group'] == group])
                    config_text.append(f"{group}: {sample_size:,}")
            
            # Add experimental factors if available
            if experiment_config.experimental_factors:
                config_text.append("")
                config_text.append("Factors:")
                for factor, value in experiment_config.experimental_factors.items():
                    config_text.append(f"{factor}: {value}")
            
            ax_config.text(0.05, 0.95, '\n'.join(config_text),
                          transform=ax_config.transAxes, fontsize=9,
                          verticalalignment='top', fontfamily='monospace')
            ax_config.set_title('Experiment Details')
            
            # Set overall figure title
            fig.suptitle(f'A/B Testing Analysis: {experiment_config.experiment_id}', 
                        fontsize=16, fontweight='bold')
            
            # Add enterprise metadata
            fig.text(0.99, 0.01, f'Analysis Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='right', va='bottom', fontsize=8, alpha=0.5)
            
            logger.info(f"Successfully created A/B testing analysis for {experiment_config.experiment_id}")
            
            # Note: FigureDataSet applies sophisticated condition-based styling automatically
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create A/B testing analysis: {e}")
        # Create minimal fallback figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'A/B Testing Analysis Failed\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('A/B Testing Analysis - Error State')
        return fig


# Advanced Reporting Pipeline Nodes

@performance_monitor("executive_dashboard_creation", target_ms=100.0)
def create_executive_performance_dashboard(
    kpi_data: pd.DataFrame,
    trend_analysis: Dict[str, Any],
    comparative_metrics: Dict[str, Any],
    executive_context: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create enterprise executive performance dashboard with sophisticated styling.
    
    This function demonstrates the pinnacle of automated visualization with publication-
    ready executive dashboard generation. The styling automatically adapts to business
    performance levels, trend directions, and stakeholder audience through advanced
    condition-based styling rules managed by figregistry-kedro.
    
    Args:
        kpi_data: DataFrame containing key performance indicators over time
        trend_analysis: Dictionary with trend analysis results and forecasts
        comparative_metrics: Dictionary with period-over-period comparisons
        executive_context: High-level business context and presentation requirements
        
    Returns:
        matplotlib.Figure: Publication-ready executive dashboard with automated styling
        
    Note:
        Demonstrates ultimate F-002 and F-004 integration with:
        - Performance tier (excellent/good/needs_attention) → professional color schemes
        - Trend direction (growth/decline/stable) → visual emphasis patterns
        - Stakeholder audience (board/executive/operational) → presentation formality
        - Business impact (high/medium/low) → chart prominence and styling
    """
    try:
        with performance_context("executive_dashboard_creation"):
            logger.info("Creating enterprise executive performance dashboard")
            
            # Create publication-quality figure layout
            fig = plt.figure(figsize=(20, 14))
            gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
            
            # Configure enterprise styling
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'sans-serif',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False
            })
            
            # Extract KPI columns and time information
            kpi_columns = [col for col in kpi_data.columns if col != 'date']
            primary_kpi = kpi_columns[0] if kpi_columns else 'revenue'
            
            # 1. Executive Summary KPI Cards (Top Row)
            for i, kpi in enumerate(kpi_columns[:4]):
                ax_kpi = fig.add_subplot(gs[0, i])
                
                current_value = kpi_data[kpi].iloc[-1]
                previous_value = kpi_data[kpi].iloc[-2] if len(kpi_data) > 1 else current_value
                change_pct = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
                
                # Format KPI value based on magnitude
                if abs(current_value) >= 1000000:
                    formatted_value = f"${current_value/1000000:.1f}M"
                elif abs(current_value) >= 1000:
                    formatted_value = f"${current_value/1000:.1f}K"
                else:
                    formatted_value = f"${current_value:.0f}"
                
                # Create KPI card visualization
                ax_kpi.text(0.5, 0.7, formatted_value, ha='center', va='center',
                           transform=ax_kpi.transAxes, fontsize=24, fontweight='bold')
                
                # Add change indicator
                change_color = 'green' if change_pct > 0 else 'red' if change_pct < 0 else 'gray'
                change_arrow = '↑' if change_pct > 0 else '↓' if change_pct < 0 else '→'
                ax_kpi.text(0.5, 0.4, f"{change_arrow} {abs(change_pct):.1f}%", 
                           ha='center', va='center', transform=ax_kpi.transAxes,
                           fontsize=14, color=change_color, fontweight='bold')
                
                # Add KPI title
                ax_kpi.text(0.5, 0.15, kpi.replace('_', ' ').title(), 
                           ha='center', va='center', transform=ax_kpi.transAxes,
                           fontsize=12, fontweight='bold')
                
                # Style KPI card background
                ax_kpi.set_xlim(0, 1)
                ax_kpi.set_ylim(0, 1)
                ax_kpi.set_xticks([])
                ax_kpi.set_yticks([])
                
                # Add subtle background color based on performance
                if change_pct > 5:
                    ax_kpi.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.2))
                elif change_pct < -5:
                    ax_kpi.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.2))
                else:
                    ax_kpi.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightyellow', alpha=0.2))
                
                # Add border
                for spine in ax_kpi.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color('gray')
            
            # 2. Primary KPI Trend Analysis (Second Row, Left)
            ax_trend = fig.add_subplot(gs[1, :3])
            
            if 'date' in kpi_data.columns:
                dates = pd.to_datetime(kpi_data['date'])
                
                # Plot primary KPI with trend line
                ax_trend.plot(dates, kpi_data[primary_kpi], linewidth=3, marker='o', 
                             markersize=6, alpha=0.8, label='Actual')
                
                # Add trend line if trend analysis available
                if 'trend_line' in trend_analysis and primary_kpi in trend_analysis['trend_line']:
                    trend_values = trend_analysis['trend_line'][primary_kpi]
                    ax_trend.plot(dates, trend_values, linestyle='--', linewidth=2, 
                                 alpha=0.7, label='Trend', color='red')
                
                # Add forecast if available
                if 'forecast' in trend_analysis and primary_kpi in trend_analysis['forecast']:
                    forecast_data = trend_analysis['forecast'][primary_kpi]
                    forecast_dates = pd.date_range(start=dates.iloc[-1], 
                                                 periods=len(forecast_data)+1, freq='D')[1:]
                    ax_trend.plot(forecast_dates, forecast_data, linestyle=':', 
                                 linewidth=2, alpha=0.6, label='Forecast', color='orange')
                    
                    # Add confidence bands if available
                    if 'forecast_upper' in trend_analysis and primary_kpi in trend_analysis['forecast_upper']:
                        upper_bound = trend_analysis['forecast_upper'][primary_kpi]
                        lower_bound = trend_analysis.get('forecast_lower', {}).get(primary_kpi, forecast_data)
                        ax_trend.fill_between(forecast_dates, lower_bound, upper_bound, 
                                            alpha=0.2, color='orange')
                
                ax_trend.set_xlabel('Date')
                ax_trend.set_ylabel(primary_kpi.replace('_', ' ').title())
                ax_trend.set_title(f'Primary KPI Trend: {primary_kpi.replace("_", " ").title()}')
                ax_trend.legend()
                ax_trend.grid(True, alpha=0.3)
                
                # Format date labels
                ax_trend.tick_params(axis='x', rotation=45)
            else:
                ax_trend.text(0.5, 0.5, 'Date Information\nNot Available',
                             ha='center', va='center', transform=ax_trend.transAxes)
                ax_trend.set_title('Primary KPI Trend Analysis')
            
            # 3. Performance Scorecard (Second Row, Right)
            ax_scorecard = fig.add_subplot(gs[1, 3])
            ax_scorecard.axis('off')
            
            scorecard_text = ["Performance Scorecard:"]
            scorecard_text.append("=" * 22)
            
            # Calculate performance scores
            for kpi in kpi_columns[:6]:  # Top 6 KPIs
                current_val = kpi_data[kpi].iloc[-1]
                previous_val = kpi_data[kpi].iloc[-2] if len(kpi_data) > 1 else current_val
                change = ((current_val - previous_val) / previous_val * 100) if previous_val != 0 else 0
                
                # Assign performance rating
                if change > 10:
                    rating = "Excellent ⭐⭐⭐"
                elif change > 5:
                    rating = "Good ⭐⭐"
                elif change > 0:
                    rating = "Fair ⭐"
                elif change > -5:
                    rating = "Caution ⚠️"
                else:
                    rating = "Action Needed 🔴"
                
                scorecard_text.append(f"{kpi.replace('_', ' ').title()}")
                scorecard_text.append(f"  {rating}")
                scorecard_text.append(f"  Change: {change:+.1f}%")
                scorecard_text.append("")
            
            ax_scorecard.text(0.05, 0.95, '\n'.join(scorecard_text),
                             transform=ax_scorecard.transAxes, fontsize=10,
                             verticalalignment='top', fontfamily='monospace')
            ax_scorecard.set_title('Performance Scorecard', fontsize=14, fontweight='bold')
            
            # 4. Comparative Analysis (Third Row)
            ax_comparison = fig.add_subplot(gs[2, :2])
            
            if 'period_comparison' in comparative_metrics:
                comparison_data = comparative_metrics['period_comparison']
                
                metrics = list(comparison_data.keys())[:8]  # Top 8 metrics
                current_period = [comparison_data[metric].get('current', 0) for metric in metrics]
                previous_period = [comparison_data[metric].get('previous', 0) for metric in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                bars1 = ax_comparison.bar(x - width/2, current_period, width, 
                                        label='Current Period', alpha=0.8)
                bars2 = ax_comparison.bar(x + width/2, previous_period, width, 
                                        label='Previous Period', alpha=0.8)
                
                ax_comparison.set_xlabel('Metrics')
                ax_comparison.set_ylabel('Values')
                ax_comparison.set_title('Period-over-Period Comparison')
                ax_comparison.set_xticks(x)
                ax_comparison.set_xticklabels([m.replace('_', ' ').title() for m in metrics], 
                                            rotation=45, ha='right')
                ax_comparison.legend()
                ax_comparison.grid(True, alpha=0.3)
                
                # Color bars based on improvement
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    if current_period[i] > previous_period[i]:
                        bar1.set_color('green')
                    elif current_period[i] < previous_period[i]:
                        bar1.set_color('red')
                    else:
                        bar1.set_color('gray')
            else:
                ax_comparison.text(0.5, 0.5, 'Comparative Metrics\nNot Available',
                                 ha='center', va='center', transform=ax_comparison.transAxes)
                ax_comparison.set_title('Period-over-Period Comparison')
            
            # 5. Market Performance Matrix (Third Row, Right)
            ax_matrix = fig.add_subplot(gs[2, 2:])
            
            if 'market_position' in comparative_metrics:
                # Create performance vs market growth matrix
                performance_data = comparative_metrics['market_position']
                
                companies = list(performance_data.keys())
                x_vals = [performance_data[comp].get('market_share_growth', 0) for comp in companies]
                y_vals = [performance_data[comp].get('revenue_growth', 0) for comp in companies]
                
                # Create quadrant analysis
                ax_matrix.scatter(x_vals, y_vals, s=100, alpha=0.7)
                
                # Add company labels
                for i, company in enumerate(companies):
                    ax_matrix.annotate(company, (x_vals[i], y_vals[i]), 
                                     xytext=(5, 5), textcoords='offset points')
                
                # Add quadrant lines
                ax_matrix.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax_matrix.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                # Add quadrant labels
                ax_matrix.text(0.05, 0.95, 'Stars\n(High Growth)', transform=ax_matrix.transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                ax_matrix.text(0.05, 0.05, 'Question Marks\n(Low Share)', transform=ax_matrix.transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                ax_matrix.text(0.65, 0.95, 'Cash Cows\n(High Share)', transform=ax_matrix.transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                ax_matrix.text(0.65, 0.05, 'Dogs\n(Low Growth)', transform=ax_matrix.transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                
                ax_matrix.set_xlabel('Market Share Growth (%)')
                ax_matrix.set_ylabel('Revenue Growth (%)')
                ax_matrix.set_title('Market Position Matrix')
                ax_matrix.grid(True, alpha=0.3)
            else:
                ax_matrix.text(0.5, 0.5, 'Market Position\nData Not Available',
                             ha='center', va='center', transform=ax_matrix.transAxes)
                ax_matrix.set_title('Market Position Matrix')
            
            # 6. Financial Health Indicators (Bottom Row, Left)
            ax_financial = fig.add_subplot(gs[3, :2])
            
            financial_metrics = ['profit_margin', 'cash_flow', 'debt_ratio', 'roa']
            financial_data = []
            financial_labels = []
            
            for metric in financial_metrics:
                if metric in kpi_data.columns:
                    current_val = kpi_data[metric].iloc[-1]
                    financial_data.append(current_val)
                    financial_labels.append(metric.replace('_', ' ').title())
            
            if financial_data:
                # Create radar chart for financial health
                angles = np.linspace(0, 2 * np.pi, len(financial_data), endpoint=False)
                financial_data = np.array(financial_data)
                
                # Normalize data to 0-1 scale for radar chart
                normalized_data = (financial_data - financial_data.min()) / (financial_data.max() - financial_data.min())
                
                # Close the plot
                angles = np.concatenate((angles, [angles[0]]))
                normalized_data = np.concatenate((normalized_data, [normalized_data[0]]))
                
                ax_financial.plot(angles, normalized_data, 'o-', linewidth=2, alpha=0.8)
                ax_financial.fill(angles, normalized_data, alpha=0.25)
                ax_financial.set_xticks(angles[:-1])
                ax_financial.set_xticklabels(financial_labels)
                ax_financial.set_title('Financial Health Radar')
                ax_financial.grid(True)
                
                # Convert to polar projection
                ax_financial.set_theta_offset(np.pi / 2)
                ax_financial.set_theta_direction(-1)
            else:
                ax_financial.text(0.5, 0.5, 'Financial Metrics\nNot Available',
                                ha='center', va='center', transform=ax_financial.transAxes)
                ax_financial.set_title('Financial Health Indicators')
            
            # 7. Executive Summary and Recommendations
            ax_summary = fig.add_subplot(gs[3, 2:])
            ax_summary.axis('off')
            
            summary_text = ["Executive Summary & Recommendations:"]
            summary_text.append("=" * 35)
            summary_text.append("")
            
            # Generate automated insights based on data
            insights = []
            
            # Revenue trend insight
            if primary_kpi in kpi_data.columns:
                recent_trend = kpi_data[primary_kpi].iloc[-5:].pct_change().mean() * 100
                if recent_trend > 5:
                    insights.append("📈 Strong revenue growth momentum")
                elif recent_trend < -5:
                    insights.append("📉 Revenue declining - action needed")
                else:
                    insights.append("📊 Revenue stable - monitor closely")
            
            # Performance consistency
            if len(kpi_columns) > 1:
                kpi_changes = []
                for kpi in kpi_columns[:3]:
                    if len(kpi_data) > 1:
                        change = ((kpi_data[kpi].iloc[-1] - kpi_data[kpi].iloc[-2]) / 
                                kpi_data[kpi].iloc[-2] * 100)
                        kpi_changes.append(change)
                
                if kpi_changes:
                    avg_change = np.mean(kpi_changes)
                    if avg_change > 3:
                        insights.append("🎯 Consistent positive performance")
                    elif avg_change < -3:
                        insights.append("⚠️ Multiple metrics declining")
                    else:
                        insights.append("⚖️ Mixed performance signals")
            
            # Add executive context insights
            if executive_context:
                if executive_context.get('quarterly_target_status') == 'on_track':
                    insights.append("✅ On track for quarterly targets")
                elif executive_context.get('quarterly_target_status') == 'at_risk':
                    insights.append("🚨 Quarterly targets at risk")
                
                if executive_context.get('market_conditions') == 'favorable':
                    insights.append("🌟 Market conditions favorable")
            
            # Format recommendations
            summary_text.extend(["Key Insights:"] + [f"  {insight}" for insight in insights])
            summary_text.append("")
            summary_text.append("Strategic Actions:")
            summary_text.append("  • Focus on high-performing segments")
            summary_text.append("  • Address declining metrics promptly")
            summary_text.append("  • Monitor competitive positioning")
            summary_text.append("  • Optimize resource allocation")
            
            if executive_context:
                summary_text.append("")
                summary_text.append("Context:")
                for key, value in executive_context.items():
                    summary_text.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            ax_summary.text(0.05, 0.95, '\n'.join(summary_text),
                           transform=ax_summary.transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='sans-serif')
            ax_summary.set_title('Executive Insights', fontsize=14, fontweight='bold')
            
            # Set overall figure title with enterprise branding
            fig.suptitle('Executive Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)
            
            # Add enterprise metadata and timestamp
            fig.text(0.01, 0.01, f'Dashboard Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='left', va='bottom', fontsize=10, alpha=0.7)
            fig.text(0.99, 0.01, 'Confidential - Executive Use Only', 
                    ha='right', va='bottom', fontsize=10, alpha=0.7, style='italic')
            
            logger.info("Successfully created enterprise executive performance dashboard")
            
            # Note: FigureDataSet applies sophisticated enterprise styling automatically
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create executive performance dashboard: {e}")
        # Create professional fallback figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, f'Executive Dashboard Creation Failed\n{str(e)}\n\nPlease contact IT support', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Executive Performance Dashboard - Service Unavailable', fontsize=16)
        return fig


@performance_monitor("statistical_analysis_report", target_ms=80.0)
def create_statistical_analysis_report(
    analysis_results: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
    statistical_config: Dict[str, Any],
    publication_context: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create comprehensive statistical analysis report with publication-ready formatting.
    
    This function demonstrates the most sophisticated statistical visualization patterns
    with automated academic/professional styling through figregistry-kedro. The report
    adapts formatting based on publication type, statistical significance levels, and
    audience requirements without any manual styling intervention.
    
    Args:
        analysis_results: Dictionary containing comprehensive statistical analysis results
        dataset_metadata: Metadata about the analyzed datasets
        statistical_config: Configuration parameters for statistical methods
        publication_context: Publication and audience context for styling adaptation
        
    Returns:
        matplotlib.Figure: Publication-ready statistical report with automated styling
        
    Note:
        Demonstrates the pinnacle of F-005 integration with:
        - Publication type (journal/conference/internal) → formatting strictness
        - Statistical significance → visual emphasis and annotation
        - Analysis complexity → layout adaptation and detail level
        - Academic/business audience → terminology and presentation style
    """
    try:
        with performance_context("statistical_report_creation"):
            logger.info("Creating comprehensive statistical analysis report")
            
            # Create publication-quality figure layout
            fig = plt.figure(figsize=(18, 16))
            gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)
            
            # Configure academic/publication styling
            plt.rcParams.update({
                'font.size': 11,
                'font.family': 'serif',
                'axes.grid': True,
                'grid.alpha': 0.2,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'figure.dpi': ENTERPRISE_FIGURE_DPI
            })
            
            # 1. Descriptive Statistics Summary
            ax_descriptive = fig.add_subplot(gs[0, 0])
            
            if 'descriptive_statistics' in analysis_results:
                desc_stats = analysis_results['descriptive_statistics']
                
                # Create statistical summary table visualization
                variables = list(desc_stats.keys())[:6]  # Top 6 variables
                statistics = ['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']
                
                # Prepare data matrix
                data_matrix = []
                for stat in statistics:
                    row = []
                    for var in variables:
                        if var in desc_stats and stat in desc_stats[var]:
                            value = desc_stats[var][stat]
                            if isinstance(value, (int, float)):
                                row.append(f"{value:.3f}")
                            else:
                                row.append(str(value))
                        else:
                            row.append("N/A")
                    data_matrix.append(row)
                
                # Create table visualization
                table = ax_descriptive.table(cellText=data_matrix,
                                           rowLabels=statistics,
                                           colLabels=[v.replace('_', ' ').title() for v in variables],
                                           cellLoc='center',
                                           loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                # Style table headers
                for i in range(len(variables)):
                    table[(0, i)].set_facecolor('#E6E6FA')
                    table[(0, i)].set_text_props(weight='bold')
                
                for i in range(len(statistics)):
                    table[(i+1, -1)].set_facecolor('#E6E6FA')
                    table[(i+1, -1)].set_text_props(weight='bold')
                
                ax_descriptive.axis('off')
                ax_descriptive.set_title('Descriptive Statistics Summary', fontweight='bold')
            else:
                ax_descriptive.text(0.5, 0.5, 'Descriptive Statistics\nNot Available',
                                  ha='center', va='center', transform=ax_descriptive.transAxes)
                ax_descriptive.set_title('Descriptive Statistics Summary')
            
            # 2. Hypothesis Testing Results
            ax_hypothesis = fig.add_subplot(gs[0, 1])
            
            if 'hypothesis_tests' in analysis_results:
                hypothesis_results = analysis_results['hypothesis_tests']
                
                test_names = []
                p_values = []
                test_statistics = []
                
                for test_type, test_data in hypothesis_results.items():
                    if isinstance(test_data, dict):
                        for test_name, result in test_data.items():
                            if 'p_value' in result:
                                test_names.append(f"{test_type}_{test_name}")
                                p_values.append(result['p_value'])
                                test_statistics.append(result.get('statistic', 0))
                
                if test_names:
                    # Create horizontal bar chart of p-values
                    y_pos = np.arange(len(test_names))
                    bars = ax_hypothesis.barh(y_pos, p_values, alpha=0.7)
                    
                    # Color bars by significance
                    significance_threshold = statistical_config.get('significance_level', 0.05)
                    for bar, p_val in zip(bars, p_values):
                        if p_val < 0.001:
                            bar.set_color('darkgreen')
                        elif p_val < 0.01:
                            bar.set_color('green')
                        elif p_val < significance_threshold:
                            bar.set_color('orange')
                        else:
                            bar.set_color('red')
                    
                    # Add significance threshold line
                    ax_hypothesis.axvline(x=significance_threshold, color='red', 
                                        linestyle='--', alpha=0.7, label=f'α = {significance_threshold}')
                    
                    ax_hypothesis.set_yticks(y_pos)
                    ax_hypothesis.set_yticklabels([name.replace('_', ' ').title() for name in test_names])
                    ax_hypothesis.set_xlabel('p-value')
                    ax_hypothesis.set_title('Hypothesis Test Results')
                    ax_hypothesis.legend()
                    ax_hypothesis.grid(True, alpha=0.3, axis='x')
                else:
                    ax_hypothesis.text(0.5, 0.5, 'Hypothesis Tests\nNot Available',
                                     ha='center', va='center', transform=ax_hypothesis.transAxes)
            else:
                ax_hypothesis.text(0.5, 0.5, 'Hypothesis Tests\nNot Available',
                                 ha='center', va='center', transform=ax_hypothesis.transAxes)
            
            # 3. Correlation Analysis Heatmap
            ax_correlation = fig.add_subplot(gs[0, 2])
            
            if 'correlation_analysis' in analysis_results:
                corr_data = analysis_results['correlation_analysis']
                
                if 'pearson' in corr_data and 'matrix' in corr_data['pearson']:
                    corr_matrix = pd.DataFrame(corr_data['pearson']['matrix'])
                    
                    # Create correlation heatmap
                    im = ax_correlation.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax_correlation.set_xticks(range(len(corr_matrix.columns)))
                    ax_correlation.set_yticks(range(len(corr_matrix.columns)))
                    ax_correlation.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                    ax_correlation.set_yticklabels(corr_matrix.columns)
                    
                    # Add correlation values to cells
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.columns)):
                            value = corr_matrix.iloc[i, j]
                            text = ax_correlation.text(j, i, f'{value:.2f}',
                                                     ha='center', va='center',
                                                     color='white' if abs(value) > 0.6 else 'black',
                                                     fontsize=8)
                    
                    ax_correlation.set_title('Correlation Matrix (Pearson)')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax_correlation, shrink=0.8)
                    cbar.set_label('Correlation Coefficient')
                else:
                    ax_correlation.text(0.5, 0.5, 'Correlation Matrix\nNot Available',
                                      ha='center', va='center', transform=ax_correlation.transAxes)
            else:
                ax_correlation.text(0.5, 0.5, 'Correlation Analysis\nNot Available',
                                  ha='center', va='center', transform=ax_correlation.transAxes)
            
            # 4. Distribution Analysis
            ax_distributions = fig.add_subplot(gs[1, :])
            
            if 'distribution_analysis' in analysis_results:
                dist_data = analysis_results['distribution_analysis']
                
                # Show distribution test results for multiple variables
                variables = list(dist_data.keys())[:4]  # Top 4 variables
                
                if variables:
                    # Create subplots for each distribution
                    n_vars = len(variables)
                    
                    for i, var in enumerate(variables):
                        var_data = dist_data[var]
                        
                        # Create mini subplot within the distribution section
                        sub_ax = plt.subplot2grid((1, n_vars), (0, i), fig=fig)
                        
                        # Extract distribution fit information
                        distributions = ['normal', 'exponential', 'lognormal']
                        dist_names = []
                        ks_stats = []
                        p_values = []
                        
                        for dist in distributions:
                            if dist in var_data:
                                dist_names.append(dist.title())
                                ks_stats.append(var_data[dist].get('ks_statistic', 0))
                                p_values.append(var_data[dist].get('ks_p_value', 1))
                        
                        if dist_names:
                            # Create bar chart of KS statistics
                            bars = sub_ax.bar(dist_names, ks_stats, alpha=0.7)
                            
                            # Color by goodness of fit (lower KS stat = better fit)
                            max_ks = max(ks_stats) if ks_stats else 1
                            for bar, ks_stat, p_val in zip(bars, ks_stats, p_values):
                                if p_val > 0.05:  # Good fit
                                    bar.set_color('green')
                                elif p_val > 0.01:  # Moderate fit
                                    bar.set_color('orange')
                                else:  # Poor fit
                                    bar.set_color('red')
                            
                            sub_ax.set_title(f'{var.replace("_", " ").title()}', fontsize=10)
                            sub_ax.set_ylabel('KS Statistic', fontsize=9)
                            sub_ax.tick_params(axis='x', rotation=45, labelsize=8)
                            sub_ax.tick_params(axis='y', labelsize=8)
                            sub_ax.grid(True, alpha=0.3)
                        else:
                            sub_ax.text(0.5, 0.5, 'No Distribution\nFits Available',
                                      ha='center', va='center', transform=sub_ax.transAxes,
                                      fontsize=8)
                            sub_ax.set_title(f'{var.replace("_", " ").title()}', fontsize=10)
                    
                    # Remove the main axis
                    ax_distributions.axis('off')
                    ax_distributions.set_title('Distribution Analysis - Goodness of Fit Tests', 
                                             fontweight='bold', pad=20)
                else:
                    ax_distributions.text(0.5, 0.5, 'Distribution Analysis\nData Not Available',
                                        ha='center', va='center', transform=ax_distributions.transAxes)
            else:
                ax_distributions.text(0.5, 0.5, 'Distribution Analysis\nNot Available',
                                    ha='center', va='center', transform=ax_distributions.transAxes)
            
            # 5. Outlier Detection Summary
            ax_outliers = fig.add_subplot(gs[2, 0])
            
            if 'outlier_analysis' in analysis_results:
                outlier_data = analysis_results['outlier_analysis']
                
                variables = list(outlier_data.keys())
                methods = ['iqr', 'zscore', 'modified_zscore']
                
                # Create stacked bar chart of outlier percentages
                outlier_percentages = {method: [] for method in methods}
                var_names = []
                
                for var in variables[:6]:  # Top 6 variables
                    var_names.append(var.replace('_', ' ').title())
                    for method in methods:
                        if method in outlier_data[var]:
                            pct = outlier_data[var][method].get('outlier_percentage', 0)
                            outlier_percentages[method].append(pct)
                        else:
                            outlier_percentages[method].append(0)
                
                if var_names:
                    x = np.arange(len(var_names))
                    width = 0.25
                    
                    for i, method in enumerate(methods):
                        ax_outliers.bar(x + i*width, outlier_percentages[method], 
                                      width, label=method.upper(), alpha=0.8)
                    
                    ax_outliers.set_xlabel('Variables')
                    ax_outliers.set_ylabel('Outlier Percentage (%)')
                    ax_outliers.set_title('Outlier Detection Results')
                    ax_outliers.set_xticks(x + width)
                    ax_outliers.set_xticklabels(var_names, rotation=45, ha='right')
                    ax_outliers.legend()
                    ax_outliers.grid(True, alpha=0.3)
                else:
                    ax_outliers.text(0.5, 0.5, 'Outlier Analysis\nNot Available',
                                   ha='center', va='center', transform=ax_outliers.transAxes)
            else:
                ax_outliers.text(0.5, 0.5, 'Outlier Analysis\nNot Available',
                                ha='center', va='center', transform=ax_outliers.transAxes)
            
            # 6. Time Series Analysis (if available)
            ax_timeseries = fig.add_subplot(gs[2, 1])
            
            if 'time_series_analysis' in analysis_results:
                ts_data = analysis_results['time_series_analysis']
                
                variables = list(ts_data.keys())[:3]  # Top 3 time series
                
                if variables:
                    # Create trend analysis summary
                    trend_directions = []
                    trend_strengths = []
                    var_labels = []
                    
                    for var in variables:
                        if 'trend' in ts_data[var]:
                            trend_info = ts_data[var]['trend']
                            direction = trend_info.get('trend_direction', 'stable')
                            r_squared = trend_info.get('r_squared', 0)
                            
                            trend_directions.append(direction)
                            trend_strengths.append(r_squared)
                            var_labels.append(var.replace('_', ' ').title())
                    
                    if var_labels:
                        # Create horizontal bar chart of trend strengths
                        y_pos = np.arange(len(var_labels))
                        bars = ax_timeseries.barh(y_pos, trend_strengths, alpha=0.7)
                        
                        # Color by trend direction
                        direction_colors = {'increasing': 'green', 'decreasing': 'red', 'stable': 'gray'}
                        for bar, direction in zip(bars, trend_directions):
                            bar.set_color(direction_colors.get(direction, 'blue'))
                        
                        ax_timeseries.set_yticks(y_pos)
                        ax_timeseries.set_yticklabels(var_labels)
                        ax_timeseries.set_xlabel('Trend Strength (R²)')
                        ax_timeseries.set_title('Time Series Trend Analysis')
                        ax_timeseries.grid(True, alpha=0.3, axis='x')
                        
                        # Add trend direction legend
                        for direction, color in direction_colors.items():
                            ax_timeseries.barh(-1, 0, color=color, label=direction.title())
                        ax_timeseries.legend()
                    else:
                        ax_timeseries.text(0.5, 0.5, 'Time Series Trends\nNot Available',
                                         ha='center', va='center', transform=ax_timeseries.transAxes)
                else:
                    ax_timeseries.text(0.5, 0.5, 'Time Series Variables\nNot Found',
                                     ha='center', va='center', transform=ax_timeseries.transAxes)
            else:
                ax_timeseries.text(0.5, 0.5, 'Time Series Analysis\nNot Available',
                                 ha='center', va='center', transform=ax_timeseries.transAxes)
            
            # 7. Regression Analysis Summary
            ax_regression = fig.add_subplot(gs[2, 2])
            
            if 'regression_analysis' in analysis_results:
                reg_data = analysis_results['regression_analysis']
                
                # Extract regression results
                relationships = []
                r_squared_values = []
                p_values = []
                
                for relationship, results in reg_data.items():
                    if 'multiple_regression' not in relationship:
                        relationships.append(relationship.replace('_vs_', ' → '))
                        r_squared_values.append(results.get('r_squared', 0))
                        p_values.append(results.get('p_value', 1))
                
                if relationships:
                    # Create scatter plot of R² vs p-values
                    colors = ['green' if p < 0.05 else 'red' for p in p_values]
                    scatter = ax_regression.scatter(p_values, r_squared_values, 
                                                  c=colors, alpha=0.7, s=100)
                    
                    # Add labels for significant relationships
                    for i, (rel, p_val, r2) in enumerate(zip(relationships, p_values, r_squared_values)):
                        if p_val < 0.05:
                            ax_regression.annotate(rel, (p_val, r2), 
                                                 xytext=(5, 5), textcoords='offset points',
                                                 fontsize=8)
                    
                    ax_regression.axvline(x=0.05, color='red', linestyle='--', alpha=0.5)
                    ax_regression.set_xlabel('p-value')
                    ax_regression.set_ylabel('R² Value')
                    ax_regression.set_title('Regression Relationships')
                    ax_regression.grid(True, alpha=0.3)
                    
                    # Add significance region shading
                    ax_regression.axvspan(0, 0.05, alpha=0.1, color='green', label='Significant')
                    ax_regression.legend()
                else:
                    ax_regression.text(0.5, 0.5, 'Regression Results\nNot Available',
                                     ha='center', va='center', transform=ax_regression.transAxes)
            else:
                ax_regression.text(0.5, 0.5, 'Regression Analysis\nNot Available',
                                 ha='center', va='center', transform=ax_regression.transAxes)
            
            # 8. Statistical Methods Summary
            ax_methods = fig.add_subplot(gs[3, :])
            ax_methods.axis('off')
            
            methods_text = ["Statistical Analysis Summary:"]
            methods_text.append("=" * 40)
            methods_text.append("")
            
            # Dataset information
            methods_text.append("Dataset Information:")
            methods_text.append(f"  Sample Size: {dataset_metadata.get('sample_size', 'Unknown')}")
            methods_text.append(f"  Variables: {dataset_metadata.get('variable_count', 'Unknown')}")
            methods_text.append(f"  Missing Data: {dataset_metadata.get('missing_percentage', 'Unknown')}%")
            methods_text.append("")
            
            # Statistical methods applied
            methods_text.append("Methods Applied:")
            for method, config in statistical_config.items():
                if isinstance(config, dict):
                    methods_text.append(f"  {method.replace('_', ' ').title()}: {config.get('description', 'Applied')}")
                else:
                    methods_text.append(f"  {method.replace('_', ' ').title()}: {config}")
            methods_text.append("")
            
            # Key findings
            methods_text.append("Key Statistical Findings:")
            
            # Hypothesis test summary
            if 'hypothesis_tests' in analysis_results:
                significant_tests = 0
                total_tests = 0
                for test_type, test_data in analysis_results['hypothesis_tests'].items():
                    if isinstance(test_data, dict):
                        for test_name, result in test_data.items():
                            total_tests += 1
                            if result.get('p_value', 1) < statistical_config.get('significance_level', 0.05):
                                significant_tests += 1
                
                methods_text.append(f"  • {significant_tests}/{total_tests} hypothesis tests significant")
            
            # Correlation summary
            if 'correlation_analysis' in analysis_results:
                corr_data = analysis_results['correlation_analysis']
                if 'pearson' in corr_data and 'strong_correlations' in corr_data['pearson']:
                    strong_corr_count = len(corr_data['pearson']['strong_correlations'])
                    methods_text.append(f"  • {strong_corr_count} strong correlations identified")
            
            # Outlier summary
            if 'outlier_analysis' in analysis_results:
                total_outlier_pct = 0
                var_count = 0
                for var, methods in analysis_results['outlier_analysis'].items():
                    for method, results in methods.items():
                        total_outlier_pct += results.get('outlier_percentage', 0)
                        var_count += 1
                
                if var_count > 0:
                    avg_outlier_pct = total_outlier_pct / var_count
                    methods_text.append(f"  • Average outlier rate: {avg_outlier_pct:.1f}%")
            
            # Publication context
            if publication_context:
                methods_text.append("")
                methods_text.append("Publication Context:")
                for key, value in publication_context.items():
                    methods_text.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Recommendations
            methods_text.append("")
            methods_text.append("Statistical Recommendations:")
            methods_text.append("  • Review significant test results for practical importance")
            methods_text.append("  • Consider effect sizes alongside p-values")  
            methods_text.append("  • Validate findings with independent datasets")
            methods_text.append("  • Address outliers based on domain knowledge")
            
            ax_methods.text(0.05, 0.95, '\n'.join(methods_text),
                           transform=ax_methods.transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='serif')
            ax_methods.set_title('Analysis Methodology & Findings Summary', 
                               fontsize=14, fontweight='bold', pad=20)
            
            # Set overall figure title
            report_title = publication_context.get('title', 'Statistical Analysis Report') if publication_context else 'Statistical Analysis Report'
            fig.suptitle(report_title, fontsize=18, fontweight='bold', y=0.96)
            
            # Add publication metadata
            fig.text(0.01, 0.01, f'Report Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='left', va='bottom', fontsize=9, alpha=0.7)
            
            # Add significance level note
            sig_level = statistical_config.get('significance_level', 0.05)
            fig.text(0.99, 0.01, f'Significance Level: α = {sig_level}', 
                    ha='right', va='bottom', fontsize=9, alpha=0.7)
            
            logger.info("Successfully created comprehensive statistical analysis report")
            
            # Note: FigureDataSet applies publication-ready styling automatically
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create statistical analysis report: {e}")
        # Create professional fallback figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, f'Statistical Analysis Report Generation Failed\n{str(e)}\n\nPlease review input data and configuration', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Statistical Analysis Report - Generation Error', fontsize=14)
        return fig


# Utility Functions for Advanced Node Support

def validate_experimental_configuration(config: Dict[str, Any]) -> ExperimentalConfiguration:
    """
    Validate and create ExperimentalConfiguration from dictionary.
    
    This utility function demonstrates sophisticated parameter validation and
    condition preparation for advanced experimental scenarios. It ensures
    that experimental configurations are properly structured for condition-
    based styling automation.
    
    Args:
        config: Dictionary containing experimental configuration parameters
        
    Returns:
        ExperimentalConfiguration: Validated experimental configuration object
        
    Raises:
        ValueError: When configuration parameters are invalid or incomplete
    """
    required_fields = ['experiment_id', 'treatment_groups', 'control_group']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required experimental configuration fields: {missing_fields}")
    
    # Validate treatment groups
    if not isinstance(config['treatment_groups'], list) or len(config['treatment_groups']) == 0:
        raise ValueError("Treatment groups must be a non-empty list")
    
    # Validate control group
    if not isinstance(config['control_group'], str) or not config['control_group'].strip():
        raise ValueError("Control group must be a non-empty string")
    
    # Create experimental configuration with defaults
    exp_config = ExperimentalConfiguration(
        experiment_id=config['experiment_id'],
        treatment_groups=config['treatment_groups'],
        control_group=config['control_group'],
        experimental_factors=config.get('experimental_factors', {}),
        baseline_metrics=config.get('baseline_metrics', {}),
        statistical_power=config.get('statistical_power', 0.8),
        effect_size_threshold=config.get('effect_size_threshold', 0.1)
    )
    
    logger.debug(f"Validated experimental configuration: {exp_config.experiment_id}")
    return exp_config


def prepare_condition_hierarchy(
    experiment_config: ExperimentalConfiguration,
    performance_context: Dict[str, Any],
    business_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare sophisticated condition hierarchy for advanced styling resolution.
    
    This function demonstrates how to construct complex experimental condition
    hierarchies that leverage figregistry-kedro's advanced pattern matching
    and style inheritance capabilities for enterprise visualization scenarios.
    
    Args:
        experiment_config: Experimental configuration with treatment information
        performance_context: Model/system performance context
        business_context: Business impact and stakeholder context
        
    Returns:
        Dict[str, Any]: Hierarchical condition structure for styling resolution
        
    Note:
        This hierarchy enables sophisticated F-005-RQ-004 condition resolution
        with inheritance, pattern matching, and dynamic style adaptation.
    """
    # Build comprehensive condition hierarchy
    condition_hierarchy = {
        "experiment": experiment_config.get_condition_hierarchy(),
        "performance": {
            "tier": performance_context.get('performance_tier', 'medium'),
            "metrics": performance_context.get('key_metrics', {}),
            "trend": performance_context.get('trend_direction', 'stable'),
            "confidence": performance_context.get('confidence_level', 0.95)
        },
        "business": {
            "impact": business_context.get('business_impact', 'medium'),
            "stakeholders": business_context.get('target_audience', 'general'),
            "urgency": business_context.get('urgency_level', 'normal'),
            "presentation_type": business_context.get('presentation_type', 'internal')
        },
        "technical": {
            "complexity": 'high',  # Advanced visualizations
            "automation_level": 'full',  # Complete figregistry-kedro automation
            "style_inheritance": 'hierarchical'  # Advanced style resolution
        }
    }
    
    logger.debug("Prepared advanced condition hierarchy for styling resolution")
    return condition_hierarchy


def calculate_enterprise_performance_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive enterprise performance metrics for visualization.
    
    This utility function demonstrates advanced metric calculation patterns
    that support sophisticated business intelligence visualizations with
    automated styling based on performance characteristics.
    
    Args:
        data: DataFrame containing business/performance data
        
    Returns:
        Dict[str, Any]: Comprehensive performance metrics and metadata
    """
    try:
        metrics = {}
        
        # Basic statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            column_data = data[column].dropna()
            
            if len(column_data) > 0:
                metrics[column] = {
                    'current_value': float(column_data.iloc[-1]),
                    'mean': float(column_data.mean()),
                    'std': float(column_data.std()),
                    'min': float(column_data.min()),
                    'max': float(column_data.max()),
                    'trend': 'increasing' if column_data.iloc[-1] > column_data.mean() else 'decreasing',
                    'volatility': float(column_data.std() / column_data.mean()) if column_data.mean() != 0 else 0
                }
                
                # Performance classification
                if column_data.iloc[-1] > column_data.quantile(0.8):
                    metrics[column]['performance_tier'] = 'excellent'
                elif column_data.iloc[-1] > column_data.quantile(0.6):
                    metrics[column]['performance_tier'] = 'good'
                elif column_data.iloc[-1] > column_data.quantile(0.4):
                    metrics[column]['performance_tier'] = 'average'
                else:
                    metrics[column]['performance_tier'] = 'needs_attention'
        
        # Growth rates if data has time component
        if len(data) > 1:
            for column in numeric_columns:
                column_data = data[column].dropna()
                if len(column_data) > 1:
                    growth_rate = ((column_data.iloc[-1] - column_data.iloc[0]) / column_data.iloc[0] * 100) if column_data.iloc[0] != 0 else 0
                    metrics[column]['growth_rate'] = float(growth_rate)
        
        logger.debug(f"Calculated enterprise metrics for {len(numeric_columns)} indicators")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate enterprise performance metrics: {e}")
        return {}


# Export comprehensive API for advanced usage
__all__ = [
    # Core node functions
    "create_advanced_training_metrics_dashboard",
    "create_hyperparameter_optimization_analysis", 
    "create_model_inference_analysis",
    "create_ab_testing_analysis",
    "create_executive_performance_dashboard",
    "create_statistical_analysis_report",
    
    # Utility classes and functions
    "ExperimentalConfiguration",
    "validate_experimental_configuration",
    "prepare_condition_hierarchy",
    "calculate_enterprise_performance_metrics",
    
    # Configuration constants
    "ENTERPRISE_STYLING_CONFIGS",
    "ENTERPRISE_FIGURE_DPI",
    "DEFAULT_CONFIDENCE_INTERVAL"
]