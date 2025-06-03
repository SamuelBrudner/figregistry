"""Advanced Kedro node functions for sophisticated figure generation with automated FigRegistry styling.

This module implements enterprise-grade node functions demonstrating sophisticated figure 
generation patterns with complex experimental conditions, multi-variable analysis, and 
production-ready visualization workflows. These functions showcase advanced figregistry-kedro 
integration patterns that eliminate manual figure management through automated styling and 
versioning, supporting complex data science workflows per F-005 requirements.

Key Features Demonstrated:
- Advanced condition-based styling with multi-variable experimental conditions per F-002
- Sophisticated matplotlib figure creation without manual styling per F-005-RQ-001
- Complex experimental design scenarios (multi-treatment studies, A/B testing) per F-002
- Enterprise-grade visualizations for training, inference, and reporting pipelines per F-004
- Production-ready patterns with comprehensive error handling and edge case management
- Automated zero-touch figure management through FigureDataSet integration per Section 0.1.1

All node functions demonstrate the core principle of outputting raw matplotlib figures that
are automatically styled and saved through the FigRegistry-Kedro integration, eliminating
manual plt.savefig() calls and ensuring consistent visualization across complex workflows.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns

# Advanced utilities for sophisticated analysis patterns
from .utils import (
    ExperimentalCondition,
    AdvancedConditionResolver, 
    DataTransformationHelper,
    StatisticalAnalysisHelper,
    ConfigurationManager,
    prepare_experimental_data,
    calculate_advanced_statistics
)

logger = logging.getLogger(__name__)

# Configure matplotlib for high-quality output
plt.style.use('default')  # Reset to default - styling handled by FigRegistry
sns.set_palette("husl")


def create_training_loss_visualization(
    training_history: pd.DataFrame,
    validation_history: pd.DataFrame,
    model_config: Dict[str, Any],
    experiment_params: Dict[str, Any]
) -> Figure:
    """Generate sophisticated training loss curves with advanced statistical analysis.
    
    This function demonstrates enterprise-grade training visualization patterns with
    complex experimental condition resolution and advanced statistical overlays.
    The figure is automatically styled through FigureDataSet based on experimental
    conditions including model type, treatment group, and hyperparameter settings.
    
    Args:
        training_history: DataFrame with columns ['epoch', 'loss', 'accuracy', 'lr']
        validation_history: DataFrame with columns ['epoch', 'val_loss', 'val_accuracy']
        model_config: Dictionary containing model architecture and hyperparameter settings
        experiment_params: Pipeline parameters for condition-based styling resolution
        
    Returns:
        matplotlib.figure.Figure: Sophisticated training visualization with multiple
            subplots, statistical overlays, and production-ready annotations. The figure
            is automatically styled and saved through FigureDataSet integration without
            manual styling or save operations per F-005 requirements.
    
    Note:
        This function demonstrates zero-touch figure management per Section 0.1.1.
        The returned figure is intercepted by FigureDataSet.save() which automatically:
        - Resolves experimental conditions from experiment_params
        - Applies condition-based styling through get_style()
        - Handles versioning and file persistence through save_figure()
        - Eliminates all manual figure management code from the node function
    """
    # Log advanced experimental condition for monitoring
    logger.info(
        f"Generating training visualization for experiment: "
        f"model={model_config.get('architecture', 'unknown')}, "
        f"treatment={experiment_params.get('treatment', 'control')}, "
        f"environment={experiment_params.get('environment', 'development')}"
    )
    
    # Create sophisticated multi-panel figure layout
    fig = plt.figure(figsize=(16, 12))
    
    # Define complex grid layout for enterprise reporting
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[2, 2, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Main loss curves subplot with advanced statistical overlays
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Prepare time series data with advanced transformations
    helper = DataTransformationHelper()
    train_data = helper.prepare_time_series_data(
        data=training_history,
        time_column='epoch',
        value_columns=['loss', 'accuracy'],
        resample_freq=None  # Keep original epoch resolution
    )
    
    val_data = helper.prepare_time_series_data(
        data=validation_history,
        time_column='epoch', 
        value_columns=['val_loss', 'val_accuracy'],
        resample_freq=None
    )
    
    # Plot training and validation curves with confidence intervals
    epochs = train_data['epoch']
    
    # Primary loss curves
    ax_main.plot(epochs, train_data['loss'], label='Training Loss', 
                linewidth=2.5, alpha=0.8)
    ax_main.plot(epochs, val_data['val_loss'], label='Validation Loss',
                linewidth=2.5, alpha=0.8)
    
    # Add confidence intervals using rolling statistics
    if 'loss_std' in train_data.columns:
        ax_main.fill_between(epochs, 
                           train_data['loss'] - train_data['loss_std'],
                           train_data['loss'] + train_data['loss_std'],
                           alpha=0.2, label='Training ±1σ')
    
    # Add smoothed trend lines using moving averages
    if 'loss_ma7' in train_data.columns:
        ax_main.plot(epochs, train_data['loss_ma7'], '--', alpha=0.7,
                    label='Training Trend (7-epoch MA)')
    
    # Statistical annotations for model convergence analysis
    min_val_loss_epoch = val_data.loc[val_data['val_loss'].idxmin(), 'epoch']
    min_val_loss = val_data['val_loss'].min()
    
    ax_main.axvline(x=min_val_loss_epoch, color='red', linestyle=':', alpha=0.7,
                   label=f'Best Model (Epoch {min_val_loss_epoch})')
    ax_main.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                    xy=(min_val_loss_epoch, min_val_loss),
                    xytext=(min_val_loss_epoch + len(epochs) * 0.1, min_val_loss + 0.1),
                    arrowprops=dict(arrowstyle='->', alpha=0.7),
                    fontsize=10, ha='left')
    
    # Configure main subplot with enterprise styling placeholders
    # Note: Actual styling applied automatically by FigRegistry based on experimental condition
    ax_main.set_xlabel('Training Epoch')
    ax_main.set_ylabel('Loss Value')
    ax_main.set_title(f'Training Progress - {model_config.get("architecture", "Model")} '
                     f'({experiment_params.get("treatment", "baseline")} treatment)')
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    
    # Secondary accuracy subplot
    ax_acc = fig.add_subplot(gs[1, :2])
    
    ax_acc.plot(epochs, train_data['accuracy'], label='Training Accuracy',
               linewidth=2.5, alpha=0.8)
    ax_acc.plot(epochs, val_data['val_accuracy'], label='Validation Accuracy',
               linewidth=2.5, alpha=0.8)
    
    # Add plateau detection for early stopping analysis
    val_acc_plateau = detect_training_plateau(val_data['val_accuracy'].values, 
                                            patience=10, min_delta=0.001)
    if val_acc_plateau is not None:
        ax_acc.axvline(x=val_acc_plateau, color='orange', linestyle='--', alpha=0.7,
                      label=f'Plateau Detected (Epoch {val_acc_plateau})')
    
    ax_acc.set_xlabel('Training Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Model Accuracy Progression')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    
    # Learning rate subplot for hyperparameter analysis
    ax_lr = fig.add_subplot(gs[0, 2])
    
    if 'lr' in train_data.columns:
        ax_lr.plot(epochs, train_data['lr'], color='purple', linewidth=2)
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_title('LR Schedule')
        ax_lr.set_yscale('log')
        ax_lr.grid(True, alpha=0.3)
    
    # Gradient norm analysis subplot (if available)
    ax_grad = fig.add_subplot(gs[1, 2])
    
    if 'grad_norm' in train_data.columns:
        ax_grad.plot(epochs, train_data['grad_norm'], color='brown', linewidth=2)
        ax_grad.set_xlabel('Epoch')
        ax_grad.set_ylabel('Gradient Norm')
        ax_grad.set_title('Gradient Analysis')
        ax_grad.set_yscale('log')
        ax_grad.grid(True, alpha=0.3)
    
    # Advanced statistical summary panel
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')  # Remove axes for text display
    
    # Calculate comprehensive training statistics
    final_train_loss = train_data['loss'].iloc[-1]
    final_val_loss = val_data['val_loss'].iloc[-1]
    final_train_acc = train_data['accuracy'].iloc[-1]
    final_val_acc = val_data['val_accuracy'].iloc[-1]
    
    overfitting_score = final_val_loss - final_train_loss
    convergence_rate = calculate_convergence_rate(train_data['loss'].values)
    
    # Format statistical summary with experimental metadata
    stats_text = f"""
    Experimental Configuration: Model={model_config.get('architecture')} | Treatment={experiment_params.get('treatment')} | Environment={experiment_params.get('environment')}
    
    Final Metrics: Train Loss={final_train_loss:.4f} | Val Loss={final_val_loss:.4f} | Train Acc={final_train_acc:.3f} | Val Acc={final_val_acc:.3f}
    
    Model Analysis: Overfitting Score={overfitting_score:.4f} | Convergence Rate={convergence_rate:.6f}/epoch | Best Epoch={min_val_loss_epoch}
    
    Hyperparameters: Batch Size={model_config.get('batch_size', 'N/A')} | Optimizer={model_config.get('optimizer', 'N/A')} | Total Epochs={len(epochs)}
    """
    
    ax_stats.text(0.02, 0.8, stats_text.strip(), transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add experimental condition metadata for FigRegistry resolution
    # These metadata will be used by AdvancedConditionResolver for sophisticated styling
    condition_metadata = {
        'model_architecture': model_config.get('architecture'),
        'treatment_group': experiment_params.get('treatment'),
        'environment': experiment_params.get('environment'),
        'hyperparameter_set': experiment_params.get('hyperparameter_set'),
        'data_split': experiment_params.get('data_split', 'train_val'),
        'experiment_id': experiment_params.get('experiment_id'),
        'final_performance': final_val_acc,
        'convergence_quality': convergence_rate,
        'overfitting_severity': overfitting_score
    }
    
    # Store metadata in figure for FigRegistry condition resolution
    fig.metadata = condition_metadata
    
    plt.tight_layout()
    
    logger.info(f"Generated training visualization with {len(epochs)} epochs, "
               f"final validation accuracy: {final_val_acc:.3f}")
    
    # Return figure for automatic FigureDataSet processing
    # No manual styling or saving - handled by figregistry-kedro integration
    return fig


def create_inference_results_analysis(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    model_metadata: Dict[str, Any],
    experiment_params: Dict[str, Any]
) -> Figure:
    """Generate comprehensive inference results visualization with statistical analysis.
    
    Demonstrates sophisticated inference pipeline visualization patterns with complex
    error analysis, confidence assessment, and production-ready performance evaluation.
    Supports multi-class classification and regression scenarios with automatic
    experimental condition resolution for advanced styling.
    
    Args:
        predictions_df: DataFrame with columns ['prediction', 'confidence', 'sample_id']
        ground_truth_df: DataFrame with columns ['true_label', 'sample_id']
        model_metadata: Dictionary with model information and performance metrics
        experiment_params: Pipeline parameters for sophisticated condition resolution
        
    Returns:
        matplotlib.figure.Figure: Advanced inference analysis visualization with
            confusion matrix, error distribution analysis, confidence calibration,
            and statistical performance metrics. Automatically styled through
            FigureDataSet based on experimental conditions per F-005-RQ-004.
    
    Note:
        Demonstrates zero-touch figure management with sophisticated error analysis
        patterns suitable for production model evaluation workflows. The figure
        output includes advanced statistical overlays and enterprise-grade annotations
        that are automatically styled based on complex experimental conditions.
    """
    logger.info(
        f"Generating inference analysis for model: "
        f"{model_metadata.get('model_name', 'unknown')}, "
        f"samples: {len(predictions_df)}, "
        f"treatment: {experiment_params.get('treatment', 'baseline')}"
    )
    
    # Merge predictions with ground truth for comprehensive analysis
    analysis_df = pd.merge(predictions_df, ground_truth_df, on='sample_id', how='inner')
    
    # Detect task type for appropriate visualization
    task_type = detect_task_type(analysis_df['true_label'])
    
    # Create sophisticated multi-panel analysis layout
    fig = plt.figure(figsize=(20, 16))
    
    if task_type == 'classification':
        # Classification-specific visualization layout
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        # Confusion matrix with advanced annotations
        ax_conf = fig.add_subplot(gs[0, 0])
        
        unique_labels = sorted(analysis_df['true_label'].unique())
        conf_matrix = confusion_matrix(analysis_df['true_label'], analysis_df['prediction'])
        
        # Create heatmap with sophisticated formatting
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_labels, yticklabels=unique_labels, ax=ax_conf)
        ax_conf.set_title('Confusion Matrix Analysis')
        ax_conf.set_xlabel('Predicted Labels')
        ax_conf.set_ylabel('True Labels')
        
        # Per-class performance analysis
        ax_class = fig.add_subplot(gs[0, 1])
        
        class_report = classification_report(analysis_df['true_label'], 
                                           analysis_df['prediction'], 
                                           output_dict=True)
        
        # Extract per-class metrics for visualization
        classes = [str(c) for c in unique_labels]
        precision_scores = [class_report[c]['precision'] for c in classes]
        recall_scores = [class_report[c]['recall'] for c in classes]
        f1_scores = [class_report[c]['f1-score'] for c in classes]
        
        x_pos = np.arange(len(classes))
        width = 0.25
        
        ax_class.bar(x_pos - width, precision_scores, width, label='Precision', alpha=0.8)
        ax_class.bar(x_pos, recall_scores, width, label='Recall', alpha=0.8)
        ax_class.bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax_class.set_xlabel('Classes')
        ax_class.set_ylabel('Score')
        ax_class.set_title('Per-Class Performance Metrics')
        ax_class.set_xticks(x_pos)
        ax_class.set_xticklabels(classes)
        ax_class.legend()
        ax_class.grid(True, alpha=0.3)
        
    else:  # Regression task
        # Regression-specific visualization layout
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        # Prediction vs actual scatter plot with statistical overlays
        ax_scatter = fig.add_subplot(gs[0, 0])
        
        actual = analysis_df['true_label'].astype(float)
        predicted = analysis_df['prediction'].astype(float)
        
        # Create sophisticated scatter plot with regression line
        ax_scatter.scatter(actual, predicted, alpha=0.6, s=30)
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', 
                       label='Perfect Prediction', linewidth=2)
        
        # Add regression line with confidence interval
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        ax_scatter.plot(actual.sort_values(), p(actual.sort_values()), 
                       'g-', label=f'Regression Line (R²={stats.pearsonr(actual, predicted)[0]**2:.3f})',
                       linewidth=2)
        
        ax_scatter.set_xlabel('Actual Values')
        ax_scatter.set_ylabel('Predicted Values')
        ax_scatter.set_title('Prediction Accuracy Analysis')
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        
        # Residual analysis subplot
        ax_residual = fig.add_subplot(gs[0, 1])
        
        residuals = predicted - actual
        ax_residual.scatter(predicted, residuals, alpha=0.6, s=30)
        ax_residual.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax_residual.set_xlabel('Predicted Values')
        ax_residual.set_ylabel('Residuals')
        ax_residual.set_title('Residual Analysis')
        ax_residual.grid(True, alpha=0.3)
    
    # Confidence calibration analysis (applies to both classification and regression)
    ax_conf_cal = fig.add_subplot(gs[0, 2])
    
    if 'confidence' in analysis_df.columns:
        # Create confidence vs accuracy calibration plot
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (analysis_df['confidence'] >= confidence_bins[i]) & \
                   (analysis_df['confidence'] < confidence_bins[i + 1])
            
            if mask.sum() > 0:
                bin_data = analysis_df[mask]
                if task_type == 'classification':
                    accuracy = (bin_data['prediction'] == bin_data['true_label']).mean()
                else:
                    # For regression, use correlation as proxy for accuracy
                    accuracy = abs(stats.pearsonr(bin_data['prediction'], 
                                                bin_data['true_label'])[0])
                
                bin_accuracies.append(accuracy)
                bin_confidences.append(bin_data['confidence'].mean())
                bin_counts.append(len(bin_data))
        
        if bin_accuracies:
            ax_conf_cal.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, 
                           markersize=8, label='Model Calibration')
            ax_conf_cal.plot([0, 1], [0, 1], 'r--', linewidth=2, 
                           label='Perfect Calibration')
            
            ax_conf_cal.set_xlabel('Mean Predicted Confidence')
            ax_conf_cal.set_ylabel('Accuracy')
            ax_conf_cal.set_title('Confidence Calibration')
            ax_conf_cal.legend()
            ax_conf_cal.grid(True, alpha=0.3)
    
    # Error distribution analysis
    ax_error = fig.add_subplot(gs[1, 0])
    
    if task_type == 'classification':
        # Classification error analysis by class
        error_by_class = {}
        for true_class in unique_labels:
            class_mask = analysis_df['true_label'] == true_class
            class_errors = (analysis_df[class_mask]['prediction'] != true_class).sum()
            error_by_class[str(true_class)] = class_errors
        
        classes = list(error_by_class.keys())
        errors = list(error_by_class.values())
        
        ax_error.bar(classes, errors, alpha=0.7)
        ax_error.set_xlabel('True Class')
        ax_error.set_ylabel('Number of Errors')
        ax_error.set_title('Error Distribution by Class')
        
    else:
        # Regression error distribution
        errors = predicted - actual
        ax_error.hist(errors, bins=30, alpha=0.7, density=True)
        
        # Add normal distribution overlay for comparison
        mu, sigma = stats.norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        ax_error.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', 
                     label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
        
        ax_error.set_xlabel('Prediction Error')
        ax_error.set_ylabel('Density')
        ax_error.set_title('Error Distribution Analysis')
        ax_error.legend()
    
    ax_error.grid(True, alpha=0.3)
    
    # Model performance over time (if timestamp available)
    ax_time = fig.add_subplot(gs[1, 1])
    
    if 'timestamp' in analysis_df.columns:
        # Convert timestamp and analyze performance trends
        analysis_df['timestamp'] = pd.to_datetime(analysis_df['timestamp'])
        
        # Group by time windows and calculate accuracy
        time_grouped = analysis_df.set_index('timestamp').resample('1H')
        
        if task_type == 'classification':
            time_accuracy = time_grouped.apply(
                lambda x: (x['prediction'] == x['true_label']).mean() if len(x) > 0 else np.nan
            )
        else:
            time_accuracy = time_grouped.apply(
                lambda x: -np.sqrt(((x['prediction'] - x['true_label']) ** 2).mean()) if len(x) > 0 else np.nan
            )
        
        time_accuracy = time_accuracy.dropna()
        
        if len(time_accuracy) > 1:
            ax_time.plot(time_accuracy.index, time_accuracy.values, 
                        linewidth=2, marker='o', markersize=4)
            ax_time.set_xlabel('Time')
            ax_time.set_ylabel('Performance Metric')
            ax_time.set_title('Performance Over Time')
            ax_time.tick_params(axis='x', rotation=45)
            ax_time.grid(True, alpha=0.3)
    
    # Advanced statistical analysis panel
    ax_advanced = fig.add_subplot(gs[1, 2])
    
    # Perform comprehensive statistical analysis
    if task_type == 'classification':
        overall_accuracy = (analysis_df['prediction'] == analysis_df['true_label']).mean()
        statistical_summary = {
            'Overall Accuracy': f'{overall_accuracy:.3f}',
            'Macro F1': f'{class_report["macro avg"]["f1-score"]:.3f}',
            'Weighted F1': f'{class_report["weighted avg"]["f1-score"]:.3f}',
            'Sample Count': f'{len(analysis_df):,}',
            'Class Balance': f'{analysis_df["true_label"].value_counts().std():.1f}'
        }
    else:
        mae = np.mean(np.abs(predicted - actual))
        mse = np.mean((predicted - actual) ** 2)
        rmse = np.sqrt(mse)
        r2 = stats.pearsonr(actual, predicted)[0] ** 2
        
        statistical_summary = {
            'R² Score': f'{r2:.3f}',
            'RMSE': f'{rmse:.3f}',
            'MAE': f'{mae:.3f}',
            'MSE': f'{mse:.3f}',
            'Sample Count': f'{len(analysis_df):,}'
        }
    
    # Create text summary with advanced formatting
    summary_text = '\n'.join([f'{k}: {v}' for k, v in statistical_summary.items()])
    ax_advanced.text(0.1, 0.9, summary_text, transform=ax_advanced.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax_advanced.set_title('Statistical Summary')
    ax_advanced.axis('off')
    
    # Experimental metadata and condition information
    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.axis('off')
    
    # Prepare comprehensive experimental metadata for condition resolution
    condition_metadata = {
        'model_name': model_metadata.get('model_name'),
        'model_version': model_metadata.get('version'),
        'treatment_group': experiment_params.get('treatment'),
        'environment': experiment_params.get('environment'),
        'data_split': experiment_params.get('data_split', 'test'),
        'experiment_id': experiment_params.get('experiment_id'),
        'task_type': task_type,
        'sample_count': len(analysis_df),
        'performance_score': overall_accuracy if task_type == 'classification' else r2,
        'confidence_available': 'confidence' in analysis_df.columns
    }
    
    # Format metadata display
    meta_text = f"""
    Experimental Configuration: Model={condition_metadata['model_name']} v{condition_metadata.get('model_version', 'N/A')} | 
    Treatment={condition_metadata['treatment_group']} | Environment={condition_metadata['environment']} | Task={condition_metadata['task_type']}
    
    Dataset Information: Samples={condition_metadata['sample_count']:,} | Split={condition_metadata['data_split']} | 
    Experiment ID={condition_metadata.get('experiment_id', 'N/A')}
    
    Performance Summary: Score={condition_metadata['performance_score']:.3f} | 
    Confidence Data={'Available' if condition_metadata['confidence_available'] else 'Not Available'}
    """
    
    ax_meta.text(0.02, 0.8, meta_text.strip(), transform=ax_meta.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Store metadata for FigRegistry condition resolution
    fig.metadata = condition_metadata
    
    plt.tight_layout()
    
    logger.info(f"Generated inference analysis: task_type={task_type}, "
               f"performance={condition_metadata['performance_score']:.3f}, "
               f"samples={len(analysis_df)}")
    
    # Return figure for automatic FigureDataSet processing
    return fig


def create_ab_test_comparison_report(
    control_results: pd.DataFrame,
    treatment_results: pd.DataFrame,
    experiment_config: Dict[str, Any],
    experiment_params: Dict[str, Any]
) -> Figure:
    """Generate comprehensive A/B test comparison with advanced statistical analysis.
    
    Demonstrates sophisticated experimental design visualization patterns with complex
    statistical testing, effect size calculations, and production-ready reporting
    suitable for enterprise decision-making workflows. Shows advanced condition-based
    styling through multi-variable experimental condition resolution.
    
    Args:
        control_results: DataFrame with control group metrics and outcomes
        treatment_results: DataFrame with treatment group metrics and outcomes  
        experiment_config: Dictionary with A/B test configuration and parameters
        experiment_params: Pipeline parameters for advanced condition resolution
        
    Returns:
        matplotlib.figure.Figure: Enterprise-grade A/B test report with comprehensive
            statistical analysis, effect size calculations, confidence intervals,
            and business impact assessment. Automatically styled through FigureDataSet
            based on sophisticated experimental conditions per F-002 requirements.
    
    Note:
        Showcases enterprise-grade statistical analysis patterns with complex
        experimental condition resolution suitable for production A/B testing
        workflows. Demonstrates zero-touch figure management with advanced
        statistical overlays and business intelligence formatting.
    """
    logger.info(
        f"Generating A/B test comparison: "
        f"experiment={experiment_config.get('experiment_name', 'unknown')}, "
        f"control_n={len(control_results)}, treatment_n={len(treatment_results)}, "
        f"environment={experiment_params.get('environment', 'production')}"
    )
    
    # Create sophisticated A/B test analysis layout  
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.35, wspace=0.3)
    
    # Primary metric comparison with statistical significance
    ax_primary = fig.add_subplot(gs[0, :2])
    
    primary_metric = experiment_config.get('primary_metric', 'conversion_rate')
    
    if primary_metric in control_results.columns and primary_metric in treatment_results.columns:
        control_values = control_results[primary_metric].dropna()
        treatment_values = treatment_results[primary_metric].dropna()
        
        # Create sophisticated comparison visualization
        positions = [1, 2]
        box_data = [control_values, treatment_values]
        labels = ['Control', 'Treatment']
        
        # Box plots with advanced statistical overlays
        bp = ax_primary.boxplot(box_data, positions=positions, patch_artist=True,
                               labels=labels, widths=0.6)
        
        # Customize box plot appearance (styling handled by FigRegistry)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual data points with jitter
        for i, data in enumerate(box_data):
            x = np.random.normal(positions[i], 0.02, size=len(data))
            ax_primary.scatter(x, data, alpha=0.4, s=20)
        
        # Calculate and display advanced statistics
        stat_helper = StatisticalAnalysisHelper()
        
        # Perform comprehensive statistical comparison
        comparison_results = stat_helper.perform_comprehensive_comparison(
            data=pd.concat([
                control_results.assign(group='control'),
                treatment_results.assign(group='treatment')
            ]),
            value_column=primary_metric,
            group_column='group'
        )
        
        # Extract key statistics for annotation
        t_test_p = comparison_results['pairwise_comparisons']['control_vs_treatment']['t_test']['p_value']
        mw_test_p = comparison_results['pairwise_comparisons']['control_vs_treatment']['mann_whitney']['p_value']
        
        effect_sizes = comparison_results['effect_sizes']['control_vs_treatment']
        cohens_d = effect_sizes.get('cohens_d', 0)
        
        # Add statistical annotations
        control_mean = control_values.mean()
        treatment_mean = treatment_values.mean()
        relative_lift = (treatment_mean - control_mean) / control_mean * 100
        
        # Statistical significance indicators
        if t_test_p < 0.001:
            sig_text = "***"
        elif t_test_p < 0.01:
            sig_text = "**"
        elif t_test_p < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        
        ax_primary.text(1.5, max(control_values.max(), treatment_values.max()) * 1.1,
                       f'Relative Lift: {relative_lift:+.2f}% {sig_text}\n'
                       f"Cohen's d: {cohens_d:.3f}\n"
                       f'p-value: {t_test_p:.4f}',
                       ha='center', va='bottom', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        ax_primary.set_ylabel(primary_metric.replace('_', ' ').title())
        ax_primary.set_title(f'Primary Metric Comparison: {primary_metric.replace("_", " ").title()}')
        ax_primary.grid(True, alpha=0.3, axis='y')
    
    # Statistical power and sample size analysis
    ax_power = fig.add_subplot(gs[0, 2])
    
    # Calculate statistical power for different effect sizes
    effect_sizes_range = np.linspace(0.01, 0.5, 50)
    power_values = []
    
    for effect_size in effect_sizes_range:
        # Simplified power calculation (normally would use more sophisticated methods)
        sample_size = min(len(control_values), len(treatment_values))
        z_alpha = stats.norm.ppf(0.975)  # 95% confidence
        z_beta = stats.norm.ppf(0.8)     # 80% power
        
        # Effect size to power approximation
        power = stats.norm.cdf(effect_size * np.sqrt(sample_size/2) - z_alpha)
        power_values.append(max(0, min(1, power)))
    
    ax_power.plot(effect_sizes_range, power_values, linewidth=2, label='Statistical Power')
    ax_power.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power Threshold')
    ax_power.axvline(x=abs(cohens_d), color='green', linestyle=':', alpha=0.7, 
                    label=f'Observed Effect Size ({abs(cohens_d):.3f})')
    
    ax_power.set_xlabel('Effect Size')
    ax_power.set_ylabel('Statistical Power')
    ax_power.set_title('Power Analysis')
    ax_power.legend()
    ax_power.grid(True, alpha=0.3)
    
    # Secondary metrics comparison dashboard
    ax_secondary = fig.add_subplot(gs[1, :2])
    
    secondary_metrics = experiment_config.get('secondary_metrics', 
                                            ['click_through_rate', 'bounce_rate', 'session_duration'])
    
    # Create multi-metric comparison
    metric_comparisons = []
    for metric in secondary_metrics:
        if metric in control_results.columns and metric in treatment_results.columns:
            control_vals = control_results[metric].dropna()
            treatment_vals = treatment_results[metric].dropna()
            
            if len(control_vals) > 0 and len(treatment_vals) > 0:
                control_mean = control_vals.mean()
                treatment_mean = treatment_vals.mean()
                
                # Statistical test
                _, p_value = stats.ttest_ind(control_vals, treatment_vals)
                
                # Effect size
                effect_sizes_secondary = stat_helper.calculate_effect_sizes(
                    control_vals.values, treatment_vals.values
                )
                
                metric_comparisons.append({
                    'metric': metric,
                    'control_mean': control_mean,
                    'treatment_mean': treatment_mean,
                    'relative_change': (treatment_mean - control_mean) / control_mean * 100,
                    'p_value': p_value,
                    'cohens_d': effect_sizes_secondary.get('cohens_d', 0),
                    'significant': p_value < 0.05
                })
    
    if metric_comparisons:
        metrics_df = pd.DataFrame(metric_comparisons)
        
        # Create horizontal bar chart for relative changes
        y_pos = np.arange(len(metrics_df))
        colors = ['green' if x > 0 else 'red' for x in metrics_df['relative_change']]
        significance_alpha = [0.8 if sig else 0.4 for sig in metrics_df['significant']]
        
        bars = ax_secondary.barh(y_pos, metrics_df['relative_change'], 
                               color=colors, alpha=significance_alpha)
        
        ax_secondary.set_yticks(y_pos)
        ax_secondary.set_yticklabels([m.replace('_', ' ').title() for m in metrics_df['metric']])
        ax_secondary.set_xlabel('Relative Change (%)')
        ax_secondary.set_title('Secondary Metrics Impact Analysis')
        ax_secondary.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax_secondary.grid(True, alpha=0.3, axis='x')
        
        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, metrics_df['p_value'])):
            if p_val < 0.05:
                ax_secondary.text(bar.get_width() + 0.5 if bar.get_width() > 0 else bar.get_width() - 0.5,
                                bar.get_y() + bar.get_height()/2, 
                                f'p={p_val:.3f}', ha='left' if bar.get_width() > 0 else 'right',
                                va='center', fontsize=9)
    
    # Time series analysis (if timestamp data available)
    ax_time = fig.add_subplot(gs[1, 2])
    
    if 'timestamp' in control_results.columns and 'timestamp' in treatment_results.columns:
        # Prepare time series data
        control_ts = control_results.copy()
        control_ts['timestamp'] = pd.to_datetime(control_ts['timestamp'])
        control_ts = control_ts.set_index('timestamp')
        
        treatment_ts = treatment_results.copy()
        treatment_ts['timestamp'] = pd.to_datetime(treatment_ts['timestamp'])
        treatment_ts = treatment_ts.set_index('timestamp')
        
        # Resample to daily averages for primary metric
        control_daily = control_ts[primary_metric].resample('D').mean()
        treatment_daily = treatment_ts[primary_metric].resample('D').mean()
        
        # Plot time series with confidence intervals
        ax_time.plot(control_daily.index, control_daily.values, 
                    label='Control', linewidth=2, alpha=0.8)
        ax_time.plot(treatment_daily.index, treatment_daily.values,
                    label='Treatment', linewidth=2, alpha=0.8)
        
        # Add rolling averages for trend analysis
        control_ma = control_daily.rolling(window=3, min_periods=1).mean()
        treatment_ma = treatment_daily.rolling(window=3, min_periods=1).mean()
        
        ax_time.plot(control_ma.index, control_ma.values, '--', alpha=0.6, 
                    label='Control Trend')
        ax_time.plot(treatment_ma.index, treatment_ma.values, '--', alpha=0.6,
                    label='Treatment Trend')
        
        ax_time.set_xlabel('Date')
        ax_time.set_ylabel(primary_metric.replace('_', ' ').title())
        ax_time.set_title('Temporal Performance Analysis')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        ax_time.tick_params(axis='x', rotation=45)
    
    # Business impact and recommendation summary
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    # Calculate business impact metrics
    business_impact = calculate_business_impact(
        control_results, treatment_results, experiment_config
    )
    
    # Determine recommendation based on statistical and business criteria
    recommendation = generate_ab_test_recommendation(
        statistical_results=comparison_results,
        business_impact=business_impact,
        experiment_config=experiment_config
    )
    
    # Prepare comprehensive experimental metadata
    condition_metadata = {
        'experiment_name': experiment_config.get('experiment_name'),
        'experiment_type': 'ab_test',
        'treatment_group': experiment_params.get('treatment', 'variant_a'),
        'environment': experiment_params.get('environment'),
        'primary_metric': primary_metric,
        'control_sample_size': len(control_results),
        'treatment_sample_size': len(treatment_results),
        'statistical_significance': t_test_p < 0.05,
        'effect_size': cohens_d,
        'relative_lift': relative_lift,
        'business_impact_score': business_impact.get('impact_score', 0),
        'recommendation': recommendation['action'],
        'confidence_level': recommendation['confidence']
    }
    
    # Format comprehensive summary
    summary_text = f"""
    A/B Test Results Summary: {condition_metadata['experiment_name']} | Environment: {condition_metadata['environment']} | Treatment: {condition_metadata['treatment_group']}
    
    Statistical Analysis: Primary Metric={primary_metric} | Relative Lift={relative_lift:+.2f}% | Effect Size (Cohen's d)={cohens_d:.3f} | p-value={t_test_p:.4f} | Significant={'Yes' if condition_metadata['statistical_significance'] else 'No'}
    
    Sample Sizes: Control={condition_metadata['control_sample_size']:,} | Treatment={condition_metadata['treatment_sample_size']:,} | Total={condition_metadata['control_sample_size'] + condition_metadata['treatment_sample_size']:,}
    
    Business Impact: Impact Score={business_impact.get('impact_score', 0):.2f} | Revenue Impact={business_impact.get('revenue_impact', 'N/A')} | User Impact={business_impact.get('user_impact', 'N/A')}
    
    Recommendation: {recommendation['action']} (Confidence: {recommendation['confidence']}) | Rationale: {recommendation.get('rationale', 'Based on statistical and business analysis')}
    """
    
    ax_summary.text(0.02, 0.9, summary_text.strip(), transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
    
    # Store metadata for FigRegistry condition resolution
    fig.metadata = condition_metadata
    
    plt.tight_layout()
    
    logger.info(f"Generated A/B test report: experiment={condition_metadata['experiment_name']}, "
               f"lift={relative_lift:+.2f}%, significant={condition_metadata['statistical_significance']}")
    
    # Return figure for automatic FigureDataSet processing
    return fig


def create_model_performance_comparison(
    model_results: List[Dict[str, Any]], 
    benchmark_data: pd.DataFrame,
    comparison_config: Dict[str, Any],
    experiment_params: Dict[str, Any]
) -> Figure:
    """Generate comprehensive model performance comparison across multiple algorithms.
    
    Demonstrates sophisticated model evaluation visualization patterns with statistical
    significance testing, performance profiling, and enterprise-grade algorithm
    comparison suitable for production model selection workflows. Shows complex
    experimental condition resolution with multi-algorithm analysis.
    
    Args:
        model_results: List of dictionaries containing model performance metrics
        benchmark_data: DataFrame with baseline/benchmark performance data
        comparison_config: Dictionary with comparison configuration and evaluation criteria
        experiment_params: Pipeline parameters for sophisticated condition resolution
        
    Returns:
        matplotlib.figure.Figure: Enterprise-grade model comparison report with
            statistical analysis, performance profiling, algorithm ranking, and
            production readiness assessment. Automatically styled through FigureDataSet
            based on complex experimental conditions per F-002-RQ-002.
    
    Note:
        Showcases production-ready model evaluation patterns with sophisticated
        statistical analysis and enterprise-grade performance profiling suitable
        for algorithmic decision-making in production environments.
    """
    logger.info(
        f"Generating model comparison: "
        f"models={len(model_results)}, "
        f"evaluation_strategy={comparison_config.get('evaluation_strategy', 'cross_validation')}, "
        f"environment={experiment_params.get('environment', 'production')}"
    )
    
    # Prepare model comparison data
    helper = DataTransformationHelper()
    models_df = helper.prepare_performance_comparison(
        results_data=model_results,
        metric_columns=comparison_config.get('metrics', ['accuracy', 'precision', 'recall', 'f1_score']),
        group_by='model_type'
    )
    
    # Create sophisticated model comparison layout
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, height_ratios=[2, 2, 1.5, 1], hspace=0.35, wspace=0.3)
    
    # Primary performance metrics comparison
    ax_primary = fig.add_subplot(gs[0, :3])
    
    primary_metric = comparison_config.get('primary_metric', 'accuracy')
    
    # Create comprehensive performance comparison
    model_types = models_df['model_type'].unique()
    metrics = models_df['metric'].unique()
    
    # Multi-metric comparison with error bars
    metric_positions = {metric: i for i, metric in enumerate(metrics)}
    model_positions = {model: i for i, model in enumerate(model_types)}
    
    # Create grouped bar chart with confidence intervals
    bar_width = 0.15
    x_positions = np.arange(len(model_types))
    
    for i, metric in enumerate(metrics):
        metric_data = models_df[models_df['metric'] == metric]
        
        means = []
        errors = []
        
        for model_type in model_types:
            model_metric_data = metric_data[metric_data['model_type'] == model_type]
            if len(model_metric_data) > 0:
                mean_val = model_metric_data['mean'].iloc[0]
                ci_error = (model_metric_data['ci_upper'].iloc[0] - model_metric_data['ci_lower'].iloc[0]) / 2
                means.append(mean_val)
                errors.append(ci_error)
            else:
                means.append(0)
                errors.append(0)
        
        ax_primary.bar(x_positions + i * bar_width, means, bar_width, 
                      yerr=errors, capsize=5, alpha=0.8, 
                      label=metric.replace('_', ' ').title())
    
    ax_primary.set_xlabel('Model Types')
    ax_primary.set_ylabel('Performance Score')
    ax_primary.set_title('Model Performance Comparison Across Metrics')
    ax_primary.set_xticks(x_positions + bar_width * (len(metrics) - 1) / 2)
    ax_primary.set_xticklabels([m.replace('_', ' ').title() for m in model_types], rotation=45)
    ax_primary.legend()
    ax_primary.grid(True, alpha=0.3, axis='y')
    
    # Statistical significance heatmap
    ax_heatmap = fig.add_subplot(gs[0, 3])
    
    # Calculate pairwise statistical significance for primary metric
    primary_data = models_df[models_df['metric'] == primary_metric]
    
    if len(primary_data) > 1:
        # Create significance matrix
        significance_matrix = np.zeros((len(model_types), len(model_types)))
        
        for i, model1 in enumerate(model_types):
            for j, model2 in enumerate(model_types):
                if i != j:
                    # Simulate significance test (in practice, would use actual test data)
                    model1_data = primary_data[primary_data['model_type'] == model1]
                    model2_data = primary_data[primary_data['model_type'] == model2]
                    
                    if len(model1_data) > 0 and len(model2_data) > 0:
                        # Simplified significance calculation
                        mean_diff = abs(model1_data['mean'].iloc[0] - model2_data['mean'].iloc[0])
                        combined_error = np.sqrt(model1_data['std'].iloc[0]**2 + model2_data['std'].iloc[0]**2)
                        
                        if combined_error > 0:
                            z_score = mean_diff / combined_error
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                            significance_matrix[i, j] = p_value
        
        # Create heatmap
        sns.heatmap(significance_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=[m[:10] for m in model_types], 
                   yticklabels=[m[:10] for m in model_types],
                   ax=ax_heatmap, cbar_kws={'label': 'p-value'})
        ax_heatmap.set_title('Statistical Significance\n(p-values)')
    
    # Performance profiling across different evaluation criteria
    ax_profile = fig.add_subplot(gs[1, :2])
    
    # Create radar chart for multi-dimensional performance profiling
    evaluation_criteria = comparison_config.get('evaluation_criteria', 
                                               ['accuracy', 'speed', 'interpretability', 'robustness'])
    
    # Calculate normalized scores for each model across criteria
    model_profiles = {}
    
    for model_type in model_types:
        profile_scores = []
        
        for criterion in evaluation_criteria:
            if criterion in [m.lower() for m in metrics]:
                # Use actual metric data
                criterion_data = models_df[
                    (models_df['model_type'] == model_type) & 
                    (models_df['metric'].str.lower() == criterion.lower())
                ]
                if len(criterion_data) > 0:
                    score = criterion_data['mean'].iloc[0]
                else:
                    score = 0.5  # Default neutral score
            else:
                # Use simulated scores for other criteria
                score = np.random.uniform(0.3, 0.9)  # Simulated for demo
            
            profile_scores.append(score)
        
        model_profiles[model_type] = profile_scores
    
    # Create spider/radar plot
    angles = np.linspace(0, 2 * np.pi, len(evaluation_criteria), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    ax_profile = plt.subplot(gs[1, 0], projection='polar')
    
    for model_type, scores in model_profiles.items():
        scores_circle = scores + [scores[0]]  # Complete the circle
        ax_profile.plot(angles, scores_circle, 'o-', linewidth=2, 
                       label=model_type.replace('_', ' ').title())
        ax_profile.fill(angles, scores_circle, alpha=0.25)
    
    ax_profile.set_xticks(angles[:-1])
    ax_profile.set_xticklabels([c.replace('_', ' ').title() for c in evaluation_criteria])
    ax_profile.set_ylim(0, 1)
    ax_profile.set_title('Model Performance Profile', y=1.08)
    ax_profile.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Training efficiency and resource utilization analysis
    ax_efficiency = fig.add_subplot(gs[1, 2])
    
    # Extract efficiency metrics from model results
    efficiency_data = []
    for model_result in model_results:
        if 'training_time' in model_result and 'model_size' in model_result:
            efficiency_data.append({
                'model_type': model_result['model_type'],
                'training_time': model_result['training_time'],
                'model_size': model_result['model_size'],
                'inference_time': model_result.get('inference_time', 0),
                'accuracy': model_result.get('accuracy', 0)
            })
    
    if efficiency_data:
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Create efficiency scatter plot (accuracy vs training time)
        scatter = ax_efficiency.scatter(efficiency_df['training_time'], 
                                      efficiency_df['accuracy'],
                                      s=efficiency_df['model_size'] * 100,  # Size represents model size
                                      alpha=0.7, c=range(len(efficiency_df)), 
                                      cmap='viridis')
        
        # Add model labels
        for i, row in efficiency_df.iterrows():
            ax_efficiency.annotate(row['model_type'][:8], 
                                 (row['training_time'], row['accuracy']),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=9, alpha=0.8)
        
        ax_efficiency.set_xlabel('Training Time (minutes)')
        ax_efficiency.set_ylabel('Accuracy')
        ax_efficiency.set_title('Training Efficiency Analysis\n(bubble size = model size)')
        ax_efficiency.grid(True, alpha=0.3)
    
    # Benchmark comparison and baseline analysis
    ax_benchmark = fig.add_subplot(gs[1, 3])
    
    if not benchmark_data.empty and primary_metric in benchmark_data.columns:
        # Compare models against benchmarks
        benchmark_scores = benchmark_data[primary_metric]
        
        # Create comparison with benchmark distribution
        ax_benchmark.hist(benchmark_scores, bins=20, alpha=0.5, density=True, 
                         label='Benchmark Distribution', color='gray')
        
        # Overlay model performance
        primary_scores = []
        for model_type in model_types:
            model_data = models_df[
                (models_df['model_type'] == model_type) & 
                (models_df['metric'] == primary_metric)
            ]
            if len(model_data) > 0:
                score = model_data['mean'].iloc[0]
                primary_scores.append(score)
                ax_benchmark.axvline(x=score, alpha=0.8, linewidth=2,
                                   label=f'{model_type[:10]}')
        
        ax_benchmark.set_xlabel(primary_metric.replace('_', ' ').title())
        ax_benchmark.set_ylabel('Density')
        ax_benchmark.set_title('Benchmark Comparison')
        ax_benchmark.legend()
        ax_benchmark.grid(True, alpha=0.3)
    
    # Model ranking and selection recommendation
    ax_ranking = fig.add_subplot(gs[2, :2])
    
    # Calculate composite ranking scores
    ranking_criteria = comparison_config.get('ranking_criteria', 
                                           {'accuracy': 0.4, 'speed': 0.3, 'interpretability': 0.3})
    
    model_rankings = []
    for model_type in model_types:
        composite_score = 0
        criteria_scores = {}
        
        for criterion, weight in ranking_criteria.items():
            if criterion in [m.lower() for m in metrics]:
                criterion_data = models_df[
                    (models_df['model_type'] == model_type) & 
                    (models_df['metric'].str.lower() == criterion.lower())
                ]
                if len(criterion_data) > 0:
                    score = criterion_data['mean'].iloc[0]
                else:
                    score = 0.5
            else:
                # Use efficiency metrics or defaults
                if criterion == 'speed' and efficiency_data:
                    speed_data = [d for d in efficiency_data if d['model_type'] == model_type]
                    if speed_data:
                        # Inverse of training time (normalized)
                        max_time = max([d['training_time'] for d in efficiency_data])
                        score = 1 - (speed_data[0]['training_time'] / max_time)
                    else:
                        score = 0.5
                else:
                    score = np.random.uniform(0.3, 0.9)  # Simulated
            
            criteria_scores[criterion] = score
            composite_score += score * weight
        
        model_rankings.append({
            'model_type': model_type,
            'composite_score': composite_score,
            'criteria_scores': criteria_scores
        })
    
    # Sort by composite score
    model_rankings.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Create ranking visualization
    ranking_df = pd.DataFrame(model_rankings)
    y_pos = np.arange(len(ranking_df))
    
    bars = ax_ranking.barh(y_pos, ranking_df['composite_score'], alpha=0.7)
    
    # Color bars based on score
    for bar, score in zip(bars, ranking_df['composite_score']):
        if score > 0.8:
            bar.set_color('green')
        elif score > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax_ranking.set_yticks(y_pos)
    ax_ranking.set_yticklabels([m['model_type'].replace('_', ' ').title() 
                               for m in model_rankings])
    ax_ranking.set_xlabel('Composite Ranking Score')
    ax_ranking.set_title('Model Ranking by Weighted Criteria')
    ax_ranking.grid(True, alpha=0.3, axis='x')
    
    # Add score annotations
    for i, (bar, score) in enumerate(zip(bars, ranking_df['composite_score'])):
        ax_ranking.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', ha='left', va='center', fontsize=10)
    
    # Production readiness assessment
    ax_readiness = fig.add_subplot(gs[2, 2:])
    
    # Assess production readiness based on multiple factors
    readiness_factors = ['performance', 'stability', 'scalability', 'interpretability', 'compliance']
    readiness_assessments = {}
    
    for model_type in model_types:
        model_assessment = {}
        
        # Performance readiness (based on benchmark comparison)
        model_perf = models_df[
            (models_df['model_type'] == model_type) & 
            (models_df['metric'] == primary_metric)
        ]
        if len(model_perf) > 0:
            perf_score = model_perf['mean'].iloc[0]
            model_assessment['performance'] = min(1.0, perf_score * 1.2)  # Scale for readiness
        else:
            model_assessment['performance'] = 0.5
        
        # Other factors (simulated for demo - would be based on actual assessments)
        model_assessment['stability'] = np.random.uniform(0.4, 0.9)
        model_assessment['scalability'] = np.random.uniform(0.3, 0.8)
        model_assessment['interpretability'] = np.random.uniform(0.2, 0.9)
        model_assessment['compliance'] = np.random.uniform(0.5, 0.95)
        
        readiness_assessments[model_type] = model_assessment
    
    # Create readiness heatmap
    readiness_matrix = np.array([
        [readiness_assessments[model][factor] for factor in readiness_factors]
        for model in model_types
    ])
    
    sns.heatmap(readiness_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
               xticklabels=[f.title() for f in readiness_factors],
               yticklabels=[m.replace('_', ' ').title() for m in model_types],
               ax=ax_readiness, cbar_kws={'label': 'Readiness Score'})
    ax_readiness.set_title('Production Readiness Assessment')
    
    # Experimental metadata and comprehensive summary
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    # Generate model selection recommendation
    best_model = model_rankings[0]
    recommendation = generate_model_selection_recommendation(
        model_rankings, comparison_config, experiment_params
    )
    
    # Prepare comprehensive experimental metadata
    condition_metadata = {
        'comparison_type': 'model_performance',
        'evaluation_strategy': comparison_config.get('evaluation_strategy'),
        'environment': experiment_params.get('environment'),
        'primary_metric': primary_metric,
        'model_count': len(model_types),
        'total_evaluations': len(model_results),
        'best_model': best_model['model_type'],
        'best_score': best_model['composite_score'],
        'evaluation_criteria': list(ranking_criteria.keys()),
        'treatment_group': experiment_params.get('treatment', 'model_comparison'),
        'data_split': experiment_params.get('data_split', 'validation'),
        'experiment_id': experiment_params.get('experiment_id'),
        'recommendation': recommendation['selected_model'],
        'confidence_level': recommendation['confidence']
    }
    
    # Format comprehensive summary
    models_summary = ' | '.join([f"{m['model_type']}({m['composite_score']:.3f})" 
                                for m in model_rankings[:3]])
    
    summary_text = f"""
    Model Performance Comparison Summary: Strategy={condition_metadata['evaluation_strategy']} | Environment={condition_metadata['environment']} | Primary Metric={primary_metric}
    
    Evaluation Results: Models Evaluated={condition_metadata['model_count']} | Total Runs={condition_metadata['total_evaluations']} | Top 3: {models_summary}
    
    Selection Criteria: {' | '.join([f'{k}({v})' for k, v in ranking_criteria.items()])} | Data Split={condition_metadata['data_split']}
    
    Recommendation: Selected Model={recommendation['selected_model']} | Confidence={recommendation['confidence']} | 
    Rationale={recommendation.get('rationale', 'Based on composite scoring and production readiness assessment')}
    
    Production Readiness: Best Model Score={best_model['composite_score']:.3f} | Deployment Ready={'Yes' if best_model['composite_score'] > 0.7 else 'Needs Review'}
    """
    
    ax_summary.text(0.02, 0.9, summary_text.strip(), transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
    
    # Store metadata for FigRegistry condition resolution
    fig.metadata = condition_metadata
    
    plt.tight_layout()
    
    logger.info(f"Generated model comparison: best_model={best_model['model_type']}, "
               f"score={best_model['composite_score']:.3f}, models={len(model_types)}")
    
    # Return figure for automatic FigureDataSet processing
    return fig


# Utility functions for advanced statistical analysis

def detect_training_plateau(values: np.ndarray, patience: int = 10, min_delta: float = 0.001) -> Optional[int]:
    """Detect plateau in training metrics for early stopping analysis."""
    if len(values) < patience:
        return None
    
    for i in range(patience, len(values)):
        window = values[i-patience:i]
        if np.all(np.abs(np.diff(window)) < min_delta):
            return i - patience
    
    return None


def calculate_convergence_rate(loss_values: np.ndarray) -> float:
    """Calculate convergence rate from loss curve."""
    if len(loss_values) < 2:
        return 0.0
    
    # Calculate rate of change in loss
    diff = np.diff(loss_values)
    
    # Return average rate of improvement (negative because loss should decrease)
    return -np.mean(diff)


def detect_task_type(labels) -> str:
    """Detect whether task is classification or regression based on labels."""
    try:
        # Try to convert to float
        float_labels = pd.to_numeric(labels, errors='coerce')
        
        # If many values can't be converted, likely classification
        if float_labels.isna().sum() > len(labels) * 0.1:
            return 'classification'
        
        # If continuous values, likely regression
        unique_ratio = len(float_labels.unique()) / len(float_labels)
        if unique_ratio > 0.1:  # More than 10% unique values
            return 'regression'
        else:
            return 'classification'
            
    except Exception:
        return 'classification'


def calculate_business_impact(control_results: pd.DataFrame, 
                            treatment_results: pd.DataFrame,
                            experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate business impact metrics for A/B test analysis."""
    primary_metric = experiment_config.get('primary_metric', 'conversion_rate')
    
    if primary_metric not in control_results.columns or primary_metric not in treatment_results.columns:
        return {'impact_score': 0, 'revenue_impact': 'N/A', 'user_impact': 'N/A'}
    
    control_mean = control_results[primary_metric].mean()
    treatment_mean = treatment_results[primary_metric].mean()
    
    relative_lift = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
    
    # Simulate business impact calculation
    revenue_per_conversion = experiment_config.get('revenue_per_conversion', 100)
    total_users = len(control_results) + len(treatment_results)
    
    revenue_impact = relative_lift * control_mean * total_users * revenue_per_conversion
    user_impact = relative_lift * control_mean * total_users
    
    # Calculate impact score (0-1 scale)
    impact_score = min(1.0, abs(relative_lift) * 10)  # Scale relative lift
    
    return {
        'impact_score': impact_score,
        'revenue_impact': f'${revenue_impact:,.0f}' if abs(revenue_impact) > 0 else 'Minimal',
        'user_impact': f'{user_impact:,.0f} users' if abs(user_impact) > 0 else 'Minimal'
    }


def generate_ab_test_recommendation(statistical_results: Dict[str, Any],
                                  business_impact: Dict[str, Any],
                                  experiment_config: Dict[str, Any]) -> Dict[str, str]:
    """Generate recommendation based on A/B test results."""
    # Extract key metrics
    significance_threshold = experiment_config.get('significance_threshold', 0.05)
    
    try:
        pairwise_results = statistical_results['pairwise_comparisons']['control_vs_treatment']
        p_value = pairwise_results['t_test']['p_value']
        effect_size = statistical_results['effect_sizes']['control_vs_treatment']['cohens_d']
        
        is_significant = p_value < significance_threshold
        has_meaningful_effect = abs(effect_size) > 0.2  # Small to medium effect size
        has_business_impact = business_impact['impact_score'] > 0.3
        
        if is_significant and has_meaningful_effect and has_business_impact:
            if effect_size > 0:
                return {
                    'action': 'Deploy Treatment',
                    'confidence': 'High',
                    'rationale': 'Statistically significant improvement with meaningful business impact'
                }
            else:
                return {
                    'action': 'Keep Control',
                    'confidence': 'High', 
                    'rationale': 'Treatment shows significant negative impact'
                }
        elif is_significant and has_meaningful_effect:
            return {
                'action': 'Deploy with Monitoring',
                'confidence': 'Medium',
                'rationale': 'Statistically significant but unclear business impact'
            }
        elif has_business_impact:
            return {
                'action': 'Extend Test Duration',
                'confidence': 'Medium',
                'rationale': 'Promising business impact but needs statistical confirmation'
            }
        else:
            return {
                'action': 'No Change',
                'confidence': 'High',
                'rationale': 'No significant improvement detected'
            }
            
    except Exception as e:
        logger.warning(f"Error generating A/B test recommendation: {e}")
        return {
            'action': 'Inconclusive',
            'confidence': 'Low',
            'rationale': 'Insufficient data for recommendation'
        }


def generate_model_selection_recommendation(model_rankings: List[Dict[str, Any]],
                                          comparison_config: Dict[str, Any],
                                          experiment_params: Dict[str, Any]) -> Dict[str, str]:
    """Generate model selection recommendation based on ranking and criteria."""
    if not model_rankings:
        return {
            'selected_model': 'None',
            'confidence': 'Low',
            'rationale': 'No models evaluated'
        }
    
    best_model = model_rankings[0]
    second_best = model_rankings[1] if len(model_rankings) > 1 else None
    
    # Determine confidence based on score gap
    score_gap = 0
    if second_best:
        score_gap = best_model['composite_score'] - second_best['composite_score']
    
    if best_model['composite_score'] > 0.8 and score_gap > 0.1:
        confidence = 'High'
        rationale = f'Clear winner with composite score {best_model["composite_score"]:.3f}'
    elif best_model['composite_score'] > 0.6:
        confidence = 'Medium'
        rationale = f'Good performance but consider validation in {experiment_params.get("environment", "production")}'
    else:
        confidence = 'Low'
        rationale = 'All models show suboptimal performance, consider additional algorithms'
    
    return {
        'selected_model': best_model['model_type'],
        'confidence': confidence,
        'rationale': rationale
    }