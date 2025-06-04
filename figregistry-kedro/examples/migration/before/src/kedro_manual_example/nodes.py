"""Traditional Kedro node functions demonstrating manual matplotlib figure management.

This module showcases the problematic approaches that figregistry-kedro eliminates:
- Manual plt.savefig() calls scattered throughout node functions
- Hardcoded styling configuration in each function
- Inconsistent file naming and path management
- Code duplication across visualization functions
- Manual experimental condition handling
- Maintenance overhead from scattered figure management logic

These patterns demonstrate the baseline state that motivated the development of
figregistry-kedro's automated figure styling and management capabilities.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


def create_exploratory_data_analysis(data: pd.DataFrame, experiment_config: Dict[str, Any]) -> None:
    """Create exploratory data analysis plots with manual figure management.
    
    This function demonstrates the traditional approach with:
    - Manual plt.savefig() calls
    - Hardcoded styling parameters
    - Manual file path construction
    - Inconsistent styling approach
    """
    # Manual styling configuration - hardcoded in function
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Hardcoded colors for different conditions - no systematic management
    if experiment_config.get('condition') == 'baseline':
        color_scheme = '#1f77b4'  # Blue
        marker_style = 'o'
    elif experiment_config.get('condition') == 'treatment_a':
        color_scheme = '#ff7f0e'  # Orange  
        marker_style = 's'
    elif experiment_config.get('condition') == 'treatment_b':
        color_scheme = '#2ca02c'  # Green
        marker_style = '^'
    else:
        # Fallback - inconsistent with other functions
        color_scheme = 'black'
        marker_style = 'x'
    
    # Create figure with manual sizing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution plot - manual styling
    ax1.hist(data['feature_1'], bins=30, color=color_scheme, alpha=0.7, edgecolor='black')
    ax1.set_title('Feature 1 Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1 Values')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot - manual styling
    ax2.scatter(data['feature_1'], data['feature_2'], 
               color=color_scheme, marker=marker_style, alpha=0.6, s=50)
    ax2.set_title('Feature 1 vs Feature 2', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.grid(True, alpha=0.3)
    
    # Box plot - manual styling
    box_data = [data[col] for col in ['feature_1', 'feature_2', 'feature_3']]
    bp = ax3.boxplot(box_data, labels=['Feature 1', 'Feature 2', 'Feature 3'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color_scheme)
        patch.set_alpha(0.7)
    ax3.set_title('Feature Distributions', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Values')
    ax3.grid(True, alpha=0.3)
    
    # Correlation heatmap - different styling approach
    corr_matrix = data[['feature_1', 'feature_2', 'feature_3']].corr()
    im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45)
    ax4.set_yticklabels(corr_matrix.columns)
    ax4.set_title('Feature Correlations', fontsize=14, fontweight='bold')
    
    # Add colorbar manually
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    
    # Manual file path construction - hardcoded and inconsistent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition = experiment_config.get('condition', 'unknown')
    
    # Hardcoded output directory - not configurable
    output_dir = Path("data/08_reporting/figures/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Manual filename construction - inconsistent naming pattern
    filename = f"exploratory_analysis_{condition}_{timestamp}.png"
    filepath = output_dir / filename
    
    # Manual plt.savefig call with hardcoded parameters
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"EDA plot saved to: {filepath}")


def create_model_performance_plots(metrics_data: Dict[str, float], 
                                 training_history: pd.DataFrame,
                                 experiment_config: Dict[str, Any]) -> None:
    """Create model performance visualization with manual figure management.
    
    Demonstrates more manual styling inconsistencies and file management issues.
    """
    # Different matplotlib style configuration - inconsistent with other functions
    plt.style.use('seaborn-v0_8')  # Different base style
    plt.rcParams.update({
        'font.size': 11,  # Different from EDA function
        'axes.labelsize': 13,  # Different sizing
        'axes.titlesize': 15,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Different color scheme logic - code duplication with variations
    condition = experiment_config.get('condition', 'default')
    if condition == 'baseline':
        primary_color = '#2E86AB'  # Different blue than EDA
        secondary_color = '#A23B72'
        line_style = '-'
    elif condition == 'treatment_a':
        primary_color = '#F18F01'  # Different orange
        secondary_color = '#C73E1D'
        line_style = '--'
    elif condition == 'treatment_b':
        primary_color = '#85BF3B'  # Different green
        secondary_color = '#592693'
        line_style = '-.'
    else:
        # Different fallback colors - inconsistent
        primary_color = '#666666'
        secondary_color = '#333333'
        line_style = ':'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))  # Different size
    
    # Training loss plot - manual styling
    epochs = range(len(training_history))
    ax1.plot(epochs, training_history['train_loss'], color=primary_color, 
             linewidth=2.5, linestyle=line_style, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, training_history['val_loss'], color=secondary_color,
             linewidth=2.5, linestyle=line_style, label='Validation Loss', marker='s', markersize=4)
    ax1.set_title('Training Progress', fontsize=16, pad=20)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4, linestyle='--')
    
    # Accuracy plot with different styling approach
    ax2.plot(epochs, training_history['train_accuracy'], color=primary_color,
             linewidth=3, alpha=0.8, label='Training Accuracy')
    ax2.plot(epochs, training_history['val_accuracy'], color=secondary_color,
             linewidth=3, alpha=0.8, label='Validation Accuracy')
    ax2.set_title('Model Accuracy', fontsize=16, pad=20)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.4)
    
    # Metrics bar chart - different styling again
    metric_names = list(metrics_data.keys())
    metric_values = list(metrics_data.values())
    
    bars = ax3.bar(metric_names, metric_values, color=primary_color, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_title('Model Performance Metrics', fontsize=16, pad=20)
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars - manual positioning
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Learning rate plot if available
    if 'learning_rate' in training_history.columns:
        ax4.plot(epochs, training_history['learning_rate'], color=secondary_color,
                linewidth=2, marker='D', markersize=3)
        ax4.set_title('Learning Rate Schedule', fontsize=16, pad=20)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.4)
    else:
        ax4.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Learning Rate Schedule', fontsize=16, pad=20)
    
    plt.tight_layout(pad=3.0)
    
    # Different file management approach - more manual and inconsistent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Different format
    model_name = experiment_config.get('model_name', 'model')
    
    # Different output directory structure
    base_dir = "data/08_reporting"
    condition_dir = os.path.join(base_dir, f"condition_{condition}")
    os.makedirs(condition_dir, exist_ok=True)
    
    # Yet another filename pattern - inconsistent
    filename = f"{model_name}_performance_{timestamp}.png"
    full_path = os.path.join(condition_dir, filename)
    
    # Different savefig parameters - inconsistent settings
    plt.savefig(full_path, dpi=200, bbox_inches='tight', pad_inches=0.2, 
                facecolor='white', transparent=False)
    plt.close()
    
    print(f"Model performance plot saved to: {full_path}")


def create_comparison_plots(baseline_data: pd.DataFrame, 
                          treatment_data: pd.DataFrame,
                          experiment_config: Dict[str, Any]) -> None:
    """Create comparison plots between experimental conditions.
    
    Shows even more code duplication and manual styling management.
    """
    # Yet another styling approach - no consistency
    plt.rcParams.update({
        'font.family': 'serif',  # Different font family
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Hardcoded colors again - duplicated logic with variations
    baseline_color = '#1f77b4'
    treatment_color = '#ff7f0e'
    
    # Manual figure creation with different approach
    fig = plt.figure(figsize=(18, 6))
    
    # Side-by-side comparison plots
    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(baseline_data['outcome'], bins=25, alpha=0.7, color=baseline_color, 
             label='Baseline', edgecolor='black', linewidth=1)
    ax1.hist(treatment_data['outcome'], bins=25, alpha=0.7, color=treatment_color,
             label='Treatment', edgecolor='black', linewidth=1)
    ax1.set_title('Outcome Distribution Comparison', fontweight='bold')
    ax1.set_xlabel('Outcome Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Box plot comparison with manual styling
    ax2 = plt.subplot(1, 3, 2)
    data_to_plot = [baseline_data['outcome'], treatment_data['outcome']]
    bp = ax2.boxplot(data_to_plot, labels=['Baseline', 'Treatment'], patch_artist=True)
    bp['boxes'][0].set_facecolor(baseline_color)
    bp['boxes'][1].set_facecolor(treatment_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_linewidth(2)
    ax2.set_title('Outcome Distributions', fontweight='bold')
    ax2.set_ylabel('Outcome Value')
    
    # Scatter plot with manual color coding
    ax3 = plt.subplot(1, 3, 3)
    ax3.scatter(baseline_data['feature_1'], baseline_data['outcome'], 
               c=baseline_color, alpha=0.6, s=30, label='Baseline', marker='o')
    ax3.scatter(treatment_data['feature_1'], treatment_data['outcome'],
               c=treatment_color, alpha=0.6, s=30, label='Treatment', marker='^')
    ax3.set_title('Feature vs Outcome', fontweight='bold')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Outcome')
    ax3.legend()
    
    plt.tight_layout()
    
    # Yet another file management approach - complete inconsistency
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    
    # Manual directory creation with hardcoded paths
    output_base = "./outputs/figures"
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    
    comparison_dir = os.path.join(output_base, "comparisons")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Another filename pattern - no standardization
    filename = f"comparison_baseline_vs_treatment_{date_str}_{time_str}.png"
    save_path = os.path.join(comparison_dir, filename)
    
    # Different save parameters again
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='w', edgecolor='w', format='png')
    plt.close()
    
    print(f"Comparison plot saved to: {save_path}")


def create_publication_figure(results_data: pd.DataFrame,
                            statistical_tests: Dict[str, float],
                            experiment_config: Dict[str, Any]) -> None:
    """Create publication-ready figure with extensive manual styling.
    
    Demonstrates the complexity and maintenance burden of manual figure styling
    for publication-quality outputs.
    """
    # Publication-specific styling - completely different approach
    plt.style.use('classic')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'lines.linewidth': 1.5,
        'patch.linewidth': 1.2,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Publication color scheme - hardcoded yet again
    colors = {
        'baseline': '#2C3E50',      # Dark blue-gray
        'treatment_a': '#E74C3C',   # Red
        'treatment_b': '#27AE60',   # Green
        'treatment_c': '#8E44AD'    # Purple
    }
    
    # Create figure with publication dimensions (manual sizing for specific journal)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6))  # Specific journal requirements
    
    # Main results plot with extensive manual styling
    conditions = results_data['condition'].unique()
    means = []
    errors = []
    
    for condition in conditions:
        condition_data = results_data[results_data['condition'] == condition]['outcome']
        means.append(condition_data.mean())
        errors.append(condition_data.std() / np.sqrt(len(condition_data)))  # SEM
    
    x_pos = np.arange(len(conditions))
    bars = ax1.bar(x_pos, means, yerr=errors, capsize=5, capthick=2,
                   color=[colors.get(cond, '#7F8C8D') for cond in conditions],
                   alpha=0.8, edgecolor='black', linewidth=1.2,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    ax1.set_xlabel('Experimental Condition', fontweight='bold')
    ax1.set_ylabel('Outcome Measure (units)', fontweight='bold')
    ax1.set_title('A) Primary Results', fontweight='bold', loc='left')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([cond.replace('_', ' ').title() for cond in conditions], rotation=45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add significance annotations manually
    if 'p_value_baseline_vs_treatment_a' in statistical_tests:
        p_val = statistical_tests['p_value_baseline_vs_treatment_a']
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Manual positioning of significance markers
        y_max = max(means) + max(errors)
        ax1.plot([0, 1], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=1)
        ax1.text(0.5, y_max * 1.15, sig_text, ha='center', va='bottom', fontweight='bold')
    
    # Time series plot with manual formatting
    ax2.errorbar(results_data.groupby('time_point')['outcome'].mean().index,
                results_data.groupby('time_point')['outcome'].mean().values,
                yerr=results_data.groupby('time_point')['outcome'].sem().values,
                marker='o', markersize=6, capsize=4, capthick=2,
                color=colors['baseline'], linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time Point (hours)', fontweight='bold')
    ax2.set_ylabel('Outcome Measure', fontweight='bold')
    ax2.set_title('B) Time Course', fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Correlation plot with manual styling
    feature_cols = [col for col in results_data.columns if col.startswith('feature_')]
    if len(feature_cols) >= 2:
        ax3.scatter(results_data[feature_cols[0]], results_data['outcome'],
                   c=[colors.get(cond, '#7F8C8D') for cond in results_data['condition']],
                   alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel(f'{feature_cols[0].replace("_", " ").title()}', fontweight='bold')
        ax3.set_ylabel('Outcome Measure', fontweight='bold')
        ax3.set_title('C) Feature Correlation', fontweight='bold', loc='left')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Add correlation coefficient manually
        corr_coef = results_data[feature_cols[0]].corr(results_data['outcome'])
        ax3.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Distribution plot with manual density estimation
    for i, condition in enumerate(conditions):
        condition_data = results_data[results_data['condition'] == condition]['outcome']
        ax4.hist(condition_data, bins=15, alpha=0.6, density=True,
                color=colors.get(condition, '#7F8C8D'), 
                label=condition.replace('_', ' ').title(),
                edgecolor='black', linewidth=0.8)
    
    ax4.set_xlabel('Outcome Measure', fontweight='bold')
    ax4.set_ylabel('Density', fontweight='bold')
    ax4.set_title('D) Distribution by Condition', fontweight='bold', loc='left')
    ax4.legend(frameon=False, loc='upper right')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    # Publication-specific file management - most complex approach yet
    date_stamp = datetime.now().strftime("%Y%m%d")
    version = experiment_config.get('figure_version', 'v1')
    
    # Multiple output formats for publication
    output_formats = ['png', 'pdf', 'svg']
    
    # Complex directory structure
    pub_base = "./outputs/publication"
    figure_dir = os.path.join(pub_base, f"figure_1_{version}")
    
    # Create nested directories manually
    for fmt in output_formats:
        fmt_dir = os.path.join(figure_dir, fmt)
        os.makedirs(fmt_dir, exist_ok=True)
    
    # Save in multiple formats with different parameters
    base_filename = f"figure_1_main_results_{date_stamp}_{version}"
    
    for fmt in output_formats:
        if fmt == 'png':
            # High DPI for submission
            save_params = {'dpi': 600, 'bbox_inches': 'tight', 'pad_inches': 0.1,
                          'facecolor': 'white', 'transparent': False}
        elif fmt == 'pdf':
            # Vector format for print
            save_params = {'bbox_inches': 'tight', 'pad_inches': 0.1,
                          'facecolor': 'white', 'transparent': False}
        elif fmt == 'svg':
            # Editable vector format
            save_params = {'bbox_inches': 'tight', 'pad_inches': 0.1,
                          'facecolor': 'white', 'transparent': False}
        
        filepath = os.path.join(figure_dir, fmt, f"{base_filename}.{fmt}")
        plt.savefig(filepath, format=fmt, **save_params)
        print(f"Publication figure saved: {filepath}")
    
    plt.close()


def generate_summary_report_figures(all_results: Dict[str, pd.DataFrame],
                                  experiment_metadata: Dict[str, Any]) -> None:
    """Generate summary figures for final report with extensive manual management.
    
    This function shows the ultimate complexity of manual figure management
    when dealing with multiple datasets and complex layouts.
    """
    # Report-specific styling - another unique approach
    plt.style.use('bmh')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': 'gray',
        'grid.alpha': 0.2,
        'grid.linewidth': 0.8
    })
    
    # Complex figure layout - manual subplot management
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.5], width_ratios=[1, 1, 1],
                         hspace=0.3, wspace=0.3)
    
    # Color palette management - hardcoded again
    experiment_colors = {
        'baseline': '#34495E',
        'treatment_a': '#E67E22',
        'treatment_b': '#16A085',
        'treatment_c': '#9B59B6',
        'treatment_d': '#E74C3C'
    }
    
    # Summary statistics plot
    ax1 = fig.add_subplot(gs[0, 0])
    summary_stats = []
    conditions = []
    
    for condition, data in all_results.items():
        summary_stats.append([
            data['outcome'].mean(),
            data['outcome'].std(),
            data['outcome'].median(),
            len(data)
        ])
        conditions.append(condition)
    
    summary_df = pd.DataFrame(summary_stats, 
                             columns=['Mean', 'Std', 'Median', 'N'],
                             index=conditions)
    
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x_pos - width/2, summary_df['Mean'], width, 
           color=[experiment_colors.get(c, '#95A5A6') for c in conditions],
           alpha=0.8, label='Mean', edgecolor='black')
    ax1.bar(x_pos + width/2, summary_df['Median'], width,
           color=[experiment_colors.get(c, '#95A5A6') for c in conditions],
           alpha=0.5, label='Median', edgecolor='black')
    
    ax1.set_title('Summary Statistics by Condition', fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in conditions], rotation=45)
    ax1.set_ylabel('Outcome Value')
    ax1.legend()
    
    # Effect size plot
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_mean = all_results.get('baseline', pd.DataFrame()).get('outcome', pd.Series()).mean()
    effect_sizes = []
    
    for condition, data in all_results.items():
        if condition != 'baseline' and baseline_mean:
            effect_size = (data['outcome'].mean() - baseline_mean) / data['outcome'].std()
            effect_sizes.append(effect_size)
        else:
            effect_sizes.append(0)
    
    bars = ax2.barh(range(len(conditions)), effect_sizes,
                    color=[experiment_colors.get(c, '#95A5A6') for c in conditions],
                    alpha=0.8, edgecolor='black')
    ax2.set_title('Effect Sizes vs Baseline', fontweight='bold', pad=15)
    ax2.set_yticks(range(len(conditions)))
    ax2.set_yticklabels([c.replace('_', ' ').title() for c in conditions])
    ax2.set_xlabel('Cohen\'s d')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax2.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    
    # Sample size plot
    ax3 = fig.add_subplot(gs[0, 2])
    sample_sizes = [len(data) for data in all_results.values()]
    pie_colors = [experiment_colors.get(c, '#95A5A6') for c in conditions]
    
    wedges, texts, autotexts = ax3.pie(sample_sizes, labels=conditions, colors=pie_colors,
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops={'fontsize': 8})
    ax3.set_title('Sample Size Distribution', fontweight='bold', pad=15)
    
    # Time series comparison
    ax4 = fig.add_subplot(gs[1, :])
    for condition, data in all_results.items():
        if 'time_point' in data.columns:
            time_series = data.groupby('time_point')['outcome'].agg(['mean', 'sem']).reset_index()
            ax4.errorbar(time_series['time_point'], time_series['mean'], 
                        yerr=time_series['sem'], 
                        color=experiment_colors.get(condition, '#95A5A6'),
                        marker='o', markersize=5, capsize=3, capthick=1.5,
                        linewidth=2, alpha=0.8, label=condition.replace('_', ' ').title())
    
    ax4.set_title('Outcome Time Series by Condition', fontweight='bold', pad=15)
    ax4.set_xlabel('Time Point (hours)')
    ax4.set_ylabel('Outcome Measure')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Feature correlation heatmap
    ax5 = fig.add_subplot(gs[2, :2])
    
    # Combine all data for correlation analysis
    combined_data = pd.concat(all_results.values(), ignore_index=True)
    feature_cols = [col for col in combined_data.columns if col.startswith('feature_')]
    
    if len(feature_cols) > 1:
        corr_matrix = combined_data[feature_cols + ['outcome']].corr()
        im = ax5.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax5.set_xticks(range(len(corr_matrix.columns)))
        ax5.set_yticks(range(len(corr_matrix.columns)))
        ax5.set_xticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns], 
                           rotation=45, ha='right')
        ax5.set_yticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns])
        ax5.set_title('Feature Correlation Matrix', fontweight='bold', pad=15)
        
        # Add correlation values manually
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                               fontsize=8, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Statistical summary table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Create summary table manually
    table_data = []
    for condition, data in all_results.items():
        table_data.append([
            condition.replace('_', ' ').title(),
            f"{data['outcome'].mean():.3f}",
            f"{data['outcome'].std():.3f}",
            f"{len(data)}"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Condition', 'Mean', 'Std Dev', 'N'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table manually
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#3498DB')
                cell.set_text_props(weight='bold', color='white')
            else:
                condition = table_data[i-1][0].lower().replace(' ', '_')
                cell.set_facecolor(experiment_colors.get(condition, '#ECF0F1'))
    
    ax6.set_title('Summary Statistics Table', fontweight='bold', pad=15, y=0.95)
    
    # Metadata summary
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    # Create metadata text manually
    metadata_text = f"""
    Experiment Summary: {experiment_metadata.get('experiment_name', 'Unknown')}
    Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Total Samples: {sum(len(data) for data in all_results.values())}
    Conditions Tested: {len(all_results)}
    Analysis Version: {experiment_metadata.get('analysis_version', '1.0')}
    """
    
    ax7.text(0.5, 0.5, metadata_text, ha='center', va='center', 
            transform=ax7.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    # Complex file management for report figures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_version = experiment_metadata.get('report_version', 'v1')
    
    # Multiple output strategies
    output_strategies = {
        'draft': {'dpi': 150, 'format': 'png', 'quality': 'medium'},
        'review': {'dpi': 300, 'format': 'png', 'quality': 'high'},
        'final': {'dpi': 600, 'format': 'pdf', 'quality': 'publication'}
    }
    
    base_dir = "./outputs/reports"
    
    for strategy_name, params in output_strategies.items():
        strategy_dir = os.path.join(base_dir, strategy_name, report_version)
        os.makedirs(strategy_dir, exist_ok=True)
        
        filename = f"summary_report_{strategy_name}_{timestamp}.{params['format']}"
        filepath = os.path.join(strategy_dir, filename)
        
        # Different save parameters for each strategy
        if params['format'] == 'png':
            save_kwargs = {
                'dpi': params['dpi'],
                'bbox_inches': 'tight',
                'pad_inches': 0.2,
                'facecolor': 'white',
                'edgecolor': 'none',
                'transparent': False
            }
        else:  # PDF
            save_kwargs = {
                'bbox_inches': 'tight',
                'pad_inches': 0.2,
                'facecolor': 'white',
                'edgecolor': 'none',
                'transparent': False
            }
        
        plt.savefig(filepath, format=params['format'], **save_kwargs)
        print(f"Summary report saved ({strategy_name}): {filepath}")
    
    plt.close()
    
    # Additional individual figure exports - more manual management
    individual_dir = os.path.join(base_dir, "individual_plots", report_version)
    os.makedirs(individual_dir, exist_ok=True)
    
    print(f"Summary report generation complete. All files saved to: {base_dir}")