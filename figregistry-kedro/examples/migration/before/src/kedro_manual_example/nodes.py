"""
Traditional Kedro node functions demonstrating problematic manual matplotlib figure management.

This module showcases the scattered plt.savefig() calls, hardcoded styling, manual file path
management, and code duplication patterns that figregistry-kedro eliminates. These functions
represent the "before" state in the migration example, highlighting the maintenance overhead
and inconsistencies that automated figure management resolves.

Key Problems Demonstrated:
- Manual plt.savefig() calls scattered throughout node functions
- Hardcoded styling without systematic condition-based management  
- Manual file path construction and naming patterns
- Code duplication with repeated styling logic across functions
- Inconsistent experimental condition handling
- Manual intervention requirements for figure management
"""

import os
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def explore_data_distribution(data: pd.DataFrame, params: Dict[str, Any]) -> None:
    """
    Create exploratory data distribution plots with manual figure management.
    
    PROBLEMS DEMONSTRATED:
    - Manual plt.savefig() calls with hardcoded paths
    - Inconsistent styling across different conditions
    - Manual timestamp generation for versioning
    - Hardcoded color schemes without systematic management
    """
    # Manual styling configuration - PROBLEM: hardcoded, not reusable
    if params.get("experiment_type") == "pilot":
        # Different hardcoded colors for pilot experiments
        primary_color = "#FF6B6B"  # Red for pilot
        secondary_color = "#4ECDC4"  # Teal
        marker_style = "o"
    elif params.get("experiment_type") == "production":
        # Different hardcoded colors for production
        primary_color = "#45B7D1"  # Blue for production  
        secondary_color = "#96CEB4"  # Green
        marker_style = "s"
    else:
        # Default hardcoded fallback - PROBLEM: inconsistent with above patterns
        primary_color = "gray"
        secondary_color = "lightgray"
        marker_style = "^"
    
    # Manual figure creation with hardcoded settings
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Data Distribution Analysis", fontsize=16, fontweight='bold')
    
    # Histogram with manual styling
    axes[0, 0].hist(data['value'], bins=30, color=primary_color, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title("Value Distribution", fontsize=14, color=primary_color)
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot with different manual styling - PROBLEM: inconsistent with histogram
    axes[0, 1].boxplot(data['value'], patch_artist=True, 
                       boxprops=dict(facecolor=secondary_color, alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_title("Value Distribution (Box Plot)", fontsize=14, color=secondary_color)
    axes[0, 1].set_ylabel("Value")
    
    # Scatter plot with manual condition-based coloring - PROBLEM: not systematic
    if 'category' in data.columns:
        categories = data['category'].unique()
        # Manual color assignment - PROBLEM: hardcoded, not scalable
        colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(categories)]
        for i, cat in enumerate(categories):
            mask = data['category'] == cat
            axes[1, 0].scatter(data.loc[mask, 'value'], data.loc[mask, 'secondary_value'], 
                             c=colors[i % len(colors)], marker=marker_style, 
                             label=cat, alpha=0.6, s=50)
        axes[1, 0].legend()
    else:
        axes[1, 0].scatter(data['value'], data['secondary_value'], 
                          c=primary_color, marker=marker_style, alpha=0.6, s=50)
    
    axes[1, 0].set_title("Value Correlation", fontsize=14)
    axes[1, 0].set_xlabel("Primary Value")
    axes[1, 0].set_ylabel("Secondary Value")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time series plot with manual date formatting - PROBLEM: repetitive setup
    if 'date' in data.columns:
        axes[1, 1].plot(data['date'], data['value'], color=primary_color, 
                       linewidth=2, marker=marker_style, markersize=4)
        axes[1, 1].set_title("Value Over Time", fontsize=14)
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Value")
        # Manual date formatting - PROBLEM: repeated code
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, "No date data available", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # PROBLEM: Manual file path construction with hardcoded patterns
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = params.get("experiment_type", "default")
    
    # PROBLEM: Different path patterns for different purposes - not systematic
    if params.get("purpose") == "exploratory":
        output_dir = "data/08_reporting/exploratory"
        filename = f"data_exploration_{experiment}_{timestamp}.png"
    elif params.get("purpose") == "presentation":
        output_dir = "data/08_reporting/presentation"  
        filename = f"presentation_data_dist_{experiment}_{timestamp}.png"
    else:
        # PROBLEM: Inconsistent fallback pattern
        output_dir = "data/08_reporting"
        filename = f"figure_{timestamp}.png"
    
    # PROBLEM: Manual directory creation with error-prone logic
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    
    # PROBLEM: Manual plt.savefig() call with hardcoded parameters
    plt.savefig(full_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved exploration plot to: {full_path}")


def analyze_correlations(data: pd.DataFrame, params: Dict[str, Any]) -> None:
    """
    Create correlation analysis plots with different manual styling patterns.
    
    PROBLEMS DEMONSTRATED:
    - Repeated styling logic from other functions (code duplication)
    - Different file naming convention (inconsistency)
    - Manual colormap selection without systematic approach
    - Different error handling patterns
    """
    # PROBLEM: Duplicated condition checking logic from above function
    experiment = params.get("experiment_type", "unknown")
    
    # PROBLEM: Different styling approach than explore_data_distribution
    if experiment == "pilot":
        colormap = "Reds"
        title_color = "#8B0000"  # Different red than above function
    elif experiment == "production":
        colormap = "Blues"
        title_color = "#000080"  # Different blue than above function
    else:
        colormap = "Greys"
        title_color = "black"
    
    # Calculate correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    # Manual figure setup with different parameters than other functions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Correlation heatmap with manual styling
    im = axes[0].imshow(corr_matrix, cmap=colormap, aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(corr_matrix.columns)))
    axes[0].set_yticks(range(len(corr_matrix.columns)))
    axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    axes[0].set_yticklabels(corr_matrix.columns)
    axes[0].set_title("Correlation Matrix", fontsize=16, color=title_color, weight='bold')
    
    # Manual colorbar creation - PROBLEM: different approach than other functions
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label("Correlation Coefficient", fontsize=12)
    
    # Add correlation values manually - PROBLEM: repetitive formatting code
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            # Manual text color selection based on value
            text_color = 'white' if abs(value) > 0.5 else 'black'
            axes[0].text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color=text_color, fontweight='bold')
    
    # Scatter plot matrix with manual subplot management
    n_vars = min(4, len(numeric_data.columns))  # Limit to 4 variables
    selected_vars = numeric_data.columns[:n_vars]
    
    # Manual grid creation - PROBLEM: complex manual layout management
    gs = fig.add_gridspec(n_vars, n_vars, left=0.55, right=0.95, 
                         top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
    
    for i, var1 in enumerate(selected_vars):
        for j, var2 in enumerate(selected_vars):
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                # Diagonal: histograms with manual styling
                ax.hist(numeric_data[var1], bins=20, color=title_color, alpha=0.7)
                ax.set_title(var1, fontsize=10)
            else:
                # Off-diagonal: scatter plots with manual styling
                ax.scatter(numeric_data[var2], numeric_data[var1], 
                          alpha=0.6, s=20, color=title_color)
                if i == len(selected_vars) - 1:
                    ax.set_xlabel(var2, fontsize=9)
                if j == 0:
                    ax.set_ylabel(var1, fontsize=9)
            
            # Manual tick formatting - PROBLEM: repeated formatting code
            ax.tick_params(labelsize=8)
            if len(ax.get_xticklabels()) > 5:
                ax.tick_params(axis='x', rotation=45)
    
    # PROBLEM: Different timestamp format than other functions
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    purpose = params.get("purpose", "analysis")
    
    # PROBLEM: Different directory structure logic
    if purpose == "presentation":
        base_dir = f"data/08_reporting/presentations/{experiment}"
        filename = f"correlations_presentation_{timestamp}.pdf"  # Different format!
    elif purpose == "exploratory":
        base_dir = f"data/08_reporting/exploration/{experiment}"  
        filename = f"corr_analysis_{timestamp}.png"
    else:
        # PROBLEM: Yet another different fallback pattern
        base_dir = "figures"  # Different base directory!
        filename = f"correlation_analysis_{experiment}_{timestamp}.svg"  # Different format!
    
    # PROBLEM: Different error handling approach
    try:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(base_dir) / filename
        
        # PROBLEM: Different save parameters than other functions
        plt.savefig(output_path, dpi=400, bbox_inches='tight',  # Different DPI!
                   format=output_path.suffix[1:], transparent=True)  # Different options!
        plt.close()
        print(f"Correlation analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to save correlation plot: {e}")
        plt.close()  # Make sure to close even on error


def create_summary_dashboard(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a comprehensive summary dashboard with manual figure management.
    
    PROBLEMS DEMONSTRATED:
    - Complex manual subplot layout management
    - Mixed styling approaches from previous functions
    - Manual aggregation and calculation logic
    - Inconsistent error handling and file management
    - Return data along with side-effect figure saving (poor separation)
    """
    # PROBLEM: Yet another different styling approach 
    condition = params.get("condition", "baseline")
    stage = params.get("stage", "development")
    
    # PROBLEM: Complex manual condition-to-style mapping
    style_config = {
        "baseline": {"color": "#2E8B57", "marker": "o", "linestyle": "-"},
        "treatment_a": {"color": "#DC143C", "marker": "s", "linestyle": "--"},
        "treatment_b": {"color": "#4169E1", "marker": "^", "linestyle": "-."},
        "control": {"color": "#8B4513", "marker": "d", "linestyle": ":"}
    }
    
    # PROBLEM: Default fallback with different pattern than other functions
    current_style = style_config.get(condition, {
        "color": "black", "marker": "x", "linestyle": "-"
    })
    
    # Manual data aggregation - this would be better as separate node
    summary_stats = {
        'mean': data.groupby('category')['value'].mean() if 'category' in data.columns else data['value'].mean(),
        'std': data.groupby('category')['value'].std() if 'category' in data.columns else data['value'].std(),
        'count': data.groupby('category')['value'].count() if 'category' in data.columns else len(data)
    }
    
    # PROBLEM: Complex manual figure layout with hardcoded dimensions
    fig = plt.figure(figsize=(16, 12))
    
    # Manual subplot creation with magic numbers
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1],
                         hspace=0.3, wspace=0.3)
    
    # Top row: summary statistics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Manual bar plots with inconsistent styling
    if isinstance(summary_stats['mean'], pd.Series):
        # Grouped data case
        categories = summary_stats['mean'].index
        x_pos = np.arange(len(categories))
        
        # Different color scheme than previous functions
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))  # PROBLEM: different approach
        
        ax1.bar(x_pos, summary_stats['mean'], color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title("Mean Values by Category", fontweight='bold', color=current_style['color'])
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories, rotation=45)
        ax1.set_ylabel("Mean Value")
        
        ax2.bar(x_pos, summary_stats['std'], color=colors, alpha=0.6, edgecolor='gray')
        ax2.set_title("Standard Deviation by Category", fontweight='bold', color=current_style['color'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, rotation=45)
        ax2.set_ylabel("Std Deviation")
        
        ax3.bar(x_pos, summary_stats['count'], color=colors, alpha=0.9, edgecolor='darkblue')
        ax3.set_title("Sample Count by Category", fontweight='bold', color=current_style['color'])
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories, rotation=45)
        ax3.set_ylabel("Count")
        
    else:
        # Single value case - PROBLEM: completely different visualization approach
        metrics = ['Mean', 'Std Dev', 'Count']
        values = [summary_stats['mean'], summary_stats['std'], summary_stats['count']]
        
        for ax, metric, value in zip([ax1, ax2, ax3], metrics, values):
            ax.bar([0], [value], color=current_style['color'], alpha=0.7, width=0.5)
            ax.set_title(f"{metric}: {value:.2f}", fontweight='bold')
            ax.set_xticks([])
            ax.set_ylabel("Value")
    
    # Middle row: distribution analysis
    ax4 = fig.add_subplot(gs[1, :])  # Span all columns
    
    if 'category' in data.columns:
        # PROBLEM: Manual violin plot setup with hardcoded parameters
        categories = data['category'].unique()
        positions = range(len(categories))
        violin_data = [data[data['category'] == cat]['value'].values for cat in categories]
        
        parts = ax4.violinplot(violin_data, positions=positions, widths=0.8, 
                              showmeans=True, showmedians=True)
        
        # Manual styling of violin plot components - PROBLEM: complex manual customization
        for pc in parts['bodies']:
            pc.set_facecolor(current_style['color'])
            pc.set_alpha(0.6)
        
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('blue') 
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')
        
        ax4.set_xticks(positions)
        ax4.set_xticklabels(categories)
        ax4.set_title("Value Distribution by Category", fontsize=14, fontweight='bold',
                     color=current_style['color'])
        ax4.set_ylabel("Value")
        ax4.grid(True, alpha=0.3)
        
    else:
        # Single distribution
        ax4.hist(data['value'], bins=50, color=current_style['color'], 
                alpha=0.7, edgecolor='black', density=True)
        ax4.set_title("Overall Value Distribution", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Value")
        ax4.set_ylabel("Density")
        ax4.grid(True, alpha=0.3)
    
    # Bottom row: trend analysis (if time data available)
    ax5 = fig.add_subplot(gs[2, :])
    
    if 'date' in data.columns:
        # PROBLEM: Manual time series aggregation
        daily_data = data.groupby(data['date'].dt.date)['value'].agg(['mean', 'std']).reset_index()
        
        ax5.plot(daily_data['date'], daily_data['mean'], 
                color=current_style['color'], linewidth=2, 
                marker=current_style['marker'], markersize=6,
                linestyle=current_style['linestyle'], label='Daily Mean')
        
        # Manual error band calculation and plotting
        ax5.fill_between(daily_data['date'], 
                        daily_data['mean'] - daily_data['std'],
                        daily_data['mean'] + daily_data['std'],
                        alpha=0.3, color=current_style['color'], label='±1 Std Dev')
        
        ax5.set_title("Daily Value Trends", fontsize=14, fontweight='bold',
                     color=current_style['color'])
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Value")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Manual date formatting - PROBLEM: different from other functions
        ax5.tick_params(axis='x', rotation=30)
        
    else:
        ax5.text(0.5, 0.5, f"No time series data available\nCondition: {condition}\nStage: {stage}", 
                ha='center', va='center', transform=ax5.transAxes, 
                fontsize=12, color=current_style['color'])
    
    # PROBLEM: Complex manual title formatting
    fig.suptitle(f"Summary Dashboard - {condition.title()} Condition ({stage.title()} Stage)", 
                fontsize=18, fontweight='bold', y=0.98)
    
    # PROBLEM: Yet another different file naming and path convention
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")  # Different format again!
    
    # PROBLEM: Complex conditional path logic
    if stage == "development":
        if params.get("purpose") == "presentation":
            output_dir = f"reports/development/presentations/{condition}"
            filename = f"dashboard_dev_{condition}_{timestamp}.png"
        else:
            output_dir = f"reports/development/analysis/{condition}"
            filename = f"summary_dashboard_dev_{timestamp}.pdf"  # Mixed formats!
    elif stage == "production":
        output_dir = f"reports/production/{condition}"
        filename = f"prod_dashboard_{condition}_{timestamp}.svg"  # Another format!
    else:
        # PROBLEM: Yet another fallback pattern
        output_dir = "outputs/dashboards"
        filename = f"dashboard_{condition}_{stage}_{timestamp}.png"
    
    # PROBLEM: Manual directory creation with different error handling
    full_output_dir = Path(output_dir)
    try:
        full_output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"WARNING: Permission denied creating {full_output_dir}, using current directory")
        full_output_dir = Path(".")
        filename = f"dashboard_fallback_{timestamp}.png"
    
    output_path = full_output_dir / filename
    
    # PROBLEM: Different save parameters again
    try:
        format_type = output_path.suffix[1:] if output_path.suffix else 'png'
        plt.savefig(output_path, format=format_type, dpi=350,  # Another different DPI!
                   bbox_inches='tight', facecolor='white', edgecolor='black')  # Different edge!
        print(f"Dashboard saved successfully to: {output_path}")
        
    except Exception as e:
        # PROBLEM: Fallback save with yet different parameters
        fallback_path = Path(f"dashboard_error_{timestamp}.png")
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')  # Low quality fallback
        print(f"ERROR saving to {output_path}: {e}")
        print(f"Saved to fallback location: {fallback_path}")
    
    finally:
        plt.close()
    
    # PROBLEM: Mixing side effects (file saving) with data return - poor separation of concerns
    summary_df = pd.DataFrame({
        'condition': [condition],
        'stage': [stage], 
        'timestamp': [timestamp],
        'output_path': [str(output_path)],
        'mean_value': [summary_stats['mean'] if not isinstance(summary_stats['mean'], pd.Series) 
                      else summary_stats['mean'].mean()],
        'total_count': [summary_stats['count'] if not isinstance(summary_stats['count'], pd.Series)
                       else summary_stats['count'].sum()]
    })
    
    return summary_df


def create_publication_plots(data: pd.DataFrame, params: Dict[str, Any]) -> None:
    """
    Generate publication-ready plots with manual high-DPI formatting.
    
    PROBLEMS DEMONSTRATED:
    - Manual publication formatting with hardcoded parameters
    - Complex manual legend and annotation management
    - Inconsistent styling approach from all previous functions
    - Manual multi-format saving with different compression settings
    - No systematic approach to publication standards
    """
    # PROBLEM: Publication styling hardcoded differently than all other functions
    condition = params.get("condition", "control")
    publication_type = params.get("publication_type", "journal")
    
    # PROBLEM: Complex manual publication style configuration
    if publication_type == "journal":
        figure_width = 6.5  # Single column width in inches
        figure_height = 5.0
        font_size = 10
        title_size = 12
        dpi_setting = 600  # Journal requirement
        formats = ['pdf', 'eps', 'png']  # Multiple formats required
    elif publication_type == "conference":
        figure_width = 8.0  # Larger for conference
        figure_height = 6.0  
        font_size = 12
        title_size = 14
        dpi_setting = 300
        formats = ['pdf', 'png']
    else:
        # PROBLEM: Default doesn't match either standard
        figure_width = 10.0
        figure_height = 8.0
        font_size = 11
        title_size = 13
        dpi_setting = 150
        formats = ['png']
    
    # PROBLEM: Manual font configuration that overrides matplotlib defaults
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': title_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'figure.titlesize': title_size + 2,
        'font.family': 'serif',  # PROBLEM: Hardcoded font choice
        'font.serif': ['Times New Roman', 'Times', 'serif'],
    })
    
    # Create figure with manual sizing
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, figure_height))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)  # Manual spacing
    
    # PROBLEM: Manual color palette that doesn't coordinate with other functions
    pub_colors = {
        'control': '#1f77b4',      # Blue
        'treatment_a': '#ff7f0e',  # Orange
        'treatment_b': '#2ca02c',  # Green  
        'baseline': '#d62728'      # Red
    }
    current_color = pub_colors.get(condition, '#7f7f7f')  # Gray fallback
    
    # Plot 1: Main effect with error bars
    if 'category' in data.columns:
        categories = data['category'].unique()
        means = []
        errors = []
        
        for cat in categories:
            cat_data = data[data['category'] == cat]['value']
            means.append(cat_data.mean())
            errors.append(cat_data.std() / np.sqrt(len(cat_data)))  # SEM
        
        x_pos = np.arange(len(categories))
        
        # PROBLEM: Manual error bar styling with hardcoded parameters
        bars = axes[0, 0].bar(x_pos, means, yerr=errors, 
                             color=current_color, alpha=0.8, edgecolor='black',
                             capsize=5, capthick=2, elinewidth=2, linewidth=1.5)
        
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(categories, rotation=0)
        axes[0, 0].set_ylabel('Mean Value ± SEM')
        axes[0, 0].set_title(f'Main Effect ({condition.title()})', fontweight='bold')
        
        # PROBLEM: Manual significance annotation - hardcoded positions
        if len(means) >= 2:
            y_max = max([m + e for m, e in zip(means, errors)])
            axes[0, 0].annotate('*', xy=(0.5, y_max * 1.1), ha='center', 
                               fontsize=16, fontweight='bold')
            axes[0, 0].plot([0, 1], [y_max * 1.05, y_max * 1.05], 'k-', lw=1)
    
    # Plot 2: Dose-response curve with manual curve fitting
    if 'dose' in data.columns:
        doses = data['dose'].unique()
        dose_means = []
        dose_errors = []
        
        for dose in sorted(doses):
            dose_data = data[data['dose'] == dose]['value']
            dose_means.append(dose_data.mean())
            dose_errors.append(dose_data.std() / np.sqrt(len(dose_data)))
        
        sorted_doses = sorted(doses)
        
        # PROBLEM: Manual curve plotting with hardcoded parameters
        axes[0, 1].errorbar(sorted_doses, dose_means, yerr=dose_errors,
                           color=current_color, marker='o', markersize=6,
                           linewidth=2, capsize=4, capthick=1.5)
        
        # Manual trend line - PROBLEM: simplified linear fit
        z = np.polyfit(sorted_doses, dose_means, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(sorted_doses, p(sorted_doses), '--', 
                       color=current_color, alpha=0.7, linewidth=1.5)
        
        axes[0, 1].set_xlabel('Dose (arbitrary units)')
        axes[0, 1].set_ylabel('Response')
        axes[0, 1].set_title('Dose-Response Relationship', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # PROBLEM: Manual R² calculation and annotation
        correlation = np.corrcoef(sorted_doses, dose_means)[0, 1]
        r_squared = correlation ** 2
        axes[0, 1].text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                       transform=axes[0, 1].transAxes, fontsize=font_size-1,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Distribution comparison with manual statistical annotations
    if 'group' in data.columns:
        groups = data['group'].unique()
        
        # PROBLEM: Manual box plot styling with publication parameters
        box_data = [data[data['group'] == group]['value'] for group in groups]
        bp = axes[1, 0].boxplot(box_data, labels=groups, patch_artist=True,
                               boxprops=dict(facecolor=current_color, alpha=0.7, linewidth=1.5),
                               medianprops=dict(color='red', linewidth=2),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
        
        axes[1, 0].set_ylabel('Value Distribution')
        axes[1, 0].set_title('Group Comparison', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # PROBLEM: Manual statistical test annotation (simplified)
        if len(groups) == 2:
            from scipy import stats
            group1_data = data[data['group'] == groups[0]]['value']
            group2_data = data[data['group'] == groups[1]]['value']
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            # Manual p-value annotation formatting
            if p_value < 0.001:
                p_text = "p < 0.001"
            elif p_value < 0.01:
                p_text = f"p < 0.01"
            elif p_value < 0.05:
                p_text = f"p < 0.05"
            else:
                p_text = f"p = {p_value:.3f}"
            
            axes[1, 0].text(0.5, 0.95, p_text, transform=axes[1, 0].transAxes,
                           ha='center', fontsize=font_size-1, fontweight='bold')
    
    # Plot 4: Time course with publication formatting
    if 'time' in data.columns:
        time_points = sorted(data['time'].unique())
        time_means = []
        time_errors = []
        
        for tp in time_points:
            tp_data = data[data['time'] == tp]['value']
            time_means.append(tp_data.mean())
            time_errors.append(tp_data.std() / np.sqrt(len(tp_data)))
        
        # PROBLEM: Manual time course plotting with publication styling
        axes[1, 1].errorbar(time_points, time_means, yerr=time_errors,
                           color=current_color, marker='s', markersize=5,
                           linewidth=2, capsize=3, capthick=1,
                           markerfacecolor='white', markeredgecolor=current_color,
                           markeredgewidth=1.5)
        
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Response')
        axes[1, 1].set_title('Time Course', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # PROBLEM: Manual axis formatting for publication
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)
        axes[1, 1].spines['left'].set_linewidth(1.5)
        axes[1, 1].spines['bottom'].set_linewidth(1.5)
    
    # PROBLEM: Manual overall figure formatting
    fig.suptitle(f'Publication Figure - {condition.title()} Condition', 
                fontsize=title_size + 2, fontweight='bold', y=0.98)
    
    # PROBLEM: Complex manual file saving with multiple formats
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"publication_{publication_type}_{condition}_{timestamp}"
    
    # PROBLEM: Different directory structure again
    if publication_type == "journal":
        output_dir = f"manuscripts/figures/{condition}"
    elif publication_type == "conference":
        output_dir = f"presentations/conference_figures/{condition}"
    else:
        output_dir = f"figures/publication/{condition}"
    
    # PROBLEM: Manual multi-format saving with different parameters per format
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filename = f"{base_name}.{fmt}"
        filepath = Path(output_dir) / filename
        
        # PROBLEM: Format-specific manual parameter settings
        if fmt == 'pdf':
            plt.savefig(filepath, format='pdf', dpi=dpi_setting, bbox_inches='tight',
                       facecolor='white', edgecolor='none', 
                       metadata={'Creator': 'Manual Figure Management', 
                                'Subject': f'{condition} analysis'})
        elif fmt == 'eps':
            plt.savefig(filepath, format='eps', dpi=dpi_setting, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        elif fmt == 'png':
            plt.savefig(filepath, format='png', dpi=dpi_setting, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       optimize=True, pnginfo=None)  # Manual PNG optimization
        
        print(f"Publication figure saved: {filepath}")
    
    plt.close()
    
    # PROBLEM: Manual rcParams reset - but not to original values
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',  # Different reset than what was set
        # Missing other resets - will affect subsequent plots!
    })


def quick_diagnostic_plot(data: pd.DataFrame, column: str, save_path: Optional[str] = None) -> None:
    """
    Create a quick diagnostic plot with minimal styling - shows inconsistency.
    
    PROBLEMS DEMONSTRATED:
    - Minimal styling that doesn't match other functions
    - Optional save path that's handled differently
    - Quick-and-dirty approach that creates technical debt
    - No consideration for systematic styling or branding
    """
    # PROBLEM: Minimal figure setup - no consideration for consistency
    plt.figure(figsize=(8, 6))
    
    # PROBLEM: Basic plot with default matplotlib styling
    plt.hist(data[column], bins=20, alpha=0.7)  # Default blue color
    plt.title(f"Quick Diagnostic: {column}")  # Basic title, no styling
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    # PROBLEM: Optional manual save with different default behavior
    if save_path:
        # User provided path - save directly without any checks
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Quick plot saved to: {save_path}")
    else:
        # PROBLEM: Yet another different auto-naming convention
        auto_name = f"quick_diagnostic_{column}_{datetime.datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(auto_name, dpi=100)  # Lower quality for "quick" plots
        print(f"Quick plot auto-saved as: {auto_name}")
    
    plt.close()


# PROBLEM: Utility function with manual styling that doesn't integrate with other functions
def apply_custom_style(ax, style_type: str = "default"):
    """
    Apply custom styling to matplotlib axes - but inconsistent with main functions.
    
    This utility function demonstrates the problem of scattered styling logic
    that doesn't integrate systematically with the main plotting functions.
    """
    # PROBLEM: Manual style definitions that don't match main function styling
    styles = {
        "default": {"color": "blue", "grid": True, "spines": False},
        "minimal": {"color": "gray", "grid": False, "spines": True},
        "bold": {"color": "red", "grid": True, "spines": True}
    }
    
    style = styles.get(style_type, styles["default"])
    
    # PROBLEM: Manual application that may conflict with existing styling
    if style["grid"]:
        ax.grid(True, alpha=0.3)
    
    if not style["spines"]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # PROBLEM: Color applied generically without context
    for line in ax.get_lines():
        line.set_color(style["color"])


def batch_plot_generator(datasets: Dict[str, pd.DataFrame], output_base: str) -> None:
    """
    Generate multiple plots in batch with manual file management.
    
    PROBLEMS DEMONSTRATED:
    - Manual iteration and file naming
    - No systematic approach to batch styling
    - Complex manual directory and file management
    - Error handling that's inconsistent with other functions
    """
    # PROBLEM: Manual timestamp for batch - different format again
    batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"{output_base}/batch_{batch_timestamp}"
    
    # PROBLEM: Manual directory creation with basic error handling
    try:
        os.makedirs(batch_dir, exist_ok=True)
    except:
        print("Failed to create batch directory, using current directory")
        batch_dir = "."
    
    # PROBLEM: Manual iteration with inconsistent styling per dataset
    for i, (name, dataset) in enumerate(datasets.items()):
        plt.figure(figsize=(10, 6))
        
        # PROBLEM: Cycling through hardcoded colors without systematic approach
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color = colors[i % len(colors)]
        
        # PROBLEM: Generic plot that doesn't consider data characteristics
        if 'value' in dataset.columns:
            plt.plot(dataset['value'], color=color, linewidth=2, alpha=0.8)
            plt.title(f"Dataset: {name}", fontsize=14, color=color)
            plt.ylabel("Value")
            plt.xlabel("Index")
        else:
            plt.text(0.5, 0.5, f"No 'value' column in {name}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # PROBLEM: Manual file naming that's different from all other functions
        filename = f"batch_plot_{i:02d}_{name.replace(' ', '_')}_{batch_timestamp}.png"
        filepath = os.path.join(batch_dir, filename)
        
        # PROBLEM: Basic save parameters without consideration for use case
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Batch plot {i+1}/{len(datasets)}: {filepath}")
    
    print(f"Batch plotting completed. Files saved in: {batch_dir}")