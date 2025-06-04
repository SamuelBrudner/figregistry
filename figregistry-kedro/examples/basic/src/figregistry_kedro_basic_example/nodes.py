"""
FigRegistry-Kedro Basic Example Node Functions

This module demonstrates zero-touch figure management through figregistry-kedro 
integration, showcasing how Kedro pipeline nodes can create matplotlib figures 
without manual styling or save operations. The FigureDataSet automatically 
applies condition-based styling and handles persistence based on experimental 
conditions and output purposes.

Key Demonstrations:
- Matplotlib figure creation without manual plt.savefig() calls (F-005)
- Condition-based styling through pipeline parameter resolution (F-005-RQ-004)
- Automated styling application based on experimental conditions (F-002)
- Different output purposes: exploratory, presentation, publication (F-004)
- Zero-touch figure management workflow (Section 0.1.1 objective)

The node functions in this module output raw matplotlib Figure objects that are
automatically intercepted by FigureDataSet during catalog save operations. The
dataset applies appropriate styling based on:
- Experimental conditions (treatment vs control)
- Output purpose (exploratory, presentation, publication)
- Pipeline parameters and context resolution
- FigRegistry configuration-driven styling rules

This approach eliminates manual figure management overhead while ensuring
consistent, publication-ready visualizations across all workflow outputs.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

# Core scientific computing imports
import numpy as np
import pandas as pd

# Matplotlib imports for figure creation
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.figure import Figure

# Configure module logger
logger = logging.getLogger(__name__)

# Set random seed for reproducible demonstrations
np.random.seed(42)


def create_sample_dataset(
    n_samples: int = 200, 
    experiment_condition: str = "treatment",
    dataset_type: str = "primary"
) -> pd.DataFrame:
    """
    Generate synthetic experimental data for visualization demonstrations.
    
    This function creates sample datasets that simulate different experimental
    conditions, enabling demonstration of condition-based styling through
    figregistry-kedro integration. The generated data includes both treatment
    and control conditions with realistic statistical properties.
    
    Args:
        n_samples: Number of data points to generate
        experiment_condition: Experimental condition for data generation
            - "treatment": Enhanced effect with higher variance
            - "control": Baseline effect with lower variance
        dataset_type: Type of dataset to generate
            - "primary": Main experimental measurements
            - "secondary": Supplementary measurements
            - "validation": Validation measurements
    
    Returns:
        pandas DataFrame containing experimental data with columns:
        - time_point: Sequential measurement timepoints
        - measurement: Primary measurement values
        - error: Measurement uncertainties
        - condition: Experimental condition label
        - category: Data categorization for grouping
    
    Note:
        This function outputs data only - no figures are created here.
        Figure creation is handled by dedicated visualization nodes that
        output raw matplotlib Figure objects for automated styling.
    """
    logger.info(
        f"Generating {n_samples} samples for condition='{experiment_condition}', "
        f"type='{dataset_type}'"
    )
    
    # Generate time series data
    time_points = np.linspace(0, 10, n_samples)
    
    # Create condition-specific effects
    if experiment_condition == "treatment":
        # Enhanced effect with higher signal-to-noise ratio
        base_signal = 2.0 * np.sin(time_points) + 0.5 * np.cos(2 * time_points)
        noise_level = 0.3
        effect_size = 1.5
    else:  # control condition
        # Baseline effect with lower signal-to-noise ratio
        base_signal = 1.0 * np.sin(time_points) + 0.2 * np.cos(2 * time_points)
        noise_level = 0.2
        effect_size = 1.0
    
    # Add noise and experimental effects
    noise = np.random.normal(0, noise_level, n_samples)
    measurements = effect_size * base_signal + noise
    
    # Generate measurement uncertainties
    base_error = 0.1
    errors = base_error + 0.05 * np.random.exponential(1, n_samples)
    
    # Create category labels for grouping demonstrations
    categories = np.random.choice(
        ["group_a", "group_b", "group_c"], 
        size=n_samples,
        p=[0.4, 0.35, 0.25]
    )
    
    # Construct result DataFrame
    dataset = pd.DataFrame({
        "time_point": time_points,
        "measurement": measurements,
        "error": errors,
        "condition": experiment_condition,
        "category": categories,
        "dataset_type": dataset_type
    })
    
    logger.debug(
        f"Generated dataset summary: "
        f"mean={measurements.mean():.3f}, "
        f"std={measurements.std():.3f}, "
        f"categories={len(np.unique(categories))}"
    )
    
    return dataset


def create_exploratory_time_series_plot(
    dataset: pd.DataFrame,
    experiment_condition: str = "treatment"
) -> Figure:
    """
    Create exploratory time series visualization for initial data analysis.
    
    This node demonstrates zero-touch figure management by creating a raw 
    matplotlib Figure object without any manual styling or save operations.
    The FigureDataSet will automatically:
    - Apply exploratory-purpose styling based on catalog configuration
    - Resolve condition-based styling using experiment_condition parameter
    - Handle file persistence with appropriate naming and versioning
    - Apply FigRegistry configuration-driven visual properties
    
    The exploratory purpose typically results in:
    - Lower DPI for faster rendering during analysis iterations
    - Simplified styling optimized for data investigation
    - Quick visual feedback without publication-ready formatting
    
    Args:
        dataset: Experimental data from create_sample_dataset()
        experiment_condition: Condition parameter for style resolution
            Used by FigureDataSet to determine appropriate styling through
            FigRegistry's condition_styles configuration
    
    Returns:
        matplotlib Figure object ready for automated styling and persistence
        
    Note:
        No manual plt.savefig() or styling calls are made in this function.
        All styling and persistence is handled automatically by FigureDataSet
        based on catalog configuration and experimental conditions.
    """
    logger.info(
        f"Creating exploratory time series plot for condition='{experiment_condition}'"
    )
    
    # Create figure and axes without manual styling
    # FigureDataSet will apply exploratory-purpose styling automatically
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw data without manual style specification
    # Style properties (color, linewidth, markers) will be applied by FigRegistry
    ax.plot(
        dataset["time_point"], 
        dataset["measurement"],
        label=f"Condition: {experiment_condition.title()}"
    )
    
    # Add error bars without style specification
    ax.errorbar(
        dataset["time_point"][::10],  # Subsample for cleaner error bars
        dataset["measurement"][::10],
        yerr=dataset["error"][::10],
        fmt='none',  # Style will be applied by FigRegistry
        alpha=0.5,
        label="Measurement uncertainty"
    )
    
    # Set basic labels and title (styling applied automatically)
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Measurement Value")
    ax.set_title(f"Exploratory Analysis: {experiment_condition.title()} Condition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistical summary as text annotation
    mean_val = dataset["measurement"].mean()
    std_val = dataset["measurement"].std()
    ax.text(
        0.02, 0.98, 
        f"μ = {mean_val:.3f}\nσ = {std_val:.3f}\nn = {len(dataset)}",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Ensure proper layout without manual styling
    plt.tight_layout()
    
    logger.debug(
        f"Created exploratory figure with {len(dataset)} data points, "
        f"condition='{experiment_condition}'"
    )
    
    # Return raw Figure object for automated styling by FigureDataSet
    # No manual save operation - catalog handles persistence automatically
    return fig


def create_presentation_comparative_plot(
    treatment_data: pd.DataFrame,
    control_data: pd.DataFrame,
    experiment_condition: str = "comparison"
) -> Figure:
    """
    Create presentation-quality comparative visualization for stakeholder communication.
    
    This node demonstrates automated figure management for presentation purposes,
    creating a publication-ready comparative analysis without manual styling.
    The FigureDataSet will automatically:
    - Apply presentation-purpose styling with enhanced visual clarity
    - Use condition-based styling for consistent treatment vs control representation
    - Handle higher DPI rendering appropriate for presentation displays
    - Apply professional color schemes and formatting from FigRegistry configuration
    
    The presentation purpose typically results in:
    - Higher DPI (200) for crisp display on screens and projectors
    - Enhanced color contrast and professional styling
    - Clear legends and annotations optimized for audience viewing
    - Consistent branding and formatting across all presentation figures
    
    Args:
        treatment_data: Experimental data for treatment condition
        control_data: Experimental data for control condition  
        experiment_condition: Condition parameter for automated style resolution
            Used by FigureDataSet to apply appropriate comparative styling
    
    Returns:
        matplotlib Figure object configured for automated presentation styling
        
    Note:
        This function focuses on data visualization logic only.
        All styling (colors, markers, line weights) and persistence
        are handled automatically by the figregistry-kedro integration.
    """
    logger.info(
        f"Creating presentation comparative plot for condition='{experiment_condition}'"
    )
    
    # Create subplot layout for comparative analysis
    # FigureDataSet will apply presentation-purpose styling automatically
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Time series comparison
    # Raw plotting without manual style specification
    ax1.plot(
        treatment_data["time_point"],
        treatment_data["measurement"],
        label="Treatment"
    )
    ax1.plot(
        control_data["time_point"],
        control_data["measurement"],
        label="Control"
    )
    
    ax1.set_xlabel("Time Point")
    ax1.set_ylabel("Measurement Value")
    ax1.set_title("Time Series Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Distribution comparison using boxplots
    # FigRegistry will apply consistent styling to all plot elements
    data_for_boxplot = [
        treatment_data["measurement"],
        control_data["measurement"]
    ]
    box_plot = ax2.boxplot(
        data_for_boxplot,
        labels=["Treatment", "Control"],
        patch_artist=True  # Enable styling of box patches
    )
    
    ax2.set_ylabel("Measurement Value")
    ax2.set_title("Distribution Comparison")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Summary statistics comparison
    # Calculate key metrics for presentation
    treatment_stats = {
        "mean": treatment_data["measurement"].mean(),
        "std": treatment_data["measurement"].std(),
        "median": treatment_data["measurement"].median()
    }
    
    control_stats = {
        "mean": control_data["measurement"].mean(),
        "std": control_data["measurement"].std(),
        "median": control_data["measurement"].median()
    }
    
    # Create bar chart for statistical comparison
    metrics = ["Mean", "Median", "Std Dev"]
    treatment_values = [
        treatment_stats["mean"],
        treatment_stats["median"], 
        treatment_stats["std"]
    ]
    control_values = [
        control_stats["mean"],
        control_stats["median"],
        control_stats["std"]
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    # Raw bar plotting - styling applied by FigRegistry
    ax3.bar(x_pos - width/2, treatment_values, width, label="Treatment")
    ax3.bar(x_pos + width/2, control_values, width, label="Control")
    
    ax3.set_xlabel("Statistical Metrics")
    ax3.set_ylabel("Value")
    ax3.set_title("Statistical Summary")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Calculate effect size for presentation
    pooled_std = np.sqrt(
        (treatment_stats["std"]**2 + control_stats["std"]**2) / 2
    )
    effect_size = (treatment_stats["mean"] - control_stats["mean"]) / pooled_std
    
    # Add effect size annotation
    fig.suptitle(
        f"Comparative Analysis: Effect Size = {effect_size:.3f}",
        fontsize=16,
        y=0.98
    )
    
    # Ensure proper layout without manual styling adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    logger.debug(
        f"Created presentation figure with effect_size={effect_size:.3f}, "
        f"treatment_n={len(treatment_data)}, control_n={len(control_data)}"
    )
    
    # Return raw Figure object for automated presentation styling
    # FigureDataSet handles all styling and persistence automatically
    return fig


def create_publication_analysis_plot(
    dataset: pd.DataFrame,
    experiment_condition: str = "publication"
) -> Figure:
    """
    Create publication-ready analysis visualization with statistical rigor.
    
    This node demonstrates the highest level of automated figure management,
    creating publication-quality visualizations without any manual styling.
    The FigureDataSet will automatically:
    - Apply publication-purpose styling with maximum quality settings
    - Use high DPI (300) rendering suitable for print publications
    - Apply professional typography and consistent formatting
    - Handle advanced statistical visualization requirements
    - Ensure compliance with journal submission standards
    
    The publication purpose typically results in:
    - Highest DPI (300) for print-quality output
    - Professional color schemes optimized for both digital and print
    - Precise typography and spacing for academic publications
    - Statistical annotations and error representations
    - Compliance with scientific visualization best practices
    
    Args:
        dataset: Experimental data for comprehensive analysis
        experiment_condition: Condition parameter for style resolution
            Used by FigureDataSet to apply publication-appropriate styling
    
    Returns:
        matplotlib Figure object configured for automated publication styling
        
    Note:
        This function implements scientific visualization logic without
        manual styling concerns. All visual properties, formatting, and
        quality settings are managed by the figregistry-kedro integration
        based on publication-purpose configuration.
    """
    logger.info(
        f"Creating publication analysis plot for condition='{experiment_condition}'"
    )
    
    # Create sophisticated subplot layout for comprehensive analysis
    # FigureDataSet will apply publication-quality styling automatically
    fig = plt.figure(figsize=(12, 10))
    
    # Define subplot grid for publication layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
    
    # Main time series panel with statistical analysis
    ax_main = fig.add_subplot(gs[0, :])
    
    # Group data by category for detailed analysis
    categories = dataset["category"].unique()
    
    for category in categories:
        category_data = dataset[dataset["category"] == category]
        
        # Plot individual category data without manual styling
        ax_main.plot(
            category_data["time_point"],
            category_data["measurement"],
            label=f"Group {category.split('_')[1].upper()}",
            alpha=0.7
        )
        
        # Add trend line using polynomial fitting
        z = np.polyfit(category_data["time_point"], category_data["measurement"], 2)
        p = np.poly1d(z)
        
        # Plot trend without manual styling
        ax_main.plot(
            category_data["time_point"],
            p(category_data["time_point"]),
            linestyle="--",
            alpha=0.8
        )
    
    # Add overall trend analysis
    z_overall = np.polyfit(dataset["time_point"], dataset["measurement"], 3)
    p_overall = np.poly1d(z_overall)
    
    ax_main.plot(
        dataset["time_point"],
        p_overall(dataset["time_point"]),
        linewidth=3,
        label="Overall Trend",
        alpha=0.9
    )
    
    # Add confidence intervals using rolling statistics
    window_size = 20
    rolling_mean = dataset["measurement"].rolling(window=window_size, center=True).mean()
    rolling_std = dataset["measurement"].rolling(window=window_size, center=True).std()
    
    # Plot confidence interval without manual styling
    ax_main.fill_between(
        dataset["time_point"],
        rolling_mean - 1.96 * rolling_std,
        rolling_mean + 1.96 * rolling_std,
        alpha=0.2,
        label="95% Confidence Interval"
    )
    
    ax_main.set_xlabel("Time Point")
    ax_main.set_ylabel("Measurement Value")
    ax_main.set_title("Comprehensive Time Series Analysis")
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    # Residual analysis panel
    ax_residual = fig.add_subplot(gs[1, 0])
    
    # Calculate residuals from overall trend
    predicted = p_overall(dataset["time_point"])
    residuals = dataset["measurement"] - predicted
    
    # Plot residuals without manual styling
    ax_residual.scatter(
        dataset["time_point"],
        residuals,
        alpha=0.6,
        s=30
    )
    ax_residual.axhline(y=0, linestyle='-', alpha=0.8)
    ax_residual.set_xlabel("Time Point")
    ax_residual.set_ylabel("Residuals")
    ax_residual.set_title("Residual Analysis")
    ax_residual.grid(True, alpha=0.3)
    
    # Statistical distribution panel
    ax_hist = fig.add_subplot(gs[1, 1])
    
    # Create histogram without manual styling
    n_bins = 25
    ax_hist.hist(
        dataset["measurement"],
        bins=n_bins,
        density=True,
        alpha=0.7,
        edgecolor='none'
    )
    
    # Add normal distribution overlay for comparison
    mu = dataset["measurement"].mean()
    sigma = dataset["measurement"].std()
    x_norm = np.linspace(dataset["measurement"].min(), dataset["measurement"].max(), 100)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
    
    ax_hist.plot(x_norm, y_norm, linewidth=2, label="Normal Distribution")
    ax_hist.set_xlabel("Measurement Value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution Analysis")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Correlation analysis panel
    ax_corr = fig.add_subplot(gs[2, :])
    
    # Create lag correlation analysis
    max_lag = 20
    lags = range(-max_lag, max_lag + 1)
    correlations = []
    
    for lag in lags:
        if lag == 0:
            correlations.append(1.0)
        else:
            # Calculate lagged correlation
            if lag > 0:
                x = dataset["measurement"][:-lag]
                y = dataset["measurement"][lag:]
            else:
                x = dataset["measurement"][-lag:]
                y = dataset["measurement"][:lag]
            
            if len(x) > 1 and len(y) > 1:
                correlation = np.corrcoef(x, y)[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0)
            else:
                correlations.append(0)
    
    # Plot correlation function without manual styling
    ax_corr.plot(lags, correlations, marker='o', markersize=4)
    ax_corr.axhline(y=0, linestyle='-', alpha=0.5)
    ax_corr.axhline(y=0.2, linestyle='--', alpha=0.5, label="Significance Threshold")
    ax_corr.axhline(y=-0.2, linestyle='--', alpha=0.5)
    ax_corr.set_xlabel("Lag")
    ax_corr.set_ylabel("Correlation Coefficient")
    ax_corr.set_title("Autocorrelation Analysis")
    ax_corr.legend()
    ax_corr.grid(True, alpha=0.3)
    
    # Add comprehensive statistical summary
    stats_text = (
        f"Statistical Summary:\n"
        f"N = {len(dataset)}\n"
        f"Mean = {mu:.4f}\n"
        f"Std = {sigma:.4f}\n"
        f"Skewness = {dataset['measurement'].skew():.4f}\n"
        f"Kurtosis = {dataset['measurement'].kurtosis():.4f}\n"
        f"Condition = {experiment_condition}"
    )
    
    fig.text(
        0.02, 0.02, stats_text,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        verticalalignment='bottom'
    )
    
    # Ensure proper layout without manual spacing adjustments
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)
    
    logger.debug(
        f"Created publication figure with comprehensive analysis: "
        f"n_categories={len(categories)}, mean={mu:.3f}, std={sigma:.3f}"
    )
    
    # Return raw Figure object for automated publication styling
    # FigureDataSet applies all quality settings and persistence automatically
    return fig


def create_condition_comparison_matrix(
    treatment_data: pd.DataFrame,
    control_data: pd.DataFrame,
    experiment_condition: str = "matrix_comparison"
) -> Figure:
    """
    Create advanced multi-panel comparison matrix for comprehensive analysis.
    
    This node demonstrates sophisticated automated figure management with
    complex multi-panel layouts and statistical comparisons. The function
    showcases how figregistry-kedro handles advanced visualizations while
    maintaining zero-touch styling and persistence.
    
    The FigureDataSet will automatically:
    - Apply condition-specific styling across all panels consistently
    - Handle complex subplot styling with unified color schemes
    - Manage high-resolution output appropriate for the specified purpose
    - Apply statistical visualization best practices from FigRegistry configuration
    
    Args:
        treatment_data: Experimental data for treatment condition
        control_data: Experimental data for control condition
        experiment_condition: Condition parameter for comprehensive style resolution
            Used to determine appropriate styling for comparative analysis
    
    Returns:
        matplotlib Figure object with complex multi-panel layout ready for
        automated styling and persistence through figregistry-kedro integration
        
    Note:
        This function demonstrates advanced visualization patterns without
        any manual styling code. All color coordination, formatting, and
        quality settings are handled by the automated styling system.
    """
    logger.info(
        f"Creating condition comparison matrix for condition='{experiment_condition}'"
    )
    
    # Create complex subplot matrix without manual styling
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel (0,0): Raw time series overlay
    ax = axes[0, 0]
    ax.plot(treatment_data["time_point"], treatment_data["measurement"], 
            label="Treatment", alpha=0.8)
    ax.plot(control_data["time_point"], control_data["measurement"], 
            label="Control", alpha=0.8)
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Measurement")
    ax.set_title("Time Series Overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel (0,1): Difference plot
    ax = axes[0, 1]
    
    # Interpolate control data to match treatment timepoints for direct comparison
    treatment_interp = np.interp(
        control_data["time_point"], 
        treatment_data["time_point"], 
        treatment_data["measurement"]
    )
    difference = treatment_interp - control_data["measurement"]
    
    ax.plot(control_data["time_point"], difference, linewidth=2)
    ax.axhline(y=0, linestyle='--', alpha=0.7)
    ax.fill_between(control_data["time_point"], 0, difference, alpha=0.3)
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Treatment - Control")
    ax.set_title("Condition Difference")
    ax.grid(True, alpha=0.3)
    
    # Panel (0,2): Statistical comparison
    ax = axes[0, 2]
    
    # Create violin plots for distribution comparison
    data_for_violin = [treatment_data["measurement"], control_data["measurement"]]
    violin_parts = ax.violinplot(data_for_violin, positions=[1, 2], 
                                widths=0.6, showmeans=True, showmedians=True)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Treatment", "Control"])
    ax.set_ylabel("Measurement Value")
    ax.set_title("Distribution Comparison")
    ax.grid(True, alpha=0.3)
    
    # Panel (1,0): Rolling statistics
    ax = axes[1, 0]
    
    # Calculate rolling statistics for both conditions
    window = 20
    treatment_rolling = treatment_data["measurement"].rolling(window=window, center=True).mean()
    control_rolling = control_data["measurement"].rolling(window=window, center=True).mean()
    
    ax.plot(treatment_data["time_point"], treatment_rolling, 
            label="Treatment (rolling)", linewidth=2)
    ax.plot(control_data["time_point"], control_rolling, 
            label="Control (rolling)", linewidth=2)
    
    # Add rolling standard deviation bands
    treatment_rolling_std = treatment_data["measurement"].rolling(window=window, center=True).std()
    control_rolling_std = control_data["measurement"].rolling(window=window, center=True).std()
    
    ax.fill_between(treatment_data["time_point"], 
                   treatment_rolling - treatment_rolling_std,
                   treatment_rolling + treatment_rolling_std, alpha=0.2)
    ax.fill_between(control_data["time_point"],
                   control_rolling - control_rolling_std, 
                   control_rolling + control_rolling_std, alpha=0.2)
    
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Rolling Mean ± Std")
    ax.set_title("Rolling Statistics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel (1,1): Correlation analysis
    ax = axes[1, 1]
    
    # Create scatter plot with regression line
    # Align data by interpolating to common timepoints
    common_time = np.linspace(
        max(treatment_data["time_point"].min(), control_data["time_point"].min()),
        min(treatment_data["time_point"].max(), control_data["time_point"].max()),
        100
    )
    
    treatment_interp = np.interp(common_time, treatment_data["time_point"], 
                                treatment_data["measurement"])
    control_interp = np.interp(common_time, control_data["time_point"], 
                              control_data["measurement"])
    
    ax.scatter(treatment_interp, control_interp, alpha=0.6, s=20)
    
    # Add regression line
    z = np.polyfit(treatment_interp, control_interp, 1)
    p = np.poly1d(z)
    x_reg = np.linspace(treatment_interp.min(), treatment_interp.max(), 100)
    ax.plot(x_reg, p(x_reg), linewidth=2, alpha=0.8)
    
    # Calculate and display correlation
    correlation = np.corrcoef(treatment_interp, control_interp)[0, 1]
    ax.text(0.05, 0.95, f"r = {correlation:.3f}", transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    ax.set_xlabel("Treatment Measurement")
    ax.set_ylabel("Control Measurement")
    ax.set_title("Cross-Condition Correlation")
    ax.grid(True, alpha=0.3)
    
    # Panel (1,2): Effect size analysis
    ax = axes[1, 2]
    
    # Calculate effect sizes across different time windows
    window_starts = np.arange(0, len(treatment_data) - window, window // 2)
    effect_sizes = []
    window_centers = []
    
    for start in window_starts:
        end = start + window
        if end <= len(treatment_data) and end <= len(control_data):
            t_window = treatment_data["measurement"].iloc[start:end]
            c_window = control_data["measurement"].iloc[start:end]
            
            # Cohen's d effect size
            pooled_std = np.sqrt((t_window.var() + c_window.var()) / 2)
            if pooled_std > 0:
                cohen_d = (t_window.mean() - c_window.mean()) / pooled_std
                effect_sizes.append(cohen_d)
                window_centers.append(treatment_data["time_point"].iloc[start + window // 2])
    
    if effect_sizes:
        ax.plot(window_centers, effect_sizes, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=0.2, linestyle='--', alpha=0.5, label="Small Effect")
        ax.axhline(y=0.5, linestyle='--', alpha=0.5, label="Medium Effect")
        ax.axhline(y=0.8, linestyle='--', alpha=0.5, label="Large Effect")
        ax.axhline(y=0, linestyle='-', alpha=0.3)
        
        ax.set_xlabel("Time Point")
        ax.set_ylabel("Effect Size (Cohen's d)")
        ax.set_title("Dynamic Effect Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add overall figure title and statistics
    overall_treatment_mean = treatment_data["measurement"].mean()
    overall_control_mean = control_data["measurement"].mean()
    overall_effect = (overall_treatment_mean - overall_control_mean) / \
                    np.sqrt((treatment_data["measurement"].var() + 
                            control_data["measurement"].var()) / 2)
    
    fig.suptitle(
        f"Comprehensive Condition Comparison Matrix\n"
        f"Overall Effect Size: {overall_effect:.3f} | "
        f"Treatment μ = {overall_treatment_mean:.3f} | "
        f"Control μ = {overall_control_mean:.3f}",
        fontsize=14, y=0.98
    )
    
    # Ensure proper layout without manual adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    logger.debug(
        f"Created comparison matrix with overall_effect={overall_effect:.3f}, "
        f"correlation={correlation:.3f}, n_windows={len(effect_sizes)}"
    )
    
    # Return raw Figure object for automated styling by FigureDataSet
    return fig


# Export all node functions for Kedro pipeline registration
__all__ = [
    "create_sample_dataset",
    "create_exploratory_time_series_plot", 
    "create_presentation_comparative_plot",
    "create_publication_analysis_plot",
    "create_condition_comparison_matrix"
]