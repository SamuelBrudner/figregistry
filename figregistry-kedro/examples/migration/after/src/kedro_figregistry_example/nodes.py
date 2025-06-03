"""
Converted Kedro node functions demonstrating the elimination of manual matplotlib figure management
through figregistry-kedro integration.

This module showcases the 'after' state where plt.savefig() calls have been removed, nodes return
matplotlib figure objects directly, and FigureDataSet handles all styling and persistence
automatically based on experimental conditions.

Key Benefits Demonstrated:
- Elimination of manual plt.savefig() calls throughout pipeline nodes
- Automatic figure styling based on experimental conditions via catalog configuration
- Centralized configuration management through figregistry.yml integration
- Clean separation between visualization logic and figure management
- Reduced code duplication and maintenance overhead

Usage:
    These node functions are designed to be used with figregistry-kedro's FigureDataSet
    in the Kedro catalog configuration. The catalog handles all styling and persistence
    automatically based on the configured purpose and condition parameters.
"""

import logging
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


def create_exploratory_scatter_plot(
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """
    Create an exploratory scatter plot for correlation analysis.
    
    This node demonstrates the 'after' state where matplotlib figures are returned
    directly without manual styling or save operations. FigureDataSet automatically
    applies condition-based styling and handles persistence based on catalog configuration.
    
    Args:
        data: Input DataFrame containing variables for correlation analysis
        parameters: Pipeline parameters that may include condition information
            
    Returns:
        matplotlib.figure.Figure: Styled figure object for automatic processing
    """
    logger.info("Creating exploratory scatter plot for correlation analysis")
    
    # Create figure - no manual styling needed, FigureDataSet handles this
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract columns for analysis (assuming data has x and y columns)
    x_col = parameters.get('x_column', 'feature_1')
    y_col = parameters.get('y_column', 'feature_2')
    
    if x_col not in data.columns or y_col not in data.columns:
        # Create sample data for demonstration
        x_data = np.random.randn(200)
        y_data = 2 * x_data + np.random.randn(200) * 0.5
        x_col, y_col = 'X Variable', 'Y Variable'
    else:
        x_data = data[x_col]
        y_data = data[y_col]
    
    # Create scatter plot - styling handled automatically by FigureDataSet
    ax.scatter(x_data, y_data, alpha=0.6)
    
    # Add regression line for analysis
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax.plot(x_data, p(x_data), "--", alpha=0.8)
    
    # Add correlation coefficient
    correlation = np.corrcoef(x_data, y_data)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Set labels and title - final styling applied by FigRegistry
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('Exploratory Correlation Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Return figure object directly - no plt.savefig() call needed
    # FigureDataSet will automatically apply styling and handle persistence
    return fig


def create_time_series_analysis(
    time_series_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """
    Generate time series analysis visualization.
    
    Demonstrates clean node implementation focused on visualization logic without
    manual figure management concerns. All styling and output management handled
    automatically by figregistry-kedro integration.
    
    Args:
        time_series_data: DataFrame with time series data
        parameters: Pipeline parameters for configuration
        
    Returns:
        matplotlib.figure.Figure: Figure object for automated processing
    """
    logger.info("Creating time series analysis visualization")
    
    # Create multi-panel figure for comprehensive analysis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Generate sample time series if data not provided
    if time_series_data.empty or 'timestamp' not in time_series_data.columns:
        logger.info("Generating sample time series data for demonstration")
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        trend = np.linspace(100, 120, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 2, len(dates))
        values = trend + seasonal + noise
        
        time_series_data = pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'trend': trend,
            'seasonal': seasonal
        })
    
    timestamps = time_series_data['timestamp']
    values = time_series_data['value']
    
    # Plot 1: Original time series
    ax1.plot(timestamps, values, linewidth=1.5, alpha=0.8, label='Observed')
    if 'trend' in time_series_data.columns:
        ax1.plot(timestamps, time_series_data['trend'], 
                linewidth=2, alpha=0.9, label='Trend')
    ax1.set_title('Time Series Analysis - Original Data')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling statistics
    window = parameters.get('rolling_window', 30)
    rolling_mean = values.rolling(window=window).mean()
    rolling_std = values.rolling(window=window).std()
    
    ax2.plot(timestamps, rolling_mean, label=f'{window}-day Moving Average', linewidth=2)
    ax2.fill_between(timestamps, 
                     rolling_mean - rolling_std, 
                     rolling_mean + rolling_std, 
                     alpha=0.3, label=f'±1 Std Dev')
    ax2.set_title('Rolling Statistics')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution analysis
    ax3.hist(values, bins=50, alpha=0.7, density=True, label='Data Distribution')
    
    # Fit normal distribution for comparison
    mu, sigma = stats.norm.fit(values)
    x = np.linspace(values.min(), values.max(), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 
             linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
    
    ax3.set_title('Value Distribution Analysis')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Return figure for automatic styling and persistence by FigureDataSet
    return fig


def create_categorical_summary(
    categorical_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """
    Generate categorical data summary visualization.
    
    Shows systematic condition-based styling through catalog configuration rather
    than manual style application. Demonstrates clean node implementation without
    figure management overhead.
    
    Args:
        categorical_data: DataFrame containing categorical variables
        parameters: Pipeline parameters for customization
        
    Returns:
        matplotlib.figure.Figure: Figure for automated processing
    """
    logger.info("Creating categorical data summary visualization")
    
    # Create figure with subplots for comprehensive categorical analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate sample categorical data if not provided
    if categorical_data.empty:
        logger.info("Generating sample categorical data for demonstration")
        np.random.seed(42)  # For reproducible demo data
        
        categories_a = np.random.choice(['Type A', 'Type B', 'Type C', 'Type D'], 
                                       size=1000, p=[0.4, 0.3, 0.2, 0.1])
        categories_b = np.random.choice(['Low', 'Medium', 'High'], 
                                       size=1000, p=[0.3, 0.5, 0.2])
        numerical_values = np.random.exponential(2, 1000)
        
        categorical_data = pd.DataFrame({
            'category_a': categories_a,
            'category_b': categories_b,
            'values': numerical_values,
            'success_rate': np.random.beta(2, 3, 1000)
        })
    
    # Plot 1: Bar chart for primary categorical variable
    category_counts = categorical_data['category_a'].value_counts()
    bars1 = ax1.bar(category_counts.index, category_counts.values)
    ax1.set_title('Distribution by Primary Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Plot 2: Grouped bar chart for cross-tabulation
    crosstab = pd.crosstab(categorical_data['category_a'], categorical_data['category_b'])
    crosstab.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Cross-tabulation Analysis')
    ax2.set_xlabel('Primary Category')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Secondary Category')
    
    # Plot 3: Box plot for numerical variable by category
    categories = categorical_data['category_a'].unique()
    box_data = [categorical_data[categorical_data['category_a'] == cat]['values'] 
                for cat in categories]
    
    box_plot = ax3.boxplot(box_data, labels=categories, patch_artist=True)
    ax3.set_title('Value Distribution by Category')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Values')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Success rate analysis
    success_stats = categorical_data.groupby('category_a')['success_rate'].agg(['mean', 'std'])
    bars4 = ax4.bar(success_stats.index, success_stats['mean'], 
                   yerr=success_stats['std'], capsize=5)
    ax4.set_title('Success Rate by Category')
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Success Rate')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1)
    
    # Add percentage labels
    for i, (bar, mean_val) in enumerate(zip(bars4, success_stats['mean'])):
        ax4.text(bar.get_x() + bar.get_width()/2., mean_val + 0.02,
                f'{mean_val:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Return figure object - FigureDataSet handles all styling and persistence
    return fig


def create_comparative_analysis(
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
    parameters: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """
    Generate comparative analysis between two datasets.
    
    Demonstrates advanced visualization patterns with clean separation between
    data analysis logic and figure management. All styling applied automatically
    based on experimental conditions via FigureDataSet configuration.
    
    Args:
        dataset_a: First dataset for comparison
        dataset_b: Second dataset for comparison  
        parameters: Pipeline parameters including comparison settings
        
    Returns:
        matplotlib.figure.Figure: Figure for automated styling and persistence
    """
    logger.info("Creating comparative analysis visualization")
    
    # Create comprehensive comparison figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Generate sample data if not provided
    if dataset_a.empty or dataset_b.empty:
        logger.info("Generating sample datasets for comparative analysis")
        np.random.seed(123)
        
        # Dataset A: Control group
        dataset_a = pd.DataFrame({
            'metric_1': np.random.normal(50, 10, 500),
            'metric_2': np.random.exponential(2, 500),
            'category': np.random.choice(['X', 'Y', 'Z'], 500),
            'success': np.random.binomial(1, 0.3, 500)
        })
        
        # Dataset B: Treatment group (with improved performance)
        dataset_b = pd.DataFrame({
            'metric_1': np.random.normal(55, 8, 500),  # Higher mean, lower variance
            'metric_2': np.random.exponential(1.8, 500),  # Slightly better
            'category': np.random.choice(['X', 'Y', 'Z'], 500),
            'success': np.random.binomial(1, 0.4, 500)  # Higher success rate
        })
    
    # Plot 1: Distribution comparison for primary metric
    ax1.hist(dataset_a['metric_1'], bins=30, alpha=0.6, density=True, 
            label='Dataset A (Control)')
    ax1.hist(dataset_b['metric_1'], bins=30, alpha=0.6, density=True, 
            label='Dataset B (Treatment)')
    
    # Add statistical annotations
    mean_a, mean_b = dataset_a['metric_1'].mean(), dataset_b['metric_1'].mean()
    ax1.axvline(mean_a, color='blue', linestyle='--', alpha=0.8, 
               label=f'Mean A: {mean_a:.2f}')
    ax1.axvline(mean_b, color='orange', linestyle='--', alpha=0.8, 
               label=f'Mean B: {mean_b:.2f}')
    
    ax1.set_title('Metric Distribution Comparison')
    ax1.set_xlabel('Metric 1 Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot comparison
    ax2.scatter(dataset_a['metric_1'], dataset_a['metric_2'], 
               alpha=0.5, label='Dataset A', s=30)
    ax2.scatter(dataset_b['metric_1'], dataset_b['metric_2'], 
               alpha=0.5, label='Dataset B', s=30)
    
    ax2.set_title('Metric Correlation Comparison')
    ax2.set_xlabel('Metric 1')
    ax2.set_ylabel('Metric 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Category-wise comparison
    category_comparison = pd.DataFrame({
        'Dataset A': dataset_a.groupby('category')['metric_1'].mean(),
        'Dataset B': dataset_b.groupby('category')['metric_1'].mean()
    })
    
    category_comparison.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Category-wise Performance Comparison')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Average Metric 1')
    ax3.tick_params(axis='x', rotation=0)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rate comparison with statistical testing
    success_a = dataset_a['success'].mean()
    success_b = dataset_b['success'].mean()
    
    # Calculate confidence intervals (approximate)
    n_a, n_b = len(dataset_a), len(dataset_b)
    se_a = np.sqrt(success_a * (1 - success_a) / n_a)
    se_b = np.sqrt(success_b * (1 - success_b) / n_b)
    
    categories = ['Dataset A', 'Dataset B']
    success_rates = [success_a, success_b]
    errors = [1.96 * se_a, 1.96 * se_b]  # 95% confidence intervals
    
    bars4 = ax4.bar(categories, success_rates, yerr=errors, capsize=10)
    ax4.set_title('Success Rate Comparison')
    ax4.set_ylabel('Success Rate')
    ax4.set_ylim(0, 1)
    
    # Add percentage labels and p-value annotation
    for bar, rate in zip(bars4, success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2., rate + 0.05,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Simple statistical test (z-test for proportions)
    pooled_p = (dataset_a['success'].sum() + dataset_b['success'].sum()) / (n_a + n_b)
    se_diff = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
    z_score = (success_b - success_a) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    ax4.text(0.5, 0.9, f'p-value: {p_value:.4f} {significance}', 
            transform=ax4.transAxes, ha='center',
            bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Return figure object for automatic processing by FigureDataSet
    return fig


def create_statistical_summary(
    analysis_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """
    Generate comprehensive statistical summary visualization.
    
    Final demonstration of clean node implementation with eliminated code duplication
    through FigureDataSet automation. Shows how complex statistical visualizations
    benefit from automated styling and figure management.
    
    Args:
        analysis_data: Dataset for statistical analysis
        parameters: Pipeline parameters for analysis configuration
        
    Returns:
        matplotlib.figure.Figure: Comprehensive statistical summary figure
    """
    logger.info("Creating comprehensive statistical summary visualization")
    
    # Create complex multi-panel statistical summary
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Generate comprehensive sample data if not provided
    if analysis_data.empty:
        logger.info("Generating comprehensive sample data for statistical analysis")
        np.random.seed(456)
        n_samples = 1000
        
        analysis_data = pd.DataFrame({
            'continuous_var': np.random.gamma(2, 2, n_samples),
            'normal_var': np.random.normal(100, 15, n_samples),
            'binary_outcome': np.random.binomial(1, 0.35, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.25, 0.35, 0.25, 0.15]),
            'treatment_group': np.random.choice(['Control', 'Treatment'], n_samples),
            'score': np.random.beta(2, 5, n_samples) * 100
        })
    
    # Panel 1: Distribution analysis with normality testing
    ax1 = fig.add_subplot(gs[0, 0])
    variable = 'continuous_var' if 'continuous_var' in analysis_data.columns else analysis_data.select_dtypes(include=[np.number]).columns[0]
    data_to_plot = analysis_data[variable].dropna()
    
    ax1.hist(data_to_plot, bins=40, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay fitted distributions
    mu, sigma = stats.norm.fit(data_to_plot)
    x = np.linspace(data_to_plot.min(), data_to_plot.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit')
    
    # Shapiro-Wilk test for normality
    if len(data_to_plot) <= 5000:  # Shapiro-Wilk limitation
        _, p_value = stats.shapiro(data_to_plot[:5000])
        ax1.text(0.05, 0.95, f'Shapiro p-value: {p_value:.4f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    ax1.set_title('Distribution Analysis')
    ax1.set_xlabel(variable)
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Q-Q plot for normality assessment
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(data_to_plot, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Box plot by categorical variable
    ax3 = fig.add_subplot(gs[0, 2])
    categorical_col = 'categorical' if 'categorical' in analysis_data.columns else None
    if categorical_col:
        box_data = [analysis_data[analysis_data[categorical_col] == cat][variable].dropna() 
                    for cat in analysis_data[categorical_col].unique()]
        box_plot = ax3.boxplot(box_data, labels=analysis_data[categorical_col].unique(), 
                              patch_artist=True)
        ax3.set_title(f'{variable} by {categorical_col}')
        ax3.set_ylabel(variable)
        ax3.tick_params(axis='x', rotation=45)
    
    # Panel 4: Correlation matrix (if multiple numeric variables)
    ax4 = fig.add_subplot(gs[1, :2])
    numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = analysis_data[numeric_cols].corr()
        im = ax4.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax4.set_xticks(range(len(correlation_matrix.columns)))
        ax4.set_yticks(range(len(correlation_matrix.index)))
        ax4.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax4.set_yticklabels(correlation_matrix.index)
        ax4.set_title('Correlation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
    
    # Panel 5: Statistical test results summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Perform various statistical tests
    test_results = []
    
    if 'treatment_group' in analysis_data.columns and len(analysis_data['treatment_group'].unique()) == 2:
        groups = analysis_data['treatment_group'].unique()
        group1_data = analysis_data[analysis_data['treatment_group'] == groups[0]][variable]
        group2_data = analysis_data[analysis_data['treatment_group'] == groups[1]][variable]
        
        # Independent t-test
        t_stat, t_p = stats.ttest_ind(group1_data, group2_data)
        test_results.append(f"Independent t-test:")
        test_results.append(f"  t = {t_stat:.3f}, p = {t_p:.4f}")
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        test_results.append(f"Mann-Whitney U test:")
        test_results.append(f"  U = {u_stat:.1f}, p = {u_p:.4f}")
    
    if categorical_col and 'binary_outcome' in analysis_data.columns:
        # Chi-square test of independence
        contingency_table = pd.crosstab(analysis_data[categorical_col], analysis_data['binary_outcome'])
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
        test_results.append(f"Chi-square test:")
        test_results.append(f"  χ² = {chi2:.3f}, p = {chi2_p:.4f}")
    
    # Add descriptive statistics
    test_results.append(f"\nDescriptive Statistics ({variable}):")
    test_results.append(f"  Mean: {data_to_plot.mean():.3f}")
    test_results.append(f"  Std: {data_to_plot.std():.3f}")
    test_results.append(f"  Median: {data_to_plot.median():.3f}")
    test_results.append(f"  Skewness: {stats.skew(data_to_plot):.3f}")
    test_results.append(f"  Kurtosis: {stats.kurtosis(data_to_plot):.3f}")
    
    ax5.text(0.05, 0.95, '\n'.join(test_results), transform=ax5.transAxes, 
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax5.set_title('Statistical Test Results')
    
    # Panel 6: Time series or trend analysis (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create synthetic time series for demonstration
    dates = pd.date_range('2023-01-01', periods=len(analysis_data), freq='D')
    if 'score' in analysis_data.columns:
        time_series = analysis_data['score'].rolling(window=7, center=True).mean()
    else:
        time_series = analysis_data[variable].rolling(window=7, center=True).mean()
    
    ax6.plot(dates, time_series, linewidth=1.5, alpha=0.8, label='7-day Moving Average')
    
    # Add trend line
    x_numeric = np.arange(len(dates))
    valid_idx = ~np.isnan(time_series)
    if valid_idx.sum() > 1:
        z = np.polyfit(x_numeric[valid_idx], time_series[valid_idx], 1)
        p = np.poly1d(z)
        ax6.plot(dates, p(x_numeric), "--", alpha=0.8, linewidth=2, 
                label=f'Trend (slope: {z[0]:.4f})')
    
    ax6.set_title('Temporal Trend Analysis')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Value')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Return comprehensive statistical summary figure for automated processing
    # FigureDataSet will apply appropriate styling based on experimental conditions
    return fig