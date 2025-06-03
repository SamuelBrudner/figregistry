"""Advanced utility functions for complex experimental scenarios and sophisticated figure generation.

This module provides production-ready utility functions supporting enterprise-grade 
figregistry-kedro integration with advanced condition-based styling, complex experimental 
condition resolution, multi-environment configuration management, and sophisticated data 
transformation patterns for automated visualization workflows.

The utilities enable complex experimental design scenarios including:
- Multi-variable condition resolution for advanced styling
- Statistical analysis helpers for reporting pipelines
- Configuration management for multi-environment deployments  
- Advanced data transformation patterns for visualization
- Production-ready patterns for enterprise workflows
- Complex experimental condition management per F-002 requirements

These functions eliminate manual figure management while supporting sophisticated 
automated styling across complex data science workflows per Section 0.1.1 objectives.
"""

import logging
import re
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns

try:
    from figregistry_kedro.config import (
        FigRegistryConfigBridge, 
        FigRegistryKedroConfig,
        ConfigurationMergeError
    )
except ImportError:
    warnings.warn(
        "FigRegistry-Kedro config module not found. Some features may be limited.",
        ImportWarning
    )
    FigRegistryConfigBridge = None
    FigRegistryKedroConfig = None
    ConfigurationMergeError = Exception

logger = logging.getLogger(__name__)

# Type aliases for better code readability
ConditionDict = Dict[str, Any]
StyleDict = Dict[str, Any]
MetricsDict = Dict[str, Union[float, int]]
ExperimentalParams = Dict[str, Any]


@dataclass
class ExperimentalCondition:
    """Advanced experimental condition with metadata and inheritance patterns.
    
    Supports complex condition-based styling scenarios per F-002 requirements
    with hierarchical inheritance, wildcard matching, and sophisticated 
    metadata management for enterprise experimental design.
    """
    
    name: str
    treatment: Optional[str] = None
    model_type: Optional[str] = None
    data_split: Optional[str] = None
    environment: Optional[str] = None
    experiment_id: Optional[str] = None
    metrics: MetricsDict = field(default_factory=dict)
    metadata: ConditionDict = field(default_factory=dict)
    parent_conditions: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize derived fields and validation."""
        if not self.name:
            raise ValueError("Condition name cannot be empty")
        
        # Generate condition hash for caching
        self._condition_hash = self._generate_hash()
        
        # Initialize tags if not provided
        if not self.tags:
            self.tags = set()
            if self.treatment:
                self.tags.add(f"treatment_{self.treatment}")
            if self.model_type:
                self.tags.add(f"model_{self.model_type}")
            if self.environment:
                self.tags.add(f"env_{self.environment}")
    
    def _generate_hash(self) -> str:
        """Generate unique hash for condition caching."""
        condition_str = f"{self.name}_{self.treatment}_{self.model_type}_{self.data_split}_{self.environment}"
        return hashlib.md5(condition_str.encode()).hexdigest()[:8]
    
    def matches_pattern(self, pattern: str) -> bool:
        """Check if condition matches wildcard pattern per F-002-RQ-002.
        
        Args:
            pattern: Wildcard pattern (supports *, ?, [abc], {alt1,alt2})
            
        Returns:
            True if condition matches pattern
        """
        # Convert glob pattern to regex
        pattern_regex = pattern.replace('*', '.*').replace('?', '.')
        
        # Support bracket patterns [abc] and brace patterns {alt1,alt2}
        pattern_regex = re.sub(r'\[([^\]]+)\]', r'[\1]', pattern_regex)
        pattern_regex = re.sub(r'\{([^}]+)\}', lambda m: f"(?:{'|'.join(m.group(1).split(','))})", pattern_regex)
        
        # Test against condition name and components
        test_strings = [
            self.name,
            f"{self.treatment}_{self.model_type}" if self.treatment and self.model_type else "",
            f"{self.environment}_{self.data_split}" if self.environment and self.data_split else "",
            "_".join(filter(None, [self.treatment, self.model_type, self.data_split, self.environment]))
        ]
        
        for test_str in filter(None, test_strings):
            if re.match(f"^{pattern_regex}$", test_str):
                return True
                
        return False
    
    def get_hierarchical_names(self) -> List[str]:
        """Get hierarchical condition names for inheritance resolution.
        
        Returns ordered list from most specific to most general for
        condition-based styling inheritance per F-002 requirements.
        """
        names = []
        
        # Most specific: full condition name
        names.append(self.name)
        
        # Include parent conditions in hierarchy
        names.extend(self.parent_conditions)
        
        # Component-based hierarchies
        if self.treatment and self.model_type and self.data_split:
            names.append(f"{self.treatment}_{self.model_type}_{self.data_split}")
        
        if self.treatment and self.model_type:
            names.append(f"{self.treatment}_{self.model_type}")
        
        if self.model_type and self.data_split:
            names.append(f"{self.model_type}_{self.data_split}")
        
        # Single component fallbacks
        if self.treatment:
            names.append(f"treatment_{self.treatment}")
        
        if self.model_type:
            names.append(f"model_{self.model_type}")
        
        if self.data_split:
            names.append(f"split_{self.data_split}")
        
        if self.environment:
            names.append(f"env_{self.environment}")
        
        # Generic fallbacks
        names.extend(["default", "fallback"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        return unique_names
    
    def to_dict(self) -> ConditionDict:
        """Convert condition to dictionary for serialization."""
        return {
            'name': self.name,
            'treatment': self.treatment,
            'model_type': self.model_type,
            'data_split': self.data_split,
            'environment': self.environment,
            'experiment_id': self.experiment_id,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'parent_conditions': self.parent_conditions,
            'tags': list(self.tags),
            'condition_hash': self._condition_hash
        }


class AdvancedConditionResolver:
    """Advanced condition resolution engine for complex experimental scenarios.
    
    Provides sophisticated condition-based styling resolution supporting 
    multi-variable experimental designs, hierarchical inheritance patterns,
    and enterprise-grade configuration management per F-002 and F-007 requirements.
    """
    
    def __init__(self, 
                 config_bridge: Optional[FigRegistryConfigBridge] = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600):
        """Initialize advanced condition resolver.
        
        Args:
            config_bridge: FigRegistry configuration bridge for Kedro integration
            enable_caching: Enable condition resolution caching for performance
            cache_ttl: Cache time-to-live in seconds for condition resolutions
        """
        self.config_bridge = config_bridge
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Thread-safe caches for performance
        self._condition_cache: Dict[str, ExperimentalCondition] = {}
        self._resolution_cache: Dict[str, Tuple[StyleDict, datetime]] = {}
        self._pattern_cache: Dict[str, List[str]] = {}
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self._resolution_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.debug("Initialized AdvancedConditionResolver with caching enabled: %s", enable_caching)
    
    def parse_condition_from_params(self, 
                                  params: ExperimentalParams, 
                                  condition_template: Optional[str] = None) -> ExperimentalCondition:
        """Parse experimental condition from pipeline parameters.
        
        Supports complex condition resolution from Kedro pipeline parameters
        enabling dynamic styling based on experimental context per F-005-RQ-004.
        
        Args:
            params: Pipeline parameters dictionary from Kedro context
            condition_template: Optional template for condition name generation
            
        Returns:
            Parsed ExperimentalCondition instance
        """
        start_time = time.perf_counter()
        
        # Extract condition components from parameters
        treatment = params.get('treatment', params.get('experimental_condition'))
        model_type = params.get('model_type', params.get('algorithm'))
        data_split = params.get('data_split', params.get('split', params.get('fold')))
        environment = params.get('environment', params.get('env', 'development'))
        experiment_id = params.get('experiment_id', params.get('run_id'))
        
        # Extract metrics if available
        metrics = {}
        for key, value in params.items():
            if key.endswith('_score') or key.endswith('_metric') or key in ['accuracy', 'precision', 'recall', 'f1']:
                if isinstance(value, (int, float)):
                    metrics[key] = value
        
        # Generate condition name using template or default pattern
        if condition_template:
            condition_name = condition_template.format(**params)
        else:
            name_parts = filter(None, [treatment, model_type, data_split])
            condition_name = "_".join(name_parts) if name_parts else "default_condition"
        
        # Extract additional metadata
        metadata = {k: v for k, v in params.items() 
                   if k.startswith('meta_') or k in ['description', 'notes', 'tags']}
        
        # Create experimental condition
        condition = ExperimentalCondition(
            name=condition_name,
            treatment=treatment,
            model_type=model_type,
            data_split=data_split,
            environment=environment,
            experiment_id=experiment_id,
            metrics=metrics,
            metadata=metadata
        )
        
        # Cache condition for reuse
        if self.enable_caching:
            with self._cache_lock:
                self._condition_cache[condition._condition_hash] = condition
        
        parse_time = time.perf_counter() - start_time
        self._resolution_times.append(parse_time)
        
        logger.debug("Parsed condition '%s' in %.2fms", condition_name, parse_time * 1000)
        return condition
    
    def resolve_style_hierarchy(self, 
                              condition: ExperimentalCondition,
                              style_config: Optional[Dict[str, StyleDict]] = None) -> StyleDict:
        """Resolve style using hierarchical inheritance patterns.
        
        Implements sophisticated style resolution with inheritance and fallback
        mechanisms for complex experimental design scenarios per F-002 requirements.
        
        Args:
            condition: Experimental condition to resolve styling for
            style_config: Optional style configuration override
            
        Returns:
            Resolved style dictionary for matplotlib application
        """
        start_time = time.perf_counter()
        
        # Check cache first if enabled
        cache_key = f"{condition._condition_hash}_{hash(str(style_config))}"
        if self.enable_caching:
            with self._cache_lock:
                if cache_key in self._resolution_cache:
                    cached_style, cached_time = self._resolution_cache[cache_key]
                    if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                        self._cache_hits += 1
                        logger.debug("Cache hit for condition '%s'", condition.name)
                        return cached_style.copy()
                    else:
                        # Remove expired cache entry
                        del self._resolution_cache[cache_key]
        
        self._cache_misses += 1
        
        # Get style configuration
        if style_config is None:
            if self.config_bridge:
                try:
                    merged_config = self.config_bridge.get_merged_config()
                    style_config = merged_config.styles or {}
                except Exception as e:
                    logger.warning("Failed to get merged config: %s", e)
                    style_config = {}
            else:
                style_config = {}
        
        # Initialize resolved style
        resolved_style = {}
        
        # Resolve style using hierarchical names
        for condition_name in condition.get_hierarchical_names():
            if condition_name in style_config:
                current_style = style_config[condition_name]
                if isinstance(current_style, dict):
                    # Deep merge styles with current taking precedence
                    resolved_style = self._deep_merge_styles(resolved_style, current_style)
                    logger.debug("Applied style from condition '%s'", condition_name)
                    break
        
        # Apply pattern-based matching if no exact match found
        if not resolved_style:
            resolved_style = self._resolve_pattern_styles(condition, style_config)
        
        # Apply metric-based styling adjustments
        resolved_style = self._apply_metric_styling(condition, resolved_style)
        
        # Apply default styling if still empty
        if not resolved_style:
            resolved_style = self._get_default_style(condition)
        
        # Validate and normalize style for matplotlib compatibility
        resolved_style = self._normalize_matplotlib_style(resolved_style)
        
        # Cache resolved style
        if self.enable_caching:
            with self._cache_lock:
                self._resolution_cache[cache_key] = (resolved_style.copy(), datetime.now())
        
        resolution_time = time.perf_counter() - start_time
        self._resolution_times.append(resolution_time)
        
        # Ensure performance requirement <1ms per F-002 SLA
        if resolution_time > 0.001:
            logger.warning("Style resolution took %.2fms, exceeding 1ms target", 
                         resolution_time * 1000)
        
        logger.debug("Resolved style for condition '%s' in %.2fms", 
                    condition.name, resolution_time * 1000)
        
        return resolved_style
    
    def _deep_merge_styles(self, base_style: StyleDict, override_style: StyleDict) -> StyleDict:
        """Deep merge style dictionaries with override precedence."""
        merged = base_style.copy()
        
        for key, value in override_style.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge_styles(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _resolve_pattern_styles(self, 
                              condition: ExperimentalCondition, 
                              style_config: Dict[str, StyleDict]) -> StyleDict:
        """Resolve styles using wildcard pattern matching."""
        resolved_style = {}
        
        # Check cached patterns first
        cache_key = f"patterns_{hash(str(list(style_config.keys())))}"
        if self.enable_caching and cache_key in self._pattern_cache:
            patterns = self._pattern_cache[cache_key]
        else:
            # Extract patterns from style config (keys with wildcards)
            patterns = [key for key in style_config.keys() 
                       if any(char in key for char in ['*', '?', '[', '{'])]
            if self.enable_caching:
                self._pattern_cache[cache_key] = patterns
        
        # Test patterns in order of specificity (longest first)
        patterns.sort(key=len, reverse=True)
        
        for pattern in patterns:
            if condition.matches_pattern(pattern):
                pattern_style = style_config[pattern]
                if isinstance(pattern_style, dict):
                    resolved_style = self._deep_merge_styles(resolved_style, pattern_style)
                    logger.debug("Applied pattern style from '%s'", pattern)
                    break
        
        return resolved_style
    
    def _apply_metric_styling(self, 
                            condition: ExperimentalCondition, 
                            base_style: StyleDict) -> StyleDict:
        """Apply metric-based styling adjustments."""
        if not condition.metrics:
            return base_style
        
        style = base_style.copy()
        
        # Performance-based color adjustments
        if 'accuracy' in condition.metrics:
            accuracy = condition.metrics['accuracy']
            if accuracy > 0.9:
                style['color'] = style.get('color', '#2e7d32')  # Dark green for high accuracy
            elif accuracy > 0.8:
                style['color'] = style.get('color', '#388e3c')  # Medium green
            elif accuracy > 0.7:
                style['color'] = style.get('color', '#ffa000')  # Orange for moderate
            else:
                style['color'] = style.get('color', '#d32f2f')  # Red for low accuracy
        
        # Line style based on confidence metrics
        if 'confidence' in condition.metrics:
            confidence = condition.metrics['confidence']
            if confidence > 0.9:
                style['linestyle'] = '-'  # Solid for high confidence
            elif confidence > 0.7:
                style['linestyle'] = '--'  # Dashed for medium confidence
            else:
                style['linestyle'] = ':'  # Dotted for low confidence
        
        # Marker size based on sample size
        if 'sample_size' in condition.metrics:
            sample_size = condition.metrics['sample_size']
            if sample_size > 10000:
                style['markersize'] = 8
            elif sample_size > 1000:
                style['markersize'] = 6
            else:
                style['markersize'] = 4
        
        return style
    
    def _get_default_style(self, condition: ExperimentalCondition) -> StyleDict:
        """Get default style for condition based on type and environment."""
        # Base default style
        default_style = {
            'color': '#1f77b4',  # Default matplotlib blue
            'linestyle': '-',
            'marker': 'o',
            'markersize': 6,
            'linewidth': 2,
            'alpha': 1.0
        }
        
        # Environment-specific adjustments
        if condition.environment == 'production':
            default_style.update({
                'color': '#2e7d32',  # Production green
                'linewidth': 2.5,
                'markersize': 7
            })
        elif condition.environment == 'staging':
            default_style.update({
                'color': '#ff9800',  # Staging orange
                'linestyle': '--'
            })
        elif condition.environment == 'development':
            default_style.update({
                'color': '#9c27b0',  # Development purple
                'linestyle': ':'
            })
        
        return default_style
    
    def _normalize_matplotlib_style(self, style: StyleDict) -> StyleDict:
        """Normalize style dictionary for matplotlib compatibility."""
        normalized = {}
        
        # Valid matplotlib style properties
        valid_props = {
            'color', 'c', 'linestyle', 'ls', 'linewidth', 'lw',
            'marker', 'markersize', 'ms', 'markerfacecolor', 'mfc',
            'markeredgecolor', 'mec', 'markeredgewidth', 'mew',
            'alpha', 'label', 'zorder', 'fillstyle', 'antialiased'
        }
        
        for key, value in style.items():
            if key in valid_props:
                normalized[key] = value
            elif key == 'line_style':  # Common alias
                normalized['linestyle'] = value
            elif key == 'line_width':  # Common alias
                normalized['linewidth'] = value
            elif key == 'marker_size':  # Common alias
                normalized['markersize'] = value
        
        return normalized
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get resolver performance metrics for monitoring."""
        with self._cache_lock:
            avg_resolution_time = np.mean(self._resolution_times) if self._resolution_times else 0
            cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            
            return {
                'average_resolution_time_ms': avg_resolution_time * 1000,
                'cache_hit_rate': cache_hit_rate,
                'total_resolutions': len(self._resolution_times),
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cached_conditions': len(self._condition_cache),
                'cached_resolutions': len(self._resolution_cache)
            }
    
    def clear_caches(self):
        """Clear all caches for memory management."""
        with self._cache_lock:
            self._condition_cache.clear()
            self._resolution_cache.clear()
            self._pattern_cache.clear()
            logger.debug("Cleared all condition resolver caches")


class DataTransformationHelper:
    """Advanced data transformation utilities for sophisticated visualization patterns.
    
    Provides production-ready data transformation functions supporting complex
    visualization scenarios with statistical analysis, time series processing,
    and multi-dimensional data preparation for automated figure generation.
    """
    
    @staticmethod
    def prepare_time_series_data(data: pd.DataFrame,
                                time_column: str,
                                value_columns: List[str],
                                resample_freq: Optional[str] = None,
                                aggregation: str = 'mean') -> pd.DataFrame:
        """Prepare time series data for advanced visualization.
        
        Args:
            data: Input DataFrame with time series data
            time_column: Name of the datetime column
            value_columns: List of columns to process
            resample_freq: Optional resampling frequency ('D', 'H', 'M', etc.)
            aggregation: Aggregation method for resampling
            
        Returns:
            Processed DataFrame optimized for time series visualization
        """
        # Ensure datetime column is properly typed
        processed_data = data.copy()
        processed_data[time_column] = pd.to_datetime(processed_data[time_column])
        processed_data = processed_data.set_index(time_column)
        
        # Filter to specified value columns
        processed_data = processed_data[value_columns]
        
        # Resample if frequency specified
        if resample_freq:
            agg_func = getattr(processed_data, aggregation, processed_data.mean)
            processed_data = agg_func().resample(resample_freq).agg(aggregation)
        
        # Add rolling statistics for trend analysis
        for col in value_columns:
            processed_data[f'{col}_ma7'] = processed_data[col].rolling(window=7, min_periods=1).mean()
            processed_data[f'{col}_ma30'] = processed_data[col].rolling(window=30, min_periods=1).mean()
            processed_data[f'{col}_std'] = processed_data[col].rolling(window=7, min_periods=1).std()
        
        # Reset index to make time column accessible
        processed_data = processed_data.reset_index()
        
        return processed_data
    
    @staticmethod
    def prepare_correlation_matrix(data: pd.DataFrame,
                                 method: str = 'pearson',
                                 min_periods: int = 30,
                                 exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare correlation matrix for advanced heatmap visualization.
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum periods for correlation calculation
            exclude_columns: Columns to exclude from correlation analysis
            
        Returns:
            Correlation matrix DataFrame
        """
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Exclude specified columns
        if exclude_columns:
            numeric_data = numeric_data.drop(columns=exclude_columns, errors='ignore')
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr(method=method, min_periods=min_periods)
        
        # Add metadata for visualization
        correlation_matrix.attrs['method'] = method
        correlation_matrix.attrs['n_features'] = len(correlation_matrix.columns)
        correlation_matrix.attrs['min_periods'] = min_periods
        
        return correlation_matrix
    
    @staticmethod
    def prepare_distribution_data(data: pd.DataFrame,
                                column: str,
                                groupby: Optional[str] = None,
                                bins: int = 30,
                                density: bool = True) -> Dict[str, Any]:
        """Prepare distribution data for advanced statistical visualization.
        
        Args:
            data: Input DataFrame
            column: Column to analyze
            groupby: Optional grouping column
            bins: Number of histogram bins
            density: Whether to normalize to density
            
        Returns:
            Dictionary with histogram data and statistical summaries
        """
        result = {
            'column': column,
            'bins': bins,
            'density': density,
            'statistics': {},
            'histogram_data': {}
        }
        
        if groupby:
            # Group-wise analysis
            for group_name, group_data in data.groupby(groupby):
                values = group_data[column].dropna()
                
                if len(values) > 0:
                    # Calculate histogram
                    counts, bin_edges = np.histogram(values, bins=bins, density=density)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    result['histogram_data'][group_name] = {
                        'counts': counts,
                        'bin_edges': bin_edges,
                        'bin_centers': bin_centers
                    }
                    
                    # Calculate statistics
                    result['statistics'][group_name] = {
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'skewness': stats.skew(values),
                        'kurtosis': stats.kurtosis(values),
                        'count': len(values),
                        'min': values.min(),
                        'max': values.max(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75)
                    }
        else:
            # Single distribution analysis
            values = data[column].dropna()
            
            if len(values) > 0:
                # Calculate histogram
                counts, bin_edges = np.histogram(values, bins=bins, density=density)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                result['histogram_data']['all'] = {
                    'counts': counts,
                    'bin_edges': bin_edges,
                    'bin_centers': bin_centers
                }
                
                # Calculate statistics
                result['statistics']['all'] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values),
                    'count': len(values),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75)
                }
        
        return result
    
    @staticmethod
    def prepare_performance_comparison(results_data: List[Dict[str, Any]],
                                     metric_columns: List[str],
                                     group_by: str = 'model_type') -> pd.DataFrame:
        """Prepare model performance comparison data for visualization.
        
        Args:
            results_data: List of dictionaries with model results
            metric_columns: List of performance metric column names
            group_by: Column to group results by
            
        Returns:
            DataFrame optimized for performance comparison visualization
        """
        # Convert to DataFrame
        df = pd.DataFrame(results_data)
        
        # Ensure metric columns are numeric
        for col in metric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate summary statistics by group
        summary_stats = []
        
        for group_name, group_data in df.groupby(group_by):
            for metric in metric_columns:
                if metric in group_data.columns:
                    values = group_data[metric].dropna()
                    
                    if len(values) > 0:
                        summary_stats.append({
                            group_by: group_name,
                            'metric': metric,
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'count': len(values),
                            'ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
                            'ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values))
                        })
        
        return pd.DataFrame(summary_stats)


class StatisticalAnalysisHelper:
    """Advanced statistical analysis utilities for reporting pipeline visualizations.
    
    Provides sophisticated statistical analysis functions supporting enterprise-grade
    reporting pipelines with hypothesis testing, effect size calculations, 
    confidence intervals, and advanced statistical summaries for automated
    visualization in production data science workflows.
    """
    
    @staticmethod
    def calculate_effect_sizes(group1: np.ndarray, 
                             group2: np.ndarray,
                             effect_types: List[str] = None) -> Dict[str, float]:
        """Calculate multiple effect size measures for group comparisons.
        
        Args:
            group1: First group data
            group2: Second group data  
            effect_types: List of effect size types to calculate
            
        Returns:
            Dictionary with calculated effect sizes
        """
        if effect_types is None:
            effect_types = ['cohens_d', 'hedges_g', 'glass_delta', 'cliff_delta']
        
        results = {}
        
        # Remove NaN values
        group1_clean = np.array(group1)[~np.isnan(group1)]
        group2_clean = np.array(group2)[~np.isnan(group2)]
        
        if len(group1_clean) == 0 or len(group2_clean) == 0:
            return {effect_type: np.nan for effect_type in effect_types}
        
        # Cohen's d
        if 'cohens_d' in effect_types:
            pooled_std = np.sqrt(((len(group1_clean) - 1) * np.var(group1_clean) + 
                                (len(group2_clean) - 1) * np.var(group2_clean)) / 
                               (len(group1_clean) + len(group2_clean) - 2))
            if pooled_std > 0:
                results['cohens_d'] = (np.mean(group1_clean) - np.mean(group2_clean)) / pooled_std
            else:
                results['cohens_d'] = 0.0
        
        # Hedges' g (bias-corrected Cohen's d)
        if 'hedges_g' in effect_types and 'cohens_d' in results:
            correction_factor = 1 - (3 / (4 * (len(group1_clean) + len(group2_clean)) - 9))
            results['hedges_g'] = results['cohens_d'] * correction_factor
        
        # Glass's delta
        if 'glass_delta' in effect_types:
            if np.std(group2_clean) > 0:
                results['glass_delta'] = (np.mean(group1_clean) - np.mean(group2_clean)) / np.std(group2_clean)
            else:
                results['glass_delta'] = 0.0
        
        # Cliff's delta (non-parametric effect size)
        if 'cliff_delta' in effect_types:
            pairs = 0
            favorable = 0
            
            for x1 in group1_clean:
                for x2 in group2_clean:
                    pairs += 1
                    if x1 > x2:
                        favorable += 1
                    elif x1 < x2:
                        favorable -= 1
            
            results['cliff_delta'] = favorable / pairs if pairs > 0 else 0.0
        
        return results
    
    @staticmethod
    def perform_comprehensive_comparison(data: pd.DataFrame,
                                       value_column: str,
                                       group_column: str,
                                       alpha: float = 0.05) -> Dict[str, Any]:
        """Perform comprehensive statistical comparison between groups.
        
        Args:
            data: DataFrame with data to compare
            value_column: Column containing values to compare
            group_column: Column containing group labels
            alpha: Significance level for tests
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        results = {
            'groups': {},
            'pairwise_comparisons': {},
            'overall_tests': {},
            'effect_sizes': {},
            'summary_statistics': {}
        }
        
        # Group-wise descriptive statistics
        for group_name, group_data in data.groupby(group_column):
            values = group_data[value_column].dropna()
            
            if len(values) > 0:
                results['groups'][group_name] = {
                    'n': len(values),
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'var': values.var(),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'iqr': values.quantile(0.75) - values.quantile(0.25),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values),
                    'shapiro_p': stats.shapiro(values)[1] if len(values) <= 5000 else np.nan
                }
        
        # Overall tests
        groups = [group[value_column].dropna() for _, group in data.groupby(group_column)]
        group_names = list(data[group_column].unique())
        
        if len(groups) >= 2:
            # ANOVA (parametric)
            try:
                f_stat, anova_p = stats.f_oneway(*groups)
                results['overall_tests']['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': anova_p,
                    'significant': anova_p < alpha
                }
            except Exception as e:
                logger.warning("ANOVA test failed: %s", e)
                results['overall_tests']['anova'] = {'error': str(e)}
            
            # Kruskal-Wallis (non-parametric)
            try:
                h_stat, kw_p = stats.kruskal(*groups)
                results['overall_tests']['kruskal_wallis'] = {
                    'h_statistic': h_stat,
                    'p_value': kw_p,
                    'significant': kw_p < alpha
                }
            except Exception as e:
                logger.warning("Kruskal-Wallis test failed: %s", e)
                results['overall_tests']['kruskal_wallis'] = {'error': str(e)}
        
        # Pairwise comparisons
        for i, group1_name in enumerate(group_names):
            for j, group2_name in enumerate(group_names[i+1:], i+1):
                group1_data = results['groups'][group1_name]
                group2_data = results['groups'][group2_name]
                
                group1_values = data[data[group_column] == group1_name][value_column].dropna()
                group2_values = data[data[group_column] == group2_name][value_column].dropna()
                
                comparison_key = f"{group1_name}_vs_{group2_name}"
                
                # T-test
                try:
                    t_stat, t_p = stats.ttest_ind(group1_values, group2_values)
                    results['pairwise_comparisons'][comparison_key] = {
                        't_test': {
                            't_statistic': t_stat,
                            'p_value': t_p,
                            'significant': t_p < alpha
                        }
                    }
                except Exception as e:
                    results['pairwise_comparisons'][comparison_key] = {
                        't_test': {'error': str(e)}
                    }
                
                # Mann-Whitney U test
                try:
                    u_stat, u_p = stats.mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                    results['pairwise_comparisons'][comparison_key]['mann_whitney'] = {
                        'u_statistic': u_stat,
                        'p_value': u_p,
                        'significant': u_p < alpha
                    }
                except Exception as e:
                    if 'mann_whitney' not in results['pairwise_comparisons'][comparison_key]:
                        results['pairwise_comparisons'][comparison_key]['mann_whitney'] = {'error': str(e)}
                
                # Effect sizes
                effect_sizes = StatisticalAnalysisHelper.calculate_effect_sizes(
                    group1_values.values, group2_values.values
                )
                results['effect_sizes'][comparison_key] = effect_sizes
        
        # Summary statistics
        all_values = data[value_column].dropna()
        results['summary_statistics'] = {
            'total_n': len(all_values),
            'n_groups': len(group_names),
            'overall_mean': all_values.mean(),
            'overall_std': all_values.std(),
            'between_group_variance': data.groupby(group_column)[value_column].mean().var(),
            'within_group_variance': np.mean([group.var() for group in groups if len(group) > 1])
        }
        
        return results
    
    @staticmethod
    def calculate_confidence_intervals(data: np.ndarray,
                                     confidence_level: float = 0.95,
                                     method: str = 'normal') -> Dict[str, float]:
        """Calculate confidence intervals using different methods.
        
        Args:
            data: Input data array
            confidence_level: Confidence level (0.95 for 95% CI)
            method: Method ('normal', 'bootstrap', 't_distribution')
            
        Returns:
            Dictionary with confidence interval bounds
        """
        data_clean = np.array(data)[~np.isnan(data)]
        
        if len(data_clean) == 0:
            return {'lower': np.nan, 'upper': np.nan, 'method': method}
        
        alpha = 1 - confidence_level
        mean = np.mean(data_clean)
        
        if method == 'normal':
            # Normal approximation
            std_error = np.std(data_clean) / np.sqrt(len(data_clean))
            z_score = stats.norm.ppf(1 - alpha/2)
            margin_error = z_score * std_error
            
            return {
                'lower': mean - margin_error,
                'upper': mean + margin_error,
                'method': method,
                'margin_error': margin_error
            }
        
        elif method == 't_distribution':
            # T-distribution (better for small samples)
            std_error = np.std(data_clean, ddof=1) / np.sqrt(len(data_clean))
            t_score = stats.t.ppf(1 - alpha/2, df=len(data_clean)-1)
            margin_error = t_score * std_error
            
            return {
                'lower': mean - margin_error,
                'upper': mean + margin_error,
                'method': method,
                'margin_error': margin_error,
                'degrees_freedom': len(data_clean) - 1
            }
        
        elif method == 'bootstrap':
            # Bootstrap confidence interval
            n_bootstrap = 10000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data_clean, size=len(data_clean), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            return {
                'lower': np.percentile(bootstrap_means, lower_percentile),
                'upper': np.percentile(bootstrap_means, upper_percentile),
                'method': method,
                'n_bootstrap': n_bootstrap
            }
        
        else:
            raise ValueError(f"Unknown confidence interval method: {method}")


class ConfigurationManager:
    """Production-ready configuration management for multi-environment deployment.
    
    Provides enterprise-grade configuration management utilities supporting
    sophisticated multi-environment deployment scenarios with validation,
    caching, monitoring, and integration with Kedro's configuration system
    per F-007 requirements for production data science workflows.
    """
    
    def __init__(self, 
                 environments: List[str] = None,
                 config_bridge: Optional[FigRegistryConfigBridge] = None):
        """Initialize configuration manager.
        
        Args:
            environments: List of supported environments
            config_bridge: Optional FigRegistry configuration bridge
        """
        self.environments = environments or ['development', 'staging', 'production']
        self.config_bridge = config_bridge
        
        # Environment-specific configuration cache
        self._environment_configs: Dict[str, Dict[str, Any]] = {}
        self._config_lock = threading.RLock()
        
        # Configuration monitoring
        self._config_access_log: List[Dict[str, Any]] = []
        self._validation_errors: List[Dict[str, Any]] = []
        
        logger.debug("Initialized ConfigurationManager for environments: %s", self.environments)
    
    def get_environment_config(self, 
                             environment: str,
                             config_section: Optional[str] = None,
                             use_cache: bool = True) -> Dict[str, Any]:
        """Get configuration for specific environment with caching.
        
        Args:
            environment: Target environment name
            config_section: Optional specific configuration section
            use_cache: Whether to use cached configuration
            
        Returns:
            Environment-specific configuration dictionary
        """
        if environment not in self.environments:
            raise ValueError(f"Unknown environment: {environment}. Supported: {self.environments}")
        
        cache_key = f"{environment}_{config_section or 'all'}"
        
        # Check cache first
        if use_cache:
            with self._config_lock:
                if cache_key in self._environment_configs:
                    self._log_config_access(environment, config_section, 'cache_hit')
                    return self._environment_configs[cache_key].copy()
        
        # Load configuration
        start_time = time.perf_counter()
        
        try:
            if self.config_bridge:
                # Use Kedro configuration bridge
                bridge = FigRegistryConfigBridge(
                    config_loader=self.config_bridge.config_loader,
                    environment=environment
                )
                merged_config = bridge.get_merged_config()
                config_dict = merged_config.dict()
                
                if config_section:
                    config_dict = config_dict.get(config_section, {})
            else:
                # Fallback to basic environment configuration
                config_dict = self._load_basic_environment_config(environment, config_section)
            
            # Cache configuration
            if use_cache:
                with self._config_lock:
                    self._environment_configs[cache_key] = config_dict.copy()
            
            load_time = time.perf_counter() - start_time
            self._log_config_access(environment, config_section, 'loaded', load_time)
            
            return config_dict
            
        except Exception as e:
            error_info = {
                'timestamp': datetime.now(),
                'environment': environment,
                'config_section': config_section,
                'error': str(e),
                'error_type': type(e).__name__
            }
            self._validation_errors.append(error_info)
            
            logger.error("Failed to load config for environment %s: %s", environment, e)
            raise ConfigurationMergeError(f"Configuration loading failed for {environment}: {str(e)}")
    
    def _load_basic_environment_config(self, 
                                     environment: str, 
                                     config_section: Optional[str]) -> Dict[str, Any]:
        """Load basic environment configuration without Kedro bridge."""
        # Implement basic environment-specific configuration loading
        base_config = {
            'environment': environment,
            'deployment_tier': self._get_deployment_tier(environment),
            'performance_targets': self._get_performance_targets(environment),
            'logging_level': self._get_logging_level(environment),
            'cache_settings': self._get_cache_settings(environment)
        }
        
        if config_section:
            return base_config.get(config_section, {})
        
        return base_config
    
    def _get_deployment_tier(self, environment: str) -> str:
        """Get deployment tier for environment."""
        tier_mapping = {
            'development': 'dev',
            'staging': 'staging', 
            'production': 'prod'
        }
        return tier_mapping.get(environment, 'dev')
    
    def _get_performance_targets(self, environment: str) -> Dict[str, float]:
        """Get performance targets for environment."""
        if environment == 'production':
            return {
                'style_resolution_ms': 0.5,
                'figure_save_ms': 50,
                'config_load_ms': 25
            }
        elif environment == 'staging':
            return {
                'style_resolution_ms': 1.0,
                'figure_save_ms': 100,
                'config_load_ms': 50
            }
        else:  # development
            return {
                'style_resolution_ms': 2.0,
                'figure_save_ms': 200,
                'config_load_ms': 100
            }
    
    def _get_logging_level(self, environment: str) -> str:
        """Get appropriate logging level for environment."""
        level_mapping = {
            'development': 'DEBUG',
            'staging': 'INFO',
            'production': 'WARNING'
        }
        return level_mapping.get(environment, 'INFO')
    
    def _get_cache_settings(self, environment: str) -> Dict[str, Any]:
        """Get cache settings for environment."""
        if environment == 'production':
            return {
                'enable_caching': True,
                'cache_ttl': 3600,  # 1 hour
                'max_cache_size': 1000
            }
        elif environment == 'staging':
            return {
                'enable_caching': True,
                'cache_ttl': 1800,  # 30 minutes
                'max_cache_size': 500
            }
        else:  # development
            return {
                'enable_caching': False,
                'cache_ttl': 300,  # 5 minutes
                'max_cache_size': 100
            }
    
    def _log_config_access(self, 
                          environment: str, 
                          config_section: Optional[str],
                          access_type: str,
                          load_time: Optional[float] = None):
        """Log configuration access for monitoring."""
        log_entry = {
            'timestamp': datetime.now(),
            'environment': environment,
            'config_section': config_section,
            'access_type': access_type,
            'load_time_ms': load_time * 1000 if load_time else None
        }
        
        self._config_access_log.append(log_entry)
        
        # Keep only recent entries (last 1000)
        if len(self._config_access_log) > 1000:
            self._config_access_log = self._config_access_log[-1000:]
    
    def validate_environment_configs(self) -> Dict[str, List[str]]:
        """Validate configurations across all environments.
        
        Returns:
            Dictionary mapping environments to validation error messages
        """
        validation_results = {}
        
        for environment in self.environments:
            errors = []
            
            try:
                config = self.get_environment_config(environment, use_cache=False)
                
                # Validate required sections
                required_sections = ['environment', 'deployment_tier']
                for section in required_sections:
                    if section not in config:
                        errors.append(f"Missing required section: {section}")
                
                # Environment-specific validations
                if environment == 'production':
                    # Production requires stricter validation
                    if 'performance_targets' not in config:
                        errors.append("Production environment missing performance targets")
                    
                    performance_targets = config.get('performance_targets', {})
                    if performance_targets.get('style_resolution_ms', 0) > 1.0:
                        errors.append("Production style resolution target too high")
                
                validation_results[environment] = errors
                
            except Exception as e:
                validation_results[environment] = [f"Configuration load error: {str(e)}"]
        
        return validation_results
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get configuration monitoring data for observability."""
        with self._config_lock:
            # Calculate access statistics
            total_accesses = len(self._config_access_log)
            cache_hits = sum(1 for entry in self._config_access_log if entry['access_type'] == 'cache_hit')
            cache_hit_rate = cache_hits / total_accesses if total_accesses > 0 else 0
            
            # Calculate average load times
            load_times = [entry['load_time_ms'] for entry in self._config_access_log 
                         if entry['load_time_ms'] is not None]
            avg_load_time = np.mean(load_times) if load_times else 0
            
            # Environment access distribution
            env_distribution = defaultdict(int)
            for entry in self._config_access_log:
                env_distribution[entry['environment']] += 1
            
            return {
                'total_config_accesses': total_accesses,
                'cache_hit_rate': cache_hit_rate,
                'average_load_time_ms': avg_load_time,
                'environment_distribution': dict(env_distribution),
                'cached_configs': len(self._environment_configs),
                'validation_errors': len(self._validation_errors),
                'supported_environments': self.environments,
                'recent_errors': self._validation_errors[-10:] if self._validation_errors else []
            }
    
    def clear_cache(self, environment: Optional[str] = None):
        """Clear configuration cache for specified environment or all."""
        with self._config_lock:
            if environment:
                # Clear specific environment
                keys_to_remove = [key for key in self._environment_configs.keys() 
                                if key.startswith(f"{environment}_")]
                for key in keys_to_remove:
                    del self._environment_configs[key]
                logger.debug("Cleared configuration cache for environment: %s", environment)
            else:
                # Clear all caches
                self._environment_configs.clear()
                logger.debug("Cleared all configuration caches")


# Convenience functions for common operations

def create_condition_resolver(config_bridge: Optional[FigRegistryConfigBridge] = None,
                            **kwargs) -> AdvancedConditionResolver:
    """Create an advanced condition resolver with optimal settings.
    
    Args:
        config_bridge: Optional FigRegistry configuration bridge
        **kwargs: Additional resolver configuration
        
    Returns:
        Configured AdvancedConditionResolver instance
    """
    return AdvancedConditionResolver(config_bridge=config_bridge, **kwargs)


def create_configuration_manager(environments: List[str] = None,
                               config_bridge: Optional[FigRegistryConfigBridge] = None) -> ConfigurationManager:
    """Create a configuration manager with enterprise settings.
    
    Args:
        environments: List of supported environments
        config_bridge: Optional FigRegistry configuration bridge
        
    Returns:
        Configured ConfigurationManager instance
    """
    return ConfigurationManager(environments=environments, config_bridge=config_bridge)


def prepare_experimental_data(data: pd.DataFrame,
                            condition_params: ExperimentalParams,
                            analysis_type: str = 'time_series',
                            **kwargs) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Prepare experimental data for advanced visualization.
    
    Convenience function that combines data transformation with condition
    resolution for streamlined preparation of complex experimental datasets.
    
    Args:
        data: Input experimental data DataFrame
        condition_params: Experimental condition parameters
        analysis_type: Type of analysis ('time_series', 'correlation', 'distribution', 'comparison')
        **kwargs: Additional parameters for specific analysis types
        
    Returns:
        Processed data optimized for the specified analysis type
    """
    helper = DataTransformationHelper()
    
    if analysis_type == 'time_series':
        return helper.prepare_time_series_data(
            data=data,
            time_column=kwargs.get('time_column', 'timestamp'),
            value_columns=kwargs.get('value_columns', ['value']),
            resample_freq=kwargs.get('resample_freq'),
            aggregation=kwargs.get('aggregation', 'mean')
        )
    
    elif analysis_type == 'correlation':
        return helper.prepare_correlation_matrix(
            data=data,
            method=kwargs.get('method', 'pearson'),
            min_periods=kwargs.get('min_periods', 30),
            exclude_columns=kwargs.get('exclude_columns')
        )
    
    elif analysis_type == 'distribution':
        return helper.prepare_distribution_data(
            data=data,
            column=kwargs.get('column', 'value'),
            groupby=kwargs.get('groupby'),
            bins=kwargs.get('bins', 30),
            density=kwargs.get('density', True)
        )
    
    elif analysis_type == 'comparison':
        # Convert DataFrame to list of dictionaries for comparison
        results_data = data.to_dict('records')
        return helper.prepare_performance_comparison(
            results_data=results_data,
            metric_columns=kwargs.get('metric_columns', ['accuracy', 'precision', 'recall']),
            group_by=kwargs.get('group_by', 'model_type')
        )
    
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def calculate_advanced_statistics(data: pd.DataFrame,
                                value_column: str,
                                group_column: Optional[str] = None,
                                confidence_level: float = 0.95) -> Dict[str, Any]:
    """Calculate comprehensive statistical analysis for reporting.
    
    Convenience function providing complete statistical analysis including
    effect sizes, confidence intervals, and hypothesis testing for
    enterprise reporting pipeline visualizations.
    
    Args:
        data: Input DataFrame with experimental results
        value_column: Column containing values to analyze
        group_column: Optional column for group comparisons
        confidence_level: Confidence level for intervals and tests
        
    Returns:
        Comprehensive statistical analysis results
    """
    helper = StatisticalAnalysisHelper()
    
    if group_column and group_column in data.columns:
        # Group comparison analysis
        comparison_results = helper.perform_comprehensive_comparison(
            data=data,
            value_column=value_column,
            group_column=group_column,
            alpha=1 - confidence_level
        )
        
        # Add confidence intervals for each group
        for group_name, group_stats in comparison_results['groups'].items():
            group_data = data[data[group_column] == group_name][value_column].dropna()
            
            if len(group_data) > 0:
                ci_results = helper.calculate_confidence_intervals(
                    data=group_data.values,
                    confidence_level=confidence_level,
                    method='t_distribution' if len(group_data) < 30 else 'normal'
                )
                group_stats['confidence_interval'] = ci_results
        
        return comparison_results
    
    else:
        # Single sample analysis
        values = data[value_column].dropna()
        
        if len(values) == 0:
            return {'error': 'No valid data found'}
        
        # Basic descriptive statistics
        basic_stats = {
            'n': len(values),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
        
        # Confidence intervals
        confidence_intervals = {}
        for method in ['normal', 't_distribution', 'bootstrap']:
            try:
                ci_results = helper.calculate_confidence_intervals(
                    data=values.values,
                    confidence_level=confidence_level,
                    method=method
                )
                confidence_intervals[method] = ci_results
            except Exception as e:
                confidence_intervals[method] = {'error': str(e)}
        
        return {
            'basic_statistics': basic_stats,
            'confidence_intervals': confidence_intervals,
            'normality_test': {
                'shapiro_p': stats.shapiro(values)[1] if len(values) <= 5000 else None
            }
        }