"""FigRegistry Kedro Dataset Integration.

This module implements FigureDataSet, a custom Kedro AbstractDataSet that bridges 
matplotlib figure objects with FigRegistry's automated styling and versioning system. 
The dataset intercepts figure save operations within Kedro's catalog workflow, 
automatically applying condition-based styling through FigRegistry's get_style() API 
and managing file persistence through save_figure().

The implementation maintains full compatibility with Kedro's versioning system while 
eliminating manual styling code from pipeline nodes and ensuring consistent figure 
outputs across all workflow stages.
"""

import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import copy

# Core dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    matplotlib_available = True
except ImportError:
    warnings.warn(
        "Matplotlib not found. Please ensure matplotlib>=3.9.0 is installed.",
        ImportWarning
    )
    matplotlib_available = False
    Figure = None

try:
    from kedro.io import AbstractDataSet
    from kedro.io.core import get_filepath_str, get_protocol_and_path
    kedro_available = True
except ImportError:
    warnings.warn(
        "Kedro not found. Please ensure kedro>=0.18.0,<0.20.0 is installed.",
        ImportWarning
    )
    kedro_available = False
    AbstractDataSet = object

try:
    import figregistry
    from figregistry import get_style, save_figure
    figregistry_available = True
except ImportError:
    warnings.warn(
        "FigRegistry not found. Please ensure figregistry>=0.3.0 is installed.",
        ImportWarning
    )
    figregistry_available = False
    get_style = None
    save_figure = None

# Internal dependencies
from .config import FigRegistryConfigBridge, get_bridge_instance, init_config

logger = logging.getLogger(__name__)

class FigureDataSetError(Exception):
    """Custom exception for FigureDataSet operations."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class StyleResolutionCache:
    """Thread-safe cache for style resolution to meet <1ms lookup requirement."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._lock = threading.RLock()
        self._stats = {'hits': 0, 'misses': 0}
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached style with LRU ordering."""
        with self._lock:
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                self._stats['hits'] += 1
                return copy.deepcopy(self._cache[cache_key])
            
            self._stats['misses'] += 1
            return None
    
    def put(self, cache_key: str, style: Dict[str, Any]):
        """Cache style with LRU eviction."""
        with self._lock:
            if cache_key in self._cache:
                # Update existing entry
                self._access_order.remove(cache_key)
            elif len(self._cache) >= self._max_size:
                # Evict least recently used
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
            
            self._cache[cache_key] = copy.deepcopy(style)
            self._access_order.append(cache_key)
    
    def clear(self):
        """Clear cache and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = {'hits': 0, 'misses': 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            hit_rate = self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) if (self._stats['hits'] + self._stats['misses']) > 0 else 0
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hit_rate': hit_rate,
                **self._stats
            }


class FigureDataSet(AbstractDataSet):
    """Custom Kedro AbstractDataSet for matplotlib figures with FigRegistry integration.
    
    This dataset implementation bridges matplotlib figure objects with FigRegistry's 
    automated styling and versioning system, enabling seamless integration within 
    Kedro's catalog-based workflow management.
    
    Features:
    - Automatic condition-based styling through FigRegistry
    - Compatibility with Kedro's versioning and environment management
    - Thread-safe operation for parallel pipeline execution
    - <5% performance overhead compared to manual matplotlib operations
    - Configuration-driven styling with runtime parameter resolution
    
    Configuration Parameters:
        filepath (str): Output file path for the figure
        purpose (str, optional): Output categorization - 'exploratory', 'presentation', 
            or 'publication'. Defaults to 'exploratory'
        condition_param (str, optional): Parameter name for dynamic condition resolution 
            from pipeline context. If provided, the dataset will resolve the condition 
            value from the pipeline parameters
        style_params (Dict[str, Any], optional): Dataset-specific styling overrides 
            that take precedence over condition-based styles
        save_args (Dict[str, Any], optional): Additional arguments passed to matplotlib's 
            savefig method (e.g., dpi, format, bbox_inches)
        load_args (Dict[str, Any], optional): Additional arguments for figure loading 
            (currently not used as figures are typically write-only)
        version (str, optional): Dataset version for Kedro's versioning system
    
    Example:
        In catalog.yml:
        ```yaml
        experiment_plots:
          type: figregistry_kedro.FigureDataSet
          filepath: data/08_reporting/experiment_results.png
          purpose: presentation
          condition_param: experiment_condition
          style_params:
            figure.dpi: 300
            figure.facecolor: white
          save_args:
            bbox_inches: tight
            transparent: false
        ```
    """
    
    # Class-level style cache for performance optimization
    _style_cache = StyleResolutionCache(max_size=1000)
    _performance_stats = {
        'total_saves': 0,
        'total_style_time': 0.0,
        'total_save_time': 0.0,
        'cache_stats': {}
    }
    _stats_lock = threading.RLock()
    
    def __init__(self,
                 filepath: str,
                 purpose: str = "exploratory",
                 condition_param: Optional[str] = None,
                 style_params: Optional[Dict[str, Any]] = None,
                 save_args: Optional[Dict[str, Any]] = None,
                 load_args: Optional[Dict[str, Any]] = None,
                 version: Optional[str] = None,
                 credentials: Optional[Dict[str, Any]] = None,
                 fs_args: Optional[Dict[str, Any]] = None):
        """Initialize FigureDataSet with configuration parameters.
        
        Args:
            filepath: Output file path for the figure
            purpose: Output categorization ('exploratory', 'presentation', 'publication')
            condition_param: Parameter name for dynamic condition resolution
            style_params: Dataset-specific styling overrides
            save_args: Additional arguments for matplotlib savefig
            load_args: Additional arguments for figure loading (future use)
            version: Dataset version for Kedro versioning
            credentials: Credentials for accessing file systems (future use)
            fs_args: File system arguments (future use)
        
        Raises:
            FigureDataSetError: If required dependencies are not available
        """
        # Validate dependencies
        if not matplotlib_available:
            raise FigureDataSetError(
                "Matplotlib is required but not available. "
                "Please install matplotlib>=3.9.0"
            )
        
        if not kedro_available:
            raise FigureDataSetError(
                "Kedro is required but not available. "
                "Please install kedro>=0.18.0,<0.20.0"
            )
        
        if not figregistry_available:
            raise FigureDataSetError(
                "FigRegistry is required but not available. "
                "Please install figregistry>=0.3.0"
            )
        
        # Store configuration parameters
        self._filepath = filepath
        self._purpose = purpose
        self._condition_param = condition_param
        self._style_params = style_params or {}
        self._save_args = save_args or {}
        self._load_args = load_args or {}
        self._version = version
        
        # Validate purpose parameter
        valid_purposes = {'exploratory', 'presentation', 'publication'}
        if purpose not in valid_purposes:
            logger.warning(
                f"Purpose '{purpose}' not in recommended values {valid_purposes}. "
                f"Using as-is for FigRegistry condition resolution."
            )
        
        # Set up performance monitoring
        self._operation_stats = {
            'saves': 0,
            'style_resolution_time': 0.0,
            'save_operation_time': 0.0
        }
        
        # Thread-local storage for pipeline context
        self._context = threading.local()
        
        logger.debug(
            f"Initialized FigureDataSet: filepath={filepath}, purpose={purpose}, "
            f"condition_param={condition_param}, style_params={bool(style_params)}"
        )
    
    def _get_pipeline_context(self) -> Dict[str, Any]:
        """Get current pipeline context from thread-local storage.
        
        Returns:
            Pipeline context dictionary, empty if not available
        """
        return getattr(self._context, 'pipeline_params', {})
    
    def _set_pipeline_context(self, context: Dict[str, Any]):
        """Set pipeline context in thread-local storage.
        
        Args:
            context: Pipeline context dictionary
        """
        self._context.pipeline_params = context
    
    def _generate_cache_key(self, condition: str, style_params: Dict[str, Any]) -> str:
        """Generate cache key for style resolution.
        
        Args:
            condition: Condition string for style lookup
            style_params: Additional style parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        import json
        
        cache_data = {
            'condition': condition,
            'style_params': style_params,
            'purpose': self._purpose
        }
        
        return hashlib.md5(
            json.dumps(cache_data, sort_keys=True).encode()
        ).hexdigest()
    
    def _resolve_condition(self, pipeline_params: Optional[Dict[str, Any]] = None) -> str:
        """Resolve condition value for style lookup.
        
        Args:
            pipeline_params: Pipeline parameters for condition resolution
            
        Returns:
            Resolved condition string
        """
        # If no condition_param specified, use purpose as condition
        if not self._condition_param:
            return self._purpose
        
        # Get pipeline parameters from context or parameter
        params = pipeline_params or self._get_pipeline_context()
        
        # Resolve condition from parameters
        if self._condition_param in params:
            condition_value = params[self._condition_param]
            logger.debug(f"Resolved condition '{self._condition_param}' = '{condition_value}'")
            return str(condition_value)
        
        # Fallback to purpose if condition parameter not found
        logger.debug(
            f"Condition parameter '{self._condition_param}' not found in pipeline params. "
            f"Falling back to purpose: '{self._purpose}'"
        )
        return self._purpose
    
    def _get_figure_style(self, condition: str) -> Dict[str, Any]:
        """Get styling parameters for the given condition.
        
        Args:
            condition: Condition string for style lookup
            
        Returns:
            Style dictionary for matplotlib application
        """
        start_time = time.perf_counter()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(condition, self._style_params)
            
            # Check cache first
            cached_style = self._style_cache.get(cache_key)
            if cached_style is not None:
                style_time = time.perf_counter() - start_time
                logger.debug(f"Style cache hit for condition '{condition}' in {style_time*1000:.2f}ms")
                return cached_style
            
            # Get base style from FigRegistry
            base_style = {}
            if figregistry_available and get_style:
                try:
                    base_style = get_style(condition) or {}
                    logger.debug(f"Retrieved FigRegistry style for condition '{condition}': {len(base_style)} parameters")
                except Exception as e:
                    logger.warning(f"Failed to get FigRegistry style for condition '{condition}': {e}")
                    base_style = {}
            
            # Merge with dataset-specific style parameters
            final_style = {**base_style, **self._style_params}
            
            # Cache the result
            self._style_cache.put(cache_key, final_style)
            
            style_time = time.perf_counter() - start_time
            
            # Ensure performance requirement of <1ms
            if style_time > 0.001:  # 1ms
                logger.warning(f"Style resolution took {style_time*1000:.2f}ms, exceeding 1ms target")
            
            # Update performance stats
            self._operation_stats['style_resolution_time'] += style_time
            
            logger.debug(f"Style resolution completed in {style_time*1000:.2f}ms for condition '{condition}'")
            return final_style
            
        except Exception as e:
            style_time = time.perf_counter() - start_time
            error_msg = f"Style resolution failed for condition '{condition}': {str(e)}"
            logger.error(error_msg)
            raise FigureDataSetError(error_msg, e)
    
    def _apply_style_to_figure(self, figure: Figure, style: Dict[str, Any]):
        """Apply styling parameters to matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
            style: Style parameters dictionary
        """
        if not style:
            return
        
        try:
            # Apply rcParams-style parameters
            with matplotlib.rc_context(style):
                # Update figure properties that can be changed after creation
                if 'figure.facecolor' in style:
                    figure.patch.set_facecolor(style['figure.facecolor'])
                
                if 'figure.edgecolor' in style:
                    figure.patch.set_edgecolor(style['figure.edgecolor'])
                
                if 'figure.dpi' in style:
                    figure.set_dpi(float(style['figure.dpi']))
                
                # Apply any custom properties
                for key, value in style.items():
                    if key.startswith('figure.') and hasattr(figure, key.split('.', 1)[1]):
                        try:
                            setattr(figure, key.split('.', 1)[1], value)
                        except Exception as e:
                            logger.debug(f"Could not set figure property {key} to {value}: {e}")
            
            logger.debug(f"Applied {len(style)} style parameters to figure")
            
        except Exception as e:
            logger.warning(f"Failed to apply some style parameters: {e}")
    
    def _save_figure_with_figregistry(self, figure: Figure, filepath: str, 
                                    condition: str, style: Dict[str, Any]) -> str:
        """Save figure using FigRegistry's save_figure function.
        
        Args:
            figure: Matplotlib figure object
            filepath: Output file path
            condition: Condition string for FigRegistry
            style: Style parameters
            
        Returns:
            Actual saved file path
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare save arguments
            save_kwargs = {
                **self._save_args,
                'figure': figure,
                'filepath': filepath,
                'condition': condition
            }
            
            # Use FigRegistry save_figure if available
            if figregistry_available and save_figure:
                try:
                    actual_path = save_figure(**save_kwargs)
                    logger.debug(f"Saved figure using FigRegistry to: {actual_path}")
                    return actual_path
                except Exception as e:
                    logger.warning(f"FigRegistry save_figure failed, falling back to matplotlib: {e}")
            
            # Fallback to direct matplotlib save
            figure.savefig(filepath, **self._save_args)
            logger.debug(f"Saved figure using matplotlib to: {filepath}")
            return filepath
            
        except Exception as e:
            save_time = time.perf_counter() - start_time
            error_msg = f"Figure save failed after {save_time*1000:.2f}ms: {str(e)}"
            logger.error(error_msg)
            raise FigureDataSetError(error_msg, e)
        
        finally:
            save_time = time.perf_counter() - start_time
            self._operation_stats['save_operation_time'] += save_time
            
            # Ensure performance requirement of <50ms overhead
            if save_time > 0.05:  # 50ms
                logger.warning(f"Figure save took {save_time*1000:.2f}ms, exceeding 50ms target")
    
    def _save(self, data: Figure) -> None:
        """Save matplotlib figure with FigRegistry styling applied.
        
        This method implements the Kedro AbstractDataSet._save interface,
        applying FigRegistry styling and saving the figure with versioning support.
        
        Args:
            data: Matplotlib Figure object to save
            
        Raises:
            FigureDataSetError: If save operation fails
        """
        operation_start = time.perf_counter()
        
        try:
            # Validate input
            if not isinstance(data, Figure):
                raise FigureDataSetError(
                    f"Expected matplotlib Figure object, got {type(data)}"
                )
            
            # Resolve condition for styling
            condition = self._resolve_condition()
            
            # Get styling parameters
            style = self._get_figure_style(condition)
            
            # Apply styling to figure
            self._apply_style_to_figure(data, style)
            
            # Resolve file path (handle Kedro versioning)
            try:
                filepath = get_filepath_str(self._filepath, self._version)
            except:
                # Fallback for older Kedro versions
                filepath = str(self._filepath)
            
            # Ensure output directory exists
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure with FigRegistry integration
            actual_path = self._save_figure_with_figregistry(
                data, filepath, condition, style
            )
            
            # Update statistics
            with self._stats_lock:
                self._operation_stats['saves'] += 1
                self._performance_stats['total_saves'] += 1
                
                operation_time = time.perf_counter() - operation_start
                self._performance_stats['total_save_time'] += operation_time
                self._performance_stats['cache_stats'] = self._style_cache.get_stats()
            
            logger.info(
                f"Successfully saved figure to {actual_path} "
                f"(condition: {condition}, operation time: {operation_time*1000:.2f}ms)"
            )
            
        except Exception as e:
            if isinstance(e, FigureDataSetError):
                raise
            
            operation_time = time.perf_counter() - operation_start
            error_msg = f"Figure save operation failed after {operation_time*1000:.2f}ms: {str(e)}"
            logger.error(error_msg)
            raise FigureDataSetError(error_msg, e)
    
    def _load(self) -> None:
        """Load operation for FigureDataSet.
        
        Note: FigureDataSet is primarily designed for saving figures.
        Loading is not typically required as figures are generated by pipeline nodes.
        
        Raises:
            FigureDataSetError: Always raised as loading is not supported
        """
        raise FigureDataSetError(
            "Loading figures is not supported by FigureDataSet. "
            "Figures should be generated by pipeline nodes and saved through the catalog."
        )
    
    def _describe(self) -> Dict[str, Any]:
        """Describe the dataset configuration and current state.
        
        Returns:
            Dictionary containing dataset metadata and configuration
        """
        return {
            "filepath": self._filepath,
            "purpose": self._purpose,
            "condition_param": self._condition_param,
            "style_params": self._style_params,
            "save_args": self._save_args,
            "version": self._version,
            "operation_stats": self._operation_stats.copy(),
            "cache_stats": self._style_cache.get_stats(),
            "dependencies": {
                "matplotlib_available": matplotlib_available,
                "kedro_available": kedro_available,
                "figregistry_available": figregistry_available
            }
        }
    
    def _exists(self) -> bool:
        """Check if the dataset file exists.
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            filepath = get_filepath_str(self._filepath, self._version)
            return Path(filepath).exists()
        except:
            return Path(self._filepath).exists()
    
    @classmethod
    def get_performance_stats(cls) -> Dict[str, Any]:
        """Get global performance statistics for all FigureDataSet instances.
        
        Returns:
            Dictionary containing performance metrics
        """
        with cls._stats_lock:
            stats = cls._performance_stats.copy()
            
            # Calculate averages
            if stats['total_saves'] > 0:
                stats['avg_style_time'] = stats['total_style_time'] / stats['total_saves']
                stats['avg_save_time'] = stats['total_save_time'] / stats['total_saves']
                stats['avg_total_time'] = (stats['total_style_time'] + stats['total_save_time']) / stats['total_saves']
            else:
                stats['avg_style_time'] = 0.0
                stats['avg_save_time'] = 0.0
                stats['avg_total_time'] = 0.0
            
            return stats
    
    @classmethod
    def clear_cache(cls):
        """Clear the global style cache."""
        cls._style_cache.clear()
        logger.debug("Cleared FigureDataSet style cache")
    
    @classmethod
    def reset_performance_stats(cls):
        """Reset global performance statistics."""
        with cls._stats_lock:
            cls._performance_stats = {
                'total_saves': 0,
                'total_style_time': 0.0,
                'total_save_time': 0.0,
                'cache_stats': {}
            }
        logger.debug("Reset FigureDataSet performance statistics")
    
    def set_pipeline_context(self, context: Dict[str, Any]):
        """Set pipeline context for condition resolution.
        
        This method should be called by Kedro hooks to provide pipeline
        parameters for dynamic condition resolution.
        
        Args:
            context: Pipeline context dictionary containing parameters
        """
        self._set_pipeline_context(context)
        logger.debug(f"Set pipeline context with {len(context)} parameters")


# Utility functions for integration

def create_figure_dataset(filepath: str, 
                         purpose: str = "exploratory",
                         condition_param: Optional[str] = None,
                         style_params: Optional[Dict[str, Any]] = None,
                         **kwargs) -> FigureDataSet:
    """Factory function for creating FigureDataSet instances.
    
    Args:
        filepath: Output file path for the figure
        purpose: Output categorization
        condition_param: Parameter name for dynamic condition resolution
        style_params: Dataset-specific styling overrides
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured FigureDataSet instance
    """
    return FigureDataSet(
        filepath=filepath,
        purpose=purpose,
        condition_param=condition_param,
        style_params=style_params,
        **kwargs
    )


def validate_figure_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate FigureDataSet configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Required parameters
    if 'filepath' not in config:
        raise ValueError("FigureDataSet requires 'filepath' parameter")
    
    # Validate purpose
    purpose = config.get('purpose', 'exploratory')
    valid_purposes = {'exploratory', 'presentation', 'publication'}
    if purpose not in valid_purposes:
        logger.warning(f"Purpose '{purpose}' not in recommended values {valid_purposes}")
    
    # Validate optional parameters
    if 'condition_param' in config and not isinstance(config['condition_param'], str):
        raise ValueError("condition_param must be a string")
    
    if 'style_params' in config and not isinstance(config['style_params'], dict):
        raise ValueError("style_params must be a dictionary")
    
    if 'save_args' in config and not isinstance(config['save_args'], dict):
        raise ValueError("save_args must be a dictionary")
    
    return config


# Export public API
__all__ = [
    'FigureDataSet',
    'FigureDataSetError',
    'StyleResolutionCache',
    'create_figure_dataset',
    'validate_figure_dataset_config'
]