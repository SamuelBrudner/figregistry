"""
Advanced Utility Functions for FigRegistry-Kedro Integration

This module provides sophisticated utility functions supporting complex experimental scenarios
and advanced figure generation patterns in enterprise data science workflows. The utilities
enable complex condition-based styling (F-002), multi-environment configuration management
(F-007), and production-ready statistical analysis patterns for automated visualization
generation within Kedro data pipelines.

Key Capabilities:
- Advanced experimental condition resolution with hierarchical inheritance
- Multi-environment configuration management for enterprise deployment
- Statistical analysis helpers for sophisticated reporting pipelines
- Data transformation utilities for complex visualization scenarios
- Production-ready performance monitoring and error handling
- Thread-safe operations for parallel Kedro pipeline execution

The module is designed for enterprise-grade usage with comprehensive error handling,
performance optimization, and extensive logging support suitable for production
data science workflows.
"""

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
import copy
import hashlib
import json
import re
from collections import defaultdict, ChainMap
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import configuration bridge from dependency
from figregistry_kedro.config import (
    FigRegistryConfigBridge,
    FigRegistryConfigSchema,
    ConfigMergeError,
    ConfigValidationError
)

# Configure module logger
logger = logging.getLogger(__name__)

# Module-level constants for enterprise configuration
DEFAULT_PERFORMANCE_TARGET_MS = 5.0
MAX_CONDITION_CACHE_SIZE = 10000
CONDITION_RESOLUTION_TIMEOUT_SECONDS = 30.0
STATISTICAL_ANALYSIS_PRECISION = 1e-6

# Thread-safe caches and state management
_condition_resolution_cache: Dict[str, Any] = {}
_statistical_cache: Dict[str, Any] = {}
_performance_metrics: Dict[str, List[float]] = defaultdict(list)
_cache_lock = Lock()

# Advanced condition matching patterns
WILDCARD_PATTERNS = {
    '*': r'.*',
    '?': r'.',
    '[!...]': r'[^...]',
    '[...]': r'[...]'
}

# Enterprise-grade error types
class ConditionResolutionError(Exception):
    """Raised when experimental condition resolution fails."""
    pass

class ConfigurationEnvironmentError(Exception):
    """Raised when multi-environment configuration operations fail."""
    pass

class StatisticalAnalysisError(Exception):
    """Raised when statistical analysis operations fail."""
    pass

class DataTransformationError(Exception):
    """Raised when data transformation operations fail."""
    pass


@dataclass
class ExperimentalCondition:
    """
    Represents a complex experimental condition with hierarchical inheritance and metadata.
    
    This class supports the advanced condition-based styling requirements (F-002) by providing
    a structured representation of experimental conditions that can be resolved against
    configuration hierarchies and matched using sophisticated pattern matching algorithms.
    """
    
    condition_key: str
    condition_value: Any
    condition_type: str = "categorical"
    hierarchy_level: int = 0
    parent_condition: Optional['ExperimentalCondition'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    
    def __post_init__(self):
        """Validate condition properties and set derived attributes."""
        if not self.condition_key:
            raise ValueError("Condition key cannot be empty")
        
        # Generate condition hash for caching
        self.condition_hash = self._generate_condition_hash()
        
        # Set default metadata
        if not self.metadata:
            self.metadata = {
                "created_at": time.time(),
                "validation_status": "pending"
            }
    
    def _generate_condition_hash(self) -> str:
        """Generate unique hash for condition caching."""
        condition_data = {
            "key": self.condition_key,
            "value": str(self.condition_value),
            "type": self.condition_type,
            "level": self.hierarchy_level
        }
        return hashlib.md5(
            json.dumps(condition_data, sort_keys=True).encode()
        ).hexdigest()
    
    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if condition matches a given pattern using advanced wildcard support.
        
        Args:
            pattern: Pattern string with wildcard support
            
        Returns:
            True if condition matches pattern, False otherwise
        """
        # Convert pattern to regex
        regex_pattern = pattern
        for wildcard, regex in WILDCARD_PATTERNS.items():
            if wildcard in pattern:
                regex_pattern = regex_pattern.replace(wildcard, regex)
        
        try:
            return bool(re.match(regex_pattern, str(self.condition_value)))
        except re.error:
            logger.warning(f"Invalid pattern '{pattern}' for condition matching")
            return False
    
    def get_inheritance_chain(self) -> List['ExperimentalCondition']:
        """Get full inheritance chain from root to current condition."""
        chain = []
        current = self
        while current:
            chain.insert(0, current)
            current = current.parent_condition
        return chain
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary representation."""
        return {
            "condition_key": self.condition_key,
            "condition_value": self.condition_value,
            "condition_type": self.condition_type,
            "hierarchy_level": self.hierarchy_level,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "priority": self.priority,
            "condition_hash": self.condition_hash
        }


@dataclass
class EnvironmentConfiguration:
    """
    Multi-environment configuration context supporting sophisticated deployment scenarios.
    
    This class implements advanced configuration management for enterprise deployments
    supporting the F-007 configuration bridge requirements with environment-specific
    override capabilities and validation.
    """
    
    environment_name: str
    config_hierarchy: List[Dict[str, Any]] = field(default_factory=list)
    override_rules: Dict[str, Any] = field(default_factory=dict)
    validation_schema: Optional[Dict[str, Any]] = None
    deployment_context: Dict[str, Any] = field(default_factory=dict)
    performance_constraints: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize environment-specific defaults."""
        if not self.performance_constraints:
            self.performance_constraints = {
                "max_merge_time_ms": DEFAULT_PERFORMANCE_TARGET_MS,
                "max_validation_time_ms": 50.0,
                "max_condition_resolution_ms": 20.0
            }
        
        # Set environment-specific defaults
        self.deployment_context.setdefault("environment_type", self._classify_environment())
        self.deployment_context.setdefault("security_level", self._determine_security_level())
    
    def _classify_environment(self) -> str:
        """Classify environment type based on name patterns."""
        env_lower = self.environment_name.lower()
        if any(term in env_lower for term in ['dev', 'development', 'local']):
            return "development"
        elif any(term in env_lower for term in ['test', 'testing', 'staging']):
            return "testing"
        elif any(term in env_lower for term in ['prod', 'production', 'live']):
            return "production"
        else:
            return "custom"
    
    def _determine_security_level(self) -> str:
        """Determine security level based on environment classification."""
        env_type = self.deployment_context.get("environment_type", "custom")
        security_mapping = {
            "development": "low",
            "testing": "medium",
            "production": "high",
            "custom": "medium"
        }
        return security_mapping.get(env_type, "medium")


def performance_monitor(operation_name: str, target_ms: float = DEFAULT_PERFORMANCE_TARGET_MS):
    """
    Decorator for monitoring operation performance with enterprise-grade metrics.
    
    This decorator supports production-ready performance monitoring by tracking
    operation execution times and providing warnings when performance targets
    are exceeded. Metrics are stored for analysis and alerting.
    
    Args:
        operation_name: Name of the operation being monitored
        target_ms: Target execution time in milliseconds
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Store metrics thread-safely
                with _cache_lock:
                    _performance_metrics[operation_name].append(execution_time_ms)
                    # Limit metrics storage
                    if len(_performance_metrics[operation_name]) > 1000:
                        _performance_metrics[operation_name] = _performance_metrics[operation_name][-500:]
                
                # Log performance warnings
                if execution_time_ms > target_ms:
                    logger.warning(
                        f"Operation '{operation_name}' took {execution_time_ms:.2f}ms, "
                        f"exceeding target {target_ms}ms"
                    )
                else:
                    logger.debug(
                        f"Operation '{operation_name}' completed in {execution_time_ms:.2f}ms"
                    )
            
            return result
        return wrapper
    return decorator


class AdvancedConditionResolver:
    """
    Advanced experimental condition resolution engine supporting complex scenarios.
    
    This class implements sophisticated condition matching and resolution capabilities
    for the F-002 condition-based styling requirements, providing hierarchical
    inheritance, pattern matching, and performance-optimized condition resolution
    suitable for enterprise data science workflows.
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        max_cache_size: int = MAX_CONDITION_CACHE_SIZE,
        resolution_timeout: float = CONDITION_RESOLUTION_TIMEOUT_SECONDS
    ):
        """
        Initialize the advanced condition resolver.
        
        Args:
            cache_enabled: Enable condition resolution caching
            max_cache_size: Maximum cache entries for resolved conditions
            resolution_timeout: Timeout for complex condition resolution operations
        """
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.resolution_timeout = resolution_timeout
        
        # Thread-safe caching and metrics
        self._condition_cache: Dict[str, Any] = {}
        self._resolution_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "resolution_failures": 0,
            "pattern_matches": 0
        }
        self._lock = Lock()
        
        logger.info(f"Initialized AdvancedConditionResolver with cache_size={max_cache_size}")
    
    @performance_monitor("condition_resolution", target_ms=10.0)
    def resolve_experimental_condition(
        self,
        condition_parameters: Dict[str, Any],
        style_config: Dict[str, Any],
        inheritance_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve experimental conditions against style configuration with advanced matching.
        
        This method implements the core F-002 functionality by matching experimental
        condition parameters against style configuration hierarchies with support for
        wildcards, inheritance, and complex matching rules.
        
        Args:
            condition_parameters: Dictionary of experimental condition parameters
            style_config: Style configuration hierarchy to match against
            inheritance_rules: Optional inheritance and override rules
            
        Returns:
            Resolved style dictionary with applied conditions
            
        Raises:
            ConditionResolutionError: When condition resolution fails
        """
        try:
            # Generate cache key
            cache_key = self._generate_resolution_cache_key(
                condition_parameters, style_config, inheritance_rules
            )
            
            # Check cache first
            if self.cache_enabled:
                cached_result = self._get_cached_resolution(cache_key)
                if cached_result is not None:
                    self._resolution_stats["cache_hits"] += 1
                    return cached_result
                self._resolution_stats["cache_misses"] += 1
            
            logger.debug(f"Resolving conditions: {list(condition_parameters.keys())}")
            
            # Create condition objects for advanced processing
            conditions = self._create_condition_objects(condition_parameters)
            
            # Resolve conditions with inheritance
            resolved_styles = self._resolve_with_inheritance(
                conditions, style_config, inheritance_rules or {}
            )
            
            # Apply pattern matching and wildcards
            pattern_matched_styles = self._apply_pattern_matching(
                conditions, resolved_styles, style_config
            )
            
            # Merge and validate final styles
            final_styles = self._merge_and_validate_styles(
                pattern_matched_styles, style_config
            )
            
            # Cache results
            if self.cache_enabled:
                self._cache_resolution(cache_key, final_styles)
            
            logger.debug(f"Successfully resolved {len(final_styles)} style properties")
            return final_styles
            
        except Exception as e:
            self._resolution_stats["resolution_failures"] += 1
            logger.error(f"Condition resolution failed: {e}")
            raise ConditionResolutionError(f"Failed to resolve experimental conditions: {e}") from e
    
    def _create_condition_objects(self, condition_parameters: Dict[str, Any]) -> List[ExperimentalCondition]:
        """Create structured condition objects from parameters."""
        conditions = []
        
        for key, value in condition_parameters.items():
            if value is None:
                continue
            
            # Determine condition type
            condition_type = self._infer_condition_type(value)
            
            # Extract hierarchy information
            hierarchy_level = key.count('.') if '.' in key else 0
            
            condition = ExperimentalCondition(
                condition_key=key,
                condition_value=value,
                condition_type=condition_type,
                hierarchy_level=hierarchy_level,
                metadata={"source": "kedro_parameters"}
            )
            
            conditions.append(condition)
        
        # Sort by hierarchy level and priority
        conditions.sort(key=lambda c: (c.hierarchy_level, -c.priority))
        return conditions
    
    def _infer_condition_type(self, value: Any) -> str:
        """Infer condition type from value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "numeric"
        elif isinstance(value, str):
            return "categorical"
        elif isinstance(value, (list, tuple)):
            return "multi_value"
        elif isinstance(value, dict):
            return "structured"
        else:
            return "object"
    
    def _resolve_with_inheritance(
        self,
        conditions: List[ExperimentalCondition],
        style_config: Dict[str, Any],
        inheritance_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conditions with hierarchical inheritance."""
        resolved_styles = {}
        
        # Start with base styles
        if "defaults" in style_config:
            resolved_styles.update(style_config["defaults"])
        
        # Apply inheritance hierarchy
        for condition in conditions:
            # Find matching styles for this condition
            condition_styles = self._find_matching_styles(condition, style_config)
            
            if condition_styles:
                # Apply inheritance rules
                inherited_styles = self._apply_inheritance_rules(
                    condition_styles, inheritance_rules, condition
                )
                
                # Merge with existing styles
                resolved_styles = self._deep_merge_styles(resolved_styles, inherited_styles)
        
        return resolved_styles
    
    def _find_matching_styles(
        self,
        condition: ExperimentalCondition,
        style_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find styles matching a specific condition."""
        matching_styles = {}
        
        # Check direct matches first
        styles_section = style_config.get("styles", {})
        
        # Direct key match
        if condition.condition_key in styles_section:
            direct_match = styles_section[condition.condition_key]
            if isinstance(direct_match, dict):
                # Check for value-specific styles
                if str(condition.condition_value) in direct_match:
                    matching_styles.update(direct_match[str(condition.condition_value)])
                elif "default" in direct_match:
                    matching_styles.update(direct_match["default"])
                else:
                    matching_styles.update(direct_match)
        
        # Pattern-based matching
        for style_key, style_value in styles_section.items():
            if self._matches_condition_pattern(condition, style_key):
                if isinstance(style_value, dict):
                    matching_styles = self._deep_merge_styles(matching_styles, style_value)
        
        return matching_styles
    
    def _matches_condition_pattern(self, condition: ExperimentalCondition, pattern: str) -> bool:
        """Check if condition matches a pattern."""
        # Exact match
        if condition.condition_key == pattern:
            return True
        
        # Wildcard pattern matching
        if any(wildcard in pattern for wildcard in WILDCARD_PATTERNS.keys()):
            return condition.matches_pattern(pattern)
        
        # Hierarchical matching
        if '.' in pattern and '.' in condition.condition_key:
            pattern_parts = pattern.split('.')
            condition_parts = condition.condition_key.split('.')
            
            # Check if pattern is a prefix of condition
            if len(pattern_parts) <= len(condition_parts):
                return pattern_parts == condition_parts[:len(pattern_parts)]
        
        return False
    
    def _apply_inheritance_rules(
        self,
        styles: Dict[str, Any],
        inheritance_rules: Dict[str, Any],
        condition: ExperimentalCondition
    ) -> Dict[str, Any]:
        """Apply inheritance rules to style resolution."""
        if not inheritance_rules:
            return styles
        
        # Apply inheritance transformations
        inherited_styles = copy.deepcopy(styles)
        
        # Color inheritance
        if "color_inheritance" in inheritance_rules:
            inherited_styles = self._apply_color_inheritance(
                inherited_styles, inheritance_rules["color_inheritance"], condition
            )
        
        # Style property inheritance
        if "property_inheritance" in inheritance_rules:
            inherited_styles = self._apply_property_inheritance(
                inherited_styles, inheritance_rules["property_inheritance"], condition
            )
        
        return inherited_styles
    
    def _apply_color_inheritance(
        self,
        styles: Dict[str, Any],
        color_rules: Dict[str, Any],
        condition: ExperimentalCondition
    ) -> Dict[str, Any]:
        """Apply color inheritance rules."""
        if "color" not in styles and "base_color" in color_rules:
            # Generate color based on condition properties
            base_color = color_rules["base_color"]
            
            if condition.condition_type == "numeric":
                # Numeric conditions get interpolated colors
                styles["color"] = self._interpolate_color(
                    base_color, condition.condition_value, color_rules
                )
            elif condition.condition_type == "categorical":
                # Categorical conditions get derived colors
                styles["color"] = self._derive_categorical_color(
                    base_color, condition.condition_value, color_rules
                )
        
        return styles
    
    def _interpolate_color(self, base_color: str, value: float, color_rules: Dict[str, Any]) -> str:
        """Interpolate color based on numeric value."""
        try:
            # Convert base color to RGB
            base_rgb = mcolors.to_rgb(base_color)
            
            # Get interpolation range
            min_val = color_rules.get("min_value", 0.0)
            max_val = color_rules.get("max_value", 1.0)
            
            # Normalize value
            normalized = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
            
            # Interpolate to target color
            target_color = color_rules.get("target_color", "#FF0000")
            target_rgb = mcolors.to_rgb(target_color)
            
            interpolated_rgb = [
                base_rgb[i] + normalized * (target_rgb[i] - base_rgb[i])
                for i in range(3)
            ]
            
            return mcolors.to_hex(interpolated_rgb)
            
        except Exception as e:
            logger.warning(f"Color interpolation failed: {e}")
            return base_color
    
    def _derive_categorical_color(self, base_color: str, value: str, color_rules: Dict[str, Any]) -> str:
        """Derive color for categorical values."""
        try:
            # Use hash-based color derivation for consistency
            value_hash = hashlib.md5(str(value).encode()).hexdigest()
            hash_int = int(value_hash[:8], 16)
            
            # Convert base color to HSV for manipulation
            base_rgb = mcolors.to_rgb(base_color)
            base_hsv = mcolors.rgb_to_hsv(base_rgb)
            
            # Modify hue based on hash
            hue_shift = (hash_int % 360) / 360.0
            shift_amount = color_rules.get("hue_variation", 0.2)
            
            new_hue = (base_hsv[0] + hue_shift * shift_amount) % 1.0
            new_hsv = (new_hue, base_hsv[1], base_hsv[2])
            
            # Convert back to RGB and hex
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            return mcolors.to_hex(new_rgb)
            
        except Exception as e:
            logger.warning(f"Categorical color derivation failed: {e}")
            return base_color
    
    def _apply_property_inheritance(
        self,
        styles: Dict[str, Any],
        property_rules: Dict[str, Any],
        condition: ExperimentalCondition
    ) -> Dict[str, Any]:
        """Apply property inheritance rules."""
        for property_name, rule in property_rules.items():
            if property_name not in styles and "base_value" in rule:
                # Apply inheritance rule
                base_value = rule["base_value"]
                
                if "modifier" in rule:
                    modifier = rule["modifier"]
                    if callable(modifier):
                        styles[property_name] = modifier(base_value, condition)
                    elif isinstance(modifier, (int, float)):
                        styles[property_name] = base_value * modifier
                else:
                    styles[property_name] = base_value
        
        return styles
    
    def _apply_pattern_matching(
        self,
        conditions: List[ExperimentalCondition],
        resolved_styles: Dict[str, Any],
        style_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply advanced pattern matching for style resolution."""
        pattern_styles = copy.deepcopy(resolved_styles)
        
        # Get pattern rules from configuration
        pattern_rules = style_config.get("pattern_rules", {})
        
        for condition in conditions:
            for pattern, rule in pattern_rules.items():
                if condition.matches_pattern(pattern):
                    self._resolution_stats["pattern_matches"] += 1
                    
                    # Apply pattern-specific styles
                    if "styles" in rule:
                        pattern_styles = self._deep_merge_styles(
                            pattern_styles, rule["styles"]
                        )
                    
                    # Apply transformations
                    if "transformations" in rule:
                        pattern_styles = self._apply_transformations(
                            pattern_styles, rule["transformations"], condition
                        )
        
        return pattern_styles
    
    def _apply_transformations(
        self,
        styles: Dict[str, Any],
        transformations: Dict[str, Any],
        condition: ExperimentalCondition
    ) -> Dict[str, Any]:
        """Apply style transformations based on condition properties."""
        transformed_styles = copy.deepcopy(styles)
        
        for transform_name, transform_config in transformations.items():
            if transform_name == "scale_numeric":
                # Scale numeric properties based on condition value
                if condition.condition_type == "numeric":
                    for prop in transform_config.get("properties", []):
                        if prop in transformed_styles:
                            scale_factor = transform_config.get("scale_factor", 1.0)
                            base_value = transformed_styles[prop]
                            scaled_value = base_value * (1 + scale_factor * condition.condition_value)
                            transformed_styles[prop] = scaled_value
            
            elif transform_name == "alpha_by_priority":
                # Adjust alpha based on condition priority
                if "alpha" in transformed_styles:
                    priority_factor = transform_config.get("priority_factor", 0.1)
                    base_alpha = transformed_styles["alpha"]
                    priority_alpha = base_alpha * (1 + priority_factor * condition.priority)
                    transformed_styles["alpha"] = max(0.0, min(1.0, priority_alpha))
        
        return transformed_styles
    
    def _merge_and_validate_styles(
        self,
        pattern_styles: Dict[str, Any],
        style_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge and validate final style configuration."""
        # Apply final fallback styles if needed
        fallback_styles = style_config.get("fallback_style", {})
        final_styles = self._deep_merge_styles(fallback_styles, pattern_styles)
        
        # Validate required properties
        required_props = style_config.get("required_properties", ["color"])
        for prop in required_props:
            if prop not in final_styles:
                logger.warning(f"Required property '{prop}' missing from resolved styles")
                # Apply sensible defaults
                if prop == "color":
                    final_styles[prop] = "#1f77b4"  # Default matplotlib blue
                elif prop == "linewidth":
                    final_styles[prop] = 1.5
                elif prop == "alpha":
                    final_styles[prop] = 0.8
        
        # Validate property values
        final_styles = self._validate_style_properties(final_styles)
        
        return final_styles
    
    def _validate_style_properties(self, styles: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct style property values."""
        validated_styles = copy.deepcopy(styles)
        
        # Color validation
        if "color" in validated_styles:
            try:
                mcolors.to_rgb(validated_styles["color"])
            except ValueError:
                logger.warning(f"Invalid color '{validated_styles['color']}', using default")
                validated_styles["color"] = "#1f77b4"
        
        # Numeric property validation
        numeric_props = {
            "linewidth": (0.1, 10.0),
            "markersize": (1.0, 20.0),
            "alpha": (0.0, 1.0),
            "fontsize": (6.0, 24.0)
        }
        
        for prop, (min_val, max_val) in numeric_props.items():
            if prop in validated_styles:
                try:
                    value = float(validated_styles[prop])
                    validated_styles[prop] = max(min_val, min(max_val, value))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid {prop} value, using default")
                    del validated_styles[prop]
        
        return validated_styles
    
    def _deep_merge_styles(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge style dictionaries with override precedence."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_styles(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _generate_resolution_cache_key(self, *args) -> str:
        """Generate cache key for condition resolution."""
        key_data = []
        for arg in args:
            if isinstance(arg, dict):
                key_data.append(json.dumps(arg, sort_keys=True))
            else:
                key_data.append(str(arg))
        
        return hashlib.md5("_".join(key_data).encode()).hexdigest()
    
    def _get_cached_resolution(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached condition resolution."""
        with self._lock:
            return self._condition_cache.get(cache_key)
    
    def _cache_resolution(self, cache_key: str, resolution: Dict[str, Any]) -> None:
        """Cache condition resolution."""
        with self._lock:
            if len(self._condition_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._condition_cache))
                del self._condition_cache[oldest_key]
            
            self._condition_cache[cache_key] = copy.deepcopy(resolution)
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get condition resolution performance statistics."""
        with self._lock:
            return {
                "cache_stats": self._resolution_stats.copy(),
                "cache_size": len(self._condition_cache),
                "max_cache_size": self.max_cache_size
            }
    
    def clear_cache(self) -> None:
        """Clear condition resolution cache."""
        with self._lock:
            self._condition_cache.clear()
            self._resolution_stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "resolution_failures": 0,
                "pattern_matches": 0
            }


class MultiEnvironmentConfigManager:
    """
    Advanced configuration manager for multi-environment deployment scenarios.
    
    This class supports the F-007 configuration bridge requirements by providing
    sophisticated configuration management across development, staging, and production
    environments with environment-specific overrides, validation, and deployment
    context awareness.
    """
    
    def __init__(
        self,
        config_bridge: Optional[FigRegistryConfigBridge] = None,
        default_environments: Optional[List[str]] = None
    ):
        """
        Initialize multi-environment configuration manager.
        
        Args:
            config_bridge: FigRegistryConfigBridge instance for configuration merging
            default_environments: List of default environment names to support
        """
        self.config_bridge = config_bridge or FigRegistryConfigBridge()
        self.default_environments = default_environments or [
            "development", "testing", "staging", "production"
        ]
        
        # Environment-specific configuration cache
        self._environment_configs: Dict[str, EnvironmentConfiguration] = {}
        self._config_lock = Lock()
        
        # Performance tracking
        self._environment_stats = {
            "config_loads": 0,
            "cache_hits": 0,
            "validation_failures": 0
        }
        
        logger.info(f"Initialized MultiEnvironmentConfigManager for environments: {self.default_environments}")
    
    @performance_monitor("environment_config_load", target_ms=15.0)
    def load_environment_config(
        self,
        environment_name: str,
        config_loader: Optional[Any] = None,
        project_path: Optional[Path] = None,
        validation_strict: bool = True,
        **override_params
    ) -> EnvironmentConfiguration:
        """
        Load and validate environment-specific configuration.
        
        This method implements sophisticated environment-aware configuration loading
        that supports the F-007 requirements for multi-environment deployment with
        proper validation and override capabilities.
        
        Args:
            environment_name: Name of the environment to load configuration for
            config_loader: Kedro ConfigLoader instance
            project_path: Project root path for configuration discovery
            validation_strict: Enable strict validation for loaded configuration
            **override_params: Additional override parameters
            
        Returns:
            EnvironmentConfiguration object with loaded and validated configuration
            
        Raises:
            ConfigurationEnvironmentError: When environment configuration loading fails
        """
        try:
            self._environment_stats["config_loads"] += 1
            
            # Check cache first
            cache_key = f"{environment_name}_{hash(str(override_params))}"
            with self._config_lock:
                if cache_key in self._environment_configs:
                    self._environment_stats["cache_hits"] += 1
                    return self._environment_configs[cache_key]
            
            logger.info(f"Loading configuration for environment: {environment_name}")
            
            # Load base configuration through bridge
            base_config = self.config_bridge.merge_configurations(
                config_loader=config_loader,
                environment=environment_name,
                project_path=project_path,
                validation_strict=validation_strict,
                **override_params
            )
            
            # Create environment-specific configuration hierarchy
            config_hierarchy = self._build_environment_hierarchy(
                environment_name, base_config, project_path
            )
            
            # Determine environment-specific override rules
            override_rules = self._determine_override_rules(environment_name, base_config)
            
            # Create validation schema for this environment
            validation_schema = self._create_environment_validation_schema(
                environment_name, base_config
            )
            
            # Build deployment context
            deployment_context = self._build_deployment_context(
                environment_name, config_loader, project_path
            )
            
            # Create environment configuration
            env_config = EnvironmentConfiguration(
                environment_name=environment_name,
                config_hierarchy=config_hierarchy,
                override_rules=override_rules,
                validation_schema=validation_schema,
                deployment_context=deployment_context
            )
            
            # Validate environment configuration
            if validation_strict:
                self._validate_environment_config(env_config)
            
            # Cache for future use
            with self._config_lock:
                self._environment_configs[cache_key] = env_config
            
            logger.debug(f"Successfully loaded configuration for environment: {environment_name}")
            return env_config
            
        except Exception as e:
            self._environment_stats["validation_failures"] += 1
            logger.error(f"Failed to load environment configuration for {environment_name}: {e}")
            raise ConfigurationEnvironmentError(
                f"Failed to load environment configuration: {e}"
            ) from e
    
    def _build_environment_hierarchy(
        self,
        environment_name: str,
        base_config: Dict[str, Any],
        project_path: Optional[Path]
    ) -> List[Dict[str, Any]]:
        """Build configuration hierarchy for environment."""
        hierarchy = []
        
        # Add base configuration
        hierarchy.append({
            "source": "base",
            "priority": 0,
            "config": base_config
        })
        
        # Add environment-specific overlays
        if project_path:
            env_config_paths = [
                project_path / "conf" / environment_name / "figregistry.yml",
                project_path / "conf" / environment_name / "parameters.yml"
            ]
            
            for config_path in env_config_paths:
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path, 'r') as f:
                            env_config = yaml.safe_load(f) or {}
                        
                        hierarchy.append({
                            "source": str(config_path),
                            "priority": 10,
                            "config": env_config
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load environment config from {config_path}: {e}")
        
        # Sort by priority
        hierarchy.sort(key=lambda x: x["priority"])
        return hierarchy
    
    def _determine_override_rules(
        self,
        environment_name: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine environment-specific override rules."""
        # Default override rules by environment type
        env_type = self._classify_environment_type(environment_name)
        
        default_rules = {
            "development": {
                "performance_constraints": {"relaxed": True},
                "validation_level": "permissive",
                "debug_enabled": True,
                "cache_ttl": 300  # 5 minutes
            },
            "testing": {
                "performance_constraints": {"strict": True},
                "validation_level": "strict",
                "debug_enabled": True,
                "cache_ttl": 600  # 10 minutes
            },
            "staging": {
                "performance_constraints": {"strict": True},
                "validation_level": "strict",
                "debug_enabled": False,
                "cache_ttl": 1800  # 30 minutes
            },
            "production": {
                "performance_constraints": {"strict": True},
                "validation_level": "strict",
                "debug_enabled": False,
                "cache_ttl": 3600  # 1 hour
            }
        }
        
        rules = default_rules.get(env_type, default_rules["development"])
        
        # Merge with configuration-specific rules
        config_rules = base_config.get("environment_rules", {}).get(environment_name, {})
        rules.update(config_rules)
        
        return rules
    
    def _classify_environment_type(self, environment_name: str) -> str:
        """Classify environment type from name."""
        env_lower = environment_name.lower()
        
        if any(term in env_lower for term in ['dev', 'development', 'local']):
            return "development"
        elif any(term in env_lower for term in ['test', 'testing']):
            return "testing"
        elif any(term in env_lower for term in ['stage', 'staging', 'uat']):
            return "staging"
        elif any(term in env_lower for term in ['prod', 'production', 'live']):
            return "production"
        else:
            return "development"  # Default to development for unknown types
    
    def _create_environment_validation_schema(
        self,
        environment_name: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create validation schema for environment-specific configuration."""
        # Base validation schema from FigRegistry
        base_schema = {
            "required_sections": ["styles", "defaults", "outputs"],
            "optional_sections": ["palettes", "kedro", "metadata"],
            "property_constraints": {
                "styles": {"min_entries": 1},
                "outputs.base_path": {"required": True},
                "defaults.figure.figsize": {"type": "list", "length": 2}
            }
        }
        
        # Environment-specific validation rules
        env_type = self._classify_environment_type(environment_name)
        
        env_specific_rules = {
            "development": {
                "validation_level": "permissive",
                "allow_unknown_properties": True,
                "require_all_sections": False
            },
            "testing": {
                "validation_level": "strict",
                "allow_unknown_properties": False,
                "require_all_sections": True,
                "additional_validations": ["performance_constraints"]
            },
            "staging": {
                "validation_level": "strict",
                "allow_unknown_properties": False,
                "require_all_sections": True,
                "additional_validations": ["performance_constraints", "security_checks"]
            },
            "production": {
                "validation_level": "strict",
                "allow_unknown_properties": False,
                "require_all_sections": True,
                "additional_validations": [
                    "performance_constraints", 
                    "security_checks", 
                    "compliance_checks"
                ]
            }
        }
        
        # Merge schemas
        env_rules = env_specific_rules.get(env_type, env_specific_rules["development"])
        validation_schema = {**base_schema, **env_rules}
        
        return validation_schema
    
    def _build_deployment_context(
        self,
        environment_name: str,
        config_loader: Optional[Any],
        project_path: Optional[Path]
    ) -> Dict[str, Any]:
        """Build deployment context for environment."""
        context = {
            "environment_name": environment_name,
            "environment_type": self._classify_environment_type(environment_name),
            "deployment_timestamp": time.time(),
            "project_path": str(project_path) if project_path else None
        }
        
        # Add Kedro context information if available
        if config_loader:
            try:
                context["kedro_config_loader"] = str(type(config_loader))
                # Add additional Kedro metadata if accessible
                if hasattr(config_loader, 'config_patterns'):
                    context["kedro_config_patterns"] = config_loader.config_patterns
            except Exception as e:
                logger.debug(f"Could not extract Kedro context: {e}")
        
        # Add system information
        context["system_info"] = {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "platform": __import__('platform').system()
        }
        
        return context
    
    def _validate_environment_config(self, env_config: EnvironmentConfiguration) -> None:
        """Validate environment configuration against schema."""
        if not env_config.validation_schema:
            return
        
        validation_level = env_config.validation_schema.get("validation_level", "strict")
        
        # Check required sections
        if env_config.validation_schema.get("require_all_sections", False):
            required_sections = env_config.validation_schema.get("required_sections", [])
            
            for hierarchy_item in env_config.config_hierarchy:
                config = hierarchy_item["config"]
                missing_sections = [
                    section for section in required_sections 
                    if section not in config
                ]
                
                if missing_sections and validation_level == "strict":
                    raise ConfigurationEnvironmentError(
                        f"Missing required sections: {missing_sections}"
                    )
                elif missing_sections:
                    logger.warning(f"Missing sections: {missing_sections}")
        
        # Validate property constraints
        property_constraints = env_config.validation_schema.get("property_constraints", {})
        for constraint_path, constraint_rules in property_constraints.items():
            self._validate_property_constraint(
                env_config, constraint_path, constraint_rules, validation_level
            )
    
    def _validate_property_constraint(
        self,
        env_config: EnvironmentConfiguration,
        constraint_path: str,
        constraint_rules: Dict[str, Any],
        validation_level: str
    ) -> None:
        """Validate specific property constraint."""
        # Extract property value from config hierarchy
        path_parts = constraint_path.split('.')
        
        for hierarchy_item in env_config.config_hierarchy:
            config = hierarchy_item["config"]
            value = config
            
            # Navigate to property
            try:
                for part in path_parts:
                    value = value[part]
                
                # Apply constraint rules
                if "required" in constraint_rules and constraint_rules["required"]:
                    if value is None:
                        if validation_level == "strict":
                            raise ConfigurationEnvironmentError(
                                f"Required property {constraint_path} is missing"
                            )
                        else:
                            logger.warning(f"Required property {constraint_path} is missing")
                
                if "type" in constraint_rules:
                    expected_type = constraint_rules["type"]
                    if expected_type == "list" and not isinstance(value, list):
                        if validation_level == "strict":
                            raise ConfigurationEnvironmentError(
                                f"Property {constraint_path} must be a list"
                            )
                
                if "length" in constraint_rules and isinstance(value, (list, tuple)):
                    expected_length = constraint_rules["length"]
                    if len(value) != expected_length:
                        if validation_level == "strict":
                            raise ConfigurationEnvironmentError(
                                f"Property {constraint_path} must have length {expected_length}"
                            )
                
                break  # Property found and validated
                
            except (KeyError, TypeError):
                # Property not found in this hierarchy level
                continue
    
    def merge_environment_configs(
        self,
        environments: List[str],
        merge_strategy: str = "override"
    ) -> Dict[str, Any]:
        """
        Merge configurations from multiple environments.
        
        Args:
            environments: List of environment names to merge
            merge_strategy: Strategy for merging ("override", "deep_merge", "selective")
            
        Returns:
            Merged configuration dictionary
        """
        if not environments:
            raise ConfigurationEnvironmentError("No environments specified for merging")
        
        logger.info(f"Merging configurations for environments: {environments}")
        
        merged_config = {}
        
        for env_name in environments:
            if env_name not in self._environment_configs:
                logger.warning(f"Environment {env_name} not loaded, skipping")
                continue
            
            env_config = self._environment_configs[env_name]
            
            # Merge configuration hierarchy
            for hierarchy_item in env_config.config_hierarchy:
                config = hierarchy_item["config"]
                
                if merge_strategy == "override":
                    merged_config.update(config)
                elif merge_strategy == "deep_merge":
                    merged_config = self._deep_merge_configs(merged_config, config)
                elif merge_strategy == "selective":
                    merged_config = self._selective_merge_configs(
                        merged_config, config, env_config.override_rules
                    )
        
        return merged_config
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _selective_merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Selectively merge configurations based on rules."""
        result = copy.deepcopy(base)
        
        # Get selective merge rules
        merge_rules = rules.get("selective_merge", {})
        allowed_sections = merge_rules.get("allowed_sections", list(override.keys()))
        blocked_sections = merge_rules.get("blocked_sections", [])
        
        for key, value in override.items():
            if key in blocked_sections:
                continue
            
            if key in allowed_sections:
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge_configs(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
        
        return result
    
    def get_environment_statistics(self) -> Dict[str, Any]:
        """Get environment configuration statistics."""
        with self._config_lock:
            return {
                "loaded_environments": list(self._environment_configs.keys()),
                "environment_count": len(self._environment_configs),
                "load_statistics": self._environment_stats.copy()
            }


class AdvancedDataTransformer:
    """
    Advanced data transformation utilities for sophisticated visualization generation.
    
    This class provides production-ready data transformation patterns supporting
    complex visualization scenarios in enterprise data science workflows. The
    transformations are optimized for performance and include comprehensive
    error handling and validation.
    """
    
    def __init__(self, precision: float = STATISTICAL_ANALYSIS_PRECISION):
        """
        Initialize advanced data transformer.
        
        Args:
            precision: Numerical precision for statistical calculations
        """
        self.precision = precision
        self.transformation_cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        
        logger.info("Initialized AdvancedDataTransformer")
    
    @performance_monitor("data_transformation", target_ms=20.0)
    def transform_for_visualization(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        transformation_config: Dict[str, Any],
        validation_strict: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Transform data for advanced visualization scenarios.
        
        This method provides sophisticated data transformation capabilities supporting
        complex visualization generation patterns with comprehensive validation and
        error handling suitable for enterprise workflows.
        
        Args:
            data: Input data to transform (DataFrame, array, or dictionary)
            transformation_config: Configuration specifying transformations to apply
            validation_strict: Enable strict validation of transformation results
            
        Returns:
            Tuple of (transformed_data, transformation_metadata)
            
        Raises:
            DataTransformationError: When data transformation fails
        """
        try:
            logger.debug(f"Transforming data with config: {list(transformation_config.keys())}")
            
            # Validate input data
            validated_data = self._validate_input_data(data, validation_strict)
            
            # Initialize transformation metadata
            metadata = {
                "input_shape": self._get_data_shape(validated_data),
                "input_type": type(validated_data).__name__,
                "transformations_applied": [],
                "warnings": [],
                "performance_metrics": {}
            }
            
            # Apply transformations in sequence
            transformed_data = validated_data
            
            for transform_name, transform_params in transformation_config.items():
                start_time = time.time()
                
                try:
                    if transform_name == "normalize":
                        transformed_data = self._apply_normalization(
                            transformed_data, transform_params
                        )
                    elif transform_name == "aggregate":
                        transformed_data = self._apply_aggregation(
                            transformed_data, transform_params
                        )
                    elif transform_name == "filter":
                        transformed_data = self._apply_filtering(
                            transformed_data, transform_params
                        )
                    elif transform_name == "pivot":
                        transformed_data = self._apply_pivot_transformation(
                            transformed_data, transform_params
                        )
                    elif transform_name == "statistical_summary":
                        transformed_data = self._apply_statistical_summary(
                            transformed_data, transform_params
                        )
                    elif transform_name == "time_series_resample":
                        transformed_data = self._apply_time_series_resampling(
                            transformed_data, transform_params
                        )
                    elif transform_name == "outlier_treatment":
                        transformed_data = self._apply_outlier_treatment(
                            transformed_data, transform_params
                        )
                    elif transform_name == "feature_engineering":
                        transformed_data = self._apply_feature_engineering(
                            transformed_data, transform_params
                        )
                    else:
                        logger.warning(f"Unknown transformation: {transform_name}")
                        continue
                    
                    # Record transformation metadata
                    execution_time = (time.time() - start_time) * 1000
                    metadata["transformations_applied"].append({
                        "name": transform_name,
                        "parameters": transform_params,
                        "execution_time_ms": execution_time,
                        "output_shape": self._get_data_shape(transformed_data)
                    })
                    
                    metadata["performance_metrics"][transform_name] = execution_time
                    
                except Exception as e:
                    error_msg = f"Transformation '{transform_name}' failed: {e}"
                    logger.error(error_msg)
                    metadata["warnings"].append(error_msg)
                    
                    if validation_strict:
                        raise DataTransformationError(error_msg) from e
            
            # Final validation
            if validation_strict:
                self._validate_transformation_output(transformed_data, metadata)
            
            metadata["output_shape"] = self._get_data_shape(transformed_data)
            metadata["output_type"] = type(transformed_data).__name__
            
            logger.debug(f"Successfully applied {len(metadata['transformations_applied'])} transformations")
            return transformed_data, metadata
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise DataTransformationError(f"Failed to transform data: {e}") from e
    
    def _validate_input_data(self, data: Any, strict: bool) -> Any:
        """Validate input data for transformation."""
        if data is None:
            raise DataTransformationError("Input data cannot be None")
        
        # Convert to appropriate format
        if isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except Exception as e:
                if strict:
                    raise DataTransformationError(f"Cannot convert dict to DataFrame: {e}")
                return data
        
        elif isinstance(data, (list, tuple)):
            try:
                return np.array(data)
            except Exception as e:
                if strict:
                    raise DataTransformationError(f"Cannot convert to numpy array: {e}")
                return data
        
        elif isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            return data
        
        else:
            if strict:
                raise DataTransformationError(f"Unsupported data type: {type(data)}")
            return data
    
    def _get_data_shape(self, data: Any) -> Tuple[int, ...]:
        """Get shape of data regardless of type."""
        if hasattr(data, 'shape'):
            return data.shape
        elif hasattr(data, '__len__'):
            return (len(data),)
        else:
            return (1,)
    
    def _apply_normalization(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply normalization transformation."""
        method = params.get("method", "zscore")
        
        if isinstance(data, pd.DataFrame):
            if method == "zscore":
                return (data - data.mean()) / data.std()
            elif method == "minmax":
                return (data - data.min()) / (data.max() - data.min())
            elif method == "robust":
                median = data.median()
                mad = np.abs(data - median).median()
                return (data - median) / mad
            else:
                raise DataTransformationError(f"Unknown normalization method: {method}")
        
        elif isinstance(data, np.ndarray):
            if method == "zscore":
                return stats.zscore(data, axis=0, nan_policy='omit')
            elif method == "minmax":
                min_vals = np.nanmin(data, axis=0)
                max_vals = np.nanmax(data, axis=0)
                return (data - min_vals) / (max_vals - min_vals)
            else:
                raise DataTransformationError(f"Normalization method {method} not supported for arrays")
        
        else:
            raise DataTransformationError("Normalization requires DataFrame or array input")
    
    def _apply_aggregation(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply aggregation transformation."""
        if not isinstance(data, pd.DataFrame):
            raise DataTransformationError("Aggregation requires DataFrame input")
        
        group_by = params.get("group_by", [])
        agg_functions = params.get("functions", {"count": "count"})
        
        if not group_by:
            # Global aggregation
            return data.agg(agg_functions)
        else:
            # Grouped aggregation
            return data.groupby(group_by).agg(agg_functions)
    
    def _apply_filtering(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply filtering transformation."""
        if not isinstance(data, pd.DataFrame):
            raise DataTransformationError("Filtering requires DataFrame input")
        
        filter_conditions = params.get("conditions", [])
        filter_method = params.get("method", "and")
        
        if not filter_conditions:
            return data
        
        # Build combined filter
        combined_filter = None
        
        for condition in filter_conditions:
            column = condition.get("column")
            operator = condition.get("operator", "==")
            value = condition.get("value")
            
            if column not in data.columns:
                logger.warning(f"Filter column '{column}' not found in data")
                continue
            
            # Create condition filter
            if operator == "==":
                condition_filter = data[column] == value
            elif operator == "!=":
                condition_filter = data[column] != value
            elif operator == ">":
                condition_filter = data[column] > value
            elif operator == ">=":
                condition_filter = data[column] >= value
            elif operator == "<":
                condition_filter = data[column] < value
            elif operator == "<=":
                condition_filter = data[column] <= value
            elif operator == "in":
                condition_filter = data[column].isin(value)
            elif operator == "not_in":
                condition_filter = ~data[column].isin(value)
            else:
                logger.warning(f"Unknown filter operator: {operator}")
                continue
            
            # Combine filters
            if combined_filter is None:
                combined_filter = condition_filter
            elif filter_method == "and":
                combined_filter = combined_filter & condition_filter
            elif filter_method == "or":
                combined_filter = combined_filter | condition_filter
        
        if combined_filter is not None:
            return data[combined_filter]
        else:
            return data
    
    def _apply_pivot_transformation(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply pivot transformation."""
        if not isinstance(data, pd.DataFrame):
            raise DataTransformationError("Pivot requires DataFrame input")
        
        index = params.get("index")
        columns = params.get("columns")
        values = params.get("values")
        aggfunc = params.get("aggfunc", "mean")
        
        if not all([index, columns]):
            raise DataTransformationError("Pivot requires both index and columns parameters")
        
        return data.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0
        )
    
    def _apply_statistical_summary(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply statistical summary transformation."""
        if isinstance(data, pd.DataFrame):
            summary_types = params.get("statistics", ["mean", "std", "count"])
            
            summary_data = {}
            for stat in summary_types:
                if stat == "mean":
                    summary_data["mean"] = data.mean()
                elif stat == "std":
                    summary_data["std"] = data.std()
                elif stat == "count":
                    summary_data["count"] = data.count()
                elif stat == "median":
                    summary_data["median"] = data.median()
                elif stat == "min":
                    summary_data["min"] = data.min()
                elif stat == "max":
                    summary_data["max"] = data.max()
                elif stat == "percentiles":
                    percentiles = params.get("percentile_values", [25, 50, 75])
                    for p in percentiles:
                        summary_data[f"p{p}"] = data.quantile(p / 100)
            
            return pd.DataFrame(summary_data)
        
        elif isinstance(data, np.ndarray):
            return {
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0)
            }
        
        else:
            raise DataTransformationError("Statistical summary requires DataFrame or array input")
    
    def _apply_time_series_resampling(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply time series resampling transformation."""
        if not isinstance(data, pd.DataFrame):
            raise DataTransformationError("Time series resampling requires DataFrame input")
        
        time_column = params.get("time_column", "timestamp")
        frequency = params.get("frequency", "D")
        agg_method = params.get("aggregation", "mean")
        
        if time_column not in data.columns:
            raise DataTransformationError(f"Time column '{time_column}' not found in data")
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        # Set time column as index for resampling
        data_resampled = data.set_index(time_column)
        
        # Apply resampling
        if agg_method == "mean":
            return data_resampled.resample(frequency).mean()
        elif agg_method == "sum":
            return data_resampled.resample(frequency).sum()
        elif agg_method == "count":
            return data_resampled.resample(frequency).count()
        elif agg_method == "first":
            return data_resampled.resample(frequency).first()
        elif agg_method == "last":
            return data_resampled.resample(frequency).last()
        else:
            raise DataTransformationError(f"Unknown aggregation method: {agg_method}")
    
    def _apply_outlier_treatment(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply outlier treatment transformation."""
        method = params.get("method", "iqr")
        action = params.get("action", "remove")  # remove, cap, transform
        
        if isinstance(data, pd.DataFrame):
            treated_data = data.copy()
            
            for column in data.select_dtypes(include=[np.number]).columns:
                if method == "iqr":
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                elif method == "zscore":
                    z_scores = np.abs(stats.zscore(data[column], nan_policy='omit'))
                    threshold = params.get("z_threshold", 3.0)
                    outlier_mask = z_scores > threshold
                
                elif method == "isolation_forest":
                    try:
                        from sklearn.ensemble import IsolationForest
                        iso_forest = IsolationForest(contamination=params.get("contamination", 0.1))
                        outlier_labels = iso_forest.fit_predict(data[[column]])
                        outlier_mask = outlier_labels == -1
                    except ImportError:
                        logger.warning("sklearn not available, falling back to IQR method")
                        continue
                
                else:
                    logger.warning(f"Unknown outlier detection method: {method}")
                    continue
                
                # Apply outlier treatment
                if method == "iqr":
                    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                
                if action == "remove":
                    treated_data = treated_data[~outlier_mask]
                elif action == "cap":
                    if method == "iqr":
                        treated_data.loc[data[column] < lower_bound, column] = lower_bound
                        treated_data.loc[data[column] > upper_bound, column] = upper_bound
                elif action == "transform":
                    # Log transformation for positive outliers
                    treated_data.loc[outlier_mask, column] = np.log1p(
                        treated_data.loc[outlier_mask, column]
                    )
            
            return treated_data
        
        else:
            raise DataTransformationError("Outlier treatment requires DataFrame input")
    
    def _apply_feature_engineering(self, data: Any, params: Dict[str, Any]) -> Any:
        """Apply feature engineering transformation."""
        if not isinstance(data, pd.DataFrame):
            raise DataTransformationError("Feature engineering requires DataFrame input")
        
        engineered_data = data.copy()
        feature_specs = params.get("features", [])
        
        for feature_spec in feature_specs:
            feature_name = feature_spec.get("name")
            feature_type = feature_spec.get("type")
            feature_params = feature_spec.get("parameters", {})
            
            try:
                if feature_type == "polynomial":
                    # Create polynomial features
                    source_columns = feature_params.get("columns", [])
                    degree = feature_params.get("degree", 2)
                    
                    for col in source_columns:
                        if col in engineered_data.columns:
                            for d in range(2, degree + 1):
                                new_col_name = f"{col}_power_{d}"
                                engineered_data[new_col_name] = engineered_data[col] ** d
                
                elif feature_type == "interaction":
                    # Create interaction features
                    col1 = feature_params.get("column1")
                    col2 = feature_params.get("column2")
                    
                    if col1 in engineered_data.columns and col2 in engineered_data.columns:
                        interaction_name = feature_name or f"{col1}_{col2}_interaction"
                        engineered_data[interaction_name] = (
                            engineered_data[col1] * engineered_data[col2]
                        )
                
                elif feature_type == "binning":
                    # Create binned features
                    source_column = feature_params.get("column")
                    bins = feature_params.get("bins", 5)
                    
                    if source_column in engineered_data.columns:
                        bin_name = feature_name or f"{source_column}_binned"
                        engineered_data[bin_name] = pd.cut(
                            engineered_data[source_column], 
                            bins=bins, 
                            labels=False
                        )
                
                elif feature_type == "rolling_stats":
                    # Create rolling statistics
                    source_column = feature_params.get("column")
                    window = feature_params.get("window", 5)
                    stat_type = feature_params.get("statistic", "mean")
                    
                    if source_column in engineered_data.columns:
                        stat_name = feature_name or f"{source_column}_{stat_type}_{window}"
                        
                        if stat_type == "mean":
                            engineered_data[stat_name] = (
                                engineered_data[source_column].rolling(window).mean()
                            )
                        elif stat_type == "std":
                            engineered_data[stat_name] = (
                                engineered_data[source_column].rolling(window).std()
                            )
                        elif stat_type == "min":
                            engineered_data[stat_name] = (
                                engineered_data[source_column].rolling(window).min()
                            )
                        elif stat_type == "max":
                            engineered_data[stat_name] = (
                                engineered_data[source_column].rolling(window).max()
                            )
                
                logger.debug(f"Created feature: {feature_name}")
                
            except Exception as e:
                logger.warning(f"Failed to create feature {feature_name}: {e}")
        
        return engineered_data
    
    def _validate_transformation_output(self, data: Any, metadata: Dict[str, Any]) -> None:
        """Validate transformation output."""
        if data is None:
            raise DataTransformationError("Transformation produced None output")
        
        # Check for empty data
        if hasattr(data, '__len__') and len(data) == 0:
            logger.warning("Transformation produced empty dataset")
        
        # Check for infinite or NaN values if numeric data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.isnull().any().any():
                logger.warning("Transformation produced NaN values")
            
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty and np.isinf(numeric_data).any().any():
                logger.warning("Transformation produced infinite values")
        
        elif isinstance(data, np.ndarray):
            if np.isnan(data).any():
                logger.warning("Transformation produced NaN values")
            if np.isinf(data).any():
                logger.warning("Transformation produced infinite values")


class StatisticalAnalysisHelper:
    """
    Statistical analysis utilities for advanced reporting pipeline visualizations.
    
    This class provides production-ready statistical analysis functions optimized
    for enterprise data science workflows with comprehensive error handling,
    performance monitoring, and advanced analytical capabilities.
    """
    
    def __init__(self, precision: float = STATISTICAL_ANALYSIS_PRECISION):
        """
        Initialize statistical analysis helper.
        
        Args:
            precision: Numerical precision for statistical calculations
        """
        self.precision = precision
        self.analysis_cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        
        logger.info("Initialized StatisticalAnalysisHelper")
    
    @performance_monitor("statistical_analysis", target_ms=30.0)
    def perform_comprehensive_analysis(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        analysis_config: Dict[str, Any],
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis for visualization support.
        
        This method provides advanced statistical analysis capabilities supporting
        enterprise reporting pipelines with comprehensive hypothesis testing,
        distribution analysis, and correlation studies.
        
        Args:
            data: Input data for analysis
            analysis_config: Configuration specifying analyses to perform
            significance_level: Significance level for hypothesis tests
            
        Returns:
            Dictionary containing comprehensive analysis results
            
        Raises:
            StatisticalAnalysisError: When statistical analysis fails
        """
        try:
            logger.debug(f"Performing statistical analysis with config: {list(analysis_config.keys())}")
            
            # Initialize results dictionary
            results = {
                "descriptive_statistics": {},
                "hypothesis_tests": {},
                "correlation_analysis": {},
                "distribution_analysis": {},
                "outlier_analysis": {},
                "time_series_analysis": {},
                "regression_analysis": {},
                "metadata": {
                    "input_shape": self._get_data_shape(data),
                    "significance_level": significance_level,
                    "analysis_timestamp": time.time()
                }
            }
            
            # Validate input data
            validated_data = self._validate_analysis_data(data)
            
            # Perform requested analyses
            for analysis_type, analysis_params in analysis_config.items():
                try:
                    if analysis_type == "descriptive":
                        results["descriptive_statistics"] = self._compute_descriptive_statistics(
                            validated_data, analysis_params
                        )
                    
                    elif analysis_type == "hypothesis_tests":
                        results["hypothesis_tests"] = self._perform_hypothesis_tests(
                            validated_data, analysis_params, significance_level
                        )
                    
                    elif analysis_type == "correlation":
                        results["correlation_analysis"] = self._analyze_correlations(
                            validated_data, analysis_params
                        )
                    
                    elif analysis_type == "distribution":
                        results["distribution_analysis"] = self._analyze_distributions(
                            validated_data, analysis_params
                        )
                    
                    elif analysis_type == "outliers":
                        results["outlier_analysis"] = self._detect_outliers(
                            validated_data, analysis_params
                        )
                    
                    elif analysis_type == "time_series":
                        results["time_series_analysis"] = self._analyze_time_series(
                            validated_data, analysis_params
                        )
                    
                    elif analysis_type == "regression":
                        results["regression_analysis"] = self._perform_regression_analysis(
                            validated_data, analysis_params
                        )
                    
                    else:
                        logger.warning(f"Unknown analysis type: {analysis_type}")
                
                except Exception as e:
                    error_msg = f"Analysis '{analysis_type}' failed: {e}"
                    logger.error(error_msg)
                    results[f"{analysis_type}_error"] = error_msg
            
            logger.debug(f"Completed statistical analysis with {len(results)} result sections")
            return results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            raise StatisticalAnalysisError(f"Failed to perform statistical analysis: {e}") from e
    
    def _validate_analysis_data(self, data: Any) -> Union[pd.DataFrame, np.ndarray]:
        """Validate data for statistical analysis."""
        if data is None:
            raise StatisticalAnalysisError("Analysis data cannot be None")
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise StatisticalAnalysisError("DataFrame is empty")
            return data
        
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                raise StatisticalAnalysisError("Array is empty")
            return data
        
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        
        else:
            raise StatisticalAnalysisError(f"Unsupported data type: {type(data)}")
    
    def _get_data_shape(self, data: Any) -> Tuple[int, ...]:
        """Get shape of data for analysis."""
        if hasattr(data, 'shape'):
            return data.shape
        elif hasattr(data, '__len__'):
            return (len(data),)
        else:
            return (1,)
    
    def _compute_descriptive_statistics(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute comprehensive descriptive statistics."""
        statistics = {}
        
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                col_data = data[column].dropna()
                
                if len(col_data) == 0:
                    continue
                
                statistics[column] = {
                    "count": len(col_data),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "var": float(col_data.var()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "range": float(col_data.max() - col_data.min()),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                    "percentiles": {
                        "p25": float(col_data.quantile(0.25)),
                        "p50": float(col_data.quantile(0.50)),
                        "p75": float(col_data.quantile(0.75)),
                        "p90": float(col_data.quantile(0.90)),
                        "p95": float(col_data.quantile(0.95)),
                        "p99": float(col_data.quantile(0.99))
                    }
                }
                
                # Add advanced statistics if requested
                if params.get("include_advanced", False):
                    statistics[column].update({
                        "coefficient_of_variation": statistics[column]["std"] / statistics[column]["mean"],
                        "mad": float(np.median(np.abs(col_data - col_data.median()))),
                        "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25))
                    })
        
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                clean_data = data[~np.isnan(data)]
                statistics["array"] = {
                    "count": len(clean_data),
                    "mean": float(np.mean(clean_data)),
                    "median": float(np.median(clean_data)),
                    "std": float(np.std(clean_data)),
                    "var": float(np.var(clean_data)),
                    "min": float(np.min(clean_data)),
                    "max": float(np.max(clean_data)),
                    "range": float(np.max(clean_data) - np.min(clean_data))
                }
            else:
                # Multi-dimensional array statistics
                for i in range(data.shape[1]):
                    col_data = data[:, i]
                    clean_data = col_data[~np.isnan(col_data)]
                    
                    statistics[f"column_{i}"] = {
                        "count": len(clean_data),
                        "mean": float(np.mean(clean_data)),
                        "std": float(np.std(clean_data)),
                        "min": float(np.min(clean_data)),
                        "max": float(np.max(clean_data))
                    }
        
        return statistics
    
    def _perform_hypothesis_tests(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        params: Dict[str, Any],
        significance_level: float
    ) -> Dict[str, Any]:
        """Perform various hypothesis tests."""
        test_results = {}
        test_types = params.get("tests", ["normality", "t_test"])
        
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for test_type in test_types:
                test_results[test_type] = {}
                
                if test_type == "normality":
                    # Shapiro-Wilk test for normality
                    for column in numeric_columns:
                        col_data = data[column].dropna()
                        
                        if len(col_data) < 3:
                            continue
                        
                        try:
                            statistic, p_value = stats.shapiro(col_data[:5000])  # Limit for performance
                            test_results[test_type][column] = {
                                "statistic": float(statistic),
                                "p_value": float(p_value),
                                "is_normal": p_value > significance_level,
                                "test_name": "Shapiro-Wilk"
                            }
                        except Exception as e:
                            logger.warning(f"Normality test failed for {column}: {e}")
                
                elif test_type == "t_test":
                    # One-sample t-test against specified mean
                    test_mean = params.get("test_mean", 0.0)
                    
                    for column in numeric_columns:
                        col_data = data[column].dropna()
                        
                        if len(col_data) < 2:
                            continue
                        
                        try:
                            statistic, p_value = stats.ttest_1samp(col_data, test_mean)
                            test_results[test_type][column] = {
                                "statistic": float(statistic),
                                "p_value": float(p_value),
                                "is_significant": p_value < significance_level,
                                "test_mean": test_mean,
                                "sample_mean": float(col_data.mean()),
                                "test_name": "One-sample t-test"
                            }
                        except Exception as e:
                            logger.warning(f"T-test failed for {column}: {e}")
                
                elif test_type == "anova":
                    # One-way ANOVA if grouping variable specified
                    group_column = params.get("group_column")
                    value_column = params.get("value_column")
                    
                    if group_column and value_column and both in data.columns:
                        try:
                            groups = [group[value_column].dropna() for name, group in data.groupby(group_column)]
                            
                            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                                statistic, p_value = stats.f_oneway(*groups)
                                test_results[test_type]["anova_result"] = {
                                    "statistic": float(statistic),
                                    "p_value": float(p_value),
                                    "is_significant": p_value < significance_level,
                                    "groups_count": len(groups),
                                    "test_name": "One-way ANOVA"
                                }
                        except Exception as e:
                            logger.warning(f"ANOVA test failed: {e}")
        
        return test_results
    
    def _analyze_correlations(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        correlation_results = {}
        
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return {"error": "Insufficient numeric columns for correlation analysis"}
            
            # Compute correlation matrices
            methods = params.get("methods", ["pearson", "spearman"])
            
            for method in methods:
                try:
                    if method == "pearson":
                        corr_matrix = numeric_data.corr(method='pearson')
                    elif method == "spearman":
                        corr_matrix = numeric_data.corr(method='spearman')
                    elif method == "kendall":
                        corr_matrix = numeric_data.corr(method='kendall')
                    else:
                        continue
                    
                    # Convert to dictionary format
                    correlation_results[method] = {
                        "matrix": corr_matrix.to_dict(),
                        "strong_correlations": self._find_strong_correlations(
                            corr_matrix, params.get("correlation_threshold", 0.7)
                        )
                    }
                    
                except Exception as e:
                    logger.warning(f"Correlation analysis failed for method {method}: {e}")
        
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            # Numpy array correlation
            try:
                corr_matrix = np.corrcoef(data.T)
                correlation_results["pearson"] = {
                    "matrix": corr_matrix.tolist(),
                    "shape": corr_matrix.shape
                }
            except Exception as e:
                logger.warning(f"Array correlation analysis failed: {e}")
        
        return correlation_results
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Find strong correlations in correlation matrix."""
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        "variable_1": corr_matrix.columns[i],
                        "variable_2": corr_matrix.columns[j],
                        "correlation": float(corr_value),
                        "strength": "strong" if abs(corr_value) >= 0.8 else "moderate"
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return strong_correlations
    
    def _analyze_distributions(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data distributions."""
        distribution_results = {}
        
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                col_data = data[column].dropna()
                
                if len(col_data) < 10:
                    continue
                
                try:
                    # Test against common distributions
                    distribution_fits = {}
                    
                    # Test normal distribution
                    normal_params = stats.norm.fit(col_data)
                    ks_stat, ks_p = stats.kstest(col_data, lambda x: stats.norm.cdf(x, *normal_params))
                    distribution_fits["normal"] = {
                        "parameters": normal_params,
                        "ks_statistic": float(ks_stat),
                        "ks_p_value": float(ks_p),
                        "fits_well": ks_p > 0.05
                    }
                    
                    # Test exponential distribution
                    if (col_data >= 0).all():
                        exp_params = stats.expon.fit(col_data)
                        ks_stat, ks_p = stats.kstest(col_data, lambda x: stats.expon.cdf(x, *exp_params))
                        distribution_fits["exponential"] = {
                            "parameters": exp_params,
                            "ks_statistic": float(ks_stat),
                            "ks_p_value": float(ks_p),
                            "fits_well": ks_p > 0.05
                        }
                    
                    # Test log-normal distribution
                    if (col_data > 0).all():
                        lognorm_params = stats.lognorm.fit(col_data)
                        ks_stat, ks_p = stats.kstest(col_data, lambda x: stats.lognorm.cdf(x, *lognorm_params))
                        distribution_fits["lognormal"] = {
                            "parameters": lognorm_params,
                            "ks_statistic": float(ks_stat),
                            "ks_p_value": float(ks_p),
                            "fits_well": ks_p > 0.05
                        }
                    
                    distribution_results[column] = distribution_fits
                    
                except Exception as e:
                    logger.warning(f"Distribution analysis failed for {column}: {e}")
        
        return distribution_results
    
    def _detect_outliers(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        outlier_results = {}
        methods = params.get("methods", ["iqr", "zscore"])
        
        if isinstance(data, pd.DataFrame):
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                col_data = data[column].dropna()
                
                if len(col_data) < 4:
                    continue
                
                outlier_results[column] = {}
                
                for method in methods:
                    try:
                        if method == "iqr":
                            Q1 = col_data.quantile(0.25)
                            Q3 = col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                            outlier_results[column][method] = {
                                "outlier_count": len(outliers),
                                "outlier_percentage": (len(outliers) / len(col_data)) * 100,
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound),
                                "outlier_values": outliers.tolist()[:10]  # Limit for performance
                            }
                        
                        elif method == "zscore":
                            z_scores = np.abs(stats.zscore(col_data))
                            threshold = params.get("z_threshold", 3.0)
                            outliers = col_data[z_scores > threshold]
                            
                            outlier_results[column][method] = {
                                "outlier_count": len(outliers),
                                "outlier_percentage": (len(outliers) / len(col_data)) * 100,
                                "threshold": threshold,
                                "outlier_values": outliers.tolist()[:10]
                            }
                        
                        elif method == "modified_zscore":
                            median = col_data.median()
                            mad = np.median(np.abs(col_data - median))
                            modified_z_scores = 0.6745 * (col_data - median) / mad
                            threshold = params.get("modified_z_threshold", 3.5)
                            outliers = col_data[np.abs(modified_z_scores) > threshold]
                            
                            outlier_results[column][method] = {
                                "outlier_count": len(outliers),
                                "outlier_percentage": (len(outliers) / len(col_data)) * 100,
                                "threshold": threshold,
                                "outlier_values": outliers.tolist()[:10]
                            }
                    
                    except Exception as e:
                        logger.warning(f"Outlier detection method {method} failed for {column}: {e}")
        
        return outlier_results
    
    def _analyze_time_series(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze time series properties."""
        time_series_results = {}
        
        if isinstance(data, pd.DataFrame):
            time_column = params.get("time_column")
            value_columns = params.get("value_columns", [])
            
            if not time_column or time_column not in data.columns:
                return {"error": "Time column not specified or not found"}
            
            if not value_columns:
                value_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for value_column in value_columns:
                if value_column not in data.columns:
                    continue
                
                try:
                    # Sort by time
                    ts_data = data[[time_column, value_column]].sort_values(time_column)
                    ts_data = ts_data.dropna()
                    
                    if len(ts_data) < 10:
                        continue
                    
                    values = ts_data[value_column].values
                    
                    # Basic time series statistics
                    time_series_results[value_column] = {
                        "length": len(values),
                        "mean": float(np.mean(values)),
                        "trend": self._calculate_trend(values),
                        "seasonality": self._detect_seasonality(values, params),
                        "stationarity": self._test_stationarity(values),
                        "autocorrelation": self._calculate_autocorrelation(values, params)
                    }
                    
                except Exception as e:
                    logger.warning(f"Time series analysis failed for {value_column}: {e}")
        
        return time_series_results
    
    def _calculate_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate trend in time series."""
        try:
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            return {
                "slope": float(slope),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05,
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            }
        except Exception:
            return {"error": "Trend calculation failed"}
    
    def _detect_seasonality(self, values: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect seasonality in time series."""
        try:
            # Simple autocorrelation-based seasonality detection
            period = params.get("seasonal_period", 12)
            
            if len(values) < 2 * period:
                return {"error": "Insufficient data for seasonality detection"}
            
            # Calculate autocorrelation at seasonal lag
            if len(values) > period:
                seasonal_autocorr = np.corrcoef(values[:-period], values[period:])[0, 1]
                
                return {
                    "seasonal_period": period,
                    "seasonal_autocorrelation": float(seasonal_autocorr),
                    "has_seasonality": abs(seasonal_autocorr) > 0.3
                }
            else:
                return {"error": "Insufficient data length"}
                
        except Exception:
            return {"error": "Seasonality detection failed"}
    
    def _test_stationarity(self, values: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        try:
            # Simple variance-based stationarity test
            # Split series into two halves and compare variances
            mid = len(values) // 2
            first_half_var = np.var(values[:mid])
            second_half_var = np.var(values[mid:])
            
            # F-test for equality of variances
            f_stat = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
            
            return {
                "variance_ratio": float(f_stat),
                "is_stationary": f_stat < 2.0,  # Simple threshold
                "first_half_variance": float(first_half_var),
                "second_half_variance": float(second_half_var)
            }
            
        except Exception:
            return {"error": "Stationarity test failed"}
    
    def _calculate_autocorrelation(self, values: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate autocorrelation function."""
        try:
            max_lags = min(params.get("max_lags", 20), len(values) // 4)
            
            autocorrelations = []
            for lag in range(1, max_lags + 1):
                if len(values) > lag:
                    autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    autocorrelations.append(float(autocorr))
            
            return {
                "autocorrelations": autocorrelations,
                "max_autocorr": float(max(autocorrelations)) if autocorrelations else 0.0,
                "significant_lags": [
                    i + 1 for i, ac in enumerate(autocorrelations) 
                    if abs(ac) > 0.2
                ]
            }
            
        except Exception:
            return {"error": "Autocorrelation calculation failed"}
    
    def _perform_regression_analysis(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform regression analysis."""
        regression_results = {}
        
        if isinstance(data, pd.DataFrame):
            target_column = params.get("target_column")
            feature_columns = params.get("feature_columns", [])
            
            if not target_column or target_column not in data.columns:
                return {"error": "Target column not specified or not found"}
            
            if not feature_columns:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != target_column]
            
            try:
                # Prepare data
                analysis_data = data[feature_columns + [target_column]].dropna()
                
                if len(analysis_data) < 10:
                    return {"error": "Insufficient data for regression analysis"}
                
                X = analysis_data[feature_columns].values
                y = analysis_data[target_column].values
                
                # Simple linear regression for each feature
                for feature in feature_columns:
                    try:
                        feature_data = analysis_data[[feature, target_column]].dropna()
                        x_vals = feature_data[feature].values
                        y_vals = feature_data[target_column].values
                        
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                        
                        regression_results[f"{feature}_vs_{target_column}"] = {
                            "slope": float(slope),
                            "intercept": float(intercept),
                            "r_squared": float(r_value**2),
                            "p_value": float(p_value),
                            "standard_error": float(std_err),
                            "is_significant": p_value < 0.05
                        }
                        
                    except Exception as e:
                        logger.warning(f"Regression failed for {feature}: {e}")
                
                # Multiple regression summary if multiple features
                if len(feature_columns) > 1:
                    try:
                        # Calculate multiple correlation coefficient
                        corr_matrix = analysis_data.corr()
                        target_corrs = corr_matrix[target_column][feature_columns]
                        
                        regression_results["multiple_regression"] = {
                            "feature_correlations": target_corrs.to_dict(),
                            "max_correlation": float(target_corrs.abs().max()),
                            "feature_count": len(feature_columns)
                        }
                        
                    except Exception as e:
                        logger.warning(f"Multiple regression analysis failed: {e}")
                
            except Exception as e:
                logger.error(f"Regression analysis failed: {e}")
                return {"error": f"Regression analysis failed: {e}"}
        
        return regression_results


@contextmanager
def performance_context(operation_name: str, target_ms: float = DEFAULT_PERFORMANCE_TARGET_MS):
    """
    Context manager for performance monitoring with enterprise-grade tracking.
    
    This context manager provides comprehensive performance tracking for
    enterprise operations with automatic metric collection and alerting
    when performance targets are exceeded.
    
    Args:
        operation_name: Name of the operation being monitored
        target_ms: Target execution time in milliseconds
        
    Yields:
        Performance tracking context
    """
    start_time = time.time()
    success = False
    
    try:
        yield
        success = True
    finally:
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Store metrics thread-safely
        with _cache_lock:
            _performance_metrics[operation_name].append(execution_time_ms)
            # Limit metrics storage
            if len(_performance_metrics[operation_name]) > 1000:
                _performance_metrics[operation_name] = _performance_metrics[operation_name][-500:]
        
        # Log performance information
        if execution_time_ms > target_ms:
            logger.warning(
                f"Operation '{operation_name}' took {execution_time_ms:.2f}ms, "
                f"exceeding target {target_ms}ms (success={success})"
            )
        else:
            logger.debug(
                f"Operation '{operation_name}' completed in {execution_time_ms:.2f}ms (success={success})"
            )


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive performance metrics for all monitored operations.
    
    Returns:
        Dictionary containing performance statistics and metrics
    """
    with _cache_lock:
        metrics = {}
        
        for operation_name, times in _performance_metrics.items():
            if times:
                metrics[operation_name] = {
                    "count": len(times),
                    "average_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "median_ms": sorted(times)[len(times) // 2],
                    "p95_ms": sorted(times)[int(len(times) * 0.95)],
                    "total_time_ms": sum(times)
                }
        
        return {
            "operation_metrics": metrics,
            "total_operations": sum(len(times) for times in _performance_metrics.values()),
            "cache_sizes": {
                "condition_cache": len(_condition_resolution_cache),
                "statistical_cache": len(_statistical_cache)
            }
        }


def clear_all_caches() -> None:
    """Clear all utility caches for memory management."""
    with _cache_lock:
        _condition_resolution_cache.clear()
        _statistical_cache.clear()
        _performance_metrics.clear()
    
    logger.info("Cleared all utility caches")


# Export public API
__all__ = [
    # Core classes
    "AdvancedConditionResolver",
    "MultiEnvironmentConfigManager", 
    "AdvancedDataTransformer",
    "StatisticalAnalysisHelper",
    
    # Data structures
    "ExperimentalCondition",
    "EnvironmentConfiguration",
    
    # Decorators and context managers
    "performance_monitor",
    "performance_context",
    
    # Utility functions
    "get_performance_metrics",
    "clear_all_caches",
    
    # Exception classes
    "ConditionResolutionError",
    "ConfigurationEnvironmentError",
    "StatisticalAnalysisError",
    "DataTransformationError"
]