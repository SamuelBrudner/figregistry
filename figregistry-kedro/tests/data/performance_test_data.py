"""
Performance testing data generators and utilities for figregistry-kedro plugin.

This module provides comprehensive data generation capabilities for performance testing and
benchmarking of the figregistry-kedro plugin operations against technical specification 
requirements per Section 6.6.4.3:

Performance Targets:
- Plugin Pipeline Execution Overhead: <200ms per FigureDataSet save
- Configuration Bridge Merge Time: <50ms per pipeline run  
- Hook Initialization Overhead: <25ms per project startup
- Plugin Memory Footprint: <5MB overhead

Key Capabilities:
- Large-scale configuration scenario generation for config bridge performance testing
- High-volume catalog entry generation for concurrent execution validation
- Complex matplotlib figure generation with varying complexity levels
- Concurrent execution test data for parallel Kedro runner scenarios
- Memory usage profiling scenarios for plugin footprint validation
- Precision timing utilities for performance measurement and SLA validation
- Stress testing data generators for high-load scenario validation

This module integrates with pytest-benchmark for automated performance regression testing
and provides utilities for measuring plugin overhead against manual matplotlib operations.
"""

import gc
import time
import threading
import multiprocessing
import psutil
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator, Callable, Generator, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# =============================================================================
# PERFORMANCE TESTING CONFIGURATION
# =============================================================================

@dataclass
class PerformanceTargets:
    """Performance targets from Section 6.6.4.3 for validation."""
    
    # Plugin-specific performance targets (milliseconds)
    figuredataset_save_overhead: float = 200.0  # <200ms per FigureDataSet save
    config_bridge_resolution: float = 50.0     # <50ms per pipeline run
    hook_initialization: float = 25.0          # <25ms per project startup
    
    # Memory targets (megabytes)
    plugin_memory_overhead: float = 5.0        # <5MB plugin overhead
    max_concurrent_figures: int = 10           # Maximum concurrent open figures
    
    # Core performance targets for comparison
    configuration_load: float = 100.0          # <100ms config load
    style_lookup: float = 1.0                  # <1ms style lookup
    file_io_operation: float = 50.0            # <50ms file I/O
    api_overhead: float = 10.0                 # <10ms API overhead


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def meets_target(self, target_ms: float) -> bool:
        """Check if execution time meets target."""
        return self.execution_time_ms <= target_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'operation_name': self.operation_name,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


# =============================================================================
# LARGE CONFIGURATION GENERATORS
# =============================================================================

def large_config_generators() -> Iterator[Tuple[str, Dict[str, Any], float]]:
    """
    Generate large-scale configuration scenarios for testing configuration bridge 
    merge performance per Section 6.6.4.3.
    
    Creates progressively complex configuration scenarios to validate that
    FigRegistryConfigBridge resolution remains under 50ms target even with
    enterprise-scale configuration complexity.
    
    Yields:
        Tuple of (scenario_name, config_dict, expected_time_ms)
    """
    
    # Small baseline configuration
    yield ("baseline_small", {
        "figregistry_version": "0.3.0",
        "styles": {
            "exploratory": {"color": "#1f77b4", "marker": "o"},
            "presentation": {"color": "#ff7f0e", "marker": "s"},
            "publication": {"color": "#2ca02c", "marker": "^"}
        },
        "outputs": {
            "base_path": "data/08_reporting",
            "format": "png",
            "dpi": 300
        }
    }, 5.0)
    
    # Medium complexity with multiple conditions
    medium_styles = {}
    for i in range(50):
        medium_styles[f"condition_{i}"] = {
            "color": f"#{random.randint(0, 16777215):06x}",
            "marker": random.choice(["o", "s", "^", "v", "D", "p", "*"]),
            "linestyle": random.choice(["-", "--", "-.", ":"]),
            "linewidth": random.uniform(0.5, 3.0),
            "markersize": random.uniform(4, 12)
        }
    
    yield ("medium_complexity", {
        "figregistry_version": "0.3.0",
        "styles": medium_styles,
        "outputs": {
            "base_path": "data/08_reporting",
            "format": "png",
            "dpi": 300,
            "timestamp": True,
            "versioning": "kedro"
        },
        "environments": {
            "development": {"dpi": 150},
            "staging": {"dpi": 200},
            "production": {"dpi": 300}
        }
    }, 15.0)
    
    # Large enterprise-scale configuration
    large_styles = {}
    for category in ["exploratory", "presentation", "publication", "report", "thesis"]:
        for experiment in range(100):
            for variant in ["default", "high_contrast", "colorblind", "print"]:
                condition = f"{category}_{experiment}_{variant}"
                large_styles[condition] = {
                    "color": f"#{random.randint(0, 16777215):06x}",
                    "marker": random.choice(["o", "s", "^", "v", "D", "p", "*", "h", "H", "+"]),
                    "linestyle": random.choice(["-", "--", "-.", ":"]),
                    "linewidth": random.uniform(0.5, 5.0),
                    "markersize": random.uniform(2, 15),
                    "alpha": random.uniform(0.3, 1.0),
                    "markerfacecolor": f"#{random.randint(0, 16777215):06x}",
                    "markeredgecolor": f"#{random.randint(0, 16777215):06x}",
                    "markeredgewidth": random.uniform(0.1, 2.0)
                }
    
    # Complex nested configuration with multiple environments
    environment_configs = {}
    for env in ["development", "testing", "staging", "production", "research"]:
        environment_configs[env] = {
            "outputs": {
                "base_path": f"data/08_reporting/{env}",
                "dpi": random.choice([150, 200, 300, 600]),
                "format": random.choice(["png", "pdf", "svg"]),
                "transparent": random.choice([True, False])
            },
            "style_overrides": {
                condition: {
                    "dpi": random.choice([150, 200, 300]),
                    "format": random.choice(["png", "pdf"])
                } for condition in random.sample(list(large_styles.keys()), 50)
            }
        }
    
    yield ("enterprise_large", {
        "figregistry_version": "0.3.0",
        "styles": large_styles,
        "outputs": {
            "base_path": "data/08_reporting",
            "format": "png",
            "dpi": 300,
            "timestamp": True,
            "versioning": "kedro",
            "path_aliases": {
                alias: f"path/to/{alias}" for alias in 
                [''.join(random.choices(string.ascii_lowercase, k=8)) for _ in range(100)]
            }
        },
        "environments": environment_configs,
        "metadata": {
            "created_by": "performance_test_generator",
            "complexity_level": "enterprise_large",
            "style_count": len(large_styles),
            "environment_count": len(environment_configs)
        }
    }, 40.0)
    
    # Stress test configuration - maximum complexity
    stress_styles = {}
    for i in range(1000):
        condition_name = f"stress_test_condition_{i:04d}"
        stress_styles[condition_name] = {
            # Full matplotlib rcParams coverage
            "figure.figsize": [random.uniform(4, 16), random.uniform(3, 12)],
            "figure.dpi": random.choice([100, 150, 200, 300, 600]),
            "figure.facecolor": f"#{random.randint(0, 16777215):06x}",
            "figure.edgecolor": f"#{random.randint(0, 16777215):06x}",
            "axes.linewidth": random.uniform(0.5, 3.0),
            "axes.spines.left": random.choice([True, False]),
            "axes.spines.bottom": random.choice([True, False]),
            "axes.spines.top": random.choice([True, False]),
            "axes.spines.right": random.choice([True, False]),
            "axes.facecolor": f"#{random.randint(0, 16777215):06x}",
            "axes.edgecolor": f"#{random.randint(0, 16777215):06x}",
            "axes.labelsize": random.uniform(8, 16),
            "axes.titlesize": random.uniform(10, 20),
            "xtick.labelsize": random.uniform(6, 14),
            "ytick.labelsize": random.uniform(6, 14),
            "legend.fontsize": random.uniform(8, 14),
            "font.family": random.choice(["sans-serif", "serif", "monospace"]),
            "font.size": random.uniform(8, 16),
            "lines.linewidth": random.uniform(0.5, 4.0),
            "lines.markersize": random.uniform(2, 12),
            "grid.alpha": random.uniform(0.1, 0.8),
            "grid.linewidth": random.uniform(0.3, 1.5)
        }
    
    yield ("stress_maximum", {
        "figregistry_version": "0.3.0",
        "styles": stress_styles,
        "outputs": {
            "base_path": "data/08_reporting",
            "format": "png",
            "dpi": 300,
            "timestamp": True,
            "versioning": "kedro",
            "path_aliases": {
                f"alias_{i:04d}": f"path/level1/level2/level3/{i:04d}"
                for i in range(500)
            }
        },
        "environments": {
            f"env_{i:02d}": {
                "outputs": {"dpi": random.choice([150, 300, 600])},
                "style_overrides": {
                    f"stress_test_condition_{j:04d}": {"alpha": random.uniform(0.1, 1.0)}
                    for j in random.sample(range(1000), 100)
                }
            } for i in range(20)
        },
        "metadata": {
            "complexity_level": "stress_maximum",
            "total_conditions": 1000,
            "total_environments": 20,
            "expected_resolution_time_ms": 45.0
        }
    }, 45.0)


def generate_kedro_config_scenarios() -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Generate Kedro-specific configuration scenarios for testing config bridge merge behavior.
    
    Creates scenarios testing the merge behavior between Kedro's environment-specific
    configuration system and FigRegistry's traditional YAML configuration structure.
    
    Yields:
        Tuple of (scenario_name, kedro_config, figregistry_config)
    """
    
    # Basic merge scenario
    yield ("basic_merge", {
        # Kedro conf/base/figregistry.yml
        "figregistry": {
            "styles": {
                "default": {"color": "#1f77b4", "marker": "o"}
            },
            "outputs": {
                "base_path": "${base.data_path}/08_reporting",
                "dpi": 300
            }
        }
    }, {
        # Traditional figregistry.yaml
        "figregistry_version": "0.3.0",
        "styles": {
            "default": {"color": "#ff7f0e", "marker": "s"},  # Should be overridden
            "exploratory": {"color": "#2ca02c", "marker": "^"}
        },
        "outputs": {
            "format": "png",
            "timestamp": True
        }
    })
    
    # Complex environment override scenario
    yield ("environment_override", {
        # Kedro configuration with environment-specific overrides
        "figregistry": {
            "styles": {
                "development": {"color": "#ff0000", "dpi": 150},
                "production": {"color": "#0000ff", "dpi": 600}
            },
            "outputs": {
                "base_path": "${runtime.environment_path}/figures",
                "format": "pdf"
            },
            "environment_specific": {
                "local": {
                    "outputs": {"dpi": 100, "format": "png"}
                },
                "production": {
                    "outputs": {"dpi": 600, "format": "pdf"}
                }
            }
        }
    }, {
        # FigRegistry base configuration
        "figregistry_version": "0.3.0",
        "styles": {
            "default": {"color": "#888888", "marker": "o"},
            "development": {"color": "#00ff00", "dpi": 200},  # Should be overridden
            "production": {"color": "#ffff00", "dpi": 300}    # Should be overridden
        },
        "outputs": {
            "base_path": "default/path",
            "dpi": 300,
            "timestamp": False
        }
    })
    
    # Parameter substitution scenario
    yield ("parameter_substitution", {
        # Kedro with parameter substitution
        "figregistry": {
            "styles": {
                "${experiment.type}": {
                    "color": "${experiment.color}",
                    "marker": "${experiment.marker}",
                    "linewidth": "${experiment.linewidth}"
                }
            },
            "outputs": {
                "base_path": "${paths.reporting}/${experiment.name}",
                "dpi": "${output.dpi}",
                "format": "${output.format}"
            }
        },
        "experiment": {
            "type": "machine_learning",
            "name": "model_evaluation",
            "color": "#e74c3c",
            "marker": "D",
            "linewidth": 2.5
        },
        "paths": {
            "reporting": "data/08_reporting"
        },
        "output": {
            "dpi": 300,
            "format": "svg"
        }
    }, {
        # Static FigRegistry configuration
        "figregistry_version": "0.3.0",
        "styles": {
            "machine_learning": {"color": "#3498db", "marker": "o"},  # Should be overridden
            "baseline": {"color": "#95a5a6", "marker": "s"}
        },
        "outputs": {
            "base_path": "static/path",
            "dpi": 150,
            "format": "png"
        }
    })


# =============================================================================
# HIGH-VOLUME CATALOG GENERATORS
# =============================================================================

def high_volume_catalog_generators() -> Iterator[Tuple[str, List[Dict[str, Any]], int]]:
    """
    Generate high-volume catalog entries for concurrent execution testing per Section 5.2.8.
    
    Creates large numbers of FigureDataSet catalog entries to validate plugin performance
    under concurrent execution scenarios with parallel Kedro runners.
    
    Yields:
        Tuple of (scenario_name, catalog_entries_list, expected_concurrent_load)
    """
    
    # Small concurrent load - 10 concurrent figures
    small_entries = []
    for i in range(10):
        small_entries.append({
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": f"data/08_reporting/concurrent_small_{i:02d}.png",
            "purpose": "exploratory",
            "condition_param": "experiment_type",
            "style_params": {
                "color": f"#{random.randint(0, 16777215):06x}",
                "marker": random.choice(["o", "s", "^"]),
                "dpi": 300
            },
            "versioned": True
        })
    
    yield ("concurrent_small", small_entries, 10)
    
    # Medium concurrent load - 50 concurrent figures
    medium_entries = []
    purposes = ["exploratory", "presentation", "publication"]
    conditions = ["baseline", "treatment_a", "treatment_b", "control"]
    
    for i in range(50):
        medium_entries.append({
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": f"data/08_reporting/concurrent_medium_{i:03d}.png",
            "purpose": random.choice(purposes),
            "condition_param": "experiment_condition",
            "style_params": {
                "color": f"#{random.randint(0, 16777215):06x}",
                "marker": random.choice(["o", "s", "^", "v", "D"]),
                "linestyle": random.choice(["-", "--", "-.", ":"]),
                "linewidth": random.uniform(1.0, 3.0),
                "markersize": random.uniform(6, 12),
                "alpha": random.uniform(0.5, 1.0),
                "dpi": random.choice([150, 200, 300])
            },
            "save_args": {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "transparent": random.choice([True, False])
            },
            "versioned": True
        })
    
    yield ("concurrent_medium", medium_entries, 50)
    
    # Large concurrent load - 200 concurrent figures
    large_entries = []
    
    for i in range(200):
        # Simulate complex pipeline with multiple experiment conditions
        experiment_id = f"exp_{i//20:02d}"
        condition_id = f"condition_{i%10}"
        figure_type = random.choice(["line_plot", "scatter", "histogram", "boxplot", "heatmap"])
        
        large_entries.append({
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": f"data/08_reporting/{experiment_id}/{condition_id}/{figure_type}_{i:03d}.png",
            "purpose": random.choice(["exploratory", "presentation", "publication", "report"]),
            "condition_param": "experiment_condition",
            "style_params": {
                # Complex styling to stress-test style resolution
                "figure.figsize": [random.uniform(6, 16), random.uniform(4, 12)],
                "figure.dpi": random.choice([150, 200, 300, 600]),
                "axes.linewidth": random.uniform(0.5, 2.0),
                "axes.labelsize": random.uniform(10, 16),
                "axes.titlesize": random.uniform(12, 18),
                "lines.linewidth": random.uniform(1.0, 4.0),
                "lines.markersize": random.uniform(4, 14),
                "font.size": random.uniform(8, 14),
                "legend.fontsize": random.uniform(8, 12),
                "grid.alpha": random.uniform(0.2, 0.7),
                "color": f"#{random.randint(0, 16777215):06x}",
                "marker": random.choice(["o", "s", "^", "v", "D", "p", "*", "h", "H", "+"]),
                "linestyle": random.choice(["-", "--", "-.", ":"]),
                "alpha": random.uniform(0.3, 1.0)
            },
            "save_args": {
                "format": random.choice(["png", "pdf", "svg"]),
                "dpi": random.choice([200, 300, 600]),
                "bbox_inches": "tight",
                "pad_inches": random.uniform(0.05, 0.2),
                "transparent": random.choice([True, False]),
                "facecolor": "white" if random.random() > 0.3 else "none"
            },
            "versioned": True,
            "metadata": {
                "experiment_id": experiment_id,
                "condition_id": condition_id,
                "figure_type": figure_type,
                "complexity_level": "high"
            }
        })
    
    yield ("concurrent_large", large_entries, 200)
    
    # Stress test - 500 concurrent figures with complex dependencies
    stress_entries = []
    
    # Simulate enterprise pipeline with multiple teams and experiments
    teams = ["data_science", "ml_research", "analytics", "visualization", "reporting"]
    projects = ["project_alpha", "project_beta", "project_gamma", "project_delta"]
    
    for i in range(500):
        team = random.choice(teams)
        project = random.choice(projects)
        pipeline_stage = random.choice(["preprocessing", "training", "evaluation", "reporting"])
        
        stress_entries.append({
            "type": "figregistry_kedro.datasets.FigureDataSet",
            "filepath": f"data/08_reporting/{team}/{project}/{pipeline_stage}/figure_{i:04d}.png",
            "purpose": random.choice(["exploratory", "presentation", "publication", "report", "thesis"]),
            "condition_param": f"{team}_condition",
            "style_params": {
                # Maximum styling complexity
                "figure.figsize": [random.uniform(8, 20), random.uniform(6, 16)],
                "figure.dpi": random.choice([150, 200, 300, 600, 1200]),
                "figure.facecolor": f"#{random.randint(0, 16777215):06x}",
                "figure.edgecolor": f"#{random.randint(0, 16777215):06x}",
                "axes.linewidth": random.uniform(0.5, 3.0),
                "axes.spines.left": random.choice([True, False]),
                "axes.spines.bottom": random.choice([True, False]),
                "axes.spines.top": random.choice([True, False]),
                "axes.spines.right": random.choice([True, False]),
                "axes.facecolor": f"#{random.randint(0, 16777215):06x}",
                "axes.edgecolor": f"#{random.randint(0, 16777215):06x}",
                "axes.labelsize": random.uniform(8, 18),
                "axes.titlesize": random.uniform(10, 24),
                "xtick.labelsize": random.uniform(6, 16),
                "ytick.labelsize": random.uniform(6, 16),
                "legend.fontsize": random.uniform(6, 16),
                "font.family": random.choice(["sans-serif", "serif", "monospace"]),
                "font.size": random.uniform(8, 18),
                "lines.linewidth": random.uniform(0.5, 5.0),
                "lines.markersize": random.uniform(2, 16),
                "grid.alpha": random.uniform(0.1, 0.9),
                "grid.linewidth": random.uniform(0.2, 2.0),
                "color": f"#{random.randint(0, 16777215):06x}",
                "marker": random.choice(["o", "s", "^", "v", "D", "p", "*", "h", "H", "+", "x", "X", "d"]),
                "linestyle": random.choice(["-", "--", "-.", ":"]),
                "alpha": random.uniform(0.2, 1.0),
                "markerfacecolor": f"#{random.randint(0, 16777215):06x}",
                "markeredgecolor": f"#{random.randint(0, 16777215):06x}",
                "markeredgewidth": random.uniform(0.1, 2.0)
            },
            "save_args": {
                "format": random.choice(["png", "pdf", "svg", "eps"]),
                "dpi": random.choice([150, 200, 300, 600, 1200]),
                "bbox_inches": random.choice(["tight", None]),
                "pad_inches": random.uniform(0.0, 0.3),
                "transparent": random.choice([True, False]),
                "facecolor": random.choice(["white", "none", "auto"]),
                "edgecolor": random.choice(["white", "none", "auto"])
            },
            "versioned": True,
            "metadata": {
                "team": team,
                "project": project,
                "pipeline_stage": pipeline_stage,
                "complexity_level": "stress_maximum",
                "concurrent_index": i
            }
        })
    
    yield ("concurrent_stress", stress_entries, 500)


# =============================================================================
# COMPLEX FIGURE GENERATORS
# =============================================================================

def complex_figure_generators() -> Iterator[Tuple[str, Figure, Dict[str, Any], float]]:
    """
    Generate matplotlib figures with varying complexity levels for dataset save 
    performance benchmarking.
    
    Creates figures ranging from simple line plots to complex multi-panel visualizations
    to validate FigureDataSet save operations maintain <200ms overhead target.
    
    Yields:
        Tuple of (complexity_level, figure_object, metadata, expected_save_time_ms)
    """
    
    # Simple figure - baseline performance
    fig_simple, ax_simple = plt.subplots(figsize=(8, 6))
    x_simple = np.linspace(0, 10, 100)
    y_simple = np.sin(x_simple)
    ax_simple.plot(x_simple, y_simple, 'b-', linewidth=2)
    ax_simple.set_xlabel('X values')
    ax_simple.set_ylabel('Y values')
    ax_simple.set_title('Simple Performance Test Figure')
    ax_simple.grid(True, alpha=0.3)
    
    yield ("simple", fig_simple, {
        "data_points": 100,
        "plot_elements": 1,
        "complexity_score": 1,
        "expected_memory_mb": 0.5
    }, 20.0)
    
    # Medium complexity - multiple series and annotations
    fig_medium, ax_medium = plt.subplots(figsize=(12, 8))
    
    # Multiple data series
    x_medium = np.linspace(0, 20, 1000)
    for i in range(5):
        y_series = np.sin(x_medium + i) * np.exp(-x_medium / 20) + np.random.normal(0, 0.1, len(x_medium))
        ax_medium.plot(x_medium, y_series, linewidth=2, label=f'Series {i+1}', 
                      marker='o', markersize=4, alpha=0.8)
    
    # Add annotations and styling
    ax_medium.set_xlabel('Time (s)', fontsize=14)
    ax_medium.set_ylabel('Amplitude', fontsize=14)
    ax_medium.set_title('Medium Complexity Performance Test', fontsize=16)
    ax_medium.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_medium.grid(True, alpha=0.4)
    
    # Add text annotations
    for i in range(3):
        ax_medium.annotate(f'Peak {i+1}', 
                          xy=(i*7 + 2, 0.8), 
                          xytext=(i*7 + 4, 1.2),
                          arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                          fontsize=10)
    
    plt.tight_layout()
    
    yield ("medium", fig_medium, {
        "data_points": 5000,
        "plot_elements": 8,  # 5 series + 3 annotations
        "complexity_score": 5,
        "expected_memory_mb": 2.0
    }, 50.0)
    
    # High complexity - multi-panel with various plot types
    fig_complex = plt.figure(figsize=(16, 12))
    
    # Create complex subplot layout
    gs = fig_complex.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Large main plot
    ax_main = fig_complex.add_subplot(gs[0, :])
    x_main = np.linspace(0, 50, 5000)
    y_main = np.sin(x_main) * np.cos(x_main/3) + np.random.normal(0, 0.1, len(x_main))
    ax_main.plot(x_main, y_main, 'b-', linewidth=1, alpha=0.7)
    ax_main.fill_between(x_main, y_main, alpha=0.3)
    ax_main.set_title('Main Time Series Analysis', fontsize=16)
    ax_main.grid(True, alpha=0.3)
    
    # Histogram
    ax_hist = fig_complex.add_subplot(gs[1, 0])
    hist_data = np.random.normal(0, 1, 10000)
    ax_hist.hist(hist_data, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax_hist.set_title('Distribution')
    ax_hist.grid(True, alpha=0.3)
    
    # Scatter plot with color mapping
    ax_scatter = fig_complex.add_subplot(gs[1, 1])
    x_scatter = np.random.randn(1000)
    y_scatter = x_scatter * 2 + np.random.randn(1000)
    colors = np.random.rand(1000)
    scatter = ax_scatter.scatter(x_scatter, y_scatter, c=colors, alpha=0.6, s=50)
    ax_scatter.set_title('Correlation Analysis')
    plt.colorbar(scatter, ax=ax_scatter)
    
    # Heatmap
    ax_heatmap = fig_complex.add_subplot(gs[1, 2])
    heatmap_data = np.random.rand(20, 20)
    im = ax_heatmap.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax_heatmap.set_title('Feature Heatmap')
    plt.colorbar(im, ax=ax_heatmap)
    
    # Box plots
    ax_box = fig_complex.add_subplot(gs[2, 0])
    box_data = [np.random.normal(0, std, 1000) for std in range(1, 5)]
    ax_box.boxplot(box_data, labels=[f'Group {i+1}' for i in range(4)])
    ax_box.set_title('Group Comparisons')
    ax_box.grid(True, alpha=0.3)
    
    # Bar chart
    ax_bar = fig_complex.add_subplot(gs[2, 1])
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [random.uniform(10, 100) for _ in categories]
    bars = ax_bar.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
    ax_bar.set_title('Category Analysis')
    ax_bar.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}', ha='center', va='bottom')
    
    # Polar plot
    ax_polar = fig_complex.add_subplot(gs[2, 2], projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3 * np.sin(4*theta) + 0.1 * np.random.randn(100)
    ax_polar.plot(theta, r, 'r-', linewidth=2)
    ax_polar.fill(theta, r, alpha=0.3)
    ax_polar.set_title('Polar Analysis')
    
    plt.suptitle('Complex Multi-Panel Performance Test', fontsize=20)
    
    yield ("high", fig_complex, {
        "data_points": 17200,  # Sum of all data points across subplots
        "plot_elements": 15,   # Multiple plots, colorbars, annotations
        "complexity_score": 10,
        "expected_memory_mb": 8.0
    }, 150.0)
    
    # Stress test figure - maximum complexity
    fig_stress = plt.figure(figsize=(24, 18))
    
    # Create massive subplot grid
    rows, cols = 6, 6
    
    plot_types = ['line', 'scatter', 'histogram', 'contour', 'bar', 'pie']
    
    for i in range(rows * cols):
        ax = plt.subplot(rows, cols, i + 1)
        plot_type = plot_types[i % len(plot_types)]
        
        if plot_type == 'line':
            x = np.linspace(0, 10, 1000)
            for j in range(3):
                y = np.sin(x + j) + np.random.normal(0, 0.1, len(x))
                ax.plot(x, y, linewidth=1.5, alpha=0.8, label=f'Series {j+1}')
            ax.legend(fontsize=6)
            
        elif plot_type == 'scatter':
            x = np.random.randn(2000)
            y = x * 2 + np.random.randn(2000)
            colors = np.random.rand(2000)
            ax.scatter(x, y, c=colors, alpha=0.5, s=10)
            
        elif plot_type == 'histogram':
            data = np.random.gamma(2, 2, 5000)
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
            
        elif plot_type == 'contour':
            x = np.linspace(-3, 3, 100)
            y = np.linspace(-3, 3, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-(X**2 + Y**2))
            ax.contour(X, Y, Z, levels=10)
            
        elif plot_type == 'bar':
            categories = [f'Cat{j}' for j in range(8)]
            values = [random.uniform(1, 10) for _ in categories]
            ax.bar(categories, values, alpha=0.7)
            ax.tick_params(axis='x', rotation=45, labelsize=6)
            
        elif plot_type == 'pie':
            sizes = [random.uniform(1, 10) for _ in range(6)]
            ax.pie(sizes, autopct='%1.1f%%', textprops={'fontsize': 6})
        
        ax.set_title(f'{plot_type.title()} {i+1}', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
    
    plt.suptitle('Stress Test: Maximum Complexity Figure', fontsize=24)
    plt.tight_layout()
    
    yield ("stress", fig_stress, {
        "data_points": 50000,  # Massive dataset across all subplots
        "plot_elements": 36,   # 6x6 grid of plots
        "complexity_score": 20,
        "expected_memory_mb": 15.0
    }, 180.0)


# =============================================================================
# CONCURRENT EXECUTION DATA
# =============================================================================

def concurrent_execution_data() -> Iterator[Tuple[str, Dict[str, Any], int]]:
    """
    Generate test scenarios for parallel Kedro runner validation per thread-safety requirements.
    
    Creates data structures and scenarios for testing plugin behavior under concurrent
    execution with multiple pipeline runners and parallel processing scenarios.
    
    Yields:
        Tuple of (scenario_name, execution_config, concurrent_thread_count)
    """
    
    # Basic concurrent execution - 2 threads
    yield ("basic_concurrent", {
        "execution_type": "parallel_simple",
        "pipeline_configs": [
            {
                "pipeline_name": "pipeline_a",
                "figregistry_config": {
                    "styles": {"baseline": {"color": "#1f77b4", "marker": "o"}},
                    "outputs": {"base_path": "data/08_reporting/pipeline_a"}
                },
                "catalog_entries": {
                    "figure_a1": {
                        "type": "figregistry_kedro.datasets.FigureDataSet",
                        "filepath": "data/08_reporting/pipeline_a/figure_a1.png",
                        "condition_param": "baseline"
                    }
                },
                "execution_time_target_ms": 100
            },
            {
                "pipeline_name": "pipeline_b", 
                "figregistry_config": {
                    "styles": {"treatment": {"color": "#ff7f0e", "marker": "s"}},
                    "outputs": {"base_path": "data/08_reporting/pipeline_b"}
                },
                "catalog_entries": {
                    "figure_b1": {
                        "type": "figregistry_kedro.datasets.FigureDataSet",
                        "filepath": "data/08_reporting/pipeline_b/figure_b1.png",
                        "condition_param": "treatment"
                    }
                },
                "execution_time_target_ms": 100
            }
        ],
        "thread_safety_requirements": {
            "configuration_isolation": True,
            "figure_object_isolation": True,
            "file_output_isolation": True
        }
    }, 2)
    
    # Medium concurrent execution - 4 threads with shared config
    yield ("medium_concurrent", {
        "execution_type": "parallel_shared_config",
        "shared_figregistry_config": {
            "styles": {
                "experiment_a": {"color": "#e74c3c", "marker": "^"},
                "experiment_b": {"color": "#3498db", "marker": "v"},
                "experiment_c": {"color": "#2ecc71", "marker": "D"},
                "control": {"color": "#95a5a6", "marker": "o"}
            },
            "outputs": {
                "base_path": "data/08_reporting/shared",
                "timestamp": True,
                "versioning": "kedro"
            }
        },
        "pipeline_configs": [
            {
                "pipeline_name": f"experiment_{chr(65+i)}", 
                "condition": f"experiment_{chr(97+i)}" if i < 3 else "control",
                "catalog_entries": {
                    f"figure_{chr(97+i)}{j}": {
                        "type": "figregistry_kedro.datasets.FigureDataSet",
                        "filepath": f"data/08_reporting/shared/exp_{chr(97+i)}/figure_{j:02d}.png",
                        "condition_param": f"experiment_{chr(97+i)}" if i < 3 else "control",
                        "versioned": True
                    } for j in range(3)
                },
                "execution_time_target_ms": 150
            } for i in range(4)
        ],
        "concurrency_challenges": {
            "shared_configuration_access": True,
            "concurrent_file_creation": True,
            "memory_pressure_simulation": True
        }
    }, 4)
    
    # High concurrent execution - 8 threads with complex interactions
    yield ("high_concurrent", {
        "execution_type": "parallel_complex",
        "pipeline_configs": [
            {
                "pipeline_name": f"complex_pipeline_{i:02d}",
                "figregistry_config": {
                    "styles": {
                        f"condition_{i}_{j}": {
                            "color": f"#{random.randint(0, 16777215):06x}",
                            "marker": random.choice(["o", "s", "^", "v", "D"]),
                            "linewidth": random.uniform(1.0, 3.0),
                            "alpha": random.uniform(0.5, 1.0)
                        } for j in range(5)
                    },
                    "outputs": {
                        "base_path": f"data/08_reporting/complex_{i:02d}",
                        "dpi": random.choice([150, 200, 300]),
                        "format": random.choice(["png", "pdf", "svg"])
                    }
                },
                "catalog_entries": {
                    f"figure_{i:02d}_{j:02d}": {
                        "type": "figregistry_kedro.datasets.FigureDataSet",
                        "filepath": f"data/08_reporting/complex_{i:02d}/figure_{j:02d}.png",
                        "condition_param": f"condition_{i}_{j%5}",
                        "style_params": {
                            "dpi": random.choice([200, 300, 600]),
                            "transparent": random.choice([True, False])
                        },
                        "versioned": True
                    } for j in range(10)
                },
                "execution_time_target_ms": 200,
                "memory_target_mb": 10
            } for i in range(8)
        ],
        "stress_factors": {
            "high_memory_usage": True,
            "complex_style_resolution": True,
            "concurrent_file_io": True,
            "configuration_cache_pressure": True
        }
    }, 8)
    
    # Stress concurrent execution - 16 threads maximum load
    yield ("stress_concurrent", {
        "execution_type": "parallel_stress_maximum",
        "global_config": {
            "figregistry_version": "0.3.0",
            "memory_limit_mb": 100,
            "concurrent_figure_limit": 10,
            "cache_size_limit": 50
        },
        "pipeline_configs": [
            {
                "pipeline_name": f"stress_pipeline_{i:02d}",
                "team": f"team_{i//4}",
                "project": f"project_{chr(65 + i%4)}",
                "figregistry_config": {
                    "styles": {
                        f"stress_condition_{i}_{j}": {
                            # Complex styling that stresses resolution system
                            "figure.figsize": [random.uniform(8, 16), random.uniform(6, 12)],
                            "figure.dpi": random.choice([150, 200, 300, 600]),
                            "axes.linewidth": random.uniform(0.5, 2.0),
                            "lines.linewidth": random.uniform(1.0, 4.0),
                            "lines.markersize": random.uniform(4, 12),
                            "font.size": random.uniform(8, 16),
                            "color": f"#{random.randint(0, 16777215):06x}",
                            "marker": random.choice(["o", "s", "^", "v", "D", "p", "*"]),
                            "alpha": random.uniform(0.3, 1.0)
                        } for j in range(20)
                    },
                    "outputs": {
                        "base_path": f"data/08_reporting/stress/team_{i//4}/project_{chr(65 + i%4)}",
                        "dpi": random.choice([200, 300, 600]),
                        "format": random.choice(["png", "pdf", "svg"]),
                        "timestamp": True,
                        "versioning": "kedro"
                    }
                },
                "catalog_entries": {
                    f"stress_figure_{i:02d}_{j:03d}": {
                        "type": "figregistry_kedro.datasets.FigureDataSet",
                        "filepath": f"data/08_reporting/stress/team_{i//4}/project_{chr(65 + i%4)}/figure_{j:03d}.png",
                        "condition_param": f"stress_condition_{i}_{j%20}",
                        "style_params": {
                            "dpi": random.choice([150, 200, 300, 600]),
                            "transparent": random.choice([True, False]),
                            "bbox_inches": "tight",
                            "pad_inches": random.uniform(0.05, 0.2)
                        },
                        "versioned": True,
                        "metadata": {
                            "stress_level": "maximum",
                            "concurrent_index": j,
                            "team": f"team_{i//4}",
                            "project": f"project_{chr(65 + i%4)}"
                        }
                    } for j in range(25)
                },
                "execution_time_target_ms": 250,
                "memory_target_mb": 6,
                "concurrency_stress_level": "maximum"
            } for i in range(16)
        ],
        "maximum_stress_factors": {
            "extreme_memory_pressure": True,
            "maximum_concurrent_io": True,
            "configuration_cache_thrashing": True,
            "figure_object_memory_pressure": True,
            "thread_contention_simulation": True
        }
    }, 16)


# =============================================================================
# MEMORY USAGE SCENARIOS
# =============================================================================

def memory_usage_scenarios() -> Iterator[Tuple[str, Dict[str, Any], float]]:
    """
    Generate memory usage scenarios for testing plugin memory footprint targeting 
    <5MB overhead per Section 6.6.4.3.
    
    Creates scenarios that stress-test memory usage patterns to validate plugin
    memory efficiency and prevent memory leaks during extended pipeline execution.
    
    Yields:
        Tuple of (scenario_name, memory_test_config, expected_memory_mb)
    """
    
    # Baseline memory usage
    yield ("baseline_minimal", {
        "scenario_type": "minimal_plugin_load",
        "configuration_size": "minimal",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {"default": {"color": "#1f77b4"}},
            "outputs": {"base_path": "data/08_reporting"}
        },
        "catalog_entries": 1,
        "expected_operations": ["hook_initialization", "config_loading"],
        "memory_tracking": {
            "track_hook_memory": True,
            "track_config_memory": True,
            "track_dataset_memory": True
        }
    }, 1.0)
    
    # Standard configuration memory usage
    yield ("standard_config", {
        "scenario_type": "standard_configuration_load",
        "configuration_size": "standard",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"condition_{i}": {
                    "color": f"#{random.randint(0, 16777215):06x}",
                    "marker": random.choice(["o", "s", "^", "v", "D"]),
                    "linewidth": random.uniform(1.0, 3.0),
                    "alpha": random.uniform(0.5, 1.0)
                } for i in range(50)
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "dpi": 300,
                "format": "png",
                "timestamp": True
            }
        },
        "catalog_entries": 10,
        "concurrent_figures": 5,
        "memory_tracking": {
            "track_configuration_cache": True,
            "track_style_resolution_cache": True,
            "track_figure_object_memory": True
        }
    }, 2.5)
    
    # Large configuration memory stress
    yield ("large_config_stress", {
        "scenario_type": "large_configuration_stress",
        "configuration_size": "large",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"complex_condition_{i}_{j}": {
                    # Complex nested styling configuration
                    "figure.figsize": [random.uniform(6, 16), random.uniform(4, 12)],
                    "figure.dpi": random.choice([150, 200, 300, 600]),
                    "axes.linewidth": random.uniform(0.5, 3.0),
                    "axes.labelsize": random.uniform(8, 16),
                    "lines.linewidth": random.uniform(0.5, 4.0),
                    "lines.markersize": random.uniform(2, 12),
                    "font.size": random.uniform(8, 16),
                    "color": f"#{random.randint(0, 16777215):06x}",
                    "marker": random.choice(["o", "s", "^", "v", "D", "p", "*", "h", "H", "+"]),
                    "alpha": random.uniform(0.2, 1.0)
                } for i in range(10) for j in range(50)
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "path_aliases": {
                    f"alias_{k}": f"path/level1/level2/{k}" for k in range(100)
                }
            },
            "environments": {
                f"env_{env}": {
                    "outputs": {"dpi": random.choice([150, 300, 600])},
                    "style_overrides": {
                        f"complex_condition_{i}_{j}": {"alpha": random.uniform(0.1, 1.0)}
                        for i in range(5) for j in range(10)
                    }
                } for env in range(10)
            }
        },
        "catalog_entries": 50,
        "concurrent_figures": 10,
        "memory_tracking": {
            "track_deep_configuration_memory": True,
            "track_environment_resolution_memory": True,
            "track_cache_growth_patterns": True
        }
    }, 4.5)
    
    # Memory leak detection scenario
    yield ("memory_leak_detection", {
        "scenario_type": "memory_leak_detection",
        "configuration_size": "medium",
        "test_duration_minutes": 5,
        "operations_per_minute": 100,
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"leak_test_condition_{i}": {
                    "color": f"#{random.randint(0, 16777215):06x}",
                    "marker": random.choice(["o", "s", "^"]),
                    "linewidth": random.uniform(1.0, 2.0)
                } for i in range(20)
            },
            "outputs": {"base_path": "data/08_reporting"}
        },
        "repeated_operations": {
            "figure_creation_cycles": 1000,
            "config_resolution_cycles": 500,
            "style_application_cycles": 2000
        },
        "memory_tracking": {
            "track_memory_growth": True,
            "detect_memory_leaks": True,
            "monitor_garbage_collection": True,
            "track_figure_object_cleanup": True
        },
        "gc_settings": {
            "force_gc_between_operations": True,
            "track_gc_collections": True
        }
    }, 3.0)
    
    # Concurrent memory pressure scenario
    yield ("concurrent_memory_pressure", {
        "scenario_type": "concurrent_memory_pressure",
        "configuration_size": "large",
        "concurrent_threads": 8,
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"pressure_condition_{i}_{j}": {
                    # Memory-intensive styling configuration
                    "figure.figsize": [random.uniform(12, 20), random.uniform(8, 16)],
                    "figure.dpi": random.choice([300, 600, 1200]),
                    "complex_data": [random.random() for _ in range(1000)],  # Large data arrays
                    "color_palette": [f"#{random.randint(0, 16777215):06x}" for _ in range(50)],
                    "marker_styles": {
                        f"marker_{k}": {
                            "size": random.uniform(2, 15),
                            "color": f"#{random.randint(0, 16777215):06x}",
                            "alpha": random.uniform(0.1, 1.0)
                        } for k in range(20)
                    }
                } for i in range(5) for j in range(10)
            },
            "outputs": {"base_path": "data/08_reporting"}
        },
        "concurrent_operations": {
            "simultaneous_figure_creation": 8,
            "simultaneous_config_resolution": 16,
            "simultaneous_style_application": 32
        },
        "memory_tracking": {
            "track_peak_memory_usage": True,
            "track_memory_per_thread": True,
            "monitor_memory_contention": True
        }
    }, 4.8)
    
    # Memory overflow prevention test
    yield ("memory_overflow_prevention", {
        "scenario_type": "memory_overflow_prevention",
        "configuration_size": "maximum",
        "stress_level": "extreme",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"overflow_test_{i:04d}": {
                    # Extremely large configuration entries
                    "figure.figsize": [random.uniform(16, 24), random.uniform(12, 18)],
                    "figure.dpi": 1200,  # Maximum DPI
                    "massive_array": [random.random() for _ in range(10000)],  # Large arrays
                    "complex_nested_data": {
                        f"level_{j}": {
                            f"sublevel_{k}": [random.random() for _ in range(100)]
                            for k in range(10)
                        } for j in range(10)
                    },
                    "color_gradients": [
                        [f"#{random.randint(0, 16777215):06x}" for _ in range(100)]
                        for _ in range(10)
                    ]
                } for i in range(100)
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "massive_path_aliases": {
                    f"path_alias_{i:04d}": f"very/deep/path/structure/level{i//10}/sublevel{i%10}"
                    for i in range(1000)
                }
            }
        },
        "overflow_protection": {
            "max_memory_limit_mb": 10,  # Hard limit above target
            "memory_monitoring_interval_ms": 100,
            "automatic_cleanup_threshold_mb": 8
        },
        "memory_tracking": {
            "track_memory_ceiling": True,
            "monitor_overflow_prevention": True,
            "track_automatic_cleanup": True
        }
    }, 5.0)  # At the target limit


# =============================================================================
# BENCHMARK TIMING UTILITIES
# =============================================================================

class BenchmarkTimer:
    """
    Precision timing utilities for measuring hook initialization, config resolution,
    and dataset operations per Section 6.6.4.3 performance requirements.
    """
    
    def __init__(self):
        self.measurements: List[PerformanceMetrics] = []
        self.targets = PerformanceTargets()
        
    @contextmanager
    def time_operation(self, operation_name: str, **metadata):
        """
        Context manager for timing operations with memory and CPU monitoring.
        
        Args:
            operation_name: Name of the operation being timed
            **metadata: Additional metadata to store with measurement
        """
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Get final system state
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            
            # Create measurement record
            measurement = PerformanceMetrics(
                operation_name=operation_name,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=final_memory - initial_memory,
                cpu_usage_percent=final_cpu - initial_cpu,
                metadata=metadata
            )
            
            self.measurements.append(measurement)
    
    def time_hook_initialization(self, hook_callable: Callable) -> PerformanceMetrics:
        """
        Time FigRegistryHooks initialization targeting <25ms per Section 6.6.4.3.
        
        Args:
            hook_callable: Function that initializes FigRegistryHooks
            
        Returns:
            PerformanceMetrics with timing data
        """
        with self.time_operation("hook_initialization") as timer:
            result = hook_callable()
        
        measurement = self.measurements[-1]
        measurement.metadata.update({
            "target_ms": self.targets.hook_initialization,
            "meets_target": measurement.meets_target(self.targets.hook_initialization),
            "result": str(result)
        })
        
        return measurement
    
    def time_config_bridge_resolution(self, bridge_callable: Callable) -> PerformanceMetrics:
        """
        Time FigRegistryConfigBridge resolution targeting <50ms per Section 6.6.4.3.
        
        Args:
            bridge_callable: Function that performs config bridge resolution
            
        Returns:
            PerformanceMetrics with timing data
        """
        with self.time_operation("config_bridge_resolution") as timer:
            result = bridge_callable()
        
        measurement = self.measurements[-1]
        measurement.metadata.update({
            "target_ms": self.targets.config_bridge_resolution,
            "meets_target": measurement.meets_target(self.targets.config_bridge_resolution),
            "config_size_kb": len(str(result)) / 1024 if result else 0
        })
        
        return measurement
    
    def time_dataset_save_operation(self, dataset_save_callable: Callable) -> PerformanceMetrics:
        """
        Time FigureDataSet save operation targeting <200ms per Section 6.6.4.3.
        
        Args:
            dataset_save_callable: Function that performs dataset save
            
        Returns:
            PerformanceMetrics with timing data
        """
        with self.time_operation("dataset_save_operation") as timer:
            result = dataset_save_callable()
        
        measurement = self.measurements[-1]
        measurement.metadata.update({
            "target_ms": self.targets.figuredataset_save_overhead,
            "meets_target": measurement.meets_target(self.targets.figuredataset_save_overhead),
            "file_path": str(result) if result else None
        })
        
        return measurement
    
    def batch_time_operations(self, operations: List[Tuple[str, Callable]], iterations: int = 10) -> Dict[str, List[PerformanceMetrics]]:
        """
        Time multiple operations with multiple iterations for statistical analysis.
        
        Args:
            operations: List of (operation_name, callable) tuples
            iterations: Number of iterations per operation
            
        Returns:
            Dictionary mapping operation names to lists of measurements
        """
        results = defaultdict(list)
        
        for operation_name, operation_callable in operations:
            for iteration in range(iterations):
                with self.time_operation(f"{operation_name}_iter_{iteration}") as timer:
                    operation_callable()
                
                measurement = self.measurements[-1]
                measurement.metadata.update({
                    "iteration": iteration,
                    "total_iterations": iterations
                })
                
                results[operation_name].append(measurement)
        
        return dict(results)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with SLA validation.
        
        Returns:
            Dictionary containing performance analysis and recommendations
        """
        if not self.measurements:
            return {"error": "No measurements available"}
        
        # Group measurements by operation type
        operations = defaultdict(list)
        for measurement in self.measurements:
            base_operation = measurement.operation_name.split('_iter_')[0]
            operations[base_operation].append(measurement)
        
        report = {
            "summary": {
                "total_measurements": len(self.measurements),
                "operation_types": len(operations),
                "measurement_period": {
                    "start": min(m.timestamp for m in self.measurements).isoformat(),
                    "end": max(m.timestamp for m in self.measurements).isoformat()
                }
            },
            "performance_targets": {
                "hook_initialization_ms": self.targets.hook_initialization,
                "config_bridge_resolution_ms": self.targets.config_bridge_resolution,
                "dataset_save_overhead_ms": self.targets.figuredataset_save_overhead,
                "plugin_memory_overhead_mb": self.targets.plugin_memory_overhead
            },
            "operation_analysis": {},
            "sla_compliance": {},
            "recommendations": []
        }
        
        # Analyze each operation type
        for operation_name, measurements in operations.items():
            times = [m.execution_time_ms for m in measurements]
            memory_usage = [m.memory_usage_mb for m in measurements]
            
            # Statistical analysis
            analysis = {
                "count": len(measurements),
                "timing_stats": {
                    "mean_ms": np.mean(times),
                    "median_ms": np.median(times),
                    "std_ms": np.std(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "p95_ms": np.percentile(times, 95),
                    "p99_ms": np.percentile(times, 99)
                },
                "memory_stats": {
                    "mean_mb": np.mean(memory_usage),
                    "median_mb": np.median(memory_usage),
                    "max_mb": np.max(memory_usage)
                }
            }
            
            # SLA compliance check
            target_ms = getattr(self.targets, operation_name.replace('_operation', ''), None)
            if target_ms:
                compliant_count = sum(1 for t in times if t <= target_ms)
                analysis["sla_compliance"] = {
                    "target_ms": target_ms,
                    "compliance_rate": compliant_count / len(times),
                    "violations": len(times) - compliant_count,
                    "worst_violation_ms": max(times) - target_ms if max(times) > target_ms else 0
                }
                
                report["sla_compliance"][operation_name] = analysis["sla_compliance"]
                
                # Generate recommendations
                if analysis["sla_compliance"]["compliance_rate"] < 0.95:
                    report["recommendations"].append({
                        "operation": operation_name,
                        "issue": "SLA compliance below 95%",
                        "current_rate": analysis["sla_compliance"]["compliance_rate"],
                        "suggested_actions": [
                            "Profile operation for bottlenecks",
                            "Consider caching optimizations",
                            "Review configuration complexity"
                        ]
                    })
            
            report["operation_analysis"][operation_name] = analysis
        
        return report


# =============================================================================
# STRESS TEST DATA GENERATORS
# =============================================================================

def stress_test_data_generators() -> Iterator[Tuple[str, Dict[str, Any], Dict[str, float]]]:
    """
    Generate stress testing data for validating plugin performance under high-load 
    scenarios per performance requirements.
    
    Creates extreme scenarios that push plugin components to their limits while
    validating graceful degradation and error handling under stress conditions.
    
    Yields:
        Tuple of (stress_scenario_name, stress_config, performance_thresholds)
    """
    
    # Configuration complexity stress test
    yield ("config_complexity_stress", {
        "stress_type": "configuration_complexity",
        "description": "Maximum configuration complexity with thousands of conditions",
        "figregistry_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"stress_condition_{category}_{experiment}_{variant}_{iteration:04d}": {
                    "figure.figsize": [random.uniform(8, 20), random.uniform(6, 16)],
                    "figure.dpi": random.choice([150, 200, 300, 600, 1200]),
                    "axes.linewidth": random.uniform(0.5, 3.0),
                    "lines.linewidth": random.uniform(0.5, 5.0),
                    "font.size": random.uniform(8, 18),
                    "color": f"#{random.randint(0, 16777215):06x}",
                    "marker": random.choice(["o", "s", "^", "v", "D", "p", "*", "h", "H", "+", "x", "X"]),
                    "alpha": random.uniform(0.1, 1.0),
                    "complex_nested_params": {
                        f"param_{j}": random.random() for j in range(50)
                    }
                }
                for category in ["exp", "ctrl", "treat", "base", "test"]
                for experiment in range(20)
                for variant in ["a", "b", "c", "d"]
                for iteration in range(25)
            },
            "outputs": {
                "base_path": "data/08_reporting/stress",
                "path_aliases": {
                    f"stress_alias_{i:05d}": f"deep/nested/path/level{i//1000}/sublevel{(i//100)%10}/item{i%100}"
                    for i in range(10000)
                }
            }
        },
        "stress_parameters": {
            "total_conditions": 10000,
            "config_file_size_mb": 50,
            "resolution_attempts": 1000,
            "concurrent_resolutions": 16
        }
    }, {
        "config_bridge_resolution_ms": 100.0,  # Relaxed under stress
        "memory_usage_mb": 10.0,  # Increased limit for stress test
        "style_resolution_ms": 5.0
    })
    
    # Concurrent execution stress test
    yield ("concurrent_execution_stress", {
        "stress_type": "concurrent_execution",
        "description": "Maximum concurrent pipeline execution with resource contention",
        "concurrent_pipelines": 32,
        "pipeline_template": {
            "figregistry_config": {
                "styles": {
                    f"concurrent_condition_{i}": {
                        "color": f"#{random.randint(0, 16777215):06x}",
                        "marker": random.choice(["o", "s", "^", "v", "D"]),
                        "linewidth": random.uniform(1.0, 3.0)
                    } for i in range(100)
                },
                "outputs": {"base_path": "data/08_reporting/concurrent"}
            },
            "catalog_entries_per_pipeline": 50,
            "figures_per_entry": 5
        },
        "stress_parameters": {
            "total_concurrent_operations": 1600,  # 32 pipelines * 50 entries
            "memory_pressure_simulation": True,
            "disk_io_contention": True,
            "configuration_cache_thrashing": True
        }
    }, {
        "figuredataset_save_overhead_ms": 300.0,  # Relaxed under stress
        "hook_initialization_ms": 50.0,
        "memory_usage_per_pipeline_mb": 2.0,
        "total_memory_usage_mb": 64.0
    })
    
    # Memory pressure stress test
    yield ("memory_pressure_stress", {
        "stress_type": "memory_pressure",
        "description": "Extreme memory usage with large figures and configurations",
        "memory_intensive_config": {
            "figregistry_version": "0.3.0",
            "styles": {
                f"memory_stress_{i:04d}": {
                    # Memory-intensive configuration entries
                    "large_data_arrays": [random.random() for _ in range(100000)],
                    "complex_color_palettes": [
                        [f"#{random.randint(0, 16777215):06x}" for _ in range(1000)]
                        for _ in range(10)
                    ],
                    "nested_configuration": {
                        f"level_{j}": {
                            f"sublevel_{k}": {
                                f"item_{l}": random.random()
                                for l in range(100)
                            } for k in range(10)
                        } for j in range(10)
                    }
                } for i in range(500)
            }
        },
        "figure_generation": {
            "massive_figures": True,
            "figure_sizes": [(24, 18), (32, 24), (40, 30)],
            "data_points_per_figure": 1000000,
            "concurrent_figure_limit": 20
        },
        "stress_parameters": {
            "target_memory_usage_gb": 2.0,
            "memory_leak_detection": True,
            "garbage_collection_monitoring": True
        }
    }, {
        "memory_usage_mb": 15.0,  # Increased for stress test
        "figuredataset_save_overhead_ms": 500.0,
        "memory_leak_tolerance_mb": 1.0
    })
    
    # I/O throughput stress test
    yield ("io_throughput_stress", {
        "stress_type": "io_throughput",
        "description": "Maximum I/O throughput with rapid figure generation and saving",
        "io_intensive_scenario": {
            "rapid_figure_generation": {
                "figures_per_second": 100,
                "test_duration_seconds": 60,
                "concurrent_save_operations": 50
            },
            "file_format_diversity": {
                "formats": ["png", "pdf", "svg", "eps"],
                "dpi_settings": [150, 200, 300, 600, 1200],
                "size_variations": [(8, 6), (12, 9), (16, 12), (20, 15)]
            },
            "directory_structure_stress": {
                "deep_nesting_levels": 10,
                "directories_per_level": 20,
                "files_per_directory": 100
            }
        },
        "stress_parameters": {
            "total_files_generated": 6000,
            "peak_io_operations_per_second": 200,
            "disk_space_usage_gb": 5.0
        }
    }, {
        "figuredataset_save_overhead_ms": 400.0,
        "file_io_operation_ms": 200.0,
        "directory_creation_ms": 50.0
    })
    
    # System resource exhaustion stress test  
    yield ("resource_exhaustion_stress", {
        "stress_type": "resource_exhaustion",
        "description": "Resource exhaustion simulation with recovery testing",
        "exhaustion_scenarios": {
            "memory_exhaustion": {
                "progressive_memory_increase": True,
                "memory_limit_approach": 0.95,  # Approach 95% of available memory
                "recovery_mechanisms": ["cache_cleanup", "figure_release", "gc_force"]
            },
            "cpu_exhaustion": {
                "cpu_intensive_operations": True,
                "complex_style_calculations": True,
                "parallel_processing_load": 16
            },
            "disk_space_exhaustion": {
                "large_file_generation": True,
                "space_monitoring": True,
                "cleanup_on_threshold": True
            }
        },
        "recovery_testing": {
            "graceful_degradation": True,
            "error_handling_validation": True,
            "automatic_recovery_mechanisms": True
        }
    }, {
        "memory_usage_mb": 20.0,  # Higher limit for exhaustion testing
        "cpu_usage_percent": 90.0,
        "disk_usage_gb": 10.0,
        "recovery_time_ms": 1000.0
    })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_performance_test_suite() -> Dict[str, Any]:
    """
    Create a comprehensive performance test suite combining all generators.
    
    Returns:
        Dictionary containing all performance test data organized by category
    """
    suite = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "performance_targets": PerformanceTargets().__dict__,
            "test_categories": [
                "large_configs", "high_volume_catalogs", "complex_figures",
                "concurrent_execution", "memory_usage", "stress_tests"
            ]
        },
        "large_configs": list(large_config_generators()),
        "kedro_config_scenarios": list(generate_kedro_config_scenarios()),
        "high_volume_catalogs": list(high_volume_catalog_generators()),
        "complex_figures": list(complex_figure_generators()),
        "concurrent_execution": list(concurrent_execution_data()),
        "memory_usage": list(memory_usage_scenarios()),
        "stress_tests": list(stress_test_data_generators()),
        "benchmark_timer": BenchmarkTimer(),
        "utilities": {
            "create_timer": lambda: BenchmarkTimer(),
            "validate_performance": lambda metrics, targets: all(
                m.meets_target(getattr(targets, m.operation_name.replace('_operation', ''), float('inf')))
                for m in metrics
            )
        }
    }
    
    return suite


def validate_performance_targets(measurements: List[PerformanceMetrics]) -> Dict[str, bool]:
    """
    Validate performance measurements against Section 6.6.4.3 targets.
    
    Args:
        measurements: List of performance measurements to validate
        
    Returns:
        Dictionary mapping operation names to compliance status
    """
    targets = PerformanceTargets()
    compliance = {}
    
    for measurement in measurements:
        operation_base = measurement.operation_name.split('_iter_')[0]
        
        if 'hook_initialization' in operation_base:
            compliance[operation_base] = measurement.meets_target(targets.hook_initialization)
        elif 'config_bridge_resolution' in operation_base:
            compliance[operation_base] = measurement.meets_target(targets.config_bridge_resolution)
        elif 'dataset_save_operation' in operation_base:
            compliance[operation_base] = measurement.meets_target(targets.figuredataset_save_overhead)
        else:
            compliance[operation_base] = True  # Unknown operation, assume compliant
    
    return compliance


def generate_performance_summary_report(test_suite: Dict[str, Any]) -> str:
    """
    Generate a human-readable performance test summary report.
    
    Args:
        test_suite: Complete performance test suite from create_performance_test_suite()
        
    Returns:
        Formatted string report of performance test capabilities
    """
    report_lines = [
        "=" * 80,
        "FIGREGISTRY-KEDRO PLUGIN PERFORMANCE TEST DATA SUMMARY",
        "=" * 80,
        "",
        f"Generated: {test_suite['metadata']['created_at']}",
        "",
        "PERFORMANCE TARGETS (Section 6.6.4.3):",
        f"  • Plugin Pipeline Execution Overhead: <{test_suite['metadata']['performance_targets']['figuredataset_save_overhead']}ms",
        f"  • Configuration Bridge Merge Time: <{test_suite['metadata']['performance_targets']['config_bridge_resolution']}ms", 
        f"  • Hook Initialization Overhead: <{test_suite['metadata']['performance_targets']['hook_initialization']}ms",
        f"  • Plugin Memory Footprint: <{test_suite['metadata']['performance_targets']['plugin_memory_overhead']}MB",
        "",
        "TEST DATA CATEGORIES:",
        ""
    ]
    
    # Summarize each category
    categories = [
        ("large_configs", "Large Configuration Scenarios"),
        ("kedro_config_scenarios", "Kedro Configuration Merge Scenarios"),
        ("high_volume_catalogs", "High-Volume Catalog Entries"),
        ("complex_figures", "Complex Figure Generation"),
        ("concurrent_execution", "Concurrent Execution Scenarios"),
        ("memory_usage", "Memory Usage Scenarios"),
        ("stress_tests", "Stress Test Data")
    ]
    
    for key, title in categories:
        if key in test_suite:
            count = len(test_suite[key])
            report_lines.append(f"{title}:")
            report_lines.append(f"  • {count} scenarios available")
            
            if key == "large_configs":
                report_lines.append("  • Tests configuration bridge resolution performance")
                report_lines.append("  • Validates <50ms merge target under load")
            elif key == "high_volume_catalogs":
                report_lines.append("  • Tests concurrent FigureDataSet operations")
                report_lines.append("  • Validates thread-safety and resource contention")
            elif key == "complex_figures":
                report_lines.append("  • Tests figure save operation overhead")
                report_lines.append("  • Validates <200ms dataset save target")
            elif key == "concurrent_execution":
                report_lines.append("  • Tests parallel Kedro runner scenarios")
                report_lines.append("  • Validates plugin thread-safety")
            elif key == "memory_usage":
                report_lines.append("  • Tests plugin memory footprint")
                report_lines.append("  • Validates <5MB overhead target")
            elif key == "stress_tests":
                report_lines.append("  • Tests extreme load scenarios")
                report_lines.append("  • Validates graceful degradation")
            
            report_lines.append("")
    
    report_lines.extend([
        "UTILITIES AVAILABLE:",
        "  • BenchmarkTimer: Precision timing with memory/CPU monitoring",
        "  • Performance validation functions",
        "  • Automated SLA compliance checking",
        "  • Memory leak detection capabilities",
        "  • Comprehensive reporting and analysis",
        "",
        "USAGE:",
        "  from figregistry_kedro.tests.data.performance_test_data import create_performance_test_suite",
        "  suite = create_performance_test_suite()",
        "  timer = suite['benchmark_timer']",
        "",
        "=" * 80
    ])
    
    return "\n".join(report_lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'PerformanceTargets',
    'PerformanceMetrics', 
    'BenchmarkTimer',
    
    # Data generators
    'large_config_generators',
    'generate_kedro_config_scenarios',
    'high_volume_catalog_generators',
    'complex_figure_generators',
    'concurrent_execution_data',
    'memory_usage_scenarios',
    'stress_test_data_generators',
    
    # Utility functions
    'create_performance_test_suite',
    'validate_performance_targets',
    'generate_performance_summary_report'
]


# Module initialization
if __name__ == "__main__":
    # Generate and display summary when run directly
    suite = create_performance_test_suite()
    print(generate_performance_summary_report(suite))