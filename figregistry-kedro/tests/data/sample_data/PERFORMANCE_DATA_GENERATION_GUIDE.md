# Performance Benchmark Data Generation Guide

## Overview

The `performance_benchmark_data.csv` file contains a comprehensive dataset designed for testing the figregistry-kedro plugin's performance characteristics against the technical specification targets defined in Section 6.6.4.3.

## Performance Targets Validation

This dataset is specifically designed to validate the following performance targets:

### Core Performance Metrics
- **Configuration Bridge Merge Time**: < 50ms per pipeline run
- **FigureDataSet Save Overhead**: < 200ms per save operation  
- **Hook Initialization Overhead**: < 25ms per project startup
- **Plugin Memory Overhead**: < 5MB total footprint
- **Pipeline Execution Overhead**: < 200ms per FigureDataSet save

### Scalability Requirements
- **10,000+ Rows**: Large-scale dataset for comprehensive performance testing
- **Concurrent Execution**: Thread-safe operation validation per Section 5.2.8
- **Style Resolution Caching**: Stress testing per Section 5.2.2
- **Memory Footprint**: <5MB overhead validation

## Dataset Structure

### Column Definitions

| Column | Purpose | Performance Test Coverage |
|--------|---------|---------------------------|
| `row_id` | Unique identifier | Data indexing performance |
| `test_scenario` | Test scenario category | Scenario classification and filtering |
| `data_complexity` | Complexity level (simple/medium/high/extreme) | Performance scaling validation |
| `condition_type` | Type of condition resolution | Style resolution performance |
| `condition_value` | Actual condition string | Condition matching performance |
| `experiment_type` | Experiment classification | Catalog parameter extraction |
| `expected_overhead_ms` | Target overhead for scenario | Performance validation thresholds |
| `memory_target_mb` | Memory usage target | Memory footprint validation |
| `concurrent_group` | Grouping for parallel tests | Concurrent execution testing |
| `thread_id` | Thread identifier | Thread-safety validation |
| `cache_scenario` | Cache behavior type | Caching mechanism testing |
| `config_merge_time_target` | Bridge merge time target | Configuration performance |
| `hook_init_time_target` | Hook initialization target | Lifecycle performance |
| `dataset_save_time_target` | Dataset save target | Core operation performance |

## Test Scenario Categories

### 1. Baseline Performance Tests (Rows 1-50)
Simple scenarios establishing performance baselines for:
- Basic condition resolution
- Simple style application  
- Minimal memory usage
- Single-threaded execution

### 2. Concurrent Execution Tests (Expand to Rows 51-2550)
**Target: 2,500 rows covering parallel execution scenarios**

Pattern for expansion:
```csv
# For each concurrent_group (1-25):
#   For each thread_count (1, 2, 4, 8, 16):
#     For each test_iteration (1-20):
#       Generate row with:
#         - concurrent_group: group_N
#         - thread_id: thread_M  
#         - parallel_execution: true
#         - memory_scenario: concurrent_memory
#         - expected_overhead_ms: varies by thread_count
```

### 3. Memory Stress Tests (Expand to Rows 2551-4050)
**Target: 1,500 rows covering memory usage scenarios**

Memory test categories:
- Low memory (1-2MB): Simple configurations
- Medium memory (2-4MB): Standard operations
- High memory (4-6MB): Complex operations
- Stress memory (6-10MB): Maximum load testing

### 4. Cache Performance Tests (Expand to Rows 4051-5550)
**Target: 1,500 rows covering caching scenarios**

Cache scenarios:
- `cache_miss`: Cold cache performance
- `cache_hit`: Warm cache performance  
- `cache_turbulence`: Rapid cache changes
- `cache_contention`: Concurrent cache access
- `cache_stress`: High-volume caching

### 5. Complex Style Resolution (Expand to Rows 5551-7550)
**Target: 2,000 rows covering style resolution complexity**

Complexity patterns:
- Nested conditions: `parent.child.grandchild`
- Wildcard patterns: `pattern_*`, `pattern_**`
- Multiple condition combinations
- Conditional conflicts and resolution

### 6. Plugin Component Tests (Expand to Rows 7551-10050)
**Target: 2,500 rows covering specific plugin components**

Component-specific scenarios:
- FigRegistryConfigBridge performance
- FigRegistryHooks lifecycle testing
- FigureDataSet operation benchmarks
- Kedro integration overhead

### 7. Scalability and Load Tests (Expand to Rows 10051-12000+)
**Target: 2,000+ rows for extreme load testing**

Load test patterns:
- Bulk operations (100+ figures)
- Sustained load (extended duration)
- Resource utilization limits
- Performance degradation analysis

## Systematic Generation Patterns

### Thread Safety Pattern
```python
# Generate concurrent execution data
for group_id in range(1, 26):  # 25 concurrent groups
    for thread_count in [1, 2, 4, 8, 16]:
        for iteration in range(1, 21):  # 20 iterations each
            for thread_id in range(1, thread_count + 1):
                row = {
                    'test_scenario': 'concurrent_execution_test',
                    'concurrent_group': f'group_{group_id}',
                    'thread_id': f'thread_{thread_id}',
                    'parallel_execution': True,
                    'expected_overhead_ms': base_overhead * thread_scaling_factor,
                    # ... additional fields
                }
```

### Memory Scaling Pattern
```python
# Generate memory stress data
memory_levels = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]  # MB targets
for memory_mb in memory_levels:
    for complexity in ['simple', 'medium', 'high', 'extreme']:
        for scenario_count in range(100):  # 100 scenarios per level
            row = {
                'test_scenario': 'memory_stress_test',
                'memory_target_mb': memory_mb,
                'data_complexity': complexity,
                'expected_overhead_ms': calculate_overhead(memory_mb, complexity),
                # ... additional fields
            }
```

### Cache Performance Pattern
```python
# Generate cache performance data
cache_scenarios = ['cache_miss', 'cache_hit', 'cache_turbulence', 'cache_contention']
for cache_type in cache_scenarios:
    for cache_size in [10, 50, 100, 500, 1000]:
        for access_pattern in range(75):  # 75 patterns per scenario
            row = {
                'test_scenario': 'cache_performance_test',
                'cache_scenario': cache_type,
                'style_cache_size': cache_size,
                'expected_overhead_ms': calculate_cache_overhead(cache_type, cache_size),
                # ... additional fields
            }
```

## Performance Validation Usage

### Test Execution Strategy
1. **Load entire dataset** into test framework
2. **Group by test_scenario** for targeted testing
3. **Execute performance measurements** for each row
4. **Compare actual vs expected_overhead_ms** values
5. **Validate memory usage** against memory_target_mb
6. **Report performance compliance** per target metrics

### Expected Performance Distributions
- **90% of rows** should meet their target overhead times
- **95% of memory tests** should stay within target limits
- **100% of concurrent tests** should complete without conflicts
- **Cache hit scenarios** should show 5x improvement over cache miss

### Failure Analysis Patterns
- **Overhead > 200ms**: Flag for optimization
- **Memory > 5MB**: Memory leak investigation
- **Cache misses > expected**: Cache tuning required
- **Concurrent failures**: Thread safety issues

## Data Generation Implementation

To generate the full 10,000+ row dataset programmatically:

```python
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta

def generate_performance_benchmark_data():
    rows = []
    row_id = 1
    base_timestamp = datetime(2024, 1, 1)
    
    # Generate each category systematically
    rows.extend(generate_baseline_tests(row_id, base_timestamp))
    row_id += len(rows)
    
    rows.extend(generate_concurrent_tests(row_id, base_timestamp))
    row_id += len(rows) 
    
    rows.extend(generate_memory_tests(row_id, base_timestamp))
    row_id += len(rows)
    
    rows.extend(generate_cache_tests(row_id, base_timestamp))
    row_id += len(rows)
    
    rows.extend(generate_style_tests(row_id, base_timestamp))
    row_id += len(rows)
    
    rows.extend(generate_plugin_tests(row_id, base_timestamp))
    row_id += len(rows)
    
    rows.extend(generate_scalability_tests(row_id, base_timestamp))
    
    return pd.DataFrame(rows)

# Save to CSV
df = generate_performance_benchmark_data()
df.to_csv('performance_benchmark_data.csv', index=False)
print(f"Generated {len(df)} rows for performance testing")
```

## Quality Assurance

### Data Validation Requirements
- **Row count**: Minimum 10,000 rows
- **Scenario coverage**: All test scenarios represented
- **Performance targets**: All targets within specification limits
- **Data consistency**: No missing required fields
- **Timestamp progression**: Sequential timestamps for testing

### Test Data Integrity
- **Unique row_ids**: No duplicates
- **Valid timestamps**: ISO 8601 format
- **Consistent grouping**: Proper concurrent_group assignments
- **Realistic values**: All values within expected ranges
- **Complete coverage**: All performance dimensions tested

This systematic approach ensures comprehensive coverage of all performance testing requirements while maintaining the ability to validate the <200ms plugin overhead target and <5MB memory footprint specified in the technical requirements.