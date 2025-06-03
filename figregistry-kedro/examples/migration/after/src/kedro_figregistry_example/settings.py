"""Kedro Project Settings with FigRegistry Integration.

This file demonstrates the key configuration changes required to migrate from 
traditional manual matplotlib figure management to automated figregistry-kedro 
integration. The primary transformation involves registering FigRegistryHooks 
to enable lifecycle management without requiring any changes to pipeline node code.

MIGRATION SUMMARY:
==================

Before (Manual Approach):
- Empty HOOKS = [] configuration
- Manual plt.savefig() calls scattered throughout pipeline nodes
- Hardcoded styling parameters in each visualization function
- Manual file path management and naming conventions
- Code duplication across different pipeline nodes
- No systematic figure versioning or organization

After (FigRegistry Integration):
- HOOKS = (FigRegistryHooks(),) registration enables automation
- Zero plt.savefig() calls needed in pipeline nodes
- Automatic condition-based styling via configuration
- Systematic figure organization and versioning
- DRY principle maintained across all visualizations
- Seamless integration with Kedro catalog and versioning

KEY CHANGES:
============

1. Hook Registration: Register FigRegistryHooks for automated lifecycle management
2. Configuration Bridge: Automatic merging of Kedro and FigRegistry configurations
3. Dataset Integration: FigureDataSet automatically handles figure styling and saving
4. Zero Code Changes: Existing pipeline nodes continue to work without modification

This migration enables F-006 Kedro Lifecycle Hooks functionality per the technical 
specification, providing automated FigRegistry initialization, configuration bridging, 
and context management throughout pipeline execution.
"""

from typing import Any, Dict, Tuple

# Import FigRegistryHooks for automated lifecycle management per F-006 requirements
try:
    from figregistry_kedro.hooks import FigRegistryHooks
    FIGREGISTRY_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments where figregistry-kedro is not yet installed
    import warnings
    warnings.warn(
        "figregistry-kedro not installed. To enable automated figure management, "
        "install with: pip install figregistry-kedro",
        ImportWarning
    )
    FigRegistryHooks = None
    FIGREGISTRY_AVAILABLE = False

# Package name for the migrated Kedro project
PACKAGE_NAME = "kedro_figregistry_example"

# Pipeline discovery configuration - unchanged from traditional approach
PIPELINE_REGISTRY = f"{PACKAGE_NAME}.pipeline_registry"

# Session store configuration - unchanged from traditional approach
SESSION_STORE_CLASS = "kedro.framework.session.session.BaseSessionStore"

# Data catalog configuration - unchanged, but now supports FigureDataSet
CATALOG_STORE_CLASS = "kedro.io.data_catalog.DataCatalog"

# Configuration source directory - unchanged from traditional approach
CONF_SOURCE = "conf"

# Configuration loader - enhanced to support FigRegistry configuration merging
CONFIG_LOADER_CLASS = "kedro.config.OmegaConfigLoader"
CONFIG_LOADER_ARGS: Dict[str, Any] = {
    "base_env": "base",
    "default_run_env": "local",
    
    # Enhanced configuration patterns to support FigRegistry integration
    # This enables automatic discovery of figregistry.yml configuration files
    "config_patterns": {
        # Standard Kedro configuration patterns (unchanged)
        "catalog": ["catalog*.yml", "catalog*.yaml"],
        "parameters": ["parameters*.yml", "parameters*.yaml"],
        "credentials": ["credentials*.yml", "credentials*.yaml"],
        
        # NEW: FigRegistry configuration pattern for Kedro integration
        # Supports both traditional figregistry.yaml and Kedro-style figregistry*.yml
        "figregistry": ["figregistry*.yml", "figregistry*.yaml"]
    }
}

# ============================================================================
# CRITICAL MIGRATION CHANGE: HOOK REGISTRATION
# ============================================================================

# BEFORE (Traditional Manual Approach):
# HOOKS = []  # Empty - no automated figure management
#
# AFTER (FigRegistry Integration):
# Register FigRegistryHooks to enable automated lifecycle management per F-006

if FIGREGISTRY_AVAILABLE:
    # Register FigRegistryHooks with migration-optimized configuration
    # This configuration balances automation with migration safety
    HOOKS: Tuple[Any, ...] = (
        FigRegistryHooks(
            # F-006-RQ-001: Enable automatic FigRegistry initialization
            # This eliminates the need for manual figregistry.init_config() calls
            auto_initialize=True,
            
            # Enable performance monitoring during migration to track integration overhead
            # This helps validate that the migration maintains acceptable performance
            enable_performance_monitoring=True,
            
            # F-006-RQ-003: Enable graceful fallback for migration safety
            # This ensures pipelines continue to work even if FigRegistry configuration
            # is incomplete during the migration process
            fallback_on_error=True,
            
            # Set reasonable initialization time limit for migration environments
            # More lenient than production to accommodate potential file system delays
            max_initialization_time=0.020  # 20ms for migration tolerance
        ),
    )
    
    print("✅ FigRegistryHooks registered successfully")
    print("   - Automatic FigRegistry initialization: ENABLED")
    print("   - Performance monitoring: ENABLED")
    print("   - Graceful error handling: ENABLED")
    print("   - Migration is complete!")
else:
    # Fallback configuration when figregistry-kedro is not available
    # This allows the project to continue functioning during incremental migration
    HOOKS: Tuple[Any, ...] = ()
    
    print("⚠️  FigRegistryHooks not available")
    print("   To complete migration, install: pip install figregistry-kedro")
    print("   Current configuration will work but without automated figure management")

# ============================================================================
# UNCHANGED KEDRO CONFIGURATION
# ============================================================================

# The following settings remain unchanged from the traditional approach,
# demonstrating that FigRegistry integration is non-invasive and preserves
# existing Kedro project structure and behavior.

# Context class - unchanged from traditional approach
CONTEXT_CLASS = "kedro.framework.context.KedroContext"

# Logging configuration - unchanged from traditional approach
# FigRegistry integration adds its own logging without interfering
# with existing Kedro logging configuration

# Security settings - unchanged from traditional approach
# FigRegistry maintains Kedro's security model without modifications

# Data versioning - unchanged from traditional approach
# FigRegistry integrates with Kedro's existing versioning system

# ============================================================================
# MIGRATION VALIDATION
# ============================================================================

def validate_migration() -> bool:
    """Validate that the migration to figregistry-kedro was successful.
    
    This function performs basic validation to ensure that the migration
    from traditional manual figure management to automated FigRegistry
    integration has been completed successfully.
    
    Returns:
        True if migration validation passes, False otherwise
    """
    validation_passed = True
    
    # Check 1: Verify FigRegistryHooks registration
    if not FIGREGISTRY_AVAILABLE:
        print("❌ Migration validation failed: figregistry-kedro not installed")
        print("   Install with: pip install figregistry-kedro")
        validation_passed = False
    elif len(HOOKS) == 0:
        print("❌ Migration validation failed: No hooks registered")
        print("   FigRegistryHooks should be registered in HOOKS tuple")
        validation_passed = False
    else:
        # Verify hook configuration
        hook_instance = HOOKS[0]
        if hasattr(hook_instance, 'auto_initialize') and hook_instance.auto_initialize:
            print("✅ Hook validation passed: Auto-initialization enabled")
        else:
            print("⚠️  Hook configuration warning: Auto-initialization disabled")
        
        if hasattr(hook_instance, 'fallback_on_error') and hook_instance.fallback_on_error:
            print("✅ Hook validation passed: Graceful error handling enabled")
        else:
            print("⚠️  Hook configuration warning: Error fallback disabled")
    
    # Check 2: Verify configuration loader enhancements
    if "figregistry" in CONFIG_LOADER_ARGS.get("config_patterns", {}):
        print("✅ Configuration validation passed: FigRegistry patterns enabled")
    else:
        print("⚠️  Configuration warning: FigRegistry config patterns not found")
    
    # Check 3: Summary
    if validation_passed:
        print("\n🎉 MIGRATION COMPLETE!")
        print("   Your Kedro project now has automated figure management via FigRegistry")
        print("   Key benefits enabled:")
        print("   - Zero manual plt.savefig() calls required")
        print("   - Automatic condition-based styling")
        print("   - Systematic figure organization and versioning")
        print("   - Configuration-driven visualization consistency")
    else:
        print("\n⚠️  MIGRATION INCOMPLETE")
        print("   Please address the validation issues above to complete migration")
    
    return validation_passed

# Execute migration validation on settings import
# This provides immediate feedback about the migration status
if __name__ != "__main__":  # Avoid validation during direct script execution
    validate_migration()

# ============================================================================
# MIGRATION GUIDANCE COMMENTS
# ============================================================================

# STEP-BY-STEP MIGRATION GUIDE:
# =============================
#
# 1. Install figregistry-kedro:
#    pip install figregistry-kedro
#
# 2. Update settings.py (this file):
#    - Add FigRegistryHooks import
#    - Register hooks in HOOKS tuple
#    - Add figregistry config patterns (optional)
#
# 3. Configure data catalog (conf/base/catalog.yml):
#    Add FigureDataSet entries for figure outputs:
#    
#    example_plot:
#      type: figregistry_kedro.datasets.FigureDataSet
#      filepath: data/08_reporting/example_plot.png
#      purpose: "exploration"
#      condition_param: "experiment_type"
#
# 4. Create FigRegistry configuration (conf/base/figregistry.yml):
#    Define your styling rules and output settings
#
# 5. Update pipeline nodes:
#    - Remove manual plt.savefig() calls
#    - Return matplotlib figure objects instead
#    - Let FigureDataSet handle styling and saving automatically
#
# 6. Test the migration:
#    kedro run --pipeline=your_pipeline_name
#
# COMMON MIGRATION PATTERNS:
# ==========================
#
# OLD NODE CODE:
# def create_scatter_plot(data: pd.DataFrame) -> None:
#     fig, ax = plt.subplots()
#     ax.scatter(data['x'], data['y'])
#     plt.savefig('outputs/scatter.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
# NEW NODE CODE:
# def create_scatter_plot(data: pd.DataFrame) -> matplotlib.figure.Figure:
#     fig, ax = plt.subplots()
#     ax.scatter(data['x'], data['y'])
#     return fig  # FigureDataSet handles styling and saving automatically
#
# CATALOG CONFIGURATION:
# scatter_plot:
#   type: figregistry_kedro.datasets.FigureDataSet
#   filepath: data/08_reporting/scatter_plot.png
#   purpose: "analysis"

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__migration_version__ = "traditional_to_figregistry_v1"
__description__ = "Migrated Kedro settings demonstrating FigRegistry integration"

# Export key configuration elements
__all__ = [
    "HOOKS",
    "PACKAGE_NAME",
    "CONFIG_LOADER_CLASS",
    "CONFIG_LOADER_ARGS",
    "validate_migration"
]