"""Project settings for traditional Kedro project without figregistry-kedro integration.

This file demonstrates standard Kedro settings configuration patterns without
automated FigRegistry lifecycle management. Projects using this approach require
manual figure management setup and scattered plt.savefig() calls throughout
pipeline nodes.

This serves as a baseline example showing the traditional configuration
patterns that figregistry-kedro automation eliminates.
"""

# Package name for Kedro project
PACKAGE_NAME = "kedro_manual_example"

# Project pipelines discovery
PIPELINE_REGISTRY = f"{PACKAGE_NAME}.pipeline_registry"

# Session store for pipeline tracking
SESSION_STORE_CLASS = "kedro.framework.session.session.BaseSessionStore"

# Data catalog store
CATALOG_STORE_CLASS = "kedro.io.data_catalog.DataCatalog"

# Configuration source
CONF_SOURCE = "conf"

# Configuration loader - using standard OmegaConfigLoader
CONFIG_LOADER_CLASS = "kedro.config.OmegaConfigLoader"
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local"
}

# Traditional approach: No FigRegistryHooks registration
# HOOKS = []  # Empty hooks list - no automated figure management

# Kedro plugins (none for traditional approach)
# DISABLE_HOOKS_FOR_PLUGINS = ()

# Logging configuration (standard Kedro logging)
# from kedro.config import ConfigLoader
# conf_path = str(Path.cwd() / CONF_SOURCE)
# conf_loader = ConfigLoader(conf_source=conf_path)
# LOGGING = conf_loader["logging"]

# Security settings for non-sensitive local development
# SECURITY_CONTEXT_CLASS = "kedro.framework.security.SecurityContext"

# Data versioning (optional)
# VERSIONED_DATASETS = []

# Custom dataset mappings (none for traditional approach)
# DATASET_MAPPING = {}

# This traditional configuration requires manual figure management:
# - Manual plt.savefig() calls in each pipeline node
# - Hardcoded file paths and styling parameters
# - No automated condition-based styling
# - No systematic figure versioning
# - Code duplication across visualization functions
#
# Compare this with figregistry-kedro integration which provides:
# - Automatic FigRegistry initialization via hooks
# - FigureDataSet for automated styling and persistence
# - Configuration bridge for merged settings
# - Zero manual plt.savefig() calls required