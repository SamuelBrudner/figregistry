"""FigRegistry: Configuration-driven figure generation and management."""

from pathlib import Path
from typing import Optional

from .core.config import Config, load_config
from .core.style import get_style
from .core.io import save_figure

__version__ = "0.3.0"

# Re-export public API
__all__ = ["Config", "load_config", "get_style", "save_figure"]

# Initialize default configuration
try:
    CONFIG = load_config("figregistry.yaml") if Path("figregistry.yaml").exists() else None
except Exception as e:
    import warnings
    warnings.warn(f"Failed to load default config: {e}")
    CONFIG = None

def get_config() -> Optional[Config]:
    """Get the current configuration."""
    return CONFIG
