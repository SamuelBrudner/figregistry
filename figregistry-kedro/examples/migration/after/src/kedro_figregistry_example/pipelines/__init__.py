"""Pipelines package for the Kedro FigRegistry integration migration example.

This package contains pipeline modules that demonstrate the conversion from manual
matplotlib figure management to automated figregistry-kedro integration. The pipelines
showcase how FigureDataSet automatically intercepts figure outputs, applies condition-based
styling, and handles versioned persistence without manual plt.savefig() calls.

The package structure follows standard Kedro conventions for pipeline organization,
enabling proper discovery and import of individual pipeline modules through Kedro's
framework while demonstrating the seamless integration patterns achievable with
figregistry-kedro.
"""

# Standard Python package initialization for Kedro pipeline discovery
# This enables proper namespace management and import patterns as required by
# F-008 packaging requirements and F-005 integration specifications

__all__ = []  # No explicit exports needed - pipeline discovery handled by registry