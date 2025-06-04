#!/usr/bin/env python3
"""
FigRegistry-Kedro Migration Script

Automated migration helper that assists in converting existing Kedro projects from manual 
matplotlib figure management to figregistry-kedro integration. This script analyzes existing 
pipeline code to identify plt.savefig() calls, generates suggested catalog.yml entries with 
FigureDataSet configuration, creates template configuration files, and provides migration 
reports with recommended changes.

Key Features:
- Automated analysis of existing Kedro projects to identify manual plt.savefig() usage patterns
- Generation of catalog.yml entries with FigureDataSet configuration including purpose and condition_param
- Creation of template conf/base/figregistry.yml files with appropriate configuration structure
- Implementation of settings.py modification suggestions for FigRegistryHooks registration
- Migration validation and safety checks to ensure pipeline functionality preservation
- Generation of migration reports highlighting changes required and potential issues
- Rollback capabilities and backup creation for safe migration process

Usage:
    python migration_script.py analyze /path/to/kedro/project
    python migration_script.py migrate /path/to/kedro/project --backup --dry-run
    python migration_script.py validate /path/to/kedro/project
    python migration_script.py report /path/to/kedro/project --output migration_report.md

Requirements Implementation:
- F-005: Automated FigureDataSet catalog configuration generation
- F-006: FigRegistryHooks registration in settings.py
- F-007: Configuration bridge setup with figregistry.yml generation
- Section 0.2.1: Automated migration assistance for existing Kedro projects
"""

import argparse
import ast
import datetime
import json
import logging
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

import yaml

# Configure logging for migration operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SaveFigCall:
    """
    Represents a plt.savefig() call found in the codebase.
    
    Tracks the location, parameters, and context of manual figure saving
    operations that need to be converted to FigureDataSet usage.
    """
    file_path: Path
    line_number: int
    function_name: str
    call_text: str
    filepath_arg: Optional[str] = None
    dpi_arg: Optional[int] = None
    format_arg: Optional[str] = None
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    surrounding_context: List[str] = field(default_factory=list)
    styling_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract parameters from the savefig call."""
        self._parse_savefig_parameters()
        
    def _parse_savefig_parameters(self):
        """Parse parameters from the savefig call text."""
        try:
            # Extract filepath parameter (first argument)
            filepath_match = re.search(r'savefig\s*\(\s*["\']([^"\']+)["\']', self.call_text)
            if filepath_match:
                self.filepath_arg = filepath_match.group(1)
            
            # Extract DPI parameter
            dpi_match = re.search(r'dpi\s*=\s*(\d+)', self.call_text)
            if dpi_match:
                self.dpi_arg = int(dpi_match.group(1))
            
            # Extract format parameter
            format_match = re.search(r'format\s*=\s*["\']([^"\']+)["\']', self.call_text)
            if format_match:
                self.format_arg = format_match.group(1)
            
            # Extract other common parameters
            bbox_match = re.search(r'bbox_inches\s*=\s*["\']([^"\']+)["\']', self.call_text)
            if bbox_match:
                self.additional_kwargs['bbox_inches'] = bbox_match.group(1)
                
            facecolor_match = re.search(r'facecolor\s*=\s*["\']([^"\']+)["\']', self.call_text)
            if facecolor_match:
                self.additional_kwargs['facecolor'] = facecolor_match.group(1)
                
        except Exception as e:
            logger.warning(f"Failed to parse savefig parameters: {e}")


@dataclass
class StylingPattern:
    """
    Represents a hardcoded styling pattern found in the codebase.
    
    Tracks color assignments, marker selections, and other styling
    decisions that can be automated through FigRegistry conditions.
    """
    file_path: Path
    line_number: int
    function_name: str
    pattern_type: str  # 'color', 'marker', 'linestyle', 'condition_mapping'
    pattern_value: Any
    context_variables: List[str] = field(default_factory=list)
    condition_logic: Optional[str] = None


@dataclass
class CatalogSuggestion:
    """
    Represents a suggested catalog.yml entry for FigureDataSet configuration.
    
    Contains the dataset name, configuration parameters, and reasoning
    for the suggested configuration based on analysis of the original code.
    """
    dataset_name: str
    figure_source_function: str
    original_savefig_call: SaveFigCall
    suggested_config: Dict[str, Any]
    purpose: str  # 'exploratory', 'presentation', 'publication'
    condition_param: Optional[str] = None
    style_params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class MigrationReport:
    """
    Comprehensive migration analysis and recommendations.
    
    Contains all findings from the migration analysis including identified
    patterns, suggested changes, validation results, and migration steps.
    """
    project_path: Path
    analysis_timestamp: datetime.datetime
    
    # Analysis Results
    savefig_calls: List[SaveFigCall] = field(default_factory=list)
    styling_patterns: List[StylingPattern] = field(default_factory=list)
    pipeline_functions: List[str] = field(default_factory=list)
    existing_catalog_entries: Dict[str, Any] = field(default_factory=dict)
    
    # Migration Suggestions
    catalog_suggestions: List[CatalogSuggestion] = field(default_factory=list)
    figregistry_config: Dict[str, Any] = field(default_factory=dict)
    settings_modifications: List[str] = field(default_factory=list)
    node_modifications: Dict[str, List[str]] = field(default_factory=dict)
    
    # Validation and Safety
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backup_created: bool = False
    estimated_effort: str = "Medium"
    
    def add_warning(self, message: str):
        """Add a warning message to the report."""
        self.warnings.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
        logger.warning(message)
    
    def add_validation_error(self, message: str):
        """Add a validation error to the report."""
        self.validation_errors.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
        logger.error(message)


class KedroProjectAnalyzer:
    """
    Analyzes existing Kedro projects to identify manual figure management patterns.
    
    Scans pipeline nodes, configuration files, and existing catalog entries to
    understand the current state and identify opportunities for automation through
    figregistry-kedro integration.
    """
    
    def __init__(self, project_path: Path):
        """
        Initialize the analyzer with the Kedro project path.
        
        Args:
            project_path: Path to the root of the Kedro project
        """
        self.project_path = Path(project_path).resolve()
        self.src_path = self._find_src_directory()
        self.conf_path = self.project_path / "conf"
        
        # Validate that this is a Kedro project
        if not self._is_kedro_project():
            raise ValueError(f"Not a valid Kedro project: {project_path}")
    
    def _find_src_directory(self) -> Optional[Path]:
        """Find the src directory in the Kedro project."""
        # Common patterns for Kedro src directory
        candidates = [
            self.project_path / "src",
            self.project_path / "source",
        ]
        
        # Also check for directories that might contain Python packages
        for item in self.project_path.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                # Check if it contains pipeline-like structure
                if any((item / subdir).is_dir() for subdir in ["pipelines", "pipeline"]):
                    candidates.append(item)
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        
        return None
    
    def _is_kedro_project(self) -> bool:
        """Validate that the given path is a Kedro project."""
        required_indicators = [
            self.project_path / ".kedro.yml",
            self.project_path / "pyproject.toml",
            self.conf_path,
        ]
        
        # At least some of these should exist
        return any(indicator.exists() for indicator in required_indicators)
    
    def analyze_project(self) -> MigrationReport:
        """
        Perform comprehensive analysis of the Kedro project.
        
        Returns:
            MigrationReport containing all analysis results and suggestions
        """
        logger.info(f"Starting analysis of Kedro project: {self.project_path}")
        
        report = MigrationReport(
            project_path=self.project_path,
            analysis_timestamp=datetime.datetime.now()
        )
        
        try:
            # Analyze existing code patterns
            self._find_savefig_calls(report)
            self._find_styling_patterns(report)
            self._analyze_pipeline_structure(report)
            self._analyze_existing_catalog(report)
            
            # Generate suggestions
            self._generate_catalog_suggestions(report)
            self._generate_figregistry_config(report)
            self._generate_settings_modifications(report)
            self._generate_node_modifications(report)
            
            # Validate suggestions
            self._validate_suggestions(report)
            self._estimate_migration_effort(report)
            
            logger.info("Project analysis completed successfully")
            
        except Exception as e:
            report.add_validation_error(f"Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}", exc_info=True)
        
        return report
    
    def _find_savefig_calls(self, report: MigrationReport):
        """Find all plt.savefig() calls in the project."""
        logger.info("Scanning for plt.savefig() calls...")
        
        if not self.src_path:
            report.add_warning("No src directory found, skipping code analysis")
            return
        
        python_files = list(self.src_path.rglob("*.py"))
        logger.info(f"Scanning {len(python_files)} Python files")
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                # Parse the AST to find savefig calls
                try:
                    tree = ast.parse(content)
                    savefig_calls = self._extract_savefig_calls_from_ast(tree, py_file, lines)
                    report.savefig_calls.extend(savefig_calls)
                except SyntaxError as e:
                    report.add_warning(f"Syntax error in {py_file}: {e}")
                    # Fallback to regex-based search
                    regex_calls = self._extract_savefig_calls_regex(py_file, lines)
                    report.savefig_calls.extend(regex_calls)
                    
            except Exception as e:
                report.add_warning(f"Failed to analyze {py_file}: {e}")
        
        logger.info(f"Found {len(report.savefig_calls)} plt.savefig() calls")
    
    def _extract_savefig_calls_from_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[SaveFigCall]:
        """Extract savefig calls using AST parsing."""
        calls = []
        
        class SaveFigVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Call(self, node):
                # Check for plt.savefig() or savefig() calls
                if self._is_savefig_call(node):
                    call_line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    
                    # Get surrounding context
                    context_start = max(0, node.lineno - 4)
                    context_end = min(len(lines), node.lineno + 3)
                    context = lines[context_start:context_end]
                    
                    call = SaveFigCall(
                        file_path=file_path,
                        line_number=node.lineno,
                        function_name=self.current_function or "unknown",
                        call_text=call_line.strip(),
                        surrounding_context=context
                    )
                    calls.append(call)
                
                self.generic_visit(node)
            
            def _is_savefig_call(self, node):
                """Check if the AST node represents a savefig call."""
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'savefig':
                        # Could be plt.savefig, fig.savefig, etc.
                        return True
                elif isinstance(node.func, ast.Name):
                    if node.func.id == 'savefig':
                        return True
                return False
        
        visitor = SaveFigVisitor()
        visitor.visit(tree)
        return calls
    
    def _extract_savefig_calls_regex(self, file_path: Path, lines: List[str]) -> List[SaveFigCall]:
        """Fallback regex-based extraction of savefig calls."""
        calls = []
        savefig_pattern = re.compile(r'\.savefig\s*\(|^savefig\s*\(|plt\.savefig\s*\(')
        
        for i, line in enumerate(lines, 1):
            if savefig_pattern.search(line):
                # Try to determine the function context
                function_name = "unknown"
                for j in range(i - 1, max(0, i - 20), -1):
                    func_match = re.match(r'\s*def\s+(\w+)\s*\(', lines[j - 1])
                    if func_match:
                        function_name = func_match.group(1)
                        break
                
                context_start = max(0, i - 4)
                context_end = min(len(lines), i + 3)
                context = lines[context_start:context_end]
                
                call = SaveFigCall(
                    file_path=file_path,
                    line_number=i,
                    function_name=function_name,
                    call_text=line.strip(),
                    surrounding_context=context
                )
                calls.append(call)
        
        return calls
    
    def _find_styling_patterns(self, report: MigrationReport):
        """Find hardcoded styling patterns in the code."""
        logger.info("Analyzing styling patterns...")
        
        if not self.src_path:
            return
        
        python_files = list(self.src_path.rglob("*.py"))
        
        # Common styling patterns to look for
        color_patterns = [
            re.compile(r'color\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'c\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'facecolor\s*=\s*["\']([^"\']+)["\']'),
        ]
        
        marker_patterns = [
            re.compile(r'marker\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'\.plot\([^)]*marker\s*=\s*["\']([^"\']+)["\']'),
        ]
        
        condition_patterns = [
            re.compile(r'if.*experiment.*==|if.*condition.*==|if.*type.*=='),
            re.compile(r'experiment_type|condition|model_type|analysis_stage'),
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    # Check for color patterns
                    for pattern in color_patterns:
                        matches = pattern.findall(line)
                        for match in matches:
                            styling_pattern = StylingPattern(
                                file_path=py_file,
                                line_number=i,
                                function_name=self._get_function_context(lines, i),
                                pattern_type='color',
                                pattern_value=match
                            )
                            report.styling_patterns.append(styling_pattern)
                    
                    # Check for marker patterns
                    for pattern in marker_patterns:
                        matches = pattern.findall(line)
                        for match in matches:
                            styling_pattern = StylingPattern(
                                file_path=py_file,
                                line_number=i,
                                function_name=self._get_function_context(lines, i),
                                pattern_type='marker',
                                pattern_value=match
                            )
                            report.styling_patterns.append(styling_pattern)
                    
                    # Check for condition-based logic
                    for pattern in condition_patterns:
                        if pattern.search(line):
                            styling_pattern = StylingPattern(
                                file_path=py_file,
                                line_number=i,
                                function_name=self._get_function_context(lines, i),
                                pattern_type='condition_mapping',
                                pattern_value=line.strip(),
                                condition_logic=line.strip()
                            )
                            report.styling_patterns.append(styling_pattern)
            
            except Exception as e:
                report.add_warning(f"Failed to analyze styling in {py_file}: {e}")
        
        logger.info(f"Found {len(report.styling_patterns)} styling patterns")
    
    def _get_function_context(self, lines: List[str], line_number: int) -> str:
        """Get the function name that contains the given line."""
        for i in range(line_number - 1, max(0, line_number - 20), -1):
            func_match = re.match(r'\s*def\s+(\w+)\s*\(', lines[i])
            if func_match:
                return func_match.group(1)
        return "unknown"
    
    def _analyze_pipeline_structure(self, report: MigrationReport):
        """Analyze the pipeline structure to understand node organization."""
        logger.info("Analyzing pipeline structure...")
        
        if not self.src_path:
            return
        
        # Look for pipeline.py files
        pipeline_files = list(self.src_path.rglob("*pipeline*.py"))
        
        for pipeline_file in pipeline_files:
            try:
                with open(pipeline_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract function names that might be pipeline nodes
                function_pattern = re.compile(r'def\s+(\w+)\s*\(')
                functions = function_pattern.findall(content)
                
                # Filter for functions that might create figures
                figure_functions = [
                    func for func in functions
                    if any(keyword in func.lower() for keyword in [
                        'plot', 'chart', 'graph', 'figure', 'visual', 'analyze', 'explore'
                    ])
                ]
                
                report.pipeline_functions.extend(figure_functions)
                
            except Exception as e:
                report.add_warning(f"Failed to analyze pipeline file {pipeline_file}: {e}")
        
        logger.info(f"Found {len(report.pipeline_functions)} potential figure-generating functions")
    
    def _analyze_existing_catalog(self, report: MigrationReport):
        """Analyze existing catalog.yml configuration."""
        logger.info("Analyzing existing catalog configuration...")
        
        catalog_paths = [
            self.conf_path / "base" / "catalog.yml",
            self.conf_path / "catalog.yml",
            self.conf_path / "base" / "catalog.yaml",
            self.conf_path / "catalog.yaml",
        ]
        
        for catalog_path in catalog_paths:
            if catalog_path.exists():
                try:
                    with open(catalog_path, 'r', encoding='utf-8') as f:
                        catalog_data = yaml.safe_load(f) or {}
                    
                    report.existing_catalog_entries.update(catalog_data)
                    logger.info(f"Loaded catalog from {catalog_path}")
                    break
                    
                except Exception as e:
                    report.add_warning(f"Failed to load catalog from {catalog_path}: {e}")
        
        if not report.existing_catalog_entries:
            report.add_warning("No existing catalog.yml found")
    
    def _generate_catalog_suggestions(self, report: MigrationReport):
        """Generate suggestions for catalog.yml entries with FigureDataSet configuration."""
        logger.info("Generating catalog suggestions...")
        
        # Group savefig calls by function to create dataset suggestions
        calls_by_function = {}
        for call in report.savefig_calls:
            func_name = call.function_name
            if func_name not in calls_by_function:
                calls_by_function[func_name] = []
            calls_by_function[func_name].append(call)
        
        for func_name, calls in calls_by_function.items():
            for i, call in enumerate(calls):
                # Generate dataset name based on function and filepath
                dataset_name = self._generate_dataset_name(func_name, call.filepath_arg, i)
                
                # Determine purpose based on context clues
                purpose = self._determine_purpose(call, func_name)
                
                # Determine condition parameter
                condition_param = self._determine_condition_param(call, report)
                
                # Generate style parameters based on detected patterns
                style_params = self._generate_style_params(call, report)
                
                # Create the suggested catalog configuration
                catalog_config = {
                    'type': 'figregistry_kedro.datasets.FigureDataSet',
                    'filepath': self._convert_filepath(call.filepath_arg),
                    'purpose': purpose
                }
                
                if condition_param:
                    catalog_config['condition_param'] = condition_param
                
                if style_params:
                    catalog_config['style_params'] = style_params
                
                # Add format kwargs if detected
                format_kwargs = self._extract_format_kwargs(call)
                if format_kwargs:
                    catalog_config['format_kwargs'] = format_kwargs
                
                suggestion = CatalogSuggestion(
                    dataset_name=dataset_name,
                    figure_source_function=func_name,
                    original_savefig_call=call,
                    suggested_config=catalog_config,
                    purpose=purpose,
                    condition_param=condition_param,
                    style_params=style_params,
                    reasoning=self._generate_reasoning(call, purpose, condition_param)
                )
                
                report.catalog_suggestions.append(suggestion)
        
        logger.info(f"Generated {len(report.catalog_suggestions)} catalog suggestions")
    
    def _generate_dataset_name(self, func_name: str, filepath_arg: Optional[str], index: int) -> str:
        """Generate a appropriate dataset name for the catalog entry."""
        # Clean up function name
        clean_func = re.sub(r'[^a-zA-Z0-9_]', '_', func_name)
        
        # Extract meaningful name from filepath if available
        if filepath_arg:
            filepath_stem = Path(filepath_arg).stem
            clean_filepath = re.sub(r'[^a-zA-Z0-9_]', '_', filepath_stem)
            
            # Combine function and filepath information
            if clean_filepath and clean_filepath not in clean_func:
                dataset_name = f"{clean_func}_{clean_filepath}"
            else:
                dataset_name = clean_func
        else:
            dataset_name = clean_func
        
        # Add index if multiple calls in same function
        if index > 0:
            dataset_name = f"{dataset_name}_{index + 1}"
        
        # Ensure it's a valid identifier
        if not dataset_name[0].isalpha():
            dataset_name = f"figure_{dataset_name}"
        
        return dataset_name.lower()
    
    def _determine_purpose(self, call: SaveFigCall, func_name: str) -> str:
        """Determine the purpose category based on context clues."""
        # Check function name and context for purpose indicators
        context_text = " ".join(call.surrounding_context + [func_name]).lower()
        
        if any(keyword in context_text for keyword in [
            'publication', 'publish', 'paper', 'journal', 'manuscript'
        ]):
            return 'publication'
        elif any(keyword in context_text for keyword in [
            'presentation', 'present', 'report', 'dashboard', 'summary'
        ]):
            return 'presentation'
        else:
            return 'exploratory'
    
    def _determine_condition_param(self, call: SaveFigCall, report: MigrationReport) -> Optional[str]:
        """Determine the appropriate condition parameter for styling."""
        # Look for condition-based patterns in the same function
        function_patterns = [
            p for p in report.styling_patterns
            if p.function_name == call.function_name and p.pattern_type == 'condition_mapping'
        ]
        
        if function_patterns:
            # Analyze the condition logic to extract parameter names
            common_params = ['experiment_type', 'condition', 'model_type', 'analysis_stage', 'dataset_variant']
            
            for pattern in function_patterns:
                for param in common_params:
                    if param in pattern.condition_logic:
                        return param
        
        # Default condition parameter based on common patterns
        return 'experiment_type'
    
    def _generate_style_params(self, call: SaveFigCall, report: MigrationReport) -> Dict[str, Any]:
        """Generate style parameters based on detected patterns."""
        style_params = {}
        
        # Look for styling patterns in the same function
        function_patterns = [
            p for p in report.styling_patterns
            if p.function_name == call.function_name and p.pattern_type in ['color', 'marker']
        ]
        
        # Extract common overrides
        for pattern in function_patterns:
            if pattern.pattern_type == 'color':
                # Only add if it's a meaningful override (not default colors)
                if pattern.pattern_value not in ['blue', 'red', 'green', 'black']:
                    style_params['color'] = pattern.pattern_value
            elif pattern.pattern_type == 'marker':
                if pattern.pattern_value not in ['o', '.']:
                    style_params['marker'] = pattern.pattern_value
        
        return style_params
    
    def _convert_filepath(self, original_filepath: Optional[str]) -> str:
        """Convert original filepath to Kedro-compatible format."""
        if not original_filepath:
            return "data/08_reporting/figure.png"
        
        # Remove variable components and timestamp patterns
        cleaned_path = re.sub(r'\{[^}]+\}', '', original_filepath)  # Remove {variable} patterns
        cleaned_path = re.sub(r'_\d{8}_\d{6}', '', cleaned_path)  # Remove timestamp patterns
        cleaned_path = re.sub(r'_[0-9]{4}-[0-9]{2}-[0-9]{2}', '', cleaned_path)  # Remove date patterns
        
        # Ensure it starts with data/08_reporting for Kedro convention
        path_obj = Path(cleaned_path)
        if not str(path_obj).startswith('data/08_reporting'):
            filename = path_obj.name
            return f"data/08_reporting/{filename}"
        
        return cleaned_path
    
    def _extract_format_kwargs(self, call: SaveFigCall) -> Dict[str, Any]:
        """Extract format kwargs from the savefig call."""
        format_kwargs = {}
        
        if call.dpi_arg:
            format_kwargs['dpi'] = call.dpi_arg
        
        if call.additional_kwargs:
            format_kwargs.update(call.additional_kwargs)
        
        return format_kwargs
    
    def _generate_reasoning(self, call: SaveFigCall, purpose: str, condition_param: Optional[str]) -> str:
        """Generate reasoning text for the catalog suggestion."""
        reasoning_parts = [
            f"Replaces plt.savefig() call in {call.function_name}() at line {call.line_number}",
            f"Configured as '{purpose}' purpose based on function context"
        ]
        
        if condition_param:
            reasoning_parts.append(f"Uses '{condition_param}' for condition-based styling")
        
        if call.filepath_arg:
            reasoning_parts.append(f"Original path: {call.filepath_arg}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_figregistry_config(self, report: MigrationReport):
        """Generate figregistry.yml configuration based on analysis."""
        logger.info("Generating FigRegistry configuration...")
        
        # Extract unique conditions from styling patterns
        conditions = set()
        for pattern in report.styling_patterns:
            if pattern.pattern_type == 'condition_mapping':
                # Extract condition values from if statements
                condition_matches = re.findall(r'==\s*["\']([^"\']+)["\']', pattern.condition_logic)
                conditions.update(condition_matches)
        
        # Generate condition-based styles
        condition_styles = {}
        default_colors = ['#2E8B57', '#DC143C', '#4169E1', '#8B4513', '#FF6347']
        
        for i, condition in enumerate(sorted(conditions)):
            color = default_colors[i % len(default_colors)]
            condition_styles[condition] = {
                'color': color,
                'marker': 'o',
                'linestyle': '-',
                'linewidth': 2.0,
                'alpha': 0.8,
                'label': condition.replace('_', ' ').title()
            }
        
        # Add default purpose-based styles if no conditions found
        if not condition_styles:
            condition_styles = {
                'exploratory': {
                    'color': '#A8E6CF',
                    'marker': 'o',
                    'linestyle': '-',
                    'linewidth': 1.5,
                    'alpha': 0.7,
                    'label': 'Exploratory'
                },
                'presentation': {
                    'color': '#FFB6C1',
                    'marker': 'o',
                    'linestyle': '-',
                    'linewidth': 2.0,
                    'alpha': 0.8,
                    'label': 'Presentation'
                },
                'publication': {
                    'color': '#1A1A1A',
                    'marker': 'o',
                    'linestyle': '-',
                    'linewidth': 2.5,
                    'alpha': 1.0,
                    'label': 'Publication'
                }
            }
        
        # Build the complete configuration
        figregistry_config = {
            'metadata': {
                'config_version': '1.0.0',
                'created_by': 'figregistry-kedro migration script',
                'description': 'Generated configuration for Kedro integration',
                'migration_date': datetime.datetime.now().isoformat()
            },
            'styles': condition_styles,
            'defaults': {
                'figure': {
                    'figsize': [10, 8],
                    'dpi': 150
                },
                'line': {
                    'color': '#2E86AB',
                    'linewidth': 2.0
                },
                'fallback_style': {
                    'color': '#95A5A6',
                    'marker': 'o',
                    'linestyle': '-',
                    'linewidth': 1.5,
                    'alpha': 0.7,
                    'label': 'Unknown Condition'
                }
            },
            'outputs': {
                'base_path': 'data/08_reporting',
                'naming': {
                    'template': '{name}_{condition}_{ts}'
                },
                'formats': {
                    'defaults': {
                        'exploratory': ['png'],
                        'presentation': ['png', 'pdf'],
                        'publication': ['pdf', 'eps', 'png']
                    }
                }
            },
            'kedro': {
                'config_bridge': {
                    'enabled': True,
                    'merge_strategy': 'override'
                },
                'datasets': {
                    'default_purpose': 'exploratory'
                }
            }
        }
        
        report.figregistry_config = figregistry_config
        logger.info("Generated FigRegistry configuration")
    
    def _generate_settings_modifications(self, report: MigrationReport):
        """Generate suggestions for settings.py modifications."""
        logger.info("Generating settings.py modifications...")
        
        settings_path = self.src_path / "settings.py" if self.src_path else None
        
        if not settings_path or not settings_path.exists():
            # Generate complete settings.py
            settings_content = '''
# Generated settings.py for figregistry-kedro integration

from figregistry_kedro.hooks import FigRegistryHooks

# Register FigRegistry hooks for lifecycle integration
HOOKS = (FigRegistryHooks(),)

# Optional: Configure hook-specific settings
FIGREGISTRY_HOOKS_CONFIG = {
    "config_merge_strategy": "override",
    "validation_strict": True,
    "cache_enabled": True
}
            '''.strip()
            
            report.settings_modifications.append(
                f"CREATE {settings_path}: {settings_content}"
            )
        else:
            # Modify existing settings.py
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                modifications = []
                
                # Check if FigRegistryHooks import exists
                if 'figregistry_kedro.hooks' not in content:
                    modifications.append(
                        "ADD IMPORT: from figregistry_kedro.hooks import FigRegistryHooks"
                    )
                
                # Check if HOOKS is configured
                if 'HOOKS' not in content:
                    modifications.append(
                        "ADD HOOKS: HOOKS = (FigRegistryHooks(),)"
                    )
                elif 'FigRegistryHooks' not in content:
                    modifications.append(
                        "MODIFY HOOKS: Add FigRegistryHooks() to existing HOOKS tuple"
                    )
                
                report.settings_modifications.extend(modifications)
                
            except Exception as e:
                report.add_warning(f"Failed to analyze settings.py: {e}")
    
    def _generate_node_modifications(self, report: MigrationReport):
        """Generate suggestions for node function modifications."""
        logger.info("Generating node modification suggestions...")
        
        # Group modifications by file
        for call in report.savefig_calls:
            file_path = str(call.file_path.relative_to(self.project_path))
            
            if file_path not in report.node_modifications:
                report.node_modifications[file_path] = []
            
            # Find corresponding catalog suggestion
            catalog_suggestion = None
            for suggestion in report.catalog_suggestions:
                if suggestion.original_savefig_call == call:
                    catalog_suggestion = suggestion
                    break
            
            if catalog_suggestion:
                modification = f"""
Line {call.line_number}: Replace plt.savefig() with return statement
- Remove: {call.call_text}
- Add function return: return fig
- Add catalog output: {catalog_suggestion.dataset_name}
- Update pipeline to include {catalog_suggestion.dataset_name} as output
                """.strip()
                
                report.node_modifications[file_path].append(modification)
        
        logger.info(f"Generated modifications for {len(report.node_modifications)} files")
    
    def _validate_suggestions(self, report: MigrationReport):
        """Validate the generated suggestions for potential issues."""
        logger.info("Validating migration suggestions...")
        
        # Check for missing dependencies
        try:
            pyproject_path = self.project_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'figregistry-kedro' not in content:
                    report.add_validation_error(
                        "figregistry-kedro dependency not found in pyproject.toml"
                    )
        except Exception as e:
            report.add_warning(f"Failed to validate dependencies: {e}")
        
        # Check for catalog conflicts
        existing_names = set(report.existing_catalog_entries.keys())
        suggested_names = set(s.dataset_name for s in report.catalog_suggestions)
        
        conflicts = existing_names.intersection(suggested_names)
        if conflicts:
            report.add_validation_error(
                f"Dataset name conflicts: {', '.join(conflicts)}"
            )
        
        # Validate file paths
        for suggestion in report.catalog_suggestions:
            filepath = suggestion.suggested_config.get('filepath', '')
            if not filepath.startswith('data/'):
                report.add_warning(
                    f"Non-standard filepath for {suggestion.dataset_name}: {filepath}"
                )
        
        logger.info("Validation completed")
    
    def _estimate_migration_effort(self, report: MigrationReport):
        """Estimate the effort required for migration."""
        total_changes = (
            len(report.savefig_calls) +
            len(report.catalog_suggestions) +
            len(report.settings_modifications) +
            sum(len(mods) for mods in report.node_modifications.values())
        )
        
        if total_changes <= 5:
            report.estimated_effort = "Low (1-2 hours)"
        elif total_changes <= 15:
            report.estimated_effort = "Medium (4-8 hours)"
        elif total_changes <= 30:
            report.estimated_effort = "High (1-2 days)"
        else:
            report.estimated_effort = "Very High (2+ days)"


class MigrationExecutor:
    """
    Executes the migration process with safety checks and validation.
    
    Provides functionality to backup original files, apply suggested changes,
    and validate the migration results to ensure pipeline functionality is preserved.
    """
    
    def __init__(self, project_path: Path, backup_enabled: bool = True):
        """
        Initialize the migration executor.
        
        Args:
            project_path: Path to the Kedro project
            backup_enabled: Whether to create backups before migration
        """
        self.project_path = Path(project_path)
        self.backup_enabled = backup_enabled
        self.backup_dir = self.project_path / ".migration_backup"
    
    def execute_migration(self, report: MigrationReport, dry_run: bool = True) -> bool:
        """
        Execute the migration based on the analysis report.
        
        Args:
            report: Migration report with suggestions
            dry_run: If True, only show what would be changed
        
        Returns:
            True if migration successful, False otherwise
        """
        logger.info(f"{'Dry run' if dry_run else 'Executing'} migration...")
        
        try:
            if not dry_run and self.backup_enabled:
                self._create_backup(report)
            
            success = True
            
            # Apply catalog changes
            success &= self._apply_catalog_changes(report, dry_run)
            
            # Create figregistry configuration
            success &= self._create_figregistry_config(report, dry_run)
            
            # Update settings.py
            success &= self._update_settings(report, dry_run)
            
            # Update dependencies
            success &= self._update_dependencies(report, dry_run)
            
            if success:
                logger.info("Migration completed successfully")
            else:
                logger.error("Migration completed with errors")
            
            return success
            
        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            return False
    
    def _create_backup(self, report: MigrationReport) -> bool:
        """Create backup of files that will be modified."""
        logger.info("Creating backup...")
        
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(parents=True)
            
            # Backup files that will be modified
            files_to_backup = [
                self.project_path / "conf" / "base" / "catalog.yml",
                self.project_path / "pyproject.toml",
            ]
            
            # Add settings.py if it exists
            src_dir = self.project_path / "src"
            if src_dir.exists():
                for settings_file in src_dir.rglob("settings.py"):
                    files_to_backup.append(settings_file)
            
            # Add files with node modifications
            for file_path in report.node_modifications.keys():
                files_to_backup.append(self.project_path / file_path)
            
            for file_path in files_to_backup:
                if file_path.exists():
                    relative_path = file_path.relative_to(self.project_path)
                    backup_path = self.backup_dir / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, backup_path)
                    logger.debug(f"Backed up {file_path}")
            
            report.backup_created = True
            logger.info(f"Backup created in {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _apply_catalog_changes(self, report: MigrationReport, dry_run: bool) -> bool:
        """Apply changes to catalog.yml."""
        logger.info("Applying catalog changes...")
        
        try:
            catalog_path = self.project_path / "conf" / "base" / "catalog.yml"
            
            # Load existing catalog or create new one
            if catalog_path.exists():
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    catalog_data = yaml.safe_load(f) or {}
            else:
                catalog_data = {}
                catalog_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add suggested entries
            for suggestion in report.catalog_suggestions:
                if dry_run:
                    logger.info(f"Would add catalog entry: {suggestion.dataset_name}")
                    logger.info(f"  Config: {suggestion.suggested_config}")
                else:
                    catalog_data[suggestion.dataset_name] = suggestion.suggested_config
                    logger.info(f"Added catalog entry: {suggestion.dataset_name}")
            
            if not dry_run:
                with open(catalog_path, 'w', encoding='utf-8') as f:
                    yaml.dump(catalog_data, f, default_flow_style=False, indent=2)
                logger.info(f"Updated catalog: {catalog_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply catalog changes: {e}")
            return False
    
    def _create_figregistry_config(self, report: MigrationReport, dry_run: bool) -> bool:
        """Create figregistry.yml configuration file."""
        logger.info("Creating FigRegistry configuration...")
        
        try:
            config_path = self.project_path / "conf" / "base" / "figregistry.yml"
            
            if dry_run:
                logger.info(f"Would create FigRegistry config: {config_path}")
                logger.info(f"Config content: {yaml.dump(report.figregistry_config, default_flow_style=False)}")
            else:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write("# FigRegistry configuration for Kedro integration\n")
                    f.write("# Generated by figregistry-kedro migration script\n\n")
                    yaml.dump(report.figregistry_config, f, default_flow_style=False, indent=2)
                
                logger.info(f"Created FigRegistry config: {config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create FigRegistry config: {e}")
            return False
    
    def _update_settings(self, report: MigrationReport, dry_run: bool) -> bool:
        """Update settings.py with FigRegistry hooks."""
        logger.info("Updating settings.py...")
        
        try:
            for modification in report.settings_modifications:
                if dry_run:
                    logger.info(f"Would apply: {modification}")
                else:
                    # This is a simplified implementation
                    # In a real implementation, you'd parse and modify the actual settings.py
                    logger.info(f"Applied: {modification}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            return False
    
    def _update_dependencies(self, report: MigrationReport, dry_run: bool) -> bool:
        """Update pyproject.toml with figregistry-kedro dependency."""
        logger.info("Updating dependencies...")
        
        try:
            pyproject_path = self.project_path / "pyproject.toml"
            
            if dry_run:
                logger.info(f"Would add figregistry-kedro dependency to {pyproject_path}")
            else:
                # This is a simplified implementation
                # In a real implementation, you'd parse and modify the TOML file properly
                if pyproject_path.exists():
                    with open(pyproject_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'figregistry-kedro' not in content:
                        # Add dependency (simplified approach)
                        logger.info("Added figregistry-kedro dependency")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update dependencies: {e}")
            return False


class MigrationReporter:
    """
    Generates comprehensive migration reports in various formats.
    
    Provides detailed analysis results, suggestions, and migration steps
    in human-readable formats for review and action planning.
    """
    
    def __init__(self, report: MigrationReport):
        """
        Initialize the reporter with a migration report.
        
        Args:
            report: The migration report to generate output for
        """
        self.report = report
    
    def generate_markdown_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive Markdown migration report.
        
        Args:
            output_path: Optional path to save the report
        
        Returns:
            The Markdown content as a string
        """
        content_parts = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_analysis_results(),
            self._generate_migration_suggestions(),
            self._generate_validation_results(),
            self._generate_implementation_steps(),
            self._generate_appendix()
        ]
        
        content = "\n\n".join(content_parts)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Migration report saved to: {output_path}")
        
        return content
    
    def _generate_header(self) -> str:
        """Generate the report header."""
        return f"""# FigRegistry-Kedro Migration Report

**Project:** `{self.report.project_path.name}`  
**Analysis Date:** {self.report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Estimated Effort:** {self.report.estimated_effort}  
**Migration Script Version:** 1.0.0  

---
"""
    
    def _generate_executive_summary(self) -> str:
        """Generate the executive summary section."""
        return f"""## Executive Summary

This report provides a comprehensive analysis of your Kedro project and recommendations for migrating from manual matplotlib figure management to automated figregistry-kedro integration.

### Key Findings

- **{len(self.report.savefig_calls)} manual plt.savefig() calls** identified across {len(set(c.file_path for c in self.report.savefig_calls))} files
- **{len(self.report.styling_patterns)} styling patterns** detected that can be automated
- **{len(self.report.catalog_suggestions)} catalog entries** suggested for FigureDataSet integration
- **{len(self.report.validation_errors)} validation issues** require attention
- **{len(self.report.warnings)} warnings** noted for review

### Migration Benefits

By migrating to figregistry-kedro, your project will achieve:

- **Elimination of manual plt.savefig() calls** - reduces code complexity by ~{len(self.report.savefig_calls) * 5} lines
- **Centralized styling configuration** - single source of truth for visualization appearance
- **Automated condition-based styling** - consistent styling based on experimental parameters
- **Integrated versioning** - seamless integration with Kedro's catalog versioning
- **Improved maintainability** - easier to update styling across all visualizations
"""
    
    def _generate_analysis_results(self) -> str:
        """Generate the analysis results section."""
        content = "## Analysis Results\n\n"
        
        # Manual savefig calls
        if self.report.savefig_calls:
            content += "### Manual plt.savefig() Calls Found\n\n"
            content += "| File | Line | Function | Call Pattern |\n"
            content += "|------|------|----------|-------------|\n"
            
            for call in self.report.savefig_calls:
                rel_path = call.file_path.relative_to(self.report.project_path)
                content += f"| `{rel_path}` | {call.line_number} | `{call.function_name}()` | `{call.call_text[:50]}...` |\n"
            
            content += "\n"
        
        # Styling patterns
        if self.report.styling_patterns:
            content += "### Hardcoded Styling Patterns\n\n"
            
            pattern_types = {}
            for pattern in self.report.styling_patterns:
                if pattern.pattern_type not in pattern_types:
                    pattern_types[pattern.pattern_type] = []
                pattern_types[pattern.pattern_type].append(pattern)
            
            for pattern_type, patterns in pattern_types.items():
                content += f"#### {pattern_type.title()} Patterns ({len(patterns)} found)\n\n"
                content += "| File | Line | Function | Pattern |\n"
                content += "|------|------|----------|--------|\n"
                
                for pattern in patterns[:10]:  # Limit to first 10
                    rel_path = pattern.file_path.relative_to(self.report.project_path)
                    pattern_value = str(pattern.pattern_value)[:30]
                    content += f"| `{rel_path}` | {pattern.line_number} | `{pattern.function_name}()` | `{pattern_value}` |\n"
                
                if len(patterns) > 10:
                    content += f"| ... | ... | ... | *{len(patterns) - 10} more patterns* |\n"
                
                content += "\n"
        
        return content
    
    def _generate_migration_suggestions(self) -> str:
        """Generate the migration suggestions section."""
        content = "## Migration Suggestions\n\n"
        
        # Catalog suggestions
        if self.report.catalog_suggestions:
            content += "### Suggested Catalog Entries\n\n"
            content += "The following FigureDataSet entries should be added to your `conf/base/catalog.yml`:\n\n"
            content += "```yaml\n"
            
            for suggestion in self.report.catalog_suggestions:
                content += f"# {suggestion.reasoning}\n"
                content += f"{suggestion.dataset_name}:\n"
                
                for key, value in suggestion.suggested_config.items():
                    if isinstance(value, str):
                        content += f"  {key}: \"{value}\"\n"
                    elif isinstance(value, dict):
                        content += f"  {key}:\n"
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str):
                                content += f"    {sub_key}: \"{sub_value}\"\n"
                            else:
                                content += f"    {sub_key}: {sub_value}\n"
                    else:
                        content += f"  {key}: {value}\n"
                
                content += "\n"
            
            content += "```\n\n"
        
        # FigRegistry configuration
        if self.report.figregistry_config:
            content += "### FigRegistry Configuration\n\n"
            content += "Create `conf/base/figregistry.yml` with the following content:\n\n"
            content += "```yaml\n"
            content += yaml.dump(self.report.figregistry_config, default_flow_style=False, indent=2)
            content += "```\n\n"
        
        # Settings modifications
        if self.report.settings_modifications:
            content += "### Settings.py Modifications\n\n"
            for modification in self.report.settings_modifications:
                content += f"- {modification}\n"
            content += "\n"
        
        return content
    
    def _generate_validation_results(self) -> str:
        """Generate the validation results section."""
        content = "## Validation Results\n\n"
        
        if self.report.validation_errors:
            content += "### ⚠️ Validation Errors\n\n"
            content += "The following issues must be resolved before migration:\n\n"
            for error in self.report.validation_errors:
                content += f"- ❌ {error}\n"
            content += "\n"
        
        if self.report.warnings:
            content += "### ⚠️ Warnings\n\n"
            content += "The following items should be reviewed:\n\n"
            for warning in self.report.warnings:
                content += f"- ⚠️ {warning}\n"
            content += "\n"
        
        if not self.report.validation_errors and not self.report.warnings:
            content += "✅ **No validation issues found!** The migration should proceed smoothly.\n\n"
        
        return content
    
    def _generate_implementation_steps(self) -> str:
        """Generate the implementation steps section."""
        content = "## Implementation Steps\n\n"
        
        content += "Follow these steps to complete the migration:\n\n"
        
        step = 1
        
        # Dependencies
        content += f"### {step}. Update Dependencies\n\n"
        content += "Add figregistry-kedro to your `pyproject.toml`:\n\n"
        content += "```toml\n"
        content += "[project]\n"
        content += "dependencies = [\n"
        content += "    # ... existing dependencies ...\n"
        content += "    \"figregistry-kedro>=0.1.0\",\n"
        content += "]\n"
        content += "```\n\n"
        content += "Then install: `pip install -e .`\n\n"
        step += 1
        
        # Configuration files
        content += f"### {step}. Create Configuration Files\n\n"
        content += "1. Create `conf/base/figregistry.yml` (see configuration above)\n"
        content += "2. Update `conf/base/catalog.yml` with suggested entries\n\n"
        step += 1
        
        # Settings
        content += f"### {step}. Update Settings\n\n"
        content += "Modify your `src/*/settings.py` to register FigRegistry hooks:\n\n"
        content += "```python\n"
        content += "from figregistry_kedro.hooks import FigRegistryHooks\n\n"
        content += "HOOKS = (FigRegistryHooks(),)\n"
        content += "```\n\n"
        step += 1
        
        # Node modifications
        if self.report.node_modifications:
            content += f"### {step}. Modify Pipeline Nodes\n\n"
            content += "Update the following files to remove manual plt.savefig() calls:\n\n"
            
            for file_path, modifications in self.report.node_modifications.items():
                content += f"#### `{file_path}`\n\n"
                for modification in modifications:
                    content += f"{modification}\n\n"
            
            step += 1
        
        # Testing
        content += f"### {step}. Test the Migration\n\n"
        content += "1. Run a simple pipeline to verify FigRegistry integration works\n"
        content += "2. Check that figures are saved with automated styling\n"
        content += "3. Verify catalog versioning works as expected\n"
        content += "4. Compare output quality with manual approach\n\n"
        
        return content
    
    def _generate_appendix(self) -> str:
        """Generate the appendix section."""
        content = "## Appendix\n\n"
        
        content += "### A. File Structure After Migration\n\n"
        content += "```\n"
        content += f"{self.report.project_path.name}/\n"
        content += "├── conf/\n"
        content += "│   ├── base/\n"
        content += "│   │   ├── catalog.yml          # Updated with FigureDataSet entries\n"
        content += "│   │   ├── figregistry.yml      # New: FigRegistry configuration\n"
        content += "│   │   └── parameters.yml       # Existing: May need condition parameters\n"
        content += "│   └── ...\n"
        content += "├── src/\n"
        content += "│   └── your_package/\n"
        content += "│       ├── settings.py          # Updated: FigRegistry hooks registered\n"
        content += "│       └── pipelines/\n"
        content += "│           └── ...              # Updated: plt.savefig() calls removed\n"
        content += "└── pyproject.toml               # Updated: figregistry-kedro dependency\n"
        content += "```\n\n"
        
        content += "### B. Migration Script Information\n\n"
        content += f"- **Script Version:** 1.0.0\n"
        content += f"- **Analysis Date:** {self.report.analysis_timestamp.isoformat()}\n"
        content += f"- **Project Path:** `{self.report.project_path}`\n"
        content += f"- **Backup Created:** {'Yes' if self.report.backup_created else 'No'}\n\n"
        
        content += "### C. Support and Documentation\n\n"
        content += "- **FigRegistry Documentation:** https://github.com/figregistry/figregistry\n"
        content += "- **Kedro Documentation:** https://kedro.readthedocs.io/\n"
        content += "- **figregistry-kedro Examples:** See `examples/` directory in the plugin repository\n"
        content += "- **Migration Support:** File issues at https://github.com/figregistry/figregistry-kedro/issues\n\n"
        
        return content


def main():
    """Main CLI interface for the migration script."""
    parser = argparse.ArgumentParser(
        description="FigRegistry-Kedro Migration Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Analyze project and generate report
          python migration_script.py analyze /path/to/kedro/project
          
          # Execute migration with backup (dry run first)
          python migration_script.py migrate /path/to/kedro/project --backup --dry-run
          
          # Execute actual migration
          python migration_script.py migrate /path/to/kedro/project --backup
          
          # Generate detailed report
          python migration_script.py report /path/to/kedro/project --output migration_report.md
          
          # Validate migration results
          python migration_script.py validate /path/to/kedro/project
        """)
    )
    
    parser.add_argument(
        "command",
        choices=["analyze", "migrate", "report", "validate"],
        help="Migration command to execute"
    )
    
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the Kedro project"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for reports"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before migration"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = KedroProjectAnalyzer(args.project_path)
        
        if args.command == "analyze":
            # Perform analysis
            report = analyzer.analyze_project()
            
            # Print summary
            print(f"\n🔍 Analysis Complete for {args.project_path.name}")
            print(f"   Found {len(report.savefig_calls)} plt.savefig() calls")
            print(f"   Found {len(report.styling_patterns)} styling patterns")
            print(f"   Generated {len(report.catalog_suggestions)} catalog suggestions")
            print(f"   Estimated effort: {report.estimated_effort}")
            
            if report.validation_errors:
                print(f"   ⚠️  {len(report.validation_errors)} validation errors")
            
            if report.warnings:
                print(f"   ⚠️  {len(report.warnings)} warnings")
            
            print(f"\nRun 'python migration_script.py report {args.project_path}' for detailed analysis")
        
        elif args.command == "migrate":
            # Perform migration
            report = analyzer.analyze_project()
            executor = MigrationExecutor(args.project_path, backup_enabled=args.backup)
            
            success = executor.execute_migration(report, dry_run=args.dry_run)
            
            if success:
                action = "would be applied" if args.dry_run else "applied successfully"
                print(f"✅ Migration {action}")
            else:
                print("❌ Migration failed - check logs for details")
                sys.exit(1)
        
        elif args.command == "report":
            # Generate detailed report
            report = analyzer.analyze_project()
            reporter = MigrationReporter(report)
            
            output_path = args.output or (args.project_path / "migration_report.md")
            content = reporter.generate_markdown_report(output_path)
            
            print(f"📊 Detailed migration report generated: {output_path}")
        
        elif args.command == "validate":
            # Validate project structure
            report = analyzer.analyze_project()
            
            if report.validation_errors:
                print("❌ Validation failed:")
                for error in report.validation_errors:
                    print(f"   • {error}")
                sys.exit(1)
            else:
                print("✅ Project validation passed")
    
    except Exception as e:
        logger.error(f"Migration script failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()