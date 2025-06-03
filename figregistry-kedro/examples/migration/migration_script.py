#!/usr/bin/env python3
"""FigRegistry-Kedro Migration Script.

This automated migration tool assists in converting existing Kedro projects from manual 
matplotlib figure management to figregistry-kedro integration. It analyzes existing 
pipeline code to identify plt.savefig() calls, generates catalog.yml entries with 
FigureDataSet configuration, creates template configuration files, and provides 
comprehensive migration reports with recommended changes.

Features:
- Automated analysis of existing Kedro projects for manual figure management patterns
- Generation of catalog.yml entries with FigureDataSet configuration
- Creation of template figregistry.yml configuration files
- Settings.py modification suggestions for FigRegistryHooks registration
- Validation and safety checks to ensure migration success
- Comprehensive rollback capabilities and backup creation
- Detailed migration reports highlighting changes and potential issues

Usage:
    python migration_script.py /path/to/kedro/project [options]

Requirements:
- Must provide automated migration assistance per Section 0.2.1 implementation goals
- Must analyze and suggest FigureDataSet configurations per F-005 requirements
- Must generate appropriate configuration files per F-007 requirements  
- Must include validation and safety checks for successful migration
"""

import argparse
import ast
import logging
import os
import re
import shutil
import sys
import textwrap
import yaml
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from kedro.config import ConfigLoader
    from kedro.framework.project import find_pipelines
    kedro_available = True
except ImportError:
    kedro_available = False
    ConfigLoader = None

try:
    import figregistry
    figregistry_available = True
except ImportError:
    figregistry_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('figregistry_migration')

# Migration analysis data structures
SaveFigCall = namedtuple('SaveFigCall', [
    'file_path', 'line_number', 'line_content', 'function_name', 
    'suggested_dataset_name', 'inferred_purpose', 'filepath_arg'
])

CatalogEntry = namedtuple('CatalogEntry', [
    'name', 'type', 'filepath', 'purpose', 'condition_param', 
    'style_params', 'save_args'
])

MigrationIssue = namedtuple('MigrationIssue', [
    'severity', 'category', 'description', 'file_path', 
    'line_number', 'suggestion'
])


class MigrationError(Exception):
    """Exception raised when migration analysis or execution fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class KedroProjectAnalyzer:
    """Analyzes existing Kedro projects to identify migration opportunities and issues.
    
    This class implements comprehensive analysis of Kedro projects to identify:
    - Manual plt.savefig() calls that can be automated with FigureDataSet
    - Existing catalog structure and naming patterns
    - Pipeline organization and function signatures
    - Configuration files and their structure
    - Potential migration issues and compatibility concerns
    """
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize analyzer with Kedro project path.
        
        Args:
            project_path: Path to the Kedro project root directory
            
        Raises:
            MigrationError: If project path is invalid or not a Kedro project
        """
        self.project_path = Path(project_path).resolve()
        self.src_path = self._find_src_directory()
        self.conf_path = self.project_path / "conf"
        
        # Analysis results
        self.savefig_calls: List[SaveFigCall] = []
        self.existing_catalog: Dict[str, Any] = {}
        self.pipeline_structure: Dict[str, List[str]] = {}
        self.migration_issues: List[MigrationIssue] = []
        
        # Configuration
        self.supported_extensions = {'.py'}
        self.figure_purposes = ['exploratory', 'presentation', 'publication']
        
        # Validate project structure
        self._validate_kedro_project()
        
        logger.info(f"Initialized analyzer for Kedro project: {self.project_path}")
    
    def _find_src_directory(self) -> Path:
        """Find the source directory containing pipeline code."""
        potential_paths = [
            self.project_path / "src",
            self.project_path / "kedro_project" / "src",
        ]
        
        # Look for any directory containing __init__.py in src structure
        for path in potential_paths:
            if path.exists() and any(path.rglob("__init__.py")):
                return path
        
        # Fallback: look for Python files in typical locations
        if (self.project_path / "pipelines").exists():
            return self.project_path
        
        raise MigrationError(
            f"Could not find source directory in Kedro project: {self.project_path}",
            {"searched_paths": [str(p) for p in potential_paths]}
        )
    
    def _validate_kedro_project(self) -> None:
        """Validate that the target directory is a valid Kedro project."""
        required_files = [".kedro.yml", "pyproject.toml"]
        missing_files = []
        
        for file_name in required_files:
            if not (self.project_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise MigrationError(
                f"Not a valid Kedro project - missing files: {missing_files}",
                {"project_path": str(self.project_path), "missing_files": missing_files}
            )
        
        # Check for conf directory
        if not self.conf_path.exists():
            self.migration_issues.append(MigrationIssue(
                severity="warning",
                category="structure",
                description="No conf directory found - will create during migration",
                file_path=str(self.conf_path),
                line_number=None,
                suggestion="Create conf/base directory structure for configuration files"
            ))
    
    def analyze_project(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the Kedro project.
        
        Returns:
            Dictionary containing analysis results including:
            - savefig_calls: List of identified plt.savefig() calls
            - catalog_entries: Suggested FigureDataSet catalog entries
            - migration_issues: List of potential migration issues
            - project_info: Basic project metadata and structure
        """
        logger.info("Starting comprehensive project analysis...")
        
        # Analyze Python files for plt.savefig() calls
        self._analyze_python_files()
        
        # Load and analyze existing catalog
        self._analyze_existing_catalog()
        
        # Analyze pipeline structure
        self._analyze_pipeline_structure()
        
        # Generate suggested catalog entries
        suggested_entries = self._generate_catalog_entries()
        
        # Perform validation checks
        self._validate_migration_feasibility()
        
        analysis_results = {
            "savefig_calls": self.savefig_calls,
            "catalog_entries": suggested_entries,
            "migration_issues": self.migration_issues,
            "project_info": {
                "path": str(self.project_path),
                "src_path": str(self.src_path),
                "conf_path": str(self.conf_path),
                "python_files_analyzed": len(self._get_python_files()),
                "pipelines_found": list(self.pipeline_structure.keys())
            },
            "existing_catalog": self.existing_catalog
        }
        
        logger.info(f"Analysis complete: found {len(self.savefig_calls)} plt.savefig() calls")
        return analysis_results
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project source directory."""
        python_files = []
        
        for ext in self.supported_extensions:
            python_files.extend(self.src_path.rglob(f"*{ext}"))
        
        # Filter out __pycache__ and test files for now
        filtered_files = [
            f for f in python_files 
            if '__pycache__' not in str(f) and 'test_' not in f.name
        ]
        
        return filtered_files
    
    def _analyze_python_files(self) -> None:
        """Analyze Python files to identify plt.savefig() calls."""
        python_files = self._get_python_files()
        
        logger.info(f"Analyzing {len(python_files)} Python files for plt.savefig() calls...")
        
        for file_path in python_files:
            try:
                self._analyze_single_file(file_path)
            except Exception as e:
                self.migration_issues.append(MigrationIssue(
                    severity="warning",
                    category="analysis",
                    description=f"Failed to analyze file: {e}",
                    file_path=str(file_path),
                    line_number=None,
                    suggestion="Manual review required for this file"
                ))
                logger.warning(f"Failed to analyze {file_path}: {e}")
    
    def _analyze_single_file(self, file_path: Path) -> None:
        """Analyze a single Python file for plt.savefig() calls."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Could not read {file_path} - encoding issue")
            return
        
        # Use both regex and AST analysis for comprehensive detection
        self._analyze_with_regex(file_path, content)
        self._analyze_with_ast(file_path, content)
    
    def _analyze_with_regex(self, file_path: Path, content: str) -> None:
        """Use regex patterns to find plt.savefig() calls."""
        patterns = [
            r'plt\.savefig\s*\(',
            r'figure\.savefig\s*\(',
            r'fig\.savefig\s*\(',
            r'ax\.figure\.savefig\s*\(',
            r'matplotlib\.pyplot\.savefig\s*\('
        ]
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    # Extract function context
                    function_name = self._extract_function_name(lines, line_num - 1)
                    
                    # Extract filepath argument if possible
                    filepath_arg = self._extract_filepath_argument(line)
                    
                    # Generate suggested dataset name and purpose
                    dataset_name = self._generate_dataset_name(file_path, function_name, filepath_arg)
                    purpose = self._infer_purpose(file_path, function_name, filepath_arg)
                    
                    savefig_call = SaveFigCall(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.strip(),
                        function_name=function_name,
                        suggested_dataset_name=dataset_name,
                        inferred_purpose=purpose,
                        filepath_arg=filepath_arg
                    )
                    
                    self.savefig_calls.append(savefig_call)
                    break  # Only one match per line
    
    def _analyze_with_ast(self, file_path: Path, content: str) -> None:
        """Use AST analysis to find plt.savefig() calls with more context."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.migration_issues.append(MigrationIssue(
                severity="warning",
                category="syntax",
                description=f"Syntax error in file: {e}",
                file_path=str(file_path),
                line_number=e.lineno,
                suggestion="Fix syntax errors before migration"
            ))
            return
        
        # AST visitor to find savefig calls
        class SaveFigVisitor(ast.NodeVisitor):
            def __init__(self, analyzer_instance):
                self.analyzer = analyzer_instance
                self.file_path = file_path
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Call(self, node):
                # Check for savefig method calls
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'savefig':
                    # Get the line content
                    lines = content.split('\n')
                    line_content = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    
                    # Extract filepath from arguments if possible
                    filepath_arg = None
                    if node.args:
                        if isinstance(node.args[0], ast.Str):
                            filepath_arg = node.args[0].s
                        elif isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            filepath_arg = node.args[0].value
                    
                    # Check if this call is already found by regex
                    already_found = any(
                        call.file_path == str(file_path) and call.line_number == node.lineno
                        for call in self.analyzer.savefig_calls
                    )
                    
                    if not already_found:
                        dataset_name = self.analyzer._generate_dataset_name(
                            file_path, self.current_function, filepath_arg
                        )
                        purpose = self.analyzer._infer_purpose(
                            file_path, self.current_function, filepath_arg
                        )
                        
                        savefig_call = SaveFigCall(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            line_content=line_content,
                            function_name=self.current_function,
                            suggested_dataset_name=dataset_name,
                            inferred_purpose=purpose,
                            filepath_arg=filepath_arg
                        )
                        
                        self.analyzer.savefig_calls.append(savefig_call)
                
                self.generic_visit(node)
        
        visitor = SaveFigVisitor(self)
        visitor.visit(tree)
    
    def _extract_function_name(self, lines: List[str], line_index: int) -> Optional[str]:
        """Extract the function name containing the plt.savefig() call."""
        # Look backwards for function definition
        for i in range(line_index, max(-1, line_index - 50), -1):
            line = lines[i].strip()
            if line.startswith('def '):
                match = re.match(r'def\s+(\w+)\s*\(', line)
                if match:
                    return match.group(1)
        return None
    
    def _extract_filepath_argument(self, line: str) -> Optional[str]:
        """Extract filepath argument from savefig() call."""
        # Simple regex to extract string arguments
        patterns = [
            r'savefig\s*\(\s*["\']([^"\']+)["\']',
            r'savefig\s*\(\s*([^,\)]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1).strip('\'"')
        
        return None
    
    def _generate_dataset_name(self, file_path: Path, function_name: Optional[str], 
                             filepath_arg: Optional[str]) -> str:
        """Generate a suggested dataset name for the catalog entry."""
        # Priority order for naming
        if filepath_arg:
            # Use filename without extension
            name = Path(filepath_arg).stem
            # Clean up the name
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            if name and not name[0].isdigit():
                return name
        
        if function_name:
            # Use function name as base
            name = function_name
            if not name.endswith('_plot') and not name.endswith('_figure'):
                name += '_plot'
            return name
        
        # Use file name as fallback
        file_stem = file_path.stem
        if file_stem not in ['__init__', 'nodes', 'pipeline']:
            return f"{file_stem}_plot"
        
        return "generated_plot"
    
    def _infer_purpose(self, file_path: Path, function_name: Optional[str], 
                      filepath_arg: Optional[str]) -> str:
        """Infer the purpose (exploratory/presentation/publication) from context."""
        # Check file path patterns
        path_str = str(file_path).lower()
        
        if any(term in path_str for term in ['report', 'presentation', 'pres']):
            return 'presentation'
        elif any(term in path_str for term in ['publish', 'publication', 'paper']):
            return 'publication'
        elif any(term in path_str for term in ['explore', 'eda', 'analysis']):
            return 'exploratory'
        
        # Check function name patterns
        if function_name:
            func_lower = function_name.lower()
            if any(term in func_lower for term in ['report', 'present']):
                return 'presentation'
            elif any(term in func_lower for term in ['publish', 'final']):
                return 'publication'
            elif any(term in func_lower for term in ['explore', 'quick', 'debug']):
                return 'exploratory'
        
        # Check filepath argument patterns
        if filepath_arg:
            filepath_lower = filepath_arg.lower()
            if any(term in filepath_lower for term in ['report', 'present']):
                return 'presentation'
            elif any(term in filepath_lower for term in ['publish', 'final']):
                return 'publication'
        
        # Default to exploratory
        return 'exploratory'
    
    def _analyze_existing_catalog(self) -> None:
        """Analyze existing catalog.yml files."""
        catalog_files = [
            self.conf_path / "base" / "catalog.yml",
            self.conf_path / "catalog.yml",
            self.project_path / "catalog.yml"
        ]
        
        for catalog_file in catalog_files:
            if catalog_file.exists():
                try:
                    with open(catalog_file, 'r') as f:
                        catalog_data = yaml.safe_load(f) or {}
                    
                    # Merge catalog data
                    self.existing_catalog.update(catalog_data)
                    logger.info(f"Loaded catalog from {catalog_file}")
                    break
                    
                except Exception as e:
                    self.migration_issues.append(MigrationIssue(
                        severity="warning",
                        category="catalog",
                        description=f"Failed to load catalog: {e}",
                        file_path=str(catalog_file),
                        line_number=None,
                        suggestion="Check catalog.yml syntax"
                    ))
    
    def _analyze_pipeline_structure(self) -> None:
        """Analyze pipeline structure to understand project organization."""
        # Look for pipeline directories
        pipeline_dirs = []
        
        if (self.src_path / "pipelines").exists():
            pipeline_dirs.extend((self.src_path / "pipelines").glob("*/"))
        
        # Also check for top-level pipeline.py files
        for pipeline_dir in pipeline_dirs:
            if pipeline_dir.is_dir():
                pipeline_name = pipeline_dir.name
                python_files = list(pipeline_dir.glob("*.py"))
                self.pipeline_structure[pipeline_name] = [f.name for f in python_files]
    
    def _generate_catalog_entries(self) -> List[CatalogEntry]:
        """Generate suggested catalog entries for FigureDataSet."""
        catalog_entries = []
        
        for call in self.savefig_calls:
            # Determine output filepath
            if call.filepath_arg:
                # Use the filepath from the original call, but in data/08_reporting
                filepath = f"data/08_reporting/{Path(call.filepath_arg).name}"
            else:
                # Generate filepath based on dataset name
                filepath = f"data/08_reporting/{call.suggested_dataset_name}.png"
            
            # Determine condition parameter
            condition_param = None
            if call.function_name:
                # Look for common parameter patterns
                if any(term in call.function_name.lower() for term in ['condition', 'experiment']):
                    condition_param = "experiment_condition"
            
            # Generate style parameters based on purpose
            style_params = {}
            if call.inferred_purpose == 'publication':
                style_params = {
                    "figure.dpi": 300,
                    "figure.facecolor": "white",
                    "axes.labelsize": 12,
                    "axes.titlesize": 14
                }
            elif call.inferred_purpose == 'presentation':
                style_params = {
                    "figure.dpi": 150,
                    "figure.facecolor": "white",
                    "axes.labelsize": 14,
                    "axes.titlesize": 16
                }
            
            # Generate save arguments
            save_args = {"bbox_inches": "tight"}
            if call.inferred_purpose in ['publication', 'presentation']:
                save_args["transparent"] = False
            
            catalog_entry = CatalogEntry(
                name=call.suggested_dataset_name,
                type="figregistry_kedro.FigureDataSet",
                filepath=filepath,
                purpose=call.inferred_purpose,
                condition_param=condition_param,
                style_params=style_params if style_params else None,
                save_args=save_args
            )
            
            catalog_entries.append(catalog_entry)
        
        return catalog_entries
    
    def _validate_migration_feasibility(self) -> None:
        """Validate that migration is feasible and identify potential issues."""
        # Check for dependencies
        if not kedro_available:
            self.migration_issues.append(MigrationIssue(
                severity="error",
                category="dependency",
                description="Kedro is not installed",
                file_path=None,
                line_number=None,
                suggestion="Install kedro>=0.18.0,<0.20.0"
            ))
        
        # Check for complex savefig patterns that might need manual review
        for call in self.savefig_calls:
            line = call.line_content.lower()
            
            # Check for complex arguments
            if any(term in line for term in ['**kwargs', '*args', 'format=', 'dpi=']):
                self.migration_issues.append(MigrationIssue(
                    severity="info",
                    category="complex_usage",
                    description="Complex savefig() call may need manual review",
                    file_path=call.file_path,
                    line_number=call.line_number,
                    suggestion="Review save_args configuration for this dataset"
                ))
        
        # Check for naming conflicts
        dataset_names = [call.suggested_dataset_name for call in self.savefig_calls]
        existing_names = set(self.existing_catalog.keys())
        
        for name in dataset_names:
            if name in existing_names:
                self.migration_issues.append(MigrationIssue(
                    severity="warning",
                    category="naming",
                    description=f"Dataset name '{name}' already exists in catalog",
                    file_path=None,
                    line_number=None,
                    suggestion=f"Consider renaming to '{name}_figure' or similar"
                ))


class FigRegistryKedroMigrator:
    """Executes migration from manual matplotlib to figregistry-kedro integration.
    
    This class handles the actual migration process including:
    - Creating backup of original project
    - Generating and updating configuration files
    - Modifying catalog.yml with FigureDataSet entries
    - Creating template figregistry.yml configuration
    - Updating settings.py for hook registration
    - Generating migration reports and validation
    """
    
    def __init__(self, analyzer: KedroProjectAnalyzer):
        """Initialize migrator with analysis results.
        
        Args:
            analyzer: Configured KedroProjectAnalyzer with completed analysis
        """
        self.analyzer = analyzer
        self.project_path = analyzer.project_path
        self.backup_path = None
        self.migration_results = {}
        
        logger.info(f"Initialized migrator for project: {self.project_path}")
    
    def create_backup(self, backup_suffix: str = None) -> Path:
        """Create a backup of the original project before migration.
        
        Args:
            backup_suffix: Optional suffix for backup directory name
            
        Returns:
            Path to the created backup directory
        """
        if backup_suffix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_suffix = f"backup_{timestamp}"
        
        backup_name = f"{self.project_path.name}_{backup_suffix}"
        self.backup_path = self.project_path.parent / backup_name
        
        logger.info(f"Creating backup at: {self.backup_path}")
        
        try:
            shutil.copytree(self.project_path, self.backup_path)
            logger.info("Backup created successfully")
            return self.backup_path
        except Exception as e:
            raise MigrationError(f"Failed to create backup: {e}")
    
    def execute_migration(self, dry_run: bool = False, 
                         create_backup: bool = True) -> Dict[str, Any]:
        """Execute the complete migration process.
        
        Args:
            dry_run: If True, show what would be done without making changes
            create_backup: If True, create backup before migration
            
        Returns:
            Dictionary containing migration results and status
        """
        logger.info(f"Starting migration (dry_run={dry_run})")
        
        if create_backup and not dry_run:
            self.create_backup()
        
        results = {
            "status": "success",
            "dry_run": dry_run,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "files_created": [],
            "files_modified": [],
            "catalog_entries_added": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 1. Update pyproject.toml with figregistry-kedro dependency
            self._update_dependencies(dry_run, results)
            
            # 2. Create or update catalog.yml with FigureDataSet entries
            self._update_catalog(dry_run, results)
            
            # 3. Create figregistry.yml configuration file
            self._create_figregistry_config(dry_run, results)
            
            # 4. Update settings.py for hooks registration
            self._update_settings(dry_run, results)
            
            # 5. Generate code modification suggestions
            self._generate_code_suggestions(results)
            
            # 6. Create migration report
            self._create_migration_report(dry_run, results)
            
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            logger.error(f"Migration failed: {e}")
        
        self.migration_results = results
        return results
    
    def _update_dependencies(self, dry_run: bool, results: Dict[str, Any]) -> None:
        """Update pyproject.toml to include figregistry-kedro dependency."""
        pyproject_path = self.project_path / "pyproject.toml"
        
        if not pyproject_path.exists():
            results["warnings"].append("pyproject.toml not found - manual dependency addition required")
            return
        
        try:
            with open(pyproject_path, 'r') as f:
                content = f.read()
            
            # Check if figregistry-kedro is already in dependencies
            if 'figregistry-kedro' in content:
                logger.info("figregistry-kedro dependency already present")
                return
            
            # Find dependencies section and add figregistry-kedro
            lines = content.split('\n')
            updated_lines = []
            in_dependencies = False
            dependency_added = False
            
            for line in lines:
                if line.strip().startswith('[project.dependencies]') or line.strip().startswith('dependencies = ['):
                    in_dependencies = True
                elif line.strip().startswith('[') and in_dependencies:
                    # End of dependencies section
                    in_dependencies = False
                
                updated_lines.append(line)
                
                # Add figregistry-kedro dependency
                if in_dependencies and not dependency_added and line.strip().endswith(','):
                    # Add after other dependencies
                    updated_lines.append('    "figregistry-kedro>=0.1.0",')
                    dependency_added = True
            
            if not dependency_added:
                # Try to add to the end of dependencies array
                for i, line in enumerate(updated_lines):
                    if 'dependencies' in line and '[' in line:
                        # Find the closing bracket
                        for j in range(i + 1, len(updated_lines)):
                            if ']' in updated_lines[j]:
                                # Insert before closing bracket
                                updated_lines.insert(j, '    "figregistry-kedro>=0.1.0",')
                                dependency_added = True
                                break
                        break
            
            if dependency_added:
                if not dry_run:
                    with open(pyproject_path, 'w') as f:
                        f.write('\n'.join(updated_lines))
                    logger.info("Updated pyproject.toml with figregistry-kedro dependency")
                    results["files_modified"].append(str(pyproject_path))
                else:
                    logger.info("Would update pyproject.toml with figregistry-kedro dependency")
            else:
                results["warnings"].append("Could not automatically add figregistry-kedro dependency - manual addition required")
                
        except Exception as e:
            results["warnings"].append(f"Failed to update pyproject.toml: {e}")
    
    def _update_catalog(self, dry_run: bool, results: Dict[str, Any]) -> None:
        """Update catalog.yml with FigureDataSet entries."""
        # Determine catalog file location
        catalog_path = self.project_path / "conf" / "base" / "catalog.yml"
        
        # Create conf/base directory if it doesn't exist
        if not catalog_path.parent.exists():
            if not dry_run:
                catalog_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {catalog_path.parent}")
            else:
                logger.info(f"Would create directory: {catalog_path.parent}")
        
        # Load existing catalog or create new one
        catalog_data = {}
        if catalog_path.exists():
            try:
                with open(catalog_path, 'r') as f:
                    catalog_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded existing catalog from {catalog_path}")
            except Exception as e:
                results["warnings"].append(f"Failed to load existing catalog: {e}")
                catalog_data = {}
        
        # Generate catalog entries
        analysis_results = self.analyzer.analyze_project()
        catalog_entries = analysis_results["catalog_entries"]
        
        # Add FigureDataSet entries
        entries_added = 0
        for entry in catalog_entries:
            if entry.name not in catalog_data:
                catalog_entry = {
                    "type": entry.type,
                    "filepath": entry.filepath,
                    "purpose": entry.purpose
                }
                
                if entry.condition_param:
                    catalog_entry["condition_param"] = entry.condition_param
                
                if entry.style_params:
                    catalog_entry["style_params"] = entry.style_params
                
                if entry.save_args:
                    catalog_entry["save_args"] = entry.save_args
                
                catalog_data[entry.name] = catalog_entry
                entries_added += 1
                logger.info(f"Added catalog entry: {entry.name}")
            else:
                results["warnings"].append(f"Catalog entry '{entry.name}' already exists - skipping")
        
        # Write updated catalog
        if entries_added > 0:
            if not dry_run:
                with open(catalog_path, 'w') as f:
                    yaml.dump(catalog_data, f, default_flow_style=False, sort_keys=False)
                logger.info(f"Updated catalog with {entries_added} FigureDataSet entries")
                results["files_modified"].append(str(catalog_path))
            else:
                logger.info(f"Would add {entries_added} FigureDataSet entries to catalog")
            
            results["catalog_entries_added"] = entries_added
    
    def _create_figregistry_config(self, dry_run: bool, results: Dict[str, Any]) -> None:
        """Create template figregistry.yml configuration file."""
        config_path = self.project_path / "conf" / "base" / "figregistry.yml"
        
        if config_path.exists():
            logger.info("figregistry.yml already exists - skipping creation")
            return
        
        # Generate template configuration
        template_config = {
            "figregistry_version": "0.3.0",
            "styles": {
                "exploratory": {
                    "figure.figsize": [10, 6],
                    "figure.dpi": 100,
                    "axes.labelsize": 10,
                    "axes.titlesize": 12,
                    "lines.linewidth": 1.5
                },
                "presentation": {
                    "figure.figsize": [12, 8],
                    "figure.dpi": 150,
                    "figure.facecolor": "white",
                    "axes.labelsize": 14,
                    "axes.titlesize": 16,
                    "lines.linewidth": 2
                },
                "publication": {
                    "figure.figsize": [8, 6],
                    "figure.dpi": 300,
                    "figure.facecolor": "white",
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "lines.linewidth": 1.5,
                    "font.family": "serif"
                }
            },
            "outputs": {
                "base_path": "data/08_reporting",
                "timestamp_format": "{name}_{ts:%Y%m%d_%H%M%S}",
                "purpose_paths": {
                    "exploratory": "exploratory",
                    "presentation": "presentation", 
                    "publication": "publication"
                }
            },
            "defaults": {
                "purpose": "exploratory",
                "format": "png",
                "save_args": {
                    "bbox_inches": "tight",
                    "transparent": False
                }
            }
        }
        
        if not dry_run:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(template_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Created figregistry.yml configuration: {config_path}")
            results["files_created"].append(str(config_path))
        else:
            logger.info(f"Would create figregistry.yml configuration: {config_path}")
    
    def _update_settings(self, dry_run: bool, results: Dict[str, Any]) -> None:
        """Update settings.py to register FigRegistryHooks."""
        # Find settings.py location
        potential_paths = [
            self.project_path / "src" / "settings.py",
            self.project_path / "settings.py"
        ]
        
        # Look for settings.py in package directories
        for src_dir in self.project_path.rglob("src"):
            for package_dir in src_dir.iterdir():
                if package_dir.is_dir() and (package_dir / "__init__.py").exists():
                    potential_paths.append(package_dir / "settings.py")
        
        settings_path = None
        for path in potential_paths:
            if path.exists():
                settings_path = path
                break
        
        if not settings_path:
            # Create basic settings.py
            settings_path = self.project_path / "src" / "settings.py"
            
            settings_content = '''"""Project settings and hooks registration for Kedro."""

from figregistry_kedro.hooks import FigRegistryHooks

# Register FigRegistry hooks for automated configuration management
HOOKS = (FigRegistryHooks(),)
'''
            
            if not dry_run:
                settings_path.parent.mkdir(parents=True, exist_ok=True)
                with open(settings_path, 'w') as f:
                    f.write(settings_content)
                logger.info(f"Created settings.py with FigRegistryHooks: {settings_path}")
                results["files_created"].append(str(settings_path))
            else:
                logger.info(f"Would create settings.py with FigRegistryHooks: {settings_path}")
            return
        
        # Update existing settings.py
        try:
            with open(settings_path, 'r') as f:
                content = f.read()
            
            # Check if FigRegistryHooks is already imported
            if 'FigRegistryHooks' in content:
                logger.info("FigRegistryHooks already configured in settings.py")
                return
            
            lines = content.split('\n')
            updated_lines = []
            hooks_added = False
            
            # Add import
            import_added = False
            for line in lines:
                if line.startswith('from ') or line.startswith('import '):
                    updated_lines.append(line)
                    if not import_added:
                        updated_lines.append('from figregistry_kedro.hooks import FigRegistryHooks')
                        import_added = True
                elif 'HOOKS' in line and '=' in line:
                    # Update HOOKS tuple
                    if 'FigRegistryHooks()' not in line:
                        if line.strip().endswith('()'):
                            # Replace single hook
                            updated_lines.append('HOOKS = (FigRegistryHooks(),)')
                        elif line.strip().endswith(',)'):
                            # Add to existing tuple
                            updated_lines.append(line.replace(',)', ', FigRegistryHooks(),)'))
                        else:
                            updated_lines.append(line)
                        hooks_added = True
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            
            # Add HOOKS if not found
            if not hooks_added:
                updated_lines.append('')
                updated_lines.append('# Register FigRegistry hooks for automated configuration management')
                updated_lines.append('HOOKS = (FigRegistryHooks(),)')
            
            if not dry_run:
                with open(settings_path, 'w') as f:
                    f.write('\n'.join(updated_lines))
                logger.info(f"Updated settings.py with FigRegistryHooks: {settings_path}")
                results["files_modified"].append(str(settings_path))
            else:
                logger.info(f"Would update settings.py with FigRegistryHooks: {settings_path}")
                
        except Exception as e:
            results["warnings"].append(f"Failed to update settings.py: {e}")
    
    def _generate_code_suggestions(self, results: Dict[str, Any]) -> None:
        """Generate code modification suggestions for pipeline nodes."""
        suggestions = []
        
        for call in self.analyzer.savefig_calls:
            suggestion = {
                "file": call.file_path,
                "line": call.line_number,
                "original_code": call.line_content,
                "suggested_change": f"Remove this line - figure will be saved via catalog as '{call.suggested_dataset_name}'",
                "function": call.function_name,
                "dataset_name": call.suggested_dataset_name,
                "additional_notes": []
            }
            
            # Add specific suggestions based on the call pattern
            if 'plt.show()' in call.line_content:
                suggestion["additional_notes"].append("Also remove plt.show() call if present")
            
            if any(arg in call.line_content for arg in ['dpi=', 'format=', 'bbox_inches=']):
                suggestion["additional_notes"].append("Consider moving arguments to save_args in catalog configuration")
            
            suggestions.append(suggestion)
        
        results["code_suggestions"] = suggestions
    
    def _create_migration_report(self, dry_run: bool, results: Dict[str, Any]) -> None:
        """Create comprehensive migration report."""
        report_path = self.project_path / "figregistry_migration_report.md"
        
        report_content = self._generate_report_content(results)
        
        if not dry_run:
            with open(report_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Created migration report: {report_path}")
            results["files_created"].append(str(report_path))
        else:
            logger.info(f"Would create migration report: {report_path}")
    
    def _generate_report_content(self, results: Dict[str, Any]) -> str:
        """Generate the content for the migration report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# FigRegistry-Kedro Migration Report

**Generated:** {timestamp}  
**Project:** {self.project_path.name}  
**Dry Run:** {results['dry_run']}  
**Status:** {results['status']}

## Summary

This report details the migration from manual matplotlib figure management to 
automated figregistry-kedro integration.

### Migration Statistics

- **plt.savefig() calls found:** {len(self.analyzer.savefig_calls)}
- **Catalog entries added:** {results.get('catalog_entries_added', 0)}
- **Files created:** {len(results.get('files_created', []))}
- **Files modified:** {len(results.get('files_modified', []))}
- **Warnings:** {len(results.get('warnings', []))}
- **Errors:** {len(results.get('errors', []))}

## Files Changed

### Created Files
"""

        for file_path in results.get('files_created', []):
            report += f"- `{file_path}`\n"

        report += "\n### Modified Files\n"
        for file_path in results.get('files_modified', []):
            report += f"- `{file_path}`\n"

        report += "\n## plt.savefig() Calls Found\n\n"
        
        for i, call in enumerate(self.analyzer.savefig_calls, 1):
            report += f"### {i}. {Path(call.file_path).name}:{call.line_number}\n\n"
            report += f"**Function:** `{call.function_name or 'Unknown'}`  \n"
            report += f"**Original Code:** `{call.line_content}`  \n"
            report += f"**Dataset Name:** `{call.suggested_dataset_name}`  \n"
            report += f"**Inferred Purpose:** `{call.inferred_purpose}`  \n"
            if call.filepath_arg:
                report += f"**Original Filepath:** `{call.filepath_arg}`  \n"
            report += "\n"

        if results.get('code_suggestions'):
            report += "\n## Code Modification Instructions\n\n"
            for suggestion in results['code_suggestions']:
                report += f"### {Path(suggestion['file']).name}:{suggestion['line']}\n\n"
                report += f"**Remove this line:**\n```python\n{suggestion['original_code']}\n```\n\n"
                report += f"**Reason:** {suggestion['suggested_change']}\n\n"
                if suggestion['additional_notes']:
                    report += "**Additional Notes:**\n"
                    for note in suggestion['additional_notes']:
                        report += f"- {note}\n"
                report += "\n"

        # Add warnings and errors
        if results.get('warnings'):
            report += "\n## Warnings\n\n"
            for warning in results['warnings']:
                report += f"- ⚠️ {warning}\n"

        if results.get('errors'):
            report += "\n## Errors\n\n"
            for error in results['errors']:
                report += f"- ❌ {error}\n"

        # Add next steps
        report += f"""
## Next Steps

1. **Install Dependencies**
   ```bash
   pip install figregistry-kedro>=0.1.0
   ```

2. **Update Pipeline Functions**
   - Remove all `plt.savefig()` calls identified above
   - Ensure functions return matplotlib Figure objects
   - Update function outputs in pipeline definitions to match catalog entries

3. **Test Migration**
   ```bash
   kedro run
   ```

4. **Validate Outputs**
   - Check that figures are saved in `data/08_reporting/`
   - Verify styling is applied correctly
   - Confirm all pipeline outputs are generated

5. **Clean Up** (if migration successful)
   - Remove backup directory: `{results.get('backup_path', 'N/A')}`
   - Remove this migration report
   - Commit changes to version control

## Configuration Reference

### FigureDataSet Catalog Entry Example
```yaml
example_plot:
  type: figregistry_kedro.FigureDataSet
  filepath: data/08_reporting/example_plot.png
  purpose: exploratory
  condition_param: experiment_condition
  style_params:
    figure.dpi: 150
    figure.facecolor: white
  save_args:
    bbox_inches: tight
    transparent: false
```

### FigRegistry Configuration (conf/base/figregistry.yml)
The migration created a template configuration with styles for:
- `exploratory`: Quick analysis plots
- `presentation`: Presentation-ready figures  
- `publication`: Publication-quality outputs

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `figregistry-kedro>=0.1.0` is installed
2. **Hook Not Found**: Verify `FigRegistryHooks` is in `settings.py`
3. **Figure Not Saved**: Check that pipeline function returns Figure object
4. **Style Not Applied**: Verify `condition_param` matches pipeline parameters

### Rollback Instructions

If migration causes issues:
1. Restore from backup: `{results.get('backup_path', 'N/A')}`
2. Or manually revert the files listed in "Files Changed" above

---

*This report was generated by the FigRegistry-Kedro migration script.*
"""

        return report
    
    def rollback(self) -> bool:
        """Rollback migration by restoring from backup.
        
        Returns:
            True if rollback successful, False otherwise
        """
        if not self.backup_path or not self.backup_path.exists():
            logger.error("No backup available for rollback")
            return False
        
        try:
            # Remove current project directory
            shutil.rmtree(self.project_path)
            
            # Restore from backup
            shutil.copytree(self.backup_path, self.project_path)
            
            logger.info(f"Successfully rolled back from backup: {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate Kedro projects from manual matplotlib to figregistry-kedro integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
            # Analyze project and show what would be changed
            python migration_script.py /path/to/kedro/project --dry-run
            
            # Execute migration with backup
            python migration_script.py /path/to/kedro/project --execute
            
            # Execute migration without backup (not recommended)
            python migration_script.py /path/to/kedro/project --execute --no-backup
        """)
    )
    
    parser.add_argument(
        "project_path",
        help="Path to the Kedro project to migrate"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making modifications"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true", 
        help="Execute the migration (creates backup by default)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (not recommended)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.dry_run and not args.execute:
        parser.error("Must specify either --dry-run or --execute")
    
    try:
        # Initialize analyzer
        analyzer = KedroProjectAnalyzer(args.project_path)
        
        # Perform analysis
        logger.info("Analyzing Kedro project...")
        analysis_results = analyzer.analyze_project()
        
        print(f"\n📊 Analysis Results:")
        print(f"   └─ Found {len(analysis_results['savefig_calls'])} plt.savefig() calls")
        print(f"   └─ Will generate {len(analysis_results['catalog_entries'])} catalog entries")
        print(f"   └─ Identified {len(analysis_results['migration_issues'])} potential issues")
        
        # Show issues if any
        for issue in analysis_results['migration_issues']:
            icon = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
            print(f"   {icon} {issue.category}: {issue.description}")
        
        # Check for blocking errors
        error_issues = [i for i in analysis_results['migration_issues'] if i.severity == "error"]
        if error_issues:
            print(f"\n❌ Cannot proceed - {len(error_issues)} blocking errors found")
            return 1
        
        if args.dry_run:
            print(f"\n🔍 Dry Run Results:")
            print(f"   └─ Would create figregistry.yml configuration")
            print(f"   └─ Would update catalog.yml with {len(analysis_results['catalog_entries'])} entries")
            print(f"   └─ Would update settings.py for hooks registration") 
            print(f"   └─ Would add figregistry-kedro dependency to pyproject.toml")
            print(f"   └─ Would generate migration report")
            
            print(f"\n📝 Suggested catalog entries:")
            for entry in analysis_results['catalog_entries']:
                print(f"   └─ {entry.name} ({entry.purpose}) -> {entry.filepath}")
        
        if args.execute:
            # Initialize migrator and execute
            migrator = FigRegistryKedroMigrator(analyzer)
            
            print(f"\n🚀 Executing migration...")
            migration_results = migrator.execute_migration(
                dry_run=False,
                create_backup=not args.no_backup
            )
            
            if migration_results["status"] == "success":
                print(f"✅ Migration completed successfully!")
                print(f"   └─ Backup created: {migration_results['backup_path']}")
                print(f"   └─ Files created: {len(migration_results['files_created'])}")
                print(f"   └─ Files modified: {len(migration_results['files_modified'])}")
                print(f"   └─ Catalog entries added: {migration_results['catalog_entries_added']}")
                
                if migration_results['warnings']:
                    print(f"   ⚠️ Warnings: {len(migration_results['warnings'])}")
                
                print(f"\n📋 Next steps:")
                print(f"   1. Install: pip install figregistry-kedro>=0.1.0")
                print(f"   2. Review migration report: figregistry_migration_report.md")
                print(f"   3. Update pipeline functions to remove plt.savefig() calls")
                print(f"   4. Test with: kedro run")
                
            else:
                print(f"❌ Migration failed!")
                for error in migration_results.get('errors', []):
                    print(f"   └─ {error}")
                return 1
        
        return 0
        
    except MigrationError as e:
        logger.error(f"Migration error: {e}")
        if e.details:
            logger.debug(f"Error details: {e.details}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())