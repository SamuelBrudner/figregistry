#!/bin/bash
# setup_env.sh - Script to set up and manage conda environments for FigRegistry
# Follows academic-biology software development guidelines (v2025-05-16)

# Exit on error, unset variables, and pipe failures
set -euo pipefail

# Default values
RUN_TESTS=true
DEVELOPMENT_MODE=true
ENV_NAME_PREFIX="figregistry"
ENV_PATH="${PWD}/.venv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="environment.yml"  # Default to production environment

# Source utility functions
UTILS_SCRIPT="${SCRIPT_DIR}/setup_utils.sh"
if [ ! -f "$UTILS_SCRIPT" ]; then
    echo "Error: Could not find utility script at $UTILS_SCRIPT" >&2
    exit 1
fi
# shellcheck source=./setup_utils.sh
source "$UTILS_SCRIPT" || {
    echo "Error: Failed to source utility functions from $UTILS_SCRIPT" >&2
    exit 1
}

# Initialize conda
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    log "error" "Could not initialize conda. Make sure conda is installed and available in your PATH."
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --prod|--production)
            DEVELOPMENT_MODE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-tests     Skip running the test suite after setup"
            echo "  --prod         Set up production environment (default: development)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set environment-specific variables
if [ "$DEVELOPMENT_MODE" = true ]; then
    ENV_FILE="environment-dev.yml"
    ENV_NAME="${ENV_NAME_PREFIX}-dev"
    ENV_TYPE="development"
else
    ENV_FILE="environment.yml"
    ENV_NAME="${ENV_NAME_PREFIX}"
    ENV_TYPE="production"
fi

# Print a section header for the script start
# Capitalize first letter of ENV_TYPE for display
ENV_TYPE_DISPLAY=$(echo "${ENV_TYPE}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')

section "FigRegistry ${ENV_TYPE_DISPLAY} Environment Setup"

log "info" "Starting FigRegistry ${ENV_TYPE} environment setup in ${SCRIPT_DIR}"
log "info" "Using environment file: ${ENV_FILE}"

# --- Check for Conda ---
section "Checking for Conda Installation"

# Check if we're running in a container
if [ -f "/.dockerenv" ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    log "info" "Running inside a Docker container"
    
    # In Docker, conda might be installed but not in PATH
    if ! command_exists conda; then
        # Try to source conda.sh from common locations
        for conda_path in \
            "/opt/conda/etc/profile.d/conda.sh" \
            "/usr/local/miniconda/etc/profile.d/conda.sh"
        do
            if [ -f "$conda_path" ]; then
                log "info" "Sourcing conda from $conda_path"
                # shellcheck source=/dev/null
                source "$conda_path"
                break
            fi
        done
        
        # If conda still not found, install Miniconda
        if ! command_exists conda; then
            log "warning" "Conda not found. Installing Miniconda..."
            
            # Install prerequisites
            if command -v apt-get >/dev/null 2>&1; then
                log "info" "Installing prerequisites using apt-get..."
                apt-get update && apt-get install -y --no-install-recommends \
                    wget \
                    bzip2 \
                    ca-certificates \
                    libglib2.0-0 \
                    libxext6 \
                    libsm6 \
                    libxrender1 \
                    git \
                    procps \
                    && rm -rf /var/lib/apt/lists/*
            elif command -v yum >/dev/null 2>&1; then
                log "info" "Installing prerequisites using yum..."
                yum install -y \
                    wget \
                    bzip2 \
                    ca-certificates \
                    glibc \
                    libXext \
                    libSM \
                    libXrender \
                    git \
                    procps-ng \
                    && yum clean all
            else
                error "Cannot install Miniconda: Unsupported package manager (neither apt-get nor yum found)"
            fi
            
            # Install Miniconda
            MINICONDA_INSTALLER="/tmp/miniconda.sh"
            MINICONDA_PATH="/usr/local/miniconda"
            
            log "info" "Downloading Miniconda3..."
            wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER"
            
            log "info" "Installing Miniconda3 to $MINICONDA_PATH..."
            bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_PATH"
            rm "$MINICONDA_INSTALLER"
            
            # Initialize conda
            export PATH="$MINICONDA_PATH/bin:$PATH"
            # shellcheck source=/dev/null
            source "$MINICONDA_PATH/etc/profile.d/conda.sh"
            
            # Verify installation
            if ! command_exists conda; then
                error "Failed to install Miniconda. Please install it manually."
            fi
            
            log "success" "Miniconda installed successfully"
        fi
    fi
else
    # Standard non-Docker environment
    if ! command_exists conda; then
        error "Conda not found in PATH. Please install Miniconda or Anaconda first.\nDownload Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    fi
fi

# Initialize conda for this shell
log "info" "Initializing conda..."

# Try standard conda initialization
if ! eval "$(conda shell.bash hook 2> /dev/null)" >/dev/null 2>&1; then
    log "warning" "Failed to initialize conda with shell.bash hook"
    
    # Try alternative initialization methods
    local conda_sh=""
    for prefix in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda" "/usr/local/miniconda" "/usr/local/aniconda3" "/opt/conda"; do
        if [ -f "${prefix}/etc/profile.d/conda.sh" ]; then
            conda_sh="${prefix}/etc/profile.d/conda.sh"
            break
        fi
    done
    
    if [ -n "$conda_sh" ]; then
        log "info" "Sourcing conda.sh from ${conda_sh}"
        # shellcheck source=/dev/null
        if ! source "$conda_sh" 2>/dev/null; then
            log "warning" "Failed to source $conda_sh"
        fi
    fi
    
    # Final check if conda is available
    if ! command -v conda >/dev/null 2>&1; then
        error "Could not initialize conda. Please ensure Conda is properly installed and initialized."
    fi
fi

# Ensure conda is in PATH
export PATH="${CONDA_PREFIX:-$HOME/miniconda3}/bin:$PATH"

# --- Environment Setup ---
section "Setting up Conda Environment"

# Check if environment file exists
if [ ! -f "${ENV_FILE}" ]; then
    error "Environment file '${ENV_FILE}' not found. Please run this script from the project root."
fi

# Check if environment already exists
section "Setting up Conda Environment"

if conda env list | grep -q "^${ENV_NAME}\s"; then
    log "info" "Updating existing ${ENV_TYPE} environment '${ENV_NAME}'"
    run_command_verbose conda env update -f "${ENV_FILE}" -n ${ENV_NAME} --prune
else
    # Create the conda environment without the pip section
    log "info" "Creating new development environment '$ENV_NAME'"
    if ! conda env create -f "$ENV_FILE" -n "$ENV_NAME"; then
        error "Failed to create conda environment. See above for details."
    fi

    # Activate the environment
    log "info" "Activating environment '$ENV_NAME'"
    if ! conda activate "$ENV_NAME"; then
        error "Failed to activate conda environment. Try running 'conda init <your-shell>' and restarting your shell."
    fi

    # Install the package in development mode
    log "info" "Installing package in development mode..."
    # Activate conda environment to ensure pip is available
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    
    # Use the full path to pip to ensure we're using the correct one
    PIP_PATH="$(conda run -n "${ENV_NAME}" which pip)"
    if [ -z "${PIP_PATH}" ]; then
        log "error" "Could not find pip in the conda environment"
        exit 1
    fi
    
    run_command "${PIP_PATH} install -e ."
    if [ $? -eq 0 ]; then
        log "success" "Package installed in development mode"
    else
        log "error" "Failed to install package in development mode"
        exit 1
    fi
fi

# --- Install Package ---
section "Installing FigRegistry in Development Mode"

# Install development tools if in development mode
if [ "$DEVELOPMENT_MODE" = true ]; then
    log "info" "Development tools are already included in the environment."
fi
run_command_verbose pip install -e "${SCRIPT_DIR}"

# --- Setup Development Tools ---
section "Setting Up Development Tools"

# Install pre-commit hooks
log "info" "Setting up pre-commit hooks"
if command -v pre-commit &> /dev/null; then
    run_command_verbose pre-commit install
else
    log "warning" "pre-commit not found. Installing..."
    run_command_verbose pip install pre-commit
    run_command_verbose pre-commit install
fi

# --- Run Tests ---
if [ "$RUN_TESTS" = true ]; then
    section "Running Tests"
    if command -v pytest &> /dev/null; then
        run_command_verbose pytest -v --cov=figregistry --cov-report=term-missing
    else
        log "warning" "pytest not found. Skipping tests."
    fi
fi

# --- Completion ---
section "Setup Complete!"
log "success" "FigRegistry ${ENV_TYPE} environment is ready!"

echo -e "\n${GREEN}Activation Instructions:${NC}"
echo -e "${YELLOW}To activate the ${ENV_TYPE} environment, run:${NC}"
echo "  conda activate ${ENV_NAME}"

echo -e "${YELLOW}Or add this to your shell profile (e.g., .bashrc, .zshrc) for easier activation:${NC}"
echo "  alias activate_${ENV_NAME}='conda activate ${ENV_NAME}'"

if [ "$DEVELOPMENT_MODE" = true ]; then
    echo -e "\n${YELLOW}Development tools installed:${NC}"
    echo "- Testing: pytest, pytest-cov, pytest-mock, hypothesis"
    echo "- Code Quality: black, isort, mypy, ruff, pre-commit"
    echo "- Documentation: mkdocs, jupyter, ipython"
    echo "- Optional: seaborn, plotly, jupyterlab"
    echo -e "\n${YELLOW}To run tests:${NC}"
    echo "  pytest -v --cov=figregistry"
fi

echo -e "\n${GREEN}Happy coding! 🚀${NC}"
