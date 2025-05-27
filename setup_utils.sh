#!/bin/bash
# setup_utils.sh - Utility functions and constants for FigRegistry setup scripts
# Following academic-biology software development guidelines (v2025-05-16)

# --- Colors for output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Helper Functions ---

# Function to print section headers
# Usage: section "My Section Title"
section() {
    if [ -z "$1" ]; then
        echo -e "\n${RED}Error: section() requires a title.${NC}" >&2
        return 1
    fi
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

# Function to handle and print errors, then exit
# Usage: error "Something went wrong"
error() {
    if [ -z "$1" ]; then
        echo -e "${RED}An unspecified error occurred.${NC}" >&2
    else
        echo -e "${RED}Error: $1${NC}" >&2
    fi
    exit 1
}

# Function to run a command with verbose output and error handling
# Usage: run_command_verbose my_command arg1 arg2
run_command_verbose() {
    if [ "$#" -eq 0 ]; then
        error "run_command_verbose: No command provided."
    fi
    
    # Special handling for pip commands to ensure we use the conda environment's pip
    if { [ "$1" = "pip" ] || [ "$1" = "pip3" ] || 
         { [ "$1" = "python" ] && [ "${2:-}" = "-m" ] && [ "${3:-}" = "pip" ]; }; }; then
        # If we're in a conda environment, use the full path to pip
        if [ -n "${CONDA_PREFIX:-}" ]; then
            PIP_PATH="${CONDA_PREFIX}/bin/pip"
            if [ -f "$PIP_PATH" ]; then
                # Replace pip/python -m pip with the full path
                if [ "$1" = "python" ]; then
                    shift 2  # Remove 'python -m pip'
                    set -- "$PIP_PATH" "$@"
                else
                    set -- "$PIP_PATH" "${@:2}"
                fi
            fi
        fi
    fi
    
    echo -e "${YELLOW}Running: $@${NC}"
    if ! "$@"; then
        error "Command failed: '$*'"
    fi
}

# Function to log messages with different levels
# Usage: log "info" "This is an informational message"
log() {
    local level="${1:-info}"
    local message="${2:-}"
    local timestamp
    timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        info)
            echo -e "[${timestamp}] [INFO] ${message}"
            ;;
        success)
            echo -e "[${timestamp}] [${GREEN}SUCCESS${NC}] ${message}"
            ;;
        warning)
            echo -e "[${timestamp}] [${YELLOW}WARNING${NC}] ${message}" >&2
            ;;
        error)
            echo -e "[${timestamp}] [${RED}ERROR${NC}] ${message}" >&2
            ;;
        *)
            echo -e "[${timestamp}] [${level}] ${message}"
            ;;
    esac
}

# Function to check if a command exists
# Usage: if command_exists conda; then ...
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# This makes the script safe to source and prevents it from executing
# commands if it's run directly, other than defining functions/variables.
return 0 2>/dev/null || exit 0
