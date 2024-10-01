#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to load the necessary modules for Python 3.10 on Snellius
load_modules() {
    echo "Loading necessary modules..."
    module purge
    module load 2022
    module load Anaconda3/2022.05
}

# Function to create the Python virtual environment with Python 3.10
create_virtualenv() {
    ENV_NAME=$1
    echo "Creating virtual environment: $ENV_NAME with Python 3.10"
    
    # Ensure Python 3.10 is available
    python --version | grep "3.10" >/dev/null 2>&1
    if [[ $? -ne 0 ]]; then
        echo "Python 3.10 not found. Installing Python 3.10."
        load_modules
        conda create -n py310 python=3.10 -y
        source activate py310
    else
        echo "Python 3.10 already available."
    fi
    
    # Remove the environment if it already exists
    if [[ -d "$ENV_NAME" ]]; then
        echo "The environment $ENV_NAME already exists."
        echo "Do you want to overwrite it? (y/n): "
        read -r answer
        if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
            echo "Overwriting and removing the existing environment $ENV_NAME."
            rm -rf "$ENV_NAME"
        else
            echo "Aborting the setup. No changes made."
            exit 1
        fi
    fi

    # Create new virtual environment
    python -m venv "$ENV_NAME"
    source "$ENV_NAME/bin/activate"

    # Verify Python version inside virtual environment
    python_version=$(python --version)
    echo "Virtual environment $ENV_NAME is using: $python_version"

    # Install the requirements.txt if available
    if [[ -f "requirements.txt" ]]; then
        echo "Installing dependencies from requirements.txt"
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "No requirements.txt found. Skipping package installation."
    fi
}

# Main script execution
ENV_NAME="ai4mi"

# Check if Python 3.10 is already installed
python --version 2>/dev/null | grep "3.10" >/dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Python 3.10 is not installed or not active. Setting it up..."
    load_modules
fi

# Create the virtual environment
create_virtualenv "$ENV_NAME"

# Provide usage feedback
echo "To activate the environment, use:"
echo "source $ENV_NAME/bin/activate"
