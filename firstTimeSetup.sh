#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create a new virtual environment
echo "Creating new virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install the required libraries
if [ -f "requirements.txt" ]; then
    echo "Installing libraries from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

echo "Setup complete."