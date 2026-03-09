#!/bin/bash
# Installation script for KeyChain data synthesis pipeline

set -e  # Exit on error

echo "Installing KeyChain dependencies..."

# Install core Python packages
pip install transformers tenacity openai azure-identity tqdm gdown datasets

# Install optional dependencies
pip install tiktoken nanoid

# Install system dependencies
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    sudo apt-get update -y
    sudo apt-get install -y unzip wget
elif command -v yum &> /dev/null; then
    # RHEL/CentOS
    sudo yum install -y unzip wget
elif command -v brew &> /dev/null; then
    # macOS
    brew install wget
else
    echo "Warning: Could not detect package manager. Please install unzip and wget manually."
fi

echo "✓ Installation complete!"
echo ""
echo "Optional dependencies for advanced features:"
echo "  - vllm: For question filtering stage (requires GPU)"
echo "  - pandas: For secondary quality control"
echo ""
echo "To download datasets, run:"
echo "  bash scripts/synth.sh"
