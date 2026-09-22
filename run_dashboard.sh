#!/bin/bash
# CropGuardAI Training Dashboard Launcher (Linux/macOS)
# Usage: ./run_dashboard.sh

set -e

cd "$(dirname "$0")"

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing streamlit..."
    pip install streamlit --quiet
fi

echo
echo "=========================================="
echo "  CropGuardAI Training Dashboard"
echo "=========================================="
echo
echo "Starting Streamlit server..."
echo "Open http://localhost:8501 in your browser"
echo
echo "Press Ctrl+C to stop"
echo

streamlit run scripts/training_dashboard.py --server.port 8501 --server.headless true