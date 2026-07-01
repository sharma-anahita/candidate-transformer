#!/bin/bash
set -e

echo "Installing dependencies..."
if command -v pip3 &>/dev/null; then
    pip3 install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo ""
echo "Starting app..."
streamlit run streamlit_app.py
