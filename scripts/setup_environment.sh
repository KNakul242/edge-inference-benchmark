#!/usr/bin/env bash
# Install pinned dependencies. Run once per environment after cloning.
set -euo pipefail

PLATFORM=$(uname -s)

echo "Platform: $PLATFORM"
echo "Python: $(python --version)"

pip install --upgrade pip

if [[ "$PLATFORM" == "Darwin" ]]; then
    echo "Installing Mac M5 dependencies (includes coremltools)..."
    pip install -r requirements.txt
    pip install coremltools==7.2
else
    echo "Installing Linux dependencies (CoreML EP not available)..."
    pip install -r requirements.txt
fi

echo "Verifying key versions..."
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import onnxruntime; print(f'onnxruntime: {onnxruntime.__version__}')"
python -c "import ultralytics; print(f'ultralytics: {ultralytics.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

echo "Setup complete."
