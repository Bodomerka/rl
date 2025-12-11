#!/bin/bash
# Vast.ai Instance Setup Script
# Run this after connecting to your instance

set -e

echo "========================================"
echo "Setting up Cooperation Collapse Project"
echo "========================================"

# Update system
apt-get update && apt-get install -y git tmux htop nvtop

# Install Python dependencies
pip install --upgrade pip
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax numpy matplotlib

# Verify JAX GPU
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Clone project (replace with your repo)
# git clone https://github.com/YOUR_USERNAME/cooperation-collapse.git
# cd cooperation-collapse

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Upload or clone your project"
echo "2. Run training with: python scripts/train_cooperative.py"
echo "3. Use tmux for long-running jobs: tmux new -s training"
echo ""
