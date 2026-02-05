#!/bin/bash
set -e

echo "======================================"
echo "MRD Repository Setup Script"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "This doesn't appear to be a git repository."
    print_warning "Please clone the repository first: git clone <repo-url>"
    exit 1
fi

# Step 1: Update git submodules
print_step "Updating git submodules..."
if git submodule update --init --recursive; then
    print_success "Submodules updated successfully"
else
    print_error "Failed to update submodules"
    exit 1
fi

# Step 2: Detect GPU and set up .envrc
print_step "Detecting GPU and configuring environment..."

VARIANT="llvm_ad_rgb"  # Default to CPU variant

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected!"
        VARIANT="cuda_ad_rgb"
    else
        print_warning "nvidia-smi found but no GPU detected. Using CPU variant."
    fi
else
    print_warning "No NVIDIA GPU detected. Using CPU variant (llvm_ad_rgb)."
fi

# Update or create .envrc file
print_step "Configuring .envrc with variant: $VARIANT"
cat > .envrc << EOF
export MODELVSHUMANDIR=./vendor/model-vs-human/
export DEFAULT_MI_VARIANT='$VARIANT'
EOF

print_success ".envrc configured with DEFAULT_MI_VARIANT='$VARIANT'"

# Step 3: Extract assets archive
print_step "Extracting assets archive..."

if [ ! -f "assets.tar.gz" ]; then
    print_error "assets.tar.gz not found in the repository."
    exit 1
fi

if [ -d "assets" ]; then
    print_warning "assets directory already exists. Skipping extraction."
else
    if tar -xzf assets.tar.gz; then
        print_success "Assets extracted successfully"
    else
        print_error "Failed to extract assets.tar.gz"
        exit 1
    fi
fi

# Step 4: Set up Python environment
print_step "Setting up Python environment..."

if command -v uv &> /dev/null; then
    print_step "Installing dependencies with uv..."
    if uv sync; then
        print_success "Dependencies installed successfully with uv"
    else
        print_error "Failed to install dependencies with uv"
        exit 1
    fi
else
    print_warning "uv not found. Skipping Python environment setup."
    echo "You can install dependencies manually with:"
    echo "  - Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  - Then run: uv sync"
    echo "  - Or use pip: pip install -e ."
fi

# Step 5: Source environment file
print_step "Environment setup complete!"
echo ""
echo "======================================"
print_success "Setup completed successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Source the environment file and/or install package as editable:"
echo "     source .envrc"
echo '     (uv) pip install -e ".[dev]"'
echo ""
echo "  (Alternatively, if using direnv):"
echo "     direnv allow"
echo ""
echo "  2. Activate the python environment."
echo ""
echo "  3. Run an experiment (PYTHONPATH is only required if the package wasn't installed):"
echo "     PYTHONPATH=. python src/main.py dragon hallstatt lpips --epochs 50"
echo ""
echo "Configuration:"
echo "  - GPU Variant: $VARIANT"
echo "  - Assets: Extracted"
echo "  - Submodules: Updated"
echo ""
