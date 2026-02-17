#!/usr/bin/env bash
# deploy.sh — Launch a federated learning simulation with Docker Compose.
#
# Usage:
#   ./scripts/deploy.sh              # Build and run 3-site simulation
#   ./scripts/deploy.sh --sites 5    # Build and run 5-site simulation
#   ./scripts/deploy.sh --down       # Stop the simulation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NUM_SITES=3
ACTION="up"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sites)
            NUM_SITES="$2"
            shift 2
            ;;
        --down)
            ACTION="down"
            shift
            ;;
        --help)
            echo "Usage: $0 [--sites N] [--down]"
            echo ""
            echo "Options:"
            echo "  --sites N   Number of federated client sites (default: 3)"
            echo "  --down      Stop the simulation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

if [[ "$ACTION" == "down" ]]; then
    echo "Stopping federated learning simulation..."
    docker compose down
    exit 0
fi

echo "============================================"
echo "  Physical AI Federated Oncology Trial"
echo "  Deploying with $NUM_SITES sites"
echo "============================================"

# Build the image
echo ""
echo "[1/3] Building Docker image..."
docker compose build

# Run the simulation
echo ""
echo "[2/3] Starting federated learning simulation..."
docker compose run --rm coordinator \
    python examples/run_federation.py \
    --num-sites "$NUM_SITES" \
    --rounds 10 \
    --local-epochs 5 \
    --dp-epsilon 2.0 \
    --secure-agg

echo ""
echo "[3/3] Simulation complete."
echo "============================================"
