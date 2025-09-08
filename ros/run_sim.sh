#!/usr/bin/env bash
# Usage: ./ros/run_sim.sh /abs/path/to/simple6r.urdf
set -euo pipefail
URDF="${1:-simple6r.urdf}"
# You may want to 'source /opt/ros/noetic/setup.bash' and your venv before calling this script.
python "$(dirname "$0")/sim_node.py" --urdf "$URDF" --dt 0.004 --kp 18,12,14,9,7,5
