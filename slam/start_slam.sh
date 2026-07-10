#!/bin/bash
# ============================================================
# D435 + RTAB-Map SLAM — Startup Script
# ============================================================
# Sources ROS Noetic, handles Conda Python path conflicts,
# and launches the D435 camera + RTAB-Map SLAM pipeline.
#
# Usage:
#   bash start_slam.sh                 # fresh map each run
#   bash start_slam.sh reset_db:=false # keep map across restarts
#   bash start_slam.sh rviz:=true      # also open RViz
#
# Controls:
#   - Move the camera slowly for best mapping
#   - Close rtabmapviz window to stop
#   - Press Ctrl-C in terminal to stop everything
#
# Map data saved to: ~/.ros/rtabmap.db
# To start fresh, delete the database: rm ~/.ros/rtabmap.db
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "  D435 + RTAB-Map RGB-D SLAM"
echo "  Workspace: $WORKSPACE_DIR"
echo "=============================================="

# --- Source ROS Noetic ---
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
else
    echo "[ERROR] ROS Noetic not found at /opt/ros/noetic/"
    exit 1
fi

# --- Handle Conda/ROS Python path conflicts ---
# ROS Noetic uses system Python 3.8. The Conda env 'yl39' has Python 3.9
# with its own site-packages. We prepend ROS Python paths so rospy, cv_bridge,
# etc. resolve correctly for any Python-based ROS nodes.
if [ -n "$CONDA_PREFIX" ]; then
    echo "[INFO] Conda env active ($CONDA_PREFIX), adjusting PYTHONPATH..."
    export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"
fi

# --- Launch ---
LAUNCH_FILE="$SCRIPT_DIR/launch/d435_rtabmap.launch"

if [ ! -f "$LAUNCH_FILE" ]; then
    echo "[ERROR] Launch file not found: $LAUNCH_FILE"
    exit 1
fi

echo "[INFO] Launching: $LAUNCH_FILE"
echo "[INFO] Arguments: $*"
echo ""
echo "  rtabmapviz will open showing the 3D map in real-time."
echo "  Move the camera slowly. Press Ctrl-C to stop."
echo ""

roslaunch "$LAUNCH_FILE" "$@"
