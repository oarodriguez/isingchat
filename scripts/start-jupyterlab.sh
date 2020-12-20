#!/bin/bash
# Prepares the environment for executing a Jupyter lab server.
# This script is intended for development purposes, and should be called
# from the root of the project directory.

# Script directory. It corresponds to PROJECT_DIR/scripts.
# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# The project base directory.
export PROJECT_DIR="$script_dir/../"

# If the data and media directories are located outside the project directory,
# we can set them through these environment variables.
#export PROJECT_DATA_DIR="$script_dir/../data";
#export PROJECT_MEDIA_DIR="$script_dir/../media";

# Execute Jupyter lab.
jupyter lab --no-browser
