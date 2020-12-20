# Prepares the environment for executing a Jupyter lab server.
# This script is intended for development purposes, and should be called
# from the root of the project directory.

# Script directory. It corresponds to PROJECT_DIR/scripts.
$scriptDir = $PSScriptRoot;

# The project base directory.
$Env:PROJECT_DIR = "$scriptDir/../"

# If the data and media directories are located outside the project directory,
# we can set them through these environkment variables.
#$Env:PROJECT_DATA_DIR = "$scriptDir/../data";
#$Env:PROJECT_MEDIA_DIR = "$scriptDir/../media";

# Execute Jupyter lab.
jupyter lab --no-browser
