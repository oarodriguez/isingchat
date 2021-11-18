#!/usr/bin/env bash

#BSUB -q q_expl
#BSUB -n 1
#BSUB -m mn461
#BSUB -R "affinity[core(1):cpubind=numa:membind=localprefer:distribute=pack]"
#BSUB -oo conda-env-info_output-job#%J.txt
#BSUB -eo conda-env-info_error-job#%J.txt

# Display information of the current conda environment.

# Load system modules
module load SO/ambiente-7
#module load miniconda/4.10.3

# NOTE: We can retrieve the number of processors (slots) allocated to the job through the environment variable LSB_DJOB_NUMPROC.
# See https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=variables-environment-variable-reference.

# We have to source our ~/.bashrc file since conda activate command
# relies on some instructions written in this file.
source ~/.bashrc

echo "================================"
echo "System PATH"
echo "$PATH"
echo "--------------------------------"

echo "================================"
echo "Show conda information"
conda info -a
echo "--------------------------------"

echo "================================"
echo "Conda build information"
conda build --help
echo "--------------------------------"

echo "All tasks finished"
