#!/usr/bin/env bash

#BSUB -q q_expl
#BSUB -n 4
#BSUB -m mn461
#BSUB -R "affinity[core(1):cpubind=numa:membind=localprefer:distribute=pack]"
#BSUB -oo conda-env-update_output-job#%J.txt
#BSUB -eo conda-env-update_error-job#%J.txt

# Update all packages in the isingchat conda environment.

# Load system modules
module load SO/ambiente-7
#module load miniconda/4.10.3

# NOTE: We can retrieve the number of processors (slots) allocated to the job through the environment variable LSB_DJOB_NUMPROC.
# See https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=variables-environment-variable-reference.

# We have to source our ~/.bashrc file since conda activate command
# relies on some instructions written in this file.
source ~/.bashrc

echo "================================"
echo "Show conda information"
conda info -a
echo "--------------------------------"

echo "================================"
echo "System PATH"
echo "$PATH"
echo "--------------------------------"

echo "================================"
echo "Activate conda environment"
conda activate isingchat
conda info -e
echo "--------------------------------"

echo "================================"
echo "Update isingchat"
#conda update --dry-run -c intel -c conda-forge -c oarodriguez isingchat --no-deps
conda update -c intel -c conda-forge -c oarodriguez isingchat --yes --no-deps
echo "--------------------------------"

echo "All tasks finished"
