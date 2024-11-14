#!/bin/bash -l
# Standard output and error:
#SBATCH -o /raven/u/dvoss/al_pmssmwithgp/model/slurm/job-management/job_%j.out
#SBATCH -e /raven/u/dvoss/al_pmssmwithgp/model/slurm/job-management/job_%j.err
# Initial working directory:
#SBATCH -D /raven/u/dvoss/al_pmssmwithgp/model
#
# For CPU (uncomment)
# Number of nodes and MPI tasks per node:
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
#
# For GPU
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dvoss@mpcdf.mpg.de
#
#SBATCH --partition=gpudev


