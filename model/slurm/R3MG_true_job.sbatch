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
# #SBATCH --mem=100000
#
# For GPU
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=125000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dvoss@mpcdf.mpg.de

# Load compiler and MPI modules (must be the same as used in conda env)
module purge
module load anaconda/3/2021.11 # <-> python 3.9.
module load cuda/11.6
module load cudnn/8.8
module load pytorch/gpu-cuda-11.6/2.0.0
module load gpytorch/gpu-cuda-11.6/pytorch-1.11.0/1.8.0
pip install uproot
pip install argparse
pip install pyslha
pip install structlog
pip install imageio[ffmpeg] Pillow
pip install seaborn

# Start run
srun /raven/u/dvoss/al_pmssmwithgp/model/slurm/job_Run3ModelGen_true.sh

