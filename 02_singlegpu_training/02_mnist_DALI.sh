#!/bin/bash
#SBATCH --job-name=mnist_DALI    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Specify GPU partition for GPU Nodes
#SBATCH --reservation=SCA        # Reservation incase of urgent requirement
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=logs_%j.out     # output logfile name
#SBATCH --error=logs_%j.err      # error logfile name
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# which gpu node was used
echo "Running on host of RUDRA" $(hostname)

module purge #remove unneccesary loaded modules

#load the module
module load miniconda

#activate the environment
conda activate tutorial

time python 02_mnist_DALI.py --epochs=10
