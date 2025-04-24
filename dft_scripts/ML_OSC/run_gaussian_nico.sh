#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --gres=gpu:4           # Number of GPU(s) per node
#SBATCH --gres=gpu:p100:4      # Type of GPU(s) per node
#SBATCH --cpus-per-task=24     # CPU cores/threads
#SBATCH --output=slurm-%j.out     # Output file for the job
#SBATCH --error=slurm-%j.err     # Output file for the job
#SBATCH --mem=90GB              # Memory per node
#SBATCH --time=00-00:45         # Time (DD-HH:MM)

# Check if an ID was provided
if [ $# -eq 0 ]; then
    echo "Please provide an ID as an argument."
    exit 1
fi

# Set the ID
JOB_ID=$1

# Load the Gaussian module
module load gaussian/g16.b01

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Gaussian with specified input file
g16 < ${JOB_ID}.com >& ${JOB_ID}.log

