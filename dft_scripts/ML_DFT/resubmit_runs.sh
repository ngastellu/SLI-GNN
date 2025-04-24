#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --gres=gpu:4           # Number of GPU(s) per node
#SBATCH --gres=gpu:p100:4      # Type of GPU(s) per node
#SBATCH --cpus-per-task=24     # CPU cores/threads
#SBATCH --array=0-21
#SBATCH --output=reslurm_%j-%a.out     # Output file for the job
#SBATCH --error=reslurm_%j-%a.err     # Output file for the job
#SBATCH --mem=90GB              # Memory per node
#SBATCH --time=00-03:00         # Time (DD-HH:MM)


# Parse input
if [[ $# == 1 ]]; then
	rundir="C${1}"
elif [[ $# == 2 ]]; then
	rundir="C${1}/C${1}_C${2}"
else
    echo "Please provide directory index (or indices) as an argument."
	echo 'If a single argument X is given; this script will consider directory C$X/'
	echo 'If two arguments X and Y are given; this script will consider directory C$X/C$X_C$Y/'
    exit 1
fi
	

njobs=$((1 + $SLURM_ARRAY_TASK_MAX - $SLURM_ARRAY_TASK_MIN))

fail_file="${rundir}/failed_runs.txt"

nfail=$(wc -l < "$fail_file") # use '<' to avoid capturing the file name
echo "Working on ${rundir}/"
echo "Found $nfail failed runs."

if [[ $njobs != $nfail ]] ; then
	nmissing=$(($nfail - $njobs))
	echo '!!! # of jobs =/= # of failed runs !!!'
	echo "njobs = $njobs ; nfail = $nfail"
	echo "Suggestion: once these jobs finish running; delete the first $njobs lines from ${fail_file} and resubmit using this script with $nmissing jobs in the array."
	echo '!!!!!!!!!!!!!!!!!!!!'
fi

# Get molecule ID from failed run
run_line=$(head -n $SLURM_ARRAY_TASK_ID "$fail_file" | tail -n 1) # line containing name of failed run
read -ra split_line <<< "$run_line" # split line using white space
job_ID=${rundir}/${split_line[0]}


echo "Working on file ${JOB_ID}.com"

# Load the Gaussian module
module load gaussian/g16.b01

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Gaussian with specified input file
g16 < ${JOB_ID}.com >& ${JOB_ID}.log
