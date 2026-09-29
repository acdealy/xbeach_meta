#!/bin/bash
# SLURM job array: each array task runs ONE case (line N+1 of the list file) as its own
# short job, so no job ever gets near a walltime limit and several cases can run at once.
#
# Submit from the project folder (login-node terminal, not inside Jupyter):
#   mkdir -p logs
#   python post_process.py --case_list case_list.txt --write_pending pending.txt
#   sbatch --array=0-$(( $(wc -l < pending.txt) - 1 ))%5 xbeach_array.sbatch pending.txt
#
# %5 = at most 5 cases running at the same time (5 x 40 cores). Change it later with:
#   scontrol update JobId=<array_job_id> ArrayTaskThrottle=8
#
#SBATCH --job-name=xbeach
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xb_%A_%a.out
##SBATCH --partition=common      # remove one '#' to pick a partition (e.g. your lab's) instead of the default

# ---- EDIT: python from your env. In a terminal where post_process.py works, run `which python`.
export PYTHON=/hpc/home/acd99/.conda/envs/bluemath2/bin/python

LIST="${1:?usage: sbatch --array=0-N%K xbeach_array.sbatch <case_list_file>}"
cd "$SLURM_SUBMIT_DIR" || exit 1

CASE_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")"
if [[ -z "$CASE_DIR" ]]; then
    echo "No line $((SLURM_ARRAY_TASK_ID + 1)) in $LIST"
    exit 1
fi

module load OpenMPI/4.1.6

# Two ways to request the 40 cores; this script handles both:
#   --ntasks=40 --cpus-per-task=1   40 MPI slots, MPI binds each rank to cores
#   --ntasks=1  --cpus-per-task=40  same layout as the Jupyter runs: 1 slot, so mpirun
#                                   needs --oversubscribe and leaves ranks unbound
# Command-line flags override the #SBATCH lines above, e.g.
#   sbatch --ntasks=1 --cpus-per-task=40 --array=0 xbeach_array.sbatch pending.txt
if (( ${SLURM_NTASKS:-1} > 1 )); then
    export NP="$SLURM_NTASKS"
    export MPI_EXTRA="--report-bindings"
else
    export NP="${SLURM_CPUS_PER_TASK:-40}"
    export MPI_EXTRA="--oversubscribe --report-bindings"
fi
# --report-bindings writes where each MPI rank runs to the top of xbeach_stdout.txt
export KEEP_NC=0   # uncomment to delete each multi-GB xboutput.nc after q_tot.json is saved

echo "array task $SLURM_ARRAY_TASK_ID on $(hostname): $CASE_DIR"
echo "layout: ntasks=${SLURM_NTASKS:-?} cpus-per-task=${SLURM_CPUS_PER_TASK:-?} -> mpirun -n $NP $MPI_EXTRA"
bash ./run_case.sh "$CASE_DIR"
