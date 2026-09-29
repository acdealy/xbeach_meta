#!/bin/bash
# FALLBACK: serial, resumable runner for use inside a Jupyter/OnDemand session.
# (Prefer xbeach_array.sbatch. It has no 168 h limit and can run several cases at once.)
#
# Safe to stop and restart: finished cases (q_tot.json present) are skipped, so when a
# 7-day session ends, start a new session and launch this again. It carries on where it stopped.
#
# Start it so it survives closing the browser tab:
#   nohup ./run_cases.sh case_list.txt > run_cases.log 2>&1 &
#   tail -f run_cases.log
# Stop it:  pkill -f run_cases.sh; pkill mpirun      (Ctrl-Z only *pauses* a job, it doesn't stop it)
 
module load OpenMPI/4.1.6
 
LIST="${1:-case_list.txt}"
RUN_DIR="xbeach_cases"
SAVE_EVERY=10                     # rebuild the summary CSV this often (takes seconds)
export NP=40
export MPI_EXTRA="--oversubscribe"  # needed inside a Jupyter session (1 task x 40 cpus)
export PYTHON="${PYTHON:-python}"
export KEEP_NC=0                # uncomment to delete each xboutput.nc after q_tot.json is saved
 
[[ -f "$LIST" ]] || { echo "ERROR: case list '$LIST' not found (run the setup notebook first)"; exit 1; }
 
n=0
failed=()
# read the list on fd 3 so nothing inside the loop (e.g. mpirun) can consume it
while IFS= read -r case_dir <&3; do
    [[ -z "$case_dir" ]] && continue
    echo "========================================"
    bash ./run_case.sh "$case_dir" || failed+=("$case_dir")
    n=$((n + 1))
    if (( n % SAVE_EVERY == 0 )); then
        "$PYTHON" post_process.py --run_dir "$RUN_DIR" --out q_tot_summary_3d.csv
    fi
done 3< "$LIST"
 
echo "========================================"
"$PYTHON" post_process.py --run_dir "$RUN_DIR" --out q_tot_summary_3d.csv
if (( ${#failed[@]} )); then
    echo "${#failed[@]} case(s) failed this pass (re-running this script retries them):"
    printf '  %s\n' "${failed[@]}"
fi
echo "Done :P"