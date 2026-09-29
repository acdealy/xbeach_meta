#!/bin/bash
# Runs ONE XBeach case start to finish. A case counts as finished once its folder has a
# q_tot.json computed with the current averaging window (T_AVG_START in post_process.py),
# so this script is safe to re-run on any case at any time:
#   - finished case                      -> skipped in seconds
#   - case another live job is running   -> skipped (lock file)
#   - complete xboutput.nc, no up-to-date q_tot.json -> only post-processed
#   - missing / partial xboutput.nc      -> XBeach is (re)run, then post-processed
#
# Usage:  bash run_case.sh <case_folder>        (run from the project folder)
# Called by xbeach_array.sbatch (SLURM) or run_cases.sh (Jupyter fallback).
#
# Settings (environment variables):
#   NP          MPI ranks                          (default 40)
#   MPI_EXTRA   extra mpirun flags                 (default --oversubscribe; sbatch sets it empty)
#   PYTHON      python with xarray + dask          (default: python)
#   XBEACH_BIN  XBeach executable
#   KEEP_NC     1 = keep xboutput.nc (default); 0 = delete it once q_tot.json and the
#               small xb_timemean.nc have been written (saves ~8-15 GB per case)
# XBeach's *.bcf boundary files (~400 MB per case) are always deleted once a case is done.
 
set -o pipefail
 
NP="${NP:-40}"
MPI_EXTRA="${MPI_EXTRA---oversubscribe}"
PYTHON="${PYTHON:-python}"
XBEACH_BIN="${XBEACH_BIN:-/hpc/home/acd99/xbeach/src/xbeach/xbeach}"
KEEP_NC="${KEEP_NC:-1}"
COMMON_FILES=("bathy.dep" "x.grd" "y.grd" "params.txt")
 
PROJECT_DIR="$PWD"
CASE_DIR="${1:?usage: bash run_case.sh <case_folder>}"
name="$(basename "$CASE_DIR")"
log() { echo "[$(date '+%F %T')] $name: $*"; }
 
[[ -d "$CASE_DIR" ]] || { log "ERROR: folder not found"; exit 1; }
CASE_DIR="$(cd "$CASE_DIR" && pwd)"
[[ -f "$CASE_DIR/jonswap.txt" ]] || { log "ERROR: jonswap.txt missing"; exit 1; }
 
# --- 1. python must work: it's needed to check results and to post-process ----------
#        (checked before spending ~40 min on XBeach)
if ! "$PYTHON" -c "import xarray, dask" 2>/dev/null; then
    log "ERROR: '$PYTHON' cannot import xarray/dask; set PYTHON to your env's python"
    exit 1
fi
 
# --- already finished? (q_tot.json exists AND used the current averaging window) ---------
if "$PYTHON" "$PROJECT_DIR/post_process.py" --is_done "$CASE_DIR"; then
    log "already done (up-to-date q_tot.json present), skipping"
    exit 0
fi
[[ -f "$CASE_DIR/q_tot.json" ]] && log "q_tot.json is from a different averaging window, redoing"
 
# --- 2. lock, so two jobs never run the same case --------------------------------
LOCK="$CASE_DIR/.running"
ME="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
ME="${ME:-${SLURM_JOB_ID:-local-$$}}"
if [[ -f "$LOCK" ]]; then
    owner="$(cat "$LOCK" 2>/dev/null)"
    if [[ -n "$owner" && "$owner" != "$ME" ]] && squeue -h -j "$owner" 2>/dev/null | grep -q .; then
        log "being run by live job $owner, skipping"
        exit 0
    fi
    log "removing stale lock left by $owner (that job is gone)"
fi
echo "$ME" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
 
POST_ARGS=()
[[ "$KEEP_NC" == "0" ]] && POST_ARGS+=(--delete_nc)
post_process() {
    "$PYTHON" "$PROJECT_DIR/post_process.py" --case "$CASE_DIR" ${POST_ARGS[@]+"${POST_ARGS[@]}"}
}
# XBeach's boundary-condition files (nh_reuse.bcf is ~400 MB per case) are regenerated on
# every run and aren't used by post-processing, so remove them once a case is finished
clean_bcf() { rm -f "$CASE_DIR"/*.bcf; }
 
cd "$CASE_DIR" || exit 1
 
# --- 3. output from an earlier run? just post-process it ---------------------------
if [[ -f xboutput.nc ]]; then
    log "found existing xboutput.nc, checking it"
    post_process
    rc=$?
    if (( rc == 0 )); then
        clean_bcf
        log "existing output was complete, done"
        exit 0
    elif (( rc != 3 )); then   # 3 = output truncated/unreadable; anything else is a script problem
        log "ERROR: post-processing failed on existing output (kept it); see q_tot_error.json"
        exit 1
    fi
    log "existing output incomplete or unreadable, re-running XBeach"
    rm -f xboutput.nc
fi
 
# --- 4. run XBeach --------------------------------------------------------------
for f in "${COMMON_FILES[@]}"; do
    if [[ ! -f "$PROJECT_DIR/$f" ]]; then
        log "ERROR: required file '$f' not found in $PROJECT_DIR"
        exit 1
    fi
    cp "$PROJECT_DIR/$f" .
done
 
log "starting XBeach on $NP MPI ranks (host $(hostname))"
start=$SECONDS
# stdin from /dev/null so mpirun can't swallow the case list in run_cases.sh
mpirun -n "$NP" $MPI_EXTRA "$XBEACH_BIN" < /dev/null > xbeach_stdout.txt 2>&1
rc=$?
log "XBeach exited with code $rc after $(( (SECONDS - start) / 60 )) min"
if (( rc != 0 )); then
    log "ERROR: XBeach failed; see xbeach_stdout.txt and XBeach's log files in $CASE_DIR"
    exit 1
fi
 
# --- 5. post-process right away -> q_tot.json (+ xb_timemean.nc) ---------------------
if ! post_process; then
    log "ERROR: post-processing failed; see $CASE_DIR/q_tot_error.json"
    exit 1
fi
clean_bcf
log "done"
 





