#!/bin/bash

module load OpenMPI/4.1.6

RUN_DIR="xbeach_cases"

COMMON_FILES=("bathy.dep" "x.grd" "y.grd" "params.txt")

shopt -s nullglob

for d in "$RUN_DIR"/run_tp_*_ang_*/; do
    echo "========================================"
    echo "Processing $d"
    
    for file in "${COMMON_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$d"
        else
            echo "ERROR: Required file '$file' not found in the base directory"
            exit 1
        fi
    done
    
    cd "$d" || exit
    
    mpirun -n 40 --oversubscribe /hpc/home/acd99/xbeach/src/xbeach/xbeach

    cd - > /dev/null
done

echo "========================================"
echo "Done :P"