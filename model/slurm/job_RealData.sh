#!/bin/bash

# python /raven/u/dvoss/al_pmssmwithgp/model/GPmodelRealData.py

## Set the paths and variables
GP_MODEL_SCRIPT="/u/dvoss/al_pmssmwithgp/model/GPmodelRealData.py"
ITERATIONS=1


for i in $(seq 1 ${ITERATIONS}); do
    echo "Starting iteration ${i}"

    OUTPUT_DIR="/raven/u/dvoss/al_pmssmwithgp/model/plots/Iter${i}"
    mkdir -p ${OUTPUT_DIR}

    # Step 1: Train GP Model and get new points
    GP_COMMAND=("python" "${GP_MODEL_SCRIPT}" "--iteration=${i}" "--output_dir=${OUTPUT_DIR}")
    "${GP_COMMAND[@]}"
    if [ $? -ne 0 ]; then
        echo "Error during GP model script execution"
        exit 1
    fi

done

echo "All iterations completed."