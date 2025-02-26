#!/bin/bash

# python /raven/u/dvoss/al_pmssmwithgp/model/TwoCirclesnD.py

# Set the paths and variables
GP_MODEL_SCRIPT="/raven/u/dvoss/al_pmssmwithgp/model/TwoCirclesnD.py"
ITERATIONS=100


for i in $(seq 1 ${ITERATIONS}); do
    echo "Starting iteration ${i}"

    OUTPUT_DIR="/raven/u/dvoss/al_pmssmwithgp/model/plots/plots_two_circles/Iter${i}"
    mkdir -p ${OUTPUT_DIR}

    # Train GP Model and get new points
    GP_COMMAND=("python" "${GP_MODEL_SCRIPT}" "--iteration=${i}" "--output_dir=${OUTPUT_DIR}")
    "${GP_COMMAND[@]}"
    if [ $? -ne 0 ]; then
        echo "Error during GP model script execution"
        exit 1
    fi

    echo "Completed iteration ${i}"
done

echo "All iterations completed."