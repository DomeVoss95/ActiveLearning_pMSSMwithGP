#!/bin/bash

# Set the paths and variables
GP_MODEL_SCRIPT="/u/dvoss/al_pmssmwithgp/model/GPmodelRealData_hyperScan.py"
ITERATIONS=1

# Get the parameters passed from the sbatch script
lengthscale_min=$1
lengthscale_max=$2
outputscale_min=$3
noise_min=$4
noise_max=$5
learning_rate=$6
iterations=$7
optimizer=$8

for i in $(seq 1 ${ITERATIONS}); do
    echo "Starting iteration ${i} with parameters: ls_min=${lengthscale_min}, ls_max=${lengthscale_max}, os_min=${outputscale_min}, noise_min=${noise_min}, noise_max=${noise_max}, lr=${learning_rate}, iter=${iterations}, opt=${optimizer}"

    OUTPUT_DIR="/raven/u/dvoss/al_pmssmwithgp/model/plots/Iter${i}"
    mkdir -p ${OUTPUT_DIR}

    # Step 1: Train GP Model and get new points
    GP_COMMAND=("python" "${GP_MODEL_SCRIPT}" "--iteration=${i}" "--output_dir=${OUTPUT_DIR}" \
                "--lengthscale_min=${lengthscale_min}" "--lengthscale_max=${lengthscale_max}" \
                "--outputscale_min=${outputscale_min}" "--noise_min=${noise_min}" \
                "--noise_max=${noise_max}" "--learning_rate=${learning_rate}" \
                "--iterations=${iterations}" "--optimizer=${optimizer}")
    "${GP_COMMAND[@]}"
    if [ $? -ne 0 ]; then
        echo "Error during GP model script execution"
        exit 1
    fi
done

echo "All iterations completed."