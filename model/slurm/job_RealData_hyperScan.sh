#!/bin/bash

# Set the paths and variables
GP_MODEL_SCRIPT="/u/dvoss/al_pmssmwithgp/model/GPmodelRealData_hyperScan.py"
PARAM_FILE="/u/dvoss/al_pmssmwithgp/model/slurm/param_list.txt"

# Read the parameter file line by line
while IFS=' ' read -r lengthscale_min lengthscale_max outputscale_min noise_min noise_max learning_rate iterations optimizer; do
    for i in $(seq 1 ${iterations}); do
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
done < ${PARAM_FILE}

echo "All iterations completed."