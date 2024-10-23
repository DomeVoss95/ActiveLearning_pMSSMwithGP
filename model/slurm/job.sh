#!/bin/bash

# python /raven/u/dvoss/al_pmssmwithgp/model/GPmodel2Dnew.py # adapt

## Set the paths and variables
GP_MODEL_SCRIPT="/u/dvoss/al_pmssmwithgp/model/GPmodel2Dnew.py"
GEN_MODELS_SCRIPT="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scripts/genModels.py"
SETUP_SCRIPT="/u/dvoss/al_pmssmwithgp/Run3ModelGen/build/setup.sh"
PIX_WORKING_DIR="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen"
RUN3MODELGEN_PARENT_DIR="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/"
ITERATIONS=1

# Function to wait for a file to be created
wait_for_file() {
    local file_path=$1
    local timeout=300
    local start_time=$(date +%s)

    while [ ! -e ${file_path} ]; do
        if (( $(date +%s) - ${start_time} > ${timeout} )); then
            echo "Timed out waiting for ${file_path} to be created."
            exit 1
        fi
        sleep 5
    done
}

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

    # Step 2: Generate new models with genModels.py
    CONFIG_PATH="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/data/new_config.yaml"
    SCAN_DIRECTORY="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_${i}"

    GEN_MODELS_COMMAND="cd ${PIX_WORKING_DIR} && source ${SETUP_SCRIPT} && export PYTHONPATH=${RUN3MODELGEN_PARENT_DIR}:\$PYTHONPATH && python ${GEN_MODELS_SCRIPT} --config_file ${CONFIG_PATH} --scan_dir ${SCAN_DIRECTORY}"
    bash -c "${GEN_MODELS_COMMAND}"
    if [ $? -ne 0 ]; then
        echo "Error during genModels script execution"
        exit 1
    fi

    # Wait for the expected ROOT file to be created
    ROOT_FILE_PATH="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_${i}/ntuple.0.0.root"
    wait_for_file ${ROOT_FILE_PATH}

    echo "Completed iteration ${i}"
done

echo "All iterations completed."