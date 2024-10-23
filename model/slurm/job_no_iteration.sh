#!/bin/bash

# python /raven/u/dvoss/al_pmssmwithgp/model/GPmodel2Dnew.py 

# ## Set the paths and variables
# GP_MODEL_SCRIPT="/u/dvoss/al_pmssmwithgp/model/GPmodel2Dnew.py"
# GEN_MODELS_SCRIPT="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scripts/genModels.py"
# SETUP_SCRIPT="/u/dvoss/al_pmssmwithgp/Run3ModelGen/build/setup.sh"
# PIX_WORKING_DIR="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen"
# RUN3MODELGEN_PARENT_DIR="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/"
# ITERATIONS=20

# # Function to wait for a file to be created
# wait_for_file() {
#     local file_path=$1
#     local timeout=300
#     local start_time=$(date +%s)

#     while [ ! -e ${file_path} ]; do
#         if (( $(date +%s) - ${start_time} > ${timeout} )); then
#             echo "Timed out waiting for ${file_path} to be created."
#             exit 1
#         fi
#         sleep 5
#     done
# }

# echo "Starting training"

# OUTPUT_DIR="/raven/u/dvoss/al_pmssmwithgp/model/plots/plots_no_iteration"
# mkdir -p ${OUTPUT_DIR}

# # Step 1: Train GP Model and get new points
# GP_COMMAND=("python" "${GP_MODEL_SCRIPT}" "--output_dir=${OUTPUT_DIR}")
# "${GP_COMMAND[@]}"
# if [ $? -ne 0 ]; then
#     echo "Error during GP model script execution"
#     exit 1
# fi

# echo "Training completed."