#!/bin/bash

# python /u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scripts/genModels.py

GEN_MODELS_SCRIPT="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scripts/genModels.py"
SETUP_SCRIPT="/u/dvoss/al_pmssmwithgp/Run3ModelGen/build/setup.sh"
PIX_WORKING_DIR="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen"
RUN3MODELGEN_PARENT_DIR="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/"


# Generate new models with genModels.py
CONFIG_PATH="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/data/start_config.yaml"
SCAN_DIRECTORY="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_start"

GEN_MODELS_COMMAND="cd ${PIX_WORKING_DIR} && source ${SETUP_SCRIPT} && export PYTHONPATH=${RUN3MODELGEN_PARENT_DIR}:\$PYTHONPATH && python ${GEN_MODELS_SCRIPT} --config_file ${CONFIG_PATH} --scan_dir ${SCAN_DIRECTORY}"
bash -c "${GEN_MODELS_COMMAND}"
if [ $? -ne 0 ]; then
    echo "Error during genModels script execution"
    exit 1
fi

# Wait for the expected ROOT file to be created
ROOT_FILE_PATH="/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_start/ntuple.0.0.root"
wait_for_file ${ROOT_FILE_PATH}