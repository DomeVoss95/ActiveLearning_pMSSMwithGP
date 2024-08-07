import subprocess
import os
import sys
import time

# Paths
gp_model_script = "/u/dvoss/al_pmssmwithgp/model/GPmodel1Dtest.py"
gen_models_script = "/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scripts/genModels.py"
setup_script = "/u/dvoss/al_pmssmwithgp/Run3ModelGen/build/setup.sh"
pixi_working_dir = "/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen"
run3modelgen_parent_dir = "/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/"

# Iterations
iterations = 10

def wait_for_file(file_path, timeout=300):
    """Wait for a file to be created, with a timeout."""
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for {file_path} to be created.")
        time.sleep(5)  # Wait for 5 seconds before checking again

for i in range(0, iterations + 1):
    print(f"Starting iteration {i}")

    # Step 1: Train GP Model and get new points
    output_dir = f'/raven/u/dvoss/al_pmssmwithgp/model/plots/Iter{i}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute the GP model script
    gp_command = [
        "conda", "run", "-n", "ALenv", "python", gp_model_script,
        f"--iteration={i}",
        f"--output_dir={output_dir}"
    ]
    try:
        result = subprocess.run(gp_command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during GP model script execution: {e}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
    
    # Step 2: Generate new models with genModels.py
    config_path = "/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/data/new_config.yaml"
    scan_directory = f"/u/dvoss/al_pmssmwithgp/Run3ModelGen/scans/scan_{i+1}"

    gen_models_command = f"""
    cd {pixi_working_dir}
    source {setup_script}
    export PYTHONPATH={run3modelgen_parent_dir}:$PYTHONPATH
    python {gen_models_script} --config_file {config_path} --scan_dir {scan_directory}
    """
    
    try:
        result = subprocess.run(['bash', '-c', gen_models_command], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during genModels script execution: {e}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

    # Wait for the expected ROOT file to be created
    root_file_path = f"/u/dvoss/al_pmssmwithgp/Run3ModelGen/scans/scan_{i+1}/ntuple.0.0.root"
    try:
        wait_for_file(root_file_path)
    except TimeoutError as e:
        print(e)
        sys.exit(1)

    print(f"Completed iteration {i}")

print("All iterations completed.")
