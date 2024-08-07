import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def normalize_dependencies(dependencies):
    """ Normalize dependencies to a consistent string representation """
    normalized = []
    for dep in dependencies:
        if isinstance(dep, str):
            normalized.append(dep)
        elif isinstance(dep, dict):
            for key, value in dep.items():
                if key == 'pip':
                    # Flatten pip dependencies for comparison
                    for pip_dep in value:
                        normalized.append(f"pip::{pip_dep}")
                else:
                    normalized.append(f"{key}::{value}")
    return normalized

def merge_dependencies(dependencies1, dependencies2):
    set1 = set(normalize_dependencies(dependencies1))
    set2 = set(normalize_dependencies(dependencies2))
    unique_deps = list(set1.symmetric_difference(set2))
    return unique_deps

# Load both YAML files
pixi_env = load_yaml('pixi_environment.yaml')
alenvironment_env = load_yaml('alenvironment.yaml')

# Extract dependencies
pixi_deps = pixi_env.get('dependencies', [])
alenvironment_deps = alenvironment_env.get('dependencies', [])

# Merge dependencies
merged_deps = merge_dependencies(pixi_deps, alenvironment_deps)

# Create a new environment with merged dependencies
merged_env = {
    'name': 'combined_env',
    'channels': list(set(pixi_env.get('channels', []) + alenvironment_env.get('channels', []))),
    'dependencies': merged_deps
}

# Save the merged environment to a new YAML file
save_yaml(merged_env, 'combined_env.yaml')

print("Merged environment saved to combined_env.yaml")
