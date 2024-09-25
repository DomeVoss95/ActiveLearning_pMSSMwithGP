import yaml
import torch
import os

class InlineListDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(InlineListDumper, self).increase_indent(flow, False)

    def represent_list(self, data):
        if isinstance(data, list) and (all(isinstance(i, int) for i in data) or all(isinstance(i, float) for i in data)):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

def create_config(new_points, output_file='output.yaml'):
    parameters = {
        "tanb": [60],
        "M_1": [],
        "M_2": [], # Change for 1D
        "M_3": [4000],
        "AT": [4000],
        "Ab": [2000],
        "Atau": [2000],
        "mu": [2000],
        "mA": [2000],
        "meL": [2000],
        "mtauL": [2000],
        "meR": [2000],
        "mtauR": [2000],
        "mqL1": [4000],
        "mqL3": [4000],
        "muR": [4000],
        "mtR": [4000],
        "mdR": [4000],
        "mbR": [4000]
    }

    # Define the output directory
    output_dir = '/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/data'

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert new_points to floats and round to two decimal places
    new_points_float = new_points.tolist()
    new_points_rounded = [[round(point, 2) for point in points] for points in new_points_float]

    # new_points_rounded is a list of lists: [[M_1_values], [M_2_values]]
    # Now we need to extract corresponding M_1 and M_2 values

    # Unzip new_points_rounded into separate M_1 and M_2 lists
    M_1_values = [point[0] for point in new_points_rounded]  # First column
    M_2_values = [point[1] for point in new_points_rounded]  # Second column

    # Append the rounded values to parameters
    parameters["M_1"].extend(M_1_values)
    parameters["M_2"].extend(M_2_values)


    # # Convert new_points to floats and round to two decimal places, then append to the M_1 list
    # new_points_float = new_points.tolist()
    # new_points_rounded = [[round(point, 2) for point in points] for points in new_points_float]
    # # new_points_rounded = [round(point, 2) for point in new_points_float]
    # parameters["M_1"].extend(new_points_rounded[0]) # Change for 1D
    # parameters["M_2"].extend(new_points_rounded[1]) # Change for 1D

    # Calculate the new length of M_1
    new_length = len(parameters["M_1"])

    # Extend other lists to match the new length
    for key, value in parameters.items():
        if key != "M_1" or "M_2":
            parameters[key].extend([value[-1]] * (new_length - len(value)))

    # Define the complete dictionary with additional keys
    data = {
        "prior": "fixed",
        "num_models": new_length,
        "isGMSB": False,
        "parameters": parameters,
        "steps": [
            {
                "name": "prep_input",
                "output_dir": "input",
                "prefix": "IN"
            },
            {
                "name": "SPheno",
                "input_dir": "input",
                "output_dir": "SPheno",
                "log_dir": "SPheno_log",
                "prefix": "SP"
            },
            {
                "name": "softsusy",
                "input_dir": "input",
                "output_dir": "softsusy",
                "prefix": "SS"
            },
            {
                "name": "micromegas",
                "input_dir": "SPheno",
                "output_dir": "micromegas",
                "prefix": "MO"
            },
            {
                "name": "superiso",
                "input_dir": "SPheno",
                "output_dir": "superiso",
                "prefix": "SI"
            },
            {
                "name": "gm2calc",
                "input_dir": "SPheno",
                "output_dir": "gm2calc",
                "prefix": "GM2"
            },
            {
                "name": "evade",
                "input_dir": "SPheno",
                "output_dir": "evade",
                "prefix": "EV"
            }
        ]
    }

    # Use the custom representer for lists in parameters
    dumper = InlineListDumper
    dumper.add_representer(list, dumper.represent_list)

    # Create the full path for the output file
    output_path = os.path.join(output_dir, output_file)

    # Write to a YAML file
    with open(output_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False, Dumper=dumper)

    print(f"YAML file has been created: {output_path}")

# Example usage
# new_points = torch.tensor([-1981.6007, -1854.4359, 1718.0997, -801.3525, 92.7748, -1047.7344, 609.3817, -1921.9922, 1658.4913, 271.6005])

# create_config(new_points)

