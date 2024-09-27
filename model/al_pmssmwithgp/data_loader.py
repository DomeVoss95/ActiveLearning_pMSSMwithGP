import pandas as pd
import uproot
import torch

def load_initial_data(csv_file):
    final_df = pd.read_csv(csv_file)
    M_1 = final_df['IN_M_1']
    M_2 = final_df['IN_M_2']
    Omega = final_df['MO_Omega']

    mask = Omega > 0
    M_1_filtered = M_1[mask]
    M_2_filtered = M_2[mask]
    Omega_filtered = Omega[mask]

    x_train = torch.stack([torch.tensor(M_1_filtered.values, dtype=torch.float32), torch.tensor(M_2_filtered.values, dtype=torch.float32)], dim=1)
    y_train = torch.log(torch.tensor(Omega_filtered.values, dtype=torch.float32) / 0.12)

    x_train, data_min, data_max = normalize(x_train)
    return x_train, y_train, data_min, data_max

def load_additional_data(root_file_path, additional_points_per_iter, data_min, data_max):
    file = uproot.open(root_file_path)
    tree = file["susy"]

    al_df = tree.arrays(library="pd")
    M_1_al = al_df['IN_M_1']
    M_2_al = al_df['IN_M_2']
    Omega_al = al_df['MO_Omega']

    M_1_al_tensor = torch.tensor(M_1_al.values, dtype=torch.float32)
    M_2_al_tensor = torch.tensor(M_2_al.values, dtype=torch.float32)

    additional_x_train = torch.stack([normalize(M_1_al_tensor, data_min, data_max)[0], normalize(M_2_al_tensor, data_min, data_max)[0]], dim=1)[:additional_points_per_iter]
    additional_y_train = torch.log(torch.tensor(Omega_al.values, dtype=torch.float32) / 0.12)[:additional_points_per_iter]

    return additional_x_train, additional_y_train

def normalize(data, data_min=-2000, data_max=2000):
    return (data - data_min) / (data_max - data_min), data_min, data_max

def unnormalize(data, data_min, data_max):
    return data * (data_max - data_min) + data_min
