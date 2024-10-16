import uproot
import torch
import pandas as pd

class DataLoader:
    def __init__(self, device):
        self.device = device

    def load_initial_data(self, root_file_path, initial_train_points, valid_points):
        file = uproot.open(root_file_path)
        tree = file["susy"]
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1']
        M_2 = df['IN_M_2']
        Omega = df['MO_Omega']
        
        mask = Omega > 0
        M_1_filtered = M_1[mask]
        M_2_filtered = M_2[mask]
        Omega_filtered = Omega[mask]

        x_train = torch.stack([
            torch.tensor(M_1_filtered.values[:initial_train_points], dtype=torch.float32),
            torch.tensor(M_2_filtered.values[:initial_train_points], dtype=torch.float32)
        ], dim=1).to(self.device)

        y_train = torch.log(torch.tensor(Omega_filtered.values[:initial_train_points], dtype=torch.float32).to(self.device) / 0.12)

        x_valid = torch.stack([
            torch.tensor(M_1_filtered.values[valid_points:], dtype=torch.float32),
            torch.tensor(M_2_filtered.values[valid_points:], dtype=torch.float32)
        ], dim=1).to(self.device)

        y_valid = torch.log(torch.tensor(Omega_filtered.values[valid_points:], dtype=torch.float32).to(self.device) / 0.12)

        return x_train, y_train, x_valid, y_valid

    def load_additional_data(self, root_file_path, additional_points_per_iter, normalize_fn, data_min, data_max):
        file = uproot.open(root_file_path)
        tree = file["susy"]
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1']
        M_2 = df['IN_M_2']
        Omega = df['MO_Omega']

        mask = Omega > 0
        M_1_filtered = M_1[mask]
        M_2_filtered = M_2[mask]
        Omega_filtered = Omega[mask]

        # Convert to tensor and normalize
        M_1_tensor = torch.tensor(M_1_filtered.values, dtype=torch.float32).to(self.device)
        M_2_tensor = torch.tensor(M_2_filtered.values, dtype=torch.float32).to(self.device)
        Omega_tensor = torch.tensor(Omega_filtered.values, dtype=torch.float32).to(self.device)

        M_1_normalized = normalize_fn(M_1_tensor, data_min, data_max)[0][:additional_points_per_iter]
        M_2_normalized = normalize_fn(M_2_tensor, data_min, data_max)[0][:additional_points_per_iter]

        x_train_additional = torch.cat([M_1_normalized.unsqueeze(1), M_2_normalized.unsqueeze(1)], dim=1)
        y_train_additional = torch.log(Omega_tensor[:additional_points_per_iter] / 0.12)

        return x_train_additional, y_train_additional
