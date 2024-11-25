import uproot
import torch
import pandas as pd
import pickle
from utils import normalize

# Define the custom CPU_Unpickler class
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # Override to load tensors directly to CPU
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class DataLoader:
    def __init__(self, start_root_file_path=None, true_root_file_path=None, root_file_path=None, initial_train_points=1000, valid_points=400, device=None):
        self.start_root_file_path = start_root_file_path
        self.true_root_file_path = true_root_file_path
        self.root_file_path = root_file_path
        self.initial_train_points = initial_train_points
        self.valid_points = valid_points
        self.device = device

        

    def load_initial_data(self):

        # Open the ROOT file
        file = uproot.open(self.start_root_file_path)
        
        tree_name = "susy"
        tree = file[tree_name]
        
        # Convert tree to pandas DataFrame
        final_df = tree.arrays(library="pd")

        # M_1 = final_df['IN_M_1']
        # M_2 = final_df['IN_M_2']
        # tanb = final_df['IN_tanb']
        # mu = final_df['IN_mu']
        Omega = final_df['MO_Omega']

        # TODO: these kind of implementation writ in a function wich i can call load_dataframe

        # Define the dictionary to map each n to the appropriate parameter names
        order = {1: ["IN_M_1"],
                 2: ["IN_M_1", "IN_M_2"],
                 3: ["IN_M_1", "IN_M_2", "IN_tanb"],
                 4: ["IN_M_1", "IN_M_2", "IN_tanb", "IN_mu"]}

        # Get the parameters to include based on n
        selected_columns = order.get(n, None)

        # Apply the mask to filter only valid Omega values
        mask = Omega > 0
        filtered_data = {param: final_df[f"{param}"][mask] for param in selected_columns}
        filtered_data['MO_Omega'] = Omega[mask]  # Always include Omega for filtering

        # Construct the limited DataFrame dynamically with only the selected columns
        limited_df = pd.DataFrame(filtered_data)
        
        # mask = Omega > 0
        # M_1_filtered = M_1[mask]
        # M_2_filtered = M_2[mask]
        # Omega_filtered = Omega[mask]
        
        # limited_df = pd.DataFrame({'IN_M_1': M_1_filtered, 'IN_M_2': M_2_filtered, 'MO_Omega': Omega_filtered})
        
        # Drop rows where IN_M_1 is between 0 and 50, because chance of outliers is high
        # limited_df = limited_df.drop(limited_df[(limited_df['IN_M_1'] > 0) & (limited_df['IN_M_1'] < 50)].index)

        # Convert to tensors
        # self.x_train = torch.stack([torch.tensor(limited_df['IN_M_1'].values[:self.initial_train_points], dtype=torch.float32), torch.tensor(limited_df['IN_M_2'].values[:self.initial_train_points], dtype=torch.float32)], dim=1).to(self.device)
        self.x_train = torch.stack([torch.tensor(limited_df[col].values[:self.initial_train_points], dtype=torch.float32) for col in selected_columns], dim=1).to(self.device)
        self.y_train = torch.log(torch.tensor(limited_df['MO_Omega'].values[:self.initial_train_points], dtype=torch.float32).to(self.device) / 0.12)
        
        # self.x_valid = torch.stack([torch.tensor(limited_df['IN_M_1'].values[self.valid_points:], dtype=torch.float32), torch.tensor(limited_df['IN_M_2'].values[self.valid_points:], dtype=torch.float32)], dim=1).to(self.device)
        self.x_valid = torch.stack([torch.tensor(limited_df[col].values[self.valid_points:], dtype=torch.float32) for col in selected_columns], dim=1).to(self.device)
        self.y_valid = torch.log(torch.tensor(limited_df['MO_Omega'].values[self.valid_points:], dtype=torch.float32).to(self.device) / 0.12)
        
        self.x_train, self.data_min, self.data_max = normalize(self.x_train)
        self.x_valid = normalize(self.x_valid, self.data_min, self.data_max)[0]

        print("Initial Training points: ", self.x_train, self.x_train.shape)

    # Add the generated data from Run3ModelGen
    def load_additional_data(self):

        # TODO: these part i can actually put into a function for DRY purpose
        # Open the ROOT file
        file = uproot.open(self.root_file_path)
        
        tree_name = "susy"
        tree = file[tree_name]
        
        # Convert tree to pandas DataFrame
        al_df = tree.arrays(library="pd")
        M_1_al = al_df['IN_M_1']
        M_2_al = al_df['IN_M_2']
        Omega_al = al_df['MO_Omega']
        
        # Convert al data to tensors and concatenate with existing training data
        M_1_al_tensor = torch.tensor(M_1_al.values, dtype=torch.float32).to(self.device)
        M_2_al_tensor = torch.tensor(M_2_al.values, dtype=torch.float32).to(self.device)
        Omega_al_tensor = torch.tensor(Omega_al.values, dtype=torch.float32).to(self.device)
        
        # Normalize M_1 and M_2
        M_1_al_normalized = normalize(M_1_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter]
        M_2_al_normalized = normalize(M_2_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter]

        # Concatenate M_1 and M_2 into a single tensor of shape [N, 2]
        additional_x_train = torch.cat([M_1_al_normalized.unsqueeze(1), M_2_al_normalized.unsqueeze(1)], dim=1)

        # additional_x_train = torch.stack([self._normalize(M_1_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter], self._normalize(M_2_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter]])
        additional_y_train = torch.log(Omega_al_tensor / 0.12)[:self.additional_points_per_iter]

        # Debugging: Print shapes of both tensors before concatenating
        print(f"self.x_train shape: {self.x_train.shape}")  # Expecting [N, 2]
        print(f"additional_x_train shape: {additional_x_train.shape}")  # Should also be [M, 2]

        print(f"additional_x_train: {additional_x_train}")

        # Ensure the additional_x_train has the same number of columns as self.x_train
        if additional_x_train.shape[1] != self.x_train.shape[1]:
            additional_x_train = additional_x_train[:, :self.x_train.shape[1]]  # Adjust to the correct number of columns
        
        # Append the new points to the existing training data
        self.x_train = torch.cat((self.x_train, additional_x_train))
        self.y_train = torch.cat((self.y_train, additional_y_train))

        # Combine x_train (which is 2D) and y_train into tuples (x1, x2, y)
        combined_set = {(float(x1), float(x2), float(y.item())) for (x1, x2), y in zip(self.x_train, self.y_train)}

        # Unpack the combined_set into x_train and y_train
        x_train, y_train = zip(*[(x[:2], x[2]) for x in combined_set])

        # Convert the unpacked x_train and y_train back to torch tensors
        self.x_train = torch.tensor(list(x_train), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(list(y_train), dtype=torch.float32).to(self.device)

        # # Convert the concatenated x_train and y_train to a set of tuples to remove duplicates
        # combined_set = {(float(x1), float(x2), float(y.item())) for (x1, x2), y in zip(self.x_train, self.y_train)}
        # print(self.x_train.shape)
        # print(self.y_train.shape)
        # # combined_set = {(float(x), float(y)) for x, y in zip(self.x_train, self.y_train)}

        # # Convert the set back to two tensors
        # self.x_train, self.y_train = zip(*combined_set)
        # self.x_train = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        # self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)

        #print("Unique training points: ", self.x_train, self.x_train.shape)

        print("Training points after adding: ", self.x_train, self.x_train.shape)

    # def _normalize(self, data, data_min=None, data_max=None):
    #     if data_min is None:
    #         data_min = data.min(dim=0, keepdim=True).values
    #     if data_max is None:
    #         data_max = data.max(dim=0, keepdim=True).values
    #     return (data - data_min) / (data_max - data_min), data_min, data_max

    # Save the trained data in a pickle file
    def save_training_data(filepath, x_train, y_train, data_min, data_max):
        with open(filepath, 'wb') as f:
            pickle.dump((x_train, y_train, data_min, data_max), f)

    # Load the trained data in a pickle file, which is GPU specific for raven
    def load_training_data_raven(self, filepath):
        with open(filepath, 'rb') as f:
            self.x_train, self.y_train, self.data_min, self.data_max = pickle.load(f)  

    # Load the trained data in a pickle file, which can be processed by CPU
    def load_training_data(self, filepath):
        with open(filepath, 'rb') as f:
            self.x_train, self.y_train, self.data_min, self.data_max = CPU_Unpickler(f).load()

    # Save the model state dictionary in checkpoint
    def save_model(self, model_checkpoint_path):
        torch.save(self.model.state_dict(), model_checkpoint_path)

    # Load the state_dict from checkpoint
    def load_model(self, model_checkpoint_path):
        # Initialize the model architecture
        self.initialize_model()

        # # Load the saved dict
        # self.model.load_state_dict(torch.load(model_checkpoint_path))
        # print(f"Model loaded from the {model_checkpoint_path}")
        
        # Ensure the model loads on the correct device (GPU or CPU)
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved state dict
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=map_location))
        print(f"Model loaded from {model_checkpoint_path}")

