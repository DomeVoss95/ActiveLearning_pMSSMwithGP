import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
import pandas as pd
import numpy as np

class MultitaskGPModel:
    def __init__(self, csv_file, n_points=50, n_train=30, n_test=100000):
        self.csv_file = csv_file
        self.n_points = n_points
        self.n_train = n_train
        self.n_test = n_test
        
        self.likelihood = GaussianLikelihood()
        self.model = None
        self.best_model = None
        self.losses = None
        self.losses_valid = None
        
    def load_data(self):
        final_df = pd.read_csv(self.csv_file)
        M_1 = final_df['IN_M_1']
        Omega = final_df['MO_Omega']

        # Filter Omega, M_1
        mask = (Omega > 0)
        M_1_filtered = M_1[mask]
        Omega_filtered = Omega[mask]

        # Limit to only 100 points
        M_1_limited = M_1_filtered.iloc[:self.n_points]
        Omega_limited = Omega_filtered.iloc[:self.n_points]

        # Convert to train and valid sets
        self.x_train = torch.tensor(M_1_limited.values[:self.n_train], dtype=torch.float32)
        self.y_train = torch.log(torch.tensor(Omega_limited.values[:self.n_train], dtype=torch.float32) / 0.12)
        self.x_valid = torch.tensor(M_1_limited.values[self.n_train:], dtype=torch.float32)
        self.y_valid = torch.log(torch.tensor(Omega_limited.values[self.n_train:], dtype=torch.float32) / 0.12)
        
        # Normalize data
        self.x_train = self._normalize(self.x_train)
        self.x_valid = self._normalize(self.x_valid)
        
    def _normalize(self, data):
        data_min = data.min(dim=0, keepdim=True).values
        data_max = data.max(dim=0, keepdim=True).values
        return (data - data_min) / (data_max - data_min)
    
    def initialize_model(self):
        from multitaskGP import MultitaskGP
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood)
        self.best_model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood)
        
    def train_model(self, iters=200):
        if self.model is None:
            raise ValueError("Model has not been initialized. Call initialize_model() first.")
        self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(iters=iters)
        
    def save_model(self, path='multitask_gp_state_dict.pth'):
        if self.model is None:
            raise ValueError("Model has not been initialized or trained. Call initialize_model() and train_model() first.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'best_model_state_dict': self.best_model.state_dict() if self.best_model else None,
            'losses': self.losses,
            'losses_valid': self.losses_valid
        }, path)
        
    def load_model(self, path='multitask_gp_state_dict.pth'):
        if self.model is None:
            self.initialize_model()
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        if checkpoint['best_model_state_dict']:
            if self.best_model is None:
                from multitaskGP import MultitaskGP
                self.best_model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood)
            self.best_model.load_state_dict(checkpoint['best_model_state_dict'])
        self.losses = checkpoint['losses']
        self.losses_valid = checkpoint['losses_valid']
        self.model.eval()  # Set the model to evaluation mode
        self.best_model.eval()  # Set the best model to evaluation mode


# Usage
gp_model = MultitaskGPModel(csv_file='M_2_fixed.csv')
gp_model.load_data()
gp_model.initialize_model()
gp_model.train_model(iters=20)
gp_model.save_model('multitask_gp_state_dict.pth')

# Access the loaded attributes
best_model = gp_model.best_model
losses = gp_model.losses
losses_valid = gp_model.losses_valid

# Use the attributes as needed, for example:
print("Best model:", best_model)
print("Training losses:", losses)
print("Validation losses:", losses_valid)
