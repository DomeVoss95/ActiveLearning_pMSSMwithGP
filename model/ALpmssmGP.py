import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood
import numpy as np
import os
import uproot
import pandas as pd

class GPDataProcessor:
    def __init__(self, base_dir, subdir_range):
        self.base_dir = base_dir
        self.subdir_range = subdir_range
        self.all_data = []
        self.final_df = None
    
    def load_data(self):
        for subdir in self.subdir_range:
            file_path = os.path.join(self.base_dir, str(subdir), f"ntuple.6249447.{subdir}.root")
            try:
                file = uproot.open(file_path)
                tree_name = "susy"  # Replace with the actual tree name
                tree = file[tree_name]
                df = tree.arrays(library="pd")
                self.all_data.append(df)
            except Exception as e:
                print(f"Failed to open or process {file_path}: {e}")
        self.final_df = pd.concat(self.all_data, ignore_index=True)
        return self.final_df

    def filter_data(self):
        M_1 = self.final_df['IN_M_1']
        M_2 = self.final_df['IN_M_2']
        Omega = self.final_df['MO_Omega']
        
        mask = (Omega > 0) & (Omega < 1.0)
        M_1_filtered = M_1[mask]
        M_2_filtered = M_2[mask]
        Omega_filtered = Omega[mask]

        return M_1_filtered, M_2_filtered, Omega_filtered

class GPTorchModel:
    def __init__(self, x_train, y_train, x_valid, y_valid):
        self.likelihood = GaussianLikelihood()
        self.model = MultitaskGP2D(x_train, y_train, x_valid, y_valid, self.likelihood, 2)
    
    def normalize_data(self, x_train, x_valid):
        x_train_min = x_train.min(dim=0, keepdim=True).values
        x_train_max = x_train.max(dim=0, keepdim=True).values
        x_train = (x_train - x_train_min) / (x_train_max - x_train_min)

        x_valid_min = x_valid.min(dim=0, keepdim=True).values
        x_valid_max = x_valid.max(dim=0, keepdim=True).values
        x_valid = (x_valid - x_valid_min) / (x_valid_max - x_valid_min)
        
        return x_train, x_valid

    def train_model(self, iters=2000):
        best_model, losses, losses_valid = self.model.do_train_loop(iters=iters)
        return best_model, losses, losses_valid
    
    def evaluate_model(self, x_test):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            observed_pred = self.likelihood(self.model(x_test))
            mean = observed_pred.mean.detach().reshape(-1, 1)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float32
            var = observed_pred.variance.detach().reshape(-1, 1)
            thr = torch.Tensor([0.])
            entropy = entropy_local(mean, var, thr, device, dtype)
        return observed_pred, entropy

class GPVisualizer:
    @staticmethod
    def plot_losses(losses, losses_valid):
        plt.plot(losses, label='training loss')
        plt.plot(losses_valid, label='validation loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss on Logarithmic Scale')
        plt.show()

    @staticmethod
    def plot_heatmap(x_test, y_test, z, top_10_points=None, title='Heatmap', cbar_label='Value', levels=50, contour=False):
        heatmap, xedges, yedges = np.histogram2d(x_test, y_test, bins=50, weights=z, density=True)
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        xmesh, ymesh = np.meshgrid(xcenters, ycenters)

        plt.figure(figsize=(8, 6))
        contour_plot = plt.contourf(xmesh, ymesh, heatmap.T, cmap='inferno', levels=levels)
        plt.colorbar(contour_plot, label=cbar_label)
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')
        plt.title(title)
        
        if top_10_points is not None:
            plt.scatter(top_10_points[:, 0], top_10_points[:, 1], marker='*', s=200, c='r', label='Top 10 High Entropy Points')
            plt.legend()

        if contour:
            plt.contour(xmesh, ymesh, heatmap.T, levels=[0], colors='white', linewidths=2, linestyles='solid')

        plt.show()

# Example usage
if __name__ == "__main__":
    base_dir = "/eos/user/d/dvoss/Run3ModelGen/6249447/"
    subdir_range = range(0, 2)  # Adjust range as needed

    processor = GPDataProcessor(base_dir, subdir_range)
    processor.load_data()
    M_1_filtered, M_2_filtered, Omega_filtered = processor.filter_data()

    n_points = 50
    n_train = 30
    n_test = 100000

    M_1_limited = M_1_filtered.iloc[:n_points]
    M_2_limited = M_2_filtered.iloc[:n_points]
    Omega_limited = Omega_filtered.iloc[:n_points]

    x_train = torch.stack([torch.tensor(M_1_limited.values[:n_train], dtype=torch.float32), torch.tensor(M_2_limited.values[:n_train], dtype=torch.float32)], dim=1)
    y_train = torch.log(torch.tensor(Omega_limited.values[:n_train], dtype=torch.float32)/0.12)

    x_valid = torch.stack([torch.tensor(M_1_limited.values[n_train:], dtype=torch.float32), torch.tensor(M_2_limited.values[n_train:], dtype=torch.float32)], dim=1)
    y_valid = torch.log(torch.tensor(Omega_limited.values[n_train:], dtype=torch.float32)/0.12)

    model = GPTorchModel(x_train, y_train, x_valid, y_valid)
    x_train, x_valid = model.normalize_data(x_train, x_valid)

    best_model, losses, losses_valid = model.train_model(iters=2000)
    
    visualizer = GPVisualizer()
    visualizer.plot_losses(losses, losses_valid)
    
    x_test = torch.stack([torch.tensor(M_1_filtered.values[n_points:n_test], dtype=torch.float32), torch.tensor(M_2_filtered.values[n_points:n_test], dtype=torch.float32)], dim=1)
    x_test = (x_test - x_test.mean(dim=0)) / x_test.std(dim=0)

    observed_pred, entropy = model.evaluate_model(x_test)
    
    mean = observed_pred.mean.numpy()
    
    # Plotting various heatmaps
    visualizer.plot_heatmap(x_test[:, 0], x_test[:, 1], mean, title='Gaussian Process Mean Heatmap', cbar_label='Mean log(Omega/0.12)')
    
    diff = torch.tensor(mean) - torch.log(torch.tensor(Omega_filtered.values[n_points:n_test], dtype=torch.float32)/0.12)
    visualizer.plot_heatmap(x_test[:, 0], x_test[:, 1], diff, title='Difference GP Mean/ True Heatmap', cbar_label='Difference log(Omega/0.12)')
    
    true_log_omega = torch.log(torch.tensor(Omega_filtered.values[n_points:n_test], dtype=torch.float32)/0.12)
    visualizer.plot_heatmap(x_test[:, 0], x_test[:, 1], true_log_omega, title='True Function Heatmap', cbar_label='log(Omega/0.12)')
    
    visualizer.plot_heatmap(x_test[:, 0], x_test[:, 1], entropy, title='Entropy Heatmap', cbar_label='Entropy')

    # Highlighting top 10 high entropy points
    topk_indices = torch.argsort(entropy, descending=True)[:10]
    top_10_points = x_test[topk_indices]
    visualizer.plot_heatmap(x_test[:, 0], x_test[:, 1], mean, top_10_points=top_10_points, title='Gaussian Process Mean Heatmap with Top 10 Points', cbar_label='Mean log(Omega/0.12)', contour=True)
