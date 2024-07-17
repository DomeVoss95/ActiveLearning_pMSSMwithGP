import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood
import numpy as np
import os
import uproot
import pandas as pd

from multitaskGP2D import MultitaskGP2D
from entropy import entropy_local

class GPModel:
    def __init__(self, base_dir, subdir_range, tree_name="susy"):
        self.base_dir = base_dir
        self.subdir_range = subdir_range
        self.tree_name = tree_name
        self.final_df = None
        self.likelihood = GaussianLikelihood()
        self.multitask_gp = None

    def load_data(self):
        # all_data = []
        # for subdir in self.subdir_range:
        #     file_path = os.path.join(self.base_dir, str(subdir), f"ntuple.6249447.{subdir}.root")
        #     try:
        #         file = uproot.open(file_path)
        #         tree = file[self.tree_name]
        #         df = tree.arrays(library="pd")
        #         all_data.append(df)
        #     except Exception as e:
        #         print(f"Failed to open or process {file_path}: {e}")

        # self.final_df = pd.concat(all_data, ignore_index=True)
        self.final_df = pd.read_csv('output.csv')
    
    def prepare_data(self):
        M_1 = self.final_df['IN_M_1']
        M_2 = self.final_df['IN_M_2']
        Omega = self.final_df['MO_Omega']

        n_points = 50
        n_train = 30

        mask = (Omega > 0) & (Omega < 1.0)
        M_1_filtered = M_1[mask]
        M_2_filtered = M_2[mask]
        Omega_filtered = Omega[mask]

        M_1_limited = M_1_filtered.iloc[:n_points]
        M_2_limited = M_2_filtered.iloc[:n_points]
        Omega_limited = Omega_filtered.iloc[:n_points]

        self.x_train = torch.stack([torch.tensor(M_1_limited.values[:n_train], dtype=torch.float32), torch.tensor(M_2_limited.values[:n_train], dtype=torch.float32)], dim=1)
        self.y_train = torch.log(torch.tensor(Omega_limited.values[:n_train], dtype=torch.float32) / 0.12)

        self.x_valid = torch.stack([torch.tensor(M_1_limited.values[n_train:], dtype=torch.float32), torch.tensor(M_2_limited.values[n_train:], dtype=torch.float32)], dim=1)
        self.y_valid = torch.log(torch.tensor(Omega_limited.values[n_train:], dtype=torch.float32) / 0.12)

        self.normalize_data()

    def normalize_data(self):
        x_train_min = self.x_train.min(dim=0, keepdim=True).values
        x_train_max = self.x_train.max(dim=0, keepdim=True).values
        self.x_train = (self.x_train - x_train_min) / (x_train_max - x_train_min)

        x_valid_min = self.x_valid.min(dim=0, keepdim=True).values
        x_valid_max = self.x_valid.max(dim=0, keepdim=True).values
        self.x_valid = (self.x_valid - x_valid_min) / (x_valid_max - x_valid_min)

    def train_model(self, iters=200):
        self.multitask_gp = MultitaskGP2D(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood, 2)
        self.best_multitask_gp, self.losses, self.losses_valid = self.multitask_gp.do_train_loop(iters=iters)

    def plot_losses(self, filename="loss_plot.png"):
        plt.plot(self.losses, label='training loss')
        plt.plot(self.losses_valid, label='validation loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss on Logarithmic Scale')
        plt.savefig(filename)
        plt.close()

    def evaluate(self, x_test):
        self.multitask_gp.eval()
        self.likelihood.eval()

        with torch.no_grad():
            observed_pred = self.likelihood(self.multitask_gp(x_test))
            mean = observed_pred.mean.detach().reshape(-1, 1)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float32
            var = observed_pred.variance.detach().reshape(-1, 1)
            thr = torch.Tensor([0.])
            entropy = entropy_local(mean, var, thr, device, dtype)

        return observed_pred, entropy

    def plot_heatmap(self, x_test, z, label, title, filename):
        heatmap, xedges, yedges = np.histogram2d(x_test[:, 0], x_test[:, 1], bins=50, weights=z, density=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label=label)
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')
        plt.title(title)
        plt.savefig(filename)
        plt.close()

    def plot_entropy(self, x_test, entropy, filename="entropy_heatmap.png"):
        heatmap, xedges, yedges = np.histogram2d(x_test[:, 0], x_test[:, 1], bins=50, weights=entropy, density=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Entropy')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')
        plt.title('Entropy Heatmap')
        plt.savefig(filename)
        plt.close()

    def highlight_top_entropy_points(self, x_test, entropy, observed_pred, filename="highlight_top_entropy_points.png"):
        mean = observed_pred.mean.numpy()
        heatmap, xedges, yedges = np.histogram2d(x_test[:, 0], x_test[:, 1], bins=50, weights=mean, density=True)
        topk_indices = torch.argsort(entropy, descending=True)[:10]
        top_10_points = x_test[topk_indices]
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Mean log(Omega/0.12)')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')
        plt.title('Gaussian Process Mean Heatmap')
        plt.scatter(top_10_points[:, 0], top_10_points[:, 1], marker='*', s=200, c='r', label='Top 10 High Entropy Points')
        plt.contour(xedges[:-1], yedges[:-1], heatmap.T, levels=[0], colors='white', linewidths=2, linestyles='solid')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def highlight_top_entropy_points(self, x_test, entropy, observed_pred, filename="highlight_top_entropy_points.png"):
        mean = observed_pred.mean.numpy()
        heatmap, xedges, yedges = np.histogram2d(x_test[:, 0], x_test[:, 1], bins=50, weights=mean, density=True)
        topk_indices = torch.argsort(entropy, descending=True)[:10]
        top_10_points = x_test[topk_indices]
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Mean log(Omega/0.12)')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')
        plt.title('Gaussian Process Mean Heatmap')
        plt.scatter(top_10_points[:, 0], top_10_points[:, 1], marker='*', s=200, c='r', label='Top 10 High Entropy Points')
        plt.contour(xedges[:-1], yedges[:-1], heatmap.T, levels=[0], colors='white', linewidths=2, linestyles='solid')
        plt.legend()
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    base_dir = "/eos/user/d/dvoss/Run3ModelGen/6249447/"
    subdir_range = range(0, 20)
    model = GPModel(base_dir, subdir_range)
    
    model.load_data()
    model.prepare_data()
    model.train_model(iters=20)
    model.plot_losses()

    n_test = 10000
    
    # Test data
    x_test = torch.stack([torch.tensor(model.final_df['IN_M_1'].values[50:n_test], dtype=torch.float32), torch.tensor(model.final_df['IN_M_2'].values[50:n_test], dtype=torch.float32)], dim=1)
    x_test = (x_test - x_test.mean(dim=0)) / x_test.std(dim=0)
    
    observed_pred, entropy = model.evaluate(x_test)
    
    mean = observed_pred.mean.numpy()
    model.plot_heatmap(x_test, mean, 'Mean log(Omega/0.12)', 'Gaussian Process Mean Heatmap', "mean_heatmap.png")
    
    z = torch.tensor(mean) - torch.log(torch.tensor(model.final_df['MO_Omega'].values[50:n_test], dtype=torch.float32) / 0.12)
    model.plot_heatmap(x_test, z.numpy(), 'Mean log(Omega/0.12)', 'Gaussian Process Mean Heatmap (Predicted vs True)', "pred_vs_true_heatmap.png")
    
    z = torch.log(torch.tensor(model.final_df['MO_Omega'].values[50:n_test], dtype=torch.float32) / 0.12)
    model.plot_heatmap(x_test, z.numpy(), 'log(Omega/0.12)', 'True Function Heatmap', "true_function_heatmap.png")
    
    model.plot_entropy(x_test, entropy.numpy(), "entropy_heatmap.png")
    
    model.highlight_top_entropy_points(x_test, entropy, observed_pred, "highlight_top_entropy_points.png")
