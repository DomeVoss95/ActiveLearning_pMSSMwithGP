import os
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood
import pandas as pd
import numpy as np
from entropy import entropy_local  # Ensure this is the correct import path for your entropy_local function
from multitaskGP import MultitaskGP



class GPModelPipeline:
    def __init__(self, csv_file, output_dir, n_points=70, n_train=30, n_test=100000):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.n_points = n_points
        self.n_train = n_train
        self.n_test = n_test
        
        self.likelihood = GaussianLikelihood()
        self.model = None
        self.best_model = None
        self.losses = None
        self.losses_valid = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.observed_pred = None
        self.entropy = None
        self.x_test = torch.linspace(0, 1, 1000)
        
        self.load_data()
        self.initialize_model()

    def load_data(self):
        final_df = pd.read_csv(self.csv_file)
        M_1 = final_df['IN_M_1']
        Omega = final_df['MO_Omega']

        mask = (Omega > 0)
        M_1_filtered = M_1[mask]
        Omega_filtered = Omega[mask]

        M_1_limited = M_1_filtered.iloc[:self.n_points]
        Omega_limited = Omega_filtered.iloc[:self.n_points]

        self.x_train = torch.tensor(M_1_limited.values[:self.n_train], dtype=torch.float32)
        self.y_train = torch.log(torch.tensor(Omega_limited.values[:self.n_train], dtype=torch.float32) / 0.12)
        self.x_valid = torch.tensor(M_1_limited.values[self.n_train:], dtype=torch.float32)
        self.y_valid = torch.log(torch.tensor(Omega_limited.values[self.n_train:], dtype=torch.float32) / 0.12)

        self.x_train = self._normalize(self.x_train)
        self.x_valid = self._normalize(self.x_valid)

    def _normalize(self, data):
        data_min = data.min(dim=0, keepdim=True).values
        data_max = data.max(dim=0, keepdim=True).values
        return (data - data_min) / (data_max - data_min)

    def initialize_model(self):
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood)

    def train_model(self, iters=20):
        self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(iters=iters)

    def plot_losses(self):
        plt.plot(self.losses, label='training loss')
        plt.plot(self.losses_valid, label='validation loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss on Logarithmic Scale')
        plt.savefig(os.path.join(self.output_dir, 'loss_plot.png'))
        plt.show()

    def evaluate_model(self):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            self.observed_pred = self.likelihood(self.model(self.x_test))
            mean = self.observed_pred.mean.detach().reshape(-1, 1)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float32
            var = self.observed_pred.variance.detach().reshape(-1, 1)
            thr = torch.Tensor([0.]).to(device)

            self.entropy = entropy_local(mean, var, thr, device, dtype)

            print("Entropy: ", self.entropy)

    def plotGP(self, new_x=None, save_path=None):
        mean = self.observed_pred.mean.cpu().numpy()
        lower, upper = self.observed_pred.confidence_region()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()

        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(self.x_test.cpu().numpy(), mean, 'b', label='Learnt Function')
        ax.fill_between(self.x_test.cpu().numpy(), lower, upper, alpha=0.5, label='Confidence')
        
        x_true = torch.tensor(pd.read_csv(self.csv_file)['IN_M_1'].values[:], dtype=torch.float32)
        x_true_min = x_true.min(dim=0, keepdim=True).values
        x_true_max = x_true.max(dim=0, keepdim=True).values
        x_true = (x_true - x_true_min) / (x_true_max - x_true_min)
        y_true = torch.log(torch.tensor(pd.read_csv(self.csv_file)['MO_Omega'].values[:], dtype=torch.float32) / 0.12)
        
        ax.plot(x_true.cpu().numpy(), y_true.cpu().numpy(), '*', c="r", label='Truth')
        ax.plot(self.x_train.cpu().numpy(), self.y_train.cpu().numpy(), 'k*', label='Training Data')

        if new_x is not None:
            dolabel = True
            for xval in new_x:
                ax.axvline(x=xval, color='r', linestyle='--', label='new points') if dolabel else ax.axvline(x=xval, color='r', linestyle='--')
                dolabel = False

        ax2 = ax.twinx()
        ax2.set_ylabel("entropy")
        ax2.plot(self.x_test.cpu().numpy(), self.entropy.cpu().numpy(), 'g', label='Entropy')

        maxE = torch.max(self.entropy)
        maxIndex = torch.argmax(self.entropy)
        maxX = self.x_test[maxIndex]
        ax2.plot(maxX.cpu().numpy(), maxE.cpu().numpy(), 'go', label='Max. E')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)
        # ax.legend(lines, labels)

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

# Usage
output_dir = '/raven/u/dvoss/al_pmssmwithgp/model/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("")

gp_pipeline = GPModelPipeline(csv_file='M_2_fixed.csv', output_dir=output_dir)
gp_pipeline.train_model(iters=1000)
gp_pipeline.plot_losses()
gp_pipeline.evaluate_model()
gp_pipeline.plotGP(save_path=os.path.join(output_dir, 'gp_plot.png'))
