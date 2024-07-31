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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.likelihood = GaussianLikelihood().to(self.device)
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
        self.x_test = torch.linspace(0, 1, 1000).to(self.device)
        
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

        self.x_train = torch.tensor(M_1_limited.values[:self.n_train], dtype=torch.float32).to(self.device)
        self.y_train = torch.log(torch.tensor(Omega_limited.values[:self.n_train], dtype=torch.float32) / 0.12).to(self.device)
        self.x_valid = torch.tensor(M_1_limited.values[self.n_train:], dtype=torch.float32).to(self.device)
        self.y_valid = torch.log(torch.tensor(Omega_limited.values[self.n_train:], dtype=torch.float32) / 0.12).to(self.device)

        self.x_train = self._normalize(self.x_train)
        self.x_valid = self._normalize(self.x_valid)

    def _normalize(self, data):
        data_min = data.min(dim=0, keepdim=True).values
        data_max = data.max(dim=0, keepdim=True).values
        return (data - data_min) / (data_max - data_min)

    def initialize_model(self):
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood).to(self.device)

    def train_model(self, iters=20):
        self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(iters=iters)
        # Save the state dictionary of the best model
        torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, 'best_multitask_gp.pth'))

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

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            self.observed_pred = self.likelihood(self.model(self.x_test))
            mean = self.observed_pred.mean.detach().reshape(-1, 1).to(self.device)
            var = self.observed_pred.variance.detach().reshape(-1, 1).to(self.device)
            thr = torch.Tensor([0.]).to(self.device)

            self.entropy = entropy_local(mean, var, thr, self.device, torch.float32)

            print("Entropy: ", self.entropy)

    def plotGP(self, new_x=None, save_path=None):
        mean = self.observed_pred.mean.cpu().numpy()
        lower, upper = self.observed_pred.confidence_region()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()

        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(self.x_test.cpu().numpy(), mean, 'b', label='Learnt Function')
        ax.fill_between(self.x_test.cpu().numpy(), lower, upper, alpha=0.5, label='Confidence')
        
        x_true = torch.tensor(pd.read_csv(self.csv_file)['IN_M_1'].values[:], dtype=torch.float32).to(self.device)
        x_true_min = x_true.min(dim=0, keepdim=True).values
        x_true_max = x_true.max(dim=0, keepdim=True).values
        x_true = (x_true - x_true_min) / (x_true_max - x_true_min)
        y_true = torch.log(torch.tensor(pd.read_csv(self.csv_file)['MO_Omega'].values[:], dtype=torch.float32) / 0.12).to(self.device)
        
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

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def best_not_yet_chosen(self, score, previous_indices):
        candidates = torch.sort(score, descending=True)[1].to(self.device)
        for next_index in candidates:
            if int(next_index) not in previous_indices:
                return next_index

    def iterative_batch_selector(self, score_function, 
                                 choice_function=None,
                                 gp_mean=None, 
                                 gp_covar=None, 
                                 N=None):

        if gp_mean is None:
            def greedy_batch_sel(gp_mean, gp_covar, N):
                return self.iterative_batch_selector(score_function, choice_function, gp_mean, gp_covar, N)
            return greedy_batch_sel
        
        score = score_function(gp_mean[:, None], torch.diag(gp_covar)[:, None, None]).to(self.device)
        first_index = torch.argmax(score).to(self.device)
        indices = [int(first_index)]

        num_pts = len(gp_mean)

        for i in range(N-1):
            center_cov = torch.stack([gp_covar[indices, :][:, indices]] * num_pts).to(self.device)
            side_cov = gp_covar[:, None, indices].to(self.device)
            bottom_cov = gp_covar[:, indices, None].to(self.device)
            end_cov = torch.diag(gp_covar)[:, None, None].to(self.device)

            cov_batch = torch.cat([
                torch.cat([center_cov, side_cov], axis=1),
                torch.cat([bottom_cov, end_cov], axis=1),
            ], axis=2).to(self.device)
            
            center_mean = torch.stack([gp_mean[indices]] * num_pts).to(self.device)
            new_mean = gp_mean[:, None].to(self.device)

            mean_batch = torch.cat([center_mean, new_mean], axis=1).to(self.device)
            score = score_function(mean_batch, cov_batch).to(self.device)
            next_index = choice_function(score, indices)
            indices.append(int(next_index))

        return indices

    def approximate_batch_entropy(self, mean, cov):
        device = self.device
        n = mean.shape[-1]
        d = torch.diag_embed(1. / mean).to(device)
        x = d @ cov @ d.to(device)
        I = torch.eye(n)[None, :, :].to(device)
        return (torch.logdet(x + I) - torch.logdet(x + 2 * I) + n * np.log(2)) / np.log(2)

    def smoothed_batch_entropy(self, blur):
        return lambda mean, cov: self.approximate_batch_entropy(mean + blur * torch.sign(mean).to(self.device), cov)

    def gibbs_sample(self, beta):
        def sampler(score, indices=None):
            probs = torch.exp(beta * (score - torch.max(score))).to(self.device)
            probs /= torch.sum(probs).to(self.device)
            cums = torch.cumsum(probs, dim=0).to(self.device)
            rand = torch.rand(size=(1,)).to(self.device)[0]
            return torch.sum(cums < rand).to(self.device)    
        return sampler

    def select_new_points(self, N=4):
        selector = self.iterative_batch_selector(
            score_function=self.smoothed_batch_entropy(blur=0.15),
            choice_function=self.gibbs_sample(beta=50)
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            mean = self.observed_pred.mean.detach().to(self.device)
            covar = self.observed_pred.covariance_matrix.detach().to(self.device)
            thr = torch.Tensor([0.]).to(self.device)

            points = set(selector(N=N, gp_mean=mean - thr, gp_covar=covar))
            new_x = self.x_test[list(points)]
        
        print("points:", points)
        print("Corresponding xs:", new_x)
        return new_x


# Usage
output_dir = '/raven/u/dvoss/al_pmssmwithgp/model/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gp_pipeline = GPModelPipeline(csv_file='M_2_fixed.csv', output_dir=output_dir)
gp_pipeline.train_model(iters=1000)
gp_pipeline.plot_losses()
gp_pipeline.evaluate_model()
new_points = gp_pipeline.select_new_points(N=4)
gp_pipeline.plotGP(new_x=new_points, save_path=os.path.join(output_dir, 'gp_plot.png'))
