from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from scipy.stats import multivariate_normal
import argparse
import os
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from multitaskGPnD import MultitaskGP
from scipy.stats import qmc
import scipy.stats
import itertools
import linear_operator
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='GP Model Pipeline')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

class GPModelPipeline:
    def __init__(self, output_dir=None, initial_train_points=10, valid_points=20, additional_points_per_iter=1, n_dim=4, threshold=0.5, num_layers=3, n_blobs=2):
        self.output_dir = output_dir
        self.initial_train_points = initial_train_points
        self.valid_points = valid_points
        self.additional_points_per_iter = additional_points_per_iter
        
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
        self.n_dim = n_dim
        self.num_layers = num_layers
        self.thr = threshold
        self.n_blobs = n_blobs
        self.contour = None
        

        # Create a regular grid mesh for x_test
        self.n_points_per_dim = int(np.ceil(2000 ** (1 / self.n_dim)))
        grid = np.linspace(0, 1, self.n_points_per_dim)
        mesh = np.meshgrid(*[grid] * self.n_dim)
        points = np.vstack(list(map(np.ravel, mesh))).T
        self.x_test = torch.tensor(points, dtype=torch.float32).to(self.device)

        # TODO: Create this in a more random way with more blobs combining

        def create_truth_function():
            # Needs to be able to return the individual means and covs
            mean = []
            cov = []
            mean1 = [0.48,0.48]
            cov1 = np.diag([0.02,0.02])
            scale1 = 0.02
            mean2 = [0.8,0.8]
            cov2 = np.diag([0.03,0.03])
            scale2 = 0.05
            mean.append(mean1)
            mean.append(mean2)
            cov.append(cov1) 
            cov.append(cov2) 
            def gauss1(x):
                return scale1 * multivariate_normal.pdf(x, mean=mean1, cov=cov1)
            def gauss2(x):
                return scale2 * multivariate_normal.pdf(x, mean=mean2, cov=cov2)
            def combined_gaussians(x):
                return gauss1(x) + gauss2(x)

            return mean, cov, combined_gaussians
        # def create_truth_function(n_dim):
        #     means = []
        #     covs = []
        #     gaussians = []
        #     for i in range(self.n_blobs):
        #         mean = np.random.uniform(0, 1, n_dim)
        #         cov = np.diag(np.random.uniform(0.01, 0.1, n_dim))

        #         means.append(mean)
        #         covs.append(cov)

        #         gaussians.append(lambda X: np.atleast_1d(multivariate_normal.pdf(X, mean=mean, cov=cov)))
        #     return lambda X: sum(gaussian(X) for gaussian in gaussians)
        
            """ 
            mean1 = [0.3] * n_dim  # Place one Gaussian at (0.2, 0.2, ...)
            cov1 = np.diag([0.03] * n_dim)  # Smaller covariance for tighter circles
            mean2 = [0.7] * n_dim  # Place the second Gaussian at (0.8, 0.8, ...)
            cov2 = np.diag([0.03] * n_dim)  # Smaller covariance for tighter circles

            gauss1 = lambda X: np.atleast_1d(multivariate_normal.pdf(X, mean=mean1, cov=cov1))
            gauss2 = lambda X: np.atleast_1d(multivariate_normal.pdf(X, mean=mean2, cov=cov2))
            two_gaussians = lambda X: gauss1(X) + gauss2(X)
            return two_gaussians """

        self.mean, self.cov, self.truth0 = create_truth_function()

        #self.truth0 = lambda X: np.atleast_1d(scipy.stats.multivariate_normal.pdf(X,mean = [0.5,0.5,0.5,0.5], cov = np.diag([0.2,0.3,0.2,0.3])))
        self.truth1 = lambda X: np.atleast_1d(scipy.stats.multivariate_normal.pdf(X,mean = [0.6,0.8,0.6,0.8], cov = np.diag([0.2,0.3,0.2,0.3])))

        self.load_initial_data()
        self.initialize_model()

    def load_initial_data(self):

        # Generate Latin Hypercube Sampling data between 0 and 1
        sampler = qmc.LatinHypercube(d=self.n_dim)
        self.x_train = torch.tensor(sampler.random(n=self.initial_train_points), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.truth0(self.x_train.cpu()), dtype=torch.float32).to(self.device)
        self.x_valid = torch.tensor(sampler.random(n=self.valid_points), dtype=torch.float32).to(self.device)
        self.y_valid = torch.tensor(self.truth0(self.x_valid.cpu()), dtype=torch.float32).to(self.device)
        
        print("Initial Training Points:")
        print(f"X Train: {self.x_train}, Shape: {self.x_train.shape}")
        print(f"Y Train: {self.y_train}, Shape: {self.y_train.shape}")


    def load_additional_data(self, new_x):
    
        # Load new training points and evaluate truth function
        additional_x_train = torch.tensor(new_x).to(self.device)
        additional_y_train = torch.tensor(self.truth0(additional_x_train.cpu()), dtype=torch.float32).to(self.device)

        # Debugging: Print shapes of both tensors before concatenating
        print(f"self.x_train shape: {self.x_train.shape}")  # Expecting [N, 2]
        print(f"additional_x_train shape: {additional_x_train.shape}")  # Should also be [M, 2]

        print(f"additional_x_train: {additional_x_train}")
        print(f"additional_y_train: {additional_y_train}")
        
        # Append the new points to the existing training data
        self.x_train = torch.cat((self.x_train, additional_x_train))
        self.y_train = torch.cat((self.y_train, additional_y_train))

        # Combine x_train (which is nD) and y_train into tuples (x1, x2, ..., xn, y)
        combined_set = {tuple(x.tolist()) + (float(y.item()),) for x, y in zip(self.x_train, self.y_train)}

        # Unpack the combined_set into x_train and y_train
        x_train, y_train = zip(*[(x[:-1], x[-1]) for x in combined_set])

        # Convert the unpacked x_train and y_train back to torch tensors
        self.x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        print("Training points after adding: ", self.x_train, self.x_train.shape)


    def initialize_model(self):
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood, self.n_dim).to(self.device)
        # TODO: self.model = MultitaskGP(DeepGP(self.n_dim, self.num_layers))

    def train_model(self, iters=20):
        print("These training_points are used in the GP", self.x_train)
        self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(iters=iters)

        # Print the hyperparameters of the best model
        print("best model parameters: ", self.best_model.state_dict())


    def plot_losses(self, save_path=None, iteration=None):
        plt.figure(figsize=(8, 6))
        plt.plot(self.losses, label='training loss')
        plt.plot(self.losses_valid, label='validation loss')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"Log Training and Validation Loss - Iteration {iteration}")

        # Save the plot or show it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def evaluate_model(self):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            self.observed_pred = self.likelihood(self.model(self.x_test))
            print("Likelihood", self.observed_pred)
            mean = self.observed_pred.mean.detach().reshape(-1, 1).to(self.device)
            print("Mean: ", mean)
            var = self.observed_pred.variance.detach().reshape(-1, 1).to(self.device)
            print("Variance: ", var)
            thr = torch.Tensor([self.thr]).to(self.device)

    def plotEntropy(self, slice_dim_x1=0, slice_dim_x2=1, slice_value=0.5, tolerance=0.1, new_x=None, save_path=None, iteration=None):
        """
        Plot a 2D slice of the truth function.

        Parameters:
            truth0 (function): The truth function to evaluate.
            n_dim (int): The number of dimensions.
            slice_dim_x1 (int): The first dimension to plot.
            slice_dim_x2 (int): The second dimension to plot.
            slice_value (float): The value of the dimension to slice at.
            tolerance (float): The tolerance for the slice (e.g., +/- 0.01).
            new_x (torch.Tensor): New points to highlight on the plot (optional).
            save_path (str): The path to save the plot, if specified.
            iteration (int): The current iteration number, if any (for the plot title).
        """

        # Determine the remaining dimensions to plot
        remaining_dims = [dim for dim in range(self.n_dim) if dim != slice_dim_x1 and dim != slice_dim_x2]

        # Create a grid over the remaining dimensions
        grid_size = 50
        x1_range = np.linspace(0, 1, grid_size)
        x2_range = np.linspace(0, 1, grid_size)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        x1_grid_flat = x1_grid.flatten()
        x2_grid_flat = x2_grid.flatten()

        # Create a tensor for the grid points
        x_test = np.zeros((grid_size * grid_size, self.n_dim))
        x_test[:, slice_dim_x1] = x1_grid_flat 
        x_test[:, slice_dim_x2] = x2_grid_flat 

        # Set the remaining dimensions to the slice value
        for dim in remaining_dims:
            x_test[:, dim] = slice_value

        # Turn the grid points into a torch tensor
        x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(x_test)
        
        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()
        cov = observed_pred.covariance_matrix.cpu().detach().numpy()

        # Ensure mean and cov are tensors
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(cov, torch.Tensor):
            cov = torch.tensor(cov)

        blur = 0.15
        # entropy = self.approximate_batch_entropy(mean[:, None] + blur * torch.sign(mean), torch.diag(cov)[:, None, None])
        mean_adjusted = mean - self.thr + blur * torch.sign(mean)
        entropy = self.approximate_batch_entropy(mean_adjusted[:, None], torch.diag(cov)[:, None, None])

        # Reshape the mean predictions to match the grid
        entropy_grid = entropy.reshape(grid_size, grid_size)

        plt.figure(figsize=(8, 6))
        plt.imshow(entropy_grid, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Entropy')

        # Labels for the dimensions
        labels = {
            0: "M_1_normalized",
            1: "M_2_normalized",
            2: "M_3_normalized",
            3: "tanb_normalized",
            4: "mu_normalized",
            5: "AT_normalized",
            6: "Ab_normalized",
            7: "Atau_normalized",
            8: "mA_normalized",
            9: "meL_normalized",
            10: "mtauL_normalized",
            11: "meR_normalized",
            12: "mtauR_normalized",
            13: "mqL1_normalized",
            14: "mqL3_normalized",
            15: "muR_normalized",
            16: "mtR_normalized",
            17: "mdR_normalized",
            18: "mbR_normalized"
        }
        plt.xlabel(labels[slice_dim_x1])
        plt.ylabel(labels[slice_dim_x2])

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"GP Prediction Slice - Iteration {iteration}")

        # Save the plot or show it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plotTrue(self, slice_dim_x1=0, slice_dim_x2=1, slice_value=0.5):
        '''Plot the 2D GP with a Heatmap and the new points and save it in the plot folder'''
        print(f"Shape of x_test: {self.x_test.shape}")

        # Determine the remaining dimensions to plot
        remaining_dims = [dim for dim in range(self.n_dim) if dim != slice_dim_x1 and dim != slice_dim_x2]

        # Create a grid over the remaining dimensions
        grid_size = 50
        x1_range = np.linspace(0, 1, grid_size)
        x2_range = np.linspace(0, 1, grid_size)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        x1_grid_flat = x1_grid.flatten()
        x2_grid_flat = x2_grid.flatten()

        # Create a tensor for the grid points
        x_test = np.zeros((grid_size * grid_size, self.n_dim))
        x_test[:, slice_dim_x1] = x1_grid_flat 
        x_test[:, slice_dim_x2] = x2_grid_flat 

        # Set the remaining dimensions to the slice value
        for dim in remaining_dims:
            x_test[:, dim] = slice_value

        # Turn the grid points into a torch tensor
        x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)

        # Disable gradient computation for evaluation
        truth = self.truth0(x_test)

        truth_grid = truth.reshape(grid_size, grid_size)

        plt.figure(figsize=(8, 6))
        plt.imshow(truth_grid.T, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Truth')
        # Labels for the dimensions
        labels = {
            0: "M_1_normalized",
            1: "M_2_normalized",
            2: "M_3_normalized",
            3: "tanb_normalized",
            4: "mu_normalized",
            5: "AT_normalized",
            6: "Ab_normalized",
            7: "Atau_normalized",
            8: "mA_normalized",
            9: "meL_normalized",
            10: "mtauL_normalized",
            11: "meR_normalized",
            12: "mtauR_normalized",
            13: "mqL1_normalized",
            14: "mqL3_normalized",
            15: "muR_normalized",
            16: "mtR_normalized",
            17: "mdR_normalized",
            18: "mbR_normalized"
        }
        plt.xlabel(labels[slice_dim_x1])
        plt.ylabel(labels[slice_dim_x2])

        # Contour plot of the truth function
        plt.contour(x1_grid, x2_grid, truth_grid, levels=[self.thr], colors='white', linewidths=2, linestyles='solid')

        # Add the iteration number to the plot title
        plt.show()

    
    def plotSlice2D(self, slice_dim_x1=0, slice_dim_x2=1, slice_value=0.5, tolerance=0.1, new_x=None, save_path=None, iteration=None):
        """
        Plot a 2D slice of the truth function.

        Parameters:
            truth0 (function): The truth function to evaluate.
            n_dim (int): The number of dimensions.
            slice_dim_x1 (int): The first dimension to plot.
            slice_dim_x2 (int): The second dimension to plot.
            slice_value (float): The value of the dimension to slice at.
            tolerance (float): The tolerance for the slice (e.g., +/- 0.01).
            new_x (torch.Tensor): New points to highlight on the plot (optional).
            save_path (str): The path to save the plot, if specified.
            iteration (int): The current iteration number, if any (for the plot title).
        """

        # Determine the remaining dimensions to plot
        remaining_dims = [dim for dim in range(self.n_dim) if dim != slice_dim_x1 and dim != slice_dim_x2]

        # Create a grid over the remaining dimensions
        grid_size = 50
        x1_range = np.linspace(0, 1, grid_size)
        x2_range = np.linspace(0, 1, grid_size)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        x1_grid_flat = x1_grid.flatten()
        x2_grid_flat = x2_grid.flatten()

        # Create a tensor for the grid points
        x_test = np.zeros((grid_size * grid_size, self.n_dim))
        x_test[:, slice_dim_x1] = x1_grid_flat 
        x_test[:, slice_dim_x2] = x2_grid_flat 

        # Set the remaining dimensions to the slice value
        for dim in remaining_dims:
            x_test[:, dim] = slice_value

        # Turn the grid points into a torch tensor
        x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(x_test)
        
        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()

        # Reshape the mean predictions to match the grid
        mean_grid = mean.reshape(grid_size, grid_size)

        vmin = np.min(mean)
        vmax = np.max(mean)

        plt.figure(figsize=(8, 6))
        plt.imshow(mean_grid.T, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(label='GP Prediction Mean')

        # Labels for the dimensions
        labels = {
            0: "M_1_normalized",
            1: "M_2_normalized",
            2: "M_3_normalized",
            3: "tanb_normalized",
            4: "mu_normalized",
            5: "AT_normalized",
            6: "Ab_normalized",
            7: "Atau_normalized",
            8: "mA_normalized",
            9: "meL_normalized",
            10: "mtauL_normalized",
            11: "meR_normalized",
            12: "mtauR_normalized",
            13: "mqL1_normalized",
            14: "mqL3_normalized",
            15: "muR_normalized",
            16: "mtR_normalized",
            17: "mdR_normalized",
            18: "mbR_normalized"
        }
        plt.xlabel(labels[slice_dim_x1])
        plt.ylabel(labels[slice_dim_x2])

        # Create a filtered version of the training data
        filtered_x_train = self.x_train.clone()
        filtered_y_train = self.y_train.clone()

        # Filter the training data based on the slice value and tolerance in the remaining dimensions
        for dim in remaining_dims:
            indices = (filtered_x_train[:, dim].cpu().numpy() >= slice_value - tolerance) & (filtered_x_train[:, dim].cpu().numpy() <= slice_value + tolerance)
            filtered_x_train = filtered_x_train[indices, :]
            filtered_y_train = filtered_y_train[indices]
            indices_new = (new_x[:, dim].cpu().numpy() >= slice_value - tolerance) & (new_x[:, dim].cpu().numpy() <= slice_value + tolerance)
            filtered_new_x = new_x[indices_new, :]
        
        # Scatterplot of the training points
        plt.scatter(filtered_x_train[:, slice_dim_x1].numpy(), filtered_x_train[:, slice_dim_x2].numpy(), marker='o', s=50, c=filtered_y_train.numpy(), cmap='inferno', vmin=vmin, vmax=vmax, label='training points')
        
        # Calculate the contour of the mean grid as the 90% percentile and safe in variable
        self.contour = np.percentile(mean_grid, 90)
        print("Contour in Plot function: ",self.contour)
        plt.contour(x1_grid, x2_grid, mean_grid, levels=[np.percentile(mean_grid, 90)], colors='white', linewidths=2, linestyles='solid')

        # Plot new points if provided
        if new_x is not None:
            plt.scatter(filtered_new_x[:, slice_dim_x1].numpy(), filtered_new_x[:, slice_dim_x2].numpy(), marker='*', s=100, c='r', label='new training points')

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"GP Prediction Slice - Iteration {iteration}")

        # Save the plot or show it
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
        '''Chooses the next points iterativley. The point with maximum entropy is always chosen first, 
            then the next indices are selected with the choice function - gibbs_sampling or best_not_yet_chosen
            The covariance matrix and the mean vector are updated iterativly, based on the already chosen points
        '''

        if gp_mean is None:
            def greedy_batch_sel(gp_mean, gp_covar, N):
                return self.iterative_batch_selector(score_function, choice_function, gp_mean, gp_covar, N)
            return greedy_batch_sel
        
        score = score_function(gp_mean[:, None], torch.diag(gp_covar)[:, None, None]).to(self.device)
        #print("Smoothed_batch_entropy: ", score)
        self.entropy = score
        print("Entropy: ", self.entropy)
        first_index = torch.argmax(score).to(self.device)
        indices = [int(first_index)]

        num_pts = len(gp_mean)

        for i in range(N-1):
            center_cov = torch.stack([gp_covar[indices, :][:, indices]] * num_pts).to(self.device) # Covariance between selected points
            side_cov = gp_covar[:, None, indices].to(self.device) # Coveriance between already selected and remaining points
            bottom_cov = gp_covar[:, indices, None].to(self.device)
            end_cov = torch.diag(gp_covar)[:, None, None].to(self.device)

            cov_batch = torch.cat([
                torch.cat([center_cov, side_cov], axis=1),
                torch.cat([bottom_cov, end_cov], axis=1),
            ], axis=2).to(self.device)

            # print("Cov_batch Shape: ", cov_batch.shape)
            # print("center_cov Shape: ", center_cov.shape)
            # print("side_cov Shape: ", side_cov.shape)
            # print("bottom_cov Shape: ", bottom_cov.shape)
            # print("end_cov Shape: ", end_cov.shape)
            
            center_mean = torch.stack([gp_mean[indices]] * num_pts).to(self.device) # Mean between selected points
            # TODO: Sampling approach for a sampling mean
            # new_mean = torch.normal(mean=gp_mean, std=gp_covar.sqrt())
            new_mean = gp_mean[:, None].to(self.device) # Reshapes gp_mean to a column vector of shape (num_pts, 1)
            mean_batch = torch.cat([center_mean, new_mean], axis=1).to(self.device)

            # print("Mean_batch Shape: ", mean_batch.shape)
            # print("center_mean Shape: ", center_mean.shape)
            # print("new_mean Shape: ", new_mean.shape)

            score = score_function(mean_batch, cov_batch).to(self.device)
            next_index = choice_function(score, indices)
            indices.append(int(next_index))


        return indices

    def approximate_batch_entropy(self, mean, cov):
        device = self.device
        n = mean.shape[-1]
        d = torch.diag_embed(1. / mean).to(device)
        x = d @ cov @ d.to(device)
        # print("Cov Shape: ", cov.shape)
        # print("X Shape: ", x.shape)
        I = torch.eye(n)[None, :, :].to(device)
        return (torch.logdet(x + I) - torch.logdet(x + 2 * I) + n * np.log(2)) / np.log(2)

    def smoothed_batch_entropy(self, blur):
        return lambda mean, cov: self.approximate_batch_entropy(mean + blur * torch.sign(mean).to(self.device), cov)

    def gibbs_sample(self, beta):
        '''Chooses next point based on probability that a random value is smaller than the cumulative sum 
            of the probabilites. These are calculated with the smoothed batch entropy
            - if beta is high: deterministic selection with only highest entropies
            - if beta is low: more random selection with each point having a similar probility
        '''
        def sampler(score, indices=None):
            probs = torch.exp(beta * (score - torch.max(score))).to(self.device)
            probs /= torch.sum(probs).to(self.device)
            # print("probs:", probs)
            cums = torch.cumsum(probs, dim=0).to(self.device)
            # print("cumsum: ", cums)
            rand = torch.rand(size=(1,)).to(self.device)[0]
            # print("rand: ", rand)
            return torch.sum(cums < rand).to(self.device)    
        return sampler

    def select_new_points(self, N=4):
        # Initialize the selector using a combination of a smoothed batch entropy score function and a Gibbs sampling choice function.
        selector = self.iterative_batch_selector(
            score_function=self.smoothed_batch_entropy(blur=0.15),
            choice_function=self.gibbs_sample(beta=50)
            # choice_function=self.best_not_yet_chosen
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            # Initialize a random x_test_rand to be able to choose new points out of the whole parameter space
            # # # Completly random 
            # x_test_rand = torch.rand(self.x_test.shape[0], 2).to(self.device)
            # Latin Hypercube
            x_test_rand = torch.tensor(qmc.LatinHypercube(d=self.n_dim).random(n=self.x_test.shape[0]), dtype=torch.float32).to(self.device)

            # Evaluate the model
            self.model.eval()

            # Disable gradient computation for evaluation
            with torch.no_grad():
                # Get predictions of model for random test points
                predictions = self.model(x_test_rand)
                observed_pred = self.likelihood(predictions)

            mean = observed_pred.mean.detach().to(self.device)
            covar = observed_pred.covariance_matrix.detach().to(self.device)
            thr = torch.Tensor([self.thr]).to(self.device) 

            # Use the selector to choose a set of points based on the  mean and covariance matrix.
            points = set(selector(N=N, gp_mean=mean - thr, gp_covar=covar))  
            new_x = x_test_rand[list(points)]
            
        # Debugging and informational prints to check the selected points and their corresponding values.
        print("Selected points (indices):", points)
        print("Selected new x values (normalized):", new_x)
        
        return new_x
    
    def random_new_points(self, N=4):

        # Define the number of dimensions
        dimensions = self.n_dim  # Dimensionality of the space

        # Generate Latin Hypercube Samples in the range [0, 1] for all dimensions
        sampler = qmc.LatinHypercube(d=dimensions)  # Create a Latin Hypercube sampler
        lhs_samples = sampler.random(n=N)  # Generate N samples in the range [0, 1)

        # Convert the samples to a PyTorch tensor
        lhs_samples = torch.tensor(lhs_samples, dtype=torch.float32).to(self.device)

        # Return the normalized points (guaranteed to be in [0, 1])
        return lhs_samples
    
    def goodness_of_fit(self, test=None, csv_path=None):
        
        # create empty list to store the goodness of fit metrics
        accuracy_all = []

        for blob in range(self.n_blobs):
            '''
            For each blob, evaluate the model at test points concentrated around the contour for each gaussian blob.
            Get the test_points for each blob from the n_blob function of the create_truth function.
            '''
            # Evaluate model at test points concentrated around the contour for each gaussian blob
            # Sample random points around mean of corresponding blob with covariance matrix
            mean_tensor = torch.tensor(self.mean[blob], dtype=torch.float32).to(self.device)
            cov_tensor = torch.tensor(self.cov[blob], dtype=torch.float32).to(self.device)
            mvn = torch.distributions.MultivariateNormal(mean_tensor, covariance_matrix=cov_tensor/1.5)

            # Take more points for the bigger circle and less for the smaller one
            if blob == 0:
                input_data = mvn.sample((1000,)).float()
            else:
                input_data = mvn.sample((2000,)).float() 

            # Normalize the input data to be in the range [0, 1]
            mask = (input_data >= 0) & (input_data <= 1)
            mask = mask.all(axis=1)
            input_data = input_data[mask]

            # Calculate the true values (log-scaled)
            truth_values = self.truth0(input_data.cpu().numpy())  

            # Convert back to torch tensors
            true = torch.tensor(truth_values, dtype=torch.float32).to(self.device)  

            self.model.eval()

            # Disable gradient computation for evaluation
            with torch.no_grad():
                predictions = self.model(input_data)

            observed_pred = self.likelihood(predictions)
            mean = observed_pred.mean.cpu().numpy()
            lower, upper = observed_pred.confidence_region()
            lower = lower.detach().cpu().numpy()
            upper = upper.detach().cpu().numpy()

            mean = torch.tensor(mean, dtype=torch.float32).to(self.device)
            upper = torch.tensor(upper, dtype=torch.float32).to(self.device)
            lower= torch.tensor(lower, dtype=torch.float32).to(self.device)

            # print(f"The shape of the mean is: {mean.shape}")

            # # Define the various goodness_of_fit metrics
            # def mean_squared():
            #     mse = torch.mean((mean - true) ** 2)
            #     rmse = torch.sqrt(mse)
            #     return mse.cpu().item(), rmse.cpu().item()
            
            # def r_squared():
            #     ss_res = torch.sum((mean - true) ** 2)
            #     ss_tot = torch.sum((true - torch.mean(true)) ** 2)
            #     r_squared = 1 - ss_res / ss_tot
            #     return r_squared.cpu().item()
            
            # def chi_squared():
            #     pull = (mean - true) / (upper - lower)
            #     chi_squared = torch.sum(pull ** 2)
            #     dof = len(true) - 1  # Degrees of freedom
            #     reduced_chi_squared = chi_squared / dof
            #     return chi_squared.cpu().item(), reduced_chi_squared.cpu().item()

            # def average_pull():
            #     absolute_pull = torch.abs((mean - true) / (upper - lower))
            #     mean_absolute_pull = torch.mean(absolute_pull)
            #     root_mean_squared_pull = torch.sqrt(torch.mean((absolute_pull) ** 2))
            #     return mean_absolute_pull.cpu().item(), root_mean_squared_pull.cpu().item()

            # # Weights for adjusting to threshold
            # thr = self.thr
            # epsilon = 0.1 # Tolerance area around threshold, considered close
            # weights = torch.exp(-((true - thr) ** 2) / (2 * epsilon ** 2))  

            # def mean_squared_weighted():
            #     mse_weighted = torch.mean(weights * (mean - true) ** 2)
            #     rmse_weighted = torch.sqrt(mse_weighted)
            #     return mse_weighted.cpu().item(), rmse_weighted.cpu().item()

            # def r_squared_weighted():
            #     ss_res_weighted = torch.sum(weights * (mean - true) ** 2)
            #     ss_tot_weighted = torch.sum(weights * (true - torch.mean(true)) ** 2)
            #     r_squared_weighted = 1 - ss_res_weighted / ss_tot_weighted
            #     return r_squared_weighted.cpu().item()
            
            # def chi_squared_weighted():
            #     pull_weighted = (mean - true) / (upper - lower)
            #     chi_squared_weighted = torch.sum(weights * pull_weighted ** 2)
            #     dof = len(true) - 1  # Degrees of freedom
            #     reduced_chi_squared_weighted = chi_squared_weighted / dof
            #     return chi_squared_weighted.cpu().item(), reduced_chi_squared_weighted.cpu().item()
            
            # def average_pull_weighted():
            #     absolute_pull_weighted = torch.abs((mean - true) / (upper - lower))
            #     mean_absolute_pull_weighted = torch.mean(weights * absolute_pull_weighted)
            #     root_mean_squared_pull_weighted = torch.sqrt(torch.mean(weights * (absolute_pull_weighted) ** 2))
            #     return mean_absolute_pull_weighted.cpu().item(), root_mean_squared_pull_weighted.cpu().item()

            # Calculate classification matrix and accuracy
            def accuracy():
        
                # # TODO: Classification with uncertainty    
                # mean_plus = mean + (upper - lower)/2
                # mean_minus = mean + (upper - lower)/2

                TP = ((mean > self.thr) & (true > self.thr)).sum().item()  # True Positives
                FP = ((mean > self.thr) & (true < self.thr)).sum().item()  # False Positives
                FN = ((mean < self.thr) & (true > self.thr)).sum().item()  # False Negatives
                TN = ((mean < self.thr) & (true < self.thr)).sum().item()  # True Negatives

                total = TP + FP + FN + TN

                print(f"Shapes of TP, FP, FN, TN: {TP}, {FP}, {FN}, {TN}")
                accuracy = (TP + TN) / total if total > 0 else 0

                return accuracy

            accuracy_all.append(accuracy())

        # Dictionary to map test names to functions
        tests = {
            # 'mean_squared': mean_squared,
            # 'r_squared': r_squared,
            # 'chi_squared': chi_squared,
            # 'average_pull': average_pull,
            # 'mean_squared_weighted': mean_squared_weighted,
            # 'r_squared_weighted': r_squared_weighted,
            # 'chi_squared_weighted': chi_squared_weighted,
            # 'average_pull_weighted': average_pull_weighted,
            'accuracy': accuracy
        }

        print("Accuracy_all:", accuracy_all)

        # Run all tests and store the results
        results = {}
        [results.update({f'accuracy_{i}': accuracy_all[i]}) for i in range(self.n_blobs)]
        # results['accuracy_1'] = accuracy_all[0]
        # results["accuracy_2"] = accuracy_all[1]
        # for test_name, test_function in tests.items():
        #     result = test_function()
        #     if isinstance(result, tuple):
        #         for i, value in enumerate(result):
        #             results[f"{test_name}_{i}"] = value  # Store each element of the tuple separately
        #     else:
        #         results[test_name] = result  # Store single value directly

        # Save results to CSV if a path is provided
        if csv_path is not None:
            file_exists = os.path.isfile(csv_path)
            df_results = pd.DataFrame([results])  # Convert results to a DataFrame
            
            # Append data to CSV file, write header only if file does not exist
            df_results.to_csv(csv_path, mode='a', header=not file_exists, index=False)

        return results

        # Run a specific test
        if test in tests:
            result = tests[test]()
            print(f"Goodness of Fit ({test}): {result}")
            return result
        else:
            raise ValueError(f"Unknown test: {test}.")
        


    # Utility functions
    def save_training_data(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.x_train, self.y_train), f)
    
    def load_training_data(self, filepath):
        with open(filepath, 'rb') as f:
            self.x_train, self.y_train = pickle.load(f)  

    def save_model(self, model_checkpoint_path):
        torch.save(self.model.state_dict(), model_checkpoint_path)

    def load_model(self, model_checkpoint_path):
        # Initialize the model architecture
        self.initialize_model()
        
        # Ensure the model loads on the correct device (GPU or CPU)
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved state dict
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=map_location))
        print(f"Model loaded from {model_checkpoint_path}")





def run_pipeline_without_active_learning(args):

    # Active Learning configuration
    total_iterations = 750
    total_points = 100
    output_base_dir = '/u/dvoss/al_pmssmwithgp/model/plots/plots_two_circles'
    n_dim = 2
    name = f"two_circles{n_dim}D_{total_points}"
    threshold = 1
    isIteration = False

    if isIteration:    
        # Loop through iterations 1 to `total_iterations`
        for iteration in range(1, total_iterations + 1):

            print(f"Starting Iteration {args.iteration} without Active Learning")

            # Initialize the pipeline
            gp_pipeline = GPModelPipeline(
                output_dir=args.output_dir, initial_train_points=iteration, valid_points=300, n_dim=n_dim, threshold=threshold
            )

            # Initialize, train, and evaluate the model
            gp_pipeline.initialize_model()
            gp_pipeline.train_model(iters=1000)
            gp_pipeline.plot_losses(save_path=os.path.join(args.output_dir, f'loss_plot_{name}.png'),
                        iteration=args.iteration)
            gp_pipeline.evaluate_model()

            # Generate all possible combinations of two dimensions from n_dim for plotting
            combinations = list(itertools.combinations(range(n_dim), 2))

            gp_pipeline.goodness_of_fit(csv_path=f'{output_base_dir}/gof_{name}_rand.csv')
            # for (slice_dim_x1, slice_dim_x2) in combinations:
            #     gp_pipeline.plotSlice2D(
            #         slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, slice_value=0.5,
            #         save_path=os.path.join(output_dir, f'gp_plot_{slice_dim_x1}_{slice_dim_x2}_{name}_rand.png'),
            #         iteration=iteration
            #     )
            gp_pipeline.save_model(os.path.join(args.output_dir, f'model_checkpoint_{name}_rand.pth'))

            print(f"Iteration {args.iteration} completed. Output saved to {args.output_dir}")
    else:

        # Initialize the pipeline
        gp_pipeline = GPModelPipeline(
            output_dir=args.output_dir, initial_train_points=total_points, valid_points=30, n_dim=n_dim, threshold=threshold
        )

        # Initialize, train, and evaluate the model
        gp_pipeline.initialize_model()
        gp_pipeline.train_model(iters=250)
        gp_pipeline.plot_losses(save_path=os.path.join(args.output_dir, f'loss_plot_{name}.png'))
        gp_pipeline.evaluate_model()

        # Generate all possible combinations of two dimensions from n_dim for plotting
        combinations = list(itertools.combinations(range(n_dim), 2))

        for (slice_dim_x1, slice_dim_x2) in combinations:
            gp_pipeline.plotSlice2D(
                slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, slice_value=0.5,
                save_path=os.path.join(args.output_dir, f'gp_plot_{slice_dim_x1}_{slice_dim_x2}_{name}_rand.png')
            )
        gp_pipeline.goodness_of_fit(csv_path=f'{output_base_dir}/gof_{name}_rand.csv')
        gp_pipeline.save_model(os.path.join(args.output_dir, f'model_checkpoint_{name}_rand.pth'))

def run_pipeline_with_active_learning(args):

    # Active Learning configuration
    #total_iterations = 3000
    output_base_dir = '/u/dvoss/al_pmssmwithgp/model/plots/plots_two_circles'
    n_dim = 2
    name = f"two_circles{n_dim}D_raven"
    threshold = 1
    al_points = 6

    # Loop through iterations 1 to `total_iterations`
    # output_dir = os.path.join(output_base_dir, f'Iter{iteration}')
    # os.makedirs(output_dir, exist_ok=True)

    previous_iter = args.iteration - 1
    previous_output_dir = f'/raven/u/dvoss/al_pmssmwithgp/model/plots/plots_two_circles/Iter{previous_iter}'

    training_data_path = os.path.join(previous_output_dir, 'training_data.pkl')

    print(f"Starting Iteration {args.iteration} with Active Learning")

    # Initialize the pipeline
    if args.iteration == 1:
        gp_pipeline = GPModelPipeline(
            output_dir=args.output_dir, initial_train_points=1, valid_points=30, additional_points_per_iter=al_points, n_dim=n_dim, threshold=threshold
        )
    else:
        gp_pipeline = GPModelPipeline(
            output_dir=args.output_dir, initial_train_points=1, valid_points=30, additional_points_per_iter=al_points, n_dim=n_dim, threshold=threshold
        )
        # Load previous data and add new training points
        if os.path.exists(training_data_path):
            gp_pipeline.load_training_data(training_data_path)
        

    # Initialize, train, and evaluate the model
    with linear_operator.settings.max_cg_iterations(2000):
        gp_pipeline.initialize_model()
        gp_pipeline.train_model(iters=1000)
        gp_pipeline.plot_losses(save_path=os.path.join(args.output_dir, f'loss_plot_{name}.png'),
                    iteration=args.iteration)
        gp_pipeline.evaluate_model()

        # Generate all possible combinations of two dimensions from n_dim for plotting
        combinations = list(itertools.combinations(range(n_dim), 2))

        new_points = gp_pipeline.select_new_points(N=al_points)
        gp_pipeline.goodness_of_fit(csv_path=f'{output_base_dir}/gof_{name}.csv')
        # for (slice_dim_x1, slice_dim_x2) in combinations:
        #     gp_pipeline.plotSlice2D(
        #         slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, slice_value=0.5,
        #         new_x=new_points,
        #         save_path=os.path.join(args.output_dir, f'gp_plot_{slice_dim_x1}_{slice_dim_x2}_{name}.png'),
        #         iteration=args.iteration
        #     )
        #     gp_pipeline.plotEntropy(
        #         slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, slice_value=0.5,
        #         new_x=new_points,
        #         save_path=os.path.join(args.output_dir, f'entropy_{slice_dim_x1}_{slice_dim_x2}_{name}.png'),
        #         iteration=args.iteration
        #     )
        gp_pipeline.load_additional_data(new_points)
        gp_pipeline.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'))
        gp_pipeline.save_model(os.path.join(args.output_dir, f'model_checkpoint_{name}.pth'))

        print(f"Iteration {args.iteration} completed. Output saved to {args.output_dir}")


if __name__ == "__main__":
    isActiveLearning = True

    args = parse_args()

    if isActiveLearning:
        run_pipeline_with_active_learning(args)
    else:
        run_pipeline_without_active_learning(args)


    # for version in range(1,10):
    #     if isActiveLearning:
    #         run_pipeline_with_active_learning(version)
    #     else:
    #         run_pipeline_without_active_learning(version)



