import uproot
import argparse
import os
import io
import torch
import gpytorch
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood
import pandas as pd
import numpy as np
from entropy import entropy_local  
from create_confignD import create_config
from multitaskGPnD import MultitaskGP
from scipy.stats import qmc
import itertools




def parse_args():
    parser = argparse.ArgumentParser(description='GP Model Pipeline')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

class GPModelPipeline:
    def __init__(self, start_root_file_path=None, output_dir=None, initial_train_points=500, valid_points=1000, additional_points_per_iter=500, n_test=100000, n_dim=4, threshold=0.05):
        self.start_root_file_path = start_root_file_path
        self.output_dir = output_dir
        self.initial_train_points = initial_train_points
        self.valid_points = valid_points
        self.additional_points_per_iter = additional_points_per_iter
        self.n_test = n_test
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)
        
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = None
        self.best_model = None
        self.losses = None
        self.losses_valid = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_all = None
        self.y_all = None
        self.observed_pred = None
        self.entropy = None
        self.n_dim = n_dim
        self.thr = threshold
        self.conf_matrix = None

        # Define data_min and data_max according to pMSSM parameter ranges and the number of dimensions
        self.data_min = torch.tensor([-2000, -2000, 1, -2000, 1000, -8000, -2000, -2000, 0, 2000, 2000, 2000], dtype=torch.float32).to(self.device)
        self.data_max = torch.tensor([2000, 2000, 60, 2000, 5000, 8000, 2000, 2000, 5000, 5000, 5000, 5000 ], dtype=torch.float32).to(self.device)
        self.data_min = self.data_min[:self.n_dim]
        self.data_max = self.data_max[:self.n_dim]

        # Labels for the dimensions
        self.labels = {
            0: "M_1_normalized",
            1: "M_2_normalized",
            2: "tanb_normalized",
            3: "mu_normalized",
            4: "M_3_normalized",
            5: "At_normalized",
            6: "Ab_normalized",
            7: "Atau_normalized",
            8: "mA_normalized",
            9: "mqL3_normalized",
            10: "mtR_normalized",
            11: "mbR_normalized"
        }


        # # Create a regular grid mesh for x_test
        # self.n_points_per_dim = int(np.ceil( ** (1 / self.n_dim)))
        # grid = np.linspace(0, 1, self.n_points_per_dim)
        # mesh = np.meshgrid(*[grid] * self.n_dim)
        # points = np.vstack(list(map(np.ravel, mesh))).T
        # self.x_test = torch.tensor(points, dtype=torch.float32).to(self.device)
        self.x_test = torch.tensor(qmc.LatinHypercube(d=self.n_dim).random(n=5000), dtype=torch.float32).to(self.device)

        self.load_initial_data()
        self.initialize_model()



    def load_initial_data(self):

        # Open the ROOT file
        final_df = pd.read_csv(self.start_root_file_path)

        base_order = ["M_1", "M_2", "tan_beta", "mu", "M_3", "At", "Ab", "Atau", 
                      "mA", "mqL3", "mtR", "mbR"]
        order = {i: base_order[:i] for i in range(1, len(base_order) + 1)}
        
        # Get the parameters to include based on n
        selected_columns = order.get(self.n_dim, None)

        # Apply the mask to filter only valid Omega values
        CLs = final_df['Final__CLs']
        # Crosssection = final_df['xsec_TOTAL']
        mask = CLs > 0 # TODO: ask if this is a valid mask
        filtered_data = {param: final_df[f"{param}"][mask] for param in selected_columns}
        filtered_data['Final__CLs'] = CLs[mask]  # Always include Omega for filtering

        # Construct the limited DataFrame dynamically with only the selected columns
        limited_df = pd.DataFrame(filtered_data)

        # Convert to tensors and move to device
        # Take the first part (initial_train_points) of the data for training
        self.x_train = torch.stack([torch.tensor(limited_df[param].values[:self.initial_train_points], dtype=torch.float32) for param in selected_columns], dim=1).to(self.device)
        self.y_train = torch.tensor(limited_df['Final__CLs'].values[:self.initial_train_points], dtype=torch.float32).to(self.device)
        
        # Take the second part (valid_points) of the data for validation
        self.x_valid = torch.stack([torch.tensor(limited_df[param].values[self.initial_train_points:self.initial_train_points+self.valid_points], dtype=torch.float32) for param in selected_columns], dim=1).to(self.device)
        self.y_valid = torch.tensor(limited_df['Final__CLs'].values[self.initial_train_points:self.initial_train_points+self.valid_points], dtype=torch.float32).to(self.device)
        
        # Create a tensor for all available points to use in plotting true, difference and pull plots
        self.x_all = torch.stack([torch.tensor(limited_df[param].values, dtype=torch.float32) for param in selected_columns], dim=1).to(self.device)
        self.y_all = torch.tensor(limited_df['Final__CLs'].values, dtype=torch.float32).to(self.device)

        # Normalize the x_data
        self.x_train = self._normalize(self.x_train)
        self.x_valid = self._normalize(self.x_valid)
        self.x_all = self._normalize(self.x_all)

        print("Initial Training points: ", self.x_train, self.x_train.shape)
        print("Validation points: ", self.x_valid, self.x_valid.shape)
        print("All points: ", self.x_all, self.x_all.shape)

    def load_additional_data(self, new_x, new_y):

        # Load new training points and evaluate truth function
        additional_x_train = torch.tensor(new_x).to(self.device)
        additional_y_train = torch.tensor(new_y).to(self.device)

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
        combined_set = {tuple(float(x) for x in x_row) + (float(y.item()),) for x_row, y in zip(self.x_train, self.y_train)}
        
        # Unpack the combined_set into x_train and y_train
        x_train, y_train = zip(*[(x[:self.n_dim], x[self.n_dim]) for x in combined_set])

        # Convert the unpacked x_train and y_train back to torch tensors
        self.x_train = torch.tensor(list(x_train), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(list(y_train), dtype=torch.float32).to(self.device)

        print("Training points after adding: ", self.x_train, self.x_train.shape)

    def _normalize(self, data):
        return (data - self.data_min) / (self.data_max - self.data_min)
    
    # # This works only for the dataset, otherwise i use hardcoded maxima and mimima 
    # def _normalize(self, data):
    #     return (data - data.min(dim=0).values) / (data.max(dim=0).values - data.min(dim=0).values)

    def _unnormalize(self, data):
        return data * (self.data_max - self.data_min) + self.data_min

    def initialize_model(self):
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood, n_dim).to(self.device)

    def train_model(self, iters=20):
        print("These training_points are used in the GP", self.x_train)
        self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(iters=iters)
        # Print the hyperparameters of the best model
        print("best model parameters: ", self.best_model.state_dict())
        # Save the state dictionary of the best model
        # torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, 'best_multitask_gp.pth'))

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
            print("Likelihood", self.observed_pred)
            mean = self.observed_pred.mean.detach().reshape(-1, 1).to(self.device)
            print("Mean: ", mean)
            var = self.observed_pred.variance.detach().reshape(-1, 1).to(self.device)
            print("Variance: ", var)
            thr = torch.Tensor([self.thr]).to(self.device)
    
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
        plt.xlabel(self.labels[slice_dim_x1])
        plt.ylabel(self.labels[slice_dim_x2])

        # Create a filtered version of the training data
        filtered_x_train = self.x_train.clone()
        filtered_y_train = self.y_train.clone()

        # Filter the training data based on the slice value and tolerance in the remaining dimensions
        for dim in remaining_dims:
            indices = (filtered_x_train[:, dim].cpu().numpy() >= slice_value - tolerance) & (filtered_x_train[:, dim].cpu().numpy() <= slice_value + tolerance)
            filtered_x_train = filtered_x_train[indices, :]
            filtered_y_train = filtered_y_train[indices]
            # indices_new = (new_x[:, dim].cpu().numpy() >= slice_value - tolerance) & (new_x[:, dim].cpu().numpy() <= slice_value + tolerance)
            # filtered_new_x = new_x[indices_new, :]
        
        # Scatterplot of the training points
        plt.scatter(filtered_x_train[:, slice_dim_x1].cpu().numpy(), filtered_x_train[:, slice_dim_x2].cpu().numpy(), marker='o', s=50, c=filtered_y_train.cpu().numpy(), cmap='inferno', vmin=vmin, vmax=vmax, label='training points')
        
        # Contour plot of the truth function
        plt.contour(x1_grid, x2_grid, mean_grid, levels=[self.thr], colors='white', linewidths=2, linestyles='solid')

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

    def plotDifference(self, slice_dim_x1=0, slice_dim_x2=1, save_path=None, iteration=None):
        '''
        Plot the Difference of the true function and the GP prediction with a Heatmap:
        For this take all the available points and evaluate the GP on them. 
        Then substract the mean from the true value and plot the difference in a heatmap.
        '''

        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(self.x_all)
        
        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()

        # Calculate the difference between the predicted mean and the true value
        diff = torch.tensor(mean) - self.y_all

        heatmap, xedges, yedges = np.histogram2d(self.x_all[:, slice_dim_x1].numpy(), self.x_all[:, slice_dim_x2].numpy(), bins=50, weights=diff.cpu().numpy())
        heatmap_counts, xedges, yedges = np.histogram2d(self.x_all[:, slice_dim_x1].numpy(), self.x_all[:, slice_dim_x2].numpy(), bins=50)
        heatmap = heatmap/heatmap_counts

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Difference')
        plt.xlabel(self.labels[slice_dim_x1])
        plt.ylabel(self.labels[slice_dim_x2])

        # Add the iteration number to the plot title if provided
        if iteration is not None:
            plt.title(f"Difference mean vs true - Iteration {iteration}")

        # Save the plot or display it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


    def plotPull(self, slice_dim_x1=0, slice_dim_x2=1, save_path=None, iteration=None):
        '''Plot the Pull of the evaluated model against the true value'''

        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(self.x_all)

        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

        # Calculate the pull: (predicted mean - true value) / uncertainty (upper - lower)
        pull = (torch.tensor(mean) - self.y_all) / (upper - lower)

        # Use a histogram to create a 2D heatmap of the pull values
        heatmap, xedges, yedges = np.histogram2d(self.x_all[:, slice_dim_x1].numpy(), self.x_all[:, slice_dim_x2].numpy(), bins=50, weights=pull.cpu().numpy())
        heatmap_counts, xedges, yedges = np.histogram2d(self.x_all[:, slice_dim_x1].numpy(), self.x_all[:, slice_dim_x2].numpy(), bins=50)
        heatmap = heatmap/heatmap_counts

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='(Mean - True) / Uncertainty')
        plt.xlabel(self.labels[slice_dim_x1])
        plt.ylabel(self.labels[slice_dim_x2])

        # Add the iteration number to the plot title if provided
        if iteration is not None:
            plt.title(f"Pull - Iteration {iteration}")

        # Save the plot or display it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plotTrue(self, slice_dim_x1=0, slice_dim_x2=1, save_path=None):
        '''
        Plot the True Function with a Heatmap: Because the EWKino Scan only has 12800 points in 12 dimensions,
        we sum the points up instead of slicing trough the dimensions.
        '''

        heatmap, xedges, yedges = np.histogram2d(self.x_all[:, slice_dim_x1].numpy(), self.x_all[:, slice_dim_x2].numpy(), bins=50, weights=self.y_all.numpy())
        heatmap_counts, xedges, yedges = np.histogram2d(self.x_all[:, slice_dim_x1].numpy(), self.x_all[:, slice_dim_x2].numpy(), bins=50)
        heatmap = heatmap/heatmap_counts

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Final_CLs')
        plt.xlabel(self.labels[slice_dim_x1])
        plt.ylabel(self.labels[slice_dim_x2])
        plt.title(f"True Function CLs")

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plotEntropy(self, slice_dim_x1=0, slice_dim_x2=1, slice_value=0.5, save_path=None, iteration=None):
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

        plt.xlabel(self.labels[slice_dim_x1])
        plt.ylabel(self.labels[slice_dim_x2])

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"GP Prediction Slice - Iteration {iteration}")

        # Save the plot or show it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()



    def plotSlice1D(self, slice_dim=0, slice_value=0.75, tolerance=0.01, new_x=None, save_path=None, iteration=None):
        """
        Plot a 1D slice of the GP model's predictions, confidence interval, training data, and entropy for a fixed value of M_1 or M_2.

        Parameters:
            slice_dim (int): The dimension to slice (0 for M_1, 1 for M_2).
            slice_value (float): The value of the dimension to slice at.
            tolerance (float): The tolerance for the slice (e.g., +/- 0.1).
            new_x (torch.Tensor): New points to highlight on the plot (optional).
            save_path (str): The path to save the plot, if specified.
            iteration (int): The current iteration number, if any (for the plot title).
        """
        # Obtain mean, lower, and upper bounds for confidence intervals
        mean = self.observed_pred.mean.cpu().numpy()
        lower, upper = self.observed_pred.confidence_region()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()
        entropy = self.entropy.cpu().numpy()
        x_test = self.x_test.cpu().numpy()

        add_on = 0.05

        # Slice the data based on the specified slice_dim (0 for M_1, 1 for M_2)
        indices = np.where((x_test[:, slice_dim] >= slice_value - tolerance) & (x_test[:, slice_dim] <= slice_value + tolerance))[0]

        # Get the corresponding x_test[:, 1 - slice_dim] and filtered values
        x_test_filtered = x_test[indices, 1 - slice_dim]
        mean_filtered = mean[indices]
        lower_filtered = lower[indices]
        upper_filtered = upper[indices]
        entropy_filtered = entropy[indices]

        # Start plotting
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        # Plot the GP mean prediction with confidence intervals
        ax1.plot(x_test_filtered, mean_filtered, 'b-', label='Predicted Mean')
        ax1.fill_between(x_test_filtered, lower_filtered, upper_filtered, color='blue', alpha=0.3, label='Confidence Interval')
        
        ax1.set_xlabel(f'M_{"2" if slice_dim == 0 else "1"}')
        ax1.set_ylabel('log(Omega/0.12)')
        ax1.grid(True)

        # Plot true data points
        if self.true_root_file_path:
            # Open the ROOT file
            file = uproot.open(self.true_root_file_path)
            tree_name = "susy"
            tree = file[tree_name]
            
            # Convert tree to pandas DataFrame
            df = tree.arrays(library="pd")
            M_1 = df['IN_M_1']
            M_2 = df['IN_M_2']
            Omega = df['MO_Omega']
            #mask = Omega > 0
            M_1_filtered = self._normalize(M_1, self.data_min, self.data_max)[0] 
            M_2_filtered = self._normalize(M_2, self.data_min, self.data_max)[0] 
            print(f"M_1_filtered Shape: {M_1_filtered.shape}")
            print(f"M_2_filtered Shape: {M_1_filtered.shape}")
            Omega_filtered = Omega#[mask]

            # Normalize the true data
            if slice_dim == 0:
                indices_true = np.where((M_1_filtered >= slice_value - tolerance) & (M_1_filtered <= slice_value + tolerance))[0] # Creates tuple from which we take the first index
                x_true = M_2_filtered[indices_true] # Is marginalized along M_1, so M_2 is x_coordinate
            else:
                indices_true = np.where((M_2_filtered >= slice_value - tolerance) & (M_2_filtered <= slice_value + tolerance))[0]
                x_true = M_1_filtered[indices_true] # Is marginalized along M_2, so M_1 is x_coordinate
            y_true = torch.log(torch.tensor(Omega_filtered.values, dtype=torch.float32) / 0.12).cpu().numpy()
            y_true = y_true[indices_true]
            ax1.plot(x_true, y_true, '*', c='b', label = 'True Function')

            print(f"X_true: {x_true}")
            print(f"X_true Shape: {x_true.shape}")
            print(f"Y_true: {y_true}")
            print(f"Y_true Shape: {y_true.shape}")

        # Plot training data points
        if slice_dim == 0:
            indices_train = np.where((self.x_train[:, 0].cpu().numpy() >= slice_value - tolerance - add_on) & (self.x_train[:, 0].cpu().numpy() <= slice_value + tolerance + add_on))[0]
            x_train = self.x_train[:, 1].cpu().numpy()
            x_train = x_train[indices_train]
        else:
            indices_train = np.where((self.x_train[:, 1].cpu().numpy() >= slice_value - tolerance - add_on) & (self.x_train[:, 1].cpu().numpy() <= slice_value + tolerance + add_on))[0]
            x_train = self.x_train[:, 0].cpu().numpy()
            x_train = x_train[indices_train]
        y_train = self.y_train.cpu().numpy()
        y_train = y_train[indices_train]
        ax1.plot(x_train, y_train, '*', c='r', label='Training Data')

        # Plot new points if provided
        if new_x is not None:
            new_x_np = new_x[:, slice_dim].cpu().numpy()
            dolabel = True
            for xval in new_x_np:  # Loop through the new_x values and plot each as a vertical line
                ax1.axvline(x=xval, color='r', linestyle='--', label='New Points' if dolabel else "")
                dolabel = False

        # Add a second y-axis to plot entropy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Entropy')
        ax2.plot(x_test_filtered, entropy_filtered, 'g-', label='Entropy')

        # Set the lower limit of the y-axis to 0.0 without specifying an upper limit
        ax2.set_ylim(bottom=0.0)

        # Mark the maximum entropy point
        maxE = np.max(entropy_filtered)
        maxIndex = np.argmax(entropy_filtered)
        maxX = x_test_filtered[maxIndex]
        ax2.plot(maxX, maxE, 'go', label='Max. Entropy')

        # Combine legends from both y-axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        # Add the iteration number to the plot title
        if iteration is not None:
            ax1.set_title(f"GP Model Prediction Slice - Iteration {iteration}")

        # Save the plot or show it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_conf_matrix(self, save_path=None):
        '''Plot the confusion matrix as a heatmap'''

        plt.figure(figsize=(5, 4))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

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
            # Implement all the available points for acitve Learning
            # Open the data csv file
            final_df = pd.read_csv(self.start_root_file_path)

            base_order = ["M_1", "M_2", "tan_beta", "mu", "M_3", "At", "Ab", "Atau", 
                          "mA", "mqL3", "mtR", "mbR"]
            order = {i: base_order[:i] for i in range(1, len(base_order) + 1)}
            
            # Get the parameters to include based on n
            selected_columns = order.get(self.n_dim, None)

            # Apply the mask to filter only valid Omega values
            CLs = final_df['Final__CLs']
            Crosssection = final_df['xsec_TOTAL']
            mask = CLs > 0 # TODO: ask if this is a valid mask
            filtered_data = {param: final_df[f"{param}"][mask] for param in selected_columns}
            filtered_data['Final__CLs'] = CLs[mask]  # Always include Omega for filtering

            # Construct the limited DataFrame dynamically with only the selected columns
            limited_df = pd.DataFrame(filtered_data)

            # Convert to tensors
            x_data = torch.stack([torch.tensor(limited_df[param].values[:5000], dtype=torch.float32) for param in selected_columns], dim=1).to(self.device)
            x_data = self._normalize(x_data)
            y_data = torch.tensor(limited_df['Final__CLs'].values[:5000], dtype=torch.float32).to(self.device)
            # Subtract the already chosen training point from the data
            mask = torch.all((x_data.unsqueeze(1) != self.x_train), dim=2).all(dim=1)
            x_data = x_data[mask]
            y_data = y_data[mask]
            print("x_data Shape: ", x_data.shape)
            print("y_data Shape: ", y_data.shape)

            # Evaluate the model
            self.model.eval()
            print("Model Evaluation")

            # Disable gradient computation for evaluation
            with torch.no_grad():
                # Get predictions of model for random test points
                predictions = self.model(x_data)
                print("Model Predictions")
                observed_pred = self.likelihood(predictions)
                print("Likelihoof Predictions")

            mean = observed_pred.mean.detach().to(self.device)
            print("Mean Shape: ", mean.shape)
            covar = observed_pred.covariance_matrix.detach().to(self.device)
            print("Covar Shape: ", covar.shape)
            thr = torch.Tensor([self.thr]).to(self.device) 
            print("Threshold Shape: ", thr.shape) 

            # Use the selector to choose a set of points based on the  mean and covariance matrix.
            points = set(selector(N=N, gp_mean=mean - thr, gp_covar=covar))  
            print("Selected points (indices):", points)
            new_x = x_data[list(points)]
            new_y = y_data[list(points)]
            
            # Unnormalize the selected points to return them to their original scale.
            new_x_unnormalized = self._unnormalize(new_x)

        # Debugging and informational prints to check the selected points and their corresponding values.
        print("Selected points (indices):", points)
        print("Selected new x values (normalized):", new_x)
        print("Corresponding new x values (unnormalized):", new_x_unnormalized)
        
        return new_x, new_x_unnormalized, new_y
    
    def random_new_points(self, N=4):
        # TODO: adjust this method to random as well
        # Latin Hypercube
        x_test_rand = torch.tensor(qmc.LatinHypercube(d=2).random(n=self.x_test.shape[0]), dtype=torch.float32).to(self.device)
        # # Select random indices from the available test points
        # random_indices = np.random.choice(x_test_rand.shape[0], N, replace=False)  # Randomly select N indices

        # Latin Hypercube sampling
        # Define the number of samples and dimensions
        num_samples = N  # Number of indices to sample
        dimensions = 2  # Dimensionality of the space (e.g., 2 for x_test_rand)

        # Generate Latin Hypercube Samples
        sampler = qmc.LatinHypercube(d=dimensions)
        lhs_samples = sampler.random(n=num_samples)  # Generate N samples in [0, 1)

        # Map LHS samples to indices in x_test_rand
        lhs_indices = np.floor(lhs_samples[:, 0] * x_test_rand.shape[0]).astype(int)  # Use the first dimension for indices
        lhs_indices = np.clip(lhs_indices, 0, x_test_rand.shape[0] - 1)  # Ensure valid index range

        # Ensure indices are unique if necessary
        random_indices = np.unique(lhs_indices)[:N]  # Limit to N unique indices


        new_x = self.x_test[random_indices]

        # Unnormalize the selected points to return them to their original scale.
        new_x_unnormalized = self._unnormalize(new_x)

        return new_x, new_x_unnormalized
    
    def regular_grid(self, N=4):
        # TODO: adjust this method to random as well
        # Select random indices from the available test points
        random_indices = np.random.choice(self.x_test.shape[0], N, replace=False)  # Randomly select N indices
        new_x = self.x_test[random_indices]

        # Unnormalize the selected points to return them to their original scale.
        new_x_unnormalized = self._unnormalize(new_x, self.data_min, self.data_max)

        return new_x, new_x_unnormalized

    
    def goodness_of_fit(self, test=None, csv_path=None):
        # Open the data csv file
        final_df = pd.read_csv(self.start_root_file_path)

        base_order = ["M_1", "M_2", "tan_beta", "mu", "M_3", "At", "Ab", "Atau", 
                        "mA", "mqL3", "mtR", "mbR"]
        order = {i: base_order[:i] for i in range(1, len(base_order) + 1)}
        
        # Get the parameters to include based on n
        selected_columns = order.get(self.n_dim, None)

        # Apply the mask to filter only valid Omega values
        CLs = final_df['Final__CLs']
        Crosssection = final_df['xsec_TOTAL']
        mask = CLs > 0 # TODO: ask if this is a valid mask
        filtered_data = {param: final_df[f"{param}"][mask] for param in selected_columns}
        filtered_data['Final__CLs'] = CLs[mask]  # Always include Omega for filtering

        # Construct the limited DataFrame dynamically with only the selected columns
        limited_df = pd.DataFrame(filtered_data)

        # Convert to tensors
        # Take the last part of the data for testing
        x_data = torch.stack([torch.tensor(limited_df[param].values[self.initial_train_points + self.valid_points:], dtype=torch.float32) for param in selected_columns], dim=1).to(self.device)
        input_data = self._normalize(x_data)
        true = torch.tensor(limited_df['Final__CLs'].values[self.initial_train_points + self.valid_points:], dtype=torch.float32).to(self.device)

        print("Input Data Shape: ", input_data.shape)
        print("True Shape: ", true.shape)

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

        # Define the various goodness_of_fit metrics
        def mean_squared():
            mse = torch.mean((mean - true) ** 2)
            rmse = torch.sqrt(mse)
            return mse.cpu().item(), rmse.cpu().item()
        
        def r_squared():
            ss_res = torch.sum((mean - true) ** 2)
            ss_tot = torch.sum((true - torch.mean(true)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            return r_squared.cpu().item()
        
        def chi_squared():
            pull = (mean - true) / (upper - lower)
            chi_squared = torch.sum(pull ** 2)
            dof = len(true) - 1  # Degrees of freedom
            reduced_chi_squared = chi_squared / dof
            return chi_squared.cpu().item(), reduced_chi_squared.cpu().item()

        def average_pull():
            absolute_pull = torch.abs((mean - true) / (upper - lower))
            mean_absolute_pull = torch.mean(absolute_pull)
            root_mean_squared_pull = torch.sqrt(torch.mean((absolute_pull) ** 2))
            return mean_absolute_pull.cpu().item(), root_mean_squared_pull.cpu().item()

        # Weights for adjusting to threshold
        thr = self.thr
        epsilon = 0.1 # Tolerance area around threshold, considered close
        weights = torch.exp(-((true - thr) ** 2) / (2 * epsilon ** 2))  

        def mean_squared_weighted():
            mse_weighted = torch.mean(weights * (mean - true) ** 2)
            rmse_weighted = torch.sqrt(mse_weighted)
            return mse_weighted.cpu().item(), rmse_weighted.cpu().item()

        def r_squared_weighted():
            ss_res_weighted = torch.sum(weights * (mean - true) ** 2)
            ss_tot_weighted = torch.sum(weights * (true - torch.mean(true)) ** 2)
            r_squared_weighted = 1 - ss_res_weighted / ss_tot_weighted
            return r_squared_weighted.cpu().item()
        
        def chi_squared_weighted():
            pull_weighted = (mean - true) / (upper - lower)
            chi_squared_weighted = torch.sum(weights * pull_weighted ** 2)
            dof = len(true) - 1  # Degrees of freedom
            reduced_chi_squared_weighted = chi_squared_weighted / dof
            return chi_squared_weighted.cpu().item(), reduced_chi_squared_weighted.cpu().item()
        
        def average_pull_weighted():
            absolute_pull_weighted = torch.abs((mean - true) / (upper - lower))
            mean_absolute_pull_weighted = torch.mean(weights * absolute_pull_weighted)
            root_mean_squared_pull_weighted = torch.sqrt(torch.mean(weights * (absolute_pull_weighted) ** 2))
            return mean_absolute_pull_weighted.cpu().item(), root_mean_squared_pull_weighted.cpu().item()

        # Calculate classification matrix and accuracy
        def accuracy():
    
            # # TODO: Classification with uncertainty    
            # mean_plus = mean + (upper - lower)/2
            # mean_minus = mean + (upper - lower)/2
            
            # TODO: die Punkte näher an der Contour stärker gewichten
            TP = ((mean > self.thr) & (true > self.thr)).sum().item()  # True Positives
            FP = ((mean > self.thr) & (true < self.thr)).sum().item()  # False Positives
            FN = ((mean < self.thr) & (true > self.thr)).sum().item()  # False Negatives
            TN = ((mean < self.thr) & (true < self.thr)).sum().item()  # True Negatives

            self.conf_matrix = np.array([[TP, FP], [FN, TN]])

            total = TP + FP + FN + TN
            accuracy = (TP + TN) / total if total > 0 else 0

            return accuracy

            
        # Dictionary to map test names to functions
        tests = {
            'mean_squared': mean_squared,
            'r_squared': r_squared,
            'chi_squared': chi_squared,
            'average_pull': average_pull,
            'mean_squared_weighted': mean_squared_weighted,
            'r_squared_weighted': r_squared_weighted,
            'chi_squared_weighted': chi_squared_weighted,
            'average_pull_weighted': average_pull_weighted,
            'accuracy': accuracy
        }

        # If 'test' is None, run all tests and return the results
        if test is None:
            results = {}
            for test_name, test_function in tests.items():
                result = test_function()
                if isinstance(result, tuple):
                    for i, value in enumerate(result):
                        results[f"{test_name}_{i}"] = value  # Store each element of the tuple separately
                else:
                    results[test_name] = result  # Store single value directly

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

        # # Load the saved dict
        # self.model.load_state_dict(torch.load(model_checkpoint_path))
        # print(f"Model loaded from the {model_checkpoint_path}")
        
        # Ensure the model loads on the correct device (GPU or CPU)
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved state dict
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=map_location))
        print(f"Model loaded from {model_checkpoint_path}")



# Usage
if __name__ == "__main__":

    # Define Boolean to either do Active learning selection or random selection
    IsActiveLearning = False
    IsFullTraining = True

    n_dim = 3
    name = f"two_circles{n_dim}D_CLs_test"
    threshold = 0.05
    al_points = 24

    args = parse_args()
    
    if IsFullTraining:
        gp_pipeline = GPModelPipeline(
            start_root_file_path='/u/dvoss/al_pmssmwithgp/model/EWKino.csv',
            output_dir=args.output_dir, initial_train_points=10, valid_points=10, n_dim=n_dim , threshold=threshold
        )
        gp_pipeline.initialize_model()
        gp_pipeline.train_model(iters=1000)
        gp_pipeline.plot_losses()
        gp_pipeline.evaluate_model()
        gp_pipeline.goodness_of_fit(csv_path=f'/u/dvoss/al_pmssmwithgp/model/gof_{name}.csv')
        gp_pipeline.plot_conf_matrix(save_path=os.path.join(args.output_dir, f'confusion_matrix_{name}.png'))

        # Generate all possible combinations of two dimensions from n_dim for plotting
        combinations = list(itertools.combinations(range(n_dim), 2))

        for (slice_dim_x1, slice_dim_x2) in combinations:
            gp_pipeline.plotSlice2D(
                slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, slice_value=0.5,
                save_path=os.path.join(args.output_dir, f'gp_plot_{slice_dim_x1}_{slice_dim_x2}_{name}.png'),
                iteration=args.iteration
            )
            gp_pipeline.plotDifference(slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, save_path=os.path.join(args.output_dir, f'diff_plot_{name}.png'), iteration=args.iteration)
            gp_pipeline.plotPull(slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, save_path=os.path.join(args.output_dir, f'pull_plot_{name}.png'), iteration=args.iteration)
            gp_pipeline.plotEntropy(slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, save_path=os.path.join(args.output_dir, f'entropy_plot_{name}.png'), iteration=args.iteration)
        gp_pipeline.save_model(os.path.join(args.output_dir,f'model_checkpoint_{name}_rand.pth'))
    else:
        previous_iter = args.iteration - 1
        previous_output_dir = f'/raven/u/dvoss/al_pmssmwithgp/model/plots/Iter{previous_iter}'

        if IsActiveLearning == True:
            training_data_path = os.path.join(previous_output_dir, 'training_data.pkl')
        else:
            training_data_path = os.path.join(previous_output_dir, 'training_data_rand.pkl')
        
        if args.iteration == 1:
            # First iteration, initialize with initial data
            gp_pipeline = GPModelPipeline(
                start_root_file_path='/u/dvoss/al_pmssmwithgp/model/EWKino.csv',
                output_dir=args.output_dir, initial_train_points=50, valid_points=300, additional_points_per_iter=50, n_dim=n_dim
            )
        else:
            # Load previous training data and add new data from this iteration
            gp_pipeline = GPModelPipeline(
                start_root_file_path='/u/dvoss/al_pmssmwithgp/model/EWKino.csv',
                output_dir=args.output_dir, initial_train_points=50, valid_points=300, additional_points_per_iter=50, n_dim=n_dim
            )
            if os.path.exists(training_data_path):
                gp_pipeline.load_training_data(training_data_path)

        gp_pipeline.initialize_model()
        gp_pipeline.train_model(iters=100)
        gp_pipeline.plot_losses()
        gp_pipeline.evaluate_model()
        # Plot with Active Learning new points
        if IsActiveLearning:
            new_points, new_x, new_y = gp_pipeline.select_new_points(N=al_points)
            # gp_pipeline.goodness_of_fit(csv_path=f'/u/dvoss/al_pmssmwithgp/model/gof_{name}.csv')

            # Generate all possible combinations of two dimensions from n_dim for plotting
            # combinations = list(itertools.combinations(range(n_dim), 2))

            # for (slice_dim_x1, slice_dim_x2) in combinations:
            #     gp_pipeline.plotSlice2D(
            #         slice_dim_x1=slice_dim_x1, slice_dim_x2=slice_dim_x2, slice_value=0.5,
            #         new_x=new_points,
            #         save_path=os.path.join(args.output_dir, f'gp_plot_{slice_dim_x1}_{slice_dim_x2}_{name}.png'),
            #         iteration=args.iteration
            #     )
            # gp_pipeline.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, f'gp_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, f'gp_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotDifference(new_x=new_points, save_path=os.path.join(args.output_dir, f'diff_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotPull(new_x=new_points, save_path=os.path.join(args.output_dir, f'pull_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotEntropy(new_x=new_points, save_path=os.path.join(args.output_dir, f'entropy_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotTrue(new_x=new_points, save_path=os.path.join(args.output_dir, f'true_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotSlice1D(slice_dim=0, slice_value=0.75, tolerance=0.01, new_x=new_points, save_path=os.path.join(args.output_dir, f'1DsliceM2_plot_{name}.png'), iteration=args.iteration)
            # gp_pipeline.plotSlice1D(slice_dim=1, slice_value=0.75, tolerance=0.01, new_x=new_points, save_path=os.path.join(args.output_dir, f'1DsliceM1_plot_{name}.png'), iteration=args.iteration)
            gp_pipeline.load_additional_data(new_x, new_y)
            gp_pipeline.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'))
            gp_pipeline.save_model(os.path.join(args.output_dir,f'model_checkpoint_{name}.pth'))
        # Plot with random chosen new points
        else:
            gp_pipeline.goodness_of_fit(csv_path=f'/u/dvoss/al_pmssmwithgp/model/gof_{name}_rand.csv')
            new_points, new_points_unnormalized = gp_pipeline.random_new_points(N=50)
            gp_pipeline.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, f'gp_plot_{name}_rand.png'), iteration=args.iteration)
            gp_pipeline.save_training_data(os.path.join(args.output_dir, 'training_data_rand.pkl'))
            gp_pipeline.save_model(os.path.join(args.output_dir,f'model_checkpoint_{name}_rand.pth'))



    
