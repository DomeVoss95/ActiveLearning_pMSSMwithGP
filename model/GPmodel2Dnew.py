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
from create_config import create_config
from multitaskGPnD import MultitaskGP

def parse_args():
    parser = argparse.ArgumentParser(description='GP Model Pipeline')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

class GPModelPipeline:
    def __init__(self, start_root_file_path=None, true_root_file_path=None, root_file_path=None, output_dir=None, initial_train_points=1000, valid_points=400, additional_points_per_iter=3, n_test=100000):
        self.start_root_file_path = start_root_file_path
        self.true_root_file_path = true_root_file_path
        self.root_file_path = root_file_path
        self.output_dir = output_dir
        self.initial_train_points = initial_train_points
        self.valid_points = valid_points
        self.additional_points_per_iter = additional_points_per_iter
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

        x1_test = torch.linspace(0, 1, 50)
        x2_test = torch.linspace(0, 1, 50)
        x1_grid, x2_grid = torch.meshgrid(x1_test, x2_test)
        self.x_test = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1).to(self.device)

        self.load_initial_data()
        self.initialize_model()

    def load_initial_data(self):

        # Open the ROOT file
        file = uproot.open(self.start_root_file_path)
        
        tree_name = "susy"
        tree = file[tree_name]
        
        # Convert tree to pandas DataFrame
        final_df = tree.arrays(library="pd")

        M_1 = final_df['IN_M_1']
        M_2 = final_df['IN_M_2']
        Omega = final_df['MO_Omega']
        
        mask = Omega > 0
        M_1_filtered = M_1[mask]
        M_2_filtered = M_2[mask]
        Omega_filtered = Omega[mask]
        
        limited_df = pd.DataFrame({'IN_M_1': M_1_filtered, 'IN_M_2': M_2_filtered, 'MO_Omega': Omega_filtered})
        
        # Drop rows where IN_M_1 is between 0 and 50, because chance of outliers is high
        # limited_df = limited_df.drop(limited_df[(limited_df['IN_M_1'] > 0) & (limited_df['IN_M_1'] < 50)].index)

        # Convert to tensors
        self.x_train = torch.stack([torch.tensor(limited_df['IN_M_1'].values[:self.initial_train_points], dtype=torch.float32), torch.tensor(limited_df['IN_M_2'].values[:self.initial_train_points], dtype=torch.float32)], dim=1).to(self.device)
        self.y_train = torch.log(torch.tensor(limited_df['MO_Omega'].values[:self.initial_train_points], dtype=torch.float32).to(self.device) / 0.12)
        
        self.x_valid = torch.stack([torch.tensor(limited_df['IN_M_1'].values[self.valid_points:], dtype=torch.float32), torch.tensor(limited_df['IN_M_2'].values[self.valid_points:], dtype=torch.float32)], dim=1).to(self.device)
        self.y_valid = torch.log(torch.tensor(limited_df['MO_Omega'].values[self.valid_points:], dtype=torch.float32).to(self.device) / 0.12)
        
        self.x_train, self.data_min, self.data_max = self._normalize(self.x_train)
        self.x_valid = self._normalize(self.x_valid, self.data_min, self.data_max)[0]

        print("Initial Training points: ", self.x_train, self.x_train.shape)

    def load_additional_data(self):
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
        M_1_al_normalized = self._normalize(M_1_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter]
        M_2_al_normalized = self._normalize(M_2_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter]

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

    def _normalize(self, data, data_min=-2000, data_max=2000):
        if data_min is None:
            data_min = data.min(dim=0, keepdim=True).values
        if data_max is None:
            data_max = data.max(dim=0, keepdim=True).values
        return (data - data_min) / (data_max - data_min), data_min, data_max

    def _unnormalize(self, data, data_min, data_max):
        return data * (data_max - data_min) + data_min

    def initialize_model(self):
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood, 2).to(self.device)

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
            thr = torch.Tensor([0.]).to(self.device)

            #print(f"Point: {self.x_test} - Variance: {var} ")

            # self.entropy = entropy_local(mean, var, thr, self.device, torch.float32)
            # print("Entropy: ", self.entropy)

    def plotGP1D(self, new_x=None, save_path=None, iteration=None):
        """
        Plot the GP model's predictions, confidence interval, and various data points.
        
        :param new_x: New points to be highlighted on the plot (if any).
        :param save_path: Path to save the plot as an image file.
        :param iteration: The current iteration number to be displayed in the plot title.
        """
        mean = self.observed_pred.mean.cpu().numpy()
        lower, upper = self.observed_pred.confidence_region()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()
        var = self.observed_pred.variance.cpu().numpy()

        # print("Confindence difference: ", upper - lower)

        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(self.x_test.cpu().numpy(), mean, 'b', label='Learnt Function')
        ax.plot(self.x_test.cpu().numpy(), var, label='Variance')
        ax.fill_between(self.x_test.cpu().numpy(), lower, upper, alpha=0.5, label='Confidence')

        ax.set_xlabel("M_1")
        ax.set_ylabel("log(Omega/0.12)")
        
        # Plot true data points
        x_true = torch.tensor(pd.read_csv(self.csv_file)['IN_M_1'].values[:], dtype=torch.float32).to(self.device)
        x_true_min = x_true.min(dim=0, keepdim=True).values
        x_true_max = x_true.max(dim=0, keepdim=True).values
        x_true = (x_true - x_true_min) / (x_true_max - x_true_min)
        y_true = torch.log(torch.tensor(pd.read_csv(self.csv_file)['MO_Omega'].values[:], dtype=torch.float32) / 0.12).to(self.device)
        
        ax.plot(x_true.cpu().numpy(), y_true.cpu().numpy(), '*', c="r", label='Truth')
        ax.plot(self.x_train.cpu().numpy(), self.y_train.cpu().numpy(), 'k*', label='Training Data')

        if new_x is not None:
            dolabel = True
            for xval in new_x.cpu().numpy():  # Move xval to CPU before using it in axvline
                ax.axvline(x=xval, color='r', linestyle='--', label='new points') if dolabel else ax.axvline(x=xval, color='r', linestyle='--')
                dolabel = False

        # Check if root_file_path is provided before loading and plotting al_df data
        if self.root_file_path is not None:
            # Extract and normalize al_df data points
            al_df = uproot.open(self.root_file_path)["susy"].arrays(library="pd")
            M_1_al = al_df['IN_M_1']
            Omega_al = al_df['MO_Omega']

            M_1_al_tensor = torch.tensor(M_1_al.values, dtype=torch.float32).to(self.device)
            M_1_al_normalized = self._normalize(M_1_al_tensor, self.data_min, self.data_max)[0]
            Omega_al_tensor = torch.tensor(Omega_al.values, dtype=torch.float32).to(self.device)
            Omega_al_normalized = torch.log(Omega_al_tensor / 0.12)

            # Plot al_df data points in green
            ax.plot(M_1_al_normalized.cpu().numpy(), Omega_al_normalized.cpu().numpy(), '*', c="g", label='al_df Data')
        else:
            print("No root_file_path provided; skipping ROOT file data plotting.")

        ax2 = ax.twinx()
        ax2.set_ylabel("Entropy")
        # Make sure to use the filtered x_test when plotting entropy
        x_test_to_plot = self.x_test if self.entropy.shape[0] == self.x_test.shape[0] else self.x_test[self.valid_indices]

        # Now use x_test_to_plot for plotting entropy
        ax2.plot(x_test_to_plot.cpu().numpy(), self.entropy.cpu().numpy(), 'g', label='Entropy')
        # ax2.plot(self.x_test.cpu().numpy(), self.entropy.cpu().numpy(), 'g', label='Entropy')

        # Set the lower limit of the y-axis to 0.0 without specifying an upper limit
        ax2.set_ylim(bottom=0.0)

        maxE = torch.max(self.entropy)
        maxIndex = torch.argmax(self.entropy)
        maxX = x_test_to_plot[maxIndex] # self.x_test[maxIndex] #maybe use x_test_to plot here to make sure that its maxE is at peak of entropy
        ax2.plot(maxX.cpu().numpy(), maxE.cpu().numpy(), 'go', label='Max. E')

        # # Plot smoothed batch entropy
        # smoothed_batch_entropy = self.smoothed_batch_entropy(blur=0.15)
        # batch_entropy_values = smoothed_batch_entropy(mean=self.observed_pred.mean, cov=self.observed_pred.covariance_matrix)

        # print("Batch_entropy values: ", smoothed_batch_entropy)
        
        #ax2.plot(self.x_test.cpu().numpy(), batch_entropy_values.cpu().detach().numpy(), 'm', linestyle='--', label='Smoothed Batch Entropy')


        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        # Add the iteration number to the plot title
        if iteration is not None:
            ax.set_title(f"GP Model Prediction - Iteration {iteration}")

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plotGP2D(self, new_x=None, save_path=None, iteration=None):
        '''Plot the 2D GP with a Heatmap and the new points and save it in the plot folder'''

        mean = self.observed_pred.mean.cpu().numpy()

        heatmap, xedges, yedges = np.histogram2d(self.x_test[:, 0].cpu().numpy(), self.x_test[:, 1].cpu().numpy(), bins=50, weights=mean)

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='log(Omega/0.12)')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')

        # Scatterplot of the training points
        # TODO: Remove comment, when again with fewer points than 10.000
        # plt.scatter(self.x_train[:, 0].cpu().numpy(), self.x_train[:,1].cpu().numpy(), marker='*', s=200, c='b', label='training_points')

        # Contour wo mean > 0
        plt.contour(xedges[:-1], yedges[:-1], heatmap.T, levels=[0], colors='white', linewidths=2, linestyles='solid')

        # Scatter plot of additional data (al_df) if root_file_path is provided
        if self.root_file_path is not None:
            # Extract and normalize al_df data points
            al_df = uproot.open(self.root_file_path)["susy"].arrays(library="pd")
            M_1_al = al_df['IN_M_1']
            M_2_al = al_df['IN_M_2']  # Assuming M_2_al should be plotted in 2D
            Omega_al = al_df['MO_Omega']

            # Normalize the al_df points
            M_1_al_tensor = torch.tensor(M_1_al.values, dtype=torch.float32).to(self.device)
            M_2_al_tensor = torch.tensor(M_2_al.values, dtype=torch.float32).to(self.device)
            M_1_al_normalized = self._normalize(M_1_al_tensor, self.data_min, self.data_max)[0]
            M_2_al_normalized = self._normalize(M_2_al_tensor, self.data_min, self.data_max)[0]

            # Scatter plot for the additional points (M_1_al and M_2_al)
            plt.scatter(M_1_al_normalized.cpu().numpy(), M_2_al_normalized.cpu().numpy(), s=200,
                        c='g', marker='*', label='al_df Data')

        else:
            print("No root_file_path provided; skipping ROOT file data plotting.")

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"GP Model Prediction - Iteration {iteration}")

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plotDifference(self, new_x=None, save_path=None, iteration=None):
        '''Plot the 2D GP with a Heatmap and the new points and save it in the plot folder'''

        # Obtain the mean predictions from the GP model
        mean = self.observed_pred.mean.cpu().numpy()

        # Open the ROOT file
        file = uproot.open(self.true_root_file_path)
        tree_name = "susy"
        tree = file[tree_name]
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1'].values
        M_2 = df['IN_M_2'].values
        Omega = df['MO_Omega'].values

        mask = Omega > 0
        M_1_filtered = self._normalize(M_1[mask], self.data_min, self.data_max)[0]
        M_2_filtered = self._normalize(M_2[mask], self.data_min, self.data_max)[0]
        Omega_filtered = Omega[mask]

        # Calculate the true values (log-scaled)
        true = torch.log(torch.tensor(Omega_filtered, dtype=torch.float32) / 0.12)

        # Evaluate model at M_1 and M_2 coordinates of true
        input_data = torch.stack([
            torch.tensor(M_1_filtered, dtype=torch.float32),
            torch.tensor(M_2_filtered, dtype=torch.float32)
        ], dim=1).to(self.device)

        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(input_data)

        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

        # Now calculate the difference
        diff = torch.tensor(mean) - true

        # Use a histogram to create a 2D heatmap of the differences
        heatmap, xedges, yedges = np.histogram2d(M_1_filtered,
                                                M_2_filtered,
                                                bins=50, weights=diff.cpu().numpy())
        heatmap_counts, xedges, yedges = np.histogram2d(M_1_filtered, M_2_filtered, bins=50)
        heatmap = heatmap/heatmap_counts 

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Mean - True')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')

        # Add the iteration number to the plot title if provided
        if iteration is not None:
            plt.title(f"Difference mean vs true - Iteration {iteration}")

        # Save the plot or display it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


    def plotPull(self, new_x=None, save_path=None, iteration=None):
        '''Plot the 2D GP with a Heatmap and the new points and save it in the plot folder'''

        # Open the ROOT file
        file = uproot.open(self.true_root_file_path)
        tree_name = "susy"
        tree = file[tree_name]
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1'].values
        M_2 = df['IN_M_2'].values
        Omega = df['MO_Omega'].values

        mask = Omega > 0
        M_1_filtered = self._normalize(M_1[mask], self.data_min, self.data_max)[0]
        M_2_filtered = self._normalize(M_2[mask], self.data_min, self.data_max)[0]
        Omega_filtered = Omega[mask]

        # Calculate the true values (log-scaled)
        true = torch.log(torch.tensor(Omega_filtered, dtype=torch.float32) / 0.12)

        # Evaluate model at M_1 and M_2 coordinates of true
        input_data = torch.stack([
            torch.tensor(M_1_filtered, dtype=torch.float32),
            torch.tensor(M_2_filtered, dtype=torch.float32)
        ], dim=1).to(self.device)

        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(input_data)

        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

        # Calculate the pull: (predicted mean - true value) / uncertainty (upper - lower)
        pull = (torch.tensor(mean) - true) / (upper - lower)

        # Use a histogram to create a 2D heatmap of the pull values
        heatmap, xedges, yedges = np.histogram2d(M_1_filtered,
                                                M_2_filtered,
                                                bins=50, weights=pull.cpu().numpy())
        heatmap_counts, xedges, yedges = np.histogram2d(M_1_filtered, M_2_filtered, bins=50)
        heatmap = heatmap/heatmap_counts

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='(Mean - True) / Uncertainty')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')

        # Add the iteration number to the plot title if provided
        if iteration is not None:
            plt.title(f"Pull - Iteration {iteration}")

        # Save the plot or display it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


    def plotEntropy(self, new_x=None, save_path=None, iteration=None):
        '''Plot the 2D GP with a Heatmap and the new points and save it in the plot folder'''
        
        heatmap, xedges, yedges = np.histogram2d(self.x_test[:, 0].cpu().numpy(), self.x_test[:, 1].cpu().numpy(), bins=50, weights=self.entropy.cpu().numpy())

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Entropy')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"Entropy - Iteration {iteration}")

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plotTrue(self, new_x=None, save_path=None, iteration=None):
        '''Plot the 2D GP with a Heatmap and the new points and save it in the plot folder'''

        # Open the ROOT file
        file = uproot.open(self.true_root_file_path)
        
        tree_name = "susy"
        tree = file[tree_name]
        
        # Convert tree to pandas DataFrame
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1'].values
        M_2 = df['IN_M_2'].values
        Omega = df['MO_Omega'].values

        # Create a mask to filter out negative or zero values of Omega
        mask = Omega > 0

        # Apply the mask to filter M_1, M_2, and Omega
        M_1_filtered = M_1[mask]
        M_2_filtered = M_2[mask]
        Omega_filtered = Omega[mask]

        # Calculate the true values (log-scaled)
        true = torch.log(torch.tensor(Omega_filtered, dtype=torch.float32) / 0.12)
        
        heatmap, xedges, yedges = np.histogram2d(M_1_filtered, M_2_filtered, bins=50, weights=true.cpu().numpy())
        heatmap_counts, xedges, yedges = np.histogram2d(M_1_filtered, M_2_filtered, bins=50)
        heatmap = heatmap/heatmap_counts

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='log(Omega/0.12)')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')

        # Add the iteration number to the plot title
        if iteration is not None:
            plt.title(f"True Function - Iteration {iteration}")

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
            mask = Omega > 0
            M_1_filtered = self._normalize(M_1[mask], self.data_min, self.data_max)[0] 
            M_2_filtered = self._normalize(M_2[mask], self.data_min, self.data_max)[0] 
            print(f"M_1_filtered: {M_1_filtered}")
            print(f"M_2_filtered: {M_2_filtered}")
            print(f"M_1_filtered Shape: {M_1_filtered.shape}")
            print(f"M_2_filtered Shape: {M_1_filtered.shape}")
            Omega_filtered = Omega[mask]

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
            new_mean = gp_mean[:, None].to(self.device)
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
            score_function=self.smoothed_batch_entropy(blur=0.15),  # Score function with a small blur applied to smooth entropy calculation
            choice_function=self.gibbs_sample(beta=100)  # Choice function using Gibbs sampling with a specified beta
            # choice_function=self.best_not_yet_chosen
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            # Detach the mean and covariance of the observed predictions from the computational graph
            # and move them to the appropriate device (CPU or GPU).
            mean = self.observed_pred.mean.detach().to(self.device)
            covar = self.observed_pred.covariance_matrix.detach().to(self.device)
            thr = torch.Tensor([0.]).to(self.device)  # Threshold tensor (can be adjusted if needed)

            # # Filter the points in the middle out: there cannot be new points selected
            # self.valid_indices = ((self.x_test < 0.50) | (self.x_test > 0.515)) 
            # x_test_filtered = self.x_test[self.valid_indices]  
            # mean_filtered = mean[self.valid_indices]  
            # covar_filtered = covar[self.valid_indices][:, self.valid_indices]  

            # # Check x_test for NaNs
            # print("NaN in x_test_filtered:", torch.isnan(x_test_filtered).any())


            # # Use the selector to choose a set of points based on the filtered mean and covariance matrix.
            # points = set(selector(N=N, gp_mean=mean_filtered - thr, gp_covar=covar_filtered))  
            # new_x = x_test_filtered[list(points)]  

            # Use the selector to choose a set of points based on the  mean and covariance matrix.
            points = set(selector(N=N, gp_mean=mean - thr, gp_covar=covar))  
            new_x = self.x_test[list(points)]

            # Unnormalize the selected points to return them to their original scale.
            new_x_unnormalized = self._unnormalize(new_x, self.data_min, self.data_max)

        # Debugging and informational prints to check the selected points and their corresponding values.
        print("Selected points (indices):", points)  # Indices of the selected points
        print("Selected new x values (normalized):", new_x)  # The normalized new x values chosen by the selector
        print("Corresponding new x values (unnormalized):", new_x_unnormalized)  # The unnormalized values of the selected points
        
        return new_x, new_x_unnormalized  # Return both the normalized and unnormalized selected points
    
    def random_new_points(self, N=4):
        # Select random indices from the available test points
        random_indices = np.random.choice(self.x_test.shape[0], N, replace=False)  # Randomly select N indices
        new_x = self.x_test[random_indices]

        # Unnormalize the selected points to return them to their original scale.
        new_x_unnormalized = self._unnormalize(new_x, self.data_min, self.data_max)

        return new_x, new_x_unnormalized

    
    def goodness_of_fit(self, test=None, csv_path=None):
        # Chi-Squared divided by degrees of freedom - Goodness of Fit for quantification and comparison of training

        # Open the ROOT file
        file = uproot.open(self.true_root_file_path)
        tree_name = "susy"
        tree = file[tree_name]
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1'].values
        M_2 = df['IN_M_2'].values
        Omega = df['MO_Omega'].values

        mask = Omega > 0
        M_1_filtered = self._normalize(M_1[mask], self.data_min, self.data_max)[0]
        M_2_filtered = self._normalize(M_2[mask], self.data_min, self.data_max)[0]
        Omega_filtered = Omega[mask]

        # Calculate the true values (log-scaled)
        true = torch.log(torch.tensor(Omega_filtered, dtype=torch.float32) / 0.12)

        # Evaluate model at M_1 and M_2 coordinates of true
        input_data = torch.stack([
            torch.tensor(M_1_filtered, dtype=torch.float32),
            torch.tensor(M_2_filtered, dtype=torch.float32)
        ], dim=1).to(self.device)

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

        # Dictionary to map test names to functions
        tests = {
            'mean_squared': mean_squared,
            'r_squared': r_squared,
            'chi_squared': chi_squared,
            'average_pull': average_pull
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
            pickle.dump((self.x_train, self.y_train, self.data_min, self.data_max), f)
    
    def load_training_data(self, filepath):
        with open(filepath, 'rb') as f:
            self.x_train, self.y_train, self.data_min, self.data_max = pickle.load(f)     

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
    ActiveLearning = True

    args = parse_args()
    
    previous_iter = args.iteration - 1
    previous_output_dir = f'/raven/u/dvoss/al_pmssmwithgp/model/plots/Iter{previous_iter}'

    if ActiveLearning == True:
        training_data_path = os.path.join(previous_output_dir, 'training_data.pkl')
    else:
        training_data_path = os.path.join(previous_output_dir, 'training_data_rand.pkl')
    
    if args.iteration == 1:
        # First iteration, initialize with initial data
        gp_pipeline = GPModelPipeline(
            # TODO: uncomment when finisched with no iteration training
            #start_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_start/ntuple.0.0.root',
            start_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_true/ntuple.0.0.root',
            true_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_true/ntuple.0.0.root',
            root_file_path=f'/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_{args.iteration}/ntuple.0.0.root',
            output_dir=args.output_dir
        )
    else:
        # Load previous training data and add new data from this iteration
        gp_pipeline = GPModelPipeline(
            # TODO: uncomment when finisched with no iteration training
            #start_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_start/ntuple.0.0.root',
            start_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_true/ntuple.0.0.root',
            true_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_true/ntuple.0.0.root',
            root_file_path=f'/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_{previous_iter}/ntuple.0.0.root',
            output_dir=args.output_dir
        )
        if os.path.exists(training_data_path):
            gp_pipeline.load_training_data(training_data_path)
        gp_pipeline.load_additional_data()

    gp_pipeline.initialize_model()
    gp_pipeline.train_model(iters=1000)
    gp_pipeline.plot_losses()
    gp_pipeline.evaluate_model()
    
    # Plot with new points
    if ActiveLearning == True:
        new_points, new_points_unnormalized = gp_pipeline.select_new_points(N=100)
        gp_pipeline.goodness_of_fit(csv_path='/u/dvoss/al_pmssmwithgp/model/gof.csv')
        gp_pipeline.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, 'gp_plot.png'), iteration=args.iteration)
        gp_pipeline.plotDifference(new_x=new_points, save_path=os.path.join(args.output_dir, 'diff_plot.png'), iteration=args.iteration)
        gp_pipeline.plotPull(new_x=new_points, save_path=os.path.join(args.output_dir, 'pull_plot.png'), iteration=args.iteration)
        gp_pipeline.plotEntropy(new_x=new_points, save_path=os.path.join(args.output_dir, 'entropy_plot.png'), iteration=args.iteration)
        gp_pipeline.plotTrue(new_x=new_points, save_path=os.path.join(args.output_dir, 'true_plot.png'), iteration=args.iteration)
        gp_pipeline.plotSlice1D(slice_dim=0, slice_value=0.75, tolerance=0.01, new_x=new_points, save_path=os.path.join(args.output_dir, '1DsliceM2_plot.png'), iteration=args.iteration)
        gp_pipeline.plotSlice1D(slice_dim=1, slice_value=0.75, tolerance=0.01, new_x=new_points, save_path=os.path.join(args.output_dir, '1DsliceM1_plot.png'), iteration=args.iteration)
        gp_pipeline.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'))
        gp_pipeline.save_model(os.path.join(args.output_dir,'model_checkpoint_10000.pth')) # TODO: Remove 10000 when done
    # Plot without new points
    else:
        gp_pipeline.goodness_of_fit(csv_path='/u/dvoss/al_pmssmwithgp/model/gof.csv')
        new_points, new_points_unnormalized = gp_pipeline.random_new_points(N=10)
        gp_pipeline.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, 'gp_plot_rand.png'), iteration=args.iteration)
        gp_pipeline.save_training_data(os.path.join(args.output_dir, 'training_data_rand.pkl'))
        gp_pipeline.save_model(os.path.join(args.output_dir,'model_checkpoint_rand.pth'))

    create_config(new_points=new_points_unnormalized, output_file ='new_config.yaml')


    
