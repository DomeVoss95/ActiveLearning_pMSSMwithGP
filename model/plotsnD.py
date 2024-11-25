import uproot
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from data_loader import DataLoader

class Plots:
    def __init__(self, data_loader, model=None, likelihood=None, device=None):
        self.data_loader = data_loader
        self.model = model
        self.likelihood = likelihood
        self.device = device

    # TODO: Create slices out of every dimension 
    # # Obtain mean, lower, and upper bounds for confidence intervals
    #     mean = self.observed_pred.mean.cpu().numpy()
    #     lower, upper = self.observed_pred.confidence_region()
    #     lower = lower.cpu().numpy()
    #     upper = upper.cpu().numpy()
    #     entropy = self.entropy.cpu().numpy()
    #     x_test = self.x_test.cpu().numpy()

    #     add_on = 0.05

    #     # Slice the data based on the specified slice_dim (0 for M_1, 1 for M_2)
    #     indices = np.where((x_test[:, slice_dim] >= slice_value - tolerance) & (x_test[:, slice_dim] <= slice_value + tolerance))[0]

    #     # Get the corresponding x_test[:, 1 - slice_dim] and filtered values
    #     x_test_filtered = x_test[indices, 1 - slice_dim]
    #     mean_filtered = mean[indices]
    #     lower_filtered = lower[indices]
    #     upper_filtered = upper[indices]
    #     entropy_filtered = entropy[indices]

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
            M_1_al_normalized = self.data_loader._normalize(M_1_al_tensor, self.data_min, self.data_max)[0]
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
        heatmap_counts, xedges, yedges = np.histogram2d(self.x_test[:, 0].cpu().numpy(), self.x_test[:, 1].cpu().numpy(), bins=50)
        heatmap = heatmap/heatmap_counts

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='log(Omega/0.12)')
        plt.xlabel('M_1_normalized')
        plt.ylabel('M_2_normalized')

        # Scatterplot of the training points
        plt.scatter(self.x_train[:, 0].cpu().numpy(), self.x_train[:,1].cpu().numpy(), marker='*', s=200, c='b', label='training_points')

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
            M_1_al_normalized = self.data_loader._normalize(M_1_al_tensor, self.data_min, self.data_max)[0]
            M_2_al_normalized = self.data_loader._normalize(M_2_al_tensor, self.data_min, self.data_max)[0]

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

        # Convert to PyTorch tensors
        M_1 = torch.tensor(M_1, dtype=torch.float32)
        M_2 = torch.tensor(M_2, dtype=torch.float32)

        mask = Omega > 0
        M_1_filtered = self.data_loader._normalize(M_1[mask], self.data_min, self.data_max)[0]
        M_2_filtered = self.data_loader._normalize(M_2[mask], self.data_min, self.data_max)[0]
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

        # Convert to PyTorch tensors
        M_1 = torch.tensor(M_1, dtype=torch.float32)
        M_2 = torch.tensor(M_2, dtype=torch.float32)

        mask = Omega > 0
        M_1_filtered = self.data_loader._normalize(M_1[mask], self.data_min, self.data_max)[0]
        M_2_filtered = self.data_loader._normalize(M_2[mask], self.data_min, self.data_max)[0]
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

        # Convert to PyTorch tensors
        M_1 = torch.tensor(M_1, dtype=torch.float32)
        M_2 = torch.tensor(M_2, dtype=torch.float32)

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


    # TODO: Adjust do nD 
    def plotSlice1D(self, slice_dim=0, slice_value=0.75, tolerance=0.01, new_x=None, save_path=None, iteration=None):
        """
        Plot a 1D slice of the GP model's predictions, confidence interval, training data, and entropy for a fixed value of M_1 or M_2.

        Parameters:
            slice_dim (int): The dimension to slice (0 for M_1, 1 for M_2, 2 for tanb, 3 for mu).
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

            # Convert to PyTorch tensors
            M_1 = torch.tensor(M_1, dtype=torch.float32)
            M_2 = torch.tensor(M_2, dtype=torch.float32)

            #mask = Omega > 0
            M_1_filtered = self.data_loader._normalize(M_1, self.data_min, self.data_max)[0] 
            M_2_filtered = self.data_loader._normalize(M_2, self.data_min, self.data_max)[0] 
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

    def plotSlice2D(self, slice_dim_x1=0, slice_dim_x2=0, slice_value=0.75, tolerance=0.01, new_x=None, save_path=None, iteration=None):
        """
        Plot a 1D slice of the GP model's predictions, confidence interval, training data, and entropy for a fixed value of M_1 or M_2.

        Parameters:
            slice_dim (int): The dimension to slice (0 for M_1, 1 for M_2, 2 for tanb, 3 for mu).
            slice_value (float): The value of the dimension to slice at.
            tolerance (float): The tolerance for the slice (e.g., +/- 0.1).
            new_x (torch.Tensor): New points to highlight on the plot (optional).
            save_path (str): The path to save the plot, if specified.
            iteration (int): The current iteration number, if any (for the plot title).
        """
        # Obtain mean, lower, and upper bounds for confidence intervals
        mean = self.observed_pred.mean.cpu().numpy()
        x_test = self.x_test.cpu().numpy()

        add_on = 0.05

        # Slice the data based on the specified slice_dim (0 for M_1, 1 for M_2)
        indices_x1 = np.where((x_test[:, slice_dim_x1] >= slice_value - tolerance) & (x_test[:, slice_dim_x1] <= slice_value + tolerance))[0]
        x1_test_filtered = x_test[indices_x1, 1 - slice_dim_x1]

        # Slice the data based on the specified slice_dim (0 for M_1, 1 for M_2)
        indices_x2 = np.where((x_test[:, slice_dim_x2] >= slice_value - tolerance) & (x_test[:, slice_dim_x2] <= slice_value + tolerance))[0]
        x2_test_filtered = x_test[indices_x2, 1 - slice_dim_x2]

        combined_indices = np.intersect1d(indices_x1, indices_x2)
        mean = mean[combined_indices]

        heatmap, xedges, yedges = np.histogram2d(x1_test_filtered, x2_test_filtered, bins=50, weights=mean)
        heatmap_counts, xedges, yedges = np.histogram2d(x1_test_filtered, x2_test_filtered, bins=50)
        heatmap = heatmap/heatmap_counts

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='log(Omega/0.12)')
        # TODO: adjust the names depending on dimension
        labels = {0: "M_1_normalized", 1: "M_2_normalized", 2: "tanb_normalized", 3: "mu_normalized"}
        plt.xlabel(labels[slice_dim_x1])
        plt.ylabel(labels[slice_dim_x2])

        # Scatterplot of the training points
        plt.scatter(self.x_train[:, slice_dim_x1].cpu().numpy(), self.x_train[:, slice_dim_x2].cpu().numpy(), marker='*', s=200, c='b', label='training_points')

        # Contour wo mean > 0
        plt.contour(xedges[:-1], yedges[:-1], heatmap.T, levels=[0], colors='white', linewidths=2, linestyles='solid')

        # Scatter plot of additional data (al_df) if root_file_path is provided
        if self.root_file_path is not None:
            # Extract and normalize al_df data points
            al_df = uproot.open(self.root_file_path)["susy"].arrays(library="pd")
            dim = {0: "IN_M_1", 1: "IN_M_2", 2: "IN_tanb", 3: "IN_mu"}
            x1_al = al_df[dim[slice_dim_x1]]
            x2_al = al_df[dim[slice_dim_x2]]
            Omega_al = al_df['MO_Omega']

            # Normalize the al_df points
            x1_al_tensor = torch.tensor(x1_al.values, dtype=torch.float32).to(self.device)
            x2_al_tensor = torch.tensor(x2_al.values, dtype=torch.float32).to(self.device)
            x1_al_normalized = self.data_loader._normalize(x1_al_tensor, self.data_min, self.data_max)[0]
            x2_al_normalized = self.data_loader._normalize(x2_al_tensor, self.data_min, self.data_max)[0]

            # Scatter plot for the additional points (M_1_al and M_2_al)
            plt.scatter(x1_al_normalized.cpu().numpy(), x2_al_normalized.cpu().numpy(), s=200,
                        c='g', marker='*', label='al_df Data')


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

            # Convert to PyTorch tensors
            M_1 = torch.tensor(M_1, dtype=torch.float32)
            M_2 = torch.tensor(M_2, dtype=torch.float32)

            #mask = Omega > 0
            M_1_filtered = self.data_loader._normalize(M_1, self.data_min, self.data_max)[0] 
            M_2_filtered = self.data_loader._normalize(M_2, self.data_min, self.data_max)[0] 
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



        # Add the iteration number to the plot title
        if iteration is not None:
            ax1.set_title(f"GP Model Prediction Slice - Iteration {iteration}")

        # Save the plot or show it
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    