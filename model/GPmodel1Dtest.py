import uproot
import argparse
import os
import torch
import gpytorch
import pickle
from matplotlib import pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood
import pandas as pd
import numpy as np
from entropy import entropy_local  
from create_config import create_config
from multitaskGP import MultitaskGP

def parse_args():
    parser = argparse.ArgumentParser(description='GP Model Pipeline')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

class GPModelPipeline:
    def __init__(self, csv_file, root_file_path=None, output_dir=None, initial_train_points=4, valid_points=250, additional_points_per_iter=3, n_test=100000):
        self.csv_file = csv_file
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
        self.x_test = torch.linspace(0, 1, 1000).to(self.device)
        
        self.load_initial_data()
        self.initialize_model()

    def load_initial_data(self):
        final_df = pd.read_csv(self.csv_file)
        M_1 = final_df['IN_M_1']
        Omega = final_df['MO_Omega']
        
        mask = Omega > 0
        M_1_filtered = M_1[mask]
        Omega_filtered = Omega[mask]
        
        limited_df = pd.DataFrame({'IN_M_1': M_1_filtered, 'MO_Omega': Omega_filtered})
        
        # Drop rows where IN_M_1 is between 0 and 50, because chance of outliers is high
        limited_df = limited_df.drop(limited_df[(limited_df['IN_M_1'] > 0) & (limited_df['IN_M_1'] < 50)].index)

        # Convert to tensors
        self.x_train = torch.tensor(limited_df['IN_M_1'].values[:self.initial_train_points], dtype=torch.float32).to(self.device)
        self.y_train = torch.log(torch.tensor(limited_df['MO_Omega'].values[:self.initial_train_points], dtype=torch.float32).to(self.device) / 0.12)
        
        self.x_valid = torch.tensor(limited_df['IN_M_1'].values[self.valid_points:], dtype=torch.float32).to(self.device)
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
        Omega_al = al_df['MO_Omega']
        
        # Convert al data to tensors and concatenate with existing training data
        M_1_al_tensor = torch.tensor(M_1_al.values, dtype=torch.float32).to(self.device)
        Omega_al_tensor = torch.tensor(Omega_al.values, dtype=torch.float32).to(self.device)
        
        additional_x_train = self._normalize(M_1_al_tensor, self.data_min, self.data_max)[0][:self.additional_points_per_iter]
        additional_y_train = torch.log(Omega_al_tensor / 0.12)[:self.additional_points_per_iter]
        
        # Append the new points to the existing training data
        self.x_train = torch.cat((self.x_train, additional_x_train))
        self.y_train = torch.cat((self.y_train, additional_y_train))

        # Convert the concatenated x_train and y_train to a set of tuples to remove duplicates
        combined_set = {(float(x), float(y)) for x, y in zip(self.x_train, self.y_train)}

        # Convert the set back to two tensors
        self.x_train, self.y_train = zip(*combined_set)
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)

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
        self.model = MultitaskGP(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood).to(self.device)

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
            mean = self.observed_pred.mean.detach().reshape(-1, 1).to(self.device)
            var = self.observed_pred.variance.detach().reshape(-1, 1).to(self.device)
            thr = torch.Tensor([0.]).to(self.device)

            #print(f"Point: {self.x_test} - Variance: {var} ")

            # self.entropy = entropy_local(mean, var, thr, self.device, torch.float32)
            # print("Entropy: ", self.entropy)

    def plotGP(self, new_x=None, save_path=None, iteration=None):
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
        ax.plot(self.x_test.cpu().numpy(), var, 'o', label='Variance')
        ax.fill_between(self.x_test.cpu().numpy(), lower, upper, alpha=0.5, label='Confidence')
        
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
        #print("Smoothed_batch_entropy: ", score)
        self.entropy = score
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
        # Initialize the selector using a combination of a smoothed batch entropy score function and a Gibbs sampling choice function.
        selector = self.iterative_batch_selector(
            score_function=self.smoothed_batch_entropy(blur=0.15),  # Score function with a small blur applied to smooth entropy calculation
            choice_function=self.gibbs_sample(beta=50)  # Choice function using Gibbs sampling with a specified beta
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            # Detach the mean and covariance of the observed predictions from the computational graph
            # and move them to the appropriate device (CPU or GPU).
            mean = self.observed_pred.mean.detach().to(self.device)
            covar = self.observed_pred.covariance_matrix.detach().to(self.device)
            thr = torch.Tensor([0.]).to(self.device)  # Threshold tensor (can be adjusted if needed)

            # Filter the candidate points to only those between 0.1 and 0.9
            self.valid_indices = ((self.x_test < 0.50) | (self.x_test > 0.515)) #self.x_test >= 0.01) # & 
            x_test_filtered = self.x_test[self.valid_indices]  
            mean_filtered = mean[self.valid_indices]  
            covar_filtered = covar[self.valid_indices][:, self.valid_indices]  

            # Use the selector to choose a set of points based on the filtered mean and covariance matrix.
            points = set(selector(N=N, gp_mean=mean_filtered - thr, gp_covar=covar_filtered))  
            new_x = x_test_filtered[list(points)]  

            # # Use the selector to choose a set of points based on the  mean and covariance matrix.
            # points = set(selector(N=N, gp_mean=mean - thr, gp_covar=covar))  
            # new_x = self.x_test[list(points)]

            # Unnormalize the selected points to return them to their original scale.
            new_x_unnormalized = self._unnormalize(new_x, self.data_min, self.data_max)

        # Debugging and informational prints to check the selected points and their corresponding values.
        print("Selected points (indices):", points)  # Indices of the selected points
        print("Selected new x values (normalized):", new_x)  # The normalized new x values chosen by the selector
        print("Corresponding new x values (unnormalized):", new_x_unnormalized)  # The unnormalized values of the selected points
        
        return new_x, new_x_unnormalized  # Return both the normalized and unnormalized selected points


    def save_training_data(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.x_train, self.y_train, self.data_min, self.data_max), f)
    
    def load_training_data(self, filepath):
        with open(filepath, 'rb') as f:
            self.x_train, self.y_train, self.data_min, self.data_max = pickle.load(f)


# Usage
if __name__ == "__main__":
    args = parse_args()
    
    previous_iter = args.iteration - 1
    previous_output_dir = f'/raven/u/dvoss/al_pmssmwithgp/model/plots/Iter{previous_iter}'
    training_data_path = os.path.join(previous_output_dir, 'training_data.pkl')
    
    if args.iteration == 1:
        # First iteration, initialize with initial data
        gp_pipeline = GPModelPipeline(
            csv_file='M_2_fixed.csv',
            # root_file_path=f'/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_{args.iteration}/ntuple.0.0.root',
            output_dir=args.output_dir
        )
    else:
        # Load previous training data and add new data from this iteration
        gp_pipeline = GPModelPipeline(
            csv_file='M_2_fixed.csv',
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
    new_points, new_points_unnormalized = gp_pipeline.select_new_points(N=3)
    gp_pipeline.plotGP(new_x=new_points, save_path=os.path.join(args.output_dir, 'gp_plot.png'), iteration=args.iteration)
    create_config(new_points=new_points_unnormalized, output_file ='new_config.yaml')
    
    gp_pipeline.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'))

