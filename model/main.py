import os
import torch
from data_loader import DataLoader
from models import GPModel
from active_learning import ActiveLearning
from plots import Plots
from metrics import Metrics
from gpytorch.likelihoods import GaussianLikelihood
import argparse

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="GP Model Pipeline")
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

# Steering main class, which initializes all important variables and runs iterationloops
class GPModelPipeline:
    # Initialize all variables for the imported modular classes
    def __init__(self, start_root_file_path=None, true_root_file_path=None, root_file_path=None, output_dir=None, initial_train_points=1000, valid_points=400, additional_points_per_iter=3, n_test=100000):
        # Initialize all file_paths
        self.start_root_file_path = start_root_file_path
        self.true_root_file_path = true_root_file_path
        self.root_file_path = root_file_path
        self.output_dir = output_dir

        # Initialize size of train, valid and additional points
        self.initial_train_points = initial_train_points
        self.valid_points = valid_points
        self.additional_points_per_iter = additional_points_per_iter
        self.n_test = n_test
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initalize model parameters
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

        # Initialize evalution parameters
        x1_test = torch.linspace(0, 1, 50)
        x2_test = torch.linspace(0, 1, 50)
        x1_grid, x2_grid = torch.meshgrid(x1_test, x2_test)
        self.x_test = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1).to(self.device)

    # Function to run amount of iterations for iterative training
    def run_for_iterations(self, start_iter, end_iter):

        # Loop over all iterations
        for iteration in range(start_iter, end_iter + 1):
            print(f"Running for iteration {iteration}")
            
            # Define paths for models and training data
            previous_iter = iteration - 1
            previous_output_dir = f'plots/Iter{previous_iter}'
            training_data_path = os.path.join(previous_output_dir, 'training_data.pkl')
            model_checkpoint_path = f'plots/Iter{iteration}/model_checkpoint.pth'
            
            # Initialize DataLoader, GPModel, and ActiveLearning classes
            data_loader = DataLoader(
                start_root_file_path=self.start_root_file_path,
                true_root_file_path=self.true_root_file_path,
                root_file_path=f'/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_{iteration}/ntuple.0.0.root',
                initial_train_points = self.initial_train_points,
                valid_points = self.valid_points,
                device=self.device
            )
            gp_model = GPModel()
            

            # Step 1: Load data
            if iteration == 1:
                data_loader.load_initial_data()
            else:
                if os.path.exists(training_data_path):
                    data_loader.load_training_data(training_data_path)
                data_loader.load_additional_data()

            # Step 2: Initialize model and train
            gp_model.initialize_model()
            #gp_model.train_model(iters=1000)

            # Step 3: Evaluate the model
            gp_model.evaluate_model()

            # TODO: how can i fix unnormalize in ACtive lEaring
            # Assign the necessary normalization values to ActiveLearning
            active_learning.data_min = data_loader.data_min
            active_learning.data_max = data_loader.data_max

            # Set the observed_pred in ActiveLearning
            active_learning = ActiveLearning(device=self.device, x_test=self.x_test, data_min=self.data_min, data_max=self)
            active_learning.observed_pred = gp_model.observed_pred

            # Step 4: Active learning to select new points
            new_points, new_points_unnormalized = active_learning.select_new_points(N=3)

            # Step 5: Plot results
            plots = Plots(gp_model.observed_pred, gp_model.x_test, active_learning.entropy, data_loader.x_train, data_loader.y_train)
            plots.plotGP2D(new_x=new_points, save_path=os.path.join(self.output_dir, 'gp_plot.png'), iteration=iteration)

            # Step 6: Save model and data
            data_loader.save_training_data(os.path.join(self.output_dir, 'training_data.pkl'))
            data_loader.save_model(model_checkpoint_path)

if __name__ == "__main__":
    args = parse_args()

    # Define the iteration range
    start_iter = 1
    end_iter = 5  

    # Create instance of main class and run the pipeline over iterations
    gp_pipeline = GPModelPipeline(
        start_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_true/ntuple.0.0.root',
        true_root_file_path='/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_true/ntuple.0.0.root',
        root_file_path=f'/u/dvoss/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scans/scan_1/ntuple.0.0.root',
        output_dir=args.output_dir
    )

    # Run the pipeline over the specified iterations
    gp_pipeline.run_for_iterations(start_iter, end_iter)

# TODO: Write the script so modular, that training happens on raven and that i can evaluate plots and gof locally!
