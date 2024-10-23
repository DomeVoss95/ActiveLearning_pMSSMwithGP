import os
import torch
from data_loader import DataLoader
from models import GPModel
from active_learning import ActiveLearning
from plots import Plots
from metrics import Metrics
import argparse

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="GP Model Pipeline")
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def run_for_iterations(start_iter, end_iter):
    args = parse_args()

    # Loop over all iterations
    for iteration in range(start_iter, end_iter + 1):
        print(f"Running for iteration {iteration}")
        
        # Define paths for models and training data
        previous_iter = iteration - 1
        previous_output_dir = f'plots/Iter{previous_iter}'
        training_data_path = os.path.join(previous_output_dir, 'training_data.pkl')
        model_checkpoint_path = f'plots/Iter{iteration}/model_checkpoint.pth'
        
        # Initialize DataLoader, GPModel, and ActiveLearning classes
        data_loader = DataLoader()
        gp_model = GPModel()
        active_learning = ActiveLearning()

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

        # Step 4: Active learning to select new points
        new_points, new_points_unnormalized = active_learning.select_new_points(N=3)

        # Step 5: Plot results
        plots = Plots(gp_model.observed_pred, gp_model.x_test, active_learning.entropy, data_loader.x_train, data_loader.y_train)
        plots.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, 'gp_plot.png'), iteration=iteration)

        # Step 6: Save model and data
        data_loader.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'))
        data_loader.save_model(model_checkpoint_path)

if __name__ == "__main__":
    args = parse_args()

    # Define the iteration range
    start_iter = 1
    end_iter = 5  # Example range

    # Run the pipeline over iterations
    run_for_iterations(start_iter, end_iter)
