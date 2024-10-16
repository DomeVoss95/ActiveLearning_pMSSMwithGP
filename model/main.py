import os
import torch
from data_loader import DataLoader
from models import GPModel
from utils import Utils
from active_learning import ActiveLearning
from plots import Plots

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="GP Model Pipeline")
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DataLoader and GPModel objects
    data_loader = DataLoader(device)
    gp_model = GPModel(likelihood=torch.nn.GaussianLikelihood(), device=device)

    # Load initial or additional data based on the iteration
    if args.iteration == 1:
        x_train, y_train, x_valid, y_valid = data_loader.load_initial_data(
            root_file_path='/path/to/start/root/file', 
            initial_train_points=50, valid_points=150)
        data_min, data_max = -2000, 2000  # You may set this dynamically based on your data
    else:
        x_train, y_train, data_min, data_max = Utils.load_training_data(
            os.path.join(f'plots/Iter{args.iteration - 1}', 'training_data.pkl'))
        x_train_add, y_train_add = data_loader.load_additional_data(
            root_file_path=f'/path/to/root/file_{args.iteration}.root', 
            additional_points_per_iter=3, normalize_fn=Utils.normalize, 
            data_min=data_min, data_max=data_max)
        x_train = torch.cat((x_train, x_train_add))
        y_train = torch.cat((y_train, y_train_add))

    # Initialize and train the model
    gp_model.initialize_model(x_train, y_train, x_valid, y_valid)
    best_model, losses, losses_valid = gp_model.train_model(iters=750)

    # Plot losses
    Utils.plot_losses(losses, losses_valid, args.output_dir)

    # Evaluate the model
    x_test = torch.linspace(0, 1, 50).to(device)
    observed_pred = gp_model.evaluate_model(x_test)

    # Initialize Plots
    Plots = Plots(observed_pred=observed_pred, x_test=x_test, entropy=entropy, x_train=x_train, y_train=y_train)

    # Save training data and create configuration
    Utils.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'), x_train, y_train, data_min, data_max)

    # Find new strategic training points
    new_points, new_points_unnormalized = Active_Learning.select_new_points(N=3)

    # Do various kinds of 
    Plots.plotGP2D(new_x=new_points, save_path=os.path.join(args.output_dir, 'gp_plot.png'), iteration=args.iteration)
    Plots.plotDifference(new_x=new_points, save_path=os.path.join(args.output_dir, 'diff_plot.png'), iteration=args.iteration)
    Plots.plotPull(new_x=new_points, save_path=os.path.join(args.output_dir, 'pull_plot.png'), iteration=args.iteration)
    Plots.plotEntropy(new_x=new_points, save_path=os.path.join(args.output_dir, 'entropy_plot.png'), iteration=args.iteration)
    Plots.plotTrue(new_x=new_points, save_path=os.path.join(args.output_dir, 'true_plot.png'), iteration=args.iteration)

    Utils.create_config(new_points=new_points_unnormalized, output_file ='new_config.yaml')
    
    Utils.save_training_data(os.path.join(args.output_dir, 'training_data.pkl'))
