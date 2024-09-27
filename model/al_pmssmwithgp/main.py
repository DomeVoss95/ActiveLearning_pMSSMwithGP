import os
import argparse
from data_loader import load_initial_data, load_additional_data, normalize, unnormalize
from models import MultitaskGP2D
from utils import plot_losses, plotGP2D, plotGP1D, save_training_data, load_training_data
from create_config import create_config
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='GP Model Pipeline')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.iteration == 1:
        x_train, y_train, data_min, data_max = load_initial_data(csv_file='output.csv')
    else:
        x_train, y_train, data_min, data_max = load_training_data(training_data_path)

    model = MultitaskGP2D(x_train, y_train, likelihood, n_tasks=2)
    model.train_model()

    plot_losses(model.losses, model.losses_valid, output_dir)

    new_points, new_points_unnormalized = model.select_new_points(N=3)
    plotGP2D(model, new_points, output_dir, iteration=args.iteration)

    create_config(new_points=new_points_unnormalized, output_file='new_config.yaml')

    save_training_data(x_train, y_train, data_min, data_max, os.path.join(output_dir, 'training_data.pkl'))
