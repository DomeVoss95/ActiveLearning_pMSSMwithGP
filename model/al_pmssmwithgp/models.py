import torch
import gpytorch

class MultitaskGP2D(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, n_tasks):
        # GP model implementation
        pass

    def train_model(self, iters=1000):
        # Training loop
        pass

    def select_new_points(self, N=4):
        # Point selection logic
        pass
