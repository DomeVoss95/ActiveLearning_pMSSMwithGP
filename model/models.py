import torch
import gpytorch
from multitaskGP2D import MultitaskGP2D

class GPModel:
    def __init__(self, likelihood, device):
        self.likelihood = likelihood.to(device)
        self.model = None
        self.device = device

    def initialize_model(self, x_train, y_train, x_valid, y_valid):
        self.model = MultitaskGP2D(x_train, y_train, x_valid, y_valid, self.likelihood, 2).to(self.device)

    def train_model(self, iters=20):
        print("Training model with {} iterations...".format(iters))
        best_model, losses, losses_valid = self.model.do_train_loop(iters=iters)
        print("Training complete.")
        return best_model, losses, losses_valid

    def evaluate_model(self, x_test):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            observed_pred = self.likelihood(self.model(x_test))
            return observed_pred
