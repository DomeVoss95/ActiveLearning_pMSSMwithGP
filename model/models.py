import torch
import gpytorch
from multitaskGP2D import MultitaskGP2D
from gpytorch.likelihoods import GaussianLikelihood

class GPModel:
    def __init__(self):
        super(GPModel, self).__init__()
        self.model = None
        self.device = None
        self.likelihood = GaussianLikelihood().to(self.device)

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

        # Define x_test for 2 Dimensions TODO: adjust less hard coded for higher dimensions
        x1_test = torch.linspace(0, 1, 50)
        x2_test = torch.linspace(0, 1, 50)
        x1_grid, x2_grid = torch.meshgrid(x1_test, x2_test)
        self.x_test = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1).to(self.device)
        
    def initialize_model(self):
        self.model = MultitaskGP2D(self.x_train, self.y_train, self.x_valid, self.y_valid, self.likelihood, 2).to(self.device)

    def train_model(self, iters=20):
        print("These training_points are used in the GP", self.x_train)
        self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(iters=iters)
        # Print the hyperparameters of the best model
        print("best model parameters: ", self.best_model.state_dict())
        # Save the state dictionary of the best model
        # torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, 'best_multitask_gp.pth'))
    
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

        
