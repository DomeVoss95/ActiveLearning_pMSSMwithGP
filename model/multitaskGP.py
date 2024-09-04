import gpytorch
import torch
import numpy as np
from collections import OrderedDict
from linearMean import LinearMean

class MultitaskGP(gpytorch.models.ExactGP):
    '''MultitaskGP inherits from ExactGP'''

    def __init__(self, x_train, y_train, x_valid, y_valid, likelihood, seed=42):
        '''Initialize new MultitaskGP:
            x_train: stacked SimpleAnalysis and reco queries for (mzp, mdh, mdm, gx)
            y_train: stacked SimpleAnalysis and reco queries for target values
            likelihood (gpytorch.likelihoods): Gaussian Likelihood with white noise
            seed: random seed for reproducibility
        '''
        # Set the seed for reproducibility
        self.set_seed(seed)

        super().__init__(x_train, y_train, likelihood)
        self.mean_module = LinearMean(input_size=1)
        
        # Setting up the covariance module with lengthscale constraints
        self.covar_module = gpytorch.kernels.RBFKernel(nu=1.5)
        self.covar_module.raw_lengthscale.requires_grad_(False)
        self.covar_module.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1e-4, 1.0)) # (1e-4, 1.0) 1e-1
        self.covar_module.raw_lengthscale.requires_grad_(True)
        
        # Setting initial lengthscale value within the new constraints
        self.covar_module.lengthscale = torch.tensor([0.5e-3]).unsqueeze(0)  # This value is within the range [0.01, 10.0] and has correct shape

        # Task covariance module
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=1,  # Single task
            rank=1,  # Rank of the covariance matrix - higher rank for more complex relationships between tasks
            prior=gpytorch.priors.SmoothedBoxPrior(0, 1)  # If using multiple tasks, use a different prior
        )

        # Adjusting the noise constraint
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-3, 1e-2))
        self.likelihood.noise_covar.raw_noise.requires_grad_(True)

        # Adjusting the raw variance constraint
        self.task_covar_module.raw_var.requires_grad_(False)
        self.task_covar_module.register_constraint("raw_var", gpytorch.constraints.Interval(0,1e-1))
        self.task_covar_module.raw_var.requires_grad_(True)

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.likelihood = likelihood

    def set_seed(self, seed):
        '''Set the seed for reproducibility'''
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x):
        '''Evaluate the posterior GP:
            x: contains all the 4D points where we want to evaluate the GP, they must be scaled to the hypercube
            returns a multivariate distribution with mean and variance tensors for each point in x.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def do_train_loop(self, lr=0.01, iters=1200, optimizer="Adam", weight_decay=0.01):
        '''Basic method to train a MultitaskGP with marginal loglikelihood as loss
            lr: Learning rate, Default is 0.01
            iters: Number of iterations for optimizer, Defaults is 1200
            optimizer: Type either Adam or SGD without momentum. Default is Adam    
            weight_decay: L2 regularization term, Default is 0.01
        '''

        # Optimizer
        if optimizer == "SGD": 
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initial loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        best_loss = 1e10

        # Initialize best_model with the current state of the model
        best_model = self

        # Training loop
        with gpytorch.settings.cg_tolerance(1e-6), gpytorch.settings.max_preconditioner_size(10), gpytorch.settings.max_root_decomposition_size(7000):
            losses_train = []
            losses_valid = []
            
            for i in range(iters):
                self.train()
                self.likelihood.train()

                optimizer.zero_grad()
                
                output = self(self.x_train)
                loss = -mll(output, self.y_train.view(-1))
                loss.backward(retain_graph=True)

                # Save training loss:
                losses_train.append(loss.detach().item())

                # Save model:
                if loss < best_loss:
                    best_loss = loss.detach()
                    best_model = self

                optimizer.step()

                # Evaluate model and save validation loss:
                self.eval()
                self.likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred_valid = self.likelihood(self(self.x_valid))
                    loss_valid = -mll(observed_pred_valid, self.y_valid.view(-1))
                    losses_valid.append(loss_valid.item())

                # if i % 100 == 0:
                #     print(
                #         "Iter %d / %d - Loss (Train): %.3f - Loss (Val): %.3f" % (i + 1, iters, loss.detach().item(), loss_valid.detach().item())
                #     )
        return best_model, losses_train, losses_valid
