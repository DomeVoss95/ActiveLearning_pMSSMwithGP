import gpytorch
import torch
import numpy as np
from linearMean import LinearMean
from gpytorch.lazy import NonLazyTensor


class MultitaskGP(gpytorch.models.ExactGP):
    '''MultitaskGP inherits from ExactGP'''

    def __init__(self, x_train, y_train, x_valid, y_valid, likelihood, n, seed=42):
        '''Initialize new MultitaskGP:
            x_train: stacked SimpleAnalysis and reco queries for (mzp, mdh, mdm, gx)
            y_train: stacked SimpleAnalysis and reco queries for target values
            likelihood (gpytorch.likelihoods): Gaussian Likelihood with white noise
            n: dimension of x_train
            seed: random seed for reproducibility
        '''
        # Set the seed for reproducibility
        self.set_seed(seed)

        super().__init__(x_train, y_train, likelihood)
        self.mean_module = LinearMean(input_size=n)
        
        # Setting up the covariance module with lengthscale constraints
        # self.covar_module = gpytorch.kernels.RBFKernel()
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=n)
        # )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n)
        )


        # self.covar_module = ScaleKernel(
        #     RBFKernel(
        #         lengthscale_prior=LogNormalPrior(0.0, 1.0)  # Informative Prior
        #     ),
        #     outputscale_prior=LogNormalPrior(0.0, 1.0)  # Informative Prior
        # )


        # TODO: Remove when testing is done
        # # Setting up the covariance module with ARD 
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=n)
        # )

        # self.covar_module.lengthscale = 0.5e-2
        # self.task_covar_module = gpytorch.kernels.IndexKernel(
        #     num_tasks=1, # Single task
        #     rank=1, # Rank of the covariance matrix - higher rank for more complex relationsships between tasks
        #     prior=gpytorch.priors.SmoothedBoxPrior(0,1) # der auskommentierte, wenn mehrere tasks (den lkj)
        #     )
        
                # Setting up the task covariance module
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=1, # Single task
            rank=1, # Rank of the covariance matrix - higher rank for more complex relationships between tasks
            prior=gpytorch.priors.SmoothedBoxPrior(0, 1) # If using multiple tasks, use a different prior
        )

        # Adjusting the lengthscale constraint

        self.covar_module.outputscale = torch.tensor(1.0)  # Setze einen gültigen Startwert für outputscale
        self.covar_module.base_kernel.lengthscale = torch.ones(n)  # Setze initial alle lengthscales auf 1.0

        self.covar_module.register_prior(
            "outputscale_prior",
            gpytorch.priors.LogNormalPrior(0.0, 1.0),
            "raw_outputscale"
        )
        self.covar_module.register_constraint(
            "raw_outputscale",
            gpytorch.constraints.GreaterThan(1e-4)  
        )



        # Adjusting the noise constraint
        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-5, 5e-2))

        
        # self.covar_module.raw_lengthscale.requires_grad_(False)
        # self.covar_module.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1e-4, 0.1))
        # self.covar_module.raw_lengthscale.requires_grad_(True)

        self.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        self.covar_module.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1e-4, 1.0))
        #self.covar_module.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-1))
        #self.covar_module.base_kernel.raw_lengthscale.requires_grad_(True)

        
        # # Setting initial lengthscale value within the new constraints
        # self.covar_module.lengthscale = torch.tensor([0.5e-3]).unsqueeze(0) 

        # # Task covariance module
        # self.task_covar_module = gpytorch.kernels.IndexKernel(
        #     num_tasks=1,  # Single task
        #     rank=1,  # Rank of the covariance matrix - higher rank for more complex relationships between tasks
        #     prior=gpytorch.priors.SmoothedBoxPrior(0, 1)  # If using multiple tasks, use a different prior
        # )

        # # Adjusting the noise constraint
        # self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        # self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-5, 1e-2))#1e-2))
        # self.likelihood.noise_covar.raw_noise.requires_grad_(True)

        # # Adjusting the raw variance constraint
        # self.task_covar_module.raw_var.requires_grad_(False)
        # self.task_covar_module.register_constraint("raw_var", gpytorch.constraints.Interval(0,1e-1))#1e-1))
        # self.task_covar_module.raw_var.requires_grad_(True)

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

        # Debugging: Check for NaNs in mean and covariance
        # if torch.isnan(mean_x).any() or torch.isnan(covar_x.evaluate()).any():
        #     print("NaN detected in forward pass")

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def do_train_loop(self, lr=0.005, iters=1200, optimizer="Adam"):
        '''Basic method to train a MultitaskGP with marginal loglikelihood as loss
            lr: Learning rate, Default is 0.01
            iters: Number of iterations for optimizer, Defaults is 1200
            optimzer: Type either Adam or SGD without momentum. Default is Adam    
        '''

        # Optimizer
        if optimizer == "SGD": # Good: simple & memory efficient - Bad: LR sensitive, no adaptive LR, noise Updates --> may struggle  
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0) # higher momentum ~ 0.9 increase influence of past gradients, leading to smoother and more stable updates
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr) # Good: adaptive LR & momentum & bias correction & fast converge - Bad: expensive & complex
        
        # Inital loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self
        )
        best_loss = 1e10

        # Training loop
        with gpytorch.settings.cg_tolerance(
            1e-5
        ), gpytorch.settings.max_preconditioner_size(
            10
        ), gpytorch.settings.max_root_decomposition_size(
            7000
        ), gpytorch.settings.cholesky_jitter(1e-5): #, linear_operator.settings.max_cg_iterations(2000):
            losses_train = []
            losses_valid = []
            
            for i in range(iters):
                self.train()
                self.likelihood.train()

                optimizer.zero_grad()
                
                # output = self.likelihood(self(x=self.x_train))
                output = self(self.x_train)

                # # Debug: print shapes
                print(f"Iteration {i}: x_train shape: {self.x_train.shape}, y_train shape: {self.y_train.shape}")
                print(f"Output mean shape: {output.mean.shape}")

                #loss = -mll(output, self.y_train.view(-1))
                loss = -mll(output, self.y_train.view(-1))
                loss.backward(retain_graph=True)

                # Save training loss:
                losses_train.append(loss.detach().item())

                # Save model:
                if loss < best_loss:
                    best_loss = loss.detach()
                    best_model = self

                optimizer.step()

                # Evaluate model and save validtation loss:
                self.eval()
                self.likelihood.eval()

                # x_valid and y_valid are defined globally for now!
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred_valid = self.likelihood(self(self.x_valid))
                    loss_valid = -mll(observed_pred_valid, self.y_valid.view(-1))
                    losses_valid.append(loss_valid.item())

                if i % 100 == 0:
                    print(
                        "Iter %d / %d - Loss (Train): %.3f - Loss (Val): %.3f" % (i + 1, iters, loss.detach().item(), loss_valid.detach().item())
                    )
        return best_model, losses_train, losses_valid

