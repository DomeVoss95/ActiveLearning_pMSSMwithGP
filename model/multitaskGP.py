import gpytorch
import torch

from linearMean import LinearMean

class MultitaskGP(gpytorch.models.ExactGP):
    '''MultitaskGP inherits from ExactGP'''

    def __init__(self, x_train, y_train, x_valid, y_valid, likelihood):
        '''Initialize new MultitaskGP:
            x_train: stacked SimpleAnalysis and reco queries for (mzp, mdh, mdm, gx)
            y_train: stacked SimpleAnalysis and reco queries for target values
            likelihood (gpytorch.likelihoods): Gaussian Likelihood with white noise
        '''
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = LinearMean(input_size=1)
        # self.covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        self.covar_module.lengthscale = 0.5e-2
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=1, # Single task
            rank=1, # Rank of the covariance matrix - higher rank for more complex relationsships between tasks
            prior=gpytorch.priors.SmoothedBoxPrior(0,1) # der auskommentierte, wenn mehrere tasks (den lkj)
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.likelihood = likelihood

    def forward(self, x):
        '''Evaluate the posterior GP:
            x: contains all the 4D points where we want to evaluate the GP, they must be scaled to the hypercube
            returns a multivariate distribution with mean and variance tensors for each point in x.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def do_train_loop(self, lr=0.01, iters=1200, optimizer="Adam"):
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
            1e-6
        ), gpytorch.settings.max_preconditioner_size(
            10
        ), gpytorch.settings.max_root_decomposition_size(
            7000
        ):
            losses_train = []
            losses_valid = []
            
            for i in range(iters):
                self.train()
                self.likelihood.train()

                optimizer.zero_grad()
                
                # output = self.likelihood(self(x=self.x_train))
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
    


        

        
    

