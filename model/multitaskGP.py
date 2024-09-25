import gpytorch
import torch
import numpy as np
from collections import OrderedDict
from linearMean import LinearMean
from gpytorch.lazy import NonLazyTensor
import math

# class ExactMarginalLogLikelihood(MarginalLogLikelihood):
#     """
#     The exact marginal log likelihood (MLL) for an exact Gaussian process with a
#     Gaussian likelihood.

#     .. note::
#         This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
#         and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
#         stochastic optimization.

#     :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
#     :param ~gpytorch.models.ExactGP model: The exact GP model

#     Example:
#         >>> # model is a gpytorch.models.ExactGP
#         >>> # likelihood is a gpytorch.likelihoods.Likelihood
#         >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#         >>>
#         >>> output = model(train_x)
#         >>> loss = -mll(output, train_y)
#         >>> loss.backward()
#     """

#     def __init__(self, likelihood, model):
#         if not isinstance(likelihood, _GaussianLikelihoodBase):
#             raise RuntimeError("Likelihood must be Gaussian for exact inference")
#         super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

#     def _add_other_terms(self, res, params):
#         # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
#         for added_loss_term in self.model.added_loss_terms():
#             res = res.add(added_loss_term.loss(*params))

#         # Add log probs of priors on the (functions of) parameters
#         res_ndim = res.ndim
#         for name, module, prior, closure, _ in self.model.named_priors():
#             prior_term = prior.log_prob(closure(module))
#             res.add_(prior_term.view(*prior_term.shape[:res_ndim], -1).sum(dim=-1))

#         return res

#     def forward(self, function_dist, target, *params, **kwargs):
#         r"""
#         Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

#         :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
#             the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
#         :param torch.Tensor target: :math:`\mathbf y` The target values
#         :rtype: torch.Tensor
#         :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
#         """
#         if not isinstance(function_dist, MultivariateNormal):
#             raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

#         # Determine output likelihood
#         output = self.likelihood(function_dist, *params, **kwargs)

#         # Remove NaN values if enabled
#         if settings.observation_nan_policy.value() == "mask":
#             observed = settings.observation_nan_policy._get_observed(target, output.event_shape)
#             output = MultivariateNormal(
#                 mean=output.mean[..., observed],
#                 covariance_matrix=MaskedLinearOperator(
#                     output.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
#                 ),
#             )
#             target = target[..., observed]
#         elif settings.observation_nan_policy.value() == "fill":
#             raise ValueError("NaN observation policy 'fill' is not supported by ExactMarginalLogLikelihood!")

#         # Get the log prob of the marginal distribution
#         res = output.log_prob(target)
#         res = self._add_other_terms(res, params)

#         # Scale by the amount of data we have
#         num_data = function_dist.event_shape.numel()
#         return res.div_(num_data)

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
        self.covar_module.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1e-4, 1.0)) # (1e-4, 1.0) 
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
        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-5, 1e-2))#1e-2))
        self.likelihood.noise_covar.raw_noise.requires_grad_(True)

        # Adjusting the raw variance constraint
        self.task_covar_module.raw_var.requires_grad_(False)
        self.task_covar_module.register_constraint("raw_var", gpytorch.constraints.Interval(0,1e-1))#1e-1))
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

        # Debugging: Check for NaNs in mean and covariance
        # if torch.isnan(mean_x).any() or torch.isnan(covar_x.evaluate()).any():
        #     print("NaN detected in forward pass")

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

        print("Likelihood:", self.likelihood)
        
        # Initial loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        best_loss = 1e10

        # Initialize best_model with the current state of the model
        best_model = self

        # Training loop
        with gpytorch.settings.cg_tolerance(1e-6), gpytorch.settings.max_preconditioner_size(10), gpytorch.settings.max_root_decomposition_size(7000), gpytorch.settings.cholesky_jitter(1e-6):
            losses_train = []
            losses_valid = []

            for i in range(iters):
                self.train()
                self.likelihood.train()

                optimizer.zero_grad()
                
                output = self(self.x_train)

                # Calculate the Likelihood
                observed_pred_train = self.likelihood(output)

                # Debugging: print mean and variance over the likelihood
                print(f"Iter {i}: Likelihood Mean: {observed_pred_train.mean.mean().item()}, Variance: {observed_pred_train.variance.mean().item()}")

                
                # Debugging: Print model output statistics
                #print(f"Iter {i}: Output mean: {output.mean.mean().item()}, Output variance: {output.variance.mean().item()}")

                print(type(output.covariance_matrix))
                
                covariance_matrix = NonLazyTensor(output.covariance_matrix)

                # Debugging: Print quadratic covariance_matrix
                with gpytorch.settings.fast_computations(covar_root_decomposition=False):
                    inv_quad, logdet = covariance_matrix.inv_quad_logdet(
                        inv_quad_rhs=self.y_train.view(-1).unsqueeze(-1), logdet=True
                    )
      
                # inv_quad, logdet = output.covariance_matrix.inv_quad_logdet(inv_quad_rhs=self.y_train.view(-1).unsqueeze(-1), logdet=True)
                res = -0.5 * sum([inv_quad, logdet, self.y_train.size(-1) * math.log(2 * math.pi)])
                print(f"Iteration {i}: inv_quad - {inv_quad.item()}, logdet - {logdet.item()}")
                print(f"Iteration {i}: MLL - {res.item()}")

                covar_matrix = self.covar_module(self.x_train).evaluate()
                print(f"Iteration {i}: Covariance matrix determinant: {torch.det(covar_matrix).item()}")
                if torch.isnan(covar_matrix).any():
                    print("NaN detected in covariance matrix!")


                
                loss = -mll(output, self.y_train.view(-1))
                
                # Debugging: Print loss and check for NaNs
                print(f"Iteration {i}: Loss - {loss.item()}")
                # if torch.isnan(loss):
                #     print(f"Iter {i}: NaN loss detected! Breaking training loop.")
                #     break

                
                loss.backward(retain_graph=True)

                # Debugging: Print gradients for each parameter
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad: {param.grad.norm().item()}")
                
                # Save training loss
                losses_train.append(loss.detach().item())

                # Save best model
                if loss < best_loss:
                    best_loss = loss.detach()
                    best_model = self

                optimizer.step()

                # Debugging: Print parameter values (lengthscale and noise) before and after optimizer step
                # print(f"Iter {i}: Lengthscale: {self.covar_module.lengthscale.item()}, Noise: {self.likelihood.noise.item()}")

                # Evaluate on validation data
                self.eval()
                self.likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred_valid = self.likelihood(self(self.x_valid))
                    loss_valid = -mll(observed_pred_valid, self.y_valid.view(-1))
                    losses_valid.append(loss_valid.item())

                # Calculation of the logarithmic loss
                def log_loss(y_true, mean_pred, var_pred):
                    return torch.mean(torch.log(2 * torch.pi * var_pred) + ((y_true - mean_pred) ** 2) / var_pred)

                mean_pred = observed_pred_valid.mean
                var_pred = observed_pred_valid.variance  

                log_loss_value = log_loss(self.y_valid.view(-1), mean_pred, var_pred)
                print(f"Logarithmic Loss: {log_loss_value.item()}")

                # Debugging: Print validation loss
                # print(f"Iter {i}: Validation Loss: {loss_valid.item()}")

                # if i % 100 == 0:
                #     print(f"Iter {i}: Training Loss: {loss.item()}, Validation Loss: {loss_valid.item()}")

        return best_model, losses_train, losses_valid
