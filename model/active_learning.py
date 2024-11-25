import torch
import gpytorch
import numpy as np
from scipy.stats import qmc

class ActiveLearning:
    def __init__(self, device, x_test, data_min, data_max, model, likelihood):
        self.observed_pred = None
        self.entropy = None
        self.device = device
        self.x_test = x_test
        self.data_min = data_min 
        self.data_max = data_max
        self.model = model
        self.likelihood = likelihood
    
    def _unnormalize(self, data, data_min, data_max):
        return data * (data_max - data_min) + data_min
        
    def best_not_yet_chosen(self, score, previous_indices):
        candidates = torch.sort(score, descending=True)[1].to(self.device)
        for next_index in candidates:
            if int(next_index) not in previous_indices:
                return next_index

    def iterative_batch_selector(self, score_function, 
                                 choice_function=None,
                                 gp_mean=None, 
                                 gp_covar=None, 
                                 N=None):
        '''Chooses the next points iterativley. The point with maximum entropy is always chosen first, 
            then the next indices are selected with the choice function - gibbs_sampling or best_not_yet_chosen
            The covariance matrix and the mean vector are updated iterativly, based on the already chosen points
        '''

        if gp_mean is None:
            def greedy_batch_sel(gp_mean, gp_covar, N):
                return self.iterative_batch_selector(score_function, choice_function, gp_mean, gp_covar, N)
            return greedy_batch_sel
        
        score = score_function(gp_mean[:, None], torch.diag(gp_covar)[:, None, None]).to(self.device)
        #print("Smoothed_batch_entropy: ", score)
        self.entropy = score
        first_index = torch.argmax(score).to(self.device)
        indices = [int(first_index)]

        num_pts = len(gp_mean)

        for i in range(N-1):
            center_cov = torch.stack([gp_covar[indices, :][:, indices]] * num_pts).to(self.device) # Covariance between selected points
            side_cov = gp_covar[:, None, indices].to(self.device) # Coveriance between already selected and remaining points
            bottom_cov = gp_covar[:, indices, None].to(self.device)
            end_cov = torch.diag(gp_covar)[:, None, None].to(self.device)

            cov_batch = torch.cat([
                torch.cat([center_cov, side_cov], axis=1),
                torch.cat([bottom_cov, end_cov], axis=1),
            ], axis=2).to(self.device)

            # print("Cov_batch Shape: ", cov_batch.shape)
            # print("center_cov Shape: ", center_cov.shape)
            # print("side_cov Shape: ", side_cov.shape)
            # print("bottom_cov Shape: ", bottom_cov.shape)
            # print("end_cov Shape: ", end_cov.shape)
            
            center_mean = torch.stack([gp_mean[indices]] * num_pts).to(self.device) # Mean between selected points
            new_mean = gp_mean[:, None].to(self.device)
            mean_batch = torch.cat([center_mean, new_mean], axis=1).to(self.device)

            # print("Mean_batch Shape: ", mean_batch.shape)
            # print("center_mean Shape: ", center_mean.shape)
            # print("new_mean Shape: ", new_mean.shape)

            score = score_function(mean_batch, cov_batch).to(self.device)
            next_index = choice_function(score, indices)
            indices.append(int(next_index))


        return indices

    def approximate_batch_entropy(self, mean, cov):
        device = self.device
        n = mean.shape[-1]
        d = torch.diag_embed(1. / mean).to(device)
        x = d @ cov @ d.to(device)
        # print("Cov Shape: ", cov.shape)
        # print("X Shape: ", x.shape)
        I = torch.eye(n)[None, :, :].to(device)
        return (torch.logdet(x + I) - torch.logdet(x + 2 * I) + n * np.log(2)) / np.log(2)

    def smoothed_batch_entropy(self, blur):
        return lambda mean, cov: self.approximate_batch_entropy(mean + blur * torch.sign(mean).to(self.device), cov)

    def gibbs_sample(self, beta):
        '''Chooses next point based on probability that a random value is smaller than the cumulative sum 
            of the probabilites. These are calculated with the smoothed batch entropy
            - if beta is high: deterministic selection with only highest entropies
            - if beta is low: more random selection with each point having a similar probility
        '''
        def sampler(score, indices=None):
            probs = torch.exp(beta * (score - torch.max(score))).to(self.device)
            probs /= torch.sum(probs).to(self.device)
            # print("probs:", probs)
            cums = torch.cumsum(probs, dim=0).to(self.device)
            # print("cumsum: ", cums)
            rand = torch.rand(size=(1,)).to(self.device)[0]
            # print("rand: ", rand)
            return torch.sum(cums < rand).to(self.device)    
        return sampler

    def select_new_points(self, N=4):
        # Initialize the selector using a combination of a smoothed batch entropy score function and a Gibbs sampling choice function.
        selector = self.iterative_batch_selector(
            score_function=self.smoothed_batch_entropy(blur=0.15),  # Score function with a small blur applied to smooth entropy calculation
            choice_function=self.gibbs_sample(beta=100)  # Choice function using Gibbs sampling with a specified beta
            # choice_function=self.best_not_yet_chosen
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            # Initialize a random x_test_rand to be able to choose new points out of the whole parameter space
            # # # Completly random 
            # x_test_rand = torch.rand(self.x_test.shape[0], 2).to(self.device)
            # Latin Hypercube
            x_test_rand = torch.tensor(qmc.LatinHypercube(d=2).random(n=self.x_test.shape[0]), dtype=torch.float32).to(self.device)

            # Evaluate the model
            self.model.eval()

            # Disable gradient computation for evaluation
            with torch.no_grad():
                # Get predictions of model for random test points
                predictions = self.model(x_test_rand)
                observed_pred = self.likelihood(predictions)

            mean = observed_pred.mean.detach().to(self.device)
            covar = observed_pred.covariance_matrix.detach().to(self.device)
            thr = torch.Tensor([0.]).to(self.device) 

            # Use the selector to choose a set of points based on the  mean and covariance matrix.
            points = set(selector(N=N, gp_mean=mean - thr, gp_covar=covar))  
            new_x = x_test_rand[list(points)]
            
            # Unnormalize the selected points to return them to their original scale.
            new_x_unnormalized = self._unnormalize(new_x, self.data_min, self.data_max)

        # Debugging and informational prints to check the selected points and their corresponding values.
        print("Selected points (indices):", points)  # Indices of the selected points
        print("Selected new x values (normalized):", new_x)  # The normalized new x values chosen by the selector
        print("Corresponding new x values (unnormalized):", new_x_unnormalized)  # The unnormalized values of the selected points
        
        return new_x, new_x_unnormalized  # Return both the normalized and unnormalized selected points
    
    def random_new_points(self, N=4):
        # Select random indices from the available test points
        random_indices = np.random.choice(self.x_test.shape[0], N, replace=False)  # Randomly select N indices
        new_x = self.x_test[random_indices]

        # Unnormalize the selected points to return them to their original scale.
        new_x_unnormalized = self._unnormalize(new_x, self.data_min, self.data_max)

        return new_x, new_x_unnormalized