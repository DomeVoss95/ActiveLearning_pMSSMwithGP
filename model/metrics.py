import torch
import uproot
import os
import pandas as pd

class Metrics:
    def goodness_of_fit(self, test=None, csv_path=None):
        # Chi-Squared divided by degrees of freedom - Goodness of Fit for quantification and comparison of training

        # Open the ROOT file
        file = uproot.open(self.true_root_file_path)
        tree_name = "susy"
        tree = file[tree_name]
        df = tree.arrays(library="pd")

        M_1 = df['IN_M_1'].values
        M_2 = df['IN_M_2'].values
        Omega = df['MO_Omega'].values

        mask = Omega > 0
        M_1_filtered = self._normalize(M_1[mask], self.data_min, self.data_max)[0]
        M_2_filtered = self._normalize(M_2[mask], self.data_min, self.data_max)[0]
        Omega_filtered = Omega[mask]

        # Calculate the true values (log-scaled)
        true = torch.log(torch.tensor(Omega_filtered, dtype=torch.float32).to(self.device) / 0.12)

        # Evaluate model at M_1 and M_2 coordinates of true
        input_data = torch.stack([
            torch.tensor(M_1_filtered, dtype=torch.float32),
            torch.tensor(M_2_filtered, dtype=torch.float32)
        ], dim=1).to(self.device)

        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            predictions = self.model(input_data)

        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

        mean = torch.tensor(mean, dtype=torch.float32).to(self.device)
        upper = torch.tensor(upper, dtype=torch.float32).to(self.device)
        lower= torch.tensor(lower, dtype=torch.float32).to(self.device)

        # Define the various goodness_of_fit metrics
        def mean_squared():
            mse = torch.mean((mean - true) ** 2)
            rmse = torch.sqrt(mse)
            return mse.cpu().item(), rmse.cpu().item()
        
        def r_squared():
            ss_res = torch.sum((mean - true) ** 2)
            ss_tot = torch.sum((true - torch.mean(true)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            return r_squared.cpu().item()
        
        def chi_squared():
            pull = (mean - true) / (upper - lower)
            chi_squared = torch.sum(pull ** 2)
            dof = len(true) - 1  # Degrees of freedom
            reduced_chi_squared = chi_squared / dof
            return chi_squared.cpu().item(), reduced_chi_squared.cpu().item()

        def average_pull():
            absolute_pull = torch.abs((mean - true) / (upper - lower))
            mean_absolute_pull = torch.mean(absolute_pull)
            root_mean_squared_pull = torch.sqrt(torch.mean((absolute_pull) ** 2))
            return mean_absolute_pull.cpu().item(), root_mean_squared_pull.cpu().item()

        # Weights for adjusting to threshold
        thr = 0.0
        epsilon = 0.1 # Tolerance area around threshold, considered close
        weights = torch.exp(-((true - thr) ** 2) / (2 * epsilon ** 2))  

        def mean_squared_weighted():
            mse_weighted = torch.mean(weights * (mean - true) ** 2)
            rmse_weighted = torch.sqrt(mse_weighted)
            return mse_weighted.cpu().item(), rmse_weighted.cpu().item()

        def r_squared_weighted():
            ss_res_weighted = torch.sum(weights * (mean - true) ** 2)
            ss_tot_weighted = torch.sum(weights * (true - torch.mean(true)) ** 2)
            r_squared_weighted = 1 - ss_res_weighted / ss_tot_weighted
            return r_squared_weighted.cpu().item()
        
        def chi_squared_weighted():
            pull_weighted = (mean - true) / (upper - lower)
            chi_squared_weighted = torch.sum(weights * pull_weighted ** 2)
            dof = len(true) - 1  # Degrees of freedom
            reduced_chi_squared_weighted = chi_squared_weighted / dof
            return chi_squared_weighted.cpu().item(), reduced_chi_squared_weighted.cpu().item()
        
        def average_pull_weighted():
            absolute_pull_weighted = torch.abs((mean - true) / (upper - lower))
            mean_absolute_pull_weighted = torch.mean(weights * absolute_pull_weighted)
            root_mean_squared_pull_weighted = torch.sqrt(torch.mean(weights * (absolute_pull_weighted) ** 2))
            return mean_absolute_pull_weighted.cpu().item(), root_mean_squared_pull_weighted.cpu().item()

        # Calculate classification matrix and accuracy
        def accuracy():
    
            # # TODO: Classification with uncertainty    
            # mean_plus = mean + (upper - lower)/2
            # mean_minus = mean + (upper - lower)/2

            # if mean > 0 and true > 0: # If mean lies above true and the true values is above threshold -> correctly classifed
            #     TP += 1
            # elif mean > 0 and true < 0:
            #     FP += 1
            # elif mean < 0 and true > 0:
            #     FN += 1
            # elif mean < 0 and true < 0:
            #     TN += 1

            TP = ((mean > 0) & (true > 0)).sum().item()  # True Positives
            FP = ((mean > 0) & (true < 0)).sum().item()  # False Positives
            FN = ((mean < 0) & (true > 0)).sum().item()  # False Negatives
            TN = ((mean < 0) & (true < 0)).sum().item()  # True Negatives

            total = TP + FP + FN + TN
            accuracy = (TP + TN) / total if total > 0 else 0

            return accuracy

            
        # Dictionary to map test names to functions
        tests = {
            'mean_squared': mean_squared,
            'r_squared': r_squared,
            'chi_squared': chi_squared,
            'average_pull': average_pull,
            'mean_squared_weighted': mean_squared_weighted,
            'r_squared_weighted': r_squared_weighted,
            'chi_squared_weighted': chi_squared_weighted,
            'average_pull_weighted': average_pull_weighted,
            'accuarcy': accuracy
        }

        # If 'test' is None, run all tests and return the results
        if test is None:
            results = {}
            for test_name, test_function in tests.items():
                result = test_function()
                if isinstance(result, tuple):
                    for i, value in enumerate(result):
                        results[f"{test_name}_{i}"] = value  # Store each element of the tuple separately
                else:
                    results[test_name] = result  # Store single value directly

            # Save results to CSV if a path is provided
            if csv_path is not None:
                file_exists = os.path.isfile(csv_path)
                df_results = pd.DataFrame([results])  # Convert results to a DataFrame
                
                # Append data to CSV file, write header only if file does not exist
                df_results.to_csv(csv_path, mode='a', header=not file_exists, index=False)

            return results

        # Run a specific test
        if test in tests:
            result = tests[test]()
            print(f"Goodness of Fit ({test}): {result}")
            return result
        else:
            raise ValueError(f"Unknown test: {test}.")