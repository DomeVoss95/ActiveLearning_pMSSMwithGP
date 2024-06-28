import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.likelihoods import GaussianLikelihood

import numpy as np
import os
import uproot

%matplotlib inline
%load_ext autoreload 
%autoreload 2


# Define the base directory and the range of subdirectories
base_dir = "/eos/user/d/dvoss/Run3ModelGen/6249447/"
subdir_range = range(0, 20)  # Adjust range to include subdirectories to 20

# List to store data from all files
all_data = []

for subdir in subdir_range:
    file_path = os.path.join(base_dir, str(subdir), f"ntuple.6249447.{subdir}.root")
    
    try:
        # Open the ROOT file
        file = uproot.open(file_path)
        
        # Assuming the data you want is in a specific tree
        tree_name = "susy"  # Replace with the actual tree name
        tree = file[tree_name]
        
        # Convert tree to pandas DataFrame or any other format you prefer
        df = tree.arrays(library="pd")
        
        # Append the DataFrame to the list
        all_data.append(df)
        
    except Exception as e:
        print(f"Failed to open or process {file_path}: {e}")

# Concatenate all DataFrames into a single DataFrame if using pandas
import pandas as pd
final_df = pd.concat(all_data, ignore_index=True)


M_1 = final_df['IN_M_1']
M_2 = final_df['IN_M_2']
Omega = final_df['MO_Omega']

# Filter Omega, M_1, and M_2 up to max 1.0 and limit to only 100 points
mask = (Omega > 0) & (Omega < 1.0)

# Apply the mask to filter M_1, M_2, and Omega
M_1_filtered = M_1[mask]
M_2_filtered = M_2[mask]
Omega_filtered = Omega[mask]

# Limit to only 100 points
M_1_limited = M_1_filtered.iloc[:5000]
M_2_limited = M_2_filtered.iloc[:5000]
Omega_limited = Omega_filtered.iloc[:5000]

# Convert to train
x_train = torch.stack([torch.tensor(M_1_limited.values[:4000], dtype=torch.float32), torch.tensor(M_2_limited.values[:4000], dtype=torch.float32)], dim=1)
y_train = torch.log(torch.tensor(Omega_limited.values[:4000], dtype=torch.float32)/0.12)


# Convert to valid
x_valid = torch.stack([torch.tensor(M_1_limited.values[4000:], dtype=torch.float32), torch.tensor(M_2_limited.values[4000:], dtype=torch.float32)], dim=1)
y_valid = torch.log(torch.tensor(Omega_limited.values[4000:], dtype=torch.float32)/0.12)


from multitaskGP2D import MultitaskGP2D

# # Normalisierung der Daten
# x_train = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
# x_valid = (x_valid - x_valid.mean(dim=0)) / x_valid.std(dim=0)

# Normalisierung der Daten auf min = 0 und maximum = 1
# Min-Max-Normalisierung für x_train
x_train_min = x_train.min(dim=0, keepdim=True).values
x_train_max = x_train.max(dim=0, keepdim=True).values
x_train = (x_train - x_train_min) / (x_train_max - x_train_min)

# Min-Max-Normalisierung für x_valid
x_valid_min = x_valid.min(dim=0, keepdim=True).values
x_valid_max = x_valid.max(dim=0, keepdim=True).values
x_valid = (x_valid - x_valid_min) / (x_valid_max - x_valid_min)

likelihood = GaussianLikelihood()
multitask_gp = MultitaskGP2D(x_train, y_train, x_valid, y_valid, likelihood, 2)

best_multitask_gp, losses, losses_valid = multitask_gp.do_train_loop(iters=200)


# Erstellen des Plots
plt.plot(losses, label='training loss')
plt.plot(losses_valid, label='validation loss')

# Setzen der y-Achse auf logarithmische Skala
plt.yscale('log')

# Hinzufügen der Legende
plt.legend()

# Achsenbeschriftungen und Titel hinzufügen (optional)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss on Logarithmic Scale')

# Anzeigen des Plots
plt.show()



best_multitask_gp.state_dict()


params = best_multitask_gp.named_parameters()

for name, param in params:
    print(f"Name: {name}, Value: {param.data}")


x_test = torch.stack([torch.tensor(M_1_filtered.values[2000:10000], dtype=torch.float32), torch.tensor(M_2_filtered.values[2000:10000], dtype=torch.float32)], dim=1)

# Normalisierung der Daten
x_test = (x_test - x_test.mean(dim=0)) / x_test.std(dim=0)



# from PublishGP.utils import entropy_local
from entropy import entropy_local 

def evaluate(multitask_gp, likelihood, x_test):
    """Evaluate GP on test sample and calculate corresponding entropies."""

    multitask_gp.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get prediction:
        observed_pred = likelihood(multitask_gp(x_test))

        # Get upper and lower confidence bounds
        mean = observed_pred.mean.detach().reshape(-1, 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        var = observed_pred.variance.detach().reshape(-1, 1)
        thr = torch.Tensor([0.]) # Threshold in CDF calculation. Needs to be more sophisticated if params are scaled.

        entropy = entropy_local(mean, var, thr, device, dtype)
        
    return observed_pred, entropy

observed_pred, entropy = evaluate(multitask_gp, likelihood, x_test)


mean = observed_pred.mean.numpy()


x = x_test[:, 0]
y = x_test[:, 1]
z = mean

# 2D-Histogramm berechnen
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=z, density=True)

# Plotten der Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Mean log(Omega/0.12)')
plt.xlabel('M_1_normalized')
plt.ylabel('M_2_normalized')
plt.title('Gaussian Process Mean Heatmap')
plt.show()



# Predicted vs True
z = torch.tensor(mean) - torch.log(torch.tensor(Omega_filtered.values[2000:10000], dtype=torch.float32)/0.12)

# 2D-Histogramm berechnen
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=z, density=True)

# Plotten der Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Mean log(Omega/0.12)')
plt.xlabel('M_1_normalized')
plt.ylabel('M_2_normalized')
plt.title('Gaussian Process Mean Heatmap')
plt.show()


z = torch.log(torch.tensor(Omega_filtered.values[2000:10000], dtype=torch.float32)/0.12)

# 2D-Histogramm berechnen
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=z, density=True)

# Plotten der Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='log(Omega/0.12)')
plt.xlabel('M_1_normalized')
plt.ylabel('M_2_normalized')
plt.title('True Function Heatmap')
plt.show()

z = entropy

# 2D-Histogramm berechnen
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=z, density=True)

# Plotten der Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Entropy')
plt.xlabel('M_1_normalized')
plt.ylabel('M_2_normalized')
plt.title('Entropy Heatmap')

plt.show()


mean = observed_pred.mean.numpy()
lower, upper = observed_pred.confidence_region()

# Find the top 10 highest entropy values and their indices
topk_values, topk_indices = torch.topk(entropy, 10)
tenX = x_test[topk_indices]

maxE = torch.max(entropy)
maxIndex = torch.argmax(entropy)
maxX = x_test[maxIndex]



z = mean

# 2D-Histogramm berechnen
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=z, density=True)

# Extrahieren der Top 10 Punkte mit höchster Entropie
topk_indices = torch.argsort(entropy, descending=True)[:10]
top_10_points = x_test[topk_indices]

# Plotten der Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Mean log(Omega/0.12)')
plt.xlabel('M_1_normalized')
plt.ylabel('M_2_normalized')
plt.title('Gaussian Process Mean Heatmap')

# Hervorheben der Top 10 Punkte als rote Sterne
plt.scatter(top_10_points[:, 0], top_10_points[:, 1], marker='*', s=200, c='r', label='Top 10 High Entropy Points')

# Contour wo mean > 0
plt.contour(xedges[:-1], yedges[:-1], heatmap.T, levels=[0], colors='white', linewidths=2, linestyles='solid')

# Legende hinzufügen
plt.legend()

plt.show()