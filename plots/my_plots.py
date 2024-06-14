import uproot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Open the root file
file = uproot.open("/afs/cern.ch/user/d/dvoss/Run3ModelGen/run/scan/ntuple.0.0.root")

# Access the TTree
susy = file["susy"]
# Convert the TTree to a pandas DataFrame
df = susy.arrays(library="pd")

# Extract specific columns
desired_columns = ['IN_M_1', 'IN_M_2', 'MO_Omega']
extracted_df = df[desired_columns]


# Print the head of the extracted DataFrames
print(extracted_df)

# Find the index of the maximum value in the 'MO_Omega' column
max_index = df['MO_Omega'].idxmax()

# Drop the row with the maximum value
df.drop(max_index, inplace=True)
extracted_df.drop(max_index, inplace=True)

M_1 = df['IN_M_1']
M_2 = df['IN_M_2']
Omega = df['MO_Omega']

plt.scatter(M_1, M_2, c=Omega, s=25, marker='o', cmap='viridis')
plt.xlabel('M_1')
plt.ylabel('M_2')

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('MO_Omega')

M_1_filtered = [m_1 if m_1 < 130.0 else 0 for m_1 in M_1]

plt.plot(M_2_filtered)
plt.xlabel('Index')
plt.ylabel('M_1')
plt.title('M_1 values where M_1 < 130')
plt.show()

M_2_filtered = [m_2 if m_2 < 130.0 else 0 for m_2 in M_2]

plt.plot(M_2_filtered)
plt.xlabel('Index')
plt.ylabel('M_2')
plt.title('M_2 values where M_2 < 130')
plt.show()

# Assuming Omega is a list or array of values
Omega_filtered = [omega if omega < 0.1 else 0 for omega in Omega]

# Plot the filtered Omega values
plt.plot(Omega_filtered)
plt.xlabel('Index')
plt.ylabel('Omega')
plt.title('Omega values where Omega < 0.1')
plt.show()

# export data into file again