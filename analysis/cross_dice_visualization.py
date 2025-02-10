import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print('Working on OUS.....')

# Read the CSV file
ous_df = pd.read_csv('OUS_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = ous_df['pid'].unique()

# Extract the time series column names
num_tta_cols = ous_df.columns[1:]  # Assuming the first column is 'pid'
# Define new x-axis labels using a one-line for loop

new_x_labels = [str(i) for i in range(2, 21)]

# Plot the data
plt.figure(figsize=(10, 6))

for pid in patient_ids:
    # Extract the time series values for the current patient
    patient_data = ous_df[ous_df['pid'] == pid][num_tta_cols].values.flatten()
    
    # Plot the time series data
    plt.plot(num_tta_cols, patient_data, label=f'Patient {pid}')

# Add labels and title
plt.xlabel('Number of TTA')
plt.ylabel('Mean Cross Dice Score')
plt.title('Mean Cross Dice Score vs. Number of TTA for OUS Dataset')
plt.legend(loc='best')

# Set new x-axis labels
plt.xticks(ticks=num_tta_cols, labels=new_x_labels)

# Save the plot to a PDF file
plt.savefig('OUS_Cross_Dice_NumTTA.pdf', format='pdf')

# Show the plot
plt.show()

print('Working on MAASTRO.....')

# Read the CSV file
MAASTRO_df = pd.read_csv('MAASTRO_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = MAASTRO_df['pid'].unique()

# Extract the time series column names
num_tta_cols = MAASTRO_df.columns[1:]  # Assuming the first column is 'pid'

# Plot the data
plt.figure(figsize=(10, 6))

for pid in patient_ids:
    # Extract the time series values for the current patient
    patient_data = MAASTRO_df[MAASTRO_df['pid'] == pid][num_tta_cols].values.flatten()
    
    # Plot the time series data
    plt.plot(num_tta_cols, patient_data, label=f'Patient {pid}')

# Add labels and title
plt.xlabel('Number of TTA')
plt.ylabel('Mean Cross Dice Score')
plt.title('Mean Cross Dice Score vs. Number of TTA for MAASTRO Dataset')
plt.legend(loc='best')

# Set new x-axis labels
plt.xticks(ticks=num_tta_cols, labels=new_x_labels)


# Save the plot to a PDF file
plt.savefig('MAASTRO_Cross_Dice_NumTTA.pdf', format='pdf')

# Show the plot
plt.show()