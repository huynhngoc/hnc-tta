import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print('Working on OUS.....')

# Read the CSV file
ous_df = pd.read_csv('OUS_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = ous_df['pid'].unique()

# Extract the column names of the mean cross dice scores for different number of TTA-predictions
num_tta_cols = ous_df.columns[1:]  # Assuming the first column is 'pid'

# Define new x-axis labels
new_x_labels = [str(i) for i in range(2, 21)]

# Dividing patients into groups based on starting cross dice value
group1 = ous_df[ous_df['mean_dice_02'] < 0.7]
group2 = ous_df[ous_df['mean_dice_02'] >= 0.7]

# Calculate the mean of all the values in each column for each group
group1_mean_values = group1[num_tta_cols].mean()
group2_mean_values = group2[num_tta_cols].mean()

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot the data for group 1
for pid in group1['pid'].unique():
    patient_data = group1[group1['pid'] == pid][num_tta_cols].values.flatten()
    ax1.plot(num_tta_cols, patient_data, alpha=0.3)

ax1.plot(num_tta_cols, group1_mean_values, label='Mean of group 1', color='black', linewidth=2, linestyle='--', alpha=1.0)
ax1.set_xlabel('Number of TTA')
ax1.set_ylabel('Mean Cross Dice Score')
ax1.set_title('Group 1: Mean Cross Dice Score vs. Number of TTA')
ax1.legend(loc='lower left')
ax1.set_xticks(num_tta_cols)
ax1.set_xticklabels(new_x_labels)

# Plot the data for group 2
for pid in group2['pid'].unique():
    patient_data = group2[group2['pid'] == pid][num_tta_cols].values.flatten()
    ax2.plot(num_tta_cols, patient_data, alpha=0.3)

ax2.plot(num_tta_cols, group2_mean_values, label='Mean of group 2', color='black', linewidth=2, linestyle='--', alpha=1.0)
ax2.set_xlabel('Number of TTA')
ax2.set_ylabel('Mean Cross Dice Score')
ax2.set_title('Group 2: Mean Cross Dice Score vs. Number of TTA')
ax2.legend(loc='lower left')
ax2.set_xticks(num_tta_cols)
ax2.set_xticklabels(new_x_labels)

fig.suptitle('Mean Cross Dice Score vs. Number of TTA for OUS Dataset', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])   

# Save the plot to a PDF file
plt.savefig('OUS_Cross_Dice_2Groups.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

print('Working on MAASTRO.....')

# Read the CSV file
MAASTRO_df = pd.read_csv('MAASTRO_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = MAASTRO_df['pid'].unique()

# Extract the time series column names
num_tta_cols = MAASTRO_df.columns[1:]  # Assuming the first column is 'pid'

# Dividing patients into groups based on starting cross dice value
group1 = MAASTRO_df[MAASTRO_df['mean_dice_02'] < 0.7]
group2 = MAASTRO_df[MAASTRO_df['mean_dice_02'] >= 0.7]

# Calculate the mean of all the values in each column for each group
group1_mean_values = group1[num_tta_cols].mean()
group2_mean_values = group2[num_tta_cols].mean()

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot the data for group 1
for pid in group1['pid'].unique():
    patient_data = group1[group1['pid'] == pid][num_tta_cols].values.flatten()
    ax1.plot(num_tta_cols, patient_data, alpha=0.3)

ax1.plot(num_tta_cols, group1_mean_values, label='Mean of group 1', color='black', linewidth=2, linestyle='--', alpha=1.0)
ax1.set_xlabel('Number of TTA')
ax1.set_ylabel('Mean Cross Dice Score')
ax1.set_title('Group 1: Mean Cross Dice Score vs. Number of TTA')
ax1.legend(loc='lower left')
ax1.set_xticks(num_tta_cols)
ax1.set_xticklabels(new_x_labels)

# Plot the data for group 2
for pid in group2['pid'].unique():
    patient_data = group2[group2['pid'] == pid][num_tta_cols].values.flatten()
    ax2.plot(num_tta_cols, patient_data, alpha=0.3)

ax2.plot(num_tta_cols, group2_mean_values, label='Mean of group 2', color='black', linewidth=2, linestyle='--', alpha=1.0)
ax2.set_xlabel('Number of TTA')
ax2.set_ylabel('Mean Cross Dice Score')
ax2.set_title('Group 2: Mean Cross Dice Score vs. Number of TTA')
ax2.legend(loc='lower left')
ax2.set_xticks(num_tta_cols)
ax2.set_xticklabels(new_x_labels)

fig.suptitle('Mean Cross Dice Score vs. Number of TTA for MAASTRO Dataset', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

# Save the plot to a PDF file
plt.savefig('MAASTRO_Cross_Dice_2Groups.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()