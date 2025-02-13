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

# Dividing patients into groups based on the slope between mean cross dice value for 2 and 5 TTA-predictions
group1 = ous_df[ous_df['mean_dice_02'] < ous_df['mean_dice_05']]
group2 = ous_df[ous_df['mean_dice_02'] >= ous_df['mean_dice_05']]



# Function to find the x-axis value where a graph changes less than a certain threshold for several iterations
def find_threshold_index(values, threshold, num_iterations):
    differences = np.abs(np.diff(values))
    for i in range(len(differences) - num_iterations + 1):
        if np.all(differences[i:i + num_iterations] < threshold):
            return i # return the index where the graph enters 'steady state'
    return len(values) # if the graph never enters 'steady state', return the last index

# Define the threshold and number of iterations
threshold = 0.005
num_iterations = 5


# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

group1_patient_threshold_index = []
group2_patient_threshold_index = []

# Plot the data for group 1
for pid in group1['pid'].unique():
    patient_data = group1[group1['pid'] == pid][num_tta_cols].values.flatten()
    group1_patient_threshold_index.append(find_threshold_index(patient_data, threshold, num_iterations))
    ax1.plot(num_tta_cols, patient_data, alpha=0.3)

ax1.set_xlabel('Number of TTA-predictions')
ax1.set_ylabel('Mean Cross Dice Score')
ax1.set_title('Group 1: Mean Cross Dice Score for \n 2 TTA-predictions < 5 TTA-predictions')
#ax1.legend(loc='lower left')
ax1.set_xticks(num_tta_cols)
ax1.set_xticklabels(new_x_labels)

# Find mean threshold index and add vertical line for group 1
group1_mean_threshold_index = int(np.mean(group1_patient_threshold_index))
ax1.axvline(x=num_tta_cols[group1_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
ax1.legend(loc='lower left')

# Plot the data for group 2
for pid in group2['pid'].unique():
    patient_data = group2[group2['pid'] == pid][num_tta_cols].values.flatten()
    group2_patient_threshold_index.append(find_threshold_index(patient_data, threshold, num_iterations))
    ax2.plot(num_tta_cols, patient_data, alpha=0.3)

ax2.set_xlabel('Number of TTA-predictions')
ax2.set_ylabel('Mean Cross Dice Score')
ax2.set_title('Group 2: Mean Cross Dice Score for \n 2 TTA-predictions >= 5 TTA-predictions')
#ax2.legend(loc='lower left')
ax2.set_xticks(num_tta_cols)
ax2.set_xticklabels(new_x_labels)

# Find mean threshold index and add vertical line for group 2
group2_mean_threshold_index = int(np.mean(group2_patient_threshold_index))
ax2.axvline(x=num_tta_cols[group2_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
ax2.legend(loc='lower left')

fig.suptitle('OUS Dataset', fontsize=12, y=0.95)
plt.tight_layout(pad=3)

# Save the plot to a PDF file
plt.savefig('OUS_Cross_Dice_Groups_steady_state.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

print('Working on MAASTRO.....')

# Read the CSV file
MAASTRO_df = pd.read_csv('MAASTRO_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = MAASTRO_df['pid'].unique()

# Extract the column names
num_tta_cols = MAASTRO_df.columns[1:]  # Assuming the first column is 'pid'

# Dividing patients into groups based on the slope between mean cross dice value for 2 and 5 TTA-predictions
group1 = MAASTRO_df[MAASTRO_df['mean_dice_02'] < MAASTRO_df['mean_dice_05']]
group2 = MAASTRO_df[MAASTRO_df['mean_dice_02'] >= MAASTRO_df['mean_dice_05']]


# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
group1_patient_threshold_index = []
group2_patient_threshold_index = []

# Plot the data for group 1
for pid in group1['pid'].unique():
    patient_data = group1[group1['pid'] == pid][num_tta_cols].values.flatten()
    group1_patient_threshold_index.append(find_threshold_index(patient_data, threshold, num_iterations))
    ax1.plot(num_tta_cols, patient_data, alpha=0.3)

ax1.set_xlabel('Number of TTA-predictions')
ax1.set_ylabel('Mean Cross Dice Score')
ax1.set_title('Group 1: Mean Cross Dice Score for \n 2 TTA-predictions < 5 TTA-predictions')
#ax1.legend(loc='lower left')
ax1.set_xticks(num_tta_cols)
ax1.set_xticklabels(new_x_labels)

# # Find mean threshold index and add vertical line for group 1
group1_mean_threshold_index = int(np.mean(group1_patient_threshold_index))
ax1.axvline(x=num_tta_cols[group1_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
ax1.legend(loc='lower left')

# Plot the data for group 2
for pid in group2['pid'].unique():
    patient_data = group2[group2['pid'] == pid][num_tta_cols].values.flatten()
    group2_patient_threshold_index.append(find_threshold_index(patient_data, threshold, num_iterations))
    ax2.plot(num_tta_cols, patient_data, alpha=0.3)

ax2.set_xlabel('Number of TTA-predictions')
ax2.set_ylabel('Mean Cross Dice Score')
ax2.set_title('Group 2: Mean Cross Dice Score for \n 2 TTA-predictions >= 5 TTA-predictions')
#ax2.legend(loc='lower left')
ax2.set_xticks(num_tta_cols)
ax2.set_xticklabels(new_x_labels)

# Find mean threshold index and add vertical line for group 2
group2_mean_threshold_index = int(np.mean(group2_patient_threshold_index))
ax2.axvline(x=num_tta_cols[group2_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
ax2.legend(loc='lower left')

fig.suptitle('MAASTRO Dataset', fontsize=12, y=0.95)
plt.tight_layout(pad=3) 

# Save the plot to a PDF file
plt.savefig('MAASTRO_Cross_Dice_Groups_steady_state.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
