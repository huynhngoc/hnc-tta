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

# Function to find the x-axis value where a graph changes less than a certain threshold for several iterations
def find_threshold_index(values, threshold, num_iterations):
    differences = np.abs(np.diff(values))
    for i in range(len(differences) - num_iterations + 1):
        if np.all(differences[i:i + num_iterations] < threshold):
            return i # return the index where the graph enters 'steady state'
    return len(values)-1 # if the graph never enters 'steady state', return the last index

# Define the threshold and number of iterations
threshold = 0.005
num_iterations = 5


# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

ous_patient_threshold_index = []
MAASTRO_patient_threshold_index = []

# Plot the data for OUS
for pid in patient_ids:
    patient_data = ous_df[ous_df['pid'] == pid][num_tta_cols].values.flatten()
    ous_patient_threshold_index.append(find_threshold_index(patient_data, threshold, num_iterations))
    ax1.plot(num_tta_cols, patient_data, alpha=0.3)

ax1.set_xlabel('Number of TTA-predictions')
ax1.set_ylabel('Mean Cross Dice Score')
ax1.set_title('OUS Dataset')
#ax1.legend(loc='lower left')
ax1.set_xticks(num_tta_cols)
ax1.set_xticklabels(new_x_labels)

# Find mean threshold index and add vertical line for OUS
#ous_mean_threshold_index = int(np.mean(ous_patient_threshold_index))
#ous_std_threshold_index = np.std(ous_patient_threshold_index)
#ax1.axvline(x=num_tta_cols[ous_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
#ax1.axvline(x=ous_mean_threshold_index + ous_std_threshold_index, color='green', linestyle='--', label=f'std')

# Find mean and percentiles for OUS

ous_mean_threshold_index = int(np.mean(ous_patient_threshold_index))
ous_25_percentile_index = int(np.percentile(ous_patient_threshold_index, 25))
ous_75_percentile_index = int(np.percentile(ous_patient_threshold_index, 75))
ous_50_percentile_index = int(np.percentile(ous_patient_threshold_index, 50))

print(ous_patient_threshold_index)
print(ous_mean_threshold_index)
print(ous_25_percentile_index)
print(ous_75_percentile_index)
# Add vertical lines for mean and percentiles for OUS
ax1.axvline(x=num_tta_cols[ous_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
ax1.axvline(x=num_tta_cols[ous_25_percentile_index], color='green', linestyle='--', label='25th percentile')
ax1.axvline(x=num_tta_cols[ous_75_percentile_index], color='blue', linestyle='--', label='75th percentile')
ax1.axvline(x=num_tta_cols[ous_50_percentile_index], color='blue', linestyle='--', label='50th percentile')

# Shade the area between the mean and percentiles
ax1.axvspan(num_tta_cols[ous_25_percentile_index], num_tta_cols[ous_mean_threshold_index], color='green', alpha=0.1)
ax1.axvspan(num_tta_cols[ous_mean_threshold_index], num_tta_cols[ous_75_percentile_index], color='blue', alpha=0.1)
ax1.legend(loc='lower left')


print('Working on MAASTRO.....')

# Read the CSV file
MAASTRO_df = pd.read_csv('MAASTRO_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = MAASTRO_df['pid'].unique()

# Extract the column names
num_tta_cols = MAASTRO_df.columns[1:]  # Assuming the first column is 'pid'

# Plot the data for MAASTRO
for pid in patient_ids:
    patient_data = MAASTRO_df[MAASTRO_df['pid'] == pid][num_tta_cols].values.flatten()
    MAASTRO_patient_threshold_index.append(find_threshold_index(patient_data, threshold, num_iterations))
    ax2.plot(num_tta_cols, patient_data, alpha=0.3)

ax2.set_xlabel('Number of TTA-predictions')
ax2.set_ylabel('Mean Cross Dice Score')
ax2.set_title('MAASTRO Dataset')
#ax2.legend(loc='lower left')
ax2.set_xticks(num_tta_cols)
ax2.set_xticklabels(new_x_labels)

# # Find mean threshold index and add vertical line for MAASTRO
MAASTRO_mean_threshold_index = int(np.mean(MAASTRO_patient_threshold_index))
ax2.axvline(x=num_tta_cols[MAASTRO_mean_threshold_index], color='red', linestyle='--', label=f'Average steady state, Difference threshold = {threshold}')
ax2.legend(loc='lower left')

#fig.suptitle('Finding the average steady state', fontsize=12, y=0.95)
plt.tight_layout(pad=3) 

# Save the plot to a PDF file
plt.savefig('Cross_Dice_steady_state.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
