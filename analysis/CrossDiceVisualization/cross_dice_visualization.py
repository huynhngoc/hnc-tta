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

# Calculate the mean of all the values in each column to plot the mean line
ous_mean_values = ous_df[num_tta_cols].mean()

# Plot the data
#plt.figure(figsize=(7, 4))
# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

for pid in patient_ids:
    # Extract the mean cross dice values for the current patient
    patient_data = ous_df[ous_df['pid'] == pid][num_tta_cols].values.flatten()
    
    if ous_df[ous_df['pid'] == pid]['mean_dice_02'].values[0] < 0.4:# and ous_df[ous_df['pid'] == pid]['mean_dice_1'].values[0] > 0.95:
        print(f'Patient ID: {pid}') # Print the patient ID for a patient with high uncertianty
        ax1.plot(num_tta_cols, patient_data, label=f'Patient ID = {pid}', alpha=1, linewidth=2.5)
        continue
    
    if ous_df[ous_df['pid'] == pid]['mean_dice_02'].values[0] == 1 and ous_df[ous_df['pid'] == pid]['mean_dice_10'].values[0] == 1:
        print(f'Patient ID: {pid}') # Print the patient ID for a patient with high uncertianty
        ax1.plot(num_tta_cols, patient_data, label=f'Patient ID = {pid}', alpha=1, linewidth=2.5)
        continue

    if ous_df[ous_df['pid'] == pid]['mean_dice_02'].values[0] > 0.86 and ous_df[ous_df['pid'] == pid]['mean_dice_02'].values[0] < 0.9 and ous_df[ous_df['pid'] == pid]['mean_dice_04'].values[0] < 0.75:
        print(f'Patient ID: {pid}') # Print the patient ID for a patient with high uncertianty
        ax1.plot(num_tta_cols, patient_data, label=f'Patient ID = {pid}', alpha=1, linewidth=2.5)
        continue

    # Plot the data
    ax1.plot(num_tta_cols, patient_data, alpha=0.2)

# Plot the mean values
ax1.plot(num_tta_cols, ous_mean_values, label='Mean of all patients', color='black', linewidth=2, linestyle='--', alpha=1.0)

# Add labels and title
ax1.set_xlabel('Number of TTA-predictions')
ax1.set_ylabel('Mean Cross Dice Score')
#ax1.title('OUS Dataset : Mean Cross Dice Score vs. \n Number of TTA-predictions with example trends')
ax1.set_title('OUS Dataset')
ax1.legend(loc='lower left')

# Set new x-axis labels
ax1.set_xticks(num_tta_cols)
ax1.set_xticklabels(new_x_labels)


# Place the legend outside the plot
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Save the plot to a PDF file
# plt.savefig('OUS_Cross_Dice_All_Patients.pdf', format='pdf')

# Save the plot with mean line to a PDF file
#plt.savefig('OUS_Cross_Dice_All_Patients_w_mean.pdf', format='pdf')

# Show the plot
#plt.show()

print('Working on MAASTRO.....')

# Read the CSV file
MAASTRO_df = pd.read_csv('MAASTRO_avg_cross_dice_analysis.csv')

# Extract the patient IDs
patient_ids = MAASTRO_df['pid'].unique()

# Extract the column names
num_tta_cols = MAASTRO_df.columns[1:]  # Assuming the first column is 'pid'

# Calculate the mean of all the values in each column
MAASTRO_mean_values = ous_df[num_tta_cols].mean()

# Plot the data
#plt.figure(figsize=(7, 4))

for pid in patient_ids:
    # Extract the values for the current patient
    patient_data = MAASTRO_df[MAASTRO_df['pid'] == pid][num_tta_cols].values.flatten()
    if MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_02'].values[0] < 0.05 and MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_03'].values[0] < 0.05:
        print(f'Patient ID: {pid}') # Print the patient ID for a patient with high uncertianty
        ax2.plot(num_tta_cols, patient_data, label=f'Patient ID = {pid}', alpha=1, linewidth=2.5)
        continue
    
    if MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_03'].values[0] > 0.98 and MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_04'].values[0] < 0.52 and MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_20'].values[0] > 0.83:
        print(f'Patient ID: {pid}') # Print the patient ID for a patient with high uncertianty
        ax2.plot(num_tta_cols, patient_data, label=f'Patient ID = {pid}', alpha=1, linewidth=2.5)
        continue

    if MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_03'].values[0] > 0.98 and MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_06'].values[0] > 0.98 and MAASTRO_df[MAASTRO_df['pid'] == pid]['mean_dice_20'].values[0] > 0.98:
        print(f'Patient ID: {pid}') # Print the patient ID for a patient with high uncertianty
        ax2.plot(num_tta_cols, patient_data, label=f'Patient ID = {pid}', alpha=1, linewidth=2.5)
        continue

    # Plot the data
    ax2.plot(num_tta_cols, patient_data, alpha=0.2)

# Plot the mean values
ax2.plot(num_tta_cols, MAASTRO_mean_values, label='Mean of all patients', color='black', linewidth=2, linestyle='--', alpha=1.0)

# Add labels and title
ax2.set_xlabel('Number of TTA')
ax2.set_ylabel('Mean Cross Dice Score')
#ax2.title('MAASTRO Dataset : Mean Cross Dice Score vs. \n Number of TTA-predictions with example trends')
ax2.set_title('MAASTRO Dataset')
ax2.legend(loc='lower left')

# Set new x-axis labels
ax2.set_xticks(num_tta_cols)
ax2.set_xticklabels(new_x_labels)

#fig.suptitle('Mean Cross Dice Score vs. Number of TTA-predictions with example trends', y=0.97)

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.5)
#fig.subplots_adjust(top=0.90)

# Save the plot to a PDF file
#plt.savefig('MAASTRO_Cross_Dice_All_Patients.pdf', format='pdf')

# Save the plot with mean line to a PDF file
#plt.savefig('MAASTRO_Cross_Dice_All_Patients_w_mean.pdf', format='pdf')
plt.savefig('Cross_Dice_All_Patients.pdf', format='pdf')


# Show the plot
plt.show()