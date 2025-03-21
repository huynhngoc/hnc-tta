import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("source")

args, unknown = parser.parse_known_args()

base_path = args.source + '/analysis/' + args.name

print('Base_path:', base_path)
print('Augmentation file:', args.name)

n = [4,8,12,16,20,24,28,32]

ous_dsc = {}
ous_sum_entropy = {}

for i in n:
    key = i
    ous_df = pd.read_csv(base_path + f'/OUS_analysis/average_{i:02d}.csv')
    ous_dsc_mean = ous_df["f1_score"].mean()
    ous_dsc_std = ous_df["f1_score"].std()
    ous_dsc[key] = (ous_dsc_mean, ous_dsc_std)
    ous_sum_entropy[key] = ous_df["sum_entropy"].sum()

# Convert dictionary to a DataFrame
ous_tta_dsc_analysis = pd.DataFrame({
    "Number of TTA": ous_dsc.keys(),
    "Mean DSC": ous_dsc.values()})

# Save as a CSV file
ous_tta_dsc_analysis.to_csv(base_path + '/OUS_analysis/ous_tta_dsc_analysis.csv', index=False)


ous_sum_entropy_diff = {}
keys = list(ous_sum_entropy.keys())  
for index in range(len(keys) - 1):  
    i = keys[index]
    j = keys[index + 1]  # Get the next key
    key = f"{i}_{j}"
    ous_sum_entropy_diff[key] = abs(ous_sum_entropy[i] - ous_sum_entropy[j])

ous_sum_entropy_diff_analysis = pd.DataFrame({
    "Transitions": ous_sum_entropy_diff.keys(),
    "Sum Entropy Difference": ous_sum_entropy_diff.values()})

ous_sum_entropy_diff_analysis.to_csv(base_path + '/OUS_analysis/ous_sum_entropy_diff_analysis.csv', index=False)




maastro_dsc = {}
maastro_sum_entropy = {}
for i in n:
    key = f"{i:02d}"
    maastro_df = pd.read_csv(base_path + f'/MAASTRO"_analysis/average_{i:02d}.csv')
    maastro_dsc_mean = maastro_df["f1_score"].mean()
    maastro_dsc_std = maastro_df["f1_score"].std()
    maastro_dsc[key] = (maastro_dsc_mean, maastro_dsc_std)
    maastro_sum_entropy[key] = maastro_df["sum_entropy"].sum()


# Convert dictionary to a DataFrame
maastro_tta_dsc_analysis = pd.DataFrame({
    "Number of TTA": maastro_dsc.keys(),
    "Mean DSC": maastro_dsc.values()})

# Save as a CSV file
maastro_tta_dsc_analysis.to_csv(base_path + '/MAASTRO_analysis/maastro_tta_dsc_analysis.csv', index=False)


maastro_sum_entropy_diff = {}
keys = list(maastro_sum_entropy.keys())  
for index in range(len(keys) - 1):  
    i = keys[index]
    j = keys[index + 1]  # Get the next key
    key = f"{i}_{j}"
    maastro_sum_entropy_diff[key] = abs(maastro_sum_entropy[i] - maastro_sum_entropy[j])

maastro_sum_entropy_diff_analysis = pd.DataFrame({
    "Transitions": maastro_sum_entropy_diff.keys(),
    "Sum Entropy Difference": maastro_sum_entropy_diff.values()})

maastro_sum_entropy_diff_analysis.to_csv(base_path + '/MAASTRO_analysis/maastro_sum_entropy_diff_analysis.csv', index=False)
