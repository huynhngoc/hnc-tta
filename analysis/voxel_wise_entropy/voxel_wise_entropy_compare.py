import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("num_tta")

args, unknown = parser.parse_known_args()

num_tta = args.num_tta

ous_df_tta = pd.read_csv(f"OUS_average_{num_tta}.csv")
maastro_df_tta = pd.read_csv(f"MAASTRO_average_{num_tta}.csv")

# Combine DataFrames
ous_df_tta["source"] = "OUS"  # Add identifier column
maastro_df_tta["source"] = "MAASTRO"  # Add identifier column
df_tta = pd.concat([ous_df_tta, maastro_df_tta])  # Merge datasets
df_tta["method"] = "TTA"

ous_df_mc = pd.read_csv(f"MCdropout_OUS_average_{num_tta}.csv", sep=';')
maastro_df_mc = pd.read_csv(f"MCdropout_MAASTRO_average_{num_tta}.csv", sep=',')
ous_df_mc["source"] = "OUS"  # Add identifier column
maastro_df_mc["source"] = "MAASTRO"  # Add identifier column
df_mc = pd.concat([ous_df_mc, maastro_df_mc])  # Merge datasets
df_mc["method"] = "MC"

#df_mc["pid"] = df_mc['pid'].astype(int)

#print(ous_df_mc.head())
#print(maastro_df_mc.head())
#print(df_mc.head())

#df = pd.concat([df_tta, df_mc])  # Merge datasets

df_tta['entropy_TP_norm'] = df_tta['entropy_TP'] / df_tta['TP_vol']
df_tta['entropy_FP_norm'] = df_tta['entropy_FP'] / df_tta['FP_vol']
df_tta['entropy_FN_norm'] = df_tta['entropy_FN'] / df_tta['FN_vol']

df_mc['entropy_TP_norm'] = df_mc['entropy_TP'] / df_mc['TP_vol']
df_mc['entropy_FP_norm'] = df_mc['entropy_FP'] / df_mc['FP_vol']
df_mc['entropy_FN_norm'] = df_mc['entropy_FN'] / df_mc['FN_vol']
#df['entropy_TN_norm'] = df['entropy_TN'] / df['TN_vol']
#df.to_csv(f"Method_compare_analysis.csv", index=False)
#exit()
"""
df['entropy_TP_norm'] = df['entropy_TP'] / df['TP_vol']
df['entropy_FP_norm'] = df['entropy_FP'] / df['FP_vol']
df['entropy_FN_norm'] = df['entropy_FN'] / df['FN_vol']
"""
TP_mean_tta = df_tta['entropy_TP_norm'].mean()
TP_median_tta = df_tta['entropy_TP_norm'].median()
TP_std_tta = df_tta['entropy_TP_norm'].std()
TP_25_tta = df_tta['entropy_TP_norm'].quantile(0.25)
TP_25_tta = df_tta['entropy_TP_norm'].quantile(0.25)
TP_75_tta = df_tta['entropy_TP_norm'].quantile(0.75)
print(f"TP mean: {TP_mean_tta}, std: {TP_std_tta}, median: {TP_median_tta}, 25th percentile: {TP_25_tta}, 75th percentile: {TP_75_tta}")

FP_mean_tta = df_tta['entropy_FP_norm'].mean()
FP_median_tta = df_tta['entropy_FP_norm'].median()
FP_std_tta = df_tta['entropy_FP_norm'].std()
FP_25_tta = df_tta['entropy_FP_norm'].quantile(0.25)
FP_75_tta = df_tta['entropy_FP_norm'].quantile(0.75)
print(f"FP mean: {FP_mean_tta}, std: {FP_std_tta}, median: {FP_median_tta}, 25th percentile: {FP_25_tta}, 75th percentile: {FP_75_tta}")

FN_mean_tta = df_tta['entropy_FN_norm'].mean()
FN_median_tta = df_tta['entropy_FN_norm'].median()
FN_std_tta = df_tta['entropy_FN_norm'].std()
FN_25_tta = df_tta['entropy_FN_norm'].quantile(0.25)
FN_75_tta = df_tta['entropy_FN_norm'].quantile(0.75)
print(f"FN mean: {FN_mean_tta}, std: {FN_std_tta}, median: {FN_median_tta}, 25th percentile: {FN_25_tta}, 75th percentile: {FN_75_tta}")

TP_mean_mc = df_mc['entropy_TP_norm'].mean()
TP_median_mc = df_mc['entropy_TP_norm'].median()
TP_std_mc = df_mc['entropy_TP_norm'].std()
TP_25_mc = df_mc['entropy_TP_norm'].quantile(0.25)
TP_25_mc = df_mc['entropy_TP_norm'].quantile(0.25)
TP_75_mc = df_mc['entropy_TP_norm'].quantile(0.75)
print(f"TP mean: {TP_mean_mc}, std: {TP_std_mc}, median: {TP_median_mc}, 25th percentile: {TP_25_mc}, 75th percentile: {TP_75_mc}")

FP_mean_mc = df_mc['entropy_FP_norm'].mean()
FP_median_mc = df_mc['entropy_FP_norm'].median()
FP_std_mc = df_mc['entropy_FP_norm'].std()
FP_25_mc = df_mc['entropy_FP_norm'].quantile(0.25)
FP_75_mc = df_mc['entropy_FP_norm'].quantile(0.75)
print(f"FP mean: {FP_mean_mc}, std: {FP_std_mc}, median: {FP_median_mc}, 25th percentile: {FP_25_mc}, 75th percentile: {FP_75_mc}")

FN_mean_mc = df_mc['entropy_FN_norm'].mean()
FN_median_mc = df_mc['entropy_FN_norm'].median()
FN_std_mc = df_mc['entropy_FN_norm'].std()
FN_25_mc = df_mc['entropy_FN_norm'].quantile(0.25)
FN_75_mc = df_mc['entropy_FN_norm'].quantile(0.75)
print(f"FN mean: {FN_mean_mc}, std: {FN_std_mc}, median: {FN_median_mc}, 25th percentile: {FN_25_mc}, 75th percentile: {FN_75_mc}")
exit()
"""
TN_mean = df['entropy_TN_norm'].mean()
TN_median = df['entropy_TN_norm'].median()
TN_std = df['entropy_TN_norm'].std()
TN_25 = df['entropy_TN_norm'].quantile(0.25)
TN_75 = df['entropy_TN_norm'].quantile(0.75)
print(f"TN mean: {TN_mean}, std: {TN_std}, median: {TN_median}, 25th percentile: {TN_25}, 75th percentile: {TN_75}")
"""

df_new_mc = pd.melt(df_mc, id_vars=['pid', 'source', 'method'], value_vars=['entropy_TP_norm', 'entropy_FP_norm', 'entropy_FN_norm'], var_name='class', value_name='Entropy')
df_new_tta = pd.melt(df_tta, id_vars=['pid', 'source', 'method'], value_vars=['entropy_TP_norm', 'entropy_FP_norm', 'entropy_FN_norm'], var_name='class', value_name='Entropy')

#print(df_new.head())

label_map = {
    'entropy_TP_norm': 'True Positives',
    'entropy_FP_norm': 'False Positives',
    'entropy_FN_norm': 'False Negatives',
}
#custom_palette = {"TTA": "lightseagreen", "MC": "orchid"}
custom_palette = {"OUS": "lightseagreen", "MAASTRO": "orchid"}


# Apply the mapping
df_new_tta['class'] = df_new_tta['class'].map(label_map)
df_new_mc['class'] = df_new_mc['class'].map(label_map)


# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# First boxplot on ax1
sns.boxplot(data=df_new_tta, x="class", y="Entropy", hue="source", showmeans=True, palette=custom_palette,
            meanprops={"marker": "o", "markerfacecolor": "indigo", "markeredgecolor": "black"}, ax=ax1)
ax1.set_xlabel("")
ax1.legend(title=None, loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2)
ax1.set_title("Test Time Augmentation")
# Second boxplot on ax2
sns.boxplot(data=df_new_mc, x="class", y="Entropy", hue="source", showmeans=True, palette=custom_palette, 
            meanprops={"marker": "o", "markerfacecolor": "indigo", "markeredgecolor": "black"}, ax=ax2)
ax2.set_xlabel("")
ax2.legend(title=None, loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2)
ax2.set_title("Monte Carlo Dropout")
plt.tight_layout(pad=3) 

# Save the figure
plt.savefig(f'voxel_wise_entropy_compare_subplot_{num_tta}.pdf', format='pdf', bbox_inches='tight')


plt.show()