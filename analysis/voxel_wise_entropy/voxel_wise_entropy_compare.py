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
print(df_mc.tail())
exit()

df['entropy_TP_norm'] = df['entropy_TP'] / df['TP_vol']
df['entropy_FP_norm'] = df['entropy_FP'] / df['FP_vol']
df['entropy_FN_norm'] = df['entropy_FN'] / df['FN_vol']
df['entropy_TN_norm'] = df['entropy_TN'] / df['TN_vol']


TP_mean = df['entropy_TP_norm'].mean()
TP_median = df['entropy_TP_norm'].median()
TP_std = df['entropy_TP_norm'].std()
TP_25 = df['entropy_TP_norm'].quantile(0.25)
TP_25 = df['entropy_TP_norm'].quantile(0.25)
TP_75 = df['entropy_TP_norm'].quantile(0.75)
print(f"TP mean: {TP_mean}, std: {TP_std}, median: {TP_median}, 25th percentile: {TP_25}, 75th percentile: {TP_75}")

FP_mean = df['entropy_FP_norm'].mean()
FP_median = df['entropy_FP_norm'].median()
FP_std = df['entropy_FP_norm'].std()
FP_25 = df['entropy_FP_norm'].quantile(0.25)
FP_75 = df['entropy_FP_norm'].quantile(0.75)
print(f"FP mean: {FP_mean}, std: {FP_std}, median: {FP_median}, 25th percentile: {FP_25}, 75th percentile: {FP_75}")

FN_mean = df['entropy_FN_norm'].mean()
FN_median = df['entropy_FN_norm'].median()
FN_std = df['entropy_FN_norm'].std()
FN_25 = df['entropy_FN_norm'].quantile(0.25)
FN_75 = df['entropy_FN_norm'].quantile(0.75)
print(f"FN mean: {FN_mean}, std: {FN_std}, median: {FN_median}, 25th percentile: {FN_25}, 75th percentile: {FN_75}")

TN_mean = df['entropy_TN_norm'].mean()
TN_median = df['entropy_TN_norm'].median()
TN_std = df['entropy_TN_norm'].std()
TN_25 = df['entropy_TN_norm'].quantile(0.25)
TN_75 = df['entropy_TN_norm'].quantile(0.75)
print(f"TN mean: {TN_mean}, std: {TN_std}, median: {TN_median}, 25th percentile: {TN_25}, 75th percentile: {TN_75}")

exit()
df_new = pd.melt(df, id_vars=['pid', 'source'], value_vars=['entropy_TP_norm', 'entropy_FP_norm', 'entropy_FN_norm'], var_name='class', value_name='Entropy')

label_map = {
    'entropy_TP_norm': 'True Positives',
    'entropy_FP_norm': 'False Positives',
    'entropy_FN_norm': 'False Negatives',
}
custom_palette = {"OUS": "lightseagreen", "MAASTRO": "orchid"}

# Apply the mapping
df_new['class'] = df_new['class'].map(label_map)

plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=df_new, x="class", y="Entropy", hue="source", showmeans=True, palette=custom_palette, meanprops={"marker": "o", "markerfacecolor": "indigo", "markeredgecolor": "black"})
#ax.xaxis.labelpad = 10
ax.set_xlabel("")

plt.legend(title=None, loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2)

plt.savefig(f'voxel_wise_entropy_new_{num_tta}.pdf', format='pdf', bbox_inches='tight')


plt.show()