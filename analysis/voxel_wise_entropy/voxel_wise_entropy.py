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

ous_df = pd.read_csv(f"OUS_average_{num_tta}.csv")
maastro_df = pd.read_csv(f"MAASTRO_average_{num_tta}.csv")

# Combine DataFrames
ous_df["source"] = "OUS"  # Add identifier column
maastro_df["source"] = "MAASTRO"  # Add identifier column
df = pd.concat([ous_df, maastro_df])  # Merge datasets
#print(df.head())
#print(df.tail())

df['entropy_TP_norm'] = df['entropy_TP'] / df['TP_vol']
df['entropy_FP_norm'] = df['entropy_FP'] / df['FP_vol']
df['entropy_FN_norm'] = df['entropy_FN'] / df['FN_vol']

df_new = pd.melt(df, id_vars=['pid', 'source'], value_vars=['entropy_TP_norm', 'entropy_FP_norm', 'entropy_FN_norm'], var_name='class', value_name='Entropy')

label_map = {
    'entropy_TP_norm': 'True Positives',
    'entropy_FP_norm': 'False Positives',
    'entropy_FN_norm': 'False Negatives'
}

# Apply the mapping
df_new['class'] = df_new['class'].map(label_map)

plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=df_new, x="class", y="Entropy", hue="source", showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})
#ax.xaxis.labelpad = 10
ax.set_xlabel("")

plt.legend(title=None, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.savefig(f'voxel_wise_entropy_{num_tta}.pdf', format='pdf', bbox_inches='tight')


plt.show()