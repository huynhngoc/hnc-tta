import numpy as np
import pandas as pd


ous_df = pd.read_csv(f"OUS_original_results.csv")
ous_df = ous_df[ous_df['pid'] != 110]
maastro_df = pd.read_csv(f"MAASTRO_original_results.csv")

ous_dsc_mean = ous_df["f1_score"].mean()
ous_dsc_std = ous_df["f1_score"].std()
maastro_dsc_mean = maastro_df["f1_score"].mean()
maastro_dsc_std = maastro_df["f1_score"].std()

print(f"OUS DSC mean: {ous_dsc_mean:.3f} ± {ous_dsc_std:.3f}")
print(f"MAASTRO DSC mean: {maastro_dsc_mean:.3f} ± {maastro_dsc_std:.3f}")