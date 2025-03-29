import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ous_org = pd.read_csv("OUS_original_results.csv")
ous_tta = pd.read_csv("OUS_average_15.csv")

maastro_org = pd.read_csv("MAASTRO_original_results.csv")
maastro_tta = pd.read_csv("MAASTRO_average_15.csv")

# Create a DataFrame for plotting
ous_org["Model"] = "No TTA"
ous_tta["Model"] = "Average of 15 \n TTA-predictions"
maastro_org["Model"] = "No TTA"
maastro_tta["Model"] = "Average of 15 \n TTA-predictions"

ous_df = pd.concat([ous_org[["f1_score", "Model"]], ous_tta[["f1_score", "Model"]]])
maastro_df = pd.concat([maastro_org[["f1_score", "Model"]], maastro_tta[["f1_score", "Model"]]])

# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
# Custom colors
custom_palette = {"No TTA": "#1f78b4", "Average of 15 \n TTA-predictions": "#b2df8a"}

# Violin plot for OUS data
sns.violinplot(x="Model", y="f1_score", data=ous_df, ax=axes[0], cut=0, palette=custom_palette)  # Cut=0 restricts to min/max
#sns.stripplot(x="Model", y="f1_score", data=ous_df, ax=axes[0], color="black", size=4, jitter=True, alpha=0.7)  # Show individual points
axes[0].set_title("A: Distribution of DSC \n for patients in OUS dataset", fontsize=11)
axes[0].set_xlabel("")
axes[0].set_ylabel("DSC", fontsize=11)
axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(axes[0].get_xticklabels(),fontsize=11)





# Violin plot for MAASTRO data
sns.violinplot(x="Model", y="f1_score", data=maastro_df, ax=axes[1], cut=0, palette=custom_palette)
#sns.stripplot(x="Model", y="f1_score", data=maastro_df, ax=axes[1], color="black", size=4, jitter=True, alpha=0.7)
axes[1].set_title("B: Distribution of DSC \n for patients in MAASTRO dataset", fontsize=11)
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(axes[0].get_xticklabels(),fontsize=11)






# Adjust layout
plt.tight_layout()
plt.savefig('dsc_distribution.png')
plt.show()