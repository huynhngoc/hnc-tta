import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

num_tta = 12

ous_org = pd.read_csv("OUS_original_results.csv")
ous_org = ous_org[["pid","f1_score"]]

ous_summarize = pd.read_csv(f"OUS_average_{num_tta:02d}.csv")
ous_summarize = ous_summarize[["pid","entropy_region", "actual_vol"]]

ous_df = pd.merge(ous_org, ous_summarize, on='pid', how='outer')


ous_cross_dsc = pd.read_csv(f"OUS_dice_{num_tta:02d}.csv")
ous_mean_cross_dice = ous_cross_dsc.groupby('pid').agg({'dice': 'mean'}).reset_index()
ous_mean_cross_dice.rename(columns={'dice': f'mean_dice_{num_tta:02d}'}, inplace=True)
ous_cross_dsc = ous_mean_cross_dice[["pid",f"mean_dice_{num_tta:02d}"]]
ous_df = pd.merge(ous_df, ous_cross_dsc, on='pid', how='outer')

ous_iou = pd.read_csv(f"OUS_iou_{num_tta:02d}.csv")
ous_df = pd.merge(ous_df, ous_iou, on='pid', how='outer')

maastro_org = pd.read_csv("MAASTRO_original_results.csv")
maastro_org = maastro_org[["pid","f1_score"]]

maastro_summarize = pd.read_csv(f"MAASTRO_average_{num_tta:02d}.csv")
maastro_summarize = maastro_summarize[["pid","entropy_region", "actual_vol"]]

maastro_df = pd.merge(maastro_org, maastro_summarize, on='pid', how='outer')


maastro_cross_dsc = pd.read_csv(f"MAASTRO_dice_{num_tta:02d}.csv")
maastro_mean_cross_dice = maastro_cross_dsc.groupby('pid').agg({'dice': 'mean'}).reset_index()
maastro_mean_cross_dice.rename(columns={'dice': f'mean_dice_{num_tta:02d}'}, inplace=True)
maastro_cross_dsc = maastro_mean_cross_dice[["pid",f"mean_dice_{num_tta:02d}"]]
maastro_df = pd.merge(maastro_df, maastro_cross_dsc, on='pid', how='outer')

maastro_iou = pd.read_csv(f"MAASTRO_iou_{num_tta:02d}.csv")
maastro_df = pd.merge(maastro_df, maastro_iou, on='pid', how='outer')




# Combine DataFrames
ous_df["source"] = "OUS"  # Add identifier column
maastro_df["source"] = "MAASTRO"  # Add identifier column
df = pd.concat([ous_df, maastro_df])  # Merge datasets


print('Working on IoU vs original dice score visualization.....')
# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["f1_score"], subset["iou"], label=source)
    # Add labels to points
    #for i, row in subset.iterrows():
        #plt.annotate(row["pid"], (row["f1_score"], row["iou"]), textcoords="offset points", xytext=(5,5), ha="left")

  
# Labels and title
plt.xlabel("Original DSC")
plt.ylabel('$IoU_{TTA}$')
#plt.title("Scatter Plot of IoU vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.0)


# Save the plot to a PDF file
plt.savefig(f'iou_vs_dice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on average cross dice score vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["f1_score"], subset[f"mean_dice_{num_tta:02d}"], label=source)
    #for i, row in subset.iterrows():
        #plt.annotate(row["pid"], (row["f1_score"], row[f"mean_dice_{num_tta:02d}"]), textcoords="offset points", xytext=(5,5), ha="left")
# Labels and title
plt.xlabel("Original DSC")
plt.ylabel(f"Mean Cross-DSC ({num_tta} TTA)")
#plt.title("Scatter Plot of Mean Cross Dice Score vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.0)


# Save the plot to a PDF file
plt.savefig(f'crossdice_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on sum entropy predicted region vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["f1_score"], subset["entropy_region"], label=source)#, alpha=0.7)
    #for i, row in subset.iterrows():
        #plt.annotate(row["pid"], (row["f1_score"], row["entropy_region"]), textcoords="offset points", xytext=(5,5), ha="left")
# Labels and title
plt.xlabel("Original DSC")
plt.ylabel("Entropy of predicted class 1 region")
plt.yticks(rotation=45)
#plt.title("Scatter Plot of Entropy in predicted region vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'entropy_region_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()


print('Working on volume vs original dice score visualization.....')

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # Share y-axis for better comparison

# First subplot: Original DSC vs. Actual Volume
for source, subset in df.groupby("source"):
    axes[0].scatter(subset["f1_score"], subset["actual_vol"], label=source)
    #for i, row in subset.iterrows():
        #axes[0].annotate(row["pid"], (row["f1_score"], row["actual_vol"]), textcoords="offset points", xytext=(5,5), ha="left")

axes[0].set_xlabel("Original DSC")
axes[0].set_ylabel("Number of GTV voxels")
axes[0].set_yticks(axes[0].get_yticks())  # Ensure consistent ticks
axes[0].tick_params(axis="y", rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Second subplot: Mean Cross-DSC vs. Actual Volume
for source, subset in df.groupby("source"):
    axes[1].scatter(subset[f"mean_dice_{num_tta:02d}"], subset["actual_vol"], label=source)
    #for i, row in subset.iterrows():
        #axes[1].annotate(row["pid"], (row[f"mean_dice_{num_tta:02d}"], row["actual_vol"]), textcoords="offset points", xytext=(5,5), ha="left")

axes[1].set_xlabel(f"Mean Cross-DSC ({num_tta} TTA)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig(f'gtv_volume_vs_dice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()