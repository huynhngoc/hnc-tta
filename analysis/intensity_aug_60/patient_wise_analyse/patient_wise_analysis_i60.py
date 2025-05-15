import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from matplotlib.ticker import FuncFormatter


"""
This script is used to analyze the relationship between the original DSC and patient-wise uncertainty metrics.
"""

num_tta = 15

ous_org = pd.read_csv("OUS_original_results.csv")
ous_org = ous_org[ous_org['pid'] != 110]
ous_org = ous_org[["pid","f1_score"]]

ous_summarize = pd.read_csv(f"OUS_average_{num_tta:02d}.csv")
ous_summarize = ous_summarize[["pid","entropy_org_pred_region", "actual_vol", "org_predicted_vol", "sum_entropy"]]

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
maastro_summarize = maastro_summarize[["pid","entropy_org_pred_region", "actual_vol", "org_predicted_vol", "sum_entropy"]]

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


df['entropy_region_norm'] = df['entropy_org_pred_region'] / df['org_predicted_vol']
df = df.dropna(subset=["entropy_region_norm"])

df["avg_entropy"] = df["sum_entropy"] / (173*191*265)

colors = {"OUS": "#d95f02", "MAASTRO": "#7570b3"}

maastro_pids_to_annotate = [69, 23, 52]
ous_pids_to_annotate = [ 217, 82]

"""print(df.head())
print(df.tail())
print(df.isna().sum())
nan_rows = df[df.isna().any(1)]
print(nan_rows)"""
spearman_corr_dict = {}
p_value_dict = {}


print('Working on IoU vs original dice score visualization.....')
# Create scatter plot
plt.figure(figsize=(4, 4))
for source, subset in df.groupby("source"):
    print(source)
    plt.scatter(subset["f1_score"], subset["iou"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')#, alpha=0.7)
    # Calculate Spearman's correlation coefficient
    correlation_both, p_value_both = spearmanr(df["f1_score"], df["iou"])
    spearman_corr_dict["original_dice_vs_iou"] = correlation_both
    p_value_dict["original_dice_vs_iou"] = p_value_both
    print(f"original_dice_vs_iou: {correlation_both}, p-value: {p_value_both}")

    correlation, p_value = spearmanr(subset["f1_score"], subset["iou"])
    print(f"{source} original_dice_vs_iou: {correlation}, p-value: {p_value}")
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_iou"] = correlation
    p_value_dict[f"{source} original_dice_vs_iou"] = p_value
    # Add labels to points
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            plt.annotate(row["pid"], (row["f1_score"], row["iou"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            plt.annotate(row["pid"], (row["f1_score"], row["iou"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')

  
# Labels and title
plt.xlabel("Original DSC", fontsize=11)
plt.ylabel('$IoU_{TTA}$', fontsize=11)
#plt.title("Scatter Plot of IoU vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)



# Save the plot to a PDF file
plt.savefig(f'iou_vs_dice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on IoU vs original dice score cropped visualization.....')
# Create scatter plot
plt.figure(figsize=(4, 4))
for source, subset in df.groupby("source"):
    print(source)
    plt.scatter(subset["f1_score"], subset["iou"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')#, alpha=0.7)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["f1_score"], subset["iou"])
    print(f"{source} original_dice_vs_iou: {correlation}, p-value: {p_value}")
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_iou"] = correlation
    p_value_dict[f"{source} original_dice_vs_iou"] = p_value
    # Add labels to points
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            plt.annotate(row["pid"], (row["f1_score"], row["iou"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            plt.annotate(row["pid"], (row["f1_score"], row["iou"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')

  
# Labels and title
plt.xlabel("Original DSC", fontsize=11)
plt.ylabel('$IoU_{TTA}$', fontsize=11)
#plt.title("Scatter Plot of IoU vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 0.2)
plt.xlim(0.0, 1.0)
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))


# Save the plot to a PDF file
plt.savefig(f'iou_vs_dice_{num_tta}_zoom.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on average cross dice score vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(4,4))
for source, subset in df.groupby("source"):
    print(source)
    plt.scatter(subset["f1_score"], subset[f"mean_dice_{num_tta:02d}"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')
    # Calculate Spearman's correlation coefficient
    correlation_both, p_value_both = spearmanr(df["f1_score"], df[f"mean_dice_{num_tta:02d}"])
    spearman_corr_dict[f"original_dice_vs_mean_cross_dice_{num_tta}"] = correlation_both
    p_value_dict[f"original_dice_vs_mean_cross_dice_{num_tta}"] = p_value_both

    # Calculate Pearson's correlation coefficient
    correlation_both, p_value_both = pearsonr(df["f1_score"], df[f"mean_dice_{num_tta:02d}"])
    print(f"Pearsons: original_dice_vs_iou: {correlation_both}, p-value: {p_value_both}")

    #spearman_corr_dict[f"original_dice_vs_mean_cross_dice_{num_tta}"] = correlation_both
    #p_value_dict[f"original_dice_vs_mean_cross_dice_{num_tta}"] = p_value_both

    correlation, p_value = spearmanr(subset["f1_score"], subset[f"mean_dice_{num_tta:02d}"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_mean_cross_dice_{num_tta}"] = correlation
    p_value_dict[f"{source} original_dice_vs_mean_cross_dice_{num_tta}"] = p_value
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            plt.annotate(row["pid"], (row["f1_score"], row[f"mean_dice_{num_tta:02d}"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            plt.annotate(row["pid"], (row["f1_score"], row[f"mean_dice_{num_tta:02d}"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')

# Labels and title
plt.xlabel("Original DSC", fontsize=11)
plt.ylabel(f"Mean Cross-DSC ({num_tta} TTA)", fontsize=11)
#plt.title("Scatter Plot of Mean Cross Dice Score vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)


# Save the plot to a PDF file
plt.savefig(f'crossdice_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on sum entropy predicted region vs original dice score visualization.....')


# Create scatter plot
plt.figure(figsize=(5, 3))
for source, subset in df.groupby("source"):
    print(source)
    plt.scatter(subset["f1_score"], subset["entropy_region_norm"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')#, alpha=0.7)
    correlation_both, p_value_both = spearmanr(df["f1_score"], df["entropy_region_norm"])
    spearman_corr_dict[f"original_dice_vs_entropy_region_norm"] = correlation_both
    p_value_dict[f"original_dice_vs_entropy_region_norm"] = p_value_both
    
    # Calculate Pearson's correlation coefficient
    correlation_both, p_value_both = pearsonr(df["f1_score"], df["entropy_region_norm"])
    print(f"Pearsons: original_dice_vs_entropy: {correlation_both}, p-value: {p_value_both}")

    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["f1_score"], subset["entropy_region_norm"])

    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_entropy_region_norm"] = correlation
    p_value_dict[f"{source} original_dice_vs_entropy_region_norm"] = p_value
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            plt.annotate(row["pid"], (row["f1_score"], row["entropy_region_norm"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            plt.annotate(row["pid"], (row["f1_score"], row["entropy_region_norm"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
# Labels and title
plt.xlabel("Original DSC", fontsize=11)
plt.ylabel("Entropy", fontsize=11)
plt.yticks(rotation=45)
#plt.title("Average entropy level inside the predicted GTV region \n as a function of Original DSC")
#plt.title("Sum of entropy of predicted class 1 region normalized \n by predicted class 1 volume as a function of Original DSC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0.0, 1.0)


# Save the plot to a PDF file
plt.savefig(f'entropy_region_normalized_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()
exit()
print('Working on average entropy  vs original dice score visualization.....')


# Create scatter plot
plt.figure(figsize=(5, 3))
for source, subset in df.groupby("source"):
    print(source)
    plt.scatter(subset["f1_score"], subset["avg_entropy"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')#, alpha=0.7)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["f1_score"], subset["avg_entropy"])
    print(f"{source} original_dice_vs_iou: {correlation}, p-value: {p_value}")

    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_avg_entropy"] = correlation
    p_value_dict[f"{source} original_dice_vs_avg_entropy"] = p_value
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            plt.annotate(row["pid"], (row["f1_score"], row["avg_entropy"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            plt.annotate(row["pid"], (row["f1_score"], row["avg_entropy"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
# Labels and title
plt.xlabel("Original DSC", fontsize=11)
plt.ylabel("Entropy", fontsize=11)
plt.yticks(rotation=45)
plt.title("Average entropy level across all voxels \n as a function of Original DSC")
#plt.title("Sum of entropy of predicted class 1 region normalized \n by predicted class 1 volume as a function of Original DSC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0.0, 1.0)


# Save the plot to a PDF file
plt.savefig(f'avg_entropy_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on volume vs original dice score visualization.....')

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(5, 6))  # Share y-axis for better comparison

# First subplot: Original DSC vs. Actual Volume
for source, subset in df.groupby("source"):
    print(source)
    axes[0].scatter(subset["f1_score"], subset["actual_vol"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["f1_score"], subset["actual_vol"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_actual_vol"] = correlation
    p_value_dict[f"{source} original_dice_vs_actual_vol"] = p_value
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            axes[0].annotate(row["pid"], (row["f1_score"], row["actual_vol"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            axes[0].annotate(row["pid"], (row["f1_score"], row["actual_vol"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')

axes[0].set_xlabel("Original DSC", fontsize=11)
axes[0].set_ylabel("Number of GTV voxels \n in ground truth segmentation", fontsize=11)
axes[0].set_yticks(axes[0].get_yticks())  # Ensure consistent ticks
axes[0].tick_params(axis="y", rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.0, 1.0)
axes[0].set_ylim(0.0,275000)



# Second subplot: Mean Cross-DSC vs. Actual Volume
for source, subset in df.groupby("source"):
    print(source)
    axes[1].scatter(subset[f"mean_dice_{num_tta:02d}"], subset["actual_vol"], label=source, color=colors[source], linewidths=0.2, edgecolors='black')
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset[f"mean_dice_{num_tta:02d}"], subset["actual_vol"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} cross_dice_vs_actual_vol"] = correlation
    p_value_dict[f"{source} cross_dice_vs_actual_vol"] = p_value
    for i, row in subset.iterrows():
        if row["pid"] in maastro_pids_to_annotate and source == "MAASTRO":
            axes[1].annotate(row["pid"], (row[f"mean_dice_{num_tta:02d}"], row["actual_vol"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')
        if row["pid"] in ous_pids_to_annotate and source == "OUS":
            axes[1].annotate(row["pid"], (row[f"mean_dice_{num_tta:02d}"], row["actual_vol"]), textcoords="offset points", xytext=(1,1), ha="left", fontsize=11, fontweight='bold')

axes[1].set_xlabel(f"Mean Cross-DSC ({num_tta} TTA)", fontsize=11)
axes[1].set_ylabel("Number of GTV voxels \n in ground truth segmentation", fontsize=11)
axes[1].tick_params(axis="y", rotation=45)
axes[1].set_yticks(axes[1].get_yticks())  # Ensure consistent ticks
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0.0, 1.0)
axes[1].set_ylim(0.0,275000)




# Adjust layout and save figure
plt.tight_layout()
plt.savefig(f'gtv_volume_vs_dice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

# Convert dictionary to a DataFrame
results = pd.DataFrame({
    "Comparison": spearman_corr_dict.keys(),
    "Spearman_Correlation": spearman_corr_dict.values(),
    "P-Value": p_value_dict.values()
})

# Save as a CSV file
results.to_csv(f"spearman_results_{num_tta}TTA.csv", index=False)




"""# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["f1_score"], subset["entropy_region"], label=source)#, alpha=0.7)
    for i, row in subset.iterrows():
        plt.annotate(row["pid"], (row["f1_score"], row["entropy_region"]), textcoords="offset points", xytext=(1,1), ha="left")
# Labels and title
plt.xlabel("Original DSC")
plt.ylabel("Entropy of predicted class 1 region")
plt.yticks(rotation=45)
#plt.title("Scatter Plot of Entropy in predicted region vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'entropy_region_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()"""

"""filtered_df = df[df['entropy_region']<20000]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for source, subset in df.groupby("source"):
    axes[0].scatter(subset["f1_score"], subset["entropy_region"], label=source)
    for i, row in subset.iterrows():
        axes[0].annotate(row["pid"], (row["f1_score"], row["entropy_region"]), textcoords="offset points", xytext=(1,1), ha="left")

axes[0].set_xlabel("Original DSC")
axes[0].set_ylabel("Entropy of predicted class 1 region")
#axes[0].set_yticks(axes[0].get_yticks())  # Ensure consistent ticks
#axes[0].tick_params(axis="y", rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3)


for source, subset in filtered_df.groupby("source"):
    axes[1].scatter(subset["f1_score"], subset["entropy_region"], label=source)
    for i, row in subset.iterrows():
        axes[1].annotate(row["pid"], (row["f1_score"], row["entropy_region"]), textcoords="offset points", xytext=(1,1), ha="left")

axes[1].set_xlabel("Original DSC")
axes[1].set_ylabel("Entropy of predicted class 1 region < 20000")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig(f'entropy_region_subplots.pdf', format='pdf', bbox_inches='tight')

plt.show()"""