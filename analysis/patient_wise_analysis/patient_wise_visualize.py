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

ous_df = pd.read_csv(f"OUS_patient_wise_analysis_{num_tta}TTA.csv")
maastro_df = pd.read_csv(f"MAASTRO_patient_wise_analysis_{num_tta}TTA.csv")

# Combine DataFrames
ous_df["source"] = "OUS"  # Add identifier column
maastro_df["source"] = "MAASTRO"  # Add identifier column
df = pd.concat([ous_df, maastro_df])  # Merge datasets

spearman_corr_dict = {}
p_value_dict = {}

print('Working on IoU vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["iou"], label=source)

    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["original_dice_score"], subset["iou"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_iou"] = correlation
    p_value_dict[f"{source} original_dice_vs_iou"] = p_value
  
# Labels and title
plt.xlabel("Original DSC")
plt.ylabel('$IoU_{TTA}$')
#plt.title("Scatter Plot of IoU vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'iou_vs_dice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()


print('Working on average cross dice score vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset[f"mean_dice_{num_tta}"], label=source)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["original_dice_score"], subset[f"mean_dice_{num_tta}"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_mean_cross_dice_{num_tta}"] = correlation
    p_value_dict[f"{source} original_dice_vs_mean_cross_dice_{num_tta}"] = p_value

# Labels and title
plt.xlabel("Original DSC")
plt.ylabel(f"Mean Cross-DSC ({num_tta} TTA)")
#plt.title("Scatter Plot of Mean Cross Dice Score vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'crossdice_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on sum entropy predicted region vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["entropy_region"], label=source)#, alpha=0.7)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["original_dice_score"], subset["entropy_region"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_entropy_region"] = correlation
    p_value_dict[f"{source} original_dice_vs_entropy_region"] = p_value


# Labels and title
plt.xlabel("Original DSC")
plt.ylabel("Entropy of predicted region")
plt.yticks(rotation=45)
#plt.title("Scatter Plot of Entropy in predicted region vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'entropy_region_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on volume vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6.5, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["actual_vol"], label=source)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["original_dice_score"], subset["actual_vol"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_actual_vol"] = correlation
    p_value_dict[f"{source} original_dice_vs_actual_vol"] = p_value

# Labels and title
plt.xlabel("Original DSC")
plt.ylabel("Actual Volume")
plt.yticks(rotation=45)
#plt.title("Scatter Plot of Actual Volume vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'volume_vs_orgdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on volume vs uncertainty visualization.....')

# Create scatter plot
plt.figure(figsize=(6.5, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset[f"mean_dice_{num_tta}"], subset["actual_vol"], label=source)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset[f"mean_dice_{num_tta}"], subset["actual_vol"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} cross_dice_vs_actual_vol"] = correlation
    p_value_dict[f"{source} cross_dice_vs_actual_vol"] = p_value


# Labels and title
plt.xlabel("Mean Cross-DSC")
plt.ylabel("Actual Volume")
plt.yticks(rotation=45)
#plt.title("Scatter Plot of Actual Volume vs. Mean Cross Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig(f'volume_vs_crossdice_{num_tta}.pdf', format='pdf', bbox_inches='tight')

plt.show()


# Convert dictionary to a DataFrame
results = pd.DataFrame({
    "Comparison": spearman_corr_dict.keys(),
    "Spearman_Correlation": spearman_corr_dict.values(),
    "P-Value": p_value_dict.values()
})

# Save as a CSV file
results.to_csv(f"spearman_results_{num_tta}TTA.csv", index=False)

# Print to verify
print(results)


"""
print('Working on sum entropy vs original dice score visualization.....')
filtered_df = df[df['sum_entropy']<150000]
#print(filtered_df)

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["sum_entropy"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("Original Dice Score")
plt.ylabel("Sum Entropy")
plt.title("Scatter Plot of Sum Entropy vs. Original Dice Score")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('entropy_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in filtered_df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["sum_entropy"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("Original Dice Score")
plt.ylabel("Sum Entropy < 150000")
plt.title("Scatter Plot of Sum Entropy (< 150000) vs. Original Dice Score")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('entropy_filtered_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()




print('Working on volume vs uncertainty (IoU) visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["iou"], subset["actual_vol"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("IoU")
plt.ylabel("Actual Volume")
plt.title("Scatter Plot of Actual Volume vs. IoU")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('volume_vs_iou.pdf', format='pdf', bbox_inches='tight')

plt.show()

"""
