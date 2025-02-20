import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


ous_df = pd.read_csv("OUS_patient_wise_analysis.csv")
maastro_df = pd.read_csv("MAASTRO_patient_wise_analysis.csv")

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
plt.savefig('iou_vs_dice.pdf', format='pdf', bbox_inches='tight')

plt.show()


print('Working on average cross dice score vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(6, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["mean_dice_15"], label=source)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["original_dice_score"], subset["mean_dice_15"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} original_dice_vs_mean_cross_dice_15"] = correlation
    p_value_dict[f"{source} original_dice_vs_mean_cross_dice_15"] = p_value

# Labels and title
plt.xlabel("Original DSC")
plt.ylabel("Mean Cross Dice Score (15 TTA)")
#plt.title("Scatter Plot of Mean Cross Dice Score vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig('crossdice_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on sum entropy predicted region vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 4))
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
#plt.title("Scatter Plot of Entropy in predicted region vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig('entropy_region_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on volume vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 4))
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
#plt.title("Scatter Plot of Actual Volume vs. Original Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig('volume_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on volume vs uncertainty visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 4))
for source, subset in df.groupby("source"):
    plt.scatter(subset["mean_dice_15"], subset["actual_vol"], label=source)
    # Calculate Spearman's correlation coefficient
    correlation, p_value = spearmanr(subset["mean_dice_15"], subset["actual_vol"])
    # Store values in dictionaries
    spearman_corr_dict[f"{source} cross_dice_vs_actual_vol"] = correlation
    p_value_dict[f"{source} cross_dice_vs_actual_vol"] = p_value


# Labels and title
plt.xlabel("Mean Cross Dice Score")
plt.ylabel("Actual Volume")
#plt.title("Scatter Plot of Actual Volume vs. Mean Cross Dice Score")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to a PDF file
plt.savefig('volume_vs_crossdice.pdf', format='pdf', bbox_inches='tight')

plt.show()


# Convert dictionary to a DataFrame
results = pd.DataFrame({
    "Comparison": spearman_corr_dict.keys(),
    "Spearman_Correlation": spearman_corr_dict.values(),
    "P-Value": p_value_dict.values()
})

# Save as a CSV file
results.to_csv("spearman_results.csv", index=False)

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
