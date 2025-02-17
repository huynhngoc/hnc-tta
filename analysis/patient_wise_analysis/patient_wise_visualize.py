import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ous_df = pd.read_csv("OUS_patient_wise_analysis.csv")
maastro_df = pd.read_csv("MAASTRO_patient_wise_analysis.csv")

# Combine DataFrames
ous_df["source"] = "OUS"  # Add identifier column
maastro_df["source"] = "MAASTRO"  # Add identifier column
df = pd.concat([ous_df, maastro_df])  # Merge datasets


print('Working on IoU vs original dice score visualization.....')
# Create scatter plot
plt.figure(figsize=(8, 6))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["iou"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("Original Dice Score")
plt.ylabel("IoU")
plt.title("Scatter Plot of IoU vs. Original Dice Score")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('iou_vs_dice.pdf', format='pdf', bbox_inches='tight')

plt.show()


print('Working on average cross dice score vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 6))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["mean_dice_15"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("Original Dice Score")
plt.ylabel("Mean Cross Dice Score")
plt.title("Scatter Plot of Mean Cross Dice Score vs. Original Dice Score")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('crossdice_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()


print('Working on sum entropy vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 6))
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

print('Working on volume vs original dice score visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 6))
for source, subset in df.groupby("source"):
    plt.scatter(subset["original_dice_score"], subset["actual_vol"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("Original Dice Score")
plt.ylabel("Actual Volume")
plt.title("Scatter Plot of Actual Volume vs. Original Dice Score")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('volume_vs_orgdice.pdf', format='pdf', bbox_inches='tight')

plt.show()

print('Working on volume vs uncertainty visualization.....')

# Create scatter plot
plt.figure(figsize=(8, 6))
for source, subset in df.groupby("source"):
    plt.scatter(subset["mean_dice_15"], subset["actual_vol"], label=source)#, alpha=0.7)


# Labels and title
plt.xlabel("Mean Cross Dice Score")
plt.ylabel("Actual Volume")
plt.title("Scatter Plot of Actual Volume vs. Mean Cross Dice Score")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('volume_vs_crossdice.pdf', format='pdf', bbox_inches='tight')

plt.show()

