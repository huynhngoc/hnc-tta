import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print('Working on IoU vs original dice score visualization.....')

ous_df = pd.read_csv("OUS_iou_vs_original_dice.csv")
maastro_df = pd.read_csv("MAASTRO_iou_vs_original_dice.csv")

# Combine DataFrames
ous_df["source"] = "OUS"  # Add identifier column
maastro_df["source"] = "MAASTRO"  # Add identifier column
df = pd.concat([ous_df, maastro_df])  # Merge datasets

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

ous_cross_dice = pd.read_csv("../CrossDiceVisualization/OUS_avg_cross_dice_analysis.csv")
maastro_cross_dice = pd.read_csv("../CrossDiceVisualization/MAASTRO_avg_cross_dice_analysis.csv")

ous_cross_dice = ous_cross_dice[['pid','mean_dice_15']]
maastro_cross_dice = maastro_cross_dice[['pid','mean_dice_15']]


ous_transformed_df = pd.merge(ous_cross_dice, ous_df, on='pid', how='outer')
maastro_transformed_df = pd.merge(maastro_cross_dice, maastro_df, on='pid', how='outer')

# Combine DataFrames
ous_transformed_df["source"] = "OUS"  # Add identifier column
maastro_transformed_df["source"] = "MAASTRO"  # Add identifier column
df = pd.concat([ous_transformed_df, maastro_transformed_df])  # Merge datasets

print(df.head())
print(df.tail())
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

