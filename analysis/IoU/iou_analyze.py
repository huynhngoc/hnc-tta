import pandas as pd
#from tabulate import tabulate
import matplotlib.pyplot as plt

# OUS
iou_5 = pd.read_csv("ous_iou_05.csv")
iou_10 = pd.read_csv("ous_iou_10.csv")
iou_15 = pd.read_csv("ous_iou_15.csv", sep=";")
iou_20 = pd.read_csv("ous_iou_20.csv")


"""print(iou_5.head())
print(iou_10.head())
print(iou_15.head())
print(iou_20.head())"""
#tol = 0.0001

iou_5_null = iou_5[iou_5['iou'] == 0].shape[0]
iou_10_null = iou_10[iou_10['iou'] == 0].shape[0]
iou_15_null = iou_15[iou_15['iou'] == 0].shape[0]
iou_20_null = iou_20[iou_20['iou'] == 0].shape[0]

"""
print(f"5 TTA: {iou_5_null}")
print(f"10 TTA: {iou_10_null}") 
print(f"15 TTA: {iou_15_null}")
print(f"20 TTA: {iou_20_null}")
"""

# MAASTRO
iou_5 = pd.read_csv("maastro_iou_05.csv", sep=",")
iou_10 = pd.read_csv("maastro_iou_10.csv", sep=",")
iou_15 = pd.read_csv("maastro_iou_15.csv", sep=",")
iou_20 = pd.read_csv("maastro_iou_20.csv", sep=",")


maastro_iou_5_null = iou_5[iou_5['iou'] == 0].shape[0]
maastro_iou_10_null = iou_10[iou_10['iou'] == 0].shape[0]
maastro_iou_15_null = iou_15[iou_15['iou'] == 0].shape[0]
maastro_iou_20_null = iou_20[iou_20['iou'] == 0].shape[0]

"""
print(f"5 TTA: {iou_5_null}")
print(f"10 TTA: {iou_10_null}") 
print(f"15 TTA: {iou_15_null}")
print(f"20 TTA: {iou_20_null}")
"""

#   Create DataFrame for the table
results_df = pd.DataFrame({
    "TTA": ["5", "10", "15", "20"],
    "OUS": [iou_5_null, iou_10_null, iou_15_null, iou_20_null],
    "MAASTRO": [maastro_iou_5_null, maastro_iou_10_null, maastro_iou_15_null, maastro_iou_20_null]
})

"""plt.plot(results_df["TTA"], results_df["OUS"], label="OUS")
plt.plot(results_df["TTA"], results_df["MAASTRO"], label="MAASTRO")
plt.legend()
plt.show()"""

plt.bar(results_df["TTA"], results_df["OUS"], label="OUS", color="lightseagreen")
plt.bar(results_df["TTA"], results_df["MAASTRO"], bottom=results_df["OUS"], label="MAASTRO", color="orchid")
plt.legend()
plt.xlabel("# of TTA-predictions")
plt.ylabel("# of patients with IoU=0")
plt.savefig("iou_nulls.png")
plt.show()

# Print as a formatted table
#print(tabulate(results_df, headers="keys", tablefmt="github"))

c#ustom_palette = {"OUS": "lightseagreen", "MAASTRO": "orchid"}