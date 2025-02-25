import pandas as pd
from tabulate import tabulate

# OUS
iou_5 = pd.read_csv("ous_iou_05.csv")
iou_10 = pd.read_csv("ous_iou_10.csv")
iou_15 = pd.read_csv("ous_iou_15.csv", sep=";")
iou_20 = pd.read_csv("ous_iou_20.csv")


"""print(iou_5.head())
print(iou_10.head())
print(iou_15.head())
print(iou_20.head())"""

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
    "TTA": ["5 TTA", "10 TTA", "15 TTA", "20 TTA"],
    "OUS": [iou_5_null, iou_10_null, iou_15_null, iou_20_null],
    "MAASTRO": [maastro_iou_5_null, maastro_iou_10_null, maastro_iou_15_null, maastro_iou_20_null]
})

# Print as a formatted table
print(tabulate(results_df, headers="keys", tablefmt="github"))