import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import h5py
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("source")
    parser.add_argument("--num_tta", default=1, type=int)
    

    args, unknown = parser.parse_known_args()
    base_path = args.source + '/analysis/' + args.name 
    num_tta = args.num_tta

    OUS_transformed_path = base_path + '/OUS_analysis/OUS_avg_cross_dice_analysis.csv'
    MAASTRO_transformed_path = base_path + '/MAASTRO_analysis/MAASTRO_avg_cross_dice_analysis.csv'
    
    # Initialize an empty DataFrame to store the transformed data
    transformed_df = pd.DataFrame()

    print('Working on OUS.....')
    for i in range(2, num_tta + 1):
        # Read the original CSV file
        df = pd.read_csv(base_path + f'/OUS_analysis/dice_{i:02d}.csv')
        df.groupby('pid').agg({'dice': 'mean'}).reset_index()
        # Calculate the mean cross Dice score for each patient
        mean_dice = df.groupby('pid').agg({'dice': 'mean'}).reset_index()
        mean_dice.rename(columns={'dice': f'mean_dice_{i:02d}'}, inplace=True)
        
        # Merge the mean cross Dice scores into the transformed DataFrame
        if transformed_df.empty:
            transformed_df = mean_dice
        else:
            transformed_df = pd.merge(transformed_df, mean_dice, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    transformed_df.to_csv(OUS_transformed_path, index=False)

    print(f"Transformed data saved to {OUS_transformed_path}")

    print('Working on MAASTRO.....')
    for i in range(2, num_tta + 1):
        # Read the original CSV file
        df = pd.read_csv(base_path + f'/MAASTRO_analysis/dice_{i:02d}.csv')
        df.groupby('pid').agg({'dice': 'mean'}).reset_index()
        # Calculate the mean cross Dice score for each patient
        mean_dice = df.groupby('pid').agg({'dice': 'mean'}).reset_index()
        mean_dice.rename(columns={'dice': f'mean_dice_{i:02d}'}, inplace=True)
        
        # Merge the mean cross Dice scores into the transformed DataFrame
        if transformed_df.empty:
            transformed_df = mean_dice
        else:
            transformed_df = pd.merge(transformed_df, mean_dice, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    transformed_df.to_csv(MAASTRO_transformed_path, index=False)

print(f"Transformed data saved to {MAASTRO_transformed_path}")