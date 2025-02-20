import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import h5py
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("num_tta", default=1, type=int) # Number of TTA predictions to analyze
    parser.add_argument("source")
    
    

    args, unknown = parser.parse_known_args()
    base_path = args.source + '/analysis/' + args.name 
    num_tta = args.num_tta

    #OUS_transformed_path = args.source + '/hnc-tta/analysis/CrossDiceVisualization/OUS_avg_cross_dice_analysis.csv'
    #MAASTRO_transformed_path = args.source + '/hnc-tta/analysis/CrossDiceVisualization/MAASTRO_avg_cross_dice_analysis.csv'
    
    OUS_transformed_path = args.source + '/OUS_analysis/OUS_avg_cross_dice_analysis.csv'
    MAASTRO_transformed_path = args.source + '/MAASTRO_analysis/MAASTRO_avg_cross_dice_analysis.csv'
    

    # Initialize an empty DataFrame to store the transformed data
    ous_transformed_df = pd.DataFrame()
    MAASTRO_transformed_df = pd.DataFrame()

    print('Working on OUS.....')
    for i in range(2, num_tta + 1):
        # Read the original CSV file
        ous_df = pd.read_csv(base_path + f'/OUS_analysis/dice_{i:02d}.csv')
        
        # Calculate the mean cross Dice score for each patient
        ous_mean_dice = ous_df.groupby('pid').agg({'dice': 'mean'}).reset_index()
        ous_mean_dice.rename(columns={'dice': f'mean_dice_{i:02d}'}, inplace=True)
        
        # Merge the mean cross Dice scores into the transformed DataFrame
        if ous_transformed_df.empty:
            ous_transformed_df = ous_mean_dice
        else:
            ous_transformed_df = pd.merge(ous_transformed_df, ous_mean_dice, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    ous_transformed_df.to_csv(OUS_transformed_path, index=False)

    print(f"Transformed data saved to {OUS_transformed_path}")

    print('Working on MAASTRO.....')
    for i in range(2, num_tta + 1):
        # Read the original CSV file
        MAASTRO_df = pd.read_csv(base_path + f'/MAASTRO_analysis/dice_{i:02d}.csv')
        
        # Calculate the mean cross Dice score for each patient
        MAASTRO_mean_dice = MAASTRO_df.groupby('pid').agg({'dice': 'mean'}).reset_index()
        MAASTRO_mean_dice.rename(columns={'dice': f'mean_dice_{i:02d}'}, inplace=True)
        
        # Merge the mean cross Dice scores into the transformed DataFrame
        if MAASTRO_transformed_df.empty:
            MAASTRO_transformed_df = MAASTRO_mean_dice
        else:
            MAASTRO_transformed_df = pd.merge(MAASTRO_transformed_df, MAASTRO_mean_dice, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    MAASTRO_transformed_df.to_csv(MAASTRO_transformed_path, index=False)

print(f"Transformed data saved to {MAASTRO_transformed_path}")