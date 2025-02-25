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
    
    OUS_transformed_path = base_path + '/OUS_analysis/OUS_iou_dev_analysis.csv'
    MAASTRO_transformed_path = base_path + '/MAASTRO_analysis/MAASTRO_iou_dev_analysis.csv'
    

    # Initialize an empty DataFrame to store the transformed data
    ous_transformed_df = pd.DataFrame()
    MAASTRO_transformed_df = pd.DataFrame()

    print('Working on OUS.....')
    for i in range(2, num_tta + 1):
        # Read the original CSV file
        ous_df = pd.read_csv(base_path + f'/OUS_analysis/iou_{i:02d}.csv')
    
        
        # Merge the iou values into the transformed DataFrame
        if ous_transformed_df.empty:
            ous_transformed_df = ous_df
        else:
            ous_transformed_df = pd.merge(ous_transformed_df, ous_df, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    ous_transformed_df.to_csv(OUS_transformed_path, index=False)

    print(f"Transformed data saved to {OUS_transformed_path}")

    print('Working on MAASTRO.....')
    for i in range(2, num_tta + 1):
        # Read the original CSV file
        MAASTRO_df = pd.read_csv(base_path + f'/MAASTRO_analysis/iou_{i:02d}.csv')
    
        
        # Merge the iou values into the transformed DataFrame
        if MAASTRO_transformed_df.empty:
            MAASTRO_transformed_df = MAASTRO_df
        else:
            MAASTRO_transformed_df = pd.merge(MAASTRO_transformed_df, MAASTRO_df, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    MAASTRO_transformed_df.to_csv(MAASTRO_transformed_path, index=False)

print(f"Transformed data saved to {MAASTRO_transformed_path}")