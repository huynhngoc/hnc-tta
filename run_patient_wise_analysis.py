import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import h5py
import pandas as pd


def f1_score(y_true, y_pred, beta=1):
    eps = 1e-8

    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true)
    target_positive = np.sum(y_true)
    predicted_positive = np.sum(y_pred)

    fb_numerator = (1 + beta ** 2) * true_positive + eps
    fb_denominator = (
        (beta ** 2) * target_positive + predicted_positive + eps
    )

    return fb_numerator / fb_denominator


def precision(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true)
    # target_positive = np.sum(y_true)
    predicted_positive = np.sum(y_pred)

    return true_positive / predicted_positive


def recall(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_positive = np.sum(y_pred * y_true)
    target_positive = np.sum(y_true)
    # predicted_positive = np.sum(y_pred)

    return true_positive / target_positive


def specificity(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(y_pred.dtype)
    if y_pred.ndim - y_true.ndim == 1 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    true_negative = np.sum((1 - y_true) * (1 - y_pred))
    negative_pred = np.sum(1 - y_pred)

    return true_negative / negative_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("num_tta", default=1, type=int)
    parser.add_argument("source")

    args, unknown = parser.parse_known_args()

    base_path = args.source + '/analysis/' + args.name
    num_tta = args.num_tta

    print('Base_path:', base_path)
    print('Original model:', args.name)
    print('Number of TTA:', num_tta)

    OUS_transformed_path = args.source + '/hnc-tta/analysis/patient_wise_analysis/OUS_patient_wise_analysis.csv'
    MAASTRO_transformed_path = args.source + '/hnc-tta/analysis/patient_wise_analysis/MAASTRO_patient_wise_analysis.csv'
    

    if not os.path.exists(base_path):
        os.makedirs(base_path)

   
    ous_h5 = args.source + '/segmentation/ous_test.h5'
    maastro_h5 = args.source + '/segmentation/maastro_full.h5'
   

    # NOTE: exclude patient 5 from MAASTRO set
    # data = data[data.patient_idx != 5]

    
    if not os.path.exists(base_path + '/OUS_analysis'):
        os.makedirs(base_path + '/OUS_analysis')

    ous_df = pd.read_csv(base_path + f'/OUS_analysis/iou_{num_tta:02d}.csv')
    ous_df["original_dice_score"] = 0  # Initialize the new column
        
  

    print('Working on OUS.....')
    for pid in ous_df.pid:
        print('PID:', pid)
        
        with h5py.File(ous_h5, 'r') as f:
            y_true = f['y'][str(pid)][:]
            y_pred = f['predicted'][str(pid)][:]

        ous_df.loc[ous_df["pid"] == pid, "original_dice_score"] = f1_score(y_true, y_pred)


    ous_cross_dice = pd.read_csv(base_path + f'/OUS_analysis/OUS_avg_cross_dice_analysis.csv')
    ous_cross_dice = ous_cross_dice[['pid',f'mean_dice_{num_tta:02d}']]

    ous_transformed_df = pd.merge(ous_cross_dice, ous_df, on='pid', how='outer')

    ous_summarize_df = pd.read_csv(base_path + f'/OUS_analysis/average_{num_tta:02d}.csv')
    ous_summarize_df = ous_summarize_df[['pid','actual_vol', 'sum_entropy']]

    ous_transformed_df = pd.merge(ous_transformed_df, ous_summarize_df, on='pid', how='outer')


    # Save the transformed data to a new CSV file
    ous_transformed_df.to_csv(OUS_transformed_path, index=False)


    if not os.path.exists(base_path + '/MAASTRO_analysis'):
        os.makedirs(base_path + '/MAASTRO_analysis')

    maastro_df = pd.read_csv(base_path + f'/MAASTRO_analysis/iou_{num_tta:02d}.csv')
    maastro_df["original_dice_score"] = 0  # Initialize the new column
    
    print('Working on MAASTRO.....')
    for pid in maastro_df.pid:
        print('PID:', pid)
      
        with h5py.File(maastro_h5, 'r') as f:
            y_true = f['y'][str(pid)][:]
            y_pred = f['predicted'][str(pid)][:]

        maastro_df.loc[maastro_df["pid"] == pid, "original_dice_score"] = f1_score(y_true, y_pred)
        
    maastro_cross_dice = pd.read_csv(base_path + f'/MAASTRO_analysis/MAASTRO_avg_cross_dice_analysis.csv')
    maastro_cross_dice = maastro_cross_dice[['pid',f'mean_dice_{num_tta:02d}']]

    maastro_transformed_df = pd.merge(maastro_cross_dice, maastro_df, on='pid', how='outer')

    maastro_summarize_df = pd.read_csv(base_path + f'/MAASTRO_analysis/average_{num_tta:02d}.csv')
    maastro_summarize_df = maastro_summarize_df[['pid','actual_vol', 'sum_entropy']]

    maastro_transformed_df = pd.merge(maastro_transformed_df, maastro_summarize_df, on='pid', how='outer')

    # Save the transformed data to a new CSV file
    maastro_transformed_df.to_csv(MAASTRO_transformed_path, index=False)