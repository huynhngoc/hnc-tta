import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import h5py
import pandas as pd
from datetime import datetime



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
    startTime = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("source")

    args, unknown = parser.parse_known_args()

    base_path = args.source + '/analysis/' + args.name
    iter = args.iter

    print('Base_path:', base_path)
    print('Original model:', args.name)
    print('Iteration:', iter)

    if not os.path.exists(base_path):
        os.makedirs(base_path)


    ous_h5 = args.source + '/segmentation/ous_test.h5'
    ous_csv = args.source + '/segmentation/ous_test.csv'
    maastro_h5 = args.source + '/segmentation/maastro_full.h5'
    maastro_csv = args.source + '/segmentation/maastro_full.csv'

    # NOTE: exclude patient 5 from MAASTRO set
    # data = data[data.patient_idx != 5]

    
    if not os.path.exists(base_path + '/OUS_analysis'):
        os.makedirs(base_path + '/OUS_analysis')
    
    ous_df = pd.read_csv(ous_csv)
    ous_df = ous_df[ous_df['patient_idx'] != 110]


    data = []
    print('Working on OUS.....')
    for pid in ous_df.patient_idx:
        print('PID:', pid)

        with h5py.File(ous_h5, 'r') as f:
            y_true = f['y'][str(pid)][:]

        y_pred = []
        for i in range(1, iter+1):
            with open(args.source + f'/results/{args.name}/OUS/' + str(pid) + f'/{iter:02d}.npy', 'rb') as f:
                print(f"Appending prob.map from file: {args.source}/results/{args.name}/OUS/{str(pid)}/{iter:02d}.npy")
                y_pred.append(np.load(f))
        y_pred = np.stack(y_pred, axis=0).mean(axis=0)
        uncertainty_map = - y_pred * np.log(y_pred)
        true_positive = (y_pred > 0.5).astype(float) * y_true
        false_positive = (y_pred > 0.5).astype(float) - true_positive
        false_negative = y_true - true_positive
        true_negative = np.ones_like(y_true) - (true_positive + false_positive + false_negative)

        data.append({
            'pid': pid,
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'specificity': specificity(y_true, y_pred),
            'predicted_vol': (y_pred > 0.5).astype(float).sum(),
            'actual_vol': y_true.sum(),
            'TP_vol': (true_positive > 0).astype(float).sum(),
            'TN_vol': (true_negative > 0).astype(float).sum(),
            'FP_vol': (false_positive > 0).astype(float).sum(),
            'FN_vol': (false_negative > 0).astype(float).sum(),
            'sum_entropy': uncertainty_map.sum(),
            'mean_entropy': uncertainty_map.mean(),
            'entropy_region': (uncertainty_map[y_pred > 0.5]).sum(),
            'entropy_TP': (uncertainty_map[true_positive > 0]).sum(),
            'entropy_TN': (uncertainty_map[true_negative > 0]).sum(),
            'entropy_FP': (uncertainty_map[false_positive > 0]).sum(),
            'entropy_FN': (uncertainty_map[false_negative > 0]).sum(),
            'entropy_over_05_%': (uncertainty_map > 0.05).astype(float).sum(),
            'entropy_over_10_%': (uncertainty_map > 0.10).astype(float).sum(),
            'entropy_over_15_%': (uncertainty_map > 0.15).astype(float).sum(),
            'entropy_over_20_%': (uncertainty_map > 0.20).astype(float).sum(),
        })



    pd.DataFrame(data).to_csv(
        base_path + f'/OUS_analysis/average_{iter:02d}.csv', index=False
    )

    if not os.path.exists(base_path + '/MAASTRO_analysis'):
        os.makedirs(base_path + '/MAASTRO_analysis')

    data = []
    maastro_df = pd.read_csv(maastro_csv)
    print('Working on MAASTRO.....')
    for pid in maastro_df.patient_idx:
        print('PID:', pid)
    
        with h5py.File(maastro_h5, 'r') as f:
            y_true = f['y'][str(pid)][:]

        y_pred = []
        for i in range(1, iter+1):
            with open(args.source + f'/results/{args.name}/MAASTRO/' + str(pid) + f'/{iter:02d}.npy', 'rb') as f:
                y_pred.append(np.load(f))
        y_pred = np.stack(y_pred, axis=0).mean(axis=0)
        uncertainty_map = - y_pred * np.log(y_pred)
        true_positive = (y_pred > 0.5).astype(float) * y_true
        false_positive = (y_pred > 0.5).astype(float) - true_positive
        false_negative = y_true - true_positive
        true_negative = np.ones_like(y_true) - (true_positive + false_positive + false_negative)


        data.append({
            'pid': pid,
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'specificity': specificity(y_true, y_pred),
            'predicted_vol': (y_pred > 0.5).astype(float).sum(),
            'actual_vol': y_true.sum(),
            'TP_vol': (true_positive > 0).astype(float).sum(),
            'TN_vol': (true_negative > 0).astype(float).sum(),
            'FP_vol': (false_positive > 0).astype(float).sum(),
            'FN_vol': (false_negative > 0).astype(float).sum(),
            'sum_entropy': uncertainty_map.sum(),
            'mean_entropy': uncertainty_map.mean(),
            'entropy_region': (uncertainty_map[y_pred > 0.5]).sum(),
            'entropy_TP': (uncertainty_map[true_positive > 0]).sum(),
            'entropy_TN': (uncertainty_map[true_negative > 0]).sum(),
            'entropy_FP': (uncertainty_map[false_positive > 0]).sum(),
            'entropy_FN': (uncertainty_map[false_negative > 0]).sum(),
            'entropy_over_05_%': (uncertainty_map > 0.05).astype(float).sum(),
            'entropy_over_10_%': (uncertainty_map > 0.10).astype(float).sum(),
            'entropy_over_15_%': (uncertainty_map > 0.15).astype(float).sum(),
            'entropy_over_20_%': (uncertainty_map > 0.20).astype(float).sum(),
        })


    pd.DataFrame(data).to_csv(
        base_path + f'/MAASTRO_analysis/average_{iter:02d}.csv', index=False
    )

    print(f"Time taken: {datetime.now() - startTime}")
