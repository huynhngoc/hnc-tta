from scipy.stats import ranksums
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import h5py
import pandas as pd
import gc


mc_base_path = '../../segmentation/3d_unet_32_P10_aug_affine_04/'
ous_filename = '../segmentation/ous_test.h5'
maastro_filename = '../segmentation/maastro_full.h5'

with h5py.File(ous_filename, 'r') as f:
    ous_pids = list(f['y'].keys())
with h5py.File(maastro_filename, 'r') as f:
    maastro_pids = list(f['y'].keys())


data = []
for pid in ous_pids:
    gc.collect()
    print(pid)
    with h5py.File(ous_filename, 'r') as f:
        y_true = f['y'][pid][:]
        y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
    with open(mc_base_path + f'OUS_uncertainty_map/15/{pid}.npy', 'rb') as f:
        uncertainty_map = np.load(f)

    selected_TP = (y_true * y_pred) > 0
    selected_FP = ((1-y_true) * y_pred) > 0
    selected_FN = (y_true * (1-y_pred)) > 0
    selected_TN = ((1-y_true) * (1-y_pred)) > 0

    TP = uncertainty_map[selected_TP].flatten()
    FP = uncertainty_map[selected_FP].flatten()
    FN = uncertainty_map[selected_FN].flatten()
    TN = uncertainty_map[selected_TN].flatten()

    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'TP'} for d in TP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'FP'} for d in FP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'FN'} for d in FN])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'TN'} for d in TN])

for pid in maastro_pids:
    gc.collect()
    print(pid)
    if pid == '5':
        continue
    with h5py.File(maastro_filename, 'r') as f:
        y_true = f['y'][pid][:]
        y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
    with open(mc_base_path + f'MAASTRO_uncertainty_map/15/{pid}.npy', 'rb') as f:
        uncertainty_map = np.load(f)

    selected_TP = (y_true * y_pred) > 0
    selected_FP = ((1-y_true) * y_pred) > 0
    selected_FN = (y_true * (1-y_pred)) > 0
    selected_TN = ((1-y_true) * (1-y_pred)) > 0

    TP = uncertainty_map[selected_TP].flatten()
    FP = uncertainty_map[selected_FP].flatten()
    FN = uncertainty_map[selected_FN].flatten()
    TN = uncertainty_map[selected_TN].flatten()

    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'TP'} for d in TP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'FP'} for d in FP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'FN'} for d in FN])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'TN'} for d in TN])


df = pd.DataFrame(data)
df.to_csv('../entropy/mc_entropy_15_sample_area_all.csv', index=False)

# getting tta results
tta_base_path = '../analysis/noise_aug/'
data = []
for pid in ous_pids:
    gc.collect()
    print(pid)
    with h5py.File(ous_filename, 'r') as f:
        y_true = f['y'][pid][:]
        y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
    with open(tta_base_path + f'OUS_uncertainty_map/15/{pid}.npy', 'rb') as f:
        uncertainty_map = np.load(f)

    selected_TP = (y_true * y_pred) > 0
    selected_FP = ((1-y_true) * y_pred) > 0
    selected_FN = (y_true * (1-y_pred)) > 0
    selected_TN = ((1-y_true) * (1-y_pred)) > 0

    TP = uncertainty_map[selected_TP].flatten()
    FP = uncertainty_map[selected_FP].flatten()
    FN = uncertainty_map[selected_FN].flatten()
    TN = uncertainty_map[selected_TN].flatten()

    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'TP'} for d in TP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'FP'} for d in FP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'FN'} for d in FN])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'OUS', 'area': 'TN'} for d in TN])

for pid in maastro_pids:
    gc.collect()
    print(pid)
    if pid == '5':
        continue
    with h5py.File(maastro_filename, 'r') as f:
        y_true = f['y'][pid][:]
        y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
    with open(tta_base_path + f'MAASTRO_uncertainty_map/15/{pid}.npy', 'rb') as f:
        uncertainty_map = np.load(f)

    selected_TP = (y_true * y_pred) > 0
    selected_FP = ((1-y_true) * y_pred) > 0
    selected_FN = (y_true * (1-y_pred)) > 0
    selected_TN = ((1-y_true) * (1-y_pred)) > 0

    TP = uncertainty_map[selected_TP].flatten()
    FP = uncertainty_map[selected_FP].flatten()
    FN = uncertainty_map[selected_FN].flatten()
    TN = uncertainty_map[selected_TN].flatten()

    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'TP'} for d in TP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'FP'} for d in FP])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'FN'} for d in FN])
    data.extend(
        [{'pid': pid, 'entropy': d, 'center': 'MAASTRO', 'area': 'TN'} for d in TN])


df = pd.DataFrame(data)
df.to_csv('../entropy/tta_entropy_15_sample_area_all.csv', index=False)


mc_df = pd.read_csv('../entropy/mc_entropy_15_sample_area_all.csv')


sns.boxplot(data=mc_df, x='area', y='entropy', order=['TP', 'TN', 'FP', 'FN'], showmeans=True, hue='center', hue_order=['OUS', 'MAASTRO'],
            meanprops={'marker': 'o', 'markeredgecolor': 'black',
                       'markerfacecolor': 'firebrick'},
            flierprops={'marker': 'o', 'markeredgecolor': 'white', 'markeredgewidth': '0.5'}, )
plt.xlabel('')
ax = plt.gca()
ax.set_xticklabels(['True Positives', 'False Positives', 'False Negatives'])
plt.ylabel('Entropy')
plt.legend(bbox_to_anchor=(0.5, -0.15),
           loc='lower center', borderaxespad=0, ncol=2)
plt.subplots_adjust(bottom=0.15)
plt.savefig('../entropy/mc_entropy_15_sample_area_all.png')
plt.close('all')



tta_df = pd.read_csv('../entropy/tta_entropy_15_sample_area_all.csv')

sns.boxplot(data=tta_df, x='area', y='entropy', order=['TP', 'TN', 'FP', 'FN'], showmeans=True, hue='center', hue_order=['OUS', 'MAASTRO'],
            meanprops={'marker': 'o', 'markeredgecolor': 'black',
                       'markerfacecolor': 'firebrick'},
            flierprops={'marker': 'o', 'markeredgecolor': 'white', 'markeredgewidth': '0.5'}, )
plt.xlabel('')
ax = plt.gca()
ax.set_xticklabels(['True Positives', 'False Positives', 'False Negatives'])
plt.ylabel('Entropy')
plt.legend(bbox_to_anchor=(0.5, -0.15),
           loc='lower center', borderaxespad=0, ncol=2)
plt.subplots_adjust(bottom=0.15)
plt.savefig('../entropy/tta_entropy_15_sample_area_all.png')
plt.close('all')









# ranksums(df[df.area == 'FP']['entropy'], df[df.area == 'TP']['entropy'])
# ranksums(df[df.area == 'FN']['entropy'], df[df.area == 'TP']['entropy'])
# ranksums(df[(df.area == 'FN') & (df.center == 'MAASTRO')]['entropy'],
#          df[(df.area == 'FP') & (df.center == 'MAASTRO')]['entropy'])


# # TN mean
# data = []
# sum_tn = 0
# count = 0
# for pid in ous_pids:
#     gc.collect()
#     print(pid)
#     with h5py.File(ous_filename, 'r') as f:
#         y_true = f['y'][pid][:]
#         y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
#     with open(mc_base_path + f'OUS_uncertainty_map/15/{pid}.npy', 'rb') as f:
#         uncertainty_map = np.load(f)
#     selected_TN = ((1-y_true) * (1-y_pred)) > 0

#     TN = uncertainty_map[selected_TN].flatten()

#     sum_tn += TN.sum()
#     count += len(TN)

# mean_tn = sum_tn/count
# sse = 0
# thres = 0.04
# upper_thres = []
# for pid in ous_pids:
#     gc.collect()
#     print(pid)
#     with h5py.File(ous_filename, 'r') as f:
#         y_true = f['y'][pid][:]
#         y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
#     with open(mc_base_path + f'OUS_uncertainty_map/15/{pid}.npy', 'rb') as f:
#         uncertainty_map = np.load(f)

#     selected_TN = ((1-y_true) * (1-y_pred)) > 0
#     TN = uncertainty_map[selected_TN].flatten()

#     sse += (TN-mean_tn).sum()
#     upper_thres.extend([d for d in TN if d > thres])

# sd = np.math.sqrt(sse / count)
# print(count, mean_tn, sd)
# # 0.004315780026206776 2.5132249551418364e-05
# print(len(upper_thres), len(upper_thres) / count,
#       np.min(upper_thres),  np.max(upper_thres))
# # 2106424 0.006047185764280772 0.040000014 0.36787948

# # maastro
# sum_tn = 0
# count = 0
# for pid in maastro_pids:
#     gc.collect()
#     print(pid)
#     if pid == '5':
#         continue
#     with h5py.File(maastro_filename, 'r') as f:
#         y_true = f['y'][pid][:]
#         y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
#     with open(mc_base_path + f'MAASTRO_uncertainty_map/15/{pid}.npy', 'rb') as f:
#         uncertainty_map = np.load(f)

#     selected_TN = ((1-y_true) * (1-y_pred)) > 0
#     TN = uncertainty_map[selected_TN].flatten()
#     sum_tn += TN.sum()
#     count += len(TN)


# mean_tn = sum_tn/count
# sse = 0
# upper_thres = []
# for pid in maastro_pids:
#     gc.collect()
#     print(pid)
#     if pid == '5':
#         continue
#     with h5py.File(maastro_filename, 'r') as f:
#         y_true = f['y'][pid][:]
#         y_pred = (f['predicted'][pid][:] > 0.5).astype(float)
#     with open(mc_base_path + f'MAASTRO_uncertainty_map/15/{pid}.npy', 'rb') as f:
#         uncertainty_map = np.load(f)

#     selected_TN = ((1-y_true) * (1-y_pred)) > 0
#     TN = uncertainty_map[selected_TN].flatten()
#     sse += (TN - mean_tn).sum()
#     upper_thres.extend([d for d in TN if d > thres])

# sd = np.math.sqrt(sse / count)
# print(count, mean_tn, sd)
# # 984129242 0.004368372337249633 1.0930269148933532e-05
# print(len(upper_thres), len(upper_thres) / count, np.max(upper_thres))
