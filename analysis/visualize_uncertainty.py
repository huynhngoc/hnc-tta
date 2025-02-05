from mpl_toolkits.axes_grid1 import ImageGrid
from deoxys_image import normalize
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import colors as mpl_colors
from spekkhogger.singleblock import PCA
from statsmodels.stats.diagnostic import normal_ad
from scipy import stats
from matplotlib.widgets import Slider, Button
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os
import pingouin as pg
import seaborn as sns
from medvis import apply_cmap_with_blend


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


base_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/uncertainty'
name = '3d_unet_32_P10_aug_affine'
dropout = 4

path = f'{base_path}/{name}_{dropout:02d}'

ous_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/transfer/3d_unet_32_P10_aug_affine'
maastro_path = 'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/transfer/3d_unet_32_P10_aug_affine_maastro_clinic_3d'


class VisualizeUncertaintyMapV2:
    def __init__(self, image, mask, predicted, uncertainty_map, intersection, union):
        self.image = image
        self.mask = mask
        self.predicted = predicted
        self.uncertainty_map = uncertainty_map
        self.intersect = intersection
        self.union = union
        self.idx_0 = np.argmax(self.mask.sum(axis=(1, 2, 3)))
        self.idx_1 = np.argmax(self.mask.sum(axis=(0, 2, 3)))
        self.idx_2 = np.argmax(self.mask.sum(axis=(0, 1, 3)))
        self.uncertainty_threshold = 0.05
        self.image_data = [{}, {}, {}]
        self.mask_color = '#7fffd4'
        self.pred_color = '#603fef'
        # mask_color = '#7fffd4'
        # pred_color = '#603fef'

    def _setup_columns(self, dim, indice, image_2d, mask_2d, pred_2d, uncertainty_2d, intersection_2d, union_2d):
        data = self.image_data[dim]
        # 1st row
        ax_img = self.fig.add_subplot(3, 3, dim+1)
        ax_img.axis('off')
        data['ax_img'] = ax_img
        data['ct_0'] = ax_img.imshow(
            image_2d[..., 0], 'gray', vmin=0, vmax=1, origin='lower')
        data['pet'] = ax_img.imshow(
            apply_cmap_with_blend(image_2d[..., 1],
                                  'inferno', vmin=0, vmax=1), origin='lower')
        # data['pet'] = ax_img.imshow(image_2d[..., 1],'inferno', vmin=0, vmax=1, origin='lower')
        data['img_mask'] = ax_img.contour(
            mask_2d[..., 0], 1, levels=[0.5], colors=self.mask_color, origin='lower', linewidths=2)
        data['img_pred'] = ax_img.contour(
            pred_2d[..., 0], 1, levels=[0.5], colors=self.pred_color, origin='lower', linewidths=2)
        data['vline'] = ax_img.plot(
            [0, image_2d.shape[1]], [indice[0]+0.5, indice[0]+0.5], color='white', alpha=0.6, linewidth=1)[0]
        data['hline'] = ax_img.plot(
            [indice[1]+0.5, indice[1]+0.5], [0, image_2d.shape[0]], color='white', alpha=0.6, linewidth=1)[0]
        # remove padding due to lineplot
        ax_img.margins(0)

        # 2nd row
        ax_uncertain = self.fig.add_subplot(3, 3, dim+4)
        ax_uncertain.axis('off')
        data['ax_uncertain'] = ax_uncertain
        data['ct_1'] = ax_uncertain.imshow(
            image_2d[..., 0], 'gray', vmin=0, vmax=1, origin='lower')
        uncertainty_map = np.array(uncertainty_2d)
        uncertainty_map[uncertainty_map <
                        self.uncertainty_threshold] = np.nan
        data['uncertain_map'] = ax_uncertain.imshow(
            uncertainty_map, 'Reds', vmin=0,
            vmax=self.uncertainty_map.max(), origin='lower')
        data['uncertain_mask'] = ax_uncertain.contour(
            mask_2d[..., 0], 1, levels=[0.5], colors=self.mask_color, origin='lower', linewidths=2)
        data['uncertain_pred'] = ax_uncertain.contour(
            pred_2d[..., 0], 1, levels=[0.5], colors=self.pred_color, origin='lower', linewidths=2)

        # 3rd row
        ax_iou = self.fig.add_subplot(3, 3, dim+7)
        ax_iou.axis('off')
        data['ax_iou'] = ax_iou
        data['ct_2'] = ax_iou.imshow(
            image_2d[..., 0], 'gray', vmin=0, vmax=1, origin='lower')
        iou_i = np.array(intersection_2d)
        iou_u = np.array(union_2d)
        iou_i[iou_i == 0] = np.nan
        iou_u[iou_u == 0] = np.nan
        # iou_u[iou_i > 0] = np.nan
        data['iou_u'] = ax_iou.imshow(
            iou_u[..., 0], 'gnuplot', vmin=0, vmax=1, origin='lower')
        data['iou_i'] = ax_iou.imshow(
            iou_i[..., 0], 'gist_rainbow', vmin=0, vmax=1, origin='lower')
        data['iou_mask'] = ax_iou.contour(
            mask_2d[..., 0], 1, levels=[0.5], colors=self.mask_color, origin='lower', linewidths=2)
        data['iou_pred'] = ax_iou.contour(
            pred_2d[..., 0], 1, levels=[0.5], colors=self.pred_color, origin='lower', linewidths=2)

        if dim == 2:
            ax_uncertain.legend(
                [data['iou_mask'].legend_elements()[0][0],
                 data['iou_pred'].legend_elements()[0][0]],
                ['Ground Truth (g)', 'Predicted (c)'],
                loc='right', bbox_to_anchor=(1.8, 0.5),
                # ncol=2,
            )

        self.image_data[dim] = data

    def _setup(self):
        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(top=0.99, bottom=0.1, wspace=0.01,
                            hspace=0.01, left=0.05, right=0.85)
        self.fig = fig

        # 1st column
        self._setup_columns(
            0, [self.idx_1, self.idx_2],
            self.image[self.idx_0],
            self.mask[self.idx_0],
            self.predicted[self.idx_0],
            self.uncertainty_map[self.idx_0],
            self.intersect[self.idx_0],
            self.union[self.idx_0],
        )

        # 2nd column
        self._setup_columns(
            1, [self.idx_0, self.idx_2],
            self.image[:, self.idx_1],
            self.mask[:, self.idx_1],
            self.predicted[:, self.idx_1],
            self.uncertainty_map[:, self.idx_1],
            self.intersect[:, self.idx_1],
            self.union[:, self.idx_1],
        )

        # 3rd column
        self._setup_columns(
            2, [self.idx_0, self.idx_1],
            self.image[:, :, self.idx_2],
            self.mask[:, :, self.idx_2],
            self.predicted[:, :, self.idx_2],
            self.uncertainty_map[:, :, self.idx_2],
            self.intersect[:, :, self.idx_2],
            self.union[:, :, self.idx_2],
        )

        ax_slider0 = fig.add_axes([0.15, 0.07, 0.7, 0.03])
        self.slider0 = Slider(ax_slider0, 'Ax0 idx',
                              valmin=0, valmax=172, valstep=1, valinit=self.idx_0)
        self.slider0.on_changed(self._update_slider0)

        ax_slider1 = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        self.slider1 = Slider(ax_slider1, 'Ax1 idx',
                              valmin=0, valmax=190, valstep=1, valinit=self.idx_1)
        self.slider1.on_changed(self._update_slider1)

        ax_slider2 = fig.add_axes([0.15, 0.03, 0.7, 0.03])
        self.slider2 = Slider(ax_slider2, 'Ax2 idx',
                              valmin=0, valmax=264, valstep=1, valinit=self.idx_2)
        self.slider2.on_changed(self._update_slider2)

        ax_heatmap_slider = fig.add_axes([0.15, 0.01, 0.7, 0.03])
        self.uncertainty_slider = Slider(
            ax_heatmap_slider, 'Uncertainty val', valmin=0, valmax=self.uncertainty_map.max(), valinit=0.05)
        self.uncertainty_slider.on_changed(self._update_uncertainty_slider)
        # self.save_btn = Button(ax_save, 'Save')

    def update_columns(self, dim, indice, image_2d, mask_2d, pred_2d, uncertainty_2d, intersection_2d, union_2d):
        # update ct
        data = self.image_data[dim]
        for i in range(3):
            data[f'ct_{i}'].set_data(image_2d[..., 0])
        data['pet'].set_data(
            apply_cmap_with_blend(image_2d[..., 1], 'inferno', vmin=0, vmax=1))

        uncertainty_map = np.array(uncertainty_2d)
        uncertainty_map[uncertainty_map <
                        self.uncertainty_threshold] = np.nan
        data['uncertain_map'].set_data(uncertainty_map)

        iou_i = np.array(intersection_2d)
        iou_u = np.array(union_2d)
        iou_i[iou_i == 0] = np.nan
        iou_u[iou_u == 0] = np.nan
        # iou_u[iou_i > 0] = np.nan

        data['iou_i'].set_data(iou_i)
        data['iou_u'].set_data(iou_u)

        data['vline'].set_ydata([indice[0]+0.5, indice[0]+0.5])
        data['hline'].set_xdata([indice[1]+0.5, indice[1]+0.5])

        for coll in data['img_mask'].collections:
            data['ax_img'].collections.remove(coll)

        for coll in data['img_pred'].collections:
            data['ax_img'].collections.remove(coll)

        for coll in data['uncertain_mask'].collections:
            data['ax_uncertain'].collections.remove(coll)

        for coll in data['uncertain_pred'].collections:
            data['ax_uncertain'].collections.remove(coll)

        for coll in data['iou_mask'].collections:
            data['ax_iou'].collections.remove(coll)

        for coll in data['iou_pred'].collections:
            data['ax_iou'].collections.remove(coll)

        data['img_mask'] = data['ax_img'].contour(
            mask_2d[..., 0], 1, levels=[0.5], colors=self.mask_color, origin='lower', linewidths=2)
        data['img_pred'] = data['ax_img'].contour(
            pred_2d[..., 0], 1, levels=[0.5], colors=self.pred_color, origin='lower', linewidths=2)

        data['uncertain_mask'] = data['ax_uncertain'].contour(
            mask_2d[..., 0], 1, levels=[0.5], colors=self.mask_color, origin='lower', linewidths=2)
        data['uncertain_pred'] = data['ax_uncertain'].contour(
            pred_2d[..., 0], 1, levels=[0.5], colors=self.pred_color, origin='lower', linewidths=2)

        data['iou_mask'] = data['ax_iou'].contour(
            mask_2d[..., 0], 1, levels=[0.5], colors=self.mask_color, origin='lower', linewidths=2)
        data['iou_pred'] = data['ax_iou'].contour(
            pred_2d[..., 0], 1, levels=[0.5], colors=self.pred_color, origin='lower', linewidths=2)

    def _update_image(self):
        self.update_columns(
            0, [self.idx_1, self.idx_2],
            self.image[self.idx_0],
            self.mask[self.idx_0],
            self.predicted[self.idx_0],
            self.uncertainty_map[self.idx_0],
            self.intersect[self.idx_0],
            self.union[self.idx_0],
        )

        # 2nd column
        self.update_columns(
            1, [self.idx_0, self.idx_2],
            self.image[:, self.idx_1],
            self.mask[:, self.idx_1],
            self.predicted[:, self.idx_1],
            self.uncertainty_map[:, self.idx_1],
            self.intersect[:, self.idx_1],
            self.union[:, self.idx_1],
        )

        # 3rd column
        self.update_columns(
            2, [self.idx_0, self.idx_1],
            self.image[:, :, self.idx_2],
            self.mask[:, :, self.idx_2],
            self.predicted[:, :, self.idx_2],
            self.uncertainty_map[:, :, self.idx_2],
            self.intersect[:, :, self.idx_2],
            self.union[:, :, self.idx_2],
        )
        plt.draw()

    def _update_slider0(self, val):
        self.idx_0 = val
        self._update_image()

    def _update_slider1(self, val):
        self.idx_1 = val
        self._update_image()

    def _update_slider2(self, val):
        self.idx_2 = val
        self._update_image()

    def _update_uncertainty_slider(self, val):
        self.uncertainty_threshold = val
        self._update_image()

    def start(self):
        self._setup()


num_samples = 15
patient_id = 16
uncertainty_map = np.zeros((173, 191, 265, 1))

with open(path + f'/OUS_uncertainty_map/{num_samples:02d}/{patient_id}.npy', 'rb') as f:
    uncertainty_map = np.load(f)

with h5py.File(ous_path + '/test/prediction_test.h5', 'r') as f:
    y_true = f['y'][str(patient_id)][:]
    y_pred = f['predicted'][str(patient_id)][:]
    image = f['x'][str(patient_id)][:]

# find the the intersection and union of predictions from different "models"
# (optional), did not used in the paper
with open(path + f'/OUS/{patient_id}/01.npy', 'rb') as f:
    intersection = np.load(f)
    # union = np.load(f)
intersection = (intersection > 0.5).astype(float)
union = np.array(intersection)

for i in range(2, num_samples+1):
    with open(path + f'/OUS/{patient_id}/{i:02d}.npy', 'rb') as f:
        new_pred = (np.load(f) > 0.5).astype(float)
    intersection = intersection * new_pred
    union = union + new_pred


vis = VisualizeUncertaintyMapV2(
    image, y_true, y_pred, uncertainty_map, intersection, union)
vis.start()
plt.show()
