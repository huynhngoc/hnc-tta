import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import matplotlib.cm as cm
from deoxys.data.preprocessor import preprocessor_from_config
import json




def apply_cmap_with_blend(functional_data, cmap, vmin=None, vmax=None, gamma=1):
    functional_data = functional_data.astype(float)
    if vmin is None:
                    vmin = functional_data.min()
    if vmax is None:
                    vmax = functional_data.max()
    functional_data = (functional_data - vmin) / (vmax - vmin)
    functional_data = np.minimum(np.maximum(functional_data, 0), 1) ** gamma
    image = cm.get_cmap(cmap)(functional_data)
    image[..., -1] = functional_data
    return image

def augment_image(image, preprocessors):
    for preprocessor in preprocessors:
        image = preprocessor.transform(image, None)
    return image


parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("config")
#parser.add_argument("num_tta", type=int)
parser.add_argument("pid")
parser.add_argument("source")

args, unknown = parser.parse_known_args()

print("Defining arguments...")
# Define the base path and pid
source = args.source
base_path = source + '/analysis/' + args.name
#num_tta = args.num_tta  
pid = args.pid
with open(source + '/hnc-tta/' + args.config, 'r') as file:
        config = json.load(file)

print("Loading preprocessors...")
preprocessors = []
for pp_config in config:
    preprocessors.append(preprocessor_from_config(pp_config))

#output_type = "image"
center = "OUS"
aug_type = "brightness"

print("Creating directories...")
if not os.path.exists(base_path + f'/{center}_augmentation_visualization'):
    os.makedirs(base_path + f'/{center}_augmentation_visualization')
"""if not os.path.exists(base_path + f'/{center}_augmentation_visualization/{output_type}'):
    os.makedirs(base_path + f'/{center}_augmentation_visualization/{output_type}')"""

"""
if not os.path.exists(base_path + '/MAASTRO_input_visualization'):
    os.makedirs(base_path + '/MAASTRO_input_visualization')
if not os.path.exists(base_path + f'/MAASTRO_input_visualization/{output_type}'):
    os.makedirs(base_path + f'/MAASTRO_input_visualization/{output_type}'):

"""

if center == "MAASTRO":
    print("running MAASTRO...")
    with h5py.File(source + '/segmentation/maastro_full.h5', 'r') as f:
        y_true = f['y'][str(pid)][:]
        y_pred = f['predicted'][str(pid)][:]
        image = f['x'][str(pid)][:]

    for preprocessor in preprocessors:
        image = preprocessor.transform(np.array([image]), None)[0]
        image2d = image[:, :, 87]

        # Visualize the slice
        plt.imshow(image2d[..., 0], 'gray', vmin=0, vmax=1, origin='lower')
        #plt.imshow(apply_cmap_with_blend(image2d[..., 1], 'inferno', vmin=0, vmax=1), origin='lower')
        plt.title(f'PID: {pid}, Augmentation: {aug_type}: 2.0')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Save the figure as a PDF file
        if not os.path.exists(base_path + f'/MAASTRO_augmentation_visualization/{aug_type}'):
            os.makedirs(base_path + f'/MAASTRO_augmentation_visualization/{aug_type}')


        output_path = f'{base_path}/MAASTRO_augmentation_visualization/{aug_type}/pid_{pid}_CT.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close


if center == "OUS":
    print("running OUS...")
    with h5py.File(source + '/segmentation/ous_test.h5', 'r') as f:
        y_true = f['y'][str(pid)][:]
        y_pred = f['predicted'][str(pid)][:]
        image = f['x'][str(pid)][:]

    for preprocessor in preprocessors:
        image = preprocessor.transform(np.array([image]), None)[0]
        image2d = image[:, :, 87]

        # Visualize the slice
        plt.imshow(image2d[..., 0], 'gray', vmin=0, vmax=1, origin='lower')
        plt.imshow(apply_cmap_with_blend(image2d[..., 1],'inferno', vmin=0, vmax=1), origin='lower')
        plt.title(f'PID: {pid}, Augmentation: {aug_type}: 0.7')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Save the figure as a PDF file
        if not os.path.exists(base_path + f'/OUS_augmentation_visualization/{aug_type}'):
            os.makedirs(base_path + f'/OUS_augmentation_visualization/{aug_type}')


        output_path = f'{base_path}/OUS_augmentation_visualization/{aug_type}/pid{pid}_CT_PET_07.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close