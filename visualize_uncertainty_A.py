import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("num_tta", type=int)
parser.add_argument("center")
parser.add_argument("pid")
parser.add_argument("source")

args, unknown = parser.parse_known_args()

# Define the base path and pid
source = args.source
base_path = source + '/analysis/' + args.name
num_tta = args.num_tta  
center = args.center
pid = args.pid



if not os.path.exists(base_path + '/OUS_image_visualization'):
    os.makedirs(base_path + '/OUS_image_visualization')
if not os.path.exists(base_path + f'/OUS_image_visualization/{num_tta:02d}'):
    os.makedirs(base_path + f'/OUS_image_visualization/{num_tta:02d}')


if not os.path.exists(base_path + '/MAASTRO_image_visualization'):
    os.makedirs(base_path + '/MAASTRO_image_visualization')
if not os.path.exists(base_path + f'/MAASTRO_image_visualization/{num_tta:02d}'):
    os.makedirs(base_path + f'/MAASTRO_image_visualization/{num_tta:02d}')


"""print(f'Visualizing uncertainty map for PID: {pid} with {num_tta} TTA')

# Load the uncertainty map
uncertainty_map_path = f'{base_path}/MAASTRO_uncertainty_map/{num_tta:02d}/{pid}.npy'
uncertainty_map = np.load(uncertainty_map_path)

# Select a slice (e.g., the middle slice along the z-axis)
slice_index = uncertainty_map.shape[2] // 2
slice_data = uncertainty_map[:, :, slice_index]

# Visualize the slice
plt.imshow(slice_data, cmap='Reds')
plt.colorbar(label='Entropy')
plt.title(f'Uncertainty Map Slice for PID: {pid}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Save the figure as a PDF file
output_path = f'{base_path}/MAASTRO_uncertainty_map_visualization/{num_tta:02d}/pid_{pid}_slice.pdf'
plt.savefig(output_path, format='pdf')"""

#print("Preperations for interactive visualization")

if center == "MAASTRO":
    print(f"Preparing data for MAASTRO pid: {pid}")
    with h5py.File(source + '/segmentation/maastro_full.h5', 'r') as f:
        y_true = f['y'][str(pid)][:]
        y_pred = f['predicted'][str(pid)][:]
        image = f['x'][str(pid)][:]
        
    with open(base_path + f'/MAASTRO_image_visualization//{num_tta:02d}/pid_{pid}.npy', 'wb') as f:
                np.save(f, y_true)
                np.save(f, y_pred)
                np.save(f, image)



if center == "OUS":
    print(f"Preparing data for OUS pid: {pid}")

    with h5py.File(source + '/segmentation/ous_test.h5', 'r') as f:
        y_true = f['y'][str(pid)][:]
        y_pred = f['predicted'][str(pid)][:]
        image = f['x'][str(pid)][:]
        
    with open(base_path + f'/OUS_image_visualization//{num_tta:02d}/pid_{pid}.npy', 'wb') as f:
                np.save(f, y_true)
                np.save(f, y_pred)
                np.save(f, image)