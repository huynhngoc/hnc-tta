import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("pid")
parser.add_argument("num_tta", type=int)

args, unknown = parser.parse_known_args()

# Define the base path and pid
base_path = '../analysis/' + args.name
pid = args.pid
num_tta = args.num_tta  

# Load the uncertainty map
uncertainty_map_path = f'{base_path}/MAASTRO_uncertainty_map/{num_tta:02d}/{pid}.npy'
uncertainty_map = np.load(uncertainty_map_path)

# Select a slice (e.g., the middle slice along the z-axis)
slice_index = uncertainty_map.shape[2] // 2
slice_data = uncertainty_map[:, :, slice_index]

# Visualize the slice
plt.imshow(slice_data, cmap='hot')
plt.colorbar(label='Uncertainty')
plt.title(f'Uncertainty Map Slice for PID: {pid}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Save the figure as a PDF file
output_path = f'{base_path}/MAASTRO_uncertainty_map/{num_tta:02d}/{pid}_slice.pdf'