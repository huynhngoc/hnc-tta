import h5py
import argparse

"""
This script is used to check the contents of an HDF5 file.
It takes two command line arguments:
1. source: The source directory where the HDF5 file is located.
2. filename: The name of the HDF5 file to check.
The script opens the HDF5 file and prints the names of the groups and datasets within it, along with their shapes.
"""

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("filename")

args, unknown = parser.parse_known_args()
file_path = args.source + '/segmentation/' + args.filename


def print_detail(file_name):
    with h5py.File(file_name, 'r') as f:
        for group in f.keys():
            print(group)
            for ds_name in f[group].keys():
                print('--', ds_name, f[group][ds_name].shape)

print_detail(file_path)