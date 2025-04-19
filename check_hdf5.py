import h5py
import argparse

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