from deoxys.model import load_model
import numpy as np
import tensorflow as tf
import argparse
import os
import h5py
import pandas as pd
from deoxys.data.preprocessor import preprocessor_from_config
import json


def augment_image(image, preprocessors):
    for preprocessor in preprocessors:
        image = preprocessor.transform(image, None)
    return image


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("name")
    parser.add_argument("--iter", default=1, type=int)

    args, unknown = parser.parse_known_args()
    base_path = '../results/' + args.name
    with open(args.config, 'r') as file:
        config = json.load(file)
    iter = args.iter
    preprocessors = []
    for pp_config in config:
        preprocessors.append(preprocessor_from_config(pp_config))

    print('Base_path:', base_path)
    print('Augmentation from:', args.name)
    print('Iteration:', iter)

    if not os.path.exists(base_path):
        os.makedirs(base_path)


    ous_h5 = '../segmentation/ous_test.h5'
    ous_csv = '../segmentation/ous_test.csv'
    maastro_h5 = '../segmentation/maastro_full.h5'
    maastro_csv = '../segmentation/maastro_full.csv'
    model_file = '../segmentation/model.h5'

    model = load_model(model_file)

    if not os.path.exists(base_path + '/OUS'):
        os.makedirs(base_path + '/OUS')

    ous_df = pd.read_csv(ous_csv)

    print('Working on OUS.....')
    for pid in ous_df.patient_idx:
        print('PID:', pid)
        if not os.path.exists(base_path + '/OUS/' + str(pid)):
            os.makedirs(base_path + '/OUS/' + str(pid))
        with h5py.File(ous_h5, 'r') as f:
            images = f['x'][str(pid)][:][None, ...]
        preds = model.predict(augment_image(images, preprocessors))

        with open(base_path + '/OUS/' + str(pid) + f'/{iter:02d}.npy', 'wb') as f:
            np.save(f, preds[0])

    if not os.path.exists(base_path + '/MAASTRO'):
        os.makedirs(base_path + '/MAASTRO')

    maastro_df = pd.read_csv(maastro_csv)
    print('Working on MAASTRO.....')
    for pid in maastro_df.patient_idx:
        print('PID:', pid)
        if not os.path.exists(base_path + '/MAASTRO/' + str(pid)):
            os.makedirs(base_path + '/MAASTRO/' + str(pid))
        with h5py.File(maastro_h5, 'r') as f:
            images = f['x'][str(pid)][:][None, ...]
        preds = model.predict(augment_image(images, preprocessors))

        with open(base_path + '/MAASTRO/' + str(pid) + f'/{iter:02d}.npy', 'wb') as f:
            np.save(f, preds[0])
