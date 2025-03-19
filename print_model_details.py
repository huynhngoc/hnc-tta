from deoxys.model import load_model
import numpy as np
import tensorflow as tf
import argparse
import os
import h5py
import pandas as pd
from deoxys.data.preprocessor import preprocessor_from_config
import json


parser = argparse.ArgumentParser()
parser.add_argument("source")
args, unknown = parser.parse_known_args()
source = args.source

model_file = source + '/segmentation/model.h5'

model = load_model(model_file)

print(model.model.summary())