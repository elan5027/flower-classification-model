import tensorflow as tf
import pathlib
from tensorflow import keras
import numpy as np
data_dir = pathlib.Path('C:\\gitblog\\deep\\flower\\flowers') 
image_count = len(list(data_dir.glob('*/*.jpg')))

print(data_dir)