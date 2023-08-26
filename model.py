import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.python.util import compat
from tensorflow.core.protobuf import saved_model_pb2
from google.protobuf import text_format
import pprint
import json
import os

print(tf.__version__)
# needed to install object_detection library and enlarge labels
rm -rf ./models && git clone https://github.com/tensorflow/models.git
&& sed -i "s#ImageFont.truetype('arial.ttf', 24)#ImageFont.truetype('arial.ttf', 50)#g" ./models/research/object_detection/utils/visualization_utils.py
&& cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /usr/share/fonts/truetype/dejavu/arial.ttf

# install object_detection library
cd models/research
&& protoc object_detection/protos/*.proto --python_out=. \
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation
# From within TensorFlow/models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install .


