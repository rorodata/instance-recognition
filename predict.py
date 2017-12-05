import os
import sys
import random
import math
import numpy as np
import scipy.misc

#import coco

from PIL import Image, ExifTags

import keras
from keras.models import load_model
import tensorflow as tf
import io


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import modellib
import visualize

MODEL_URL = 'https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5'
MODEL_PATH = '/volumes/data/mask_rcnn_coco.h5'


MODEL_DIR = '/volumes/data/'

# Path to trained weights file
COCO_MODEL_PATH ='/volumes/data/mask_rcnn_coco.h5'

from config import Config

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()




def download_model_wts():
    """Downloads the model file.
    """
    if os.path.exists(MODEL_PATH):
        print("Model file is already downloaded.")
        return
    # Download to a tmp file and move it to final file to avoid inconsistent state
    # if download fails or cancelled.
    print("Model file is not available. downloading...")
    exit_status = os.system("wget {} -O {}.tmp".format(MODEL_URL, MODEL_PATH))
    if exit_status == 0:
        os.system("mv {}.tmp {}".format(MODEL_PATH, MODEL_PATH))
    else:
        print("Failed to download the model file", file=sys.stderr)
        sys.exit(1)

# Preload our model
download_model_wts()
print("Loading model weights")

# Create model object in inference mode.
from modellib import MaskRCNN

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

import matplotlib.patches as patches
from matplotlib.patches import Polygon

def paint_detections(image, boxes, masks, class_names, class_ids, scores):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Number of instances
    N = boxes.shape[0]

    # Generate random colors
    colors = visualize.random_colors(N)
    
    #copy the image
    masked_image = image.astype(np.uint32).copy()
    
    #paint masks on it
    for i in range(N):
        color = colors[i]
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color).astype('uint8')

        # paint BB rectangles
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (x1,y1), (x2, y2), (0,255,0),2)
        
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        cv2.putText(masked_image, caption,(x1+3, y1+8), font, 0.3,(255,255,255))
    
    return masked_image



def predict(image_file, format='jpg'):
    global model, class_names
    
    image=plt.imread(image_file, format=format)

    results = model.detect([img], verbose=1)
    r = results[0]

    boxes, masks, class_ids, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    
    img2=paint_detections(image, boxes, masks, class_names, class_ids, scores)
    f = io.BytesIO()
    plt.imsave(f, img2)
    f.seek(0)
    return f







