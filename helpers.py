# Helper functions

import os
import glob # Library for loading images from a directory
import matplotlib.image as mpimg


def load_dataset(image_dir):
    img_list = []
    image_types = ["red", "yellow", "green"]

    for type in image_types:
        # Iterate through each image file in each image_types folder
        # glob reads in any image with the extension "image_dir/type/*"
        for file in glob.glob(os.path.join(image_dir, type, "*")):
            img = mpimg.imread(file)
            if not img is None:
                img_list.append((img, type))

    return img_list
