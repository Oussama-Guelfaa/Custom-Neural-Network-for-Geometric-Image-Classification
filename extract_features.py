# extract_features.py

import os
import glob
import numpy as np
import h5py
from skimage.io import imread
from skimage import util
from skimage.measure import label, regionprops

def check_if_directory_exists(name_folder):
    """Check if a directory exists, and create it if not."""
    if not os.path.exists(name_folder):
        print(f"{name_folder} directory does not exist, created.")
        os.makedirs(name_folder)
    else:
        print(f"{name_folder} directory exists, no action performed.")

def get_shape_features(image_region):
    """Extract shape features from a binary image region."""
    shape_features = np.empty(shape=10)
    labeled_image = label(image_region)
    props = regionprops(labeled_image)[0]

    # Convex Area
    shape_features[0] = props.convex_area
    # Eccentricity
    shape_features[1] = props.eccentricity
    # Perimeter
    shape_features[2] = props.perimeter
    # Equivalent Diameter
    shape_features[3] = props.equivalent_diameter
    # Extent
    shape_features[4] = props.extent
    # Filled Area
    shape_features[5] = props.filled_area
    # Minor Axis Length
    shape_features[6] = props.minor_axis_length
    # Major Axis Length
    shape_features[7] = props.major_axis_length
    # Ratio of Major to Minor Axis
    shape_features[8] = props.major_axis_length / props.minor_axis_length
    # Solidity
    shape_features[9] = props.solidity

    return shape_features

# Main Program
dir_base = "./Images"
dir_edges = "Edges"
dir_images_edges = os.path.join(dir_base, dir_edges)
dir_output = "Output"
features_path = os.path.join(dir_output, "features_geometric.h5")
labels_path = os.path.join(dir_output, "labels_high-low.h5")

image_labels = os.listdir(dir_images_edges)

# Variables to hold features and labels
X = np.empty((0, 10))
Y = np.array([])

for lab in image_labels:
    cur_path = os.path.join(dir_images_edges, lab)
    for image_path in glob.glob(os.path.join(cur_path, "*.png")):
        print(f"[INFO] Processing image {image_path}")
        img_ori = imread(image_path)
        img = util.img_as_ubyte(img_ori)
        features = get_shape_features(img)
        print(f"[INFO] ...Storing descriptors of image {image_path}")

        # Binary classification: High wear level is class 1, Low/Medium is class 0
        lab_num = 1 if lab == "2_High" else 0

        X = np.append(X, np.array([features]), axis=0)
        Y = np.append(Y, lab_num)

print("\n[INFO] Saving descriptors in folder " + dir_output)
check_if_directory_exists(dir_output)

# Save features and labels
with h5py.File(features_path, 'w') as h5f_data:
    h5f_data.create_dataset("dataset_inserts_geometric", data=X)

with h5py.File(labels_path, 'w') as h5f_label:
    h5f_label.create_dataset("dataset_inserts_geometric", data=Y)
