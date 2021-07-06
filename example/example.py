import cv2
import numpy as np
from pyhjs import PyHJS, BinaryFrame
from pathlib import Path
import time

from typing import List, Tuple
import matplotlib.pyplot as plt

SCRIPT_DIR = str(Path().parent)


def fill_boundary_zero(image, bold=1):
    assert len(image.shape) == 2
    height, width = image.shape
    image[0, :] = 0
    image[:, 0] = 0
    image[height - 1, :] = 0
    image[:, width - 1] = 0
    return image


def closing(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


class Skeletonizer(object):
    def __init__(self, gamma, epsilon):
        self._hjs = PyHJS(gamma, epsilon)

    def compute(self, label_mask):
        frame = BinaryFrame(label_mask)
        self._hjs.compute(frame)
        skeleton = self._hjs.get_skeleton_image()
        return skeleton

    def set_parameters(self, gamma, epsilon):
        self._hjs.set_parameters(gamma, epsilon)

    def get_flux_image(self):
        return self._hjs.get_flux_image()

    def get_distance_transform_image(self):
        return self._hjs.get_distance_transform_image()


def get_separate_skeleton_mask(skeleton_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: rename img_kel
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    img_kel = cv2.filter2D(skeleton_image, -1, kernel)
    img_kel[skeleton_image == 0] = 0
    img_jc = np.zeros_like(img_kel)
    img_jc[img_kel > 2] = 1
    img_bridge = np.zeros_like(img_kel)
    img_bridge[(img_kel == 1) | (img_kel == 2)] = 1
    img_end_point = np.zeros_like(img_kel)
    img_end_point[img_kel == 1] = 1
    return (img_jc, img_bridge, img_end_point)


def pruning_skeleton_mask(skeleton_image: np.ndarray, contour_mask: np.ndarray, dilate_kernel_size=9, edge_redundant_threshold=50):
    # Get below masks
    # - end points
    # - junction points
    # - bridge area
    _, img_bridge, _ = get_separate_skeleton_mask(skeleton_image)

    # Calculate connnected component algorithm against bridge_image
    _, cc_bridge_labels = cv2.connectedComponentsWithAlgorithm(img_bridge.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U, ccltype=cv2.CCL_SAUF)

    # Removal redundant labels
    contour_mask_dilated = cv2.dilate(contour_mask, np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8))
    redundant_labels = np.unique(cc_bridge_labels[contour_mask_dilated > 0])
    label_mask_filtered = skeleton_image.copy()
    for redundant_label in redundant_labels:
        if redundant_label == 0:
            continue
        if np.sum(cc_bridge_labels == redundant_label) < edge_redundant_threshold:
            label_mask_filtered[cc_bridge_labels == redundant_label] = 0
    return label_mask_filtered


### (2) Redundant skeleton-edge removal
# build graph from skeleton
def get_binary_image_contour(binary_image: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros_like(binary_image)
    for contour in contours:
        contour_mask[contour[:, 0, 1], contour[:, 0, 0]] = 255
    return contour_mask


# load mask image and resize
image = cv2.imread(f"{SCRIPT_DIR}/example/mask.png", cv2.IMREAD_ANYDEPTH)
image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
print(image.shape)

label_mask = np.zeros_like(image)
label_mask[image > 0] = 255

skeletonizer = Skeletonizer(gamma=2.5, epsilon=1.5)
start = time.time()
skeleton = skeletonizer.compute(label_mask)
end = time.time()
print(end - start)

skeleton_img_raw = fill_boundary_zero(skeleton)
skeleton_img = np.zeros_like(skeleton_img_raw)
skeleton_img[skeleton_img_raw > 0] = 1
flux_img = skeletonizer.get_flux_image()
df_img = skeletonizer.get_distance_transform_image()

### Redundant skeleton removal
start = time.time()
contour_mask = get_binary_image_contour(label_mask)
skeleton_img = pruning_skeleton_mask(skeleton_img, contour_mask, edge_redundant_threshold=30)
end = time.time()
print(end - start)

plt.imshow(skeleton_img)
plt.show()

### save figures
"""
plt.imshow(label_mask)
plt.savefig("example/input.png")
plt.imshow(skeleton_img)
plt.savefig("example/result.png")
cv2.imwrite("example/result_binary.png", (skeleton_img * 255).astype(np.uint8))
"""
