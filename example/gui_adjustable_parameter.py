import cv2
import numpy as np
from pyhjs import PyHJS, BinaryFrame
from pathlib import Path
import cvui


SCRIPT_DIR = str(Path().parent)


def fill_boundary_zero(image, bold=1):
    assert len(image.shape) == 2
    height, width = image.shape
    image[0, :] = 0
    image[:, 0] = 0
    image[height - 1, :] = 0
    image[:, width - 1] = 0
    return image


class Skeletonizer(object):
    def __init__(self, gamma, epsilon, arc_angle_threshold=0):
        self._hjs = PyHJS(gamma, epsilon, arc_angle_threshold)

    def compute(self, input_mask):
        frame = BinaryFrame(input_mask)
        self._hjs.compute(frame, enable_anisotropic_diffusion=True)
        skeleton = self._hjs.get_skeleton_image()
        return skeleton

    def set_parameters(self, gamma, epsilon, arc_angle_threshold=0):
        self._hjs.set_parameters(gamma, epsilon, arc_angle_threshold)

    def get_flux_image(self):
        return self._hjs.get_flux_image()

    def get_distance_transform_image(self):
        return self._hjs.get_distance_transform_image()


def main():
    ### load mask image and resize
    image = cv2.imread(f"{SCRIPT_DIR}/example/mask.png", cv2.IMREAD_ANYDEPTH)
    image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    input_mask = np.zeros_like(image)
    input_mask[image > 0] = 255

    ### Initialize skeletonizer
    gamma = [2.5]
    epsilon = [1.5]
    angle_thresh = [0]
    skeletonizer = Skeletonizer(gamma=gamma[0], epsilon=epsilon[0], arc_angle_threshold=angle_thresh[0])

    ### Initialize visualizer
    scale = 2.0
    image_viz_width = int(input_mask.shape[1] * scale)
    image_viz_height = int(input_mask.shape[0] * scale)
    frame_width = np.max([image_viz_width, 200])
    frame_height = image_viz_height + 400
    frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    WINDOW_NAME = "Skeletonization"
    cvui.init(WINDOW_NAME)
    while True:
        # Fill the frame with a nice color
        frame[:] = (49, 52, 49)

        cvui.text(frame, 10, image_viz_height + 10, "gamma, (default: 2.5)")
        cvui.trackbar(frame, 10, image_viz_height + 40, 180, gamma, 0.1, 5.0)

        cvui.text(frame, 10, image_viz_height + 110, "epsilon, (default: 1.0)")
        cvui.trackbar(frame, 10, image_viz_height + 140, 180, epsilon, 0.1, 5.0)

        cvui.text(frame, 10, image_viz_height + 110, "angle_threshold, (default: 0.0)")
        cvui.trackbar(frame, 10, image_viz_height + 240, 180, angle_thresh, 0.0, 180.0)

        skeletonizer.set_parameters(gamma[0], epsilon[0], angle_thresh[0])
        skeleton_img = skeletonizer.compute(input_mask)
        skeleton_img = fill_boundary_zero(skeleton_img)
        skeleton_viz_img = (skeleton_img * 255).astype(np.uint8)
        skeleton_viz_img = cv2.cvtColor(cv2.resize(skeleton_viz_img, None, fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
        frame[:image_viz_height, :image_viz_width, :] = skeleton_viz_img

        cvui.update()
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(10) in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
