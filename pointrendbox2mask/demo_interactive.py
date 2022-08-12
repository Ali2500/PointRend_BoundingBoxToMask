from argparse import ArgumentParser

from pointrendbox2mask.core import PointRendBox2MaskInference
from pointrendbox2mask.utils import create_color_map, overlay_mask_on_image

import cv2
import numpy as np
import os
import os.path as osp
import torch

COLOR_MAP = create_color_map().tolist()
INFERENCE_MODEL = None
ORIGINAL_IMAGE = None
TORCH_IMAGE = None

LEFT_BUTTON_DOWN = None
CURRENT_IMAGE = None
BOX_POINT_1 = None
BOX_POINT_2 = None
COLOR_INDEX = 1


def mouse_event_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global BOX_POINT_1, LEFT_BUTTON_DOWN
        BOX_POINT_1 = [x, y]
        LEFT_BUTTON_DOWN = True

    elif event == cv2.EVENT_LBUTTONUP:
        global BOX_POINT_2, INFERENCE_MODEL, COLOR_INDEX, CURRENT_IMAGE
        BOX_POINT_2 = [x, y]
        if x <= BOX_POINT_1[0] or y <= BOX_POINT_1[1]:
            print("Second point must be to the lower-right of the first point.")
            return

        # do forward pass
        print(f"Running inference on box coords: {BOX_POINT_1 + BOX_POINT_2}")
        boxes = torch.as_tensor(BOX_POINT_1 + BOX_POINT_2, dtype=torch.float32)[None]  # [1, 4]

        # (4) Forward pass. Model returns a mask for each input box
        masks = INFERENCE_MODEL(image=TORCH_IMAGE, box_coords=boxes)  # [N, H, W]

        # visualization code
        box = boxes.tolist()[0]
        mask = masks.byte().cpu().numpy()[0]

        CURRENT_IMAGE = overlay_mask_on_image(CURRENT_IMAGE, mask, box, mask_color=COLOR_MAP[COLOR_INDEX])
        COLOR_INDEX += 1

        cv2.imshow("Image", CURRENT_IMAGE)


def main(args):
    global INFERENCE_MODEL, ORIGINAL_IMAGE, CURRENT_IMAGE, TORCH_IMAGE, LEFT_BUTTON_DOWN

    INFERENCE_MODEL = PointRendBox2MaskInference(
        model_checkpoint_path=args.model_checkpoint,
        config=args.config
    )

    # (2) Load the image
    assert osp.exists(args.image), f"Image file does not exist: {args.image}"
    ORIGINAL_IMAGE = cv2.imread(args.image, cv2.IMREAD_COLOR)
    TORCH_IMAGE = torch.from_numpy(ORIGINAL_IMAGE)  # [C, H, W]
    CURRENT_IMAGE = np.copy(ORIGINAL_IMAGE)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_event_callback)
    cv2.imshow("Image", ORIGINAL_IMAGE)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)

    main(parser.parse_args())
