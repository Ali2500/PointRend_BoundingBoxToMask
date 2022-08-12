from argparse import ArgumentParser

from pointrendbox2mask.core import PointRendBox2MaskInference
from pointrendbox2mask.utils import create_color_map, overlay_mask_on_image

import cv2
import os
import os.path as osp
import torch

# can be changed to any image in the 'example_images' folder
DEFAULT_IMAGE = "bear.jpg"

# bounding box coordinates for each of the images
BOUNDING_BOX_COORDS = {
    "bear.jpg": [[86, 215, 473, 437]],
    "bmx-riding.jpg": [[503, 244, 743, 385], [538, 149, 703, 339]]
}


def main(args):
    # (1) Initialize model with checkpoint and config paths
    inference_model = PointRendBox2MaskInference(
        model_checkpoint_path=args.model_checkpoint,
        config=args.config
    )

    # (2) Load the image
    image_path = osp.join(osp.dirname(__file__), os.pardir, "example_images", DEFAULT_IMAGE)
    assert osp.exists(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # (3) Convert image and bounding box coordinates to torch tensors
    image = torch.from_numpy(image)  # [C, H, W]
    boxes = torch.as_tensor(BOUNDING_BOX_COORDS[DEFAULT_IMAGE], dtype=torch.float32)  # [N, 4]

    # (4) Forward pass. Model returns a mask for each input box
    masks = inference_model(image=image, box_coords=boxes)  # [N, H, W]

    # visualization code
    color_map = create_color_map().tolist()

    image = image.numpy()
    boxes = boxes.tolist()
    masks = masks.byte().cpu().numpy()

    for i, (box, mask) in enumerate(zip(boxes, masks), 1):
        image = overlay_mask_on_image(image, mask, box, mask_color=color_map[i])

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--config", required=True)

    main(parser.parse_args())
