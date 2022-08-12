from detectron2.config import get_cfg
from detectron2.projects.point_rend.config import add_pointrend_config
from detectron2.structures import Instances, Boxes

from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling.meta_arch import GeneralizedRCNN

import os
import os.path as osp
import torch


def setup_cfg(cfg_path, model_ckpt_path):
    cfg = get_cfg()
    add_pointrend_config(cfg)

    if not osp.isabs(cfg_path):
        cfg_path = osp.realpath(osp.join(osp.dirname(__file__), os.pardir, "pointrend_configs", cfg_path))

    assert osp.exists(cfg_path), f"Config file not found at expected path: {cfg_path}"
    cfg.merge_from_file(cfg_path)

    assert osp.exists(model_ckpt_path), f"Model checkpoint file does not exist: {model_ckpt_path}"
    cfg.MODEL.WEIGHTS = model_ckpt_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    cfg.freeze()

    return cfg


class PointRendBox2MaskInference:
    def __init__(self, model_checkpoint_path: str, config: str):
        cfg = setup_cfg(config, model_checkpoint_path)
        self.predictor = DefaultPredictor(cfg)
        self.predictor.model = self.predictor.model.cuda()

    @property
    def model(self):
        return self.predictor.model

    def forward(self, image, proposals):
        batched_inputs = [{
            "image": image.cuda().permute(2, 0, 1),
            "proposals": proposals
        }]

        images = self.model.preprocess_image(batched_inputs)
        features = self.model.backbone(images.tensor)

        features_for_box_head = [features[f] for f in self.model.roi_heads.box_in_features]
        box_features = self.model.roi_heads.box_pooler(features_for_box_head, [x.proposal_boxes for x in [proposals]])
        box_features = self.model.roi_heads.box_head(box_features)

        predictions = self.model.roi_heads.box_predictor(box_features)
        pred_instances = self.proposals_to_pred_instances(proposals, predictions)

        pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
        return GeneralizedRCNN._postprocess(pred_instances, batched_inputs, images.image_sizes)

    def __call__(self, image: torch.Tensor, box_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param image: An image as a tensor of shape [H, W, 3] with pixel values in range [0, 255] and BGR channel format
        :param box_coords: tensor of box coordinates of shape [N, 4] in (x1, y1, x2, y2) format
        :return: tensor of shape [N, H, W] with a boolean mask for each input box
        """
        image = image.cuda()
        proposals = Instances(
            image.shape[:2],
            proposal_boxes=Boxes(box_coords),
            objectness_logits=torch.ones(box_coords.size(0), dtype=torch.float32)
        ).to("cuda:0")

        output = self.forward(image, proposals)

        masks = output[0]['instances'].pred_masks  # [N, H, W]
        assert masks.size(0) == box_coords.size(0), f"Shape mismatch: {masks.shape}, {box_coords.shape}"

        return masks

    @staticmethod
    def proposals_to_pred_instances(proposals, predictions):
        logits = predictions[0]
        # logits[:, -1] = float("-inf")  # lower background logit
        max_cls_id = logits[:, :-1].argmax(1)  # [N]

        pred_instances = Instances(proposals.image_size, pred_boxes=proposals.proposal_boxes, pred_classes=max_cls_id)
        return [pred_instances]
