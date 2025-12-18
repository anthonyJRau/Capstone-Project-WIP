# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import logging
from typing import Any, Dict, Optional

import numpy as np

from tracker.frame import Frame
from tracker.bbox import BBox
from utils.nusc_utils import filter_bboxes_with_nms


class BaseVersionTrackingDataset:
    def __init__(
        self,
        scene_id,
        scene_data,
        cfg,
        classifier_inference=None,
        classifier_options: Optional[Dict[str, Any]] = None,
    ):
        self.scene_id = scene_id
        self.scene_data = scene_data
        self.cfg = cfg
        self.classifier_inference = classifier_inference
        classifier_options = classifier_options or {}
        categories = classifier_options.get("categories")
        if categories is None:
            categories = {"bicycle"}
        self.classifier_categories = set(categories)
        self.classifier_kwargs = {
            "annotate": classifier_options.get("annotate", False),
            "expand_pixels": classifier_options.get("roi_expand_pixels", 0),
        }
        self.classifier_log_errors = classifier_options.get("log_errors", True)
        category_filter_cfg = cfg.get("CATEGORY_FILTER")
        if category_filter_cfg:
            self.category_filter = {str(cat) for cat in category_filter_cfg}
        else:
            self.category_filter = set()

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, index):
        frame_info = self.scene_data[index]
        frame_id = frame_info["frame_id"]
        timestamp = frame_info["timestamp"]
        cur_sample_token = frame_info["cur_sample_token"]
        transform_matrix = frame_info["transform_matrix"]
        bboxes = frame_info["bboxes"]

        cur_frame = Frame(
            frame_id=int(frame_id),
            cur_sample_token=cur_sample_token,
            timestamp=timestamp,
            transform_matrix=transform_matrix,
        )

        allowed_categories = self.cfg["CATEGORY_MAP_TO_NUMBER"]
        category_filter = self.category_filter
        bboxes = np.array(
            [
                bbox
                for bbox in bboxes
                if bbox["category"] in allowed_categories
                and (not category_filter or bbox["category"] in category_filter)
            ]
        )
        if self.cfg["TRACKING_MODE"] == "ONLINE":
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"]

        else:
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"]
        filtered_bboxes = [
            bbox
            for bbox in bboxes
            if bbox["detection_score"]
            > input_score[self.cfg["CATEGORY_MAP_TO_NUMBER"][bbox["category"]]]
        ]

        if self.cfg["DATASET"] == "nuscenes":
            if len(filtered_bboxes) != 0:
                filtered_bboxes = filter_bboxes_with_nms(filtered_bboxes, self.cfg)

        if self.classifier_inference:
            for bbox in filtered_bboxes:
                if bbox.get("category") not in self.classifier_categories:
                    continue
                bbox_image = bbox.get("bbox_image") or {}
                if not bbox_image.get("camera_type") or not bbox_image.get("x1y1x2y2"):
                    continue
                try:
                    classifier_result = self.classifier_inference.predict_bbox(
                        bbox,
                        cur_sample_token,
                        **self.classifier_kwargs,
                    )
                    bbox["classifier_result"] = classifier_result
                except Exception as exc:
                    if self.classifier_log_errors:
                        logging.warning(
                            "Classifier inference failed for scene %s frame %s: %s",
                            self.scene_id,
                            frame_id,
                            exc,
                        )

        for bbox in filtered_bboxes:
            cur_frame.bboxes.append(BBox(frame_id, bbox))
        return cur_frame
