# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import logging
from typing import Any, Dict, Optional

import numpy as np

from tracker.frame import Frame
from tracker.bbox import BBox
from utils.nusc_utils import filter_bboxes_with_nms


def get_adaptive_threshold(base_threshold, classifier_result, cfg):
    """
    Compute adaptive detection threshold based on classifier stationary probability.
    Uses smooth linear interpolation instead of binary thresholds to reduce track fragmentation.
    
    Args:
        base_threshold: Base detection score threshold (e.g., 0.2 for bicycles)
        classifier_result: Dict with 'stationary_probability' key
        cfg: Configuration dict with CLASSIFIER settings
    
    Returns:
        Adjusted threshold value
        
    Example:
        prob=1.0 (stationary) → threshold × 0.85 = 0.17
        prob=0.5 (uncertain)  → threshold × 1.0  = 0.20
        prob=0.0 (moving)     → threshold × 1.15 = 0.23
    """
    if not cfg.get("CLASSIFIER", {}).get("ADAPTIVE_THRESHOLD_ENABLED", False):
        return base_threshold
    
    if classifier_result is None:
        return base_threshold
    
    prob = classifier_result.get("stationary_probability")
    if prob is None:
        return base_threshold
    
    # Get scale factors from config
    min_scale = cfg["CLASSIFIER"].get("STATIONARY_THRESHOLD_SCALE", 0.85)
    max_scale = cfg["CLASSIFIER"].get("MOVING_THRESHOLD_SCALE", 1.15)
    
    # Smooth linear interpolation: prob=1.0 → min_scale, prob=0.0 → max_scale
    # This creates a continuous curve instead of binary jumps
    scale = max_scale - (prob * (max_scale - min_scale))
    
    return base_threshold * scale


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
        
        # Run classifier inference BEFORE filtering to enable adaptive thresholds
        if self.classifier_inference:
            for bbox in bboxes:
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
        
        # Apply detection score filtering with adaptive thresholds for classifier categories
        if self.cfg["TRACKING_MODE"] == "ONLINE":
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"]
        else:
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"]
        
        filtered_bboxes = []
        for bbox in bboxes:
            category_num = self.cfg["CATEGORY_MAP_TO_NUMBER"][bbox["category"]]
            base_threshold = input_score[category_num]
            
            # Use adaptive threshold if classifier result is available
            if bbox.get("classifier_result") is not None:
                threshold = get_adaptive_threshold(
                    base_threshold,
                    bbox["classifier_result"],
                    self.cfg
                )
            else:
                threshold = base_threshold
            
            if bbox["detection_score"] > threshold:
                filtered_bboxes.append(bbox)

        if self.cfg["DATASET"] == "nuscenes":
            if len(filtered_bboxes) != 0:
                filtered_bboxes = filter_bboxes_with_nms(filtered_bboxes, self.cfg)

        for bbox in filtered_bboxes:
            cur_frame.bboxes.append(BBox(frame_id, bbox))
        return cur_frame
