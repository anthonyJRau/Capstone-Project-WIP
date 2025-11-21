# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import json, yaml
import logging
import copy
import argparse
import os
import time
import multiprocessing
from tqdm import tqdm
from datetime import datetime
from functools import partial
from tracker.base_tracker import Base3DTracker
from dataset.baseversion_dataset import BaseVersionTrackingDataset
from evaluation.static_evaluation.kitti.evaluation_HOTA.scripts.run_kitti import (
    eval_kitti,
)
from evaluation.static_evaluation.nuscenes.eval import eval_nusc
from evaluation.static_evaluation.waymo.eval import eval_waymo
from utils.kitti_utils import save_results_kitti
from utils.nusc_utils import save_results_nuscenes, save_results_nuscenes_for_motion
from utils.waymo_utils.convert_result import save_results_waymo

logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

_CLASSIFIER_CACHE = {"inference": None, "options": None}


def get_classifier_runtime(cfg):
    """Lazily construct and cache the classifier inference helper."""
    classifier_cfg = cfg.get("CLASSIFIER", {}) or {}
    if not classifier_cfg.get("ENABLED"):
        return None, {}

    if _CLASSIFIER_CACHE["inference"] is not None:
        return _CLASSIFIER_CACHE["inference"], _CLASSIFIER_CACHE["options"]

    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        logging.error("NuScenes package required for classifier inference: %s", exc)
        return None, {}

    try:
        from tracker.classifier_model_inference import MotionClassifierInference
    except ImportError as exc:
        logging.error("Unable to import MotionClassifierInference: %s", exc)
        return None, {}

    checkpoint_path = classifier_cfg.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        logging.error("Classifier enabled but CHECKPOINT_PATH not provided; disabling inference")
        return None, {}

    dataroot = classifier_cfg.get("DATAROOT") or cfg.get("DATASET_ROOT")
    if not dataroot:
        logging.error("Classifier requires DATAROOT or DATASET_ROOT; disabling inference")
        return None, {}
    dataroot = os.path.abspath(dataroot)

    version = classifier_cfg.get("VERSION", "v1.0-trainval")
    verbose = classifier_cfg.get("VERBOSE", False)
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to initialize NuScenes for classifier inference: %s", exc)
        return None, {}

    try:
        inference = MotionClassifierInference(
            checkpoint_path=checkpoint_path,
            nusc=nusc,
            device=classifier_cfg.get("DEVICE"),
            transform=None,
            stationary_class_index=classifier_cfg.get("STATIONARY_CLASS_INDEX", 0),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to initialize classifier inference: %s", exc)
        return None, {}

    options = {
        "categories": classifier_cfg.get("CATEGORIES", ["bicycle"]),
        "annotate": classifier_cfg.get("ANNOTATE", False),
        "roi_expand_pixels": classifier_cfg.get("ROI_EXPAND_PIXELS", 0),
        "log_errors": classifier_cfg.get("LOG_ERRORS", True),
    }

    _CLASSIFIER_CACHE["inference"] = inference
    _CLASSIFIER_CACHE["options"] = options
    return inference, options

def run(scene_id, scenes_data, cfg, args, tracking_results):
    logging.debug(f'run called for scene_id: {scene_id}, cfg: {cfg}')
    """
    Info: This function tracks objects in a given scene, processes frame data, and stores tracking results.
    Parameters:
        input:
            scene_id: ID of the scene to process.
            scenes_data: Dictionary with scene data.
            cfg: Configuration settings for tracking.
            args: Additional arguments.
            tracking_results: Dictionary to store results.
        output:
            tracking_results: Updated tracking results for the scene.
    """
    scene_data = scenes_data[scene_id]
    classifier_inference = None
    classifier_options = {}
    if cfg.get("DATASET") == "nuscenes":
        classifier_inference, classifier_options = get_classifier_runtime(cfg)
    category_filter = set(cfg.get("CATEGORY_FILTER", [])) if cfg.get("CATEGORY_FILTER") else set()

    dataset = BaseVersionTrackingDataset(
        scene_id,
        scene_data,
        cfg=cfg,
        classifier_inference=classifier_inference,
        classifier_options=classifier_options,
    )
    tracker = Base3DTracker(cfg=cfg)
    all_trajs = {}

    for index in tqdm(range(len(dataset)), desc=f"Processing {scene_id}"):
        frame_info = dataset[index]
        frame_id = frame_info.frame_id
        cur_sample_token = frame_info.cur_sample_token
        all_traj = tracker.track_single_frame(frame_info)
        if category_filter:
            all_traj = {
                track_id: bbox
                for track_id, bbox in all_traj.items()
                if bbox.category in category_filter
            }
        result_info = {
            "frame_id": frame_id,
            "cur_sample_token": cur_sample_token,
            "trajs": copy.deepcopy(all_traj),
            "transform_matrix": frame_info.transform_matrix,
        }
        all_trajs[frame_id] = copy.deepcopy(result_info)
    if cfg["TRACKING_MODE"] == "GLOBAL":
        trajs = tracker.post_processing()
        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                for bbox in trajs[track_id].bboxes:
                    if (
                        bbox.frame_id == frame_id
                        and bbox.is_interpolation
                        and track_id not in all_trajs[frame_id]["trajs"].keys()
                        and (not category_filter or bbox.category in category_filter)
                    ):
                        all_trajs[frame_id]["trajs"][track_id] = bbox

        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                det_score = 0
                for bbox in trajs[track_id].bboxes:
                    det_score = bbox.det_score
                    break
                if (
                    track_id in all_trajs[frame_id]["trajs"].keys()
                    and det_score <= cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"]
                ):
                    del all_trajs[frame_id]["trajs"][track_id]

    tracking_results[scene_id] = all_trajs


if __name__ == "__main__":
    logging.debug('main.py __main__ entry')
    parser = argparse.ArgumentParser(description="MCTrack")
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        help="Which Dataset: kitti/nuscenes/waymo",
    )
    parser.add_argument("--eval", "-e", action="store_true", help="evaluation")
    parser.add_argument("--load_image", "-lm", action="store_true", help="load_image")
    parser.add_argument("--load_point", "-lp", action="store_true", help="load_point")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--mode", "-m", action="store_true", help="online or offline")
    parser.add_argument("--process", "-p", type=int, default=1, help="multi-process!")
    parser.add_argument(
        "--single-scene",
        type=str,
        default=None,
        help="Run tracking on a single scene token (e.g., scene-0003)",
    )
    parser.add_argument(
        "--enable-classifier",
        action="store_true",
        help="Enable bicycle motion classifier during tracking",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Path to classifier .pth checkpoint (overrides config)",
    )
    parser.add_argument(
        "--category-filter",
        nargs="+",
        default=None,
        help="Restrict tracking to these categories (e.g., --category-filter bicycle)",
    )
    parser.add_argument(
        "--bicycle-only",
        action="store_true",
        help="Shortcut for --category-filter bicycle",
    )
    args = parser.parse_args()
#     python main.py --dataset nuscenes --single-scene scene-0003 --enable-classifier --classifier-checkpoint "hyperparam_search_20251113_114329\best_model.pth" --bicycle-only
#     python main.py --dataset nuscenes --enable-classifier --classifier-checkpoint "hyperparam_search_20251113_114329\best_model.pth" --bicycle-only -e -p 4

    if args.dataset == "kitti":
        cfg_path = "./config/kitti.yaml"
    elif args.dataset == "nuscenes":
        cfg_path = "./config/nuscenes.yaml"
    elif args.dataset == "waymo":
        cfg_path = "./config/waymo.yaml"
    if args.mode:
        cfg_path = cfg_path.replace(".yaml", "_offline.yaml")

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

    if "CLASSIFIER" not in cfg:
        cfg["CLASSIFIER"] = {}

    if args.enable_classifier or args.classifier_checkpoint:
        cfg["CLASSIFIER"].setdefault("ENABLED", True)
        cfg["CLASSIFIER"]["ENABLED"] = True
        if args.classifier_checkpoint:
            cfg["CLASSIFIER"]["CHECKPOINT_PATH"] = args.classifier_checkpoint

    checkpoint_path = cfg["CLASSIFIER"].get("CHECKPOINT_PATH")
    if checkpoint_path:
        cfg["CLASSIFIER"]["CHECKPOINT_PATH"] = os.path.abspath(checkpoint_path)

    if args.bicycle_only:
        cfg["CATEGORY_FILTER"] = ["bicycle"]
    elif args.category_filter:
        cfg["CATEGORY_FILTER"] = [str(cat).lower() for cat in args.category_filter]

    save_path = os.path.join(
        os.path.dirname(cfg["SAVE_PATH"]),
        cfg["DATASET"],
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(save_path, exist_ok=True)
    cfg["SAVE_PATH"] = save_path

    start_time = time.time()

    detections_root = os.path.join(
        cfg["DETECTIONS_ROOT"], cfg["DETECTOR"], cfg["SPLIT"] + ".json"
    )
    with open(detections_root, "r", encoding="utf-8") as file:
        print(f"Loading data from {detections_root}...")
        data = json.load(file)
        print("Data loaded successfully.")

    if args.debug:
        if args.dataset == "kitti":
            scene_lists = [str(scene_id).zfill(4) for scene_id in cfg["TRACKING_SEQS"]]
        elif args.dataset == "nuscenes":
            scene_lists = [scene_id for scene_id in data.keys()][:2]
        else:
            scene_lists = [scene_id for scene_id in data.keys()][:2]
    else:
        if args.single_scene:
            single_scene = args.single_scene.strip()
            if single_scene not in data:
                raise KeyError(f"Requested scene '{single_scene}' not found in detections JSON")
            scene_lists = [single_scene]
        else:
            scene_lists = [scene_id for scene_id in data.keys()]

    manager = multiprocessing.Manager()
    tracking_results = manager.dict()
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        func = partial(
            run, scenes_data=data, cfg=cfg, args=args, tracking_results=tracking_results
        )
        pool.map(func, scene_lists)
        pool.close()
        pool.join()
    else:
        for scene_id in tqdm(scene_lists, desc="Running scenes"):
            run(scene_id, data, cfg, args, tracking_results)
    tracking_results = dict(tracking_results)

    if args.dataset == "kitti":
        save_results_kitti(tracking_results, cfg)
        if args.eval:
            eval_kitti(cfg)
    if args.dataset == "nuscenes":
        save_results_nuscenes(tracking_results, save_path)
        save_results_nuscenes_for_motion(tracking_results, save_path)
        if args.eval:
            eval_nusc(cfg)
    elif args.dataset == "waymo":
        save_results_waymo(tracking_results, save_path)
        if args.eval:
            eval_waymo(cfg, save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
