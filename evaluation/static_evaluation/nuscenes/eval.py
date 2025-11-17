import os
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
import json
import logging

logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def eval_nusc(args):
    logging.info(f'Called eval_nusc with args: {args}')
    result_path = os.path.join(args["SAVE_PATH"], "results.json")
    logging.info(f"Result path: {result_path}")
    with open(result_path) as f:
        raw_results = json.load(f)
        logging.info(f"Loaded results.json, keys: {list(raw_results.keys())}")
        if "results" in raw_results:
            first_key = next(iter(raw_results["results"]))
            logging.info(f"First sample_token: {first_key}")
            logging.info(f"First prediction entry: {raw_results['results'][first_key][0]}")
            logging.info(f"Total scenes in results: {len(raw_results['results'])}")
            for scene, entries in raw_results['results'].items():
                logging.info(f"Scene {scene} has {len(entries)} entries")
    eval_path = os.path.join(args["SAVE_PATH"], "eval_result/")
    nusc_path = args["DATASET_ROOT"]
    cfg = track_configs("tracking_nips_2019")
    logging.info(f'Calling TrackingEval with config: {cfg}, result_path: {result_path}, eval_set: val, output_dir: {eval_path}, nusc_version: v1.0-trainval, nusc_dataroot: {nusc_path}')
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    logging.info('TrackingEval instance created')
    print("result in " + result_path)
    try:
        metrics_summary = nusc_eval.main()
        logging.info(f"metrics_summary: {metrics_summary}")
    except Exception as e:
        logging.exception(f"Exception during nusc_eval.main(): {e}")
    logging.info('Finished eval_nusc')