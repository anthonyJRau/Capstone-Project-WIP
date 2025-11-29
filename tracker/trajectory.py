# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import copy

from .bbox import BBox
from typing import List
from kalmanfilter.extend_kalman import *
from scipy.optimize import curve_fit

np.set_printoptions(formatter={"float": "{:0.4f}".format})

def linear_func(x, a, b):
    return a * x + b

class Trajectory:
    def __init__(
        self,
        track_id,
        init_bbox=None,
        first_bbox=True,
        cfg=None,
    ):
        self.track_id = track_id
        self.category_num = cfg["CATEGORY_MAP_TO_NUMBER"][init_bbox.category]

        self.first_bbox = first_bbox
        self.track_length = 1
        self.unmatch_length = 0
        self.out_range_length = 0
        self.is_output = False
        self.first_updated_frame = init_bbox.frame_id
        self.last_updated_frame = init_bbox.frame_id
        self.cfg = cfg
        self.bboxes: List[BBox] = [init_bbox]
        self.matched_scores = []

        cv_init_pose = np.array(init_bbox.global_xyz_lwh_yaw[:2] + init_bbox.global_velocity)
        ca_init_pose = np.array(init_bbox.global_xyz_lwh_yaw[:2] + [0, 0, 0, 0])
        ctra_init_pose = np.array(
            init_bbox.global_xyz_lwh_yaw[:2] + [0.1, 0.1, 0.1, 0.1]
        )
        
        cv_init_yaw = np.array([init_bbox.global_yaw, 0.0])
        cv_init_size = np.array([init_bbox.lwh[0], init_bbox.lwh[1], 0.0, 0.0])

        self.frame_rate = cfg["FRAME_RATE"]
        self.cost_mode = cfg["MATCHING"]["BEV"]["COST_MODE"][self.category_num]
        self._cache_bbox_len = cfg["THRESHOLD"]["TRAJECTORY_THRE"]["CACHE_BBOX_LENGTH"][
            self.category_num
        ]
        self._max_predict_len = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "PREDICT_BBOX_LENGTH"
        ][self.category_num]
        self._max_unmatch_len = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "MAX_UNMATCH_LENGTH"
        ][self.category_num]
        self._confirmed_track_length = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "CONFIRMED_TRACK_LENGTH"
        ][self.category_num]
        self._delet_out_track_length = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "DELET_OUT_VIEW_LENGTH"
        ][self.category_num]
        self._confirmed_det_score = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "CONFIRMED_DET_SCORE"
        ][self.category_num]
        self._confirmed_match_score = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "CONFIRMED_MATCHED_SCORE"
        ][self.category_num]
        self._is_filter_predict_box = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "IS_FILTER_PREDICT_BOX"
        ][self.category_num]
        
        self.status_flag = 1  # 0:initialization / 1: confirmed / 2: obscured / 4: dead
        # we have tried to set the status_flag to 0, but it seems that it is not necessary

        self.cv_filter_pose = EKF_CV(
            dt=1 / self.frame_rate,
            n=cfg["KALMAN_FILTER_POSE"]["CV"]["N"],
            m=cfg["KALMAN_FILTER_POSE"]["CV"]["M"],
            P=np.diag(cfg["KALMAN_FILTER_POSE"]["CV"]["NOISE"][self.category_num]["P"]),
            Q=np.diag(cfg["KALMAN_FILTER_POSE"]["CV"]["NOISE"][self.category_num]["Q"]),
            R=np.diag(cfg["KALMAN_FILTER_POSE"]["CV"]["NOISE"][self.category_num]["R"]),
            init_x=cv_init_pose,
        )
        self.ca_filter_pose  = EKF_CA(
            dt=1 / self.frame_rate,
            n=cfg["KALMAN_FILTER_POSE"]["CA"]["N"],
            m=cfg["KALMAN_FILTER_POSE"]["CA"]["M"],
            P=np.diag(cfg["KALMAN_FILTER_POSE"]["CA"]["NOISE"][self.category_num]["P"]),
            Q=np.diag(cfg["KALMAN_FILTER_POSE"]["CA"]["NOISE"][self.category_num]["Q"]),
            R=np.diag(cfg["KALMAN_FILTER_POSE"]["CA"]["NOISE"][self.category_num]["R"]),
            init_x=ca_init_pose,
        )
        self.ctra_filter_pose  = EKF_CTRA(
            dt=1 / self.frame_rate,
            n=cfg["KALMAN_FILTER_POSE"]["CTRA"]["N"],
            m=cfg["KALMAN_FILTER_POSE"]["CTRA"]["M"],
            P=np.diag(cfg["KALMAN_FILTER_POSE"]["CTRA"]["NOISE"][self.category_num]["P"]),
            Q=np.diag(cfg["KALMAN_FILTER_POSE"]["CTRA"]["NOISE"][self.category_num]["Q"]),
            R=np.diag(cfg["KALMAN_FILTER_POSE"]["CTRA"]["NOISE"][self.category_num]["R"]),
            init_x=ctra_init_pose,
        )
        self.kalman_filter_yaw = KF_YAW(
            dt=1 / self.frame_rate,
            n=cfg["KALMAN_FILTER_YAW"]["CV"]["N"],
            m=cfg["KALMAN_FILTER_YAW"]["CV"]["M"],
            P=np.diag(cfg["KALMAN_FILTER_YAW"]["CV"]["NOISE"][self.category_num]["P"]),
            Q=np.diag(cfg["KALMAN_FILTER_YAW"]["CV"]["NOISE"][self.category_num]["Q"]),
            R=np.diag(cfg["KALMAN_FILTER_YAW"]["CV"]["NOISE"][self.category_num]["R"]),
            init_x=cv_init_yaw,
        )
        self.kalman_filter_size = KF_SIZE(
            dt=1 / self.frame_rate,
            n=cfg["KALMAN_FILTER_SIZE"]["CV"]["N"],
            m=cfg["KALMAN_FILTER_SIZE"]["CV"]["M"],
            P=np.diag(cfg["KALMAN_FILTER_SIZE"]["CV"]["NOISE"][self.category_num]["P"]),
            Q=np.diag(cfg["KALMAN_FILTER_SIZE"]["CV"]["NOISE"][self.category_num]["Q"]),
            R=np.diag(cfg["KALMAN_FILTER_SIZE"]["CV"]["NOISE"][self.category_num]["R"]),
            init_x=cv_init_size,
        )
        
        if cfg["KALMAN_FILTER_POSE"]["MOTION_MODE"][self.category_num] == "CV":
            self.kalman_filter_pose = self.cv_filter_pose
        elif cfg["KALMAN_FILTER_POSE"]["MOTION_MODE"][self.category_num] == "CA":
            self.kalman_filter_pose = self.ca_filter_pose 
        elif cfg["KALMAN_FILTER_POSE"]["MOTION_MODE"][self.category_num] == "CTRA":
            self.kalman_filter_pose = self.ctra_filter_pose 
        
        classifier_cfg = cfg.get("CLASSIFIER", {}) or {}
        self._stationary_categories = set(classifier_cfg.get("CATEGORIES", []))
        self._stationary_lock_threshold = classifier_cfg.get("STATIONARY_LOCK_THRESHOLD", 0.5)
        self._stationary_unlock_threshold = classifier_cfg.get("STATIONARY_UNLOCK_THRESHOLD", 0.4)
        self._stationary_transition_frames = classifier_cfg.get("TRANSITION_FRAMES", 3)
        self._min_track_length_for_lock = classifier_cfg.get("MIN_TRACK_LENGTH_FOR_LOCK", 5)
        self._stationary_q_scale_min = classifier_cfg.get("STATIONARY_Q_SCALE", 0.05)
        self._pose_Q_default = self.kalman_filter_pose.Q.copy()
        self._pose_Q_stationary = self._pose_Q_default * self._stationary_q_scale_min
        self._is_stationary_locked = False
        self._stationary_lock_count = 0
        self._stationary_unlock_count = 0

        # if cfg["IS_RV_MATCHING"]:
        #     xywh = init_bbox.transform_bbox_tlbr2xywh()
        #     init_rvbox = np.array(xywh.tolist() + [0.0, 0.0, 0.0, 0.0])
        #     self.kalman_filter_rvbox = EKF_RVBOX(
        #         dt=1 / self.frame_rate,
        #         n=cfg["KALMAN_FILTER_RVBOX"]["CV"]["N"],
        #         m=cfg["KALMAN_FILTER_RVBOX"]["CV"]["M"],
        #         P=np.diag(cfg["KALMAN_FILTER_RVBOX"]["CV"]["NOISE"][self.category_num]["P"]),
        #         Q=np.diag(cfg["KALMAN_FILTER_RVBOX"]["CV"]["NOISE"][self.category_num]["Q"]),
        #         R=np.diag(cfg["KALMAN_FILTER_RVBOX"]["CV"]["NOISE"][self.category_num]["R"]),
        #         init_x=init_rvbox,
        #     )


    def get_measure(self, bbox: BBox, filter_flag="pose"):
        global_xyz = bbox.global_xyz
        global_yaw = bbox.global_yaw
        global_velocity = bbox.global_velocity
        lwh = bbox.lwh
        
        if filter_flag == "pose":
            measure = np.array([global_xyz[0], global_xyz[1], global_velocity[0], global_velocity[1]])
        elif filter_flag == "yaw":
            vel_yaw = np.arctan2(global_velocity[1], global_velocity[0]+1e-5)
            vel_yaw_norm = norm_radian(vel_yaw)
            pose_yaw_norm = norm_radian(global_yaw)
            measure = np.array([pose_yaw_norm, vel_yaw_norm])
        elif filter_flag == "size":
            measure = np.array([lwh[0], lwh[1]])
        elif filter_flag == "rvbox":
            measure = np.array(bbox.transform_bbox_tlbr2xywh())
        else:
            raise ValueError(f"Unexpected filter_flag value: {filter_flag}")

        return measure
    
    def predict(self):
        predict_state = self.kalman_filter_pose.predict()
        predict_yaw = self.kalman_filter_yaw.predict()
        predict_size = self.kalman_filter_size.predict()
        # if self.cfg["IS_RV_MATCHING"]:
        #     predict_rvbox = self.kalman_filter_rvbox.predict()
        #     self.bboxes[-1].x1y1x2y2_predict = predict_rvbox[:4]

        if self._is_stationary_locked and predict_state.shape[0] >= 4:
            predict_state[2] = 0
            predict_state[3] = 0
        
        global_xyz_lwh_yaw_fusion = self.bboxes[-1].global_xyz_lwh_yaw_fusion

        predict_xyz = predict_state[:2].tolist() + [global_xyz_lwh_yaw_fusion[2]]
        # predict_lwh = predict_size[:2].tolist() + [global_xyz_lwh_yaw_fusion[5]]
        predict_lwh = [global_xyz_lwh_yaw_fusion[3], global_xyz_lwh_yaw_fusion[4], global_xyz_lwh_yaw_fusion[5]]
        self.bboxes[-1].global_xyz_lwh_yaw_predict = predict_xyz + predict_lwh + [global_xyz_lwh_yaw_fusion[6]]

        self.bboxes[-1].global_yaw_fusion = predict_yaw[0]
        self.bboxes[-1].lwh_fusion = predict_lwh   

    def update(self, bbox: BBox, matched_score):
        bbox.track_id = self.track_id
        self.track_length += 1
        bbox.track_length = self.track_length
        self.last_updated_frame = bbox.frame_id
        self.unmatch_length = 0
        self.bboxes.append(bbox)

        if len(self.bboxes) > self._cache_bbox_len:
            self.bboxes.pop(0)

        self.matched_scores.append(matched_score)
        
        global_velocity_diff = self.cal_diff_velocity()
        self.bboxes[-1].global_velocity_diff = global_velocity_diff
        global_velocity_curve = self.cal_curve_velocity()
        self.bboxes[-1].global_velocity_curve = global_velocity_curve

        self._update_stationary_state(bbox)
        if self._is_stationary_locked:
            if self.kalman_filter_pose.x.shape[0] >= 4:
                self.kalman_filter_pose.x[2] = 0
                self.kalman_filter_pose.x[3] = 0
            self.bboxes[-1].global_velocity = [0.0, 0.0]
            self.bboxes[-1].global_velocity_fusion = [0.0, 0.0]
        
        # ======== pose filter ==========
        pose_mesure = self.get_measure(bbox, filter_flag="pose")   
        if self.kalman_filter_pose.m == 2:
            pose_mesure = pose_mesure[:2]
        update_state = self.kalman_filter_pose.update(pose_mesure)
        self.bboxes[-1].global_velocity_fusion = update_state[2:4].tolist()
        if self._is_stationary_locked and update_state.shape[0] >= 4:
            update_state[2] = 0
            update_state[3] = 0
            self.kalman_filter_pose.x[2] = 0
            self.kalman_filter_pose.x[3] = 0
            self.bboxes[-1].global_velocity_fusion = [0.0, 0.0]
        
        # ======== yaw filter ==========
        yaw_mesure = self.get_measure(bbox, filter_flag="yaw")
        update_yaw = self.kalman_filter_yaw.update(yaw_mesure)
        self.bboxes[-1].global_yaw_fusion = update_yaw[0]
        
        # ======== size filter ==========
        size_mesure = self.get_measure(bbox, filter_flag="size")  
        update_size = self.kalman_filter_size.update(size_mesure)
        update_lwh = update_size[:2].tolist() + [self.bboxes[-1].lwh[2]]
        self.bboxes[-1].lwh_fusion = update_lwh

        # ======== rv box filter ==========
        # if self.cfg["IS_RV_MATCHING"]:
        #     rvbox_mesure = self.get_measure(bbox, filter_flag="rvbox")  
        #     update_rvbox = self.kalman_filter_rvbox.update(rvbox_mesure)
        #     self.bboxes[-1].x1y1x2y2_fusion = bbox.transform_bbox_xywh2tlbr(update_rvbox[:4])

        # self.bboxes[-1].global_xyz_lwh_yaw_fusion = update_xyz + update_lwh + [update_yaw[0]]
        self.bboxes[-1].global_xyz_lwh_yaw_fusion = np.append(
            update_state[:2], self.bboxes[-1].global_xyz_lwh_yaw[2:]
        )

        self.bboxes[-1].matched_score = matched_score 

        if self.track_length > self._confirmed_track_length or (
            matched_score > self._confirmed_match_score
            and self.bboxes[-1].det_score > self._confirmed_det_score
        ):
            self.status_flag = 1

        return self.bboxes[-1]

    def _update_stationary_state(self, bbox: BBox):
        if bbox.category not in self._stationary_categories:
            if self._is_stationary_locked:
                self._exit_stationary_mode()
            return

        prob = bbox.classifier_stationary_probability
        if prob is None:
            return

        # Apply smooth Q matrix adjustment based on probability
        scale_factor = self._interpolate_q_scale(prob)
        self.kalman_filter_pose.Q = self._pose_Q_default * scale_factor

        # Gradual state transition with frame counting
        if not self._is_stationary_locked:
            if prob >= self._stationary_lock_threshold:
                self._stationary_lock_count += 1
                if self._stationary_lock_count >= self._stationary_transition_frames:
                    self._enter_stationary_mode()
            else:
                self._stationary_lock_count = 0
        else:
            if prob < self._stationary_unlock_threshold:
                self._stationary_unlock_count += 1
                if self._stationary_unlock_count >= self._stationary_transition_frames:
                    self._exit_stationary_mode()
            else:
                self._stationary_unlock_count = 0

    def _interpolate_q_scale(self, stationary_prob):
        """Smoothly interpolate Q scale based on stationary probability."""
        max_scale = 1.0
        # Linear interpolation: high prob -> low scale (more constrained)
        return max_scale - (stationary_prob * (max_scale - self._stationary_q_scale_min))

    def _enter_stationary_mode(self):
        # Only lock if track is established
        if self.track_length < self._min_track_length_for_lock:
            self._stationary_lock_count = 0  # Reset counter
            return
        
        self._is_stationary_locked = True
        self._stationary_lock_count = 0
        self.kalman_filter_pose.Q = self._pose_Q_stationary.copy()
        if self.kalman_filter_pose.x.shape[0] >= 4:
            self.kalman_filter_pose.x[2] = 0
            self.kalman_filter_pose.x[3] = 0

    def _exit_stationary_mode(self):
        self._is_stationary_locked = False
        self._stationary_unlock_count = 0
        self.kalman_filter_pose.Q = self._pose_Q_default.copy()

    def unmatch_update(self, frame_id):
        self.unmatch_length += 1

        predict_state = self.kalman_filter_pose.predict()
        predict_yaw = self.kalman_filter_yaw.predict()
        predict_size = self.kalman_filter_size.predict()
        if self._is_stationary_locked and predict_state.shape[0] >= 4:
            predict_state[2] = 0
            predict_state[3] = 0
        # if self.cfg["IS_RV_MATCHING"]:
        #     predict_rvbox = self.kalman_filter_rvbox.predict()
        #     fake_update_rvbox = self.kalman_filter_rvbox.update(predict_rvbox[:4])
        if self.kalman_filter_pose.m == 2:
            predict_state = predict_state[:2]
        fake_update_state = self.kalman_filter_pose.update(predict_state)
        # fake_update_yaw = self.kalman_filter_yaw.update(predict_yaw[:2])
        # fake_update_size = self.kalman_filter_size.update(predict_size[:2])

        fake_bbox = copy.deepcopy(self.bboxes[-1])
        fake_bbox.det_score = 0
        fake_bbox.is_fake = True
        fake_bbox.frame_id = frame_id
        
        fake_xyz = fake_update_state[:2].tolist() + [fake_bbox.global_xyz_lwh_yaw[2]]
        fake_lwh = predict_size[:2].tolist() + [fake_bbox.global_xyz_lwh_yaw[5]]
        fake_bbox.global_xyz_lwh_yaw = fake_xyz + fake_lwh + [self.bboxes[-1].global_xyz_lwh_yaw[-1]]
        fake_bbox.global_xyz_lwh_yaw_fusion = fake_xyz + fake_lwh + [self.bboxes[-1].global_xyz_lwh_yaw[-1]]
        if self._is_stationary_locked:
            if self.kalman_filter_pose.x.shape[0] >= 4:
                self.kalman_filter_pose.x[2] = 0
                self.kalman_filter_pose.x[3] = 0
            fake_bbox.global_velocity = [0.0, 0.0]
            fake_bbox.global_velocity_fusion = [0.0, 0.0]
        
        self.bboxes.append(fake_bbox)
        self.matched_scores.append(0)
        self.bboxes[-1].matched_score = 0
        self.bboxes[-1].unmatch_length = self.unmatch_length
        
        global_velocity_diff = self.cal_diff_velocity()
        self.bboxes[-1].global_velocity_diff = global_velocity_diff
        global_velocity_curve = self.cal_curve_velocity()
        self.bboxes[-1].global_velocity_curve = global_velocity_curve

        if len(self.bboxes) > self._cache_bbox_len:
            self.bboxes.pop(0)

        # ========= Adaptive unmatch tolerance for stationary-locked tracks =========
        max_unmatch = self._max_unmatch_len
        max_predict = self._max_predict_len
        
        # Extend patience for stationary-locked tracks
        # Rationale: Stationary bikes have stable predictions (velocity=0), so longer
        # patience during brief occlusions helps recover tracks without degrading MOTP
        if self._is_stationary_locked:
            # Double the patience for stationary bikes (typically 1 frame -> 2-3 frames)
            max_unmatch = min(self._max_unmatch_len * 2, 3)  # Cap at 3 frames
            # Also extend prediction length slightly
            max_predict = min(self._max_predict_len + 3, 16)  # Add 3 frames, cap at 16

        if self.status_flag == 0 and self.track_length > self._confirmed_track_length: 
            self.status_flag = 4

        if self.status_flag == 1 and self.unmatch_length > max_unmatch:
            self.status_flag = 2

        if self.status_flag == 2 and self.unmatch_length > max_predict:
            self.status_flag = 4

        return

    def logit(self, y):
        if y == 0:
            return -10000
        if y <= 0 or y >= 1:
            raise ValueError("Input must be in the range (0, 1).")
        return np.log(y / (1 - y))
    
    def filtering(self):
        '''
        Refer: https://github.com/hailanyi/3D-Multi-Object-Tracker/blob/master/tracker/trajectory.py
        Info: filtering the trajectory in a global or near online way
        '''
        detected_num = 0.00001
        score_sum = 0
        if_has_unmatched = 0
        unmatch_bbox_sum = 0
        start_xyz_lwh_yaw = None 
        start_frame = 0

        last_xyz_lwh_yaw_fusion = None
        for bbox in self.bboxes:
            frame_id = bbox.frame_id
            bbox.det_score = self.logit(bbox.det_score)
            if bbox.det_score > -10000:
                detected_num += 1
                score_sum += bbox.det_score  
            if self.first_updated_frame <= frame_id <= self.last_updated_frame and bbox.is_fake and self.is_output: 
                bbox.is_interpolation = True
                if if_has_unmatched == 0:
                    start_xyz_lwh_yaw = last_xyz_lwh_yaw_fusion
                    if_has_unmatched = 1
                    unmatch_bbox_sum = 1
                    start_frame = frame_id
                else:
                    unmatch_bbox_sum += 1
            elif not bbox.is_fake and if_has_unmatched == 1:
                end_frame = frame_id - 1
                end_xyz_lwh_yaw = bbox.global_xyz_lwh_yaw_fusion
                gap = (end_xyz_lwh_yaw - start_xyz_lwh_yaw) / (unmatch_bbox_sum + 1)
                last_xyz_lwh_yaw = start_xyz_lwh_yaw
                if unmatch_bbox_sum >= 2:
                    cnt = 0
                    for bbox_tmp in self.bboxes:
                        if bbox_tmp.frame_id >= start_frame and bbox_tmp.frame_id <= end_frame:
                            cnt += 1
                            last_xyz_lwh_yaw += gap
                            bbox_tmp.global_xyz_lwh_yaw_fusion[0] = last_xyz_lwh_yaw[0]
                            bbox_tmp.global_xyz_lwh_yaw_fusion[1] = last_xyz_lwh_yaw[1]
                            bbox_tmp.global_xyz_lwh_yaw_fusion[2] = last_xyz_lwh_yaw[2]

                if_has_unmatched = 0
            last_xyz_lwh_yaw_fusion = bbox.global_xyz_lwh_yaw_fusion
                
        score = score_sum / detected_num
        for bbox in self.bboxes:
            bbox.det_score = score
            
    def cal_diff_velocity(self):
        if len(self.bboxes) > 1:
            prev_bbox = self.bboxes[-2]
            cur_bbox = self.bboxes[-1]
            time_diff = (cur_bbox.frame_id - prev_bbox.frame_id) / self.frame_rate
            if time_diff > 0:
                position_diff = np.array(cur_bbox.global_xyz[:2]) - np.array(prev_bbox.global_xyz[:2])
                global_velocity_diff = (position_diff / time_diff).tolist()
            else:
                global_velocity_diff = [0.0, 0.0]
        else:
            global_velocity_diff = [0.0, 0.0]
        return global_velocity_diff
    
    def cal_curve_velocity(self):
        if len(self.bboxes) > 2:
            x_vals = [bb.frame_id for bb in self.bboxes[-3:]]
            y_vals_x = [bb.global_xyz[0] for bb in self.bboxes[-3:]]
            y_vals_y = [bb.global_xyz[1] for bb in self.bboxes[-3:]]

            try:
                popt_x, _ = curve_fit(linear_func, x_vals, y_vals_x)
                popt_y, _ = curve_fit(linear_func, x_vals, y_vals_y)
                global_velocity_curve = [popt_x[0], popt_y[0]]  # Using the quadratic coefficient as the curvature
            except RuntimeError:
                global_velocity_curve = [0.0, 0.0]
        else:
            global_velocity_curve = [0.0, 0.0]
        
        return global_velocity_curve
