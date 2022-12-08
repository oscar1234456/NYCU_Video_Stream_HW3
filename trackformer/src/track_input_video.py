# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from matplotlib import pyplot as plt, colors
from torch.utils.data import DataLoader

from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.datasets.coco import make_coco_transforms
from trackformer.datasets.transforms import Compose
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import (evaluate_mot_accums, get_mot_accum,
                                          interpolate_tracks, plot_sequence, rand_cmap)

import cv2
from PIL import Image

mm.lap.default_solver = 'lap'

ex = sacred.Experiment('track')
ex.add_config('cfgs/track.yaml')
ex.add_named_config('reid', 'cfgs/track_reid.yaml')


@ex.automain
def main(seed, dataset_name, obj_detect_checkpoint_file, tracker_cfg,
         write_images, output_dir, interpolate, verbose, load_results_dir,
         data_root_dir, generate_attention_maps, frame_range,
         _config, _log, _run, camera, obj_detector_model=None):
    def mouse_handler(event, x, y, flags, data):
        # data[0]: need_to_plot_track_id_pool
        # data[1]: frame
        # data[2]: now_frame_show_up
        if event == cv2.EVENT_LBUTTONDOWN:
            print("2. get points: (x, y) = ({}, {})".format(x, y))
            # 顯示 (x,y) 並儲存到 list中
            # count min dist
            min_dist = 9999999
            near_track_id = None
            print(f"min_dest: {min_dist}")
            print(f"near_id:{near_track_id}")
            for now_track_id, center in data[2].items():

                dist = (x - center[0]) ** 2 + (y - center[1]) ** 2
                print(f"3. center:{center}, id: {now_track_id}, dist: {dist}")
                if dist < min_dist:
                    # 目前最近
                    print(f"change to id:{now_track_id}")
                    near_track_id = now_track_id
                    min_dist = dist
            print(f"4. final track_id: {near_track_id}")
            if near_track_id not in data[0]:
                data[0].add(near_track_id)
            else:
                data[0].remove(near_track_id)

    # 直接列印此_run之實驗配置
    sacred.commands.print_config(_run)

    # set all seeds (使用666預設配置)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True


    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    #### 讀取track.yaml中指定的obj detect model weight ####
    # 取得 obj detect model 之 path
    obj_detect_config_path = os.path.join(os.path.dirname(obj_detect_checkpoint_file), 'config.yaml')
    # 讀取 obj detect model 的 yaml file
    obj_detect_args = nested_dict_to_namespace(yaml.unsafe_load(open(obj_detect_config_path)))
    # 抓取image transform的方式
    img_transform = obj_detect_args.img_transform
    # 按照前面的args去建立模型 (model, criterion, postprocessors)
    obj_detector, _, obj_detector_post = build_model(obj_detect_args)

    # load model weight
    obj_detect_checkpoint = torch.load(
        obj_detect_checkpoint_file, map_location=lambda storage, loc: storage)

    obj_detect_state_dict = obj_detect_checkpoint['model']
    # obj_detect_state_dict = {
    #     k: obj_detect_state_dict[k] if k in obj_detect_state_dict
    #     else v
    #     for k, v in obj_detector.state_dict().items()}

    obj_detect_state_dict = {
        k.replace('detr.', ''): v
        for k, v in obj_detect_state_dict.items()
        if 'track_encoding' not in k}

    obj_detector.load_state_dict(obj_detect_state_dict)
    if 'epoch' in obj_detect_checkpoint:
        _log.info(f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")

    obj_detector.cuda()

    if hasattr(obj_detector, 'tracking'):
        obj_detector.tracking()

    track_logger = None
    tracker = Tracker(
        obj_detector, obj_detector_post, tracker_cfg,
        generate_attention_maps, track_logger, verbose)

    # 製作 Dataset (最後目的: 像frame_data即可)

    transform = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))
    tracker.reset()

    now_frame_id = -1

    if camera:
        videoCapture = cv2.VideoCapture(0)  # 影像輸入: camera
    else:
        videoCapture = cv2.VideoCapture("../test_video/test.mp4")  # 影像輸入: 影片檔

    success, frame = videoCapture.read()
    now_frame_id += 1
    need_to_plot_track_id_pool = set()
    while success:
        # 成功讀到frame，開始圖片處理/追蹤
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉RGB
        frame_image = Image.fromarray(frame_RGB)  # 轉PIL.Image
        width_orig, height_orig = frame_image.size  # 紀錄原圖Size

        frame_image_trans, _ = transform(frame_image)  # transform image
        width, height = frame_image_trans.size(2), frame_image_trans.size(1)  # transform image的size

        # 製作sample data dict
        sample = {}
        sample['img'] = frame_image_trans.unsqueeze(0)
        sample['dets'] = torch.tensor([]).reshape(1, 0)
        sample['orig_size'] = torch.as_tensor([[int(height_orig), int(width_orig)]])
        sample['size'] = torch.as_tensor([[int(height), int(width)]])
        with torch.no_grad():
            tracker.step(sample)

        results = tracker.get_results()
        mx = 0

        tracks = results

        for track_id, track_data in tracks.items():
            mx = max(mx, track_id)

        now_frame_show_up = dict()

        for track_id, track_data in tracks.items():
            if now_frame_id in track_data.keys():
                # 代表該物件(id: track_id)在此frame(id: now_frame_id)中有出現
                bbox = track_data[now_frame_id]['bbox']
                now_frame_show_up[track_id] = [int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[1] - ((bbox[1] - bbox[3]) / 2))]  # now_frame_show_up={"track_id":"bbox coordinate"} 存放該frame有的物件座標
                if 'mask' in track_data[now_frame_id]:
                    print("have mask (emitted)")
                else:
                    if track_id in need_to_plot_track_id_pool:
                        # 如果該物件(id: track_id)是被選取中的，則畫出bbox
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0))
                        fontFace = cv2.FONT_HERSHEY_COMPLEX
                        fontScale = 0.5
                        thickness = 1
                        cv2.putText(frame, str(track_id), (bbox[0], bbox[1]), fontFace, fontScale, (0, 0, 0), thickness)
                    print(f"1. circle: {(int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[1] - ((bbox[1] - bbox[3]) / 2)))}")
                    cv2.circle(frame,
                               (int(bbox[0] + ((bbox[2] - bbox[0]) / 2)), int(bbox[1] - ((bbox[1] - bbox[3]) / 2))),
                               radius=10, color=(255, 0, 0), thickness=2)

        cv2.imshow('windows', frame)
        cv2.setMouseCallback("windows", mouse_handler, (need_to_plot_track_id_pool, frame, now_frame_show_up.copy()))
        print(f"need_to_plot_track_id_pool: {need_to_plot_track_id_pool}")
        success, frame = videoCapture.read()
        now_frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
