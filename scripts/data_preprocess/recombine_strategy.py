import os
import cv2
import time
import json
import argparse
import random

import numpy as np
from tqdm import tqdm

# segment anything
'''
from segment_anything import (
    build_sam,
    SamPredictor
)
'''

from scripts.data_preprocess.recombine_utils import Robutness
from scripts.data_preprocess.recombine_utils import load_annos, process_sample, update_bbox_info, unify_extrinsic_params_tools, visual_sample_info
from scripts.data_preprocess.recombine_utils import objects_combine_tools, frame_combine_tools, frame_combine_tools_v2, save_kitti_format

def parse_option():
    parser = argparse.ArgumentParser('Mix-Teaching data preprocess', add_help=False)
    parser.add_argument('--src-root', type=str, default="data/rope3d", help='root path to src rope3d dataset')
    parser.add_argument('--dest-root', type=str, default="data/gen_dair", help='root path to result rope3d dataset')
    parser.add_argument('--vis', type=bool, default=False, help='flag to control visualization')
    args = parser.parse_args()
    return args

def read_split(split_txt):
    with open(split_txt, "r") as file:
        lines = file.readlines()
    split_list = list()
    for line in lines:
        split_list.append(line.rstrip('\n'))
    return split_list

def write_split(split_list, split_txt):
    wf = open(split_txt, "w")
    for line in split_list:
        line_string = line + "\n"
        wf.write(line_string)
    wf.close()

def write_json(json_file, data):
    with open(json_file, 'w') as file:
        json.dump(data, file)

def load_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def read_denorm(src_root, subset, frame_id):
    denorm_file = os.path.join(src_root, subset, "denorm", frame_id + ".txt")
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

if __name__ == "__main__":
    robutness = Robutness()
    args = parse_option()
    split_root = 'data/dair-in-rope3d/split_root'
    src_root, dest_root, vis = args.src_root, args.dest_root, args.vis
    train_frame_ids = read_split(os.path.join(split_root, "train.txt"))
    val_frame_ids = read_split(os.path.join(split_root, "val.txt"))
    raw_frame_ids = read_split(os.path.join(split_root, "raw.txt"))

    # train_frame_ids = random.sample(train_frame_ids, 5)
    # val_frame_ids = random.sample(val_frame_ids, 5)

    print(len(train_frame_ids), train_frame_ids[0], len(val_frame_ids), val_frame_ids[0], len(raw_frame_ids))
    print("stage 01: processing train split ...")
    for frame_id in tqdm(train_frame_ids):
        sample_info= process_sample(src_root, "training", frame_id)
        sample_info = update_bbox_info(sample_info)  
        save_kitti_format(dest_root, sample_info, sample_info["img_path"])
    
    print("stage 02: processing val split ...")
    for frame_id in tqdm(val_frame_ids):
        sample_info= process_sample(src_root, "training", frame_id, 'validation-image_2')
        sample_info = update_bbox_info(sample_info)
        save_kitti_format(dest_root, sample_info, sample_info["img_path"])

    
    raw_frame_ids = random.sample(raw_frame_ids, 5000)
    cls_focus = ["car", "van", "truck", "bus", "pedestrian", "cyclist", "motorcyclist", "tricyclist"]
    # cls_focus = ["car", "van", "truck", "bus"]
    print("stage 03: processing raw split to select background images")
    raw_ids = list()
    for frame_id in tqdm(raw_frame_ids):
        label_path = os.path.join(src_root, "training", "label_2", frame_id + ".txt")
        annos_cam = load_annos(label_path)
        obj_cnt = 0
        for anno in annos_cam:
            if anno["name"] in cls_focus:
                obj_cnt += 1
        # print('obj_cnt: ', obj_cnt)
        if obj_cnt < 15:
            raw_ids.append(frame_id)
            # print(frame_id, len(annos_cam))
    raw_ids = random.sample(raw_ids, 20)

    print("stage 03: processing conbine split ...")
    print("init sam predictor ....")
    '''
    predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth").to("cuda"))
    '''
    combine_frame_ids = list()
    count = 0
    print("raw_ids: ", raw_ids)
    for raw_id in tqdm(raw_ids):
        train_frame_ids_gen = random.sample(train_frame_ids, 500)
        for train_id in tqdm(train_frame_ids_gen):
            '''
            sample_info_combined = frame_combine_tools_v2(robutness, predictor, src_root, [train_id], raw_id, count, sample_ratio=1.0)
            '''
            sample_info_combined = frame_combine_tools(robutness, src_root, train_id, raw_id, vis)
            combine_frame_ids.append(sample_info_combined["frame_id"])
            save_kitti_format(dest_root, sample_info_combined, "training-image_2a")
            count += 1

    print("total conbine samples: ", count)
    print("stage 04: saving split set ...")
    train_frame_ids += combine_frame_ids
    write_split(train_frame_ids, os.path.join(dest_root, "train.txt"))
    write_split(val_frame_ids, os.path.join(dest_root, "val.txt"))
