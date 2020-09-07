import argparse
import os
import cv2
from tqdm import tqdm
import csv

#for verify_load_data
import numpy as np 
import random
from detectron2.utils.visualizer import Visualizer

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

import scipy.io as sio
import json 

#for evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def setup_cfg(args):
    #load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #initialize a pretrained weights
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Mica datasets for Detectron2")
    parser.add_argument(
        "--config-file",
        default="../../configs/Base-RCNN-FPN.yaml",
        metavar="FILE",
        help="path to config file",
        )
    parser.add_argument(
        "--mica-dir",
        default="./",
        help="path to mica dataset"
        )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_hand_dicts(img_dir):
    directories = os.listdir(img_dir)
    dataset_dicts = []
    for directory in directories:
        basename = str(directory) + '.txt'
        directory = os.path.join(img_dir, directory)
        print(directory)
        json_file = os.path.join(directory, "via_export_json.json")
        f = open(os.path.join(directory, basename), "w+")
        with open(json_file) as f:
            imgs_anns = json.load(f)
    
        with open(os.path.join(directory, basename), 'a+') as f:
            for idx, v in enumerate(imgs_anns.values()):       
                annos = v["regions"]
                for anno in annos:
                    region_attributes = anno["region_attributes"]
                    category_id = int(region_attributes["category_id"])
                    if category_id == 8 or category_id == 9:
                        anno = anno["shape_attributes"]
                        px = anno["all_points_x"]
                        py = anno["all_points_y"]
                        line = [idx+1, category_id - 7, np.min(px), np.min(py), np.max(px)-np.min(px), np.max(py)-np.min(py), 1]
                        print(line)
                        dataset_dicts.append(line)
                        f.write(",".join(str(item) for item in line) + "\n")
    return dataset_dicts

def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    for d in ["train", "val"]:
        get_hand_dicts(args.mica_dir + d)
    return 0

if __name__ == "__main__":
    main()
