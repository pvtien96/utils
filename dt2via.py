import argparse
import os
import time
from distutils.util import strtobool
import cv2
import numpy as np
from tqdm import tqdm
import json

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.merge_from_list(args.opts)
    return cfg

def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(setup_cfg(args))
    video = args.input + args.input.split('/')[6] + '.avi'
    assert os.path.isfile(video), "Error: path error, input file not found"
    
    inp_vid = cv2.VideoCapture(video)
    num_frames = int(inp_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    data = {}
    
    for frameID in tqdm(range(num_frames)):
        im_name = str(frameID).zfill(len(str(num_frames))) + ".png" 
        im_path = os.path.join(args.input, im_name)
        ret, im = inp_vid.read()
        cv2.imwrite(im_path, im)
        size = os.path.getsize(im_path)
        key = im_name + str(size)
        predictions = predictor(im)
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        frame_dict = {}
        frame_dict["filename"] = im_name
        frame_dict["size"] = size
        region_list = []
        for pred_id in range(len(boxes)):
            t, l, b, r = boxes[pred_id]
            pred_dict = {}
            bbox_dict = {}
            bbox_dict["name"] = "polygon"
            bbox_dict["all_points_x"] = [int(t), int(b), int(b), int(t)]
            bbox_dict["all_points_y"] = [int(l), int(l), int(r), int(r)]
            pred_dict["shape_attributes"] = bbox_dict
            pred_dict["region_attributes"] = {"category_id": str(pred_id + 1)}
            region_list.append(pred_dict)
        frame_dict["regions"] = region_list
        frame_dict["file_attributes"] = {} 
        data[key] = frame_dict 
    via_path = os.path.join(args.input, "via_export_json.json")
    with open(via_path, 'w') as outfile:
        json.dump(data, outfile)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 to via")
    parser.add_argument("--input", 
         type=str,
         default='/media/data3/EgoCentric_Nafosted/micand26/gt/',
         help='path to input folder', 
    )
    parser.add_argument(
        "--config-file",
        default="../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to detectron2 config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    main()
