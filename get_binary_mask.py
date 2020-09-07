# read png, save mask
# python3 vid2jpg.py --root-path /home/n/Downloads/Egodatasets/DHY/ --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --confidence-threshold 0.95 --opts MODEL.WEIGHTS ../output/EgteaGaze+_mask_rcnn_R_50_FPN_1x/model_final.pth MODEL.ROI_HEADS.NUM_CLASSES 1
import argparse
import os
import cv2
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import scipy.misc
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger


def get_parser():
    parser = argparse.ArgumentParser(description="mp4 to jpg")
    parser.add_argument("--root_path", default="/media/data3/EgoCentric_Nafosted/non_skip/temp/", help="Root path")
    parser.add_argument("--config-file", default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold",)
    parser.add_argument("--opts", help="config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER,)
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_binary_mask(image):
    #outputs = new_predictor(image)
    outputs = predictor(image)
    predict_masks = outputs["instances"].pred_masks
    temp = np.zeros_like(image[:, :, 0])
    num_instances = len(predict_masks)
    for i in range(num_instances):
        predict_mask_i = predict_masks[i]
        temp += np.array(predict_mask_i.to("cpu").numpy() * 255).astype(np.uint8)
    return temp


def mp42jpg(mp4_file, mp4_file_path, jpg_file_path, mask_dir_path):
    vid = cv2.VideoCapture(mp4_file_path)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(num_frames)):
        hasFrame, frame = vid.read()
        if hasFrame:
            name_frame = mp4_file.split('.')[0] + "_" + str(i).zfill(len(str(num_frames))) + ".jpg"
            path_frame = os.path.join(jpg_file_path, name_frame)
            cv2.imwrite(path_frame, frame)
            path_bmask = os.path.join(mask_dir_path, name_frame)
            bmask = get_binary_mask(frame)
            cv2.imwrite(path_bmask, bmask)
    return


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
 
    #args = get_parser().parse_args()
    directories = os.listdir(args.root_path)
    for directory in directories:
        print(directory)
        sub_path = os.path.join(args.root_path, directory)
        files = os.listdir(sub_path)
        for my_file in files:
            if my_file.endswith(".png"):
                im_path = os.path.join(sub_path, my_file)
                im = cv2.imread(im_path)
                bmask = get_binary_mask(im)
                cv2.imwrite(im_path, bmask)
