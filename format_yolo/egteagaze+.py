#convert groundtruth from .json to yolov5 format
import argparse
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import glob

def main():
    args = get_parser().parse_args()
    images_path = os.path.join(args.target_path,'images/train/')
    labels_path = os.path.join(args.target_path,'labels/train/')
    if not os.path.isdir(images_path):
        os.mkdir(images_path)
    if not os.path.isdir(labels_path):
        os.mkdir(labels_path)
    img_dir = args.root_path + 'Images/*.jpg'
    img_list = glob.glob(img_dir)
    for idx, img_file in enumerate(tqdm(img_list)):
        #get the mask of the image
        mask_file = img_file.replace('/Images/', '/Masks/')
        mask_file = mask_file.replace('.jpg', '.png')
        img = cv2.imread(img_file)
        im_name = img_file.split('.')[0].split('/')[-1]
        copyfile(img_file, os.path.join(images_path, im_name)+'.jpg')
        im_txt_name = im_name + '.txt'
        out_txt = open(os.path.join(labels_path, im_txt_name), 'w')
        mask = cv2.imread(mask_file, 0) #mask 0-255
        ret, mask_thresh = cv2.threshold(mask, 5, 1, cv2.THRESH_BINARY)
        im_height, im_width = img.shape[:2]
        contours, __ = cv2.findContours(mask_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xc = (x + w/2)/im_width
            yc = (y + h/2)/im_height
            width = w/im_width
            height = h/im_height
            line = [0, xc, yc, width, height]
            out_txt.write(" ".join(str(item) for item in line) + "\n")
    return 0

def get_parser():
    parser = argparse.ArgumentParser(description="Convert .json to yolov5 format")
    parser.add_argument("--root_path", 
        type=str,
        default='/home/n/micand26/gt/',
        help='path to input folder contains detection groundtruth', 
    )
    parser.add_argument("--target_path",
        type=str,
        default='/home/n/micand26_yolo/',
        help='path to target folder contains yolov5 format data',
    )
    return parser

if __name__ == "__main__":
    main()
