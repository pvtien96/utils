#convert groundtruth from .mat to yolov5 format
import argparse
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import scipy.io as sio

def get_bbox(point_list):
    min_x, min_y = max_x, max_y = point_list[0][:2]
    for point in point_list:
        if len(point) == 2:
            x, y = point[:2]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    bbox = [min_x, min_y, max_x, max_y]
    return bbox

def get_polygons(point_list):
    polygon = []
    for point in point_list:
        if len(point) == 2:
            polygon.extend(point)
    return [polygon]

def main():
    args = get_parser().parse_args()
    images_path = os.path.join(args.target_path,'images/train/')
    labels_path = os.path.join(args.target_path,'labels/train/')
    if not os.path.isdir(images_path):
        os.mkdir(images_path)
    if not os.path.isdir(labels_path):
        os.mkdir(labels_path)
    directories = os.listdir(args.root_path)
    for i in tqdm(range(len(directories))):
        directory = directories[i]
        path2directory = os.path.join(args.root_path, directory) 
        assert os.path.isfile(os.path.join(path2directory, 'polygons.mat')), "Error: path error, polygons.mat not found"
        mat_file = os.path.join(path2directory, 'polygons.mat')
        image_path_array = []
        for roots, dirs, files in os.walk(path2directory):
            for file in files:
                if file.endswith(".jpg"):
                    file = os.path.join(roots, file)
                    image_path_array.append(file)
                   
            #sort image_path_array
            image_path_array.sort()
            boxes = sio.loadmat(mat_file)
            polygons = boxes["polygons"][0]
            for img in range(len(image_path_array)):
                image = cv2.imread(image_path_array[img])
                im_height, im_width = image.shape[:2]
                im_name = directory + '__' + image_path_array[img].split('.')[0].split('/')[-1] + '.jpg'
                copyfile(image_path_array[img], os.path.join(images_path, im_name))
                im_txt_name = im_name.split('.')[0] + '.txt'
                out_txt = open(os.path.join(labels_path, im_txt_name), 'w')
                for point_list in polygons[img]:
                    if point_list.size:
                        top, left, bottom, right = get_bbox(point_list)
                        xc = ((top+bottom)/2)/im_width
                        yc = ((left+right)/2)/im_height
                        width = (bottom-top)/im_width
                        height = (right-left)/im_height
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
