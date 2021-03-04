#convert groundtruth from .json to yolov5 format
import argparse
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from shutil import copyfile

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
        assert os.path.isfile(os.path.join(path2directory, 'via_export_json.json')), "Error: path error, via_export_json.json not found"
        json_file = os.path.join(path2directory, 'via_export_json.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)
        for idx, v in tqdm(enumerate(imgs_anns.values()), total=len(imgs_anns.values())):
            filename = os.path.join(path2directory, v["filename"])
            im = cv2.imread(filename)
            im_height, im_width = im.shape[:2]
            im_name = directory + '__' + v["filename"]
            copyfile(filename, os.path.join(images_path, im_name))
            im_txt_name = im_name.split('.')[0] + '.txt'
            out_txt = open(os.path.join(labels_path, im_txt_name), 'w')
            annos = v["regions"]        
            for anno in annos:
                region_attributes = anno["region_attributes"]
                if not region_attributes:
                    break
                anno = anno["shape_attributes"]
                if anno["name"] != "polygon":
                    break
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                if int(region_attributes["category_id"]):
                    top = np.min(px)
                    left = np.min(py)
                    bottom = np.max(px)
                    right = np.max(py)                
                    xc = ((top+bottom)/2)/im_width
                    yc = ((left+right)/2)/im_height
                    width = (bottom-top)/im_width
                    height = (right-left)/im_height
                    '''
                    cx = (top+bottom)/2
                    cy = (left+right)/2
                    cv2.circle(im, (int(cx),int(cy)), 10, (0,255,0))
                    cv2.circle(im, (top, left), 10, (255,0,0))
                    cv2.circle(im, (bottom, right), 10, (0,0,255))
                    #cv2.rectangle(im, (top, left), (bottom, right), (255, 0, 0))
                    print(top, left, bottom, right, width, height, im_width, im_height)
                    #cv2.resize(im, (960, 720))
                    cv2.imshow(im_name, im)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
                    line = [0, xc, yc, width, height]
                    out_txt.write(" ".join(str(item) for item in line) + "\n")
           
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
