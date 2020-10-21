import argparse
import os
import glob
import cv2
from tqdm import tqdm
import numpy as np
import json

def get_parser():
    parser = argparse.ArgumentParser(description="Preparing MOT groundtruth as MOT16's format from detection groundtruth labelled by via tool")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/media/data3/EgoCentric_Nafosted/micand30/gt/",
        help="Path to groundtruth folder that contains SEQUENCE_X with via_export_json.json",
    )
    parser.add_argument(
        "--seq_name_path",
        type=str,
        default="/media/data3/EgoCentric_Nafosted/micand30/seqmaps/Sequence.txt",
        help="Sequence.txt write all sequences name, this helps pymot reconginze seqs",
    )
    return parser

def make_seqname(seqname_file, directories):
    f = open(seqname_file, "w+")
    f.write("name"+"\n")
    f.write("\n".join(directory for directory in directories))
    return 0
def make_seqinfo(path2directory, directory):
    seqinfo = os.path.join(path2directory, "seqinfo.ini")
    f = open(seqinfo, "w+")
    f.write('[Sequence]'+"\n")
    f.write('name='+directory+"\n")
    seqLength = len(glob.glob(path2directory+'/*.png'))
    f.write('seqLength='+str(seqLength))
    return 0

def make_gt(path2gt_of_seq, json_file):
    gt_txt = os.path.join(path2gt_of_seq, "gt.txt")
    f = open(gt_txt, "w+")
    with open(json_file) as fjson:
        imgs_anns = json.load(fjson)
    for idx, v in enumerate(imgs_anns.values()):
        #print(v["filename"])
        annos = v["regions"]
        for anno in annos:
            region_attributes = anno["region_attributes"]
            trackID = int(region_attributes["category_id"])
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            top = np.min(px)
            left = np.min(py)
            width = np.max(px)-np.min(px)
            height = np.max(py)-np.min(py)
            line = [idx+1, trackID, top, left, width, height, 1, 1, 1]
            f.write(",".join(str(item) for item in line) + "\n")
    return 0

def main():
    args = get_parser().parse_args()
    directories = os.listdir(args.root_path)
    #write seq_name to Sequences.txt
    make_seqname(args.seq_name_path, directories)
    for directory in directories:
        print(directory)
        path2directory = os.path.join(args.root_path, directory)
        
        #extract frames
        vid = cv2.VideoCapture(os.path.join(path2directory, directory+'.avi'))
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))    
        for i in range(num_frames):
            hasFrame, frame = vid.read()
            if hasFrame:
                name_frame = str(i).zfill(len(str(num_frames))) + ".png"
                path_frame = os.path.join(path2directory, name_frame)
                cv2.imwrite(path_frame, frame)

        #make seqinfo.ini
        make_seqinfo(path2directory, directory)

        #make gt.txt
        path2gt_of_seq = os.path.join(path2directory, 'gt')
        if not os.path.isdir(path2gt_of_seq):
            os.mkdir(path2gt_of_seq)
        json_file = os.path.join(path2directory, "via_export_json.json")
        make_gt(path2gt_of_seq, json_file)

    return 0

if __name__=='__main__':
    main()
