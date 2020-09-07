#python3 im2vid.py --root_path /media/data3/EgoCentric_Nafosted/non_skip/train/
import os
import glob
import cv2
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="im2vid")
    parser.add_argument("--root_path", type=str, default="/media/data3/EgoCentric_Nafosted/non_skip/train/", help="Root path")
    return parser

def main():
    args = get_parser().parse_args()
    directories = os.listdir(args.root_path)
    for directory in tqdm(directories):
        out_filename = os.path.join(args.root_path, directory) + '/' + str(directory) + '.avi'
        print(out_filename)
        output_file = cv2.VideoWriter(
            filename=out_filename,
            fourcc=cv2.VideoWriter_fourcc(*"XVID"),
            fps=1,
            frameSize=(1920, 1440),
            isColor=True,
        )
 
        for filename in sorted(glob.glob(os.path.join(args.root_path, directory) + '/*.png')):
            im = cv2.imread(filename)
            output_file.write(im)
        output_file.release()
    return 0

if __name__ == "__main__":
    main()
