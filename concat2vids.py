import cv2
import numpy as np
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="Concatenate 2 videos in to 1 horizontal video")
    parser.add_argument(
            "--video1",
            type=str,
            help="Path to first video path, eg. ./video1.avi",
    )
    parser.add_argument(
            "--video2",
            type=str,
            help="Path to second video path, eg. ./video2.avi",
    )
    parser.add_argument(
            "--output",
            type=str,
            help="Path to output video path, eg. ./output.avi",
    )

    return parser
def main():
    args = get_parser().parse_args()
    vid1 = cv2.VideoCapture(args.video1)
    vid2 = cv2.VideoCapture(args.video2)
    width = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = min(int(vid1.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid2.get(cv2.CAP_PROP_FRAME_COUNT)))
    output = cv2.VideoWriter(
        filename=args.output,
        fourcc=cv2.VideoWriter_fourcc(*"XVID"),
        fps=1,
        frameSize=(width*2, height),
        isColor=True,
    )
    for i in tqdm(range(num_frames)):
        hasFrame1, frame1 = vid1.read()
        hasFrame2, frame2 = vid2.read()
        if hasFrame1 and hasFrame2:
            frame = np.concatenate((frame1, frame2), axis=1)
            output.write(frame)
    output.release()
    return 0

if __name__ == '__main__':
    main()
