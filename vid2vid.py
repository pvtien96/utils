#save all 5 6 7 8 actions-related video from MicaEgohands datasets
#python3 vid2vid.py --root-path /home/minhkv/EgoCentric_Nafosted/DHY_dataset/ --gestures 5 6 7 8 --readme /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/tienpv13/README.md
import argparse
import os
import cv2
from tqdm import tqdm
def get_parser():
    parser = argparse.ArgumentParser(description="save actions-related video")
    parser.add_argument("--root-path", help="Root path")
    parser.add_argument("--gestures", nargs='+', default=['5', '6', '7', '8'], help="Gestures list")
    parser.add_argument("--readme", help="Path to readme")
    return parser

def process(root, mp4_file, gestures, readme):
    mp4_file_path = os.path.join(root, mp4_file)
    txt_file_path = mp4_file_path.replace('MP4', 'txt')
    txt = open(txt_file_path, 'r')
    while True:
        line = txt.readline()
        if not line:
            break
        if len(line) > 4:
            gesture = line.split(';')[1]
            if gesture in gestures:
                start_frame = line.split(';')[2]
                end_frame = line.split(';')[3]
                save_vid(root, mp4_file, gesture, start_frame, end_frame, readme)
    txt.close()
    return 0

def save_vid(root, mp4_file, gesture, start_frame, end_frame, readme):
    output_file_name = mp4_file.split('.')[0] + '_' + gesture + '_' + start_frame + '_' + end_frame + '.avi'
    output_file_path = os.path.join(root, output_file_name)
    print (output_file_path)
    readme.write(output_file_path + "\n")
    video = cv2.VideoCapture(os.path.join(root, mp4_file))
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = int(end_frame) - int(start_frame)
    output_file = cv2.VideoWriter(
            filename=output_file_path,
            fourcc=cv2.VideoWriter_fourcc(*"XVID"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
    )
    
    video.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame)-1)
    for i in tqdm(range(num_frames)):
        ret, frame = video.read()
        output_file.write(frame)
    return 0

def main():
    args = get_parser().parse_args()
    readme = open(args.readme, "w+")
    for root, dirs, files in os.walk(args.root_path):
        for mp4_file in files:
            if mp4_file.endswith(".MP4"):
                process(root, mp4_file, args.gestures, readme)
    readme.close()
    return 0
if __name__ == "__main__":
    main()
