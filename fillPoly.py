import cv2
import numpy as np
import argparse
import os
import json

def get_parser():
    parser = argparse.ArgumentParser(description="Convert polygon format to binary-mask format")
    parser.add_argument(
        "--path",
        type=str,
        default="/media/data3/EgoCentric_Nafosted/non_skip/train/",
        help="Path to root folder that contains subfolder",
    )

    return parser
def fillPoly(polys):
    mask=np.zeros([1920, 1440], dtype=np.uint8)
    cv2.fillPoly(mask, polys, 255) 
    return mask 

def main():
    args = get_parser().parse_args()
    directories = os.listdir(args.path)
    for directory in directories:
        print(directory)
        sub_path = os.path.join(args.path, directory)
        '''
        out_video = cv2.VideoWriter(
            filename=os.path.join(args.path, str(directory)+'.avi'),
            fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            fps=10.0,
            frameSize=(1920, 1440)
        )
        '''
        json_file = os.path.join(sub_path, 'via_export_json.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)
        for idx, v in enumerate(imgs_anns.values()):
            filename = os.path.join(sub_path, v["filename"])
            annos = v["regions"]
            polys = []
            for anno in annos:
                region_attributes = anno["region_attributes"]
                if not region_attributes:
                    break
                anno = anno["shape_attributes"]
                if anno["name"] != "polygon":
                    break
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [[x,y] for x, y in zip(px, py)]
                polys.append(poly)
            polys = np.array(polys, dtype=np.int32)
            mask = fillPoly(polys)
            cv2.imwrite(filename, mask)
            #out_video.write(mask)
        #out_video.release()
    return 0
if __name__ == '__main__':
    main()
