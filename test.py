import os
def main():
    vids_name = os.listdir('/media/data3/EgoCentric_Nafosted/full')
    readme = open('/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/tienpv13/README.txt', 'r+')
    vids_in_micand = open('/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/tienpv13/vids_in_micand.txt', 'a')
    line = True
    while line:
        line = readme.readline().rstrip()
        #command = "python3 ../demo/EgoHands/demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --video-input " + line + " --output ../projects/EgoHands/output/COCO-InstanceSegmentation-maskrcnn_R_50_FPN_1x/demo/EgoCentricNafosted/ --opts MODEL.WEIGHTS ../projects/EgoHands/output/COCO-InstanceSegmentation-maskrcnn_R_50_FPN_1x/training/model_final.pth MODEL.ROI_HEADS.NUM_CLASSES 4"
        #print(command)
        #os.system(command)
        for vid in vids_name:
            if vid in line:
                vids_in_micand.write(line)
                vids_in_micand.write('\n')
                print(line)
    readme.close()
    return 0
if __name__ == "__main__":
    main()
