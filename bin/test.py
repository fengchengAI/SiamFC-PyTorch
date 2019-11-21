import glob
import os
import cv2
import sys
sys.path.append(os.getcwd())
from siamfc.tracker import SiamFCTracker
import xml.etree.ElementTree as ET


video_dir = '/home/feng/Github/PyCharmProjects/SiamFC-PyTorch/data'
video_dir = '/media/feng/PC/data/ILSVRC2015_VID_initial/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00001004'
model_path = '/home/feng/Github/PyCharmProjects/SiamFC-PyTorch/models/siamfc_pretrained.pth'
image_name = sorted(glob.glob(os.path.join(video_dir, "*")),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
# width height xmax xmin ymax ymin
anno_name = [x.replace('Data', 'Annotations') for x in image_name]
anno_name = [x.replace('JPEG', 'xml') for x in anno_name]
gtbox = []
for i in range(len(anno_name)):
        tree = ET.parse(anno_name[i])
        root = tree.getroot()
        bboxes = []
        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')

            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            '''
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text - bbox.find('xmin').text,
                                  bbox.find('ymax').text- bbox.find('ymin').text]))
            '''
            bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
        gtbox.append(bbox)


gpu_id = 0

frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in image_name]
gt_bboxes = gtbox
title = video_dir.split('/')[-1]
# starting tracking
tracker = SiamFCTracker(model_path, gpu_id)
for idx, frame in enumerate(frames):
    if idx == 0:
        bbox = gt_bboxes[0]
        tracker.init(frame, bbox)
        bbox = (bbox[0]-1, bbox[1]-1,
                bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
    else:
        bbox = tracker.update(frame)
    # bbox  x['xmin','ymin','xmax','ymax']
    frame = cv2.rectangle(frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 255, 0),
                          2)
    gt_bbox = gt_bboxes[idx]
    gt_bbox = (gt_bbox[0], gt_bbox[1],
               gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3])
    frame = cv2.rectangle(frame,
                          (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                          (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                          (255, 0, 0),
                          1)
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    cv2.imshow(title, frame)
    cv2.waitKey(30)
