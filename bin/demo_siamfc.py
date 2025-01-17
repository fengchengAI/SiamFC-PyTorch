import glob
import os
import pandas as pd
import cv2
import sys
sys.path.append(os.getcwd())
from fire import Fire
from siamfc.tracker import SiamFCTracker

def main(video_dir, gpu_id,  model_path):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')

    title = video_dir.split('/')[-1]
    # starting tracking
    tracker = SiamFCTracker(model_path, gpu_id)
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
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
        gt_bbox = gt_bboxes.iloc[idx].values
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

if __name__ == "__main__":
    Fire(main)
