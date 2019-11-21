import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
sys.path.append(os.getcwd())  # os.getcwd()当前路径
from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

from siamfc.config import config
from siamfc.utils import get_instance_image

def worker(output_dir, video_dir):
    # video_dir是图片的上一级文件，即video_dir文件夹下有很多帧图像
    image_names = glob(os.path.join(video_dir, '*.JPEG'))
    image_names = sorted(image_names,
                        key=lambda x:int(x.split('/')[-1].split('.')[0]))
    '''
    key=lambda x:int(x.split('/')[-1].split('.')[0])是对文件名称进行排序
    如image_names中的一个image_name
    image_name: '/home/ss/fengcheng/data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/000000.JPEG'
    image_name.split('/'): ['', 'home', 'ss', 'fengcheng', 'data', 'ILSVRC2015', 'Data', 'VID', 'train', 'ILSVRC2015_VID_train_0000', 'ILSVRC2015_train_00000000', '000000.JPEG']
    再取[-1]为'000000.JPEG'即为文件名
    再split('．')为['000000','JPEG']
    再取[０]　为'000000'文件名
    '''
    video_name = video_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    '''
    新建一个文件夹然后在文件夹下新建一个与原始图像的上级文件夹路径名称一个的文件
    用于存放预处理过的图像信息
    '''
    trajs = {}  # 用来做什么？？
    '''
    应该是针对一个视频中的多个多个跟踪目标，其中每个所要跟踪的目标有个ID 
    '''
    for image_name in image_names:
        img = cv2.imread(image_name)
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        anno_name = image_name.replace('Data', 'Annotations')
        anno_name = anno_name.replace('JPEG', 'xml')  # replace(0ld,new)
        tree = ET.parse(anno_name)
        root = tree.getroot()
        bboxes = []
        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]
            instance_img, _, _ = get_instance_image(img, bbox,
                    config.exemplar_size, config.instance_size, config.context_amount, img_mean)
            instance_img_name = os.path.join(save_folder, filename+".{:02d}.x.jpg".format(trkid))
            cv2.imwrite(instance_img_name, instance_img)

    return video_name, trajs

def processing(data_dir, output_dir, num_threads=8):
    # get all 4417 videos

    video_dir = os.path.join(data_dir, 'Data/VID')
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))

    '''
    all_videos = glob(os.path.join(video_dir, 'train','*','*'))
    all_videos的最后一级是包含图片的文件夹，即图片的上一级
    '''
    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            #　functools.partial(worker, output_dir)
            # 因为pool.imap_unordered只能接收一个参数，故需要固定一个参数．
            meta_data.append(ret)

    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))
    # meta_data.pkl保存的是video_name和trajs


if __name__ == '__main__':
    Fire(processing)
    # 在命令行执行时如下操作：
    # python3 creat_dataset.py --data_dir *** --output_dir *** --num_threads **
    # 其中data_dir和output_dir后参数必有


