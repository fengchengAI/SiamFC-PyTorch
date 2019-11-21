import numpy as np
import cv2

def get_center(x):
    return (x - 1.) / 2.

def xyxy2cxcywh(bbox):
    return get_center(bbox[0]+bbox[2]), \
           get_center(bbox[1]+bbox[3]), \
           (bbox[2]-bbox[0]), \
           (bbox[3]-bbox[1])

def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    '''
    首先会根据box的中心点位置，和original_sz信息，将img裁剪到original_sz大小，对于裁剪不够的补参数img_mean
    然后在将图片缩放到model_sz尺寸
    :param img: 要裁剪的图片
    :param cx:
    :param cy:
    :param model_sz: 　要输出的大小
    :param original_sz: 　原始大小（此处并不是真正的原始，是一种映射，大致）
    :param img_mean:
    :return:
    '''
    xmin = cx - original_sz // 2
    xmax = cx + original_sz // 2
    ymin = cy - original_sz // 2
    ymax = cy + original_sz // 2
    im_h, im_w, _ = img.shape

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]
    if left != 0 or right !=0 or top!=0 or bottom!=0:
        if img_mean is None:
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=img_mean)
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch

def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    # size_z:127    context_amount:0.5    bbox:['xmin','ymin','xmax''ymax']
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)  # 模型大小
    scale_z = size_z / s_z  # 指的是面积的缩放，由127缩放到s_z时的比例,一般size_z远小于s_z
    exemplar_img = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    # 首先会在以cx, cy为中心的图片中，裁剪出s_z大小的图片，然后再缩放到127
    return exemplar_img, scale_z, s_z

def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    '''只会在训练时对每个图像进行这样的处理
    :param img: 从数据集图片中读取的原始图像
    :param bbox: ['xmin','ymin','xmax','ymax']
    :param size_z: 127
    :param size_x: 255
    :param context_amount: 0.5
    :param img_mean:
    :return:
    '''
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    '''这里有点冗余，只是方便理解
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    '''
    s_x = size_x/scale_z  # 修改后为
    scale_x = size_x / s_x  # scale_x＝＝scale_z两者缩放比例一样，
    instance_img = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    # 首先会在以cx, cy为中心的图片中，裁剪出s_x大小的图片，然后再缩放到255
    return instance_img, scale_x, s_x

def get_pyramid_instance_image(img, center, size_x, size_x_scales, img_mean=None):
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, center[0], center[1], size_x, size_x_scale, img_mean)
            for size_x_scale in size_x_scales]
    return pyramid
