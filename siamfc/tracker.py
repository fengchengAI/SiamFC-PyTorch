import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from .alexnet import SiameseAlexNet
from .config import config
from .custom_transforms import ToTensor
from .utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image

torch.set_num_threads(1) # otherwise pytorch will take all cpus

class SiamFCTracker:
    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id):
            self.model = SiameseAlexNet(gpu_id, train=False)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
            self.model.eval() 
        self.transforms = transforms.s_xCompose([
            ToTensor()
        ])

    def _cosine_window(self, size):
        """
            size = (17*16, 17*16)
            get the cosine window
            生成一个size[0]*size[1]的一个类似于距离图,即最中间的最大,最边缘的最小,
            np.sum(cos_window)是整个矩阵的和,
            除以,会有权重的意思
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [minx, miny, width, height]
        """
        self.bbox = (bbox[0] - 1, bbox[1] - 1, bbox[0] - 1 + bbox[2], bbox[1] - 1 + bbox[3])
        # self.bbox ['xmin','ymin','xmax''ymax']
        self.pos = np.array([bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2])
        self.target_sz = np.array([bbox[2], bbox[3]])
        # pos和target_sz会在下一帧的图像中改变

        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                config.exemplar_size, config.context_amount, self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None,:,:,:]  # 这里的transforms只是单纯的totensor
        with torch.cuda.device(self.gpu_id):
            exemplar_img_var = Variable(exemplar_img.cuda())
            self.model((exemplar_img_var, None))

        self.penalty = np.ones((config.num_scale)) * config.scale_penalty  # config.scale_penalty:0.9745   num_scale:3
        self.penalty[config.num_scale//2] = 1
        # penalty [0.9745,1,0.9745] 主要是对第n+1帧出现的，三个不同刻度的scales的一个惩罚权重信息

        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz  # 16*17
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))
        # 会返回一个中心大周围小的(16*17, 16*17)的矩阵，每个值都是归一化的值

        # create scalse
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                np.floor(config.num_scale/2)+1)
        '''
        np.ceil向上取整         np.floor向下取整
        scale_step = 1.0375    num_scale = 3 
        self.scales = 1.0375^[-1,0,1]
        '''
        # create s_x
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z  ## self.s_x = config.instance_size/scale_z

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image
        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales
        #  self.scales =  config.scale_step ^ [-1,0,1]
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        instance_imgs = torch.cat([self.transforms(x)[None,:,:,:] for x in pyramid], dim=0)  # (3,3,255,255)
        with torch.cuda.device(self.gpu_id):
            instance_imgs_var = Variable(instance_imgs.cuda())
            response_maps = self.model((None, instance_imgs_var))
            response_maps = response_maps.data.cpu().numpy().squeeze()  #(3,17,17)
            response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
             for x in response_maps]## 主要是将x上采样至16*17
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty
        # penalty=[0.9745,1,0.9745]

        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()  ## 归一化，便可以与归一化的cosine_window相加
        response_map = (1 - config.window_influence) * response_map + \
                config.window_influence * self.cosine_window
        # config.window_influence = 0.176
        # 为什么加上cosine_window的权重？？？？
        # 根据经验，我们所要跟踪的视频往往会出现在屏幕中心，还有训练阶段的缘故
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        # 返回response_map最大值在response_map中的索引
        # response_map此时是多维数据，所以要用unravel_index,即横纵坐标

        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        # response的最大值的位置，距离interp_response_sz中心的距离
        # displacement in input
        disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride  ## 为什么进行这样的操作？？
        # total_stride = 8 ；response_up_stride= 16
        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input * (self.s_x * scale) / config.instance_size
        # position in frame coordinates
        self.pos += disp_response_frame  ## 为什么self.pos的更新不能直接用disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz
        bbox = (self.pos[0] - self.target_sz[0]/2 + 1, # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1]/2 + 1, # ymin
                self.pos[0] + self.target_sz[0]/2 + 1, # xmax
                self.pos[1] + self.target_sz[1]/2 + 1) # ymax
        return bbox
