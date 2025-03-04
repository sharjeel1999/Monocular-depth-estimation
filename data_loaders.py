import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import os
import skimage.transform
from collections import Counter

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

class Mult(object):
    def __init__(self):
        self.mult_fact = 255
    def __call__(self, x):
        return x * self.mult_fact

class Appolo_dataset():
    def __init__(self):
        
        self._scale = 1.0
        self.height = 192
        self.width = 640
        self.aff_r = 5
        
        self._data_config = {}
        self._data_config['image_size_raw'] = [2710, 3384]
        # when need to rescale image due to large data
        self._data_config['image_size'] = [int(2710 * self._scale),
                                           int(3384 * self._scale)]
        # fx, fy, cx, cy
        self._data_config['intrinsic'] = {
            'Camera_5': np.array(
                [2304.54786556982, 2305.875668062,
                 1686.23787612802, 1354.98486439791]),
            'Camera_6': np.array(
                [2300.39065314361, 2301.31478860597,
                 1713.21615190657, 1342.91100799715])}

        # normalized intrinsic for handling image resizing
        cam_names = self._data_config['intrinsic'].keys()
        for c_name in cam_names:
            self._data_config['intrinsic'][c_name][[0, 2]] /= \
                self._data_config['image_size_raw'][1]
            self._data_config['intrinsic'][c_name][[1, 3]] /= \
                self._data_config['image_size_raw'][0]
                
        self.transform = transforms.ToTensor()
        
        # def mult(x):
        #     return x * 255
        
        # to_bgr_transform = transforms.Lambda(mult)
        normalize_transform = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        )
        
        self.transform_rcnn = transforms.Compose([
                transforms.ToTensor(),
                # Mult(),
                # normalize_transform
            ])
        
    def norm_image(self, in_image):
        in_image = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))
        return in_image
        
    def intrinsic_vec_to_mat(self, intrinsic, shape=None):
        """Convert a 4 dim intrinsic vector to a 3x3 intrinsic
           matrix
        """
        # print('shape: ', shape)
        if shape is None:
            shape = [1, 1]
    
        K = np.zeros((3, 3), dtype=np.float32)
        # print(intrinsic[0], shape[1])
        K[0, 0] = intrinsic[0] * shape[1]
        K[1, 1] = intrinsic[1] * shape[0]
        K[0, 2] = intrinsic[2] * shape[1]
        K[1, 2] = intrinsic[3] * shape[0]
        K[2, 2] = 1.0
    
        return K
    
    def rescale(self, image, intrinsic):
        """resize the image and intrinsic given a relative scale
        """
        intrinsic_out = self.intrinsic_vec_to_mat(intrinsic, [self.height, self.width])
        hs, ws = self.height, self.width
        image_out = cv2.resize(image.copy(), (ws, hs))

        return image_out, intrinsic_out
    
    def find_camera(self, image_path):
        
        for name in self._data_config['intrinsic'].keys():
            if name in image_path:
                return name
            # else:
        print('Name: ', name)
        print(image_path)
        raise ValueError('Camera not found')
    
    
    def calc_affinity(self, img_t_aff):
        # print('seg mask shape: ', img_t_aff.shape)
        # pytorch format -> [batch, channels, height, width]
        
        out_t_aff = torch.zeros((self.aff_r, self.aff_r**2,
                                 self.height, self.width))
        

        for mul in range(5):
            img_t_aff_mul = img_t_aff[0:self.height:2**mul,
                                      0:self.width:2**mul]
            img_height = self.height // (2**mul)
            img_width = self.width // (2**mul)

            
            img_t_aff_mul_2_pix = np.zeros((img_height
                                            + (self.aff_r//2)*2,
                                            img_width
                                            + (self.aff_r//2)*2, 3))
            img_t_aff_mul_2_pix[self.aff_r//2:
                                img_height+self.aff_r//2,
                                self.aff_r//2:
                                img_width+self.aff_r//2] \
                = img_t_aff_mul

            img_t_aff_compare = np.zeros((self.aff_r**2,
                                         img_height, img_width, 3))
            
            for i in range(self.aff_r):
                for j in range(self.aff_r):
                    img_t_aff_compare[i*self.aff_r+j] \
                        = img_t_aff_mul_2_pix[i:i+img_height,
                                              j:j+img_width]

            
            aff_data = np.where((img_t_aff_compare[:, :, :, 0]
                                 == img_t_aff_mul[:, :, 0])
                                & (img_t_aff_compare[:, :, :, 1]
                                   == img_t_aff_mul[:, :, 1])
                                & (img_t_aff_compare[:, :, :, 2]
                                   == img_t_aff_mul[:, :, 2]), 1, 0)
            
            aff_data = self.transform(aff_data.transpose(1, 2, 0))
            out_t_aff[mul, :, 0:img_height, 0:img_width] = aff_data
            
        return out_t_aff
    
    def prepare_sample(self, array, index):
        image_prev, image_current, image_next, mask_prev, mask_current, mask_next = array[index]
        
        camera = self.find_camera(image_current)
        intrinsic_or = self._data_config['intrinsic'][camera]
        # print('Intrinsic: ', intrinsic_or)
        image_current = image_current.replace('E:\Datasets', r'C:\Users\Sharjeel\Desktop')
        image_next = image_next.replace('E:\Datasets', r'C:\Users\Sharjeel\Desktop')
        image_prev = image_prev.replace('E:\Datasets', r'C:\Users\Sharjeel\Desktop')
        # print(image_current)
        # print(image_next)
        # print(image_prev)
        
        image_m1 = cv2.imread(image_prev)
        image_c = cv2.imread(image_current)
        image_p1 = cv2.imread(image_next)
        
        mask_m1 = cv2.imread(mask_prev)
        mask_c = cv2.imread(mask_current)
        mask_p1 = cv2.imread(mask_next)
        # print('seg mask shape: ', mask_c.shape)
        # print(np.unique(mask_c))
        
        image_m1, _ = self.rescale(image_m1, intrinsic_or) #cv2.resize(image_m1, (640, 192))
        image, intrinsic = self.rescale(image_c, intrinsic_or) #cv2.resize(image_c, (640, 192))
        image_p1, _ = self.rescale(image_p1, intrinsic_or) #cv2.resize(image_p1, (640, 192))
        
        k = np.zeros((4,4))
        k[:3, :3] = intrinsic
        k[3, 3] = 1
        
        # print('Intrinsic: ', k)
        # print('Data type: ', torch.from_numpy(k).dtype)
        
        mask_m1 = cv2.resize(mask_m1, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
        mask_c = cv2.resize(mask_c, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
        mask_p1 = cv2.resize(mask_p1, (self.width, self.height), interpolation = cv2.INTER_NEAREST)
        
        inputs = {}
        inputs[('image_aug', 0)] = self.transform_rcnn(self.norm_image(image))
        inputs[('image_aug', -1)] = self.transform_rcnn(self.norm_image(image_m1))
        inputs[('image_aug', 1)] = self.transform_rcnn(self.norm_image(image_p1))
        
        inputs[('image', 0)] = self.transform_rcnn(self.norm_image(image))
        inputs[('image', -1)] = self.transform_rcnn(self.norm_image(image_m1))
        inputs[('image', 1)] = self.transform_rcnn(self.norm_image(image_p1))
        
        # print('mask shapes: ', mask_m1.shape, mask_c.shape, mask_p1.shape)
        affinity_m1 = self.calc_affinity(mask_m1)
        # print('affinity shape: ', affinity_m1.shape)
        
        inputs[('affinity', 0)] = self.calc_affinity(mask_c)
        inputs[('affinity', -1)] = self.calc_affinity(mask_m1)
        inputs[('affinity', 1)] = self.calc_affinity(mask_p1)
        inputs[('segmentation', 0)] = mask_c[:, :, 0]
        
        inv_k = np.linalg.pinv(k)
        
        inputs['K'] = torch.from_numpy(k.astype(np.float32))
        inputs['inv_K'] = torch.from_numpy(inv_k.astype(np.float32))
        
        
        return inputs
        

class Kitti_dataset():
    def __init__(self):
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        
        self.transform = transforms.ToTensor()
        
        # def mult(x):
        #     return x * 255
        
        # to_bgr_transform = transforms.Lambda(mult)
        normalize_transform = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        )
        
        self.transform_rcnn = transforms.Compose([
                transforms.ToTensor(),
                # Mult(),
                # normalize_transform
            ])
        
        self.height = 192
        self.width = 640
        
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def norm_image(self, in_image):
        in_image = cv2.GaussianBlur(in_image, (5,5), cv2.BORDER_DEFAULT)
        in_image = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))
        return in_image
    
    def get_depth(self, velo_filename, calib_path, side, do_flip):

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    def preprocess(self, inputs, color_aug):
        for k in inputs:
            print(k)
            
    def prepare_sample(self, array, index):
        m1, c, p1 = array[index]
        image_path_m1, velo_m1, side_m1, cal_m1 = m1
        image_path, velo, side, cal = c
        image_path_p1, velo_p1, side_p1, cal_p1 = p1
        
        image_m1 = cv2.imread(image_path_m1)
        image = cv2.imread(image_path)
        image_p1 = cv2.imread(image_path_p1)
        
        image_m1 = cv2.resize(image_m1, (640, 192))
        image = cv2.resize(image, (640, 192))
        image_p1 = cv2.resize(image_p1, (640, 192))
        
        inputs = {}
        inputs[('image_aug', 0)] = self.transform_rcnn(self.norm_image(image))
        inputs[('image_aug', -1)] = self.transform_rcnn(self.norm_image(image_m1))
        inputs[('image_aug', 1)] = self.transform_rcnn(self.norm_image(image_p1))
        
        depth_gt = self.get_depth(velo, cal, side, False)
        depth_gt = np.expand_dims(depth_gt, 0)
        depth_gt = torch.from_numpy(depth_gt.astype(np.float32))
        
        # image_m1 = cv2.resize(image_m1, (640, 192))
        # image = cv2.resize(image, (640, 192))
        # image_p1 = cv2.resize(image_p1, (640, 192))
        inputs[('image', 0)] = self.transform_rcnn(self.norm_image(image))
        inputs[('image', -1)] = self.transform_rcnn(self.norm_image(image_m1))
        inputs[('image', 1)] = self.transform_rcnn(self.norm_image(image_p1))
        
        k = self.K.copy()

        k[0, :] *= self.width // (2 ** 1)
        k[1, :] *= self.height // (2 ** 1)
        
        # print(k)
        # print('Data type: ', torch.from_numpy(k).dtype)
        
        inv_k = np.linalg.pinv(k)
        
        inputs['K'] = torch.from_numpy(k)
        inputs['inv_K'] = torch.from_numpy(inv_k)
        
        inputs['depth_gt'] = depth_gt
        return inputs

class Prepare_dataset(Dataset):
    def __init__(self, data_path, seg_data_path, appolo):
        self.appolo = appolo
        
        if appolo == True:
            if isinstance(seg_data_path, str):
                self.data = np.load(seg_data_path, allow_pickle = True)
            else:
                self.data = seg_data_path
        
        else:
            if isinstance(data_path, str):
                self.data = np.load(data_path, allow_pickle = True)
            else:
                self.data = data_path
        
        self.appolo_processor = Appolo_dataset()
        self.kitti_processor = Kitti_dataset()
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        
        # reshape images to (256, 1024)
        # image_path_m1, image_path, image_path_p1, mask_path_m1, mask_path, mask_path_p1 = self.data[index]
        if self.appolo == True:
            inputs = self.appolo_processor.prepare_sample(self.data, index)
        else:
            inputs = self.kitti_processor.prepare_sample(self.data, index)
        
        
        return inputs
    

    
class Prepare_dataset_test(Dataset):
    def __init__(self, data_path):
        if isinstance(data_path, str):
            self.data = np.load(data_path, allow_pickle = True)
        else:
            self.data = data_path
        
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        
        self.transform = transforms.ToTensor()
        
        # def mult(x):
        #     return x * 255
        
        # to_bgr_transform = transforms.Lambda(mult)
        
        # normalize_transform = transforms.Normalize(
        #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        # )
        
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.227, 0.224, 0.225]
        )
        
        self.transform_rcnn = transforms.Compose([
                transforms.ToTensor(),
                # Mult(),
                # normalize_transform
            ])
        
        self.height = 192
        self.width = 640
        
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def norm_image(self, in_image):
        in_image = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))
        return in_image
    
    def get_depth(self, velo_filename, calib_path, side, do_flip):

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    def preprocess(self, inputs, color_aug):
        for k in inputs:
            print(k)
    
    def __getitem__(self, index):
        
        
        # reshape images to (256, 1024)
        # image_path_m1, image_path, image_path_p1, mask_path_m1, mask_path, mask_path_p1 = self.data[index]
        m1, c, p1 = self.data[index]
        image_path_m1, velo_m1, side_m1, cal_m1 = m1
        image_path, velo, side, cal = c
        image_path_p1, velo_p1, side_p1, cal_p1 = p1
        # print('path 1: ', image_path_m1)
        # print('path 2: ', image_path)
        # print('path 3: ', image_path_p1)
        
        image_m1 = cv2.imread(image_path_m1)
        image = cv2.imread(image_path)
        image_p1 = cv2.imread(image_path_p1)
        # print('shapes before: ', image.shape, image_m1.shape, image_p1.shape)
        
        image_m1 = cv2.resize(image_m1, (640, 192))
        image = cv2.resize(image, (640, 192))
        image_p1 = cv2.resize(image_p1, (640, 192))
        # print('image size: ', image.shape)
        
        inputs = {}
        # image_m1 = cv2.resize(image_m1, (1280, 384))
        # image = cv2.resize(image, (1280, 384))
        # image_p1 = cv2.resize(image_p1, (1280, 384))
        inputs[('original_image_path', 0)] = image_path
        inputs[('index', 0)] = index
        
        inputs[('image_aug', 0)] = self.transform_rcnn(self.norm_image(image))
        inputs[('image_aug', -1)] = self.transform_rcnn(self.norm_image(image_m1))
        inputs[('image_aug', 1)] = self.transform_rcnn(self.norm_image(image_p1))
        
        depth_gt = self.get_depth(velo, cal, side, False)
        depth_gt = np.expand_dims(depth_gt, 0)
        depth_gt = torch.from_numpy(depth_gt.astype(np.float32))
        # print('Depth shape: ', depth_gt.shape)
        
        # mask_load = cv2.imread(mask_path)
        # mask_orig = mask_load[:, :, 0]
        # #print('mask load shape: ', mask_orig.shape)
        # mask = cv2.resize(mask_orig, (1242, 375))
        # mask2 = cv2.resize(mask_orig, (1242, 375))
        
        #inputs['image_m1'] = self.norm_image(image_m1)
        #inputs['image'] = self.norm_image(image)
        #inputs['image_p1'] = self.norm_image(image_p1)
        #print('shapes after: ', image.shape, image_m1.shape, image_p1.shape)
        image_m1 = cv2.resize(image_m1, (640, 192))
        image = cv2.resize(image, (640, 192))
        image_p1 = cv2.resize(image_p1, (640, 192))
        inputs[('image', 0)] = self.transform_rcnn(self.norm_image(image))
        inputs[('image', -1)] = self.transform_rcnn(self.norm_image(image_m1))
        inputs[('image', 1)] = self.transform_rcnn(self.norm_image(image_p1))
        
        k = self.K.copy()
        
        k[0, :] *= self.width // (2 ** 1)
        k[1, :] *= self.height // (2 ** 1)
        
        inv_k = np.linalg.pinv(k)
        
        inputs['K'] = torch.from_numpy(k)
        inputs['inv_K'] = torch.from_numpy(inv_k)
        
        # mask2 = self.transform(mask2)
        # mask = self.transform(mask)
        inputs['depth_gt'] = depth_gt
        # inputs['disp'] = 1/mask
        
        return inputs