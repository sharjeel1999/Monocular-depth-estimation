import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import networks
from maskrcnn_benchmark.config import cfg

from layers import *
from eval_utils import evaluate_depth

from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt

from PIL import Image 
import PIL 

import copy

# def plot_grad_flow(named_parameters):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
    
#     Usage: Plug this function in Trainer class after loss.backwards() as 
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#     ave_grads = []
#     max_grads= []
#     layers = []
#     print('len named params: ', len(named_parameters))
#     for z in named_parameters:
#         print('z: ', len(z))
#         for n, p in z:
#             # print('Parameter name: ', n)
#             if(p.requires_grad) and ("bias" not in n):
#                 layers.append(n)
#                 ave_grads.append(p.grad.abs().mean().cpu())
#                 max_grads.append(p.grad.abs().max().cpu())
    
#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend([Line2D([0], [0], color="c", lw=4),
#                 Line2D([0], [0], color="b", lw=4),
#                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

class conv_block(nn.Module):
    def __init__(self, in_channels):
        super(conv_block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        # self.norm = nn.BatchNorm2d(in_channels)
        
        self.act = nn.ReLU(inplace = True)
        
    def forward(self, x):
        # print('inside len: ', len(x), x.shape, torch.Tensor(x).shape)
        x = self.conv(x)
        d0, d1, d2, d3 = x.shape
        self.norm1 = nn.LayerNorm(d3).cuda()
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        d0, d1, d2, d3 = x.shape
        self.norm2 = nn.LayerNorm(d3).cuda()
        x = self.norm2(x)
        
        return self.act(x)

class Trainer:
    def __init__(self, options, train_loader, val_loader, save_folder, batch_size, use_affinity):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_iter = iter(self.val_loader)
        self.use_pose_net = True
        self.val_mode = False
        self.use_affinity = use_affinity
        self.batch_size = batch_size
        
        self.opt = options
        self.cfg = cfg
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.num_scales = 1
        
        self.models = {}
        self.parameters_to_train = []
        
        self.models['initial'] = networks.Initial_student()
        self.models['initial'].to(self.device)
        self.parameters_to_train += list(self.models['initial'].parameters())
        
        self.models['encoder'] = networks.ResnetEncoder_student(num_layers = 18, pretrained = False) # options [18, 34, 50, 101, 152]
        self.models['encoder'].to(self.device)
        # self.models['dummy_encoder'] = networks.ResnetEncoder_student(num_layers = 18, pretrained = False)
        # self.models['dummy_encoder'].to(self.device)
        self.parameters_to_train += list(self.models['encoder'].parameters())   ############################### un-comment when not using multi LR ####################
        
        self.models['buffer_m1'] = conv_block(256)
        self.models['buffer_m1'].to(self.device)
        self.parameters_to_train += list(self.models['buffer_m1'].parameters())
        
        self.models['buffer_m2'] = conv_block(256)
        self.models['buffer_m2'].to(self.device)
        self.parameters_to_train += list(self.models['buffer_m2'].parameters())
        
        self.models['buffer_m3'] = conv_block(256)
        self.models['buffer_m3'].to(self.device)
        self.parameters_to_train += list(self.models['buffer_m3'].parameters())
        
        self.models['initial_teacher'] = networks.Initial_teacher()
        self.models['initial_teacher'].to(self.device)
        self.parameters_to_train += list(self.models['initial_teacher'].parameters())
        
        self.models['encoder_teacher'] = networks.ResnetEncoder_Teacher(num_layers = 18, pretrained = False) # options [18, 34, 50, 101, 152]
        self.models['encoder_teacher'].to(self.device)
        # self.parameters_to_train += list(self.models['encoder_teacher'].parameters())
        
        
        
        self.models['depth'] = networks.DepthDecoder(scales = self.opt.scales)
        # self.models['depth'] = networks.DepthDecoder_SA(scales = self.opt.scales)
        self.models['depth'].to(self.device)
        self.parameters_to_train += list(self.models['depth'].parameters())
        
        self.models['depth_teacher'] = networks.DepthDecoder_SA(scales = self.opt.scales)
        self.models['depth_teacher'].to(self.device)
        # self.parameters_to_train += list(self.models['depth_teacher'].parameters())
        
        self.models['pose'] = networks.PoseDecoder(self.num_pose_frames)
        self.models['pose'].to(self.device)
        self.parameters_to_train += list(self.models['pose'].parameters())
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)#, weight_decay = 0.001)
        # self.model_optimizer = optim.SGD(self.parameters_to_train, self.opt.learning_rate, momentum=0.9)
        
        # Multi rate optimzier
        # self.model_optimizer = optim.Adam([
        #         {'params': self.parameters_to_train},
        #         {'params': self.models['encoder'].parameters(), 'lr':1e-6}
        #     ], 1e-5)
        
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, self.opt.scheduler_gamma)
        
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        self.backproject_depth = BackprojectDepth(batch_size, height = 192, width = 640)
        self.backproject_depth_1 = BackprojectDepth(batch_size, height = 24, width = 80)
        self.backproject_depth_2 = BackprojectDepth(batch_size, height = 12, width = 40)
        self.backproject_depth_3 = BackprojectDepth(batch_size, height = 6, width = 20)
        
        self.backproject_depth_test = BackprojectDepth(batch_size, height = 48, width = 160)
        self.project_3d_test = Project3D(batch_size, height = 48, width = 160)
        
        self.project_3d = Project3D(batch_size, height = 192, width = 640)
        self.project_3d_1 = Project3D(batch_size, height = 24, width = 80)
        self.project_3d_2 = Project3D(batch_size, height = 12, width = 40)
        self.project_3d_3 = Project3D(batch_size, height = 6, width = 20)
        self.ssim = SSIM()
        
        self.momentum_schedule = self.cosine_scheduler(0.998, 1, self.opt.num_epochs, len(self.train_loader))
        
        self.record_saves = 'D:\\depth_estimation_implementation\\all_saves\\record_saves'
        self.file_path = os.path.join(self.record_saves, 'finetuning_A8_first1.txt')
        
        self.save_folder = save_folder
        
    def cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule
        
    def set_train(self):
        self.models['initial'].train()
        self.models['encoder'].train()
        self.models['depth'].train()
        self.models['pose'].train()
        
        self.models['buffer_m1'].train()
        self.models['buffer_m2'].train()
        
        self.models['initial_teacher'].train()
        self.models['encoder_teacher'].eval()
        self.models['depth_teacher'].eval()
        
        # for m in self.models.values():
        #     m.train()
            
    def set_eval(self):
        self.models['initial'].eval()
        self.models['encoder'].eval()
        self.models['depth'].eval()
        self.models['pose'].eval()
        
        self.models['buffer_m1'].eval()
        self.models['buffer_m2'].eval()
        
        self.models['initial_teacher'].eval()
        
        # for m in self.models.values():
        #     m.eval()
            
    def load_model(self):
        
        ep = 23
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\mini_data_saves\\with_A2_using_3_channels_0.1'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A2'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights_APPOLO'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A3_oneLR' #best 1
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3_L1'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A5'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A7'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A8_original'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A8'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A9'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A9'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A8_first2'
        path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A8_first1'
        
        # encoder_name = 'Encoder.pth'
        # depth_name = 'Depth.pth'
        # pose_name = 'Pose.pth'
        # initial_name = 'Initial_encoder.pth'
        
        encoder_name = 'Encoder' + str(ep) + '.pth'
        depth_name = 'Depth' + str(ep) + '.pth'
        pose_name = 'Pose' + str(ep) + '.pth'
        initial_name = 'Initial_encoder' + str(ep) + '.pth'
        
        self.models['initial'].load_state_dict(torch.load(os.path.join(path, initial_name)))
        self.models['encoder'].load_state_dict(torch.load(os.path.join(path, encoder_name)), strict=False)
        self.models['depth'].load_state_dict(torch.load(os.path.join(path, depth_name)), strict=False)
        self.models['pose'].load_state_dict(torch.load(os.path.join(path, pose_name)), strict=False)
        
        # self.models['encoder_teacher'].load_state_dict(torch.load(os.path.join(path, encoder_name)), strict = False)
        
    
    def load_entire(self):
        ep = 9
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\mini_data_saves\\with_A2_using_3_channels_0.1'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A2'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights_APPOLO'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A3'
        path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3_0.2'
        
        # encoder_name = 'Encoder.pth'
        # depth_name = 'Depth.pth'
        # pose_name = 'Pose.pth'
        # initial_name = 'Initial_encoder.pth'
        
        encoder_name = 'Encoder' + str(ep) + '.pth'
        depth_name = 'Depth' + str(ep) + '.pth'
        pose_name = 'Pose' + str(ep) + '.pth'
        initial_name = 'Initial_encoder' + str(ep) + '.pth'
        
        self.models['initial'].load_state_dict(torch.load(os.path.join(path, initial_name)))
        self.models['encoder'].load_state_dict(torch.load(os.path.join(path, encoder_name)), strict=False)
        self.models['depth'].load_state_dict(torch.load(os.path.join(path, depth_name)), strict=False)
        self.models['pose'].load_state_dict(torch.load(os.path.join(path, pose_name)), strict=False)
        
        teacher_initial_name = 'Initial_encoder_teacher' + str(ep) + '.pth'
        teacher_encoder_name = 'Encoder_teacher' + str(ep) + '.pth'
        self.models['initial_teacher'].load_state_dict(torch.load(os.path.join(path, teacher_initial_name)))
        self.models['encoder_teacher'].load_state_dict(torch.load(os.path.join(path, teacher_encoder_name)))
    
    
    def save_model(self):
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights_APPOLO'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\mini_data_saves\\with_A2_using_3_channels_0.1'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A2'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A3'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3_0.2'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A3_L1'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A5'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrain_test'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A7'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\pretrained_A8'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A8'
        path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A8_first1'
        # path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\finetuned_A9'
        
        encoder_name = 'Encoder' + str(self.epoch) + '.pth'
        depth_name = 'Depth' + str(self.epoch) + '.pth'
        pose_name = 'Pose' + str(self.epoch) + '.pth'
        initial_name = 'Initial_encoder' + str(self.epoch) + '.pth'
        
        torch.save(self.models['initial'].state_dict(), os.path.join(path, initial_name))
        torch.save(self.models['encoder'].state_dict(), os.path.join(path, encoder_name))
        torch.save(self.models['depth'].state_dict(), os.path.join(path, depth_name))
        torch.save(self.models['pose'].state_dict(), os.path.join(path, pose_name))
        
        # teacher_initial_name = 'Initial_encoder_teacher' + str(self.epoch) + '.pth'
        # teacher_encoder_name = 'Encoder_teacher' + str(self.epoch) + '.pth'
        # torch.save(self.models['initial_teacher'].state_dict(), os.path.join(path, teacher_initial_name))
        # torch.save(self.models['encoder_teacher'].state_dict(), os.path.join(path, teacher_encoder_name))
    
    def load_pretrained(self):
        weights_folder = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\mini_data_saves\\pretrain_affinity'
        self.models['initial_teacher'].load_state_dict(torch.load(os.path.join(weights_folder, 'Initial_encoder.pth')))

    
    def pretrain_affinity(self):
        self.epoch = 0
        self.step = 0
        self.val_latch = 100
        
        for self.epoch in range(0, self.opt.num_epochs):
            self.run_epoch(train_mode = 'affinity_only')
            # self.save_pretrained()
            self.save_model()
    
    def train(self):
        self.epoch = 0
        self.step = 0
        self.val_latch = 100
        # self.it = 0
        self.load_model() # -------------------------------------------  Remove this for another experiment -------
        # self.load_pretrained()
        # self.load_entire()

        for self.epoch in range(0, self.opt.num_epochs):
            self.run_epoch(train_mode = 'combined')
            
            # self.save_model()
                
    def perform_ema(self, it):
        # starting_params = self.models['encoder_teacher'].parameters()
        
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.models['encoder'].parameters(), self.models['encoder_teacher'].parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
            # for param_dq, param_dk in zip(self.models['depth'].parameters(), self.models['depth_teacher'].parameters()):
            #     param_dk.data.mul_(m).add_((1 - m) * param_dq.detach().data)
        

    def single_ema(self, it, copied_model):
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(copied_model.parameters(), self.models['encoder'].parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
            
    def ema(student_model, teacher_model, alpha):
        student_weights = student_model.get_weights()
        teacher_weights = teacher_model.get_weights()
        
        #length must be equal otherwise it will not work 
        assert len(student_weights) == len(teacher_weights), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format(
            len(student_weights), len(teacher_weights))
        
        new_layers = []
        # assigning weights
        for i, layers in enumerate(student_weights):
            new_layer = alpha*(teacher_weights[i]) + (1-alpha)*layers
            new_layers.append(new_layer)
        teacher_model.set_weights(new_layers)
        return teacher_model
            
    def inverse_ema(self):
        with torch.no_grad():
            for param_q, param_k in zip(self.models['encoder'].parameters(), self.models['encoder_teacher'].parameters()):
                print('not applied yet')
    
            
    def test_model(self):
        self.load_model()
        self.set_eval()
        self.val_mode = True
        # ind = 0
        ii = 0
        for loader in self.val_loader:
            prev_features = None
            for inputs in tqdm(loader):
                with torch.no_grad():
                    
                    outputs, losses = self.process_batch_val(inputs, prev_features)
                    prev_features = outputs
                    
                    if "depth_gt" in inputs:
                        dl, depth_errors = self.compute_depth_losses(inputs, outputs, losses)
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_errors
                        print('scores: ', ii, ' ', abs_rel, sq_rel, rmse, rmse_log)
                        ii = ii + 1
                        
                        image_prev = inputs[('image_aug', -1)].cpu().numpy()
                        image = inputs[('image_aug', 0)].cpu().numpy()
                        image_next = inputs[('image_aug', 1)].cpu().numpy()
                        
                        recons_from_prev = outputs[('color', -1)].cpu().numpy()
                        recons_from_next = outputs[('color', 1)].cpu().numpy()
                        
                        depth_pred = outputs[("disp", 0)].cpu().numpy()
                        orig_images = inputs[('original_image_path', 0)]
                        all_indexes = inputs[('index', 0)]
                        gt_depths = inputs[('depth_gt')]
                        gt_depths = np.squeeze(gt_depths, axis = 1)
                        # print('depth gt shape: ', gt_depths.shape)
                        # print('all_indexes: ', all_indexes)
                        # print(all_indexes[0])
                        
                        # print('orign images path shape: ', orig_images.shape)
                        # print('recons_from_prev: ', recons_from_prev.shape)
                        # print('recons_from_next: ', recons_from_next.shape)
                        # print('image shape: ', inputs[('image_aug', -1)].cpu().numpy().shape)
                        # print('depth shape: ', depth_pred.shape)
                        # print('depth max; ', np.max(depth_pred))
                        
                        # try:
                        
                        for b in range(4):
                            prev_image_slice = np.transpose(image_prev[b, :, :, :], (1, 2, 0))
                            image_slice = np.transpose(image[b, :, :, :], (1, 2, 0))
                            next_image_slice = np.transpose(image_next[b, :, :, :], (1, 2, 0))
                            depth_slice = depth_pred[b, 0, :, :]
                            
                            # print('depth unique: ', np.unique(depth_slice))
                            recons_prev_slice = np.transpose(recons_from_prev[b, :, :, :], (1, 2, 0))
                            recons_next_slice = np.transpose(recons_from_next[b, :, :, :], (1, 2, 0))
                            
                            depth_pred_slice = depth_pred[b, 0, :, :]
                            
                            
                            prev_image_slice = (prev_image_slice/np.max(prev_image_slice))*255
                            image_slice = (image_slice/np.max(image_slice))*255
                            next_image_slice = (next_image_slice/np.max(next_image_slice))*255
                            
                            recons_prev_slice = (recons_prev_slice/np.max(recons_prev_slice))*255
                            recons_next_slice = (recons_next_slice/np.max(recons_next_slice))*255
                            
                            orig_slice_path = orig_images[b]
                            # print('original path: ', orig_slice_path)
                            # print('splited: ', orig_slice_path.split('\\'))
                            
                            drive = orig_slice_path.split('\\')[4]
                            im_name = orig_slice_path.split('\\')[5]
                            file = orig_slice_path.split('\\')[7]
                            
                            depth_GT_path = 'E:\\Datasets\\data_depth_annotated\\train\\' + str(drive) + '\\proj_depth\\groundtruth\\' + str(im_name) + '\\' + str(file)
                            # print('#GT path: ', depth_GT_path)
                            # gt_depth = cv2.imread(depth_GT_path)
                            
                            depth_slice_gt = gt_depths[b]
                            orig_image = cv2.imread(orig_slice_path)
                            depth_slice_gt = depth_slice_gt.numpy()
                            depth_slice_gt = depth_slice_gt / np.max(depth_slice_gt)
                            # print('gt unique: ', np.unique(depth_slice_gt))
                            # print('GT shape: ', gt_depth.shape)
                            plt.imshow(orig_image)
                            plt.show()
                            plt.imshow(depth_slice, cmap = 'gray') #gt_depth*256)
                            plt.show()
                            plt.imshow(depth_slice)
                            plt.show()
                            
                            ind = all_indexes[b].cpu().numpy()
                            
                            image_m1_name = 'image_m1\\original_m1_image_' + str(ind) + '.png'
                            image_name = 'image\\original_image_' + str(ind) + '.png'
                            image_p1_name = 'image_p1\\original_p1_image_' + str(ind) + '.png'
                            
                            reproj_m1_name = 'reprojected_m1\\reprojected_m1_image_' + str(ind) + '.png'
                            reproj_p1_name = 'reprojected_p1\\reprojected_p1_image_' + str(ind) + '.png'
                            
                            orig_image_name = 'original_image\\image_' + str(ind) + '.png'
                            
                            depth_name_jpg = 'predicted_depth\\depth_' + str(ind) + '.jpg'
                            gt_depth_name_jpg = 'predicted_depth\\GT_depth_' + str(ind) + '.jpg'
                            
                            depth_name = 'predicted_depth\\depth_' + str(ind) + '.npy'
                            gt_depth_name = 'predicted_depth\\GT_depth_' + str(ind) + '.npy'
    
                            cv2.imwrite(os.path.join(self.save_folder, image_m1_name), prev_image_slice)
                            cv2.imwrite(os.path.join(self.save_folder, image_name), image_slice)
                            cv2.imwrite(os.path.join(self.save_folder, image_p1_name), next_image_slice)
                            
                            cv2.imwrite(os.path.join(self.save_folder, reproj_m1_name), recons_prev_slice)
                            cv2.imwrite(os.path.join(self.save_folder, reproj_p1_name), recons_next_slice)
                            
                            cv2.imwrite(os.path.join(self.save_folder, orig_image_name), orig_image)
                            
                            # print('pred unique: ', np.unique(depth_slice))
                            # print('gt unique: ', np.unique(gt_depth))
                            
                            depth_pil = Image.fromarray(np.uint8(depth_slice*256)).convert('RGB')
                            depth_pil.save(os.path.join(self.save_folder, depth_name_jpg))
                            
                            # gt_depth = gt_depth / np.max(gt_depth)
                            # gt_depth_pil = Image.fromarray(np.uint8(gt_depth*256)).convert('RGB')
                            # gt_depth_pil.save(os.path.join(self.save_folder, gt_depth_name_jpg))
                            
                            # cv2.imwrite(os.path.join(self.save_folder, depth_name_jpg), depth_slice)
                            # cv2.imwrite(os.path.join(self.save_folder, gt_depth_name_jpg), gt_depth)
                            
                            
                            np.save(os.path.join(self.save_folder, depth_name), depth_slice)
                            # np.save(os.path.join(self.save_folder, gt_depth_name), gt_depth)
                            # print('---saved---')
                            ind = ind + 1
                        # except:
                            # print('-------------------SKIPPED--------------------')
        
    def study_model(self):
        self.load_entire()
        self.set_eval()
        self.val_mode = True
        ii = 0
        for inputs in tqdm(self.val_loader):
            if ii >= 54:
                with torch.no_grad():
                    # outputs, losses = self.process_batch_analysis(inputs)
                    self.process_batch_analysis(inputs)
            
            ii = ii+1
        
    def run_epoch(self, train_mode):
        self.set_train()
        # reprojection = []
        abs_req = []
        sq_req = []
        rmse_req = []
        rmse_log_req = []
        epoch_loss = []
        batch_idx = 0
        for it, inputs in enumerate(tqdm(self.train_loader)):
            all_loss = 0
            batch_idx = batch_idx + 1
            
            if train_mode == 'combined':
                outputs, losses = self.process_batch(inputs)
                
                it = len(self.train_loader) * self.epoch + it
                self.perform_ema(it)
                
            elif train_mode == 'affinity_only':
                outptuts, losses = self.process_affinity_only(inputs)
            else:
                print('Please set the train_mode to combined or affinity_only')

            
            if 'depth_gt' in inputs:
                dl, depth_errors = self.compute_depth_losses(inputs, outputs, losses)
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_errors
                abs_req.append(abs_rel.detach().cpu().numpy())
                sq_req.append(sq_rel.detach().cpu().numpy())
                rmse_req.append(rmse.detach().cpu().numpy())
                rmse_log_req.append(rmse_log.detach().cpu().numpy())
                # print('depth loss: ', dl)
                #print('calc losses: ', losses['loss'])
                epoch_loss.append(losses['loss'].detach().cpu().numpy())
                # print('dl grad: ', dl.requires_grad)
                all_loss = losses['loss']# + dl
            
            else:
                epoch_loss.append(losses['loss'].detach().cpu().numpy())
                all_loss = losses['loss']
            
            # self.models['dummy_encoder'].load_state_dict(self.models['encoder'].state_dict()) ###########################################################################################
            
            # print('final loss: ', all_loss, all_loss.double())
            self.model_optimizer.zero_grad()
            all_loss.double().backward()
            self.model_optimizer.step()
            # plot_grad_flow(self.parameters_to_train)
            
            
            # self.single_ema(it, self.models['dummy_encoder']) ############################################################################################################################
        
        print('-------Training scores------------')
        print('Epoch: ', self.epoch)
        print('Epoch loss: ', np.mean(epoch_loss))
        # print('Reprojection Loss: ', np.mean(reprojection))
        print('ABS: ', np.mean(abs_req))
        print('SQ: ', np.mean(sq_req))
        print('RMSE: ', np.mean(rmse_req))
        print('RMSE Log: ', np.mean(rmse_log_req))
        
        with open(self.file_path, 'a') as f:
            f.write(f'Epoch: {self.epoch}')
            f.write('\n')
            f.write(f'Train Loss: {np.mean(epoch_loss)}')
            f.write('\n')
            f.write(f'Train ABS: {np.mean(abs_req)}')
            f.write('\n')
            f.write(f'Train SQ: {np.mean(sq_req)}')
            f.write('\n')
            f.write(f'Train RMSE: {np.mean(rmse_req)}')
            f.write('\n')
            f.write(f'Train RMSE log: {np.mean(rmse_log_req)}')
            f.write('\n')
        
        val_loss, val_abs = self.val()
        print('loss and latch: ', val_abs, self.val_latch)
        # if val_abs < self.val_latch:
        #     self.save_model()
        #     self.val_latch = val_abs
        #     print('saved model and set latch value to : ', self.val_latch)
            
        self.save_model()
            
    def val(self):
        """Validate the model on a single minibatch
        """
        # print('entered val')
        # self.load_model()
        self.set_eval()
        self.val_mode = True
        
        abs_req = []
        sq_req = []
        rmse_req = []
        rmse_log_req = []
        # reprojection = []
        
        try:
            inputs = self.val_iter.__next__()#next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.__next__()#self.val_iter.next()
        
        for loader in self.val_loader:
            prev_features = None
            
            for inputs in tqdm(loader):
                inin = inputs[('image_aug', 0)]
                if inin.shape[0] == self.batch_size:
                    
                    with torch.no_grad():
                        outputs, losses = self.process_batch_val(inputs, prev_features)
                        prev_features = outputs
                        
                        # reprojection.append(np.mean(losses['reprojection'].detach().cpu().numpy()))
            
                        if "depth_gt" in inputs:
                            dl, depth_errors = self.compute_depth_losses(inputs, outputs, losses)
                            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_errors
                            abs_req.append(abs_rel.detach().cpu().numpy())
                            sq_req.append(sq_rel.detach().cpu().numpy())
                            rmse_req.append(rmse.detach().cpu().numpy())
                            rmse_log_req.append(rmse_log.detach().cpu().numpy())
            
                        #self.log("val", inputs, outputs, losses)
                        del inputs, outputs
        
        print('-------Validation scores------------')
        # print('Reprojection Loss: ', np.mean(reprojection))
        print('ABS: ', np.mean(abs_req))
        print('SQ: ', np.mean(sq_req))
        print('RMSE: ', np.mean(rmse_req))
        print('RMSE Log: ', np.mean(rmse_log_req))
        
        with open(self.file_path, 'a') as f:
            f.write(f'Validation ABS: {np.mean(abs_req)}')
            f.write('\n')
            f.write(f'Validation SQ: {np.mean(sq_req)}')
            f.write('\n')
            f.write(f'Validation RMSE: {np.mean(rmse_req)}')
            f.write('\n')
            f.write(f'Validation RMSE log: {np.mean(rmse_log_req)}')
            f.write('\n')
            f.write('\n')
        
        self.set_train()
        self.val_mode = False
        return losses['loss'], np.mean(abs_req)
    
    def process_batch_val(self, inputs, outputs):
        
        if self.opt.pose_model_type == 'shared':
            all_colour_aug = None
            all_colour_aug = torch.cat([inputs[('image_aug', i)] for i in self.opt.frame_ids])
            # print('input images shape: ', all_colour_aug.shape)
            
            all_features = self.models['initial'](all_colour_aug.cuda().float())
            all_features = self.models['encoder'](all_features)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            
            features ={}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            
            
            if outputs != None:
                T = outputs[("cam_T_cam", -2)]
                
                disp = outputs[('disp', 0)]
                
                prev_maps = {}
                
                disp_1 = F.interpolate(disp, [24, 80], mode="bilinear", align_corners=False)
                _, depth_1 = disp_to_depth(disp_1, self.opt.min_depth, self.opt.max_depth)
                
                cam_points_1 = self.backproject_depth_1(depth_1, inputs[('inv_K')])
                pix_coords_1 = self.project_3d_1(cam_points_1, inputs[('K')], T)
                prev_maps[('feats', 1)] = F.grid_sample(outputs[('feats', 1)].cuda(), pix_coords_1, padding_mode="border")
                
                #####################################################################################################################################
                
                disp_2 = F.interpolate(disp, [12, 40], mode="bilinear", align_corners=False)
                _, depth_2 = disp_to_depth(disp_2, self.opt.min_depth, self.opt.max_depth)
                
                cam_points_2 = self.backproject_depth_2(depth_2, inputs[('inv_K')])
                pix_coords_2 = self.project_3d_2(cam_points_2, inputs[('K')], T)
                prev_maps[('feats', 2)] = F.grid_sample(outputs[('feats', 1)].cuda(), pix_coords_2, padding_mode="border")
                
                ####################################################################################################################################
                
                disp_3 = F.interpolate(disp, [6, 20], mode="bilinear", align_corners=False)
                _, depth_3 = disp_to_depth(disp_3, self.opt.min_depth, self.opt.max_depth)
                
                cam_points_3 = self.backproject_depth_3(depth_3, inputs[('inv_K')])
                pix_coords_3 = self.project_3d_3(cam_points_3, inputs[('K')], T)
                prev_maps[('feats', 3)]  = F.grid_sample(outputs[('feats', 1)].cuda(), pix_coords_3, padding_mode="border")
                
                #####################################################################################################################################
                
                # outputs_second = self.models['depth'](features[1], prev_maps)
    
    
                # disp = outputs_second[('disp', 0)]
                # disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                
                # cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                # pix_coords = self.project_3d(cam_points, inputs[("K")], T)
                # outputs[("sample", frame_id)] = pix_coords
            else:
                prev_maps = None
            
            
            outputs = self.models['depth'](features[0], prev_maps)
        
        else:
            print('separate encoders not implemented')
        
        if self.use_pose_net:
            outputs.update(self.predict_pose(inputs, features))
            
        self.generate_images_pred_single(inputs, outputs)
        losses = self.compute_losses_single(inputs, outputs)
        return outputs, losses
    
    def process_batch_analysis(self, inputs):
        
        # all_colour_aug_aff = None
        # for i in self.opt.frame_ids:
        #     image = inputs[('image_aug', i)]
        #     affinity = inputs[('affinity', i)][:, 0, :, :, :]
        #     combined = torch.cat([image, affinity], dim = 1)

        #     if all_colour_aug_aff == None:
        #         all_colour_aug_aff = combined
        #     else:
        #         all_colour_aug_aff = torch.cat([all_colour_aug_aff, combined])
        
        all_colour_aug = torch.cat([inputs[('image_aug', i)] for i in self.opt.frame_ids])
        print('input shape: ', all_colour_aug.shape)
        
        disp_im_1 = all_colour_aug[0, :, :, :].detach().cpu().numpy()
        disp_im_2 = all_colour_aug[1, :, :, :].detach().cpu().numpy()
        disp_im_3 = all_colour_aug[2, :, :, :].detach().cpu().numpy()
        disp_im_4 = all_colour_aug[3, :, :, :].detach().cpu().numpy()
        disp_im_5 = all_colour_aug[4, :, :, :].detach().cpu().numpy()
        disp_im_1 = np.transpose(disp_im_1, (1, 2, 0))
        disp_im_2 = np.transpose(disp_im_2, (1, 2, 0))
        disp_im_3 = np.transpose(disp_im_3, (1, 2, 0))
        disp_im_4 = np.transpose(disp_im_4, (1, 2, 0))
        disp_im_5 = np.transpose(disp_im_5, (1, 2, 0))
        print('Disp shapes: ', disp_im_1.shape, disp_im_2.shape)
        plt.imshow(disp_im_1)
        plt.show()
        plt.imshow(disp_im_2)
        plt.show()
        plt.imshow(disp_im_3)
        plt.show()
        plt.imshow(disp_im_4)
        plt.show()
        plt.imshow(disp_im_5)
        plt.show()
        
        all_features = self.models['initial'](all_colour_aug.cuda().float())
        all_features = self.models['encoder'](all_features)
        
        # all_features_teacher = self.models['initial_teacher'](all_colour_aug_aff.cuda().float())
        # all_features_teacher = self.models['encoder_teacher'](all_features_teacher)
        
        print('student features shape: ', all_features[3].shape)
        # print('teacher features shape: ', all_features_teacher[3].shape)
        
        plt.imshow(all_features[0][0, 0, :, :].cpu().numpy())
        plt.show()
        # plt.imshow(all_features_teacher[0][0, 0, :, :].cpu().numpy())
        # plt.show()
        
        plt.imshow(all_features[0][1, 0, :, :].cpu().numpy())
        plt.show()
        # plt.imshow(all_features_teacher[0][35, 0, :, :].cpu().numpy())
        # plt.show()
        plt.imshow(all_features[0][2, 0, :, :].cpu().numpy())
        plt.show()
        plt.imshow(all_features[0][3, 0, :, :].cpu().numpy())
        plt.show()
        plt.imshow(all_features[0][4, 0, :, :].cpu().numpy())
        plt.show()
        
        # self.load_model()
        
        # all_features = self.models['initial'](all_colour_aug.cuda().float())
        # all_features = self.models['encoder'](all_features)
        
        # plt.imshow(all_features[0][0, 0, :, :].cpu().numpy())
        # plt.show()

        # plt.imshow(all_features[0][1, 0, :, :].cpu().numpy())
        # plt.show()
        # plt.imshow(all_features[0][2, 0, :, :].cpu().numpy())
        # plt.show()
        # plt.imshow(all_features[0][3, 0, :, :].cpu().numpy())
        # plt.show()
        # plt.imshow(all_features[0][4, 0, :, :].cpu().numpy())
        # plt.show()
        

        # return outputs, losses
    
    def process_affinity_only(self, inputs):
        
        if self.opt.pose_model_type == 'shared':
            
            all_colour_aug_aff = None
            for i in self.opt.frame_ids:
                image = inputs[('image_aug', i)]
                affinity = inputs[('affinity', i)][:, 0, :, :, :]
                combined = torch.cat([image, affinity], dim = 1)
                
                if all_colour_aug_aff == None:
                    all_colour_aug_aff = combined
                else:
                    all_colour_aug_aff = torch.cat([all_colour_aug_aff, combined])
                
            all_features_teacher = self.models['initial_teacher'](all_colour_aug_aff.cuda().float())
            all_features_teacher = self.models['encoder_teacher'](all_features_teacher)
            all_features_teacher = [torch.split(f, self.opt.batch_size) for f in all_features_teacher]
            
            teacher_features = {}
            for i, k in enumerate(self.opt.frame_ids):
                teacher_features[k] = [f[i] for f in all_features_teacher]
                
            outputs_teacher = self.models['depth_teacher'](teacher_features[0])
            
        else:
            print('Separate ecnoders nto implemented')
        
        if self.use_pose_net:
            outputs_teacher.update(self.predict_pose(inputs, teacher_features))
        
        self.generate_images_pred_single(inputs, outputs_teacher)
        losses = self.compute_losses_single(inputs, outputs_teacher)
        
        return outputs_teacher, losses
    
    def process_batch(self, inputs):
        
        if self.opt.pose_model_type == 'shared':
            
            all_colour_aug_aff = None
            
            if self.use_affinity == True:
                for i in self.opt.frame_ids:
                    image = inputs[('image_aug', i)]
                    affinity = inputs[('affinity', i)][:, 0, :, :, :]
                    combined = torch.cat([image, affinity], dim = 1)
    
                    if all_colour_aug_aff == None:
                        all_colour_aug_aff = combined
                    else:
                        all_colour_aug_aff = torch.cat([all_colour_aug_aff, combined])

            all_colour_aug = torch.cat([inputs[('image_aug', i)] for i in self.opt.frame_ids])
            # print('input shape: ', len(all_colour_aug), all_colour_aug.shape)
            
            all_features = self.models['initial'](all_colour_aug.cuda().float())
            all_features = self.models['encoder'](all_features)
            
            buffered_4 = self.models['buffer_m1'](all_features[4])
            buffered_3 = self.models['buffer_m2'](all_features[3])
            buffered_2 = self.models['buffer_m3'](all_features[2])
            
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            
            buffered_4 = torch.split(buffered_4, self.opt.batch_size)
            buffered_3 = torch.split(buffered_3, self.opt.batch_size)
            buffered_2 = torch.split(buffered_2, self.opt.batch_size)
            
            buffered_features = {}
            buffered_features['buffered_m1'] = buffered_4
            buffered_features['buffered_m2'] = buffered_3
            buffered_features['buffered_m3'] = buffered_2
            
            features = {}
            
            for i, k in enumerate(self.opt.frame_ids): # order is 0, -1, 1
                features[k] = [f[i] for f in all_features]
            
            outputs = self.models['depth'](features[0])
            # outputs_second = self.models['depth'](features[1], outputs)
            
            
            if self.use_affinity == True:
                with torch.no_grad():
                    all_features_teacher = self.models['initial_teacher'](all_colour_aug_aff.cuda().float())
                    all_features_teacher = self.models['encoder_teacher'](all_features_teacher)
                    
                    all_features_teacher = [torch.split(f, self.opt.batch_size) for f in all_features_teacher]
                    
                teacher_features = {}
                for i, k in enumerate(self.opt.frame_ids):
                    teacher_features[k] = [f[i] for f in all_features_teacher]
                
                with torch.no_grad():
                    outputs_teacher = self.models['depth_teacher'](teacher_features[0])
        
        else:
            print('separate encoders not implemented')
        
        if self.use_pose_net:
            outputs.update(self.predict_pose(inputs, features))
            
        # self.generate_images_pred(inputs, outputs, outputs_teacher)
        
        if self.use_affinity == True:
            outputs_second = self.generate_images_pred(inputs, outputs, features, outputs_teacher)
            losses = self.compute_losses(inputs, outputs, outputs_second, outputs_teacher, features, teacher_features, buffered_features)
            
        if self.use_affinity == False:
            outputs_second = self.generate_images_pred(inputs, outputs, features)
            losses = self.compute_losses_single(inputs, outputs)
        
        return outputs, losses
        
    def predict_pose(self, inputs, features):
        
        outputs = {}
        
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == 'shared':
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs['image_aug', f_i] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:
                if f_i != 's':
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]
                
                    if self.opt.pose_model_type == 'separate_resnet':
                        pose_inputs = [self.model['pose_encoder'](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == 'posecnn':
                        pose_inputs = torch.cat(pose_inputs, 1)
                        
                    
                    axisangle, translation = self.models['pose'](pose_inputs)
                    outputs[('axisangle', f_i)] = axisangle
                    outputs[('translation', f_i)] = translation
                    
                    outputs[('cam_T_cam', f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert = (f_i < 0))
                    
        else:
            if self.opt.pose_model_type in ['separate_resnet', 'posecnn']:
                pose_inputs = torch.cat(
                    [inputs[('image_aug', i)] for i in self.opt.frame_ids if i != 's'], 1)
                
                if self.opt.pose_model_type == 'separate_resnet':
                    pose_inputs = [self.model['pose_encoder'](pose_inputs)]
                    
            elif self.opt.pose_model_type == 'shared':
                # ================== this one ===================
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != 's']
            
            axisangle, translation = self.models['pose'](pose_inputs)
            # print('axis angle slice: ', axisangle[:, 0].shape)
            
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                # print('FI: ', f_i)
                if f_i != 's':
                    outputs[('axisangle', f_i)] = axisangle
                    outputs[('translation', f_i)] = translation
                    outputs[('cam_T_cam', f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])
            
            outputs[('axisangle', -2)] = axisangle
            outputs[('translation', -2)] = translation
            outputs[('cam_T_cam', -2)] = transformation_from_parameters(axisangle[:, 2], translation[:, 2])
            # print('--- addition --- done ---')
                    
        return outputs
                
    def generate_images_pred_single(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[('disp', 0)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0)] = depth
            
            init_arr = self.opt.frame_ids[1:]
            for i, frame_id in enumerate(init_arr):
                # print(' frame IDs: ', frame_id)
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", frame_id)]
                    translation = outputs[("translation", frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                
                cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                pix_coords = self.project_3d(cam_points, inputs[("K")], T)

                outputs[("sample", frame_id)] = pix_coords
                outputs[("color", frame_id)] = F.grid_sample(inputs[("image", frame_id)].cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                
                # if self.use_affinity == True and self.val_mode == False:
                #     gt_affinity = inputs[("affinity", frame_id)][:, 0, :, :, :].double()
                #     outputs[("pred_affinity", frame_id)] = F.grid_sample(gt_affinity.cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                    
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id)] = \
                        inputs[("image", frame_id)]
    
    
    def generate_images_pred(self, inputs, outputs, features, outputs_teacher = 0):
        for scale in self.opt.scales:
            disp = outputs[('disp', 0)]
            # print('final out disp shape: ', disp.shape)
            if self.use_affinity == True:
                disp_teacher = outputs_teacher[('disp', 0)]
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                if self.use_affinity == True:
                    disp_teacher = F.interpolate(disp_teacher, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            
            outputs[("depth", 0)] = depth
            
            if self.use_affinity == True:
                _, depth_teacher = disp_to_depth(disp_teacher, self.opt.min_depth, self.opt.max_depth)
                outputs_teacher[('depth', 0)] = depth_teacher
            
            init_arr = np.array(self.opt.frame_ids[1:])
            print('init arr: ', init_arr)
            init_arr = np.append(init_arr, -2)
            # print('init arr: ', init_arr)
            for i, frame_id in enumerate(init_arr):
                # print('frame ID: ', frame_id)
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_id)]


                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", frame_id)]
                    translation = outputs[("translation", frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    
                    if self.use_affinity == True:
                        inv_depth_teacher = 1 / depth_teacher
                        mean_inv_depth_teacher = inv_depth_teacher.mean(3, True).mean(2, True)
                        T_teacher = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth_teacher[:, 0], frame_id < 0)
                        
                        cam_points_teacher = self.backproject_depth(depth_teacher, inputs[('inv_K')])
                        pix_coords_teacher = self.project_3d(cam_points_teacher, inputs[("K")], T)
                        outputs_teacher[('sample', frame_id)] = pix_coords_teacher
                        outputs_teacher[("color", frame_id)] = F.grid_sample(inputs[("image", frame_id)].cuda(), outputs_teacher[("sample", frame_id)].double(), padding_mode="border")
                    
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                
                cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                pix_coords = self.project_3d(cam_points, inputs[("K")], T)
                outputs[("sample", frame_id)] = pix_coords
                
                ####################################################################################################################################
                ####################################################################################################################################
                # print('---- features shape: ', features.keys(), features[-1][0].shape, features[1][0].shape)
                # plt.imshow(features[-1][0][0, 64, :, :].detach().cpu().numpy())
                # plt.show()
                # plt.imshow(features[0][0][0, 64, :, :].detach().cpu().numpy())
                # plt.show()
                # plt.imshow(features[1][0][0, 64, :, :].detach().cpu().numpy())
                # plt.show()
                
                # disp_1 = F.interpolate(disp, [48, 160], mode="bilinear", align_corners=False)
                # _, depth_1 = disp_to_depth(disp_1, self.opt.min_depth, self.opt.max_depth)
                # print('---- depth shape: ', depth_1.shape)
                # cam_points_1 = self.backproject_depth_test(depth_1, inputs[('inv_K')])
                # pix_coords_1 = self.project_3d_test(cam_points_1, inputs[('K')], T)
                # projected_feats = F.grid_sample(features[0][0].cuda(), pix_coords_1, padding_mode="border")
                
                # plt.imshow(projected_feats[0, 64, :, :].detach().cpu().numpy())
                # plt.show()
                ####################################################################################################################################
                ####################################################################################################################################
                
                if frame_id == -2: # this is predicting frame t+1 using frame t
                    prev_maps = {}
                    
                    disp_1 = F.interpolate(disp, [24, 80], mode="bilinear", align_corners=False)
                    _, depth_1 = disp_to_depth(disp_1, self.opt.min_depth, self.opt.max_depth)
                    
                    cam_points_1 = self.backproject_depth_1(depth_1, inputs[('inv_K')])
                    pix_coords_1 = self.project_3d_1(cam_points_1, inputs[('K')], T)
                    prev_maps[('feats', 1)] = F.grid_sample(outputs[('feats', 1)].cuda(), pix_coords_1, padding_mode="border")
                    
                    #####################################################################################################################################
                    
                    disp_2 = F.interpolate(disp, [12, 40], mode="bilinear", align_corners=False)
                    _, depth_2 = disp_to_depth(disp_2, self.opt.min_depth, self.opt.max_depth)
                    
                    cam_points_2 = self.backproject_depth_2(depth_2, inputs[('inv_K')])
                    pix_coords_2 = self.project_3d_2(cam_points_2, inputs[('K')], T)
                    prev_maps[('feats', 2)] = F.grid_sample(outputs[('feats', 2)].cuda(), pix_coords_2, padding_mode="border")
                    
                    ####################################################################################################################################
                    
                    disp_3 = F.interpolate(disp, [6, 20], mode="bilinear", align_corners=False)
                    _, depth_3 = disp_to_depth(disp_3, self.opt.min_depth, self.opt.max_depth)
                    
                    cam_points_3 = self.backproject_depth_3(depth_3, inputs[('inv_K')])
                    pix_coords_3 = self.project_3d_3(cam_points_3, inputs[('K')], T)
                    prev_maps[('feats', 3)]  = F.grid_sample(outputs[('feats', 3)].cuda(), pix_coords_3, padding_mode="border")
                    
                    #####################################################################################################################################
                    
                    outputs_second = self.models['depth'](features[1], prev_maps) # prev_maps here are the features for frame t and this step is predicting depth of frame t+1


                    disp = outputs_second[('disp', 0)]
                    disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                    
                    cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                    pix_coords = self.project_3d(cam_points, inputs[("K")], T)
                    outputs[("sample", frame_id)] = pix_coords

                # ###################################################################################################################################
                
                if frame_id == -2:
                    outputs[("color", frame_id)] = F.grid_sample(inputs[("image", 0)].cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                if frame_id > -2:
                    outputs[("color", frame_id)] = F.grid_sample(inputs[("image", frame_id)].cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id)] = \
                        inputs[("image", frame_id)]
                
        return outputs_second
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target.cuda() - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    
    def compute_affinity_reprojection_loss(self, input_, t_seg, target, weight):

        sumed_weight = torch.unsqueeze(torch.sum(target, dim = 1), dim = 1)
        sumed_weight = torch.where(sumed_weight < 24, 1, 0)

        bce_loss = (input_ - target) ** 2

        huber = torch.nn.HuberLoss()
        loss = huber(input_, target.double()) * sumed_weight

        return torch.mean(loss)
    
    
    def compute_losses_single(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", 0)]
            color = inputs[("image", 0)]
            target = inputs[("image", 0)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id)]
                rp = self.compute_reprojection_loss(pred, target)
                reprojection_losses.append(rp)
            
            ###################################################################
            if self.val_mode == False:
                rp2 = self.compute_reprojection_loss(outputs[('color', -2)], inputs[('image', 1)])
                reprojection_losses.append(rp2)
            ###################################################################
                    
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("image", frame_id)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses
            
            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp"]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                # weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                # loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection"] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        # print('total loss: ', total_loss)
        losses["loss"] = total_loss
        return losses
    
    def Reg(self, features):
        features = features.float()
        T = (((features**2).sum(dim=1))**0.5) + 1e-8 # this was the original code T = (((features**2).sum(dim=1))**0.5) + 1e-8
        # print('features/T: ', features.shape, T.shape)
        r_feats = features/T.unsqueeze(dim = 1)
        return r_feats
    
    def compute_losses(self, inputs, outputs, outputs_second, outputs_teacher, features, teacher_features, buffered_features):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", 0)]
            disp_teacher = outputs_teacher[("disp", 0)]
            color = inputs[("image", 0)]
            target = inputs[("image", 0)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id)]
                # pred_teacher = outputs_teacher[('color', frame_id)]
                # print('loss calculation frame ID: ', frame_id)
                rp = self.compute_reprojection_loss(pred, target)
                # rp_st = self.compute_reprojection_loss(pred, pred_teacher.detach()) # --------------------------------------------------
                reprojection_losses.append(rp*0.9)
                # if self.epoch >= 2:
                    # reprojection_losses.append(rp_st*0.05) # ----------------------------------------------------
                # losses['reprojection'] = rp
            
            ###################################################################
            rp2 = self.compute_reprojection_loss(outputs[('color', -2)], inputs[('image', 1)])
            reprojection_losses.append(rp2)
            ###################################################################
            
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("image", frame_id)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses
            
            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp"]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                # weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                # loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection"] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            # print('To optimize: ', to_optimise.mean())

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # print(norm_disp.shape, color.shape)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            
            huber = torch.nn.HuberLoss()
            mse = torch.nn.MSELoss()
            l1 = torch.nn.L1Loss()
            
            if self.epoch >= 2:
                # print('check shapes: ', len(buffered_features['buffered_m1', 1]), teacher_features[0][4].shape)
                # print('feature shapes: ', buffered_features['buffered_m1'][0].shape, teacher_features[0][4].shape)
                
                enc_dif_loss_00 = huber(self.Reg(buffered_features['buffered_m1'][0]), self.Reg(teacher_features[0][4]).detach()) # ------------------------------------------------------------------
                enc_dif_loss_01 = huber(self.Reg(buffered_features['buffered_m1'][1]), self.Reg(teacher_features[1][4]).detach()) # ------------------------------------------------------------------
                enc_dif_loss_02 = huber(self.Reg(buffered_features['buffered_m1'][2]), self.Reg(teacher_features[-1][4]).detach()) # ------------------------------------------------------------------
                
                enc_dif_loss_10 = huber(self.Reg(buffered_features['buffered_m2'][0]), self.Reg(teacher_features[0][3]).detach()) # ------------------------------------------------------------------
                enc_dif_loss_11 = huber(self.Reg(buffered_features['buffered_m2'][1]), self.Reg(teacher_features[1][3]).detach()) # ------------------------------------------------------------------
                enc_dif_loss_12 = huber(self.Reg(buffered_features['buffered_m2'][2]), self.Reg(teacher_features[-1][3]).detach()) # ------------------------------------------------------------------
                
                enc_dif_loss_20 = huber(self.Reg(buffered_features['buffered_m3'][0]), self.Reg(teacher_features[0][2]).detach()) # ------------------------------------------------------------------
                enc_dif_loss_21 = huber(self.Reg(buffered_features['buffered_m3'][1]), self.Reg(teacher_features[1][2]).detach()) # ------------------------------------------------------------------
                enc_dif_loss_22 = huber(self.Reg(buffered_features['buffered_m3'][2]), self.Reg(teacher_features[-1][2]).detach()) # ------------------------------------------------------------------
                
                
                enc_dif_loss = (enc_dif_loss_00 + enc_dif_loss_01 + enc_dif_loss_02 + enc_dif_loss_10 + enc_dif_loss_11 + enc_dif_loss_12 + enc_dif_loss_10 + enc_dif_loss_11 + enc_dif_loss_12) / 9 # ------------------------------------------------------------------
                total_loss += enc_dif_loss * 0.1
            
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0)]
        # depth_pred_teacher = outputs_teacher[('depth', 0)]
        
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        # depth_errors = compute_depth_errors(depth_gt.cuda(), depth_pred)

        depth_gt.requires_grad_()
        depth_pred.requires_grad_()
        # print('gt grad: ', depth_gt.requires_grad)
        # print('pred grad: ', depth_pred.requires_grad)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(depth_gt.cuda(), depth_pred)
        depth_errors = abs_rel + sq_rel + rmse + rmse_log + a1 + a2 + a3

        # for i, metric in enumerate(self.depth_metric_names):
        #     losses[metric] = np.array(depth_errors[i].cpu())
        return depth_errors, [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]