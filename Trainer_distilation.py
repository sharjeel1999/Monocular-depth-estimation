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
        self.parameters_to_train += list(self.models['encoder'].parameters())

        
        self.models['initial_teacher'] = networks.Initial_teacher()
        self.models['initial_teacher'].to(self.device)
        # self.parameters_to_train += list(self.models['initial_teacher'].parameters())
        
        self.models['encoder_teacher'] = networks.ResnetEncoder_Teacher(num_layers = 18, pretrained = False) # options [18, 34, 50, 101, 152]
        self.models['encoder_teacher'].to(self.device)
        # self.parameters_to_train += list(self.models['encoder_teacher'].parameters())
        
        
        
        self.models['depth'] = networks.DepthDecoder(scales = self.opt.scales)
        self.models['depth'].to(self.device)
        self.parameters_to_train += list(self.models['depth'].parameters())
        
        self.models['depth_teacher'] = networks.DepthDecoder(scales = self.opt.scales)
        self.models['depth_teacher'].to(self.device)
        # self.parameters_to_train += list(self.models['depth_teacher'].parameters())
        

        self.models['pose'] = networks.PoseDecoder(self.num_pose_frames)
        self.models['pose'].to(self.device)
        self.parameters_to_train += list(self.models['pose'].parameters())
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, self.opt.scheduler_gamma)
        
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        self.backproject_depth = BackprojectDepth(batch_size, height = 192, width = 640)
        self.project_3d = Project3D(batch_size, height = 192, width = 640)
        self.ssim = SSIM()
        
        self.momentum_schedule = self.cosine_scheduler(0.998, 1, self.opt.num_epochs, len(self.train_loader))
        
        self.record_saves = 'D:\\depth_estimation_implementation\\all_saves\\record_saves'
        self.file_path = os.path.join(self.record_saves, 'A1_training.txt')
        
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
        self.models['encoder'].eval()
        self.models['depth'].train()
        self.models['pose'].train()
        
        # for m in self.models.values():
        #     m.train()
            
    def set_eval(self):
        self.models['encoder'].eval()
        self.models['depth'].eval()
        self.models['pose'].eval()
        
        # for m in self.models.values():
        #     m.eval()
            
    def load_model(self):
        path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\original_implementation_weights'
        encoder_name = 'Encoder.pth'
        depth_name = 'Depth.pth'
        pose_name = 'Pose.pth'
        self.models['encoder'].load_state_dict(torch.load(os.path.join(path, encoder_name)))
        self.models['depth'].load_state_dict(torch.load(os.path.join(path, depth_name)))
        self.models['pose'].load_state_dict(torch.load(os.path.join(path, pose_name)))
    
    def save_model(self):
        path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\mini_data_saves\\with_A1_algorithm'
        encoder_name = 'Encoder.pth'
        depth_name = 'Depth.pth'
        pose_name = 'Pose.pth'
        torch.save(self.models['initial'].state_dict(), os.path.join(path, 'Initial_encoder.pth'))
        torch.save(self.models['encoder'].state_dict(), os.path.join(path, encoder_name))
        torch.save(self.models['depth'].state_dict(), os.path.join(path, depth_name))
        torch.save(self.models['pose'].state_dict(), os.path.join(path, pose_name))
    
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
        self.load_pretrained()

        for self.epoch in range(0, self.opt.num_epochs):
            self.run_epoch(train_mode = 'combined')
            
            # self.save_model()
                
    def perform_ema(self, it):
        
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.models['encoder'].parameters(), self.models['encoder_teacher'].parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
            for param_dq, param_dk in zip(self.models['depth'].parameters(), self.models['depth_teacher'].parameters()):
                param_dk.data.mul_(m).add_((1 - m) * param_dq.detach().data)
            
    def test_model(self):
        # self.load_model()
        self.set_eval()
        
        # ind = 0
        
        for inputs in tqdm(self.val_loader):
        
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)
                
                if "depth_gt" in inputs:
                    dl, depth_errors = self.compute_depth_losses(inputs, outputs, losses)
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_errors
                    
                    image_prev = inputs[('image_aug', -1)].cpu().numpy()
                    image = inputs[('image_aug', 0)].cpu().numpy()
                    image_next = inputs[('image_aug', 1)].cpu().numpy()
                    
                    recons_from_prev = outputs[('color', -1)].cpu().numpy()
                    recons_from_next = outputs[('color', 1)].cpu().numpy()
                    
                    depth_pred = outputs[("disp", 0)].cpu().numpy()
                    orig_images = inputs[('original_image_path', 0)]
                    all_indexes = inputs[('index', 0)]
                    # print('all_indexes: ', all_indexes)
                    # print(all_indexes[0])
                    
                    # print('orign images path shape: ', orig_images.shape)
                    # print('recons_from_prev: ', recons_from_prev.shape)
                    # print('recons_from_next: ', recons_from_next.shape)
                    # print('image shape: ', inputs[('image_aug', -1)].cpu().numpy().shape)
                    # print('depth shape: ', depth_pred.shape)
                    # print('depth max; ', np.max(depth_pred))
                    
                    # for b in range(12):
                    #     prev_image_slice = np.transpose(image_prev[b, :, :, :], (1, 2, 0))
                    #     image_slice = np.transpose(image[b, :, :, :], (1, 2, 0))
                    #     next_image_slice = np.transpose(image_next[b, :, :, :], (1, 2, 0))
                    #     depth_slice = depth_pred[b, 0, :, :]

                    #     recons_prev_slice = np.transpose(recons_from_prev[b, :, :, :], (1, 2, 0))
                    #     recons_next_slice = np.transpose(recons_from_next[b, :, :, :], (1, 2, 0))
                        
                    #     depth_pred_slice = depth_pred[b, 0, :, :]
                        
                        
                    #     prev_image_slice = (prev_image_slice/np.max(prev_image_slice))*255
                    #     image_slice = (image_slice/np.max(image_slice))*255
                    #     next_image_slice = (next_image_slice/np.max(next_image_slice))*255
                        
                    #     recons_prev_slice = (recons_prev_slice/np.max(recons_prev_slice))*255
                    #     recons_next_slice = (recons_next_slice/np.max(recons_next_slice))*255
                        
                    #     orig_slice_path = orig_images[b]
                    #     orig_image = cv2.imread(orig_slice_path)
                        
                    #     ind = all_indexes[b].cpu().numpy()
                        
                    #     image_m1_name = 'image_m1\\original_m1_image_' + str(ind) + '.png'
                    #     image_name = 'image\\original_image_' + str(ind) + '.png'
                    #     image_p1_name = 'image_p1\\original_p1_image_' + str(ind) + '.png'
                        
                    #     reproj_m1_name = 'reprojected_m1\\reprojected_m1_image_' + str(ind) + '.png'
                    #     reproj_p1_name = 'reprojected_p1\\reprojected_p1_image_' + str(ind) + '.png'
                        
                    #     orig_image_name = 'original_image\\image_' + str(ind) + '.png'
                    #     depth_name = 'predicted_depth\\depth_' + str(ind) + '.png'

                    #     cv2.imwrite(os.path.join(self.save_folder, image_m1_name), prev_image_slice)
                    #     cv2.imwrite(os.path.join(self.save_folder, image_name), image_slice)
                    #     cv2.imwrite(os.path.join(self.save_folder, image_p1_name), next_image_slice)
                        
                    #     cv2.imwrite(os.path.join(self.save_folder, reproj_m1_name), recons_prev_slice)
                    #     cv2.imwrite(os.path.join(self.save_folder, reproj_p1_name), recons_next_slice)
                        
                    #     cv2.imwrite(os.path.join(self.save_folder, orig_image_name), orig_image)
                    #     cv2.imwrite(os.path.join(self.save_folder, depth_name), depth_slice)
                    #     # print('---saved---')
                    #     ind = ind + 1
        
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
            
            # print('final loss: ', all_loss, all_loss.double())
            self.model_optimizer.zero_grad()
            all_loss.double().backward()
            self.model_optimizer.step()
        
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
        
        val_loss = self.val()
        print('loss and latch: ', val_loss, self.val_latch)
        if val_loss < self.val_latch:
            self.save_model()
            self.val_latch = val_loss
            print('saved model and set latch value to : ', self.val_latch)
            
    def val(self):
        """Validate the model on a single minibatch
        """
        # print('entered val')
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
        
        for inputs in tqdm(self.val_loader):
            inin = inputs[('image_aug', 0)]
            if inin.shape[0] == self.batch_size:
                # print('input shape: ', inin.shape)
                with torch.no_grad():
                    outputs, losses = self.process_batch_val(inputs)
                    
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
        return losses['loss']
    
    def process_batch_val(self, inputs):
        
        if self.opt.pose_model_type == 'shared':
            all_colour_aug = None
            all_colour_aug = torch.cat([inputs[('image_aug', i)] for i in self.opt.frame_ids])
            
            all_features = self.models['initial'](all_colour_aug.cuda().float())
            all_features = self.models['encoder'](all_features)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            
            features ={}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
                
            outputs = self.models['depth'](features[0])
        
        else:
            print('separate encoders not implemented')
        
        if self.use_pose_net:
            outputs.update(self.predict_pose(inputs, features))
            
        self.generate_images_pred_single(inputs, outputs)
        losses = self.compute_losses_single(inputs, outputs)
        return outputs, losses
    
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
            for i in self.opt.frame_ids:
                image = inputs[('image_aug', i)]
                affinity = inputs[('affinity', i)][:, 0, :, :, :]
                combined = torch.cat([image, affinity], dim = 1)

                if all_colour_aug_aff == None:
                    all_colour_aug_aff = combined
                else:
                    all_colour_aug_aff = torch.cat([all_colour_aug_aff, combined])

            all_colour_aug = torch.cat([inputs[('image_aug', i)] for i in self.opt.frame_ids])
            
            all_features = self.models['initial'](all_colour_aug.cuda().float())
            all_features = self.models['encoder'](all_features)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            
            with torch.no_grad():
                all_features_teacher = self.models['initial_teacher'](all_colour_aug_aff.cuda().float())
                all_features_teacher = self.models['encoder_teacher'](all_features_teacher)
                all_features_teacher = [torch.split(f, self.opt.batch_size) for f in all_features_teacher]
            
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            
            teacher_features = {}
            for i, k in enumerate(self.opt.frame_ids):
                teacher_features[k] = [f[i] for f in all_features_teacher]
                
            outputs = self.models['depth'](features[0])
            with torch.no_grad():
                outputs_teacher = self.models['depth_teacher'](teacher_features[0])
        
        else:
            print('separate encoders not implemented')
        
        if self.use_pose_net:
            outputs.update(self.predict_pose(inputs, features))
            
        self.generate_images_pred(inputs, outputs, outputs_teacher)
        losses = self.compute_losses(inputs, outputs, outputs_teacher, features, teacher_features)
        
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
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != 's']
            
            axisangle, translation = self.models['pose'](pose_inputs)
            
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != 's':
                    outputs[('axisangle', f_i)] = axisangle
                    outputs[('translation', f_i)] = translation
                    outputs[('cam_T_cam', f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])
                    
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

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

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
                
                if self.use_affinity == True and self.val_mode == False:
                    gt_affinity = inputs[("affinity", frame_id)][:, 0, :, :, :].double()
                    outputs[("pred_affinity", frame_id)] = F.grid_sample(gt_affinity.cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                    
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id)] = \
                        inputs[("image", frame_id)]
    
    
        
    def generate_images_pred(self, inputs, outputs, outputs_teacher):
        for scale in self.opt.scales:
            disp = outputs[('disp', 0)]
            disp_teacher = outputs_teacher[('disp', 0)]
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp_teacher = F.interpolate(disp_teacher, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            _, depth_teacher = disp_to_depth(disp_teacher, self.opt.min_depth, self.opt.max_depth)
            
            outputs[("depth", 0)] = depth
            outputs_teacher[('depth', 0)] = depth_teacher

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    
                    axisangle = outputs[("axisangle", frame_id)]
                    translation = outputs[("translation", frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    
                    inv_depth_teacher = 1 / depth_teacher
                    mean_inv_depth_teacher = inv_depth_teacher.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                    
                    T_teacher = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth_teacher[:, 0], frame_id < 0)
                
                
                cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                cam_points_teacher = self.backproject_depth(depth_teacher, inputs[('inv_K')])
                
                pix_coords = self.project_3d(cam_points, inputs[("K")], T)
                pix_coords_teacher = self.project_3d(cam_points_teacher, inputs[("K")], T)

                outputs[("sample", frame_id)] = pix_coords
                outputs_teacher[('sample', frame_id)] = pix_coords_teacher

                outputs[("color", frame_id)] = F.grid_sample(inputs[("image", frame_id)].cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                outputs_teacher[("color", frame_id)] = F.grid_sample(inputs[("image", frame_id)].cuda(), outputs_teacher[("sample", frame_id)].double(), padding_mode="border")
                
                if self.use_affinity == True and self.val_mode == False:
                    # print('frame ID: ', frame_id)
                    gt_affinity = inputs[("affinity", frame_id)][:, 0, :, :, :].double()
                    outputs[("pred_affinity", frame_id)] = F.grid_sample(gt_affinity.cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                    
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id)] = \
                        inputs[("image", frame_id)]
                        
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        # print('In Loss Target: ', target.get_device())
        # print('In Loss Pred: ', pred.get_device())
        abs_diff = torch.abs(target.cuda() - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def compute_affinity_reprojection_loss(self, input_, t_seg, target, weight):
        
        # print('pred shape: ', input_.shape)
        # print('target shape: ', target.shape)
        # print('target unique: ', torch.unique(target))
        sumed_weight = torch.unsqueeze(torch.sum(target, dim = 1), dim = 1)
        # print('summed shape: ', sumed_weight.shape)
        # print('summed unique: ', torch.unique(sumed_weight))
        sumed_weight = torch.where(sumed_weight < 24, 1, 0)
        # print('final weight unique: ', torch.unique(sumed_weight))
        
        # rand_1 = torch.rand_like(target) < weight[1]
        # rand_1 = rand_1.float().cuda()
        # ones = torch.ones_like(target).cuda()
        # drop = torch.where(rand_1 == 1., input_, ones)
        # target = target * drop
        
        # t_seg = torch.unsqueeze(t_seg, dim = 1)
        # t_seg_in = torch.sum(t_seg[:, :-1], 1)[:, None] # ===============
        # t_seg_out = torch.ones_like(t_seg_in) - t_seg_in
    
        bce_loss = (input_ - target) ** 2
        # loss = (loss * t_seg_in * weight[0])
        # print('loss / weight shapes: ', loss.shape, sumed_weight.shape)
        # loss = loss * sumed_weight
        
        # print('input types: ', input_.dtype, target.dtype)
        huber = torch.nn.HuberLoss()
        loss = huber(input_, target.double()) * sumed_weight
        # print('BCE / Huber type: ', bce_loss.dtype, loss.dtype)
        # print('BCE loss: ', torch.mean(bce_loss))
        # print('Huber loss: ', torch.mean(loss))
        
        # print('-- loss unique: ', torch.unique(t_seg_in), torch.unique(loss))
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
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

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
    
    def compute_losses(self, inputs, outputs, outputs_teacher, features, teacher_features):
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
                pred_teacher = outputs_teacher[('color', frame_id)]
                
                rp = self.compute_reprojection_loss(pred, target)
                rp_st = self.compute_reprojection_loss(pred, pred_teacher.detach()) # --------------------------------------------------
                reprojection_losses.append(rp)
                reprojection_losses.append(rp_st*0.5) # ----------------------------------------------------
                # losses['reprojection'] = rp

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
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

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
            
            enc_dif_loss = huber(features[0][0], teacher_features[0][0].detach())*0.5 # ------------------------------------------------------------------
            total_loss += enc_dif_loss
            
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