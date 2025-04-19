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
        
        self.opt = options
        self.cfg = cfg
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.num_scales = 1
        
        self.models = {}
        self.parameters_to_train = []
        self.models['encoder'] = networks.ResnetEncoder_pre(num_layers = 18, pretrained = False) # options [18, 34, 50, 101, 152]
        self.models['encoder'].to(self.device)
        # self.parameters_to_train += list(self.models['encoder'].parameters())
        
        self.models['depth'] = networks.DepthDecoder(scales = self.opt.scales)
        self.models['depth'].to(self.device)
        self.parameters_to_train += list(self.models['depth'].parameters())
        
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
        
        self.record_saves = 'D:\\depth_estimation_implementation\\all_saves\\record_saves'
        self.file_path = os.path.join(self.record_saves, 'save_training_record.txt')
        
        self.save_folder = save_folder
        
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
        path = 'D:\\depth_estimation_implementation\\all_saves\\weight_saves\\with_affinity'
        encoder_name = 'Encoder.pth'
        depth_name = 'Depth.pth'
        pose_name = 'Pose.pth'
        torch.save(self.models['encoder'].state_dict(), os.path.join(path, encoder_name))
        torch.save(self.models['depth'].state_dict(), os.path.join(path, depth_name))
        torch.save(self.models['pose'].state_dict(), os.path.join(path, pose_name))
    
    def train(self):
        self.epoch = 0
        self.step = 0
        self.load_model()
        
        #print('Starting training for ' + str(self.opt.num_epochs) + ' epochs')
        #print('Save frequency: ', self.opt.save_frequency)
        for self.epoch in range(0, self.opt.num_epochs):
            self.run_epoch()
            #if (self.epoch + 1) % self.opt.save_fequency == 0:
            self.save_model()
                
    def test_model(self):
        self.load_model()
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
                    print('all_indexes: ', all_indexes)
                    print(all_indexes[0])
                    
                    # print('orign images path shape: ', orig_images.shape)
                    # print('recons_from_prev: ', recons_from_prev.shape)
                    # print('recons_from_next: ', recons_from_next.shape)
                    # print('image shape: ', inputs[('image_aug', -1)].cpu().numpy().shape)
                    print('depth shape: ', depth_pred.shape)
                    print('depth max; ', np.max(depth_pred))
                    
                    for b in range(12):
                        prev_image_slice = np.transpose(image_prev[b, :, :, :], (1, 2, 0))
                        image_slice = np.transpose(image[b, :, :, :], (1, 2, 0))
                        next_image_slice = np.transpose(image_next[b, :, :, :], (1, 2, 0))
                        depth_slice = depth_pred[b, 0, :, :]
                        # depth_slice = (depth_slice / np.max(depth_slice))*255
                        
                        
                        recons_prev_slice = np.transpose(recons_from_prev[b, :, :, :], (1, 2, 0))
                        recons_next_slice = np.transpose(recons_from_next[b, :, :, :], (1, 2, 0))
                        
                        depth_pred_slice = depth_pred[b, 0, :, :]
                        
                        
                        prev_image_slice = (prev_image_slice/np.max(prev_image_slice))*255
                        image_slice = (image_slice/np.max(image_slice))*255
                        next_image_slice = (next_image_slice/np.max(next_image_slice))*255
                        
                        recons_prev_slice = (recons_prev_slice/np.max(recons_prev_slice))*255
                        recons_next_slice = (recons_next_slice/np.max(recons_next_slice))*255
                        
                        orig_slice_path = orig_images[b]
                        orig_image = cv2.imread(orig_slice_path)
                        
                        # plt.imshow(orig_image)
                        # plt.show()
                        # plt.imshow(depth_slice)
                        # plt.show()
                        
                        # print('all shapes: ', image_slice.shape, recons_prev_slice.shape, recons_next_slice.shape)
                        # print('image max: ', np.max(image_slice))
                        # print('recons prev max: ', np.max(recons_prev_slice))
                        
                        ind = all_indexes[b].cpu().numpy()
                        
                        image_m1_name = 'image_m1\\original_m1_image_' + str(ind) + '.png'
                        image_name = 'image\\original_image_' + str(ind) + '.png'
                        image_p1_name = 'image_p1\\original_p1_image_' + str(ind) + '.png'
                        
                        reproj_m1_name = 'reprojected_m1\\reprojected_m1_image_' + str(ind) + '.png'
                        reproj_p1_name = 'reprojected_p1\\reprojected_p1_image_' + str(ind) + '.png'
                        
                        orig_image_name = 'original_image\\image_' + str(ind) + '.png'
                        depth_name = 'predicted_depth\\depth_' + str(ind) + '.png'
                        
                        # print(os.path.join(self.save_folder, image_m1_name))
                        
                        cv2.imwrite(os.path.join(self.save_folder, image_m1_name), prev_image_slice)
                        cv2.imwrite(os.path.join(self.save_folder, image_name), image_slice)
                        cv2.imwrite(os.path.join(self.save_folder, image_p1_name), next_image_slice)
                        
                        cv2.imwrite(os.path.join(self.save_folder, reproj_m1_name), recons_prev_slice)
                        cv2.imwrite(os.path.join(self.save_folder, reproj_p1_name), recons_next_slice)
                        
                        cv2.imwrite(os.path.join(self.save_folder, orig_image_name), orig_image)
                        cv2.imwrite(os.path.join(self.save_folder, depth_name), depth_slice)
                        # print('---saved---')
                        ind = ind + 1
        
    def run_epoch(self):
        self.set_train()
        # reprojection = []
        abs_req = []
        sq_req = []
        rmse_req = []
        rmse_log_req = []
        epoch_loss = []
        batch_idx = 0
        for inputs in tqdm(self.train_loader):
            all_loss = 0
            # print('Epoch / itteration: ', self.epoch, batch_idx)
            batch_idx = batch_idx + 1
            outputs, losses = self.process_batch(inputs)
            # print('processing done')
            # self.model_optimizer.zero_grad()
            # losses['loss'].backward()
            # self.model_optimizer.step()
            # reprojection.append(np.mean(losses['reprojection'].detach().cpu().numpy()))
            # print('reprojection shape: ', losses['reprojection'].detach().cpu().numpy().shape, np.mean(losses['reprojection'].detach().cpu().numpy()))
            
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
                all_loss = losses['loss']
            
            self.model_optimizer.zero_grad()
            all_loss.backward()
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
        
        self.val()
            
            ###### continue #####
            
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

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            
            # reprojection.append(np.mean(losses['reprojection'].detach().cpu().numpy()))

            if "depth_gt" in inputs:
                dl, depth_errors = self.compute_depth_losses(inputs, outputs, losses)
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_errors
                abs_req.append(abs_rel.detach().cpu().numpy())
                sq_req.append(sq_rel.detach().cpu().numpy())
                rmse_req.append(rmse.detach().cpu().numpy())
                rmse_log_req.append(rmse_log.detach().cpu().numpy())

            #self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        
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
    
    def process_batch(self, inputs):
        
        if self.opt.pose_model_type == 'shared':
            all_colour_aug = torch.cat([inputs[('image_aug', i)] for i in self.opt.frame_ids])
            
            # for i in self.opt.frame_ids:
            #     print('individual images: ', inputs[('image', i)].shape)
            # print('model input shape: ', all_colour_aug.shape)
            
            all_features = self.models['encoder'](all_colour_aug.cuda().float())
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            
            features ={}
            for i, k in enumerate(self.opt.frame_ids):
                #print('i, k: ', i, k)
                features[k] = [f[i] for f in all_features]
                
            outputs = self.models['depth'](features[0])
        
        else:
            print('separate encoders not implemented')
        
        if self.use_pose_net:
            outputs.update(self.predict_pose(inputs, features))
            
        self.generate_images_pred(inputs, outputs)
        # print('pred images generated')
        losses = self.compute_losses(inputs, outputs)
        # print('loss calculated')
        
        return outputs, losses
        
    def predict_pose(self, inputs, features):
        
        outputs = {}
        
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each sources frame via a 
            # separate forward pass through the pose network.
            
            # select what features the pose network takes as input
            if self.opt.pose_model_type == 'shared':
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs['image_aug', f_i] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:
                if f_i != 's':
                    # To maintain order we always pass frames in temporal order
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
                    
                    # Invert the matrix if the frame id in negative
                    outputs[('cam_T_cam', f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert = (f_i < 0))
                    
        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ['separate_resnet', 'posecnn']:
                pose_inputs = torch.cat(
                    [inputs[('image_aug', i)] for i in self.opt.frame_ids if i != 's'], 1)
                
                if self.opt.pose_model_type == 'separate_resnet':
                    pose_inputs = [self.model['pose_encoder'](pose_inputs)]
                    
            elif self.opt.pose_model_type == 'shared':
                # print('line 242')
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != 's']
            
            axisangle, translation = self.models['pose'](pose_inputs)
            
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != 's':
                    outputs[('axisangle', f_i)] = axisangle
                    outputs[('translation', f_i)] = translation
                    outputs[('cam_T_cam', f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])
                    
        return outputs
                
        
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        #print(outputs)
        for scale in self.opt.scales:
            disp = outputs[('disp', 0)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                #print(disp)
                # print('disp shape: ', disp.shape)
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            #print('reg depth shape: ', depth.shape)
            outputs[("depth", 0)] = depth

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

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                
                # print('input depth shape: ', depth.shape)
                # print('k: ', inputs[("K")].shape)
                # print('inv k: ', inputs[("inv_K")].shape)
                cam_points = self.backproject_depth(depth, inputs[("inv_K")])
                # print('cam points: ', cam_points.shape)
                pix_coords = self.project_3d(cam_points, inputs[("K")], T)

                outputs[("sample", frame_id)] = pix_coords
                # print('added colour id: ', frame_id)
                # print('before shape: ', inputs[("image", frame_id)].shape, outputs[("sample", frame_id)].shape)
                outputs[("color", frame_id)] = F.grid_sample(inputs[("image", frame_id)].cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                if self.use_affinity == True and self.val_mode == False:
                    # print('frame ID: ', frame_id)
                    gt_affinity = inputs[("affinity", frame_id)][:, 0, :, :, :].double()
                    # print('input affinity: ', gt_affinity.shape)
                    # print('output affinity: ', outputs[("sample", frame_id)].shape)
                    outputs[("pred_affinity", frame_id)] = F.grid_sample(gt_affinity.cuda(), outputs[("sample", frame_id)].double(), padding_mode="border")
                    
                # print('output device: ', outputs[("color", frame_id)].get_device())
                
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
        
        if (input_.cuda() == target.cuda()).all():
            print('================They are the same in the loss')

        rand_1 = torch.rand_like(target) < weight[1]
        rand_1 = rand_1.float().cuda()
        ones = torch.ones_like(target).cuda()
        drop = torch.where(rand_1 == 1., input_, ones)
        target = target * drop
        
        t_seg = torch.unsqueeze(t_seg, dim = 1)
        t_seg_in = torch.sum(t_seg[:, :-1], 1)[:, None] # ===============
        t_seg_out = torch.ones_like(t_seg_in) - t_seg_in
    
        loss = (input_ - target) ** 2
        # loss = (loss * t_seg_in * weight[0])
        
        # print('-- loss unique: ', torch.unique(t_seg_in), torch.unique(loss))
        return torch.mean(loss)
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            affinity_loss = 0

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", 0)]
            color = inputs[("image", 0)]
            target = inputs[("image", 0)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id)]
                # print('pred: ', pred.shape)
                # print('target: ', target.shape)
                rp = self.compute_reprojection_loss(pred, target)
                reprojection_losses.append(rp)
                # losses['reprojection'] = rp

                if self.use_affinity == True and self.val_mode == False:
                    aff_calc_weight = [1.5, 0.5]
                    affinity_loss += self.compute_affinity_reprojection_loss(outputs[("pred_affinity", frame_id)], inputs['segmentation', 0].cuda(), inputs['affinity', 0][:, 0, :, :, :].cuda(), aff_calc_weight)
                    # print('affinity reprojection: ', l_aff.shape)
                    
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
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
            
            elif self.opt.predictive_mask:
                # use the predicted mask
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
            loss += affinity_loss
            # print('Affinity mean: ', affinity_loss.mean())

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # print(norm_disp.shape, color.shape)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
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
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        # print('outputs shape: ', depth_pred.shape)
        # print('gt depth shape: ', depth_gt.shape)
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask
        #print('depth gt: ', depth_gt.shape)
        #print('depth pred: ', depth_pred.shape)
        #print('mask: ', mask.shape)
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        # depth_errors = compute_depth_errors(depth_gt.cuda(), depth_pred)
        # print('before depth gt shape: ', depth_gt.shape)
        # print('before depth pred shape: ', depth_pred.shape)
        depth_gt.requires_grad_()
        depth_pred.requires_grad_()
        # print('gt grad: ', depth_gt.requires_grad)
        # print('pred grad: ', depth_pred.requires_grad)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(depth_gt.cuda(), depth_pred)
        depth_errors = abs_rel + sq_rel + rmse + rmse_log + a1 + a2 + a3

        # for i, metric in enumerate(self.depth_metric_names):
        #     losses[metric] = np.array(depth_errors[i].cpu())
        return depth_errors, [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]