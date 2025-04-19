import numpy as np
from layers import disp_to_depth

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate_depth(outputs, inputs, opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    
    pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
    pred_disp = pred_disp.cpu()[:, 0].numpy()
    
    pred_depth = 1 / pred_disp
    
    gt_depth = inputs['depth_gt']
    mask = gt_depth > 0
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    
    pred_depth *= opt.pred_depth_scale_factor
    
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    
    depth_errors = compute_errors(gt_depth, pred_depth)
    return depth_errors