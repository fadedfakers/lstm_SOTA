import argparse
import mmcv
import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes

# ====================================================================
# [1] 辅助函数
# ====================================================================
def render_rail_bev_from_polys(polys, pc_range, canvas_size=(512, 224), thickness=3):
    H, W = canvas_size[0], canvas_size[1]
    mask = np.zeros((H, W), dtype=np.uint8)
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    valid_polys = []
    if isinstance(polys, list):
        for p in polys:
            if isinstance(p, np.ndarray) and p.ndim == 2: valid_polys.append(p)
    elif isinstance(polys, np.ndarray) and polys.ndim == 3:
        valid_polys = [p for p in polys]
        
    for poly in valid_polys:
        if len(poly) < 2: continue
        pts_img = []
        for pt in poly:
            nx = (pt[0] - x_min) / (x_max - x_min)
            ny = (pt[1] - y_min) / (y_max - y_min)
            px = int(ny * W)
            py = int((1 - nx) * H)
            pts_img.append([px, py])
            
        if len(pts_img) > 1:
            pts_img = np.array(pts_img, dtype=np.int32)
            cv2.polylines(mask, [pts_img], isClosed=False, color=1, thickness=thickness)
    return mask

def process_pred_mask(pred_mask, target_size=(512, 224)):
    if pred_mask is None:
        return np.zeros(target_size, dtype=np.uint8)
    if pred_mask.shape != target_size:
        pred_mask = cv2.resize(pred_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return (pred_mask > 0.5).astype(np.uint8)

# ====================================================================
# [2] 主逻辑
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate Detection & Segmentation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval', type=str, nargs='+', default=['mAP'], help='eval metrics')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg.data.test, 'test_mode'):
        cfg.data.test.test_mode = True 
    
    print("🔍 [Step 1] 初始化数据集...")
    dataset = build_dataset(cfg.data.test)
    
    print("🛠️ [FIX] Patching dataset annotations...")
    for i in range(len(dataset.data_infos)):
        info = dataset.data_infos[i]
        if 'annos' not in info: info['annos'] = {}
        annos = info['annos']
        
        gt_bboxes = annos.get('gt_bboxes_3d', np.zeros((0, 7)))
        if len(gt_bboxes) > 0 and gt_bboxes.shape[1] > 7:
            gt_bboxes = gt_bboxes[:, :7]
        annos['gt_bboxes_3d'] = gt_bboxes
        
        if 'gt_labels_3d' in annos:
            annos['class'] = annos['gt_labels_3d']
        else:
            annos['class'] = np.zeros(0, dtype=int)
            annos['gt_labels_3d'] = np.zeros(0, dtype=int)
            
        if not isinstance(gt_bboxes, LiDARInstance3DBoxes):
            if isinstance(gt_bboxes, list): gt_bboxes = np.array(gt_bboxes)
            if gt_bboxes.shape[0] > 0:
                annos['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
            else:
                annos['gt_bboxes_3d'] = LiDARInstance3DBoxes(np.zeros((0, 7)), box_dim=7, origin=(0.5, 0.5, 0.5))

        annos['gt_boxes_upright_depth'] = annos['gt_bboxes_3d'].tensor.numpy()
        annos['gt_num'] = len(gt_bboxes)

    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    print("🔍 [Step 2] 构建模型...")
    # [核心修复] 去掉 test_cfg 参数，防止冲突
    model = build_model(cfg.model) 
    
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    seg_tp, seg_fp, seg_fn = 0, 0, 0
    results_list = []
    
    pc_range = cfg.point_cloud_range
    bev_h, bev_w = cfg.grid_size[0], cfg.grid_size[1] 
    canvas_size = (bev_h, bev_w)

    print(f"🚀 [Step 3] 开始推理与评估 (共 {len(dataset)} 帧)...")
    
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            if boxes.tensor.shape[1] > 7:
                new_tensor = boxes.tensor[:, :7].clone()
                result['boxes_3d'] = LiDARInstance3DBoxes(new_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
                result['scores_3d'] = result['scores_3d']
                result['labels_3d'] = result['labels_3d']
        results_list.append(result)
        
        gt_info = dataset.get_ann_info(i)
        gt_polys = gt_info.get('gt_poly_3d', [])
        if hasattr(gt_polys, 'data'): gt_polys = gt_polys.data
            
        mask_gt = render_rail_bev_from_polys(gt_polys, pc_range, canvas_size, thickness=5)
        mask_pred_raw = result.get('bev_seg_mask', None)
        mask_pred = process_pred_mask(mask_pred_raw, target_size=canvas_size)
        
        intersection = np.logical_and(mask_pred, mask_gt).sum()
        seg_tp += intersection
        seg_fp += (mask_pred.sum() - intersection)
        seg_fn += (mask_gt.sum() - intersection)

    print("\n" + "="*50)
    print("📊 FINAL EVALUATION REPORT")
    print("="*50)
    
    print("\n[1] Object Detection Results:")
    try:
        dataset.evaluate(results_list, metric=['mAP'])
    except Exception as e:
        print(f"⚠️ Detection evaluation failed: {e}")

    rail_iou = seg_tp / (seg_tp + seg_fp + seg_fn + 1e-6)
    print("\n[2] Rail Segmentation Results:")
    print(f"   +----------------+---------+")
    print(f"   | Metric         | Value   |")
    print(f"   +----------------+---------+")
    print(f"   | Rail BEV mIoU  | {rail_iou:.4f}  |")
    print(f"   +----------------+---------+")
    print("="*50)

if __name__ == '__main__':
    main()