import argparse
import mmcv
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint

# ==============================================================================
# [1] 辅助类：逻辑判断与拟合
# ==============================================================================
class IntrusionLogic:
    def __init__(self, roi_width_meters=4.0, voxel_size=[0.4, 0.4]):
        self.roi_width_meters = roi_width_meters
        self.voxel_size = voxel_size
        self.roi_width_px = roi_width_meters / voxel_size[1] 

    def fit_rail_lines(self, mask):
        """对分割 Mask 进行二次曲线拟合"""
        if mask is None: return None
        H, W = mask.shape
        ys, xs = np.where(mask > 0.5)
        if len(xs) < 10: return None
        try:
            # 拟合 col = a*row^2 + b*row + c
            coeffs = np.polyfit(ys, xs, 2)
            return coeffs 
        except:
            return None

    def check_intrusion(self, boxes, rail_coeffs, img_shape):
        """检查障碍物是否侵入轨道"""
        alarms = []
        if rail_coeffs is None: return alarms
        H, W = img_shape
        a, b, c = rail_coeffs
        
        for i, box in enumerate(boxes):
            # box: [col_center, row_center]
            bx, by = box[0], box[1] 
            rail_center_x = a * by**2 + b * by + c
            dist = abs(bx - rail_center_x)
            
            if dist < self.roi_width_px / 2:
                alarms.append({'idx': i, 'dist': dist, 'level': 'CRITICAL'})
        return alarms

# ==============================================================================
# [2] 绘图函数 (修复: 镜像 + 纵横比)
# ==============================================================================
def visualize_2x2(img, points, bev_mask, boxes_3d, scores, labels, out_path, pc_range):
    logic = IntrusionLogic()
    
    # [FIX 1] 调整画布比例，适应长条形 BEV (Height >> Width)
    fig, axes = plt.subplots(2, 2, figsize=(16, 20)) 
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    # --- [左上] Input RGB ---
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis('off')

    # --- [右上] LiDAR BEV Point Cloud ---
    bev_h, bev_w = 512, 224
    lidar_canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    
    x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    
    # 过滤范围
    mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & \
           (points[:, 1] > y_min) & (points[:, 1] < y_max)
    valid_pts = points[mask]
    
    # 映射坐标
    # x (前) -> row (0在顶部) => row = (1-nx)*H
    # y (左) -> col (0在左侧)
    # y范围: [-44.8, 44.8]. ny=0 (y=-44.8, 右侧), ny=1 (y=44.8, 左侧)
    # 图像坐标: col=0 (左侧), col=W (右侧)
    # [FIX 2] 镜像修正: 让 ny=0(右) 映射到 col=W(右), ny=1(左) 映射到 col=0(左)
    # 公式: col = (1 - ny) * W
    
    nx = (valid_pts[:, 0] - x_min) / (x_max - x_min)
    ny = (valid_pts[:, 1] - y_min) / (y_max - y_min)
    
    rows = ((1 - nx) * bev_h).astype(np.int32)
    cols = ((1 - ny) * bev_w).astype(np.int32) # 修正后的镜像映射
    
    rows = np.clip(rows, 0, bev_h-1)
    cols = np.clip(cols, 0, bev_w-1)
    
    lidar_canvas[rows, cols] = (0, 255, 0) 
    
    # 画 Box
    if boxes_3d is not None:
        centers = boxes_3d.gravity_center.numpy()
        # 对 Box 中心做同样的镜像映射
        ny_box = (centers[:, 1] - y_min) / (y_max - y_min)
        nx_box = (centers[:, 0] - x_min) / (x_max - x_min)
        
        ctx = ((1 - ny_box) * bev_w).astype(np.int32)
        cty = ((1 - nx_box) * bev_h).astype(np.int32)
        
        for k in range(len(centers)):
            if scores[k] > 0.25:
                cv2.circle(lidar_canvas, (ctx[k], cty[k]), 3, (255, 0, 0), -1)

    # [FIX 3] 强制 Equal Aspect Ratio
    axes[0, 1].imshow(lidar_canvas, aspect='equal') 
    axes[0, 1].set_title("LiDAR BEV (Corrected View)")
    axes[0, 1].axis('off')
    
    # --- [左下] Rail Segmentation ---
    seg_canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    if bev_mask is not None:
        if bev_mask.shape != (bev_h, bev_w):
            bev_mask = cv2.resize(bev_mask, (bev_w, bev_h))
        
        # [FIX 4] Mask 也要水平翻转以匹配 LiDAR
        bev_mask = cv2.flip(bev_mask, 1) 
            
        seg_canvas[bev_mask > 0.3] = (100, 100, 100)
        
        coeffs = logic.fit_rail_lines(bev_mask)
        if coeffs is not None:
            ys = np.arange(bev_h)
            a, b, c = coeffs
            xs = a * ys**2 + b * ys + c
            valid = (xs >= 0) & (xs < bev_w)
            pts = np.stack([xs[valid], ys[valid]], axis=1).astype(np.int32)
            cv2.polylines(seg_canvas, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 3) 
            title_suffix = "(Fit Success)"
        else:
            title_suffix = "(Fit Failed)"
    else:
        title_suffix = "(No Mask)"
        
    axes[1, 0].imshow(seg_canvas, aspect='equal')
    axes[1, 0].set_title(f"Rail Segmentation {title_suffix}")
    axes[1, 0].axis('off')
    
    # --- [右下] Safety Analysis ---
    safety_canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    
    if bev_mask is not None and coeffs is not None:
        ys = np.arange(bev_h)
        a, b, c = coeffs
        xs = a * ys**2 + b * ys + c
        
        width_px = logic.roi_width_px
        xs_left = xs - width_px / 2
        xs_right = xs + width_px / 2
        pts_left = np.stack([xs_left, ys], axis=1)
        pts_right = np.stack([xs_right, ys], axis=1)
        pts_poly = np.concatenate([pts_left, pts_right[::-1]]).astype(np.int32)
        
        cv2.fillPoly(safety_canvas, [pts_poly], (0, 100, 100)) 
        
    if boxes_3d is not None:
        # 使用修正后的坐标
        pixel_boxes = []
        for k in range(len(centers)):
            pixel_boxes.append([ctx[k], cty[k]])
            
        alarms = logic.check_intrusion(pixel_boxes, coeffs, (bev_h, bev_w)) if coeffs is not None else []
        alarm_indices = [a['idx'] for a in alarms]
        
        for k in range(len(centers)):
            if scores[k] > 0.25:
                color = (255, 0, 0) if k in alarm_indices else (0, 255, 0)
                cv2.circle(safety_canvas, (ctx[k], cty[k]), 5, color, -1)
                label = "CRITICAL" if k in alarm_indices else ""
                if label:
                    cv2.putText(safety_canvas, label, (ctx[k]+5, cty[k]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    axes[1, 1].imshow(safety_canvas, aspect='equal')
    axes[1, 1].set_title("Safety Analysis")
    axes[1, 1].axis('off')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ==============================================================================
# [3] 主函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default='vis_results_v2', help='Output directory')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    print(f"🚀 Starting Advanced Visualization (Mirror & Aspect Fixed)...")
    
    for i, data in enumerate(data_loader):
        if i >= 10: break 
        
        img_metas = data['img_metas'][0].data[0][0]
        if 'img_info' in img_metas: img_path = img_metas['img_info']['filename']
        else: continue
            
        raw_img = cv2.imread(img_path)
        points = data['points'][0].data[0][0].numpy()
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        
        boxes_3d = result.get('boxes_3d', None)
        scores_3d = result.get('scores_3d', None)
        labels_3d = result.get('labels_3d', None)
        bev_mask = result.get('bev_seg_mask', None)
        
        out_name = os.path.join(args.out_dir, f'vis_{os.path.basename(img_path)}')
        
        visualize_2x2(raw_img, points, bev_mask, boxes_3d, scores_3d, labels_3d, out_name, cfg.point_cloud_range)
        print(f"Saved {out_name}")

if __name__ == '__main__':
    main()