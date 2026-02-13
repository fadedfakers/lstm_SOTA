import mmcv
import numpy as np
import os
import cv2
import torch
import open3d as o3d
from mmdet.datasets import DATASETS, PIPELINES
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.core.points import LiDARPoints
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.evaluation import indoor_eval

# ====================================================================
# [1] 工具函数
# ====================================================================
def parse_osdar23_calibration(calib_file, target_camera='rgb_center'):
    if not os.path.exists(calib_file): return np.eye(4)
    with open(calib_file, 'r') as f: lines = f.readlines()
    intrinsics, extrinsics = np.eye(3), np.eye(4)
    found_sensor, idx = False, 0
    while idx < len(lines):
        line = lines[idx].strip(); idx += 1
        if line.startswith("data_folder:"): found_sensor = (line.split(":")[1].strip() == target_camera)
        if found_sensor:
            if line.startswith("camera_matrix:"):
                matrix_str = line.split("[")[1]
                while "]" not in matrix_str: matrix_str += lines[idx].strip(); idx += 1
                values = [float(x) for x in matrix_str.replace(']', '').replace(';', ',').split(',') if x.strip()]
                if len(values) == 9: intrinsics = np.array(values).reshape(3, 3)
            if line.startswith("combined homogenous transform:"):
                matrix_str = ""
                while "]" not in matrix_str: matrix_str += lines[idx].strip(); idx += 1
                if "[" in matrix_str: matrix_str = matrix_str.split("[")[1]
                values = [float(x) for x in matrix_str.replace(']', '').replace(';', ',').split(',') if x.strip()]
                if len(values) == 16: extrinsics = np.array(values).reshape(4, 4); break
    R_rect = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]], dtype=np.float32)
    view_pad = np.eye(4); view_pad[:3, :3] = intrinsics
    return view_pad @ R_rect @ extrinsics

# ====================================================================
# [2] Pipeline
# ====================================================================
@PIPELINES.register_module()
class FormatPoly(object):
    def __init__(self, bev_size=(224, 512), pc_range=[0, -44.8, -5, 204.8, 44.8, 10]):
        self.bev_size = bev_size; self.pc_range = pc_range
    def __call__(self, results):
        if 'gt_poly_3d' in results:
            polys = results['gt_poly_3d']
            W, H = self.bev_size; mask = np.zeros((H, W), dtype=np.float32)
            x_min, y_min, x_max, y_max = self.pc_range[0], self.pc_range[1], self.pc_range[3], self.pc_range[4]
            valid_polys = []
            if isinstance(polys, list):
                for p in polys:
                    if isinstance(p, np.ndarray):
                        if p.ndim == 2 and p.shape[0] > 1: valid_polys.append(p)
                        elif p.ndim == 1 and p.shape[0] > 3: valid_polys.append(p.reshape(-1, 3))
            elif isinstance(polys, np.ndarray):
                valid_polys = [p for p in polys] if polys.ndim == 3 else [polys] if polys.ndim == 2 else []
            for poly in valid_polys:
                if len(poly) < 2: continue
                pts_img = []
                for pt in poly:
                    nx = (pt[0] - x_min) / (x_max - x_min); ny = (pt[1] - y_min) / (y_max - y_min)
                    if 0 <= nx <= 1 and 0 <= ny <= 1: pts_img.append([int(ny * W), int((1 - nx) * H)])
                if len(pts_img) > 1: cv2.polylines(mask, [np.array(pts_img, dtype=np.int32)], False, 1, 3)
            results['gt_masks'] = DC(torch.from_numpy(mask).float().unsqueeze(0), stack=True)
        return results

@PIPELINES.register_module()
class LoadSOSDaRPCD(object):
    def __init__(self, load_dim=4, use_dim=4): self.load_dim = load_dim; self.use_dim = use_dim
    def __call__(self, results):
        try:
            pcd = o3d.io.read_point_cloud(results['pts_filename'])
            points = np.asarray(pcd.points)
            if points.shape[1] == 3: points = np.hstack([points, np.zeros((points.shape[0], 1))])
            results['points'] = LiDARPoints(points.astype(np.float32), points_dim=points.shape[-1], attribute_dims=None)
            results['pts_fields'] = ['x', 'y', 'z', 'intensity'][:self.use_dim]
        except: results['points'] = LiDARPoints(np.zeros((1, self.use_dim), dtype=np.float32), points_dim=self.use_dim)
        return results

# ====================================================================
# [3] Dataset (Final V11 - 终极修复)
# ====================================================================
@DATASETS.register_module()
class SOSDaRDatasetV2(Custom3DDataset):
    CLASSES = ('car', 'pedestrian', 'obstacle')
    def __init__(self, data_root, ann_file, pipeline=None, classes=None, modality=None, box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes, modality=modality, box_type_3d=box_type_3d, filter_empty_gt=filter_empty_gt, test_mode=test_mode)

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(pts_filename=info['lidar_path'], sample_idx=info['sample_idx'], img_prefix=None)
        if info.get('img_path'): input_dict['img_info'] = dict(filename=info['img_path'])
        input_dict['lidar2img'] = parse_osdar23_calibration(info['calib_path']) if info.get('calib_path') else np.eye(4)
        input_dict['box_type_3d'] = self.box_type_3d; input_dict['box_mode_3d'] = self.box_mode_3d
        if not self.test_mode:
            annos = self.get_ann_info(index); input_dict['ann_info'] = annos
            if annos:
                input_dict['gt_bboxes_3d'] = annos['gt_bboxes_3d']
                input_dict['gt_labels_3d'] = annos['gt_labels_3d']
                input_dict['gt_poly_3d'] = annos.get('gt_poly_3d', [])
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = info['annos'].get('gt_bboxes_3d', np.zeros((0, 7), dtype=np.float32))
        gt_labels_3d = info['annos'].get('gt_labels_3d', np.zeros((0,), dtype=np.long))
        
        # 兼容性修复：智能判断 Box 类型
        if isinstance(gt_bboxes_3d, LiDARInstance3DBoxes):
            gt_bboxes_3d_obj = gt_bboxes_3d
        else:
            if len(gt_bboxes_3d) > 0:
                if gt_bboxes_3d.shape[1] == 7: 
                    gt_bboxes_3d = np.hstack([gt_bboxes_3d, np.zeros((gt_bboxes_3d.shape[0], 2))])
                gt_bboxes_3d_obj = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
            else:
                gt_bboxes_3d_obj = LiDARInstance3DBoxes(np.zeros((0, 9), dtype=np.float32), box_dim=9, origin=(0.5, 0.5, 0.5))
        
        ann_info = dict(
            gt_bboxes_3d=gt_bboxes_3d_obj,
            gt_labels_3d=gt_labels_3d,
            gt_poly_3d=info['annos'].get('gt_poly_3d', []),
            gt_num=len(gt_bboxes_3d_obj),
            gt_boxes_upright_depth=gt_bboxes_3d_obj.tensor.numpy(),
            index=index
        )
        ann_info['class'] = gt_labels_3d 
        return ann_info

    def _convert_to_7dim(self, box_obj):
        if box_obj.tensor.shape[1] > 7:
            new_tensor = box_obj.tensor[:, :7].clone()
            new_box_obj = LiDARInstance3DBoxes(new_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
            return new_box_obj
        return box_obj

    def evaluate(self, results, **kwargs):
        gt_annos = [self.get_ann_info(i) for i in range(len(self.data_infos))]
        
        for i in range(len(gt_annos)):
            old_box = gt_annos[i]['gt_bboxes_3d']
            gt_annos[i]['gt_bboxes_3d'] = self._convert_to_7dim(old_box)
            gt_annos[i]['gt_boxes_upright_depth'] = gt_annos[i]['gt_bboxes_3d'].tensor.numpy()

        for i in range(len(results)):
            if 'boxes_3d' in results[i]:
                old_box = results[i]['boxes_3d']
                results[i]['boxes_3d'] = self._convert_to_7dim(old_box)

        label2cat = {i: cat for i, cat in enumerate(self.CLASSES)}
        
        # [核心修复 V11]
        # 必须先从 kwargs 里把 'label2cat' 和 'metric' 弹出来！
        # 否则 kwargs 传给 indoor_eval 时会带着它们，引发 "multiple values" 错误
        kwargs.pop('label2cat', None)
        kwargs.pop('metric', None) # <--- 这行就是为了解决 TypeError 加上的！
        
        # 强制指定 metric 为浮点数列表
        metric = [0.25, 0.5]
            
        box_mode_3d = self.box_mode_3d if self.box_mode_3d is not None else Box3DMode.LIDAR
        box_type_3d = LiDARInstance3DBoxes 
        
        return indoor_eval(
            gt_annos=gt_annos,
            dt_annos=results,
            label2cat=label2cat,
            metric=metric,
            box_mode_3d=box_mode_3d,
            box_type_3d=box_type_3d, 
            **kwargs
        )