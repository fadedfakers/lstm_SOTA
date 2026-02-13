import os
import pickle
import numpy as np
import mmcv
import torch
import open3d as o3d
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets import PIPELINES
# 导入 Dataset
from data.sosdar_adapter import SOSDaRDataset 

# ==============================================================================
# 配置部分
# ==============================================================================
DATA_ROOT = '/root/autodl-tmp/FOD/data/'
INFO_PATH = os.path.join(DATA_ROOT, 'osdar23_infos_train.pkl') 
SAVE_DIR = os.path.join(DATA_ROOT, 'osdar23_gt_database')
SAVE_INFO_PATH = os.path.join(DATA_ROOT, 'osdar23_dbinfos_train.pkl') 

# ==============================================================================
# 稳健的点云加载器
# ==============================================================================
@PIPELINES.register_module()
class LoadPointsRobust(object):
    def __init__(self, load_dim=4, use_dim=4):
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        filename = results['pts_filename']
        points = None
        
        # 1. 尝试作为 .bin 读取 (Numpy)
        if filename.endswith('.bin'):
            try:
                points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            except Exception as e:
                pass 

        # 2. 尝试作为 .pcd 读取 (Open3D)
        elif filename.endswith('.pcd'):
            try:
                pcd = o3d.io.read_point_cloud(filename)
                points = np.asarray(pcd.points)
                if points.shape[1] == 3:
                    points = np.hstack([points, np.zeros((points.shape[0], 1))])
            except Exception as e:
                pass
        
        # 3. 失败处理
        if points is None:
            points = np.zeros((1, self.load_dim), dtype=np.float32)

        points = points[:, :self.use_dim]
        results['points'] = torch.from_numpy(points).float()
        return results

def create_osdar_gt_database():
    print(f"🚀 [Step 2] 生成 GT Database (Fix V5 - Auto Shape Adapt)...")
    
    pipeline = [
        dict(type='LoadPointsRobust', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    ]
    
    try:
        dataset = SOSDaRDataset(
            data_root=DATA_ROOT,
            ann_file=INFO_PATH,
            pipeline=pipeline,
            classes=['car', 'pedestrian', 'obstacle'],
            modality=dict(use_lidar=True, use_camera=True),
            box_type_3d='LiDAR',
            filter_empty_gt=False,
            test_mode=False
        )
    except Exception as e:
        print(f"❌ Dataset 初始化失败: {e}")
        return

    mmcv.mkdir_or_exist(SAVE_DIR)
    
    all_db_infos = dict()
    for cat in dataset.CLASSES:
        all_db_infos[cat] = []

    print(f"   准备处理 {len(dataset)} 帧...")
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i in range(len(dataset)):
        try:
            data = dataset.prepare_train_data(i)
            if data is None: 
                prog_bar.update()
                continue
            
            # 1. 解包 Points
            points = data['points']
            if isinstance(points, torch.Tensor):
                points = points.numpy()
            elif hasattr(points, 'tensor'): 
                points = points.tensor.numpy()
            
            # 2. 解包 BBoxes
            gt_bboxes_3d_obj = data['gt_bboxes_3d']
            if hasattr(gt_bboxes_3d_obj, 'tensor'):
                gt_bboxes_3d = gt_bboxes_3d_obj.tensor.numpy()
            else:
                gt_bboxes_3d = gt_bboxes_3d_obj

            # 3. 解包 Labels
            gt_labels_3d = data['gt_labels_3d']
            if isinstance(gt_labels_3d, torch.Tensor):
                gt_labels_3d = gt_labels_3d.numpy()
            
            gt_names = [dataset.CLASSES[l] for l in gt_labels_3d]
            
            # [DEBUG] 打印第一帧
            if i == 0:
                print("\n" + "="*50)
                print(f"🔍 [DEBUG Frame 0]")
                print(f"   Points: {points.shape}")
                print(f"   Boxes:  {gt_bboxes_3d.shape}")
                print("="*50 + "\n")

            # 裁剪逻辑
            if gt_bboxes_3d.shape[0] > 0 and points.shape[0] > 10:
                
                # [核心修复 1] 切片 9维 -> 7维
                gt_bboxes_3d_geom = gt_bboxes_3d[:, :7]
                
                # [核心修复 2] 搬运到 GPU 
                points_cuda = torch.from_numpy(points[:, :3]).cuda().float()
                boxes_cuda = torch.from_numpy(gt_bboxes_3d_geom).cuda().float()
                
                # 计算
                gt_boxes_lidar = LiDARInstance3DBoxes(boxes_cuda, box_dim=7)
                point_indices = gt_boxes_lidar.points_in_boxes(points_cuda)
                
                # 搬回 CPU
                point_indices = point_indices.cpu()
                
                # [核心修复 3 - V5] 自动维度适配
                # 有的版本返回 (N, M)，有的返回 (N,)
                is_2d_mask = (point_indices.dim() == 2)
                
                for j in range(gt_bboxes_3d.shape[0]):
                    box_name = gt_names[j]
                    
                    if is_2d_mask:
                        # 2D 模式: (N, M) -> 取第 j 列
                        box_point_mask = point_indices[:, j].bool().numpy()
                    else:
                        # 1D 模式: (N,) -> 取值等于 j 的点
                        box_point_mask = (point_indices == j).numpy()
                    
                    box_points = points[box_point_mask]
                    
                    if box_points.shape[0] < 5: continue
                        
                    # 坐标归一化
                    box_center = gt_bboxes_3d[j][0:3]
                    box_points[:, :3] -= box_center
                    
                    info = dataset.data_infos[i]
                    filename = f"{info.get('scene_id','unk')}_{info['sample_idx']}_{box_name}_{j}.bin"
                    filepath = os.path.join(SAVE_DIR, filename)
                    box_points.tofile(filepath)
                    
                    db_info = {
                        'name': box_name,
                        'path': os.path.join('osdar23_gt_database', filename),
                        'image_idx': info['sample_idx'],
                        'gt_idx': j,
                        'box3d_lidar': gt_bboxes_3d[j],
                        'num_points_in_gt': box_points.shape[0],
                        'difficulty': 0,
                    }
                    all_db_infos[box_name].append(db_info)
                    
        except Exception as e:
            if i == 0: print(f"   ❌ 处理出错: {e}")
            pass
            
        prog_bar.update()

    print(f"\n💾 保存数据库索引到: {SAVE_INFO_PATH}")
    with open(SAVE_INFO_PATH, 'wb') as f:
        pickle.dump(all_db_infos, f)
        
    for cat, infos in all_db_infos.items():
        print(f"   -> {cat}: {len(infos)} 个样本")

if __name__ == '__main__':
    create_osdar_gt_database()