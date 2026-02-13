_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py',
    './_base_/default_runtime.py'
]

# ==========================================================
# [Phase 2.6: Operation Head Reset] - FRESH START
# ==========================================================

point_cloud_range = [0, -44.8, -5, 204.8, 44.8, 10]
voxel_size = [0.4, 0.4, 0.2] 
grid_size = [512, 224, 75] 

dataset_type = 'SOSDaRDatasetV2' 
data_root = '/root/autodl-tmp/FOD/data/'
class_names = ['car', 'pedestrian', 'obstacle'] 

input_modality = dict(use_lidar=True, use_camera=True)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# 数据库采样配置
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'osdar23_dbinfos_train.pkl', 
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(car=5, pedestrian=5, obstacle=5)
    ),
    classes=class_names,
    sample_groups=dict(
        car=15, 
        pedestrian=10, 
        obstacle=30 
    )
)

train_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='FormatPoly', bev_size=(224, 512), pc_range=point_cloud_range), 
    dict(type='Collect3D', 
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks'], 
         meta_keys=['pts_filename', 'img_prefix', 'img_info', 'lidar2img',
                    'sample_idx', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                    'box_mode_3d', 'box_type_3d'])
]

test_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', 
                 keys=['points', 'img'],
                 meta_keys=['pts_filename', 'img_prefix', 'img_info', 'lidar2img',
                            'sample_idx', 'pcd_horizontal_flip', 
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d'])
        ])
]

data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=4, 
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'osdar23_infos_train.pkl', 
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        filter_empty_gt=False 
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'osdar23_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'osdar23_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

# [核心修改 3] 学习率策略重置 (重要！)
# 因为 Head 是从零开始学的，不能用太小的 LR，恢复为 0.001
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)

runner = dict(type='EpochBasedRunner', max_epochs=24)

# 恢复完整的 Warmup 和 Step 策略
lr_config = dict(
    _delete_=True, # 覆盖 _base_ 里的配置
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22] 
)

evaluation = dict(interval=1)

custom_imports = dict(imports=['data.sosdar_adapter', 'models'], allow_failed_imports=False)

model = dict(
    type='RailFusionNet', 
    # 注意：这里虽然写了 init_cfg，但 load_from 优先级更高
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/sosdar_geometry/phase1_best.pth'),
    voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000) 
    ),
    voxel_encoder=dict(
        _delete_=True,
        type='HardSimpleVFE',
        num_features=4, 
    ),
    middle_encoder=dict(
        _delete_=True,
        type='SparseEncoder',
        in_channels=4, 
        sparse_shape=[75, 224, 512], 
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),
    backbone=dict(
        type='SECOND',
        in_channels=512, 
        out_channels=[64, 128, 256],
        layer_nums=[5, 5, 5],
        layer_strides=[1, 2, 2], 
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256], 
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4], 
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    neck=dict(
        type='TemporalFusion',
        frames_num=4,
        fusion_method='mvx' 
    ),
    bbox_head=dict(
        type='RailCenterHead', 
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=1, class_names=['obstacle']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-10, -50, -10, 210, 50, 10],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9  
        ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=2.0), 
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.5), 
        norm_bbox=True
    ),
    rail_head=dict(
        type='BEVSegHead',
        in_channels=384,    
        num_classes=1,      
        loss_seg=dict(type='DiceLoss', loss_weight=2.0)
    ),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0]
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-10, -50, -10, 210, 50, 10],
            max_per_img=500,
            max_pool_nms=False,
            # 这里的 0.175 (obstacle) 和 0.85 (pedestrian) 就是我们之前做的 Anchor Tuning (Micro Radii)
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            
            # 保持之前Plan A的优化配置，虽然之前没生效，但在新头训练好后会有奇效
            score_threshold=0.01, 
            
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=500, 
            nms_thr=0.2
        )
    )
)

# [核心修改 1] 加载刚制作好的“无头”权重
load_from = 'checkpoints/epoch_12_headless.pth'

log_config = dict(
    interval=1,  
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])