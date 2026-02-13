# 继承之前的配置
_base_ = ['./osdar23_phase2.py']

# ==========================================================
# [Phase 2.7: The Extended Play] - 收汁阶段
# ==========================================================

# 1. 加载上一阶段最好的权重
load_from = 'work_dirs/osdar23_phase2_head_reset/epoch_24.pth'

# 2. 调整优化器：降低学习率 (1/10)
# 既然已经收敛得不错了，就用小火慢炖，精细调整边界框
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)

# 3. 短周期训练 (12 Epochs)
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 4. 调整 LR 衰减策略
# 在第 8 和 11 个 epoch 进一步衰减，确保存在这个短周期内能收敛
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)

# 5. 确保评估频率
evaluation = dict(interval=1)

# 6. 工作目录
work_dir = 'work_dirs/osdar23_phase2_extended'