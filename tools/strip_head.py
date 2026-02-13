import torch
import sys
import os

def main():
    # 检查参数
    if len(sys.argv) < 3:
        print("Usage: python tools/strip_head.py <in_checkpoint> <out_checkpoint>")
        print("Example: python tools/strip_head.py work_dirs/phase2/epoch_12.pth checkpoints/epoch_12_headless.pth")
        return

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # 1. 检查输入文件是否存在
    if not os.path.exists(in_path):
        print(f"Error: Input checkpoint '{in_path}' does not exist!")
        return

    print(f"🔍 Loading checkpoint from: {in_path}")
    try:
        checkpoint = torch.load(in_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # 2. 获取 state_dict (兼容不同格式)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        meta_info = checkpoint.get('meta', {})
    else:
        state_dict = checkpoint
        meta_info = {}
        
    new_state_dict = {}
    deleted_keys = []
    
    # 3. 核心逻辑：删除检测头权重
    print("✂️  Stripping 'bbox_head' weights...")
    for k, v in state_dict.items():
        # 这里匹配 'bbox_head'，这是 MMDetection3D 中检测头的标准命名前缀
        if 'bbox_head' in k:
            deleted_keys.append(k)
        else:
            new_state_dict[k] = v
            
    # 4. 重新封装
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict
        
    # 5. 保存
    # 确保输出目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    torch.save(checkpoint, out_path)
    
    print("-" * 50)
    print(f"✅ Success! Removed {len(deleted_keys)} keys related to detection head.")
    print(f"💾 Headless checkpoint saved to: {out_path}")
    print("-" * 50)

if __name__ == '__main__':
    main()