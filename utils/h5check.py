import h5py

def print_hdf5_structure(name, obj):
    """递归打印 HDF5 的树状结构"""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

# 打开刚才解压好的 h5 文件
h5_path = r"C:\Users\RaymondZh\Desktop\github\le-wm\datasets\libero_goal.h5"
with h5py.File(h5_path, 'r') as f:
    print(f"--- 正在查看文件: {h5_path} ---")
    # 遍历打印所有内容
    f.visititems(print_hdf5_structure)
    
    # 如果你想单独读取某一个数据，像字典一样提取即可：
    # 例如：images = f['obs/agentview_image'][:]