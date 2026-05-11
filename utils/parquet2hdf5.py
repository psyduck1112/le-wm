import glob
import io
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from pathlib import Path

src_dir = Path("/root/autodl-tmp/stable-wm/libero_goal_image/data")
out_path = Path("/root/autodl-tmp/stable-wm/libero_goal.h5")

files = sorted(glob.glob(str(src_dir/ "**/*.parquet"), recursive=True)) # ** 匹配任意层级的目录，* 匹配任意文件名
print(f"found {len(files)} parquet files")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Total rows: {len(df)}")

df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True) # 先按episode_index排,相同时再按frame_index排

ep_lengths = df.groupby("episode_index")["frame_index"].count().values # 按照epi_idx分组，返回groupby对
"""
episode_index
  0      270
  1      268
  2      265
  3      271
"""
ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths)[:-1]])
task_indices = df.groupby("episode_index")["task_index"].first().values.astype(np.int32)
print(f"Episodes: {len(ep_lengths)}, frames: {ep_lengths.sum()}")

pixels = [] 
for img_dict in df["observation.images.image"]:
    img = Image.open(io.BytesIO(img_dict["bytes"])) # 原始数据中是image是以image形式存储的bytes，解码JPEG → PIL Image对象
    pixels.append(np.array(img))
pixels = np.stack(pixels) # (52042, 256, 256, 3)
print(f"pixels shape: {pixels.shape}")

actions = np.stack(df["action"].values).astype(np.float32)                                        
proprios = np.stack(df["observation.state"].values).astype(np.float32)
print(f"actions shape: {actions.shape}")                                                          
print(f"proprios shape: {proprios.shape}") 

with h5py.File(out_path, "w") as f:
      f.create_dataset("ep_len",        data=ep_lengths)
      f.create_dataset("ep_offset",     data=ep_offsets)
      f.create_dataset("task_index",    data=task_indices)
      f.create_dataset("pixels",        data=pixels,   chunks=True)
      f.create_dataset("action",        data=actions)
      f.create_dataset("proprio",       data=proprios)
print(f"Saved to {out_path}")