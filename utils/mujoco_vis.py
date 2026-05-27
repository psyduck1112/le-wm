import mujoco
import numpy as np
import imageio

XML_PATH = "/home/yikang/envs/lewm/lib/python3.10/site-packages/robosuite/models/assets/robots/panda/robot.xml"
OUT_PATH  = "/home/yikang/git/le-wm/train.py"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)

frames = []
for i in range(200):          # 200 步 = 约 4 秒（dt=0.002 * 10 frameskip）
    mujoco.mj_step(model, data)
    if i % 5 == 0:            # 每 5 步截一帧
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

imageio.mimsave(OUT_PATH, frames, fps=30)
print(f"保存到 {OUT_PATH}，共 {len(frames)} 帧")
print("用 scp 下载到本地看：scp 服务器:{}  ./panda.mp4".format(OUT_PATH))
