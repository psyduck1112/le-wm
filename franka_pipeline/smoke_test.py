"""
Smoke test: 验证 MuJoCo 装好、能 load Franka 模型、能 step、能渲染。

跑通这个再往下做，确保环境没问题。

预期输出：
  - 打印 nq, nv, nu（关节自由度等）
  - 打印 100 步后的 EEF 位置
  - 保存一张渲染图到 smoke_test_render.png

依赖：
  - mujoco
  - mujoco_menagerie（git clone 放在某处）
  - numpy, matplotlib（看图用）

待你实现的部分：
  1. 加载 menagerie 里的 panda.xml（路径放在文件顶部 const）
  2. 创建 MjData
  3. 跑 100 步 mj_step（不施加控制，看臂会不会因为重力下垂）
  4. 用 mujoco.Renderer 渲染一张图
  5. 保存图，打印关键状态

API hint:
  model = mujoco.MjModel.from_xml_path(PATH)
  data  = mujoco.MjData(model)
  mujoco.mj_step(model, data)
  renderer = mujoco.Renderer(model, height=H, width=W)
  renderer.update_scene(data)
  pixels = renderer.render()  # (H, W, 3) uint8
"""

# TODO: implement
