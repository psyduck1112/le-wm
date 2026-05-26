"""
Camera renderer wrapper。

为什么需要包一层（而不是直接用 mujoco.Renderer）：
  - 我们要支持多相机（agentview + wrist），每个相机各一个 Renderer 实例
  - 渲染前必须 update_scene；忘记会拿到旧画面（常见 bug）
  - 之后做 domain randomization 时，光照/纹理 jitter 在这里挂

待你实现：
  class FrankaRenderer:
      def __init__(self, model, height=224, width=224, camera_names=("agentview",)):
          # 为每个 camera_name 各建一个 mujoco.Renderer
          # 注意：同一个 mujoco.Renderer 可以切换 camera，但分开实例更清晰

      def render(self, data) -> dict[str, np.ndarray]:
          # 返回 {"agentview": (H,W,3) uint8, "wrist": (H,W,3) uint8, ...}

      def close(self):
          # 释放 GPU 资源

API hint:
  renderer.update_scene(data, camera=name_or_id)
  pixels = renderer.render()
"""

# TODO: implement
