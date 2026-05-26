"""
EEF 控制器：把 5-D action [dx, dy, dz, dyaw, gripper] 转成 data.ctrl。

为什么不直接给关节控制：
  - oracle / policy 输出的是"末端去哪儿"，不是"7 个关节角度多少"
  - 这一层把"任务空间控制"翻译成"关节空间控制"
  - 同时也是真机部署时唯一需要 robot-specific 调整的部分

核心算法：Jacobian 伪逆 IK
  1. 当前 EEF 位置 p_curr = data.body("attachment").xpos
  2. 目标 EEF 位置 p_target = p_curr + [dx, dy, dz] * scale
  3. 位置误差 e = p_target - p_curr
  4. 雅可比 J: mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
  5. 关节位移 dq = J⁺ @ e（伪逆）
  6. data.qpos[:7] += dq   或   data.ctrl[:7] = qpos_target

(可选) 加 yaw control:
  - 类似上面，但用 jacr（旋转 jacobian）
  - 只取 yaw 那一维

(可选) Gripper:
  - Franka 双指 gripper 在 menagerie 里有专用 actuator
  - action[4] ∈ [-1, 1] → ctrl[7], ctrl[8] 之间映射

待你实现：
  class FrankaIKController:
      def __init__(self, model, eef_body_name="attachment", trans_scale=0.05, yaw_scale=0.3):
          # 缓存 body id、actuator id 等

      def apply(self, data, action: np.ndarray) -> None:
          # action shape: (5,)
          # in-place 改 data.ctrl（不要直接改 qpos，让 mj_step 自己积分）

API hints:
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "attachment")
  mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
  data.body("attachment").xpos  # 当前 EEF position
  data.body("attachment").xmat  # 当前 EEF rotation (3x3)
  np.linalg.pinv(J)             # 伪逆
"""

# TODO: implement
