import h5py
import numpy as np

def compare_mujoco_initial_states(h5_path_1, h5_path_2, demo_key_1="data/demo_0", demo_key_2="data/demo_0"):
    """
    对比两个 HDF5 文件（或同一文件的不同 demo）的 MuJoCo 初始状态和环境。
    """
    print(f"--- 开始对比初始状态 ---")
    print(f"File 1: {h5_path_1} [{demo_key_1}]")
    print(f"File 2: {h5_path_2} [{demo_key_2}]")
    print("-" * 30)

    try:
        with h5py.File(h5_path_1, 'r') as f1, h5py.File(h5_path_2, 'r') as f2:
            
            # --- 1. 对比底层的 MuJoCo 环境参数 (XML) ---
            # 增加类型检查，兼容 str 和 bytes
            def get_xml_string(group):
                if "model_file" in group.attrs:
                    val = group.attrs["model_file"]
                    return val.decode('utf-8') if isinstance(val, bytes) else str(val)
                return None

            xml_1 = get_xml_string(f1[demo_key_1])
            xml_2 = get_xml_string(f2[demo_key_2])
            
            if xml_1 and xml_2:
                is_same_env = (xml_1 == xml_2)
                print(f"[环境结构校验] 物理环境配置 (XML) 是否完全一致: {is_same_env}")
            else:
                print(f"[环境结构校验] 未能在 attrs 中找到 'model_file' (XML) 属性。")

            # --- 2. 对比底层的物理初始状态 (qpos, qvel 等) ---
            state_1 = f1[f"{demo_key_1}/states"][0]
            state_2 = f2[f"{demo_key_2}/states"][0]
            
            is_same_state = np.allclose(state_1, state_2, atol=1e-6)
            print(f"[物理状态校验] MuJoCo 初始物理状态 (states) 是否完全一致: {is_same_state}")
            
            if not is_same_state:
                diff = np.abs(state_1 - state_2)
                max_diff_idx = np.argmax(diff)
                print(f"  -> 不一致！最大差异出现在索引 {max_diff_idx}，差值为: {diff[max_diff_idx]:.6f}")

            # --- 3. (可选) 对比机器人的初始关节角度 ---
            joint_1 = f1[f"{demo_key_1}/obs/joint_states"][0]
            joint_2 = f2[f"{demo_key_2}/obs/joint_states"][0]
            is_same_joint = np.allclose(joint_1, joint_2, atol=1e-6)
            print(f"[机器人姿态校验] 初始关节角度 (joint_states) 是否完全一致: {is_same_joint}")

    except Exception as e:
        print(f"对比时发生错误: {e}")

if __name__ == "__main__":
    file_1 = r"C:\Users\RaymondZh\Desktop\github\le-wm\datasets\open_drawer.hdf5"
    file_2 = r"C:\Users\RaymondZh\Desktop\github\le-wm\datasets\open_put.hdf5"
    
    # 对比同一文件下的 demo_0 和 demo_1
    compare_mujoco_initial_states(file_1, file_1, demo_key_1="data/demo_0", demo_key_2="data/demo_0")