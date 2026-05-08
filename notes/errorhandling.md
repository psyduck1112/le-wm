
# 故障处理方法

## 第一步：读懂报错

报错信息从下往上读，最底层的才是根本原因。

```text
Traceback (most recent call last):
  File "a.py", line 10       ← 调用链顶部（入口）
  File "b.py", line 20       ← 中间调用
  File "c.py", line 30       ← 实际出错位置
ErrorType: message           ← 根本原因，从这里开始分析

```

**先认识常见 ErrorType 的含义：**

| 报错类型 | 含义 | 第一反应 |
| --- | --- | --- |
| **ModuleNotFoundError** | 找不到这个包 | 没装 or 装错环境 |
| **ImportError: cannot import name X** | 包找到了但没有 X | 版本不对 or 路径不对 |
| **AttributeError: 'NoneType'** | 变量是 None 却被当对象用 | 往上找这个变量哪里赋值的 |
| **AttributeError: X has no attribute Y** | 对象没有这个方法/属性 | 版本不对 or 类型不对 |
| **AssertionError** | 框架的前置条件不满足 | 看 assert 的条件缺什么 |
| **OverflowError** | 数值超出类型范围 | 确认数据类型，截断 |
| **FileNotFoundError** | 文件路径不存在 | 检查路径是否正确 |
| **UnpicklingError** | 反序列化失败 | 版本不兼容 or weights_only 问题 |

## 第二步：定位问题范围

**确认是哪一类问题：**

* **环境问题** → 包装错了、路径不对、版本冲突
* **系统库问题** → `.so` 文件缺失、软链接指错
* **代码接口问题** → 两个库的 API 不匹配
* **版本兼容问题** → 新版改了默认行为

**快速判断的命令：**

```bash
# 确认 python 和 pip 是否在同一环境
which python
which pip

# 确认包装到了哪里
python -c "import xxx; print(xxx.__file__)"

# 确认包有没有装
pip show xxx

# 确认系统库在不在
ldconfig -p | grep libxxx
ls -la /usr/lib/x86_64-linux-gnu/libxxx*

# 确认文件在哪里
find / -name "xxx.so" 2>/dev/null
find / -name "xxx.py" 2>/dev/null

# 确认类/函数在哪个文件里
grep -rn "class Xxx\|def xxx" /path/to/package/

```

## 第三步：分析根本原因

**原则：不要只看表面报错，要往上追溯一层。**

*举例：*

```python
# 表面报错
ModuleNotFoundError: No module named 'libero'

# 不要停在这里，追问：
# → 装了没有？   → 装了
# → 装在哪里？   → dist-info 在 site-packages，但源码目录不在 sys.path
# → 为什么？     → uv editable install 没生成 .pth 文件
# → 根本原因     → uv 和 pip 的 editable install 实现方式不同

```

**追溯的工具：**

```bash
# 看 Python 实际去哪里找包
python -c "import sys; print('\n'.join(sys.path))"

# 看包的元数据
cat /root/autodl-tmp/envs/lewm/lib/python3.10/site-packages/xxx-*.dist-info/direct_url.json

# 看报错文件的具体代码
sed -n '开始行,结束行p' /path/to/file.py

# 看某个关键字在哪里使用
grep -rn "关键字" /path/to/package/

```

## 第四步：制定解决方案

根据根本原因选对应的解法：

* **环境问题 → 装到正确的环境：**
```bash
uv pip install xxx     # 装到 uv venv
pip install xxx        # 装到当前 python 对应的环境
# 装完用 python -c "import xxx" 验证

```


* **系统库缺失 → 补库：**

```bash
    apt-get install -y libxxx   # 优先用 apt
    apt-get download libxxx     # apt 装不上就下 .deb 手动部署
    dpkg -x xxx.deb /tmp/extracted/
    cp extracted/usr/lib/x86_64-linux-gnu/libxxx.so /usr/lib/x86_64-linux-gnu/
    ldconfig
    ```

*   **软链接指错 → 修软链接：**
    
```bash
    ls -la /usr/lib/x86_64-linux-gnu/libxxx*                   # 先看现状
    rm /usr/lib/x86_64-linux-gnu/libxxx.so.1                   # 删错误链接
    ln -sf /path/to/correct.so /usr/lib/x86_64-linux-gnu/libxxx.so.1  # 建正确链接
    ldconfig
    ```

*   **代码接口不兼容 → 写适配层：**
    
```python
    # 不要改第三方库源码
    # 在自己的代码里写子类重写方法
    # 或者在调用处加转换逻辑
    ```

*   **版本兼容问题 → 改默认参数或降版本：**
    
```bash
    # 改默认参数（改库源码）
    sed -i 's/旧写法/新写法/' /path/to/library/file.py

    # 降版本
    uv pip install "xxx==1.4.0"

    # 升级（让代码适应新版）
    # 改自己的代码
    ```

*   **`except` 没捕获到 → 改异常类型：**
    
```bash
    sed -i 's/except ImportError:/except Exception:/g' /path/to/file.py
    ```

## 第五步：验证修复

每修一处，立刻验证，不要一次修多个再测。如果验证失败，回到第一步重新读报错，不要盲目继续改。

```bash
# 验证库能 import
python -c "import xxx; print(xxx.__version__)"

# 验证系统库
ldconfig -p | grep libxxx
python -c "from OpenGL import EGL; print('OK')"

# 验证接口
python -c "from liberogymwrapper.v0 import LiberoGoalEnv; print('OK')"

# 最终验证：跑最小复现
python eval_libero.py ... eval.num_eval=1 policy=random

```

---

# 具体问题拆解

## 1. EGL 渲染问题

* **报错核心：**

```text
    AttributeError: 'NoneType' object has no attribute 'eglQueryString'
    ```
*   **问题分析：**
    PyOpenGL 在 import 时会去找系统的 `libEGL.so.1`，找到之后尝试调用 `eglQueryString` 这个标准函数。你系统的 `libEGL.so.1` 软链接指向的是 `libEGL_nvidia.so.0`，这是 NVIDIA 为 Wayland/GBM 设计的私有实现，不包含 `eglQueryString`，所以返回 `None`，调用时崩溃。
*   **解决思路：**
    EGL 的正确架构是三层：`PyOpenGL → libEGL.so.1（Mesa 标准层）→ libEGL_nvidia.so.0（NVIDIA 驱动）`。你缺的是中间的 Mesa 标准层。从 Ubuntu 仓库手动下载 `.deb` 包，把 Mesa 的 `libEGL.so.1.1.0` 部署到系统库目录，覆盖错误的软链接。
*   **学会看的点：**
    *   报错是 `AttributeError: 'NoneType'`，说明库找到了但函数不存在。
    *   如果是 `OSError: cannot open shared object file`，说明库根本没找到。
    *   两种错误解法不同，前者是错库，后者是缺库。

## 2. mujoco import 的 PyOpenGL 误报

*   **报错核心：**
    
```text
    AttributeError: /lib/x86_64-linux-gnu/libEGL.so: undefined symbol: eglQueryString
    ```
    导致 `import mujoco` 崩溃。
*   **问题分析：**
    mujoco `__init__.py` 里有这段代码（通过 `try...except ImportError` 兼容旧版渲染器）。但 `from mujoco.rendering.classic.renderer` 触发了 PyOpenGL 初始化，抛出的是 `AttributeError`，不是 `ImportError`，所以捕获不到，异常向上传播导致崩溃。
*   **解决思路：**
    把 `except ImportError` 改成 `except Exception`，捕获所有运行时异常，让 mujoco 跳过旧版渲染器，用新版正常初始化。
*   **学会看的点：**
    *   看 `except` 后面捕获的是什么类型，和实际抛出的类型对不上就会漏掉。
    *   Python 异常继承关系：`Exception → RuntimeError/AttributeError/ImportError`，`ImportError` 只是 `Exception` 的子类。
    *   遇到 `try/except` 没生效，先确认抛出的异常类型是否匹配。

## 3. numba 装到 conda base 而非 lewm 环境

*   **报错核心：**
    
```text
    ModuleNotFoundError: No module named 'numba'
    ```
*   **问题分析：**
    终端提示符是 `(lewm) (base)`，`which python` 指向 lewm 环境。但用 `pip install` 时，`pip` 命令本身来自 conda base，所以包装进了 base 而不是 lewm。
*   **解决思路：**
    用 `uv pip install`，它会自动识别当前激活的 venv 并装到正确的环境里。
*   **学会看的点：**
    *   遇到"装了还是找不到"，先查 `which pip` 和 `which python` 是否在同一个环境。
    *   双重提示符说明两个环境同时激活，极易混淆环境变量。
    *   统一用 `uv pip install` 避免路径穿透问题。

## 4. LIBERO editable install 的 .pth 问题

*   **报错核心：**
    
```text
    ModuleNotFoundError: No module named 'libero'
    ```
    （`uv pip install -e .` 成功后仍报错）
*   **问题分析：**
    pip 的 editable install 会生成 `.pth` 文件将源码路径加入 `sys.path`。而 uv 的 editable install 在某些情况下不生成 `.pth` 文件，导致 Python 找不到源码。
*   **解决思路：**
    手动创建 `.pth` 文件，把源码根目录写进去。
*   **学会看的点：**
    *   用 `python -c "import xxx; print(xxx.__file__)"` 确认是否真的找不到。
    *   检查 `site-packages` 里有没有对应的 `.pth` 或 `.egg-link` 文件。
    *   如果 `find` 只看到 `dist-info` 没有实际目录或链接，就是 editable 没生效。

## 5. torch.load 的 weights_only 默认值变更

*   **报错核心：**
    
```text
    _pickle.UnpicklingError: Weights only load failed.
    WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct
    ```
*   **问题分析：**
    PyTorch 2.6 将 `torch.load` 的 `weights_only` 默认值改为 `True`，拒绝加载非纯张量的任意 Python 对象（如 numpy 数组）。
*   **解决思路：**
    在信任数据来源的前提下，给 `torch.load` 加上 `weights_only=False`。
*   **学会看的点：**
    *   看报错提到的 "global" 类型，如果是 numpy/sklearn 或自定义类，必定是 `weights_only` 限制。
    *   这类报错常伴随框架大版本升级（如 PyTorch 2.6 的 breaking change）。

## 6. LiberoGymWrapper import 路径错误

*   **报错核心：**
    
```text
    ImportError: cannot import name 'LiberoGoalEnv' from 'liberogymwrapper.libero_env'
    ```
*   **问题分析：**
    调用代码写死了从 `libero_env` 导入，但实际目标类定义在 `v0.py` 中。这是调用侧代码与 wrapper 版本不匹配导致。
*   **解决思路：**
    用 `grep -rn "LiberoGoalEnv" /path/to/wrapper/` 找到实际定义位置，修改 import 路径。
*   **学会看的点：**
    *   `ImportError: cannot import name 'X' from 'Y'` 表示模块存在但缺少特定内容。
    *   区分于 `ModuleNotFoundError`（模块彻底不存在）。
    *   遇到前者直接全局 `grep` 搜索类的真实定义位置。

## 7. init_state_id 缺失

*   **报错核心：**
    
```text
    AssertionError: Key init_state_id not found in info dict during step.
    ```
*   **问题分析：**
    world model wrapper 强依赖 `info` 字典里的 `init_state_id` 追踪状态，但底层 Gym wrapper 的 `step` 没有返回此字段，导致接口断言失败。
*   **解决思路：**
    通过子类化重写 `reset` 和 `step` 方法。调用 `super()` 后，手动往 `info` 字典注入 `init_state_id`（`reset` 给随机 idx，`step` 填 0 占位）。
*   **学会看的点：**
    *   `AssertionError` 说明违背了框架接口契约，看 assert 缺失什么参数。
    *   用 `grep -n "init_state_id" wrapper.py` 查阅其在框架中的语义用途。
    *   **最佳实践：** 解决接口不兼容应通过继承/写适配层，绝对不要直接改上游 wrapper 源码。

## 8. seed 溢出

*   **报错核心：**
    
```text
    OverflowError: Python int too large to convert to C long
    ```
*   **问题分析：**
    NumPy 的随机种子使用 `uint64`（最大 $2^{64}-1$），而 64 位 Linux 上的 C long 是 `int64`（最大 $2^{63}-1$）。超出边界时向下转型会导致 C 扩展崩溃。
*   **解决思路：**
    通过位运算或取模操作截断，如使用 `% (2**31)` 将其限制在安全的 int32 范围内。
*   **学会看的点：**
    *   看到 `int too large to convert to C long` 就是固定的溢出模式。
    *   定位传参行，确认来源变量类型，强制转换或取模截断。

```