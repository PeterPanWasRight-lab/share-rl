# HRCrobot 接口文档

HRC 六轴机械臂的 LeRobot / SHaRe 适配器。已在真机完成全链路验证（连接、位姿反馈、笛卡尔运动、夹爪动作）。

## 文件结构

```
src/share/robots/HRCrobot/
├── __init__.py                    # 包导出
├── pyproject.toml
├── README.md                      # 本文档
├── lerobot_robot_HRCrobot/        # 适配层
│   ├── __init__.py
│   ├── config_HRCrobot.py         # 配置类 HRCrobotConfig
│   ├── HRCrobot.py                # LeRobot Robot 适配器（坐标系换算在这里）
│   └── controller.py              # 厂家 SDK 最薄封装（单位换算、限频、IO）
└── vendor/HRCrobotSDK/            # 厂家 SDK（勿修改）
    ├── hsrosi/                    # 位姿/运动链路 Python 封装 (端口 9095)
    ├── libpyhstrajproxy.so        # 轨迹代理动态库
    └── libhsc3.so                 # 夹爪 IO 动态库 (端口 23234)
```

分层约定：`HRCrobot.py` 只懂 TaskFrame 坐标语义，`controller.py` 只懂厂家 SDK（单位/端口/IO），两层通过固定契约衔接。换机器人型号只需重写 `controller.py`。

## 快速开始

```python
from share.robots.HRCrobot import HRCrobot, HRCrobotConfig

robot = HRCrobot(HRCrobotConfig())
robot.connect()

# 可选：声明任务坐标系（不设 = 与 base 系重合）
robot.set_task_frame(TaskFrame(origin=[0.40, -0.10, 0.05, 0, 0, 0]))

obs = robot.get_observation()
robot.send_action({"z.ee_pos": 0.10, "gripper.pos": 1.0})

robot.disconnect()
```

测试文件在 `tests/ HRCrobotTest/`（离线 / 在线安全 / 运动三层开关）。

## 配置项 (HRCrobotConfig)

| 字段                  | 默认值             | 说明                                    |
| --------------------- | ------------------ | --------------------------------------- |
| `robot_ip`          | `"10.10.59.211"` | 机器人控制器 IP                         |
| `hrc_port`          | `9095`           | hsrosi 位姿/运动链路端口                |
| `hsc3_port`         | `23234`          | HSC3 夹爪 IO 链路端口                   |
| `frequency`         | `100.0`          | 运动指令下发频率（Hz），见「限频机制」  |
| `use_gripper`       | `True`           | 是否启用夹爪链路；`False` 时不连 HSC3 |
| `gripper_threshold` | `0.5`            | ≥0.5 判闭合，<0.5 判打开               |
| `gripper_min_command_interval_s` | `0.5` | 夹爪目标变化的最小下发间隔，与 MuJoCo/UR 一致 |
| `gripper_target_tolerance` | `1e-4` | 夹爪目标判定"未变化"的容差（同目标永不重发） |
| `max_cartesian_step_m` | `0.005` | 单步平移限幅（m），超限抛 ValueError。只能收紧，见「行为契约」 |
| `max_rotation_step_rad` | `0.0349`（2°） | 单步旋转限幅（rad），超限抛 ValueError。只能收紧 |
| `cameras`           | `{}`             | 暂未使用                                |

## 接口列表

### 连接生命周期

| 接口                              | 功能                                      | 备注                                  |
| --------------------------------- | ----------------------------------------- | ------------------------------------- |
| `connect()`                     | 连接机器人（自动进入笛卡尔模式）+ 连 HSC3 | HSC3 失败只警告，夹爪降级为仅记录命令 |
| `disconnect()`                  | 断开两条链路                              | 耗时约 2s（HSC3 内部等待）            |
| `is_connected` (property)       | 查询连接状态                              | 微秒级                                |
| `is_calibrated` (property)      | 恒`True`                                | 不走 LeRobot 标定                     |
| `calibrate()` / `configure()` | no-op 占位                                | 保持 LeRobot 接口完整                 |

### 数据通道声明

`observation_features` / `action_features`（对称）：

```
{'x.ee_pos': float, 'y.ee_pos': float, 'z.ee_pos': float,
 'rx.ee_pos': float, 'ry.ee_pos': float, 'rz.ee_pos': float,
 'gripper.pos': float}          # 仅 use_gripper=True 时存在
```

### get_observation() -> dict

返回 **TaskFrame 系**下的当前状态（内部完成 base→task 换算）：

```python
{'x.ee_pos': 0.050,   # TCP 相对任务原点，米
 'y.ee_pos': -0.020,
 'z.ee_pos': 0.015,
 'rx.ee_pos': 0.006,  # 相对任务坐标轴姿态，rotation vector，rad
 'ry.ee_pos': ..., 'rz.ee_pos': ...,
 'gripper.pos': 0.0}  # 最后一次夹爪命令值（无真实反馈）
```

实测延迟：p50 ≈ 0.15ms，p95 < 1ms；偶发网络抖动离群值可达 ~20ms（有 servo 掉拍保护兜底）。持续吞吐上限 ≈ 5000Hz。

### send_action(action) -> dict

下发 **TaskFrame 系**下的绝对目标，返回 **executed_action**（见下方行为契约）：

```python
robot.send_action({
    "x.ee_pos": 0.010,    # 未出现的轴自动用 task_frame.target 补齐
    "z.ee_pos": 0.080,
    "gripper.pos": 1.0,   # ≥0.5 闭合；<0.5 打开；经夹爪限频器决定是否下发
})
```

`send_action()` 的 TaskFrame 姿态目标是 extrinsic XYZ Euler（rad），而
`get_observation()` 的 `rx/ry/rz.ee_pos` 是 rotation vector（rad）。不要把姿态观测
原样作为绝对动作回放；应先转换为 Euler。

内部链路：安全限幅检查 → task→base 换算 → `servo_cartesian` 限频 → SDK 流式
`move_to_cartesian_position`。

## 行为契约（2026-08-29 起，安全硬保护）

`send_action` 不是无条件透传，它是带四道闸门的唯一指令入口。上层 primitive 的
使用者必须了解以下契约：

### 1. 单步安全限幅（fail-fast）

每次下发前，适配器将目标 vendor 位姿与当前 TCP 位姿比对：

- **平移单步 > `max_cartesian_step_m`（默认 5mm）→ 抛 `ValueError`，不下发**
- **旋转单步 > `max_rotation_step_rad`（默认 2°）→ 抛 `ValueError`，不下发**

设计意图：即使上层把旋转表示搞错（如把 rotvec 当 euler 回放，曾导致真机飞车），
机器人单拍最多走 5mm/2°，物理上不可能失控。阈值只能收紧不能放宽。

**primitive 步长预算公式**：`单步距离 = 轨迹速度 ÷ 采样帧率`。
例：100mm/s @ 100fps = 1mm/拍 ✅；300mm/s @ 30fps = 10mm/拍 ❌ 必炸。
调快轨迹速度或降低采样率前，先核对预算；预算不足时在 primitive 层细分目标点，
**不要放宽阈值**。

注意：限幅比较的是**完整 vendor 位姿（含补齐轴）**，与"缺失轴由 target 补齐"
语义天然兼容。

### 2. 夹爪命令限频（GripperCommandLimiter）

- **同目标永不重发**：目标与当前执行值之差 ≤ `gripper_target_tolerance`
  （默认 1e-4）时，无论隔多久都不下发
- **冷却间隔**：不同目标之间至少隔 `gripper_min_command_interval_s`（默认 0.5s），
  冷却期内的变更被抑制
- 被抑制时不产生硬件调用

### 3. 返回值 = executed_action（现实状态）

返回的字典反映**硬件实际执行的目标**，而非请求值：

- 平移/旋转轴：请求即执行，原样返回
- 夹爪被冷却抑制时：返回**仍在执行的旧目标**（例：请求 0.0 被抑制 → 返回 1.0）

上层若依赖返回值做状态推断（如评估脚本），应以返回值为准。

### 4. 输入不可变

`send_action` 不修改传入的 action 字典，返回值是新字典。

### 5. 观测/动作的旋转表示不对称（易错点）

| 通道 | 姿态表示 |
|---|---|
| `get_observation()` 的 `rx/ry/rz.ee_pos` | rotation vector（rad） |
| `send_action()` 的 `rx/ry/rz.ee_pos` | extrinsic XYZ Euler（rad） |
| TaskFrame 的 origin/target | extrinsic XYZ Euler（rad） |

消费观测做动作时必须 `euler_xyz_from_rotvec()` 转换（见
`tests/share/robots/test_hrcrobot_send_action_contract.py` 的往返契约测试）。

### 6. 连接即验证笛卡尔模式

`connect()` 完成后会校验 vendor 侧运动模式确为 cartesian 且链路存活，否则断开
并抛 `ConnectionError`——绝不允许在模式未就绪时进入 servo 循环。

### set_task_frame(frame)

声明任务坐标系，**只保存不驱动机器人**（运动只发生在 send_action）。

- `origin`：任务原点在 **base 系**下的位姿 `[x,y,z,r,p,y]`（m + rad）。全零 = 与 base 重合（默认）
- `target`：不可学轴（`policy_mode=None`）的静态维持目标——与 origin 语义不同
- 一个对象同时只持有一个当前帧（寄存器语义），primitive 切换时由 env 层 `apply_task_frames()` 自动换写，上层一般不手动调用

## 坐标系关系

```
基座坐标系 (base, 由 SDK 定义)
    │  controller.get_tcp_pose() 报告 base→TCP（已换算为 m + rotvec rad）
    ▼
"world"  =  base（恒等，SHaRe 代码直接采纳 base 当 world）
    │
    ▼  origin 平移/旋转
TaskFrame
    │
    ▼
{axis}.ee_pos   ← 上层 primitive 看到的数值
```

- 观测 = TCP(base) 减去 origin（姿态为相对任务轴的朝向）
- 动作 = 上层目标加上 origin，翻译回 base 系
- TaskFrame 的姿态配置使用 extrinsic XYZ Euler；`*.ee_pos` 姿态观测使用 rotation vector，与 UR/MuJoCo 一致
- SDK 返回的就是 base 系读数，base 是参考系本身，"世界系在哪"不可推断也无需推断
- 切换 primitive 时机器人不动但观测值跳变（origin 换了），属设计使然；MP-Net 通过 `entry_context` 传递上一原语的 origin 保证跨切换连续性

## 限频机制（frequency 的确切语义）

`config.frequency = 100.0` → controller 内部 `period = 10ms`，**全库唯一限频点**是 `servo_cartesian()`（`controller.py:480-513`）：

- **点位永不丢弃，只会被延迟**。调用间隔 < 10ms 时函数内部 sleep 到节拍点才发——即接口**阻塞最多一个周期**后返回
- 调用间隔 ≥ 10ms → 永不阻塞
- 节拍按 `+= period` 固定推进，上层快慢抖动只转化为单次阻塞长短，长期平均下发频率严格 = frequency
- 掉拍超 2 周期 → 丢弃过期节拍、重置到当前时刻，不补发历史目标
- **无点位 buffer**：即发即弃，轨迹连续性责任在调用方（每周期送新目标）；调用中断机器人停在最后目标附近

不受限频的接口：`get_observation` / `set_gripper`（注意单次 2~4ms 的 IO 成本）/ `set_task_frame`（纯内存）/ `connect` / `disconnect`。

## SDK 调用细节

| 事项               | 说明                                                                                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| SDK 无 servoL 接口 | 用`move_to_cartesian_position` 流式下发 + 100Hz 限频逼近伺服跟随                                                                              |
| SDK 反逻辑         | C 库`HRC_move_to_d_pos` 返回 0 才算成功，hsrosi 已封装，按"返回 True 成功"使用                                                                |
| 单位换算           | SDK mm + euler degree ↔ 适配层 m + rotvec rad，隔离在`controller.py` 的 `_sdk_pose_to_rotvec_pose` / `_rotvec_pose_to_sdk_pose` 两个函数 |
| 旋转表示           | SDK rx/ry/rz 解释为 extrinsic XYZ euler（度），已经真机旋转测试验证（`test_servo_cartesian_rotation_rz`）                                     |
| 连接时序           | `init()` → `connect(ip, port, motion_mode="cartesian")`，SDK 自动以当前位姿启动笛卡尔模式                                                  |
| 动态库加载         | `libpyhstrajproxy.so` / `libhsc3.so` 加载期间需 chdir 到库目录（RUNPATH 为相对路径），controller 已处理                                     |

## 夹爪

- **链路独立**：本体走 hsrosi (9095)，夹爪走 HSC3 (23234) 的 IO 口，两条 TCP 连接
- **控制方式**：双 IO 互斥——`DO25=1, DO26=0` 打开；`DO25=0, DO26=1` 闭合
- **无反馈**：`get_io` 读回恒为 False（固件限制，已扫描 0-63 号口确认），`gripper.pos` 观测值是最后一次命令状态
- **降级策略**：HSC3 连不上时仅记录命令状态，不影响本体运动
- 单次 `set_gripper` 耗时 2~4ms（两次 IO 往返），HRCrobot 层有"值变化才发"去重

## 已知问题与注意事项

1. **`disconnect()` 耗时 ~2s**：HSC3 close 内部等待，断开重连类逻辑需预留时间
2. **夹爪测试后停在打开状态**：`test_hrcrobot_gripper.py` 最后会恢复打开，离开现场前注意
3. **vendor 目录**：含备份文件（`*.so40ms` 等），SDK 更新时原样替换整个目录
4. **未连接守卫**：所有硬件接口未连接时抛 `RuntimeError`，不静默执行
5. **单位**：对外一律 m + rad；SDK 的 mm/度只存在于 controller 内部

仿真迁移时的边界和标定清单见 [SIM_ALIGNMENT.md](SIM_ALIGNMENT.md)。

## 相关文件

| 路径                                                    | 内容                                |
| ------------------------------------------------------- | ----------------------------------- |
| `tests/ HRCrobotTest/test_hrcrobot_controller.py`     | controller 层测试（含限频/旋转，三层安全开关） |
| `tests/ HRCrobotTest/test_hrcrobot_pose_motion.py`    | controller 层位姿移动套件（三轴往返/停止保持/完整栈，需 ACK 开关） |
| `tests/ HRCrobotTest/test_while_loop_timing.py`       | while True 循环节拍与位移测试（完整栈） |
| `tests/ HRCrobotTest/test_hrcrobot_gripper.py`        | 夹爪测试（11 项）                   |
| `tests/ HRCrobotTest/verify_live_connection.py`       | 连接真实性采样脚本                  |
| `tests/ HRCrobotTest/benchmark_controller_latency.py` | 接口延迟基准                        |
| `tests/share/robots/test_hrcrobot_sim_alignment.py`   | HRCrobot.py 适配层仿真对齐测试（离线） |
| `tests/share/robots/test_hrcrobot_send_action_contract.py` | HRCrobot.py 适配层契约测试（离线，含旋转表示往返） |
| `tests/share/robots/test_gripper_command_limiter.py`  | 夹爪限频器单元测试（离线）          |
| `SIM_ALIGNMENT.md`                                    | 仿真/真机迁移边界与标定清单         |

