# HRCrobot 与 UR MuJoCo 对齐约定

当前 MuJoCo 后端使用 UR5e 几何模型，因此它可以对齐 HRCrobot 的策略接口与任务场景，
但不能自动代表 HRCrobot 的关节运动学、碰撞几何或真实控制延迟。

## 已对齐的接口

- TaskFrame 和动作目标：`[x, y, z, roll, pitch, yaw]`，单位为 m + rad。
- `*.ee_pos` 观测：平移 + rotation vector，与 UR RTDE 和 MuJoCo 相同。
- 动作模式：HRCrobot 与当前 MuJoCo 笛卡尔后端都只接受 Task-space POS。
- 真机适配器默认拒绝相邻绝对目标超过 5 mm 或 2° 的不连续命令；这项硬件保护
  不由 MuJoCo 仿真替代。
- 夹爪：`0.0` 打开、`1.0` 闭合，并共享最小命令间隔语义。
- 策略周期：MuJoCo 的 `control_dt=1/30` 应与 actor/env 的 30 Hz 外循环一致。
  HRCrobot 的 `frequency=100` 是底层命令上限，不是策略频率。

## 同一策略运行时必须一致

1. 仿真和真机使用完全相同的 primitive TaskFrame、可学习轴和动作缩放。
2. 策略的 proprioception 只选择两端都有的通道：当前是 6D EE pose 和夹爪。
3. 每次部署前标定工装在 HRCrobot base 下的 TaskFrame origin；不要复制 MuJoCo world 坐标。
4. 视觉策略必须为真机配置与仿真同名、同分辨率的相机，并标定相近的外参和视场角。

## 仍需实测标定

- HRCrobot TCP 偏置、工具质量和夹爪碰撞尺寸。
- 工装/物体相对 base 的位姿。
- 真机单步位移/转角上限，以及命令延迟和跟踪带宽。
- 相机内参、外参、曝光与视觉域随机化范围。

在这些量没有测量前，UR 仿真适合验证任务逻辑和训练流程，不应被视为精确的 HRCrobot
数字孪生。若策略主要依赖 TaskFrame 相对观测，几何差异的影响会减小，但不会消失。
