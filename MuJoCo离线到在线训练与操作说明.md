# MuJoCo 离线到在线训练与操作说明

本文整理最近三段关于 MuJoCo 插入任务的对话内容，包括训练流水线、键盘操作和成功判定。

## 1. 完整训练流水线

当前 MuJoCo SAC 流程为：

```text
离线人工示范采集
        ↓
启动 Learner，加载离线 replay
        ↓
启动 Actor，采集在线 transition
        ↓
累计 100 条在线数据
        ↓
Learner 开始混合训练
  ├─ 50% 在线数据
  └─ 50% 离线示范
        ↓
Learner 周期性同步参数给 Actor
        ↓
继续在线采集与同步训练
```

需要注意：当前 SAC 实现不是先完成纯离线训练、再开始在线训练。Learner 会加载离线示范，但默认需要收到 100 条在线 transition 后才开始 SAC 优化。

控制链路为“SAC 归一化动作 → 笛卡尔相对速度 → 虚拟末端目标 → DLS IK → MuJoCo 关节位置伺服”。DLS IK 每个周期以当前实测关节位置为基准生成新的关节位置目标，不会把跟踪误差重复累加到上一周期的伺服目标。

在线 Actor 默认启用低频力觉回退过滤器。它在 policy 推理后、`env.step()` 前读取 task-frame XYZ wrench；任一轴超过 20 N 时，覆盖该轴上继续压向接触面的动作，并按超限比例生成回退。基础回退单周期最大 0.3 mm；若 policy 已沿正确方向回退，则额外保留经过单周期位移限幅后的原动作一半。实现按 `env.fps` 在“每周期位移”和物理速度之间换算，因此 10 Hz 与 30 Hz 下的 0.3 mm 含义一致。MuJoCo 的传感器符号已标定为 `wrench_to_backoff_sign=-1`；真机必须单独做逐轴符号标定。

可通过 `--env.force_backoff.enabled=false` 关闭；阈值可用 `--env.force_backoff.force_thresholds_n='[20,20,20]'` 修改。该过滤器只处理 Actor policy 动作，不替代 UR 控制柜的功能安全、保护停止或人工遥操作安全措施。

默认将 Menagerie 的位置伺服 Kp 放大 16 倍，并将 Kd 放大 `sqrt(16)` 倍以保持阻尼关系；可通过 `position_servo_stiffness_scale` 调整。该参数只改变 MuJoCo 内部位置控制器刚度，不改变 SAC 动作含义，也不会把接口变成力矩控制。

离线采集默认每个 episode 最多 900 个控制周期，即按 30 Hz 计算约 30 秒。成功判定监测自由工件上的 `object_tip` 在夹具坐标系中的状态，而不是夹爪 TCP：插入深度至少 70 mm、径向横向误差不超过 2.0 mm（对应 24 mm 方孔与 20 mm 工件的单边间隙），并且工件轴与孔轴的对齐度至少为 0.98。成功帧写入 `reward=1, done=true` 后立即结束；超时帧写入 `reward=0, truncated=true`。两种结束都会让下一轮直接完整重置仿真并回到 `insert`。

插入阶段不再把工作台高度当作 TCP 安全平面；默认 `min_tcp_z=-1.2 m` 只是宽松的数值边界。TCP 是数学坐标系，在几何允许时可以穿过桌面平面；真正阻止运动的是夹爪、工件、夹具与工作台 geom 之间的物理接触。完整复位还会清除上一轮残留的键盘状态，并把 teleop 夹爪目标恢复为闭合，防止复位后的第一帧再次执行上一轮的松开命令。

插座沿用 ACT 资产的四面孔壁，孔中心保持为空。场景中不再包含会占据孔中心、阻挡工件继续插入的 `socket-pin` 碰撞体。

机械臂控制使用的 `tool_tcp` 与 Menagerie 原始 2F-85 的 `gripper_pinch` 抓取中心重合；工件尖端 `object_tip` 是独立的成功监测点。两者不再错误地重合。

场景包含顶部灯和左右两盏斜侧灯。MuJoCo 交互窗口默认显示腕部相机小窗、三轴力与三轴力矩滚动曲线，以及左上角六维力/矩实时数值。命令行可分别通过 `--env.viewer_wrist_camera_overlay=false`、`--env.viewer_wrench_plot=false`、`--env.viewer_wrench_overlay=false` 关闭。

### 1.1 检查仿真

```bash
cd /home/peterpan/PeterpanWorkspace/share-rl
```

| 参数或路径                                    | 含义                                 |
| --------------------------------------------- | ------------------------------------ |
| `/home/peterpan/PeterpanWorkspace/share-rl` | 仓库根目录。后续命令均在该目录执行。 |

```bash
share-mujoco-demo --viewer-camera=free
```

| 参数                     | 含义                                       |
| ------------------------ | ------------------------------------------ |
| `share-mujoco-demo`    | 启动 MuJoCo 插入任务演示程序。             |
| `--viewer-camera=free` | 使用可交互旋转、平移和缩放的自由观察视角。 |

### 1.2 采集离线演示

```bash
share-record --env.type=mujoco_ur5e_insertion --env.viewer=true --env.viewer_camera=free --env.teleop_mode=keyboard --dataset.repo_id=local/mujoco-insertion --dataset.root=outputs/mujoco/offline-demos --dataset.num_episodes=20 --dataset.single_task="Insert the held peg into the fixture" --dataset.push_to_hub=false --use_policy=false --play_sounds=false
```

| 参数                                                             | 含义                                                                              |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `share-record`                                                 | 启动 MP-Net 离线数据采集程序。                                                    |
| `--env.type=mujoco_ur5e_insertion`                             | 使用内置的 MuJoCo UR5e 插入任务环境。                                             |
| `--env.viewer=true`                                            | 打开 MuJoCo 图形窗口。                                                            |
| `--env.viewer_camera=free`                                     | Viewer 使用自由观察视角；可改为`wrist` 查看腕部相机视角。                       |
| `--env.teleop_mode=keyboard`                                   | 使用键盘控制机械臂和夹爪。                                                        |
| `--dataset.repo_id=local/mujoco-insertion`                     | 数据集基础标识；`insert` primitive 最终使用 `local/mujoco-insertion-insert`。 |
| `--dataset.root=outputs/mujoco/offline-demos`                  | 离线数据集的本地父目录。                                                          |
| `--dataset.num_episodes=20`                                    | 采集并保存 20 个 episode。                                                        |
| `--dataset.single_task="Insert the held peg into the fixture"` | 写入数据集元数据的任务描述。                                                      |
| `--dataset.push_to_hub=false`                                  | 只保存到本地，不上传 Hugging Face Hub。                                           |
| `--use_policy=false`                                           | 不加载策略，完全使用人工键盘示范。                                                |
| `--play_sounds=false`                                          | 关闭录制过程中的语音或提示音。                                                    |

如果上次在初始化阶段异常退出，程序会将零 episode、零 frame 的残留 `insert` 目录归档为 `insert.incomplete-<timestamp>` 后重新开始。如果目录已包含有效 episode，Ctrl-C 后重新执行同一条命令会自动续录，不覆盖原数据。

自适应 `insert` primitive 的数据保存在：

```text
outputs/mujoco/offline-demos/insert/
```

### 1.3 启动 Learner

在终端 1 执行：

```bash
share-learner --env.type=mujoco_ur5e_insertion --env.policy_device=cuda --dataset.repo_id=local/mujoco-insertion --dataset.root=outputs/mujoco/offline-demos --dataset.video_backend=pyav --output_dir=outputs/mujoco/run --job_name=mujoco-insertion --batch_size=128 --wandb.enable=false
```

```bash
share-actor --env.type=mujoco_ur5e_insertion --env.teleop_mode=keyboard --env.policy_device=cuda --env.viewer=true --env.viewer_camera=free --dataset.repo_id=local/mujoco-insertion --dataset.root=outputs/mujoco/offline-demos --output_dir=outputs/mujoco/run --job_name=mujoco-insertion --wandb.enable=false
```

| 参数                                            | 含义                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------- |
| `share-learner`                               | 启动 SAC Learner 和 actor/learner 通信服务。                                       |
| `--env.type=mujoco_ur5e_insertion`            | 使用与采集阶段相同的 MuJoCo 插入环境和策略结构。                                   |
| `--env.policy_device=cuda`                    | 使用 CUDA GPU 执行策略前向、反向传播和参数优化。                                   |
| `--dataset.repo_id=local/mujoco-insertion`    | 指定离线数据集基础标识。                                                           |
| `--dataset.root=outputs/mujoco/offline-demos` | 从该目录的`insert/` 子目录加载离线示范。                                         |
| `--dataset.video_backend=pyav`                | 使用当前环境已安装的 PyAV 解码视频，避开不兼容的 TorchCodec/FFmpeg 动态库。        |
| `--output_dir=outputs/mujoco/run`             | 保存 checkpoint、训练状态和在线 replay 数据的目录。                                |
| `--job_name=mujoco-insertion`                 | 本次训练任务名称，用于日志和运行标识。                                             |
| `--batch_size=128`                            | 每次优化使用 128 条 transition；默认混合训练时约为 64 条在线数据和 64 条离线数据。 |
| `--wandb.enable=false`                        | 不启用 Weights & Biases 在线实验记录。                                             |

Learner 会加载离线数据，在 `127.0.0.1:50051` 启动通信服务，并等待 Actor 发送在线 transition。策略网络和训练 batch 使用 GPU；Replay Buffer 的 `storage_device` 仍为 CPU，避免大量图像 transition 占满显存。

### 1.4 启动 Actor

确认 Learner 已启动后，在终端 2 执行：

```bash
share-actor --env.type=mujoco_ur5e_insertion --env.teleop_mode=keyboard --env.policy_device=cuda --env.viewer=true --env.viewer_camera=free --dataset.repo_id=local/mujoco-insertion --dataset.root=outputs/mujoco/offline-demos --output_dir=outputs/mujoco/run --job_name=mujoco-insertion --wandb.enable=false
```

打开人工干预：

| 参数                                            | 含义                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| `share-actor`                                 | 启动 MuJoCo rollout Actor，采集在线 transition 并接收 Learner 参数。 |
| `--env.type=mujoco_ur5e_insertion`            | 使用 MuJoCo UR5e 插入环境。                                          |
| `--env.teleop_mode=keyboard`                  | 启用人工干预、人工成功/失败以及 Esc 停止按键。                       |
| `--env.policy_device=cuda`                    | 使用 CUDA GPU 执行 Actor 策略推理，并与 Learner 的设备配置保持一致。 |
| `--env.viewer=true`                           | 打开 Actor 的 MuJoCo 图形窗口。                                      |
| `--env.viewer_camera=free`                    | 使用自由观察视角；改为`wrist` 可观察腕部相机视角。                 |
| `--dataset.repo_id=local/mujoco-insertion`    | 与 Learner 保持一致的数据集基础标识。                                |
| `--dataset.root=outputs/mujoco/offline-demos` | 与 Learner 保持一致的离线数据父目录。                                |
| `--output_dir=outputs/mujoco/run`             | 与 Learner 使用同一个运行输出目录。                                  |
| `--job_name=mujoco-insertion`                 | 与 Learner 使用相同的训练任务名称。                                  |
| `--wandb.enable=false`                        | 不启用 Weights & Biases 在线实验记录。                               |

观察腕部相机时，将 Actor 命令中的 Viewer 参数改为：

```bash
--env.viewer_camera=wrist
```

| 参数                          | 含义                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `--env.viewer_camera=wrist` | 将 MuJoCo Viewer 固定到腕部相机；它只改变显示视角，不改变策略接收的相机 observation。 |

Viewer 默认在顶部同时显示 `front` 全局相机和 `wrist` 腕部相机，并在右侧显示六维力文本与示波器曲线。可分别用 `--env.viewer_front_camera_overlay=false` 或 `--env.viewer_wrist_camera_overlay=false` 关闭对应画面。

视觉 policy 默认将两路相机 observation 缩放为 `64×64`，Viewer 诊断小窗仍保持较高显示分辨率。可通过 `--env.policy_image_size=64` 显式设置；该值只影响数据集和 policy 输入，不影响 Viewer 清晰度。

建议停止顺序为：先停止 Actor，再停止 Learner。

### 1.5 使用硬编码成功轨迹数据集

仓库中的 `hardEncodedScripts/generate_mujoco_insertion_demos.py` 使用已知夹具位姿，按“随机起点 → 横向对准 → 接近孔口 → 插入”的顺序生成确定性示范。默认生成 100 条；只有通过当前 `peg_inserted` 物理判定的 episode 才保存，失败尝试会直接清空。

```bash
python hardEncodedScripts/generate_mujoco_insertion_demos.py --episodes 100 --output-root outputs/mujoco/hardEncodedDemosXYZGenerated
```

| 参数                                                          | 含义                                                                          |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `--episodes=100`                                            | 生成或续跑到总计 100 个成功 episode。                                         |
| `--output-root=outputs/mujoco/hardEncodedDemosXYZGenerated` | 新生成的 XYZ-only 数据集基础目录；实际 LeRobot 数据位于其`insert/` 子目录。 |
| `--seed=20260824`                                           | 随机起点与速度参数的确定性种子。                                              |
| `--max-attempts=3`                                          | 单条轨迹允许的最大尝试次数；失败数据不会写入。                                |
| `--viewer`                                                  | 可选，生成时打开 MuJoCo Viewer。                                              |

使用该数据集启动离线 Learner：

当前已将旧 7 维示范投影为 `[dx, dy, dz, gripper]` 4 维纯状态数据集，位于 `outputs/mujoco/hardEncodedDemosXYZ/insert`。如需重新执行投影：

```bash
python hardEncodedScripts/project_xyz_insertion_dataset.py --source outputs/mujoco/hardEncodedDemos/insert --output-root outputs/mujoco/hardEncodedDemosXYZ
```

```bash
share-learner --env.type=mujoco_ur5e_insertion --env.episode_steps=300 --env.state_only_policy=true --dataset.repo_id=local/mujoco-hard-encoded-insertion-xyz --dataset.root=outputs/mujoco/hardEncodedDemosXYZ --policy.type=sac_dagger_bc --policy.device=cuda --policy.storage_device=cpu --policy.online_steps=5000 --policy.offline_buffer_capacity=10000 --policy.bc_lr=0.0003 --policy.bc_loss_type=mse --output_dir=outputs/mujoco/hardEncodedRunXYZ --job_name=mujoco-hard-encoded-insertion-xyz --batch_size=128 --num_workers=0 --save_freq=1000 --log_freq=500 --wandb.enable=false
```

这里必须使用 `sac_dagger_bc` 才会在没有在线 Actor 的阶段对示范动作执行行为克隆。`state_only_policy=true` 仅从 31 维机器人本体状态学习；当前该向量不含孔位或带符号的工件相对孔位。投影数据集不包含相机帧，因此加载 9012 帧约需 1 秒。

### 1.6 采集六维力引导的搜索与回退示范

`hardEncodedScripts/generate_mujoco_force_search_demos.py` 参考 ConnTact 的插入搜索状态机，但适配当前“位置接口 + 方形摩擦夹持工件”：接近孔口直到检测到轴向接触，回退最多 3 mm，再移动到离散外扩螺旋的下一个点。当孔沿与工件底部接触时，策略会使用带符号的 `Ty/Tz`、横向力和传感器到工件端部的轴向杠杆距离估计横向接触点，并将该方向以 10% 权重融入下一个螺旋目标。工件进入孔后，再用弯矩方向和横向力卸载的导纳式外环继续插入。

对于局部插入轴 `X`，使用已知轴向杠杆 `r_x` 补偿横向力造成的弯矩：`r_y=(r_x F_y-T_z)/F_x`，`r_z=(T_y+r_x F_z)/F_x`。弯矩估计只在轴向反力超过门限时启用，并经过低通滤波、偏置限幅和危险力回退保护。

螺旋方向固定为逆时针，避免无记忆、确定性的 MSE 行为克隆在相似观测上同时看到顺时针和逆时针标签并把横向动作平均为零。

```bash
MUJOCO_GL=egl python hardEncodedScripts/generate_mujoco_force_search_demos.py --episodes=100 --seed=20260825 --output-root=outputs/mujoco/hardEncodedMomentGuidedVisual64Demos --max-attempts=3 --episode-steps=1300
```

| 参数                                                                  | 含义                                                                      |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `--episodes=100`                                                    | 生成或续跑到总计 100 个成功 episode。                                     |
| `--seed=20260825`                                                   | 孔位估计偏差、接近速度和接触力的确定性种子；搜索方向固定为逆时针。        |
| `--output-root=outputs/mujoco/hardEncodedMomentGuidedVisual64Demos` | 新的弯矩导向数据集目录；不要与旧 wrench 坐标定义的数据混合。              |
| `--max-attempts=3`                                                  | 每个候选起点最多重试 3 次；不可恢复的候选会被跳过，只保存物理成功的轨迹。 |
| `--episode-steps=1300`                                              | 给最慢的多点触觉搜索保留足够控制周期。                                    |
| `--viewer`                                                          | 可选；打开 MuJoCo Viewer 查看搜索和六维力曲线。                           |

当前已确认该 ForceSearch 数据集包含 100 个 episode、75204 帧，实际数据目录为 `outputs/mujoco/hardEncodedMomentGuidedVisual64Demos/insert`。

使用该视觉与 wrench 数据集进行离线行为克隆：

```bash
MUJOCO_GL=egl share-learner --env.type=mujoco_ur5e_insertion --env.policy_image_size=64 --env.episode_steps=1300 --dataset.repo_id=local/mujoco-moment-guided-visual64 --dataset.root=outputs/mujoco/hardEncodedMomentGuidedVisual64Demos --dataset.video_backend=pyav --policy.type=sac_dagger_bc --policy.device=cuda --policy.storage_device=cpu --policy.online_steps=20000 --policy.offline_buffer_capacity=120000 --policy.bc_lr=0.0003 --policy.bc_loss_type=mse --output_dir=outputs/mujoco/momentGuidedVisual64Run20K --job_name=mujoco-moment-guided-visual64-20k --batch_size=32 --num_workers=4 --save_freq=1000 --log_freq=500 --wandb.enable=false
```

这一步只需启动 `share-learner`，不需要同时启动 Actor。它会读取 64x64 视觉、机器人状态和六维力数据，使用 `sac_dagger_bc` 在 CUDA 上执行 20000 次离线行为克隆更新，每 1000 步保存一次 checkpoint，并将结果写入 `outputs/mujoco/momentGuidedVisual64Run20K`。

这里不要添加 `--env.state_only_policy=true`，否则相机不会进入 policy。六维力仍属于机器人状态输入。启动后出现 `Loading external offline demos for 'insert'`、视频解码进度条和 torchvision 视频接口弃用警告均属正常现象；此时仍在加载和解码数据。出现训练 step 和 `bc_loss` 日志后才表示已经进入梯度训练阶段。数据加载主要使用 CPU，进入训练后 CUDA 利用率才会明显升高。

若 CUDA 显存不足，可将 `--batch_size=32` 改为 `--batch_size=16`。下文提到的约 13--15 GB 占用是系统主内存（RAM），不是 GPU 显存。

纯 `sac_dagger_bc` 会逐帧流式载入离线数据，并且不分配 BC loss 用不到的图像 `next_state` buffer。这使 50 条、38664 帧的双相机数据集可以在约 15 GB 主内存的机器上训练；普通 SAC 仍保留完整 `next_state` 以计算 TD target。

以下是旧版 wrench 坐标定义的 50 条数据实验记录，仅用于复现对照，不要与新数据合并：

```bash
MUJOCO_GL=egl share-learner --env.type=mujoco_ur5e_insertion --env.policy_image_size=64 --env.episode_steps=1300 --dataset.repo_id=local/mujoco-force-search-visual64 --dataset.root=outputs/mujoco/hardEncodedForceSearchCoherentVisual64Demos --dataset.video_backend=pyav --policy.type=sac_dagger_bc --policy.device=cuda --policy.storage_device=cpu --policy.online_steps=20000 --policy.offline_buffer_capacity=120000 --policy.bc_lr=0.0003 --policy.bc_loss_type=mse --output_dir=outputs/mujoco/forceSearchCoherentVisual64Identity50StreamRun20K --job_name=mujoco-force-search-coherent-visual64-identity-50-stream-20k --batch_size=32 --num_workers=4 --save_freq=1000 --log_freq=500 --wandb.enable=false
```

未见起点评估命令：

```bash
MUJOCO_GL=egl python hardEncodedScripts/evaluate_mujoco_insertion_policy.py --checkpoint=outputs/mujoco/forceSearchCoherentVisual64Identity50StreamRun20K/insert/checkpoints/020000/pretrained_model --episodes=5 --episode-steps=1300 --seed=20261101 --start-mode=force-search --result-path=outputs/mujoco/forceSearchCoherentVisual64Identity50StreamRun20K/evaluation_20k_in_distribution.json
```

当前实测结果：20 条数据的 20k checkpoint 在 5 个未见起点上成功 2 次（40%）；50 条数据的 20k checkpoint 为 0/5，但多数失败已从大幅漂移改为停在孔口附近。50 条数据的 10k checkpoint 也为 0/3，并存在漂移。根因是离散螺旋教师的搜索中心和步序是隐藏运行状态，无记忆 MSE policy 在相似 observation 上会将不同阶段动作平均。因此这个 checkpoint 可用于验证链路，但不应被视为可上真机的可靠插入策略。

### 1.7 从离线 BC checkpoint 启动在线 SAC

离线 checkpoint 的 policy 类型是 `sac_dagger_bc`：actor、视觉编码器和归一化处理器已经通过示范数据训练，但 critic、target critic 和 temperature 没有经过 SAC TD 优化。下面的转换将训练模式切换为普通 `sac`，并保留与 SAC 网络兼容的已有参数。它属于 BC actor warm-start，不是完整 SAC 训练状态恢复；critic 在进入在线阶段时仍相当于随机初始化，并由新的优化器开始训练。

先执行一次 checkpoint 转换：

```bash
python -c 'import json,shutil,pathlib; src=pathlib.Path("outputs/mujoco/momentGuidedVisual64Run20K/insert/checkpoints/020000/pretrained_model"); dst=pathlib.Path("outputs/mujoco/momentGuidedVisual64SACWarmStart/pretrained_model"); shutil.copytree(src,dst,dirs_exist_ok=True); p=dst/"config.json"; c=json.loads(p.read_text()); c["type"]="sac"; c["online_steps"]=40000; c["online_step_before_learning"]=100; c.pop("bc_lr",None); c.pop("bc_loss_type",None); p.write_text(json.dumps(c,indent=2)+"\n"); print("SAC warm-start checkpoint:",dst)'
```

终端 1 启动在线 SAC Learner：

```bash
MUJOCO_GL=egl share-learner --env.type=mujoco_ur5e_insertion --env.policy_image_size=64 --env.episode_steps=1300 --env.teleop_mode=none --dataset.repo_id=local/mujoco-moment-guided-visual64 --dataset.root=outputs/mujoco/hardEncodedMomentGuidedVisual64Demos --dataset.video_backend=pyav --policy.path=outputs/mujoco/momentGuidedVisual64SACWarmStart/pretrained_model --policy.device=cuda --policy.storage_device=cpu --policy.online_steps=40000 --policy.online_step_before_learning=100 --policy.offline_buffer_capacity=120000 --policy.online_buffer_capacity=100000 --output_dir=outputs/mujoco/momentGuidedVisual64OnlineSAC40K --job_name=mujoco-moment-guided-online-sac-40k --batch_size=32 --num_workers=4 --save_freq=1000 --log_freq=500 --wandb.enable=false
```

等终端 1 出现 `[LEARNER] ROBOT_RELEASED` 以及通信服务启动信息后，再在终端 2 启动在线 Actor 和人工干预：

```bash
MUJOCO_GL=glfw share-actor --env.type=mujoco_ur5e_insertion --env.policy_image_size=64 --env.episode_steps=1300 --env.teleop_mode=keyboard --env.viewer=true --env.viewer_camera=free --policy.path=outputs/mujoco/momentGuidedVisual64SACWarmStart/pretrained_model --policy.device=cuda --policy.storage_device=cpu --policy.online_steps=40000 --policy.online_step_before_learning=100 --dataset.repo_id=local/mujoco-moment-guided-visual64 --dataset.root=outputs/mujoco/hardEncodedMomentGuidedVisual64Demos --dataset.video_backend=pyav --output_dir=outputs/mujoco/momentGuidedVisual64OnlineSAC40K --job_name=mujoco-moment-guided-online-sac-40k --wandb.enable=false
```

Learner 和 Actor 的 `policy.path`、`online_steps`、通信端口及 `output_dir` 必须一致。Actor 收集满 100 条在线 transition 后，Learner 才开始 SAC 更新；如需缩短预热，可在两条命令中同时修改 `--policy.online_step_before_learning`。运行时 `/` 标记失败，`Enter` 标记成功，`Esc` 停止整个程序。Learner 默认每 4 秒向 Actor 推送一次参数。带 Viewer 的 Actor 使用 `MUJOCO_GL=glfw`，无窗口 Learner 使用 `MUJOCO_GL=egl`。

## 2. 键盘操作规则

| 功能         | 按键                        | 说明                                                                  |
| ------------ | --------------------------- | --------------------------------------------------------------------- |
| X 正/负平移  | `←` / `→`             | 按住连续移动                                                          |
| Y 正/负平移  | `↓` / `↑`             | 按住连续移动                                                          |
| Z 正/负平移  | `右 Shift` / `左 Shift` | 按住连续移动                                                          |
| 打开夹爪     | `.`                       | 保持打开目标                                                          |
| 关闭夹爪     | `,`                       | 保持闭合目标                                                          |
| 人工失败     | `/`                       | 立即结束本轮；离线录制丢弃本轮，在线采集保留为`reward=0` 的失败样本 |
| 人工成功     | `Enter`                   | 可选标记；立即以`reward=1` 结束本轮并进入下一轮                     |
| 停止整个程序 | `Esc`                     | 停止录制、Actor 或 MuJoCo Demo                                        |

该映射与父目录 LeRobot 的 `KeyboardEndEffectorTeleop` 保持一致。字母运动键和键盘姿态旋转已禁用；按住运动键持续移动，松开即停止。夹爪按键设置并保持目标开合状态。人工失败不会污染离线示范数据，但在线 Actor 会将其作为有价值的负样本同步给 Learner。

夹爪 backend 默认启用 `0.5 s` 命令安全间隔：相同目标不会重复发送，开合目标发生变化后，新的不同目标至少等待 0.5 秒才会执行。该限制同时覆盖键盘和 SAC 输出，并在 primitive 切换期间保持有效。可通过 `--env.gripper_min_command_interval_s=0.5` 调整，但真机不建议降低。

## 3. 当前插入成功判定

当前使用红色工件相对蓝色孔位的几何量判断，不再使用夹爪 TCP 的固定 Z 阈值。默认要求插入深度至少 `0.07 m`，同时满足横向误差和轴线对齐边界。

满足条件后执行：

```text
insert → release
reward += 1.0
transition_reason = "peg_inserted"
```

插入阶段超过 `episode_steps` 也会进入 `release`，但该路径属于超时，不代表插入成功。

当前判定仅检查末端高度，不能严格证明工件已经进入插孔。更可靠的成功判定应同时考虑：

- 工件与插孔的 XY 对齐误差；
- 工件插入深度；
- 工件姿态误差；
- 接触状态或末端六维力矩范围。
