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

默认将 Menagerie 的位置伺服 Kp 放大 16 倍，并将 Kd 放大 `sqrt(16)` 倍以保持阻尼关系；可通过 `position_servo_stiffness_scale` 调整。该参数只改变 MuJoCo 内部位置控制器刚度，不改变 SAC 动作含义，也不会把接口变成力矩控制。

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
share-learner --env.type=mujoco_ur5e_insertion --dataset.repo_id=local/mujoco-insertion --dataset.root=outputs/mujoco/offline-demos --output_dir=outputs/mujoco/run --job_name=mujoco-insertion --batch_size=128 --wandb.enable=false
```

| 参数                                            | 含义                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------- |
| `share-learner`                               | 启动 SAC Learner 和 actor/learner 通信服务。                                       |
| `--env.type=mujoco_ur5e_insertion`            | 使用与采集阶段相同的 MuJoCo 插入环境和策略结构。                                   |
| `--dataset.repo_id=local/mujoco-insertion`    | 指定离线数据集基础标识。                                                           |
| `--dataset.root=outputs/mujoco/offline-demos` | 从该目录的`insert/` 子目录加载离线示范。                                         |
| `--output_dir=outputs/mujoco/run`             | 保存 checkpoint、训练状态和在线 replay 数据的目录。                                |
| `--job_name=mujoco-insertion`                 | 本次训练任务名称，用于日志和运行标识。                                             |
| `--batch_size=128`                            | 每次优化使用 128 条 transition；默认混合训练时约为 64 条在线数据和 64 条离线数据。 |
| `--wandb.enable=false`                        | 不启用 Weights & Biases 在线实验记录。                                             |

Learner 会加载离线数据，在 `127.0.0.1:50051` 启动通信服务，并等待 Actor 发送在线 transition。

### 1.4 启动 Actor

确认 Learner 已启动后，在终端 2 执行：

```bash
share-actor --env.type=mujoco_ur5e_insertion --env.viewer=true --env.viewer_camera=free --dataset.repo_id=local/mujoco-insertion --dataset.root=outputs/mujoco/offline-demos --output_dir=outputs/mujoco/run --job_name=mujoco-insertion --wandb.enable=false
```

| 参数                                            | 含义                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| `share-actor`                                 | 启动 MuJoCo rollout Actor，采集在线 transition 并接收 Learner 参数。 |
| `--env.type=mujoco_ur5e_insertion`            | 使用 MuJoCo UR5e 插入环境。                                          |
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

建议停止顺序为：先停止 Actor，再停止 Learner。

## 2. 键盘操作规则

| 功能 | 正方向 | 负方向 |
|---|---|---|
| X 平移 | `←` | `→` |
| Y 平移 | `↓` | `↑` |
| Z 平移 | `右 Shift` | `左 Shift` |
| 打开夹爪 | `右 Ctrl` | — |
| 关闭夹爪 | `左 Ctrl` | — |
| 退出 | `Esc` | — |

该映射与父目录 LeRobot 的 `KeyboardEndEffectorTeleop` 保持一致。字母运动键和键盘姿态旋转已禁用；按住运动键持续移动，松开即停止。夹爪按键设置并保持目标开合状态。

## 3. 当前插入成功判定

当前通过末端执行器的 Z 高度阈值判断：

```python
main.z.ee_pos <= 0.12
```

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
