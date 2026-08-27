# SHaRe-RL 本地实验控制台

这是一个独立的 Flask + Vanilla JS 应用，用于在一个页面内管理本地 SHaRe-RL 实验。主页提供实验概况、Actor/Learner 控制、Replay Buffer 监控、算法参数和 MP-Net 状态机编辑器。

## 启动

从仓库根目录执行：

```bash
cd webConfig
/home/peterpan/miniconda3/bin/python server.py
```

浏览器打开 <http://127.0.0.1:5050>。状态机编辑器位于 <http://127.0.0.1:5050/editor>。默认仅监听本机且关闭 Flask debug/reloader。可通过以下环境变量覆盖：

- `MPNET_WEB_HOST`
- `MPNET_WEB_PORT`
- `MPNET_WEB_DEBUG=1`
- `MPNET_WEB_CONFIG_DIR`

## 控制台功能

- “实验总览”集中显示服务状态、状态机配置数量以及 Buffer 关键指标。
- “Actor / Learner”按照 Learner → Actor 的顺序启动服务，支持日志尾部查看和实际命令预览。
- “Replay Buffer”从 Learner 的本地 dashboard API 读取 online/offline transition、轨迹、成功、失败和干预统计。
- “算法参数”保存白名单字段到 `runtime_profile.json`，只影响之后由控制台启动的进程。
- “MP-Net 状态机”嵌入完整的图形编辑器；配置仍保存在 `webConfig/configs/`，可由运行时加载。
- 状态机页的“保存并运行 Viewer”会先保存当前 JSON，再调用白名单中的 MuJoCo 示例；首个已验证映射是 `pick_insert_example` 使用 `examples/demo_pick_insert.py --viewer --config=...`。

控制台不会接受任意 shell 命令。输出目录必须位于仓库内，Learner/Replay 地址仅允许本机。Actor 只有在 Learner 已由本控制台启动后才能启动；停止时必须先停 Actor。浏览器需要确认后才会发送启停请求。

Viewer 试运行同样使用参数数组而非 shell，只允许后端登记的示例。为避免争用 MuJoCo/机器人资源，Actor 或 Learner 运行期间不能启动状态机试运行。页面提供 PID、退出状态、日志和进程组停止按钮。

服务状态只跟踪当前 Web 服务器进程启动的子进程。关闭 Web 服务器后重新打开页面，不会接管其他终端手工启动的 Actor/Learner。

## 状态机编辑器

- 图中的蓝色节点是脚本原语，黄色节点包含可学习轴；绿色和红色边框分别表示起始、终止节点。
- 起始原语、复位原语和 FPS 在左侧修改，点击“保存”后统一校验并写盘。
- 点击节点可编辑说明、终止标记、目标位姿和可学习轴。
- 添加转移边时按类型显示下拉框、选项和数值输入，不需要手写 JSON。
- 点击已有边可直接选择起点、终点和转移类型，并通过同一组结构化字段修改参数。
- 新节点必须从现有节点创建入边，否则它无法从起始节点到达；非终止节点还必须创建出边。
- 转移类型及默认参数直接取自 `share.workspace.mpnet.TRANSITION_TYPES`，页面不会维护另一份硬编码列表。
- 保存使用临时文件原子替换。验证失败时不会覆盖原配置。
- `±Infinity` 位姿边界在 HTTP 和文件 JSON 中表示为 `null`；MP-Net 解码器会恢复默认无界限制。

## 验证

```bash
node --check webConfig/static/app.js
node --check webConfig/static/console.js
/home/peterpan/miniconda3/bin/python -m py_compile webConfig/server.py
/home/peterpan/miniconda3/bin/python -m py_compile webConfig/console_runtime.py
/home/peterpan/miniconda3/bin/python -m unittest -v webConfig/test_server.py
```

完整修复记录与手工检查步骤见 [walkthrough.md](walkthrough.md)。
