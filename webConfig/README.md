# MP-Net 状态机配置编辑器

这是一个独立的 Flask + Vanilla JS 应用。它只读写 `webConfig/configs/`，不修改 MP-Net 运行时代码。

## 启动

从仓库根目录执行：

```bash
cd webConfig
/home/peterpan/miniconda3/bin/python server.py
```

浏览器打开 <http://127.0.0.1:5050>。默认仅监听本机且关闭 Flask debug/reloader。可通过以下环境变量覆盖：

- `MPNET_WEB_HOST`
- `MPNET_WEB_PORT`
- `MPNET_WEB_DEBUG=1`
- `MPNET_WEB_CONFIG_DIR`

## 功能与约束

- 图中的蓝色节点是脚本原语，黄色节点包含可学习轴；绿色和红色边框分别表示起始、终止节点。
- 起始原语、复位原语和 FPS 在左侧修改，点击“保存”后统一校验并写盘。
- 点击节点可编辑说明、终止标记、目标位姿和可学习轴。
- 添加转移边时按类型显示下拉框、选项和数值输入，不需要手写 JSON。
- 点击已有边仍可通过参数 JSON 完整编辑高级字段。
- 新节点必须从现有节点创建入边，否则它无法从起始节点到达；非终止节点还必须创建出边。
- 转移类型及默认参数直接取自 `share.workspace.mpnet.TRANSITION_TYPES`，页面不会维护另一份硬编码列表。
- 保存使用临时文件原子替换。验证失败时不会覆盖原配置。
- `±Infinity` 位姿边界在 HTTP 和文件 JSON 中表示为 `null`；MP-Net 解码器会恢复默认无界限制。

## 验证

```bash
node --check webConfig/static/app.js
/home/peterpan/miniconda3/bin/python -m py_compile webConfig/server.py
/home/peterpan/miniconda3/bin/python -m unittest -v webConfig/test_server.py
```

完整修复记录与手工检查步骤见 [walkthrough.md](walkthrough.md)。
