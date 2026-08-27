# WebConfig 修复与验证记录

## 已修复的问题

1. 修正 debug summary 的字段读取：节点角色来自 `primitive.roles`，起始、复位和终止样式现在能正确显示。
2. 后端把无界位姿中的 `Infinity` 转换成标准 JSON 的 `null`，浏览器 `response.json()` 不再失败。
3. 移除不存在的 `on_info_equals`，转移类型和 dataclass 默认值改为从仓库注册表动态加载。
4. 全局设置在“保存”和“验证”前都会同步，避免验证旧值。
5. 节点属性和边参数改为可编辑；一次完整 PUT 完成验证和保存，不会出现半次编辑已落盘。
6. 新增节点时同时构造必要的入边/出边，以满足 MP-Net 的可达性和非终止死胡同校验。
7. 修复原语类型选择无效的问题；`static`、`move_delta`、`open_loop_trajectory` 均会生成对应 payload。
8. 配置名增加白名单校验，所有 URL 名称进行编码，页面不再把配置内容拼入可执行 HTML。
9. 保存改为临时文件原子替换；无效编辑保持原文件不变。
10. 服务默认改为 `127.0.0.1`、关闭 debug/reloader，避免后台出现父子两个 Flask 进程。
11. “添加转移边”改为动态选择表单：轴、比较方式、设备、图像尺寸、步数键和成功键均可直接选择，无需手写 JSON。
12. 点击已有转移边也使用同一套选择表单，可修改起点、终点、类型及对应参数，不再编辑原始 JSON。

## 自动验证

执行日期：2026-08-27。

```text
node --check webConfig/static/app.js                         PASS
python -m py_compile webConfig/server.py webConfig/test_server.py  PASS
python -m unittest -v webConfig/test_server.py               8/8 PASS
```

API 测试覆盖健康检查、CRUD、全量保存回环、结构化编辑、失败不落盘、名称校验、动态元数据及严格 JSON。

另外执行了 6 个仓库 MP-Net/workspace 测试文件：结果为 `22 passed, 4 failed`。4 项失败都来自当前源文件把校验异常文案改成中文，而测试仍匹配旧英文文案（unknown source、unknown target、dead-end、unreachable）；与 `webConfig/` 行为无关，本次按“不得修改状态机源文件”的约束未改动它们。

真实配置冒烟检查：

```text
pick_insert_example: 12 primitives, 12 transitions, strict JSON PASS
test_custom_config:    1 primitive,   0 transitions, strict JSON PASS
```

## 浏览器手工检查

1. 打开 <http://127.0.0.1:5050>，确认左侧出现两个配置。
2. 打开 `pick_insert_example`，确认图中有 12 个节点和 12 条边，`move_above_A` 为绿色起始边框，`done` 为红色终止边框。
3. 点击节点，修改说明或位姿后保存并重新加载。
4. 点击边，确认起点、终点、类型及参数均正确预填；切换类型时参数选项同步变化。
5. 新建配置，添加终止节点及入边，再验证、保存、重新加载和删除。

当前系统的 `/usr/bin/chromium-browser` 是未安装 Chromium Snap 的占位脚本，因此本次无法执行 headless 浏览器截图；页面脚本已通过语法检查，静态资源和全部 API 已从真实 5050 服务验证。

## 序列化说明

WebConfig 保存为便于浏览器编辑的 flat JSON。公开的 load_mpnet_config(path) 会自动识别这种格式，调用 flat 解码和正式校验后返回 ManipulationPrimitiveNetConfig；带顶层 type 的既有 Draccus 配置仍走原加载路径。运行程序不会自动扫描 webConfig/configs/，调用方需要显式传入所选 JSON 路径。
