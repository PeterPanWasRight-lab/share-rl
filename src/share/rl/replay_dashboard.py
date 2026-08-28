"""Dependency-free HTTP dashboard for in-memory replay-buffer metrics."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHaRe-RL Replay Buffer</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--card:#131b2e;--line:#26334f;--text:#e8edf7;--muted:#91a0ba;--blue:#58a6ff;--green:#4bd18b;--red:#ff6b7a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}
h1{margin:0 0 5px;font-size:25px}.status{color:var(--muted);margin-bottom:22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}.title{display:flex;justify-content:space-between;margin-bottom:14px}.step{color:var(--blue)}
table{width:100%;border-collapse:collapse}th,td{text-align:right;padding:8px 6px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
.bar{height:7px;background:#202a40;border-radius:9px;overflow:hidden;margin-top:12px}.fill{height:100%;background:var(--blue)}.ok{color:var(--green)}.bad{color:var(--red)}
.note{color:var(--muted);margin-top:20px;line-height:1.6}code{color:#b7d7ff}</style></head>
<body><main><h1>SHaRe-RL Replay Buffer</h1><div id="status" class="status">等待 Learner 数据…</div><div id="grid" class="grid"></div>
<div class="note">轨迹数按当前 buffer 中的 <code>done/truncated</code> transition 统计。Offline 人工干预数据可能只是纠正片段，并不保证构成完整轨迹。页面每秒刷新，数据只存在 Learner 内存中，不写入指标文件。</div></main>
<script>
const cols=[['transitions','Transitions'],['completed_trajectories','轨迹'],['partial_trajectory_fragments','片段'],['interventions','干预'],['successes','成功'],['terminal_failures','失败']];
function table(name,x){return `<div><b>${name}</b><table><thead><tr>${cols.map(c=>`<th>${c[1]}</th>`).join('')}<th>容量</th></tr></thead><tbody><tr>${cols.map(c=>`<td class="${c[0]=='successes'?'ok':c[0]=='terminal_failures'?'bad':''}">${x[c[0]]??0}</td>`).join('')}<td>${x.transitions??0}/${x.capacity??0}</td></tr></tbody></table><div class="bar"><div class="fill" style="width:${x.fill_percent??0}%"></div></div></div>`}
async function refresh(){try{const r=await fetch('/api/metrics',{cache:'no-store'});const d=await r.json();const age=Math.max(0,Date.now()/1000-d.updated_at_unix);document.getElementById('status').textContent=`更新：${d.updated_at} · ${age.toFixed(1)} 秒前`;document.getElementById('grid').innerHTML=Object.entries(d.primitives||{}).map(([id,m])=>`<section class="card"><div class="title"><b>Primitive: ${id}</b><span class="step">optimization ${m.optimization_step}</span></div>${table('Online buffer',m.online)}${table('Offline buffer',m.offline)}</section>`).join('')}catch(e){document.getElementById('status').textContent='Learner 连接失败：'+e}}setInterval(refresh,1000);refresh();
</script></body></html>"""


class ReplayDashboardServer:
    def __init__(self, host: str, port: int):
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {"updated_at_unix": 0.0, "updated_at": "-", "primitives": {}}
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/metrics":
                    with dashboard._lock:
                        body = json.dumps(dashboard._payload).encode()
                    content_type = "application/json; charset=utf-8"
                elif self.path in {"/", "/index.html"}:
                    body = _HTML.encode()
                    content_type = "text/html; charset=utf-8"
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="replay-dashboard", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread.start()

    def update(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payload = payload

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

