import json
from urllib.request import urlopen

from share.rl.replay_dashboard import ReplayDashboardServer


def test_replay_dashboard_serves_html_and_in_memory_metrics():
    server = ReplayDashboardServer("127.0.0.1", 0)
    server.start()
    host, port = server.address
    payload = {"updated_at_unix": 1.0, "updated_at": "now", "primitives": {"insert": {}}}
    server.update(payload)
    try:
        with urlopen(f"http://{host}:{port}/", timeout=2) as response:  # noqa: S310
            assert "SHaRe-RL Replay Buffer" in response.read().decode()
        with urlopen(f"http://{host}:{port}/api/metrics", timeout=2) as response:  # noqa: S310
            assert json.loads(response.read()) == payload
    finally:
        server.close()
