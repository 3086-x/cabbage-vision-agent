"""手搓一个 MCP 客户端，验证 stdio 服务器真的能对话。

不引第三方 MCP SDK：这里要验的正是"帧对不对、握手快不快、失败长什么样"，用 SDK
等于拿它的解析器给自己背书——协议错位的时候两边一起错，就测不出来了。

它检查四件事：
    1. initialize 在 1 秒内回，且版本/能力字段齐
    2. tools/list 的 schema 是完整的（有 name / description / inputSchema）
    3. 真调一次工具能拿到株数（这一步会加载权重，慢）
    4. 坏路径走 isError 而不是协议 error，且 error.code 可读

用法：
    python scripts/mcp_probe.py
    python scripts/mcp_probe.py --device cuda:0 --image examples/sample.jpg
    python scripts/mcp_probe.py --skip-call      # 只验握手与工具表，不加载权重
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDSHAKE_BUDGET = 1.0  # 秒。客户端对握手的耐心远低于对单次工具调用


def _send(p: subprocess.Popen, obj: dict) -> None:
    p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    p.stdin.flush()


def _recv(p: subprocess.Popen) -> dict:
    line = p.stdout.readline()
    if not line:
        err = p.stderr.read().strip()
        raise SystemExit(f"服务器中途退出。stderr：\n{err or '（空）'}")
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="MCP stdio 冒烟测试")
    ap.add_argument("--device", default=None, help="cpu / cuda:0")
    ap.add_argument("--image", default="examples/sample.jpg")
    ap.add_argument("--skip-call", action="store_true", help="不调工具，避免加载权重")
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "cvagent"]
    if args.device:
        # 全局参数必须在子命令之前，这是 argparse 的规则，也是实测踩过的坑
        cmd += ["--device", args.device]
    cmd += ["mcp"]

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         bufsize=1, cwd=str(ROOT), env=env)
    fails: list[str] = []

    try:
        t0 = time.perf_counter()
        _send(p, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "mcp-probe", "version": "0"}}})
        r = _recv(p)["result"]
        dt = time.perf_counter() - t0
        print(f"initialize  {dt:5.2f}s  {r['protocolVersion']}  "
              f"{r['serverInfo']['name']} {r['serverInfo']['version']}")
        if dt > HANDSHAKE_BUDGET:
            fails.append(f"握手 {dt:.2f}s 超过 {HANDSHAKE_BUDGET}s 预算")
        if "tools" not in r.get("capabilities", {}):
            fails.append("initialize 没声明 tools 能力")

        _send(p, {"jsonrpc": "2.0", "method": "notifications/initialized"})  # 通知，不回

        t0 = time.perf_counter()
        _send(p, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = _recv(p)["result"]["tools"]
        print(f"tools/list  {time.perf_counter() - t0:5.2f}s  共 {len(tools)} 个")
        for t in tools:
            print(f"  - {t['name']:<17} 必填={t['inputSchema'].get('required', [])}")
            if not t.get("description") or not t.get("inputSchema", {}).get("properties"):
                fails.append(f"{t['name']} 的 description 或 inputSchema 是空的")

        if not args.skip_call:
            t0 = time.perf_counter()
            _send(p, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "count_cabbage",
                                 "arguments": {"image_path": args.image, "conf": 0.25}}})
            res = _recv(p)["result"]
            obs = json.loads(res["content"][0]["text"])
            print(f"tools/call  {time.perf_counter() - t0:5.2f}s  isError={res['isError']}  "
                  f"count={obs.get('count')} density={obs.get('density')}")
            if res["isError"] or not isinstance(obs.get("count"), int):
                fails.append(f"正常图片没拿到株数：{obs}")

            _send(p, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                      "params": {"name": "count_cabbage",
                                 "arguments": {"image_path": "no/such.jpg"}}})
            res = _recv(p)["result"]
            obs = json.loads(res["content"][0]["text"])
            code = obs.get("error", {}).get("code")
            print(f"坏路径      isError={res['isError']} code={code} "
                  f"retryable={obs.get('error', {}).get('retryable')}")
            # 这两条是设计选择，不是实现细节：失败必须能被模型读懂并决定要不要重试
            if not res["isError"]:
                fails.append("不存在的图片没有返回 isError")
            if code != "IMAGE_NOT_FOUND":
                fails.append(f"错误码不是 IMAGE_NOT_FOUND：{code}")

        _send(p, {"jsonrpc": "2.0", "id": 9, "method": "shutdown"})
        _recv(p)
        p.wait(timeout=10)
    finally:
        if p.poll() is None:
            p.kill()

    if fails:
        print("\n失败：")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\n全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
