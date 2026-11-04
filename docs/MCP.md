# MCP：把工具表交给客户端

这个项目前面的几层解决的是"能力"——能测、能问、能在界面上看见框。这一层解决的是
"别人能不能用"。

`cvagent mcp` 起一个 stdio MCP 服务器，把六个工具原样暴露出去。任何支持 MCP 的
客户端（Claude Desktop、Cursor、IDE 插件、自己写的 agent）都能把它当成一组原生
工具，不需要知道这里有 CLI、有 Web 界面、更不需要知道"甘蓝"这个词。

## 接入

先确保装了：

```bash
pip install -e .          # 提供 cvagent 命令
```

客户端配置（Claude Desktop 的 `claude_desktop_config.json`、Cursor 的
`mcp.json` 结构相同）：

```json
{
  "mcpServers": {
    "cabbage-vision": {
      "command": "cvagent",
      "args": ["--device", "cuda:0", "mcp"]
    }
  }
}
```

没装成包、想直接跑源码的话，把 `command` 换成解释器、用 `env` 补 `PYTHONPATH`：

```json
{
  "mcpServers": {
    "cabbage-vision": {
      "command": "D:/cabbage_project/.venv/Scripts/python.exe",
      "args": ["-m", "cvagent", "--device", "cuda:0", "mcp"],
      "env": { "PYTHONPATH": "D:/cabbage-vision-agent/src" }
    }
  }
}
```

注意 `--device` 是**全局参数，必须写在 `mcp` 前面**。写成 `cvagent mcp --device cuda:0`
会被 argparse 判成"无法识别的参数"——这条踩过，客户端只会显示"服务器启动失败"。

不想让客户端等权重加载的话，加 `--warmup`：启动即加载（然后 `initialize` 也会慢）。

## 实测时序（RTX 5060 Laptop，RTX 4050 级别的卡同样量级）

```
initialize   0.26s  proto=2025-06-18  server={'name':'cabbage-vision','version':'0.1.0'}
tools/list   0.02s  共 6 个
tools/call  31.15s  isError=False  count=62 density=dense     ← 首次调用含权重加载
坏路径       isError=True code=IMAGE_NOT_FOUND retryable=False
shutdown     {}
```

复现：`python scripts/mcp_probe.py --device cuda:0`。这个脚本自己拼 JSON-RPC、
不用 MCP SDK——要验的正是帧边界和失败形状，用 SDK 等于让两边的解析器一起错。
加 `--skip-call` 只验握手与工具表，不加载权重。

**首次调用 31 秒**是这台机器上加载+预热权重的代价，后续每次 0.1 秒级。之所以不让
`initialize` 承担它，是因为客户端对握手的超时远比对单次工具调用严格：握手慢会直接
判定"服务器起不来"，工具调用慢最多是转一会儿圈。

## 三个必须做对的约定

**1. 工具表只有一份。**

`tools/list` 的 `description` 和 `inputSchema` 直接从 `tools/spec.py` 的 `Tool`
对象取，不另写一份给 MCP。喂给模型的 `description` 就是"什么时候该调用我"的判据
——两份描述一漂移，同一个问题在不同客户端里就会走出两种行为，而且**不会报错**。
这个项目已经在别处踩过同类坑（评测的禁语正则、抽取与抹除的路径正则），所以这里
连 schema 都是从同一个 `Tool` 映射出来的：

```python
{"name": fn["name"], "description": fn["description"], "inputSchema": fn["parameters"]}
```

**2. stdout 只放协议消息。**

stdio 传输是一行一个 JSON-RPC，任何一句残留的 `print` 都会破坏帧边界。客户端的
表现是"连上了但不回话"，而日志里什么都不像出问题。所以启动时先把真正的 stdout
收进变量，再把 `sys.stdout` 指向 stderr：

```python
proto_out = sys.stdout
sys.stdout = sys.stderr     # 库里的 print 顶多污染日志，不会污染协议
```

顺序不能反：抢管道之前只要有东西 print 过一行，协议流里就多一帧垃圾。

**3. 权重不在握手里加载。**

注册表惰性构造（第一次真需要时才建），权重更晚（`YoloEngine` 的模型是首次推理
才加载）。所以 `tools/list` 完全不碰 GPU——实测 0.02 秒。

## 一个语义选择：工具失败走 `isError`，不走 JSON-RPC error

`dispatch()` 从不抛异常，失败一律是 `{"ok": false, "error": {...}}`。放到 MCP 里
有两种写法，选了后者：

| 写法 | 客户端表现 | 模型能不能自我纠正 |
|---|---|---|
| JSON-RPC `error` | 这次请求被当成"不成立" | 不一定看得到内容 |
| `result.isError = true` + 文本里带 payload | 工具失败，但内容照常给模型 | 能，`hint` 就在里面 |

这个项目的整个循环建立在"错误是输入"上（参数越界 → `INVALID_ARG` → 改参数重试），
所以 `isError` 必须带着 payload 一起回：

```
坏路径  isError=True  code=IMAGE_NOT_FOUND  retryable=False
```

模型看到 `retryable=false` 就知道别换参数重试了，直接如实告诉用户。协议层的 error
只留给**请求本身不成立**的情况：没有这个工具、arguments 不是对象、方法不存在。

## 局限（一起说）

- **单连接、顺序处理。** stdio 就是一条流。要并发用 `cvagent serve`，那层的锁只包推理。
- **路径是本机的。** 工具吃 `image_path`，客户端和服务器必须在同一台机器上（或同一份挂载）。
  这和 Web 层刻意相反：那边不允许前端指定路径，这边路径就是调用方给的。
- **没有鉴权。** stdio 由父进程启动，信任边界就是进程边界。别把它包进网络端口。
- **不提供 resources / prompts。** 这个项目的能力全在"动作"里，硬凑成资源只是把 JSON 换壳。
- **不支持 `listChanged`。** 工具表编译期固定，发一个永不触发的通知不如不发。
- **首次调用慢。** 用 `--warmup` 把这份代价挪到启动，或者接受第一次转圈。
