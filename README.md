# cabbage-vision-agent

把甘蓝航拍检测封装成**可编排的视觉 Agent**：自然语言提问进，带依据的计数结论出。

检测模型来自 [cabbage-detection](https://github.com/3086-x/cabbage-detection)（YOLOv11n，单类，mAP50-95 0.970）。
那个仓库回答"用哪个结构"，这个仓库回答另一个问题：

> **模型能力怎么变成 Agent 能可靠调用、且在出错时能自我纠正的工具？**

能跑通 YOLO 的人很多，能说清工具契约、错误语义、上下文预算的人不多。这个项目只做后者。

---

## 一次真实运行

问：*这张图里有多少株甘蓝？长势密不密？*

```
[1] detect_cabbage  ✓  54ms   count=49  density=moderate  n_low=2
[2] refine_detection ✓  31ms   49 → 50  (conf 0.25 → 0.15)  stability=stable
```

回答：*该图共 50 株，密度 moderate。conf=0.25 时检出 49 株，降到 0.15 后为 50 株
（差 +1，稳定性 stable）。结论取复核后的计数。*

注意第二行**不是人写死的流程**。`detect_cabbage` 返回 `n_low=2`（2 个低置信框），
模型据此判断"这个数字可能不稳"，自己决定降阈值复核，再把两次结果对比后给出结论。
完整轨迹见 [examples/transcript_single_image.md](examples/transcript_single_image.md)。

---

## 四个设计决策

这四件事决定了它是 Agent 而不是 API wrapper。

### 1 工具返回证据，不返回图片

一张标注图对 LLM 毫无意义——它没法从红色方框里数出数值。所以工具返回结构化观测：

```json
{
  "count": 49, "density": "moderate", "coverage": 0.448,
  "conf_stats": {"min": 0.31, "median": 0.82, "max": 0.874, "n_low": 2},
  "box_size_px": {"w_median": 103.7, "h_median": 104.1},
  "boxes_head": [[977.2, 47.8, 93.6, 95.6, 0.874], ...],
  "image_id": "b5d3658e433e"
}
```

`n_low` 是关键字段：它把"这个计数有多可信"变成了模型可读的信号。
没有它，模型只能照抄数字。

### 2 错误是输入，不是异常

工具永不抛异常，失败一律变成结构化观测：

| code | retryable | 模型该怎么做 |
|---|---|---|
| `INVALID_ARG` | ✓ | 改参数重试，不用换图 |
| `NO_DETECTION` | ✓ | 阈值过高，降到 0.10 再试一次 |
| `IMAGE_NOT_FOUND` | ✗ | 路径错了，重试无意义，直接告知用户 |
| `UNSUPPORTED_FORMAT` | ✗ | 格式不支持，告知用户转 jpg |
| `ENGINE_OOM` | ✗ | 显存不足，建议换 CPU |

`retryable` + `hint` 两个字段让自纠正变成可能。人类预设的重试策略在真实任务里
经常比模型差，所以循环层**不拦截错误**，只负责把错误原样送进上下文。

### 3 阈值是参数，不是常量

`conf` 暴露在工具签名里，而不是写死在代码里。这让"要不要复核"成为模型的决策，
而不是开发者的硬编码。`refine_detection` 复用缓存，复核只要 31ms。

### 4 观测有预算

单次工具返回限制 1400 字符（`detail=full` 的检测放宽到 7000）。超预算时
砍掉长列表、保留结论字段，并显式告知 `truncated: true` —— 让模型知道信息不全，
而不是默默拿到半截数据。

历史超过 6 条观测时，旧观测替换成一行摘要。**只替换 content，不动结构**——
删消息会破坏 `tool_call_id` 配对，多数供应商直接返回 400。

---

## 快速开始

```bash
git clone https://github.com/3086-x/cabbage-vision-agent.git
cd cabbage-vision-agent
pip install -e .

# 权重（5.5MB，不走 Git，放在 Release）
# 也可以从检测仓库复制：cvagent setup-weights --from ../cabbage-detection/weights/best.pt

export DASHSCOPE_API_KEY=sk-xxx        # 或 OPENAI_API_KEY / DEEPSEEK_API_KEY
cvagent doctor --probe                 # 先自检：权重、torch、工具、LLM
cvagent ask "这张图有多少株甘蓝？" --image path/to/img.jpg
```

**没有 API key 也能完整跑通决策链**：

```bash
cvagent ask "这张图有多少株？" --image img.jpg --provider mock
```

`mock` 是脚本化 provider，按固定策略驱动工具。它的作用不是"假装有 LLM"，
而是让**决策链本身可测试**——工具契约、错误传播、观测截断、循环终止，
这些逻辑与模型无关，不该等到接上模型才能验证。

---

## 工具

| 工具 | 何时调用 | 返回 |
|---|---|---|
| `inspect_image` | 先判断图能不能用（太暗/太糊/过小） | 尺寸、格式、亮度、边缘强度 |
| `detect_cabbage` | 要株数 + 密度 + 置信度分布 | 统计 + 前 3 个框（或全量） |
| `count_cabbage` | 只要一个数字，要最省 token | 株数、密度、低置信框数 |
| `refine_detection` | 低置信框多，或用户质疑准确性 | 两次阈值对比 + 稳定性判定 |
| `batch_detect` | 给的是文件夹 | 总数、均值、极值、逐文件 |

`cvagent tools --json` 输出完整 JSON Schema。

---

## 可验证性

```bash
python -m unittest discover -s tests    # 34 个测试，0.8 秒
```

测试**不加载权重、不联网**，在任何机器上都能跑。覆盖的是契约而非模型行为：

- 未知工具 / 缺参数 / 参数越界 / 非法 JSON → 返回正确的 `code` 且不抛异常
- 扩展名是 `.jpg` 但内容是文本 → 靠魔术字节抓住，且在模型加载前就失败
- `detail=full` 超预算 → 保留 `count`，砍掉 `boxes`，标记 `truncated`
- 步数打满 → 不强杀，发一次不带 tools 的请求逼出结论，`stopped_reason=max_steps`
- 历史压缩 → 所有 `tool_call_id` 保持配对，最新一条观测保持完整

---

## 结构

```
src/cvagent/
├─ vision/     纯推理，不感知 LLM。懒加载、先校验后推理、按 image_id 缓存
├─ tools/      工具契约与分发。参数校验、错误转 payload、观测截断
├─ llm/        供应商抽象。base 定义 Message/ToolCall/Completion；
│              openai_compat 收敛所有协议差异；mock 提供离线驱动
├─ agent/      规划-执行-观察循环，含步数收束与历史压缩
├─ report.py   轨迹落盘（trajectory.json + report.md）
└─ cli.py      ask / detect / doctor / tools / replay
```

不引 langchain。全部需求是"多轮消息 + function calling"，120 行客户端就够，
链式框架只会让可调试性变差。

---

## 与检测仓库的关系

```
cabbage-detection          ← 证明「为什么选 YOLOv11n、为什么不做注意力机制」
        │  结论被引用
        ▼
cabbage-vision-agent       ← 把这份视觉能力暴露成 Agent 可编排的工具
```

选型依据是那边的实测结论：注意力模块（SE/CBAM/CA）在三种子下均无可分辨增益，
所以这里**固定基线结构，把复杂度全部投入编排层**。两个仓库合起来讲的是一条
"证据驱动选型 → 应用落地"的链条，而不是两个孤立 demo。

---

## 局限

- **密度分级是相对本数据集的**。训练集逐图株数四分位为 45/49/53，图来自同一块田，
  分布天然集中，`sparse/moderate/dense` 只有相对意义，不是农学绝对值。
- **无法换算株/亩**。缺少地面标定（GSD），任何"每亩多少株"的说法都需要先做标定。
- **单类单场景**。只认甘蓝，只在这种航高与光照下验证过。
- **工具调用正确率未做量化评测**。当前只有契约测试；真实模型的调用正确率、
  错误恢复率需要一套任务集才能给出，见 [docs/DESIGN.md](docs/DESIGN.md) 的评测计划。

---

## 许可

MIT（仅覆盖代码与文档）。权重与数据集不随仓库分发。
