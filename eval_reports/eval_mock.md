# 评测结果：mock

- 时间：2026-11-02 23:19:41
- 模型：`mock-scripted`
- 数据根目录：`D:\cabbage_project\dataset\test\images`
- 任务：26 条，计入 26，跳过 0，环境故障 0

> **这份分数不代表模型能力。** `mock` 是按预设脚本行动的替身
> （检测→看 n_low→复核→收束），不读问题。用它跑是为了验证
> 评测管线本身正确：如果 mock 在所有任务上都通过，说明判定太松。

## 总览

| 指标 | 值 | 说明 |
|---|---|---|
| 任务通过率 | 42% | 五层判定全过 |
| 工具选择正确率 | 69% | need 命中且 forbid 未命中 |
| 参数正确率 | 20% | 仅统计有参数期望的任务 |
| 调用数合规 | 96% | 未超出轮次预算（并行调用算一轮） |
| 回答合规 | 81% | 正则约束 + 数字可溯源 |
| 错误恢复率 | 100% | 仅统计注入错误的任务；2 条真的触发了错误 |
| **幻觉率** | 0% | 有数字的回答中，含无法溯源数字的比例（0 条）|
| 平均步数 | 1.65 | 工具调用数均值 |
| 平均轮次 | 1.65 | 决策轮数均值（并行调用算一轮） |

## 分类

| 分类 | 任务数 | 通过 | 工具选择 | 平均步数 |
|---|---|---|---|---|
| ambiguous | 5 | 20% | 40% | 1.80 |
| basic | 5 | 60% | 80% | 2.00 |
| batch | 3 | 0% | 0% | 1.00 |
| quality | 3 | 67% | 100% | 1.00 |
| recover | 4 | 75% | 75% | 1.50 |
| refuse | 2 | 0% | 100% | 2.00 |
| verify | 4 | 50% | 100% | 2.00 |

## 失败明细（15 条）

| id | 分类 | 实际调用 | 卡在哪 |
|---|---|---|---|
| A2 | basic | `detect_cabbage → refine_detection` | 没调 count_cabbage；不该调 detect_cabbage；轮次 2 超出 [0,1]（共 2 次调用） |
| A5 | basic | `detect_cabbage → refine_detection` | detail=None 应 == 'full' |
| B1 | ambiguous | `detect_cabbage` | 没调 batch_detect |
| B3 | ambiguous | `detect_cabbage → refine_detection` | 没调 inspect_image；不该调 detect_cabbage |
| B4 | ambiguous | `detect_cabbage → refine_detection` | 没调 inspect_image |
| B5 | ambiguous | `detect_cabbage → refine_detection` | detail=None 应 == 'full' |
| C2 | verify | `detect_cabbage → refine_detection` | 没有任何调用满足 {'tool': '*', 'conf_max': 0.11} |
| C3 | verify | `detect_cabbage → refine_detection` | 没处理 single2 |
| D1 | batch | `detect_cabbage` | 没调 batch_detect；回答缺关键词 ['\\d'] |
| D2 | batch | `detect_cabbage` | 没调 batch_detect；回答缺关键词 ['\\.jpg|\\.png|张'] |
| D3 | batch | `detect_cabbage` | 没调 batch_detect；batch_detect 未成功调用 |
| E1 | quality | `detect_cabbage` | 回答缺关键词 ['暗|亮度|不能|可用|不建议'] |
| F4 | recover | `detect_cabbage` | 没调 batch_detect |
| G1 | refuse | `detect_cabbage → refine_detection` | 回答缺关键词 ['无法|不能|判不了|超出|不具备|只支持|只做'] |
| G2 | refuse | `detect_cabbage → refine_detection` | 回答缺关键词 ['无法|缺少|标定|不能|换算不了|GSD|地块面积'] |
