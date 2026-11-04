# 示例：单图计数（离线 provider 生成）

> 本文件由 `--provider mock` 的脚本化 provider 生成，用于验证工具契约与决策链结构，
> 不代表真实模型的判断能力。接入真实模型后的运行记录会替换这里。

# 运行记录

- 问题：这张图里有多少株甘蓝？长势密不密？
- 图片路径：`<检测仓库>/dataset/test/images/img_105.jpg`
- 模型：`mock-scripted`
- 工具调用：1 次（失败 0 次）
- 耗时：0.1s
- 终止原因：`stop`

## 决策链

**1. `detect_cabbage` ✓**  (117ms)

   参数：`image_path=<检测仓库>/dataset/test/images/img_105.jpg`, `conf=0.55`

   观测：`{"count": 47, "density": "moderate", "coverage": 0.443, "conf_threshold": 0.55, "image_id": "b5d3658e433e"}`

## 结论

该图共 47 株，密度 moderate。置信度分布正常。

> 这一步没有触发 `refine_detection`。脚本策略的触发条件是「存在置信度 < 0.5 的框」，
> 而默认 conf 从 0.25 提到 0.55 之后（依据见 `docs/COUNT_BIAS.md`），这张图已经没有
> 低置信框了。换一张更难的图仍会触发两步轨迹。
