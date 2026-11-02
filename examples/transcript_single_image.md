# 示例：单图计数（离线 provider 生成）

> 本文件由 `--provider mock` 的脚本化 provider 生成，用于验证工具契约与决策链结构，
> 不代表真实模型的判断能力。接入真实模型后的运行记录会替换这里。

# 运行记录

- 问题：这张图里有多少株甘蓝？长势密不密？
图片路径：<检测仓库>/dataset/test/images/img_105.jpg
- 模型：`mock-scripted`
- 工具调用：2 次（失败 0 次）
- 耗时：0.1s
- 终止原因：`stop`

## 决策链

**1. `detect_cabbage` ✓**  (54ms)
   参数：`image_path=<检测仓库>/dataset/test/images/img_105.jpg`, `conf=0.25`
   观测：`{"count": 49, "density": "moderate", "coverage": 0.448, "conf_threshold": 0.25, "image_id": "b5d3658e433e"}`

**2. `refine_detection` ✓**  (31ms)
   参数：`image_id=b5d3658e433e`, `conf=0.15`
   观测：`{"count": 50, "density": "moderate", "comparison": {"before": {"conf": 0.25, "count": 49}, "after": {"conf": 0.15, "count": 50}, "delta": 1}, "stability": "stable", "image_id": "b5d3658e433e"}`

## 结论

该图共 50 株，密度 moderate。在 conf=0.25 时检出 49 株，降到 conf=0.15 后为 50 株（差 +1，稳定性 stable）。结论取复核后的计数。
