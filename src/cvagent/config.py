"""集中配置。所有可调项都从环境变量读，便于在不同机器上跑而不改代码。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- 推理 ----
WEIGHTS = Path(os.getenv("CVAGENT_WEIGHTS", ROOT / "weights" / "best.pt"))
DEVICE = os.getenv("CVAGENT_DEVICE", "cpu")  # cpu | cuda:0 | mps
IMG_SIZE = int(os.getenv("CVAGENT_IMGSZ", "1024"))
DEFAULT_CONF = float(os.getenv("CVAGENT_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("CVAGENT_IOU", "0.45"))
MIN_CONF = 0.05  # 低于此值的复核请求无意义，直接判为参数错误

# ---- 密度分级 ----
# 来源：D:\cabbage_project 训练集 262 张逐图标注框数的四分位点
#   min=31  Q1=45  中位=49  Q3=53  max=83
# 注意：这批图来自同一块田，株数天然集中在 45~55，绝对分级几乎无区分度。
# 因此分级只作为**相对本数据集**的粗略标签，不宣称是农学意义上的密度。
# 真正可比的量是 mean_density_per_mpx（株/百万像素），它不依赖分级阈值。
DENSITY_QUARTILES = (45, 53, 65)  # sparse < 45 <= moderate < 53 <= dense < 65 <= very_dense

# ---- Agent 循环 ----
MAX_STEPS = int(os.getenv("CVAGENT_MAX_STEPS", "8"))
MAX_TOOL_CHARS = int(os.getenv("CVAGENT_MAX_TOOL_CHARS", "1400"))  # 单个观测的字符预算
MAX_HISTORY_OBSERVATIONS = int(os.getenv("CVAGENT_MAX_HISTORY", "6"))

# ---- LLM ----
# provider: auto | openai | dashscope | deepseek | mock
LLM_PROVIDER = os.getenv("CVAGENT_LLM", "auto")
LLM_MODEL = os.getenv("CVAGENT_MODEL", "")
LLM_TEMPERATURE = float(os.getenv("CVAGENT_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(os.getenv("CVAGENT_LLM_TIMEOUT", "120"))
# 境外端点常需代理；国内端点直连更强。设成 provider 级参数而不是全局环境变量，
# 避免把 DashScope 的请求也绕一圈代理。
LLM_PROXY = os.getenv("CVAGENT_PROXY", "")

# OpenAI 兼容端点，任何供应商只要兼容 chat/completions + tools 就能接
PROVIDERS: dict[str, dict] = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
    },
    # 任意中转/自建网关：只需给出 base_url 与 key
    "custom": {
        "base_url": os.getenv("CVAGENT_BASE_URL", ""),
        "api_key_env": "CVAGENT_API_KEY",
        "default_model": os.getenv("CVAGENT_MODEL", "gpt-4o-mini"),
    },
}

RUNS_DIR = ROOT / "runs"
