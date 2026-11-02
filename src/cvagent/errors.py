"""结构化错误语义。

工具的失败不是异常本身，而是 Agent 的**下一步输入**。因此每类错误都必须携带：

    code        机器可判别的标识，规划层据此选择恢复策略
    retryable   同一调用换个参数再来是否可能成功
    hint        给模型看的自然语言纠正建议（不是给人看的堆栈）

这条约定的价值：模型不需要猜"为什么失败"，也不需要读 traceback 才能决定重试。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolError(Exception):
    """所有工具错误的基类。"""

    code: str
    message: str
    retryable: bool = False
    hint: str = ""
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"[{self.code}] {self.message}"

    def to_payload(self) -> dict:
        """转成返回给模型的 observation。刻意保持扁平、短小。"""
        payload = {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
        }
        if self.hint:
            payload["error"]["hint"] = self.hint
        if self.detail:
            payload["error"]["detail"] = self.detail
        return payload


class ImageNotFound(ToolError):
    def __init__(self, path: str) -> None:
        super().__init__(
            code="IMAGE_NOT_FOUND",
            message=f"文件不存在: {path}",
            retryable=False,
            hint="确认路径拼写；若不确定文件名，可先列出目录再调用。",
        )


class UnsupportedFormat(ToolError):
    def __init__(self, path: str, detected: str) -> None:
        super().__init__(
            code="UNSUPPORTED_FORMAT",
            message=f"不支持的图像格式: {detected}",
            retryable=False,
            hint="仅支持 jpg / jpeg / png / bmp / tif。请转成 jpg 后重试。",
            detail={"path": path},
        )


class InvalidArgument(ToolError):
    def __init__(self, name: str, value: object, reason: str, hint: str = "") -> None:
        super().__init__(
            code="INVALID_ARG",
            message=f"参数 {name}={value!r} 不合法：{reason}",
            retryable=True,
            hint=hint or "按参数说明改正后重试，无需更换图片。",
            detail={"param": name},
        )


class NoDetection(ToolError):
    """没检出目标。这不是故障，而是需要调整阈值的信号。"""

    def __init__(self, conf: float, retried: bool) -> None:
        hint = (
            "已尝试最低阈值仍为空，判断该图可能不含甘蓝，或分辨率过低。"
            if retried
            else "可能是阈值过高或目标过小。建议把 conf 降到 0.10 重试一次。"
        )
        super().__init__(
            code="NO_DETECTION",
            message=f"conf={conf} 下未检出任何目标",
            retryable=not retried,
            hint=hint,
            detail={"conf": conf, "min_conf_tried": 0.10 if retried else conf},
        )


class EngineError(ToolError):
    """推理后端故障（显存、权重缺失、CUDA 错误）。"""

    def __init__(self, message: str, *, oom: bool = False) -> None:
        super().__init__(
            code="ENGINE_OOM" if oom else "ENGINE_ERROR",
            message=message,
            retryable=False,
            hint=(
                "显存不足。建议减小图像尺寸或切换到 CPU 设备后重试。"
                if oom
                else "推理后端故障，重试通常无效，请检查权重文件与运行环境。"
            ),
        )


class UnknownTool(ToolError):
    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            code="UNKNOWN_TOOL",
            message=f"不存在的工具: {name}",
            retryable=True,
            hint=f"可用工具只有：{', '.join(available)}。",
        )
