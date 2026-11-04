"""工具契约。

一个工具由三部分组成，缺一不可：
    名称 + 描述   决定模型**何时**调用它（写的是意图，不是实现）
    参数 schema   决定模型**怎么**调用它（类型、范围、默认值都要写死）
    执行器        决定调用结果，失败必须抛 ToolError 而非裸异常

描述文本里刻意写入了使用条件（"当用户想知道…时用"）。实测这比只写
"检测甘蓝"更能减少误调用——模型靠描述里的条件判断取舍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

Handler = Callable[..., dict]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler
    tags: tuple[str, ...] = field(default=())
    # 单次观测的字符预算。None 用全局默认值。
    # 只有"模型显式要求明细"的工具才配更高预算，否则默认值足以防止上下文爆炸。
    max_chars: int | None = None

    def openai_schema(self) -> dict:
        """OpenAI / DashScope / DeepSeek 通用的 function calling 结构。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _schema(props: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_IMAGE_PATH = {
    "type": "string",
    "description": "图片文件的绝对或相对路径，支持 jpg/jpeg/png/bmp/tif/webp",
}
_CONF = {
    "type": "number",
    "description": "置信度阈值，0.05~1.0。越低检出越多但误检也越多。默认 0.25",
    "minimum": 0.05,
    "maximum": 1.0,
}
_IOU = {
    "type": "number",
    "description": "NMS 的 IoU 阈值，0.1~0.95。密集目标建议调低以减少框被抑制。默认 0.45",
    "minimum": 0.1,
    "maximum": 0.95,
}


def build_tools(engine) -> list[Tool]:
    """构造内置工具集。engine 注入而非全局，保证可测试性。"""

    def inspect_image(image_path: str) -> dict:
        from ..vision.engine import sniff_format

        p, w, h, fmt = engine.validate(image_path)
        from PIL import Image

        with Image.open(p) as im:
            rgb = im.convert("L").resize((256, 256))
            px = list(rgb.tobytes())
        mean = sum(px) / len(px)
        # 拉普拉斯近似：相邻像素差的均值，越低越模糊
        edge = 0.0
        for y in range(0, 255, 2):
            row = y * 256
            for x in range(0, 255, 2):
                v = px[row + x]
                edge += abs(v - px[row + x + 1]) + abs(v - px[row + x + 256])
        edge /= (128 * 128 * 2)
        return {
            "ok": True,
            "image_id": _iid(p),
            "format": fmt,
            "size": [w, h],
            "megapixels": round(w * h / 1e6, 2),
            "mean_brightness": round(mean, 1),
            "edge_strength": round(edge, 2),
            # 显式给出量纲与满值。不写的话模型会自己补"满值 255"这类常识，
            # 虽然是对的，但那个 255 在观测里找不到，会被溯源检查判成无出处。
            # 与其放宽检查，不如把量纲作为事实交给它。
            "brightness_scale": 255,
            "edge_scale": 255,
            "assessment": "偏暗" if mean < 60 else ("偏亮" if mean > 200 else "曝光正常"),
        }

    def detect_cabbage(
        image_path: str, conf: float = 0.25, iou: float = 0.45, detail: str = "summary", grid: int = 4
    ) -> dict:
        res = engine.detect(image_path, conf=conf, iou=iou)
        payload = res.full() if detail == "full" else res.summary()
        # summary 里固定带 4×4；要别的粒度才重算一次（纯 Python 遍历，不推理）
        if grid != 4:
            payload["grid"] = res.grid(grid)
        payload["ok"] = True
        return payload

    def count_cabbage(image_path: str, conf: float = 0.25, iou: float = 0.45) -> dict:
        res = engine.detect(image_path, conf=conf, iou=iou)
        return {
            "ok": True,
            "image_id": res.image_id,
            "count": res.count,
            "density": res.density(),
            "conf_threshold": res.conf_threshold,
            "low_conf_boxes": res.conf_stats()["n_low"],
        }

    def refine_detection(image_id: str, conf: float, iou: float = 0.45) -> dict:
        """对已检测过的图换阈值复核，复用上次结果做对比。"""
        from ..errors import InvalidArgument

        # before 取"最近一次请求的结果"，不是缓存里最低阈值那份：用户要比的
        # 是他上一条看到的数。取缓存会让连续第二次复核变成 0.1 对 0.1。
        cached = engine.get_last(image_id) or engine.get_cached(image_id)
        if cached is None:
            raise InvalidArgument(
                "image_id", image_id,
                "该图尚未检测过，无法复用缓存",
                hint="先对图片调用 detect_cabbage 或 count_cabbage 获取 image_id，再调用本工具复核。",
            )
        new = engine.detect(cached.path, conf=conf, iou=iou, use_cache=False)
        before, after = cached.count, new.count
        return {
            "ok": True,
            "image_id": new.image_id,
            "comparison": {
                "before": {"conf": cached.conf_threshold, "count": before},
                "after": {"conf": new.conf_threshold, "count": after},
                "delta": after - before,
            },
            "count": after,
            "density": new.density(),
            "conf_stats": new.conf_stats(),
            "stability": (
                "stable" if before and abs(after - before) / before <= 0.05
                else "unstable" if before else "no_baseline"
            ),
            "note": "stability=stable 表示两次阈值下计数差异 <=5%，结论较可信",
        }

    def batch_detect(directory: str, conf: float = 0.25, limit: int = 50, offset: int = 0) -> dict:
        from pathlib import Path

        from ..config import ROOT  # noqa: F401
        from ..errors import InvalidArgument

        d = Path(directory)
        if not d.exists():
            raise InvalidArgument("directory", directory, "目录不存在")
        if not d.is_dir():
            raise InvalidArgument("directory", directory, "不是目录")
        files = sorted(
            [f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
        )
        if not files:
            raise InvalidArgument("directory", directory, "目录内无受支持的图片")
        limit = max(1, min(int(limit), 50))
        offset = max(0, int(offset))
        picked = files[offset:offset + limit]
        if not picked:
            raise InvalidArgument(
                "offset", offset, f"超出范围，目录共 {len(files)} 张（offset 从 0 开始）")
        rows, failed = [], []
        for f in picked:
            try:
                res = engine.detect(f, conf=conf)
                rows.append({"file": f.name, "count": res.count, "density": res.density()})
            except Exception as e:  # noqa: BLE001 - 批量里单张失败不应中断
                failed.append({"file": f.name, "error": str(e)})
        counts = [r["count"] for r in rows]
        next_offset = offset + len(picked)
        truncated = next_offset < len(files)

        # 逐图明细只回前几名。50 张全名列会把观测顶到几千字符、被预算截断，
        # 模型看到的是残缺列表；不如直接给极值和前几名，其余用计数说明。
        ranked = sorted(rows, key=lambda r: r["count"], reverse=True)
        top = ranked[:6]
        return {
            "ok": True,
            "directory": str(d),
            "n_files_total": len(files),
            "n_processed": len(rows),
            "n_failed": len(failed),
            "offset": offset,
            # 分页信息必须显式返回：早先只回一句"仅处理前 N 张"却没有 offset 参数，
            # 模型在 75 张的目录上无法继续，只能反复重试同一次调用。
            "next_offset": next_offset if truncated else None,
            "truncated": truncated,
            "total_count": sum(counts),
            "mean_count": round(sum(counts) / len(counts), 1) if counts else 0,
            "min_count": min(counts) if counts else 0,
            "max_count": max(counts) if counts else 0,
            "max_file": ranked[0]["file"] if ranked else None,
            "min_file": ranked[-1]["file"] if ranked else None,
            "rows_top": top,
            "rows_omitted": max(0, len(rows) - len(top)),
            "failed": failed,
            "note": (
                f"本次处理第 {offset + 1}-{next_offset} 张（共 {len(files)} 张）。"
                f"还有 {len(files) - next_offset} 张未处理，用 offset={next_offset} 继续。"
                if truncated else f"已处理全部 {len(files)} 张"
            ),
        }

    _docs_index = None

    def search_docs(query: str, top_k: int = 3) -> dict:
        """在项目知识库里检索口径、依据与局限，返回带出处的片段。"""
        nonlocal _docs_index
        from ..config import DOCS_DIR
        from .docs import DocsIndex

        if _docs_index is None:
            _docs_index = DocsIndex(DOCS_DIR)
        return _docs_index.search(query, top_k=top_k)

    return [
        Tool(
            name="inspect_image",
            description=(
                "不检测、只读图片元信息：尺寸、格式、平均亮度、边缘强度。"
                "当需要判断图片是否可用（太暗/太糊/分辨率过低）时先调用它，比直接检测更省时。"
            ),
            parameters=_schema({"image_path": _IMAGE_PATH}, ["image_path"]),
            handler=inspect_image,
            tags=("meta", "cheap"),
        ),
        Tool(
            name="detect_cabbage",
            description=(
                "对单张航拍图做甘蓝检测，返回株数、密度分级、置信度分布、框尺寸分布、"
                "4×4 网格的逐格株数（回答『分布均不均匀、哪块空着』只能用这个字段），以及前几个框的坐标。"
                "当用户关心『这张图里有多少株、长势密度如何、分布是否均匀』时用它。"
                "detail=full 才返回全部框坐标，仅在需要逐个位置分析时使用。"
            ),
            parameters=_schema(
                {
                    "image_path": _IMAGE_PATH,
                    "conf": _CONF,
                    "iou": _IOU,
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": "summary 返回统计、网格与前 3 个框（默认，省 token）；full 返回全部框坐标",
                    },
                    "grid": {
                        "type": "integer",
                        "description": "网格粒度 n（n×n），默认 4，范围 2~8。只有需要更细/更粗的地块划分时才改",
                        "minimum": 2,
                        "maximum": 8,
                    },
                },
                ["image_path"],
            ),
            handler=detect_cabbage,
            max_chars=7000,  # detail=full 时 50 个框约 2000 字符，留足余量
        ),
        Tool(
            name="count_cabbage",
            description=(
                "只返回株数与密度，不返回框坐标，速度最快、最省 token。"
                "当只需一个数字、或批量核对时使用。"
            ),
            parameters=_schema({"image_path": _IMAGE_PATH, "conf": _CONF, "iou": _IOU}, ["image_path"]),
            handler=count_cabbage,
            tags=("cheap",),
        ),
        Tool(
            name="refine_detection",
            description=(
                "对**已经检测过**的图片换一个置信度阈值重新计数，并给出与上次结果的对比和稳定性判定。"
                "当上一次结果里 low_conf_boxes 偏多、或用户质疑计数是否可靠时调用，"
                "比重新传路径检测更快，因为复用了解码与缓存。"
            ),
            parameters=_schema(
                {
                    "image_id": {"type": "string", "description": "上次检测结果里返回的 image_id，12 位十六进制"},
                    "conf": _CONF,
                    "iou": _IOU,
                },
                ["image_id", "conf"],
            ),
            handler=refine_detection,
        ),
        Tool(
            name="batch_detect",
            description=(
                "对目录内的图片批量检测并汇总（总数、均值、极值、逐文件计数）。"
                "当用户给出一个文件夹、或要求对比多张图时使用。单次最多 50 张；"
                "返回里的 truncated=true 时，用 next_offset 继续处理剩下的。"
            ),
            parameters=_schema(
                {
                    "directory": {"type": "string", "description": "包含图片的目录路径"},
                    "conf": _CONF,
                    "limit": {"type": "integer", "description": "本次最多处理多少张，默认 50（一次跑完尽量别分页）", "minimum": 1, "maximum": 50},
                    "offset": {"type": "integer", "description": "从第几张开始（从 0 计）。上一页返回的 next_offset 直接填这里", "minimum": 0},
                },
                ["directory"],
            ),
            handler=batch_detect,
        ),
        Tool(
            name="search_docs",
            description=(
                "在项目知识库里检索计数口径、密度分级依据、评测指标定义、项目局限。"
                "当用户问『为什么是这个阈值』『密度怎么分级』『这个数字可不可信』"
                "『能不能换算株/亩』『评测怎么算通过』时先调用它，按返回片段作答。"
                "不要用它查图片内容——图片问题用 inspect_image / detect_cabbage。"
            ),
            parameters=_schema(
                {
                    "query": {"type": "string", "description": "检索关键词，中文短语比整句更好，如『密度分级』『复核稳定性』"},
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数，1~5，默认 3",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                ["query"],
            ),
            handler=search_docs,
            tags=("docs", "cheap"),
            max_chars=2000,
        ),
    ]


def _iid(p) -> str:
    from ..vision.engine import image_id_of

    return image_id_of(p)
