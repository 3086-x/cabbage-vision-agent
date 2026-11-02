"""推理后端封装。

三条纪律：
1. **懒加载**：没调用检测就不 import torch，让 `cvagent tools` 这类命令秒开。
2. **图像先校验再推理**：格式/尺寸在进入模型前判掉，避免把 10 秒的推理时间
   浪费在一张打不开的文件上，也让错误分类准确。
3. **按 image_id 缓存**：同一张图换阈值复核时复用已解码张量，这是 `refine`
   工具能"只改阈值不重读盘"的前提。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .types import Box, DetectionResult
from ..errors import EngineError, ImageNotFound, InvalidArgument, NoDetection, UnsupportedFormat

_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
}


def sniff_format(path: Path) -> str:
    """按魔术字节判断真实格式，不信任扩展名。"""
    with open(path, "rb") as f:
        head = f.read(12)
    for magic, name in _MAGIC.items():
        if head.startswith(magic):
            return name
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def image_id_of(path: Path) -> str:
    """内容指纹。同路径换了文件也要变，避免缓存串味。"""
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


class YoloEngine:
    def __init__(self, weights: Path | str | None = None, device: str | None = None, imgsz: int | None = None):
        from ..config import DEVICE, IMG_SIZE, WEIGHTS

        self.weights = Path(weights or WEIGHTS)
        self.device = device or DEVICE
        self.imgsz = imgsz or IMG_SIZE
        self._model = None
        self._cache: dict[str, DetectionResult] = {}
        self._cache_order: list[str] = []

    # ---- 生命周期 ----
    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.weights.exists():
            raise EngineError(
                f"权重文件不存在: {self.weights}。请先运行 cvagent setup-weights 或设置 CVAGENT_WEIGHTS。"
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise EngineError(f"未安装 ultralytics: {e}") from e
        try:
            model = YOLO(str(self.weights))
            model.to(self.device)
            # 预热一次，把首次 cuDNN autotune 的耗时从用户请求里移出去
            import numpy as np

            model.predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8), verbose=False, device=self.device)
            self._model = model
        except Exception as e:  # noqa: BLE001 - 统一转成结构化错误
            raise EngineError(f"加载权重失败: {e}", oom="out of memory" in str(e).lower()) from e

    # ---- 校验 ----
    def validate(self, path: str | Path) -> tuple[Path, int, int, str]:
        p = Path(path)
        if not p.exists():
            raise ImageNotFound(str(p))
        if not p.is_file():
            raise InvalidArgument("image_path", str(p), "这是一个目录，不是图片文件")
        if p.suffix.lower() not in _SUPPORTED_EXT:
            raise UnsupportedFormat(str(p), f"扩展名 {p.suffix}")
        fmt = sniff_format(p)
        if fmt == "unknown":
            raise UnsupportedFormat(str(p), "无法识别的内容（可能已损坏）")
        try:
            from PIL import Image

            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            raise UnsupportedFormat(str(p), f"解码失败: {e}") from e
        if w < 32 or h < 32:
            raise InvalidArgument("image_path", str(p), f"图像过小（{w}x{h}），无法检测")
        return p, w, h, fmt

    # ---- 推理 ----
    def detect(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        use_cache: bool = True,
    ) -> DetectionResult:
        from ..config import MIN_CONF

        if not 0 < conf <= 1:
            raise InvalidArgument("conf", conf, "必须落在 (0, 1] 区间")
        if not 0 < iou <= 1:
            raise InvalidArgument("iou", iou, "必须落在 (0, 1] 区间")
        if conf < MIN_CONF:
            raise InvalidArgument("conf", conf, f"低于 {MIN_CONF} 已无实际意义")

        p, w, h, _fmt = self.validate(image_path)
        iid = image_id_of(p)

        if use_cache:
            hit = self._cache.get(iid)
            # 缓存里存的是该图已跑过的**最低**阈值结果，所以阈值更高时，
            # 只需按本次阈值过滤掉低置信框即可，不必重新推理。
            if hit is not None and hit.conf_threshold <= conf and hit.iou_threshold == iou:
                if hit.conf_threshold == conf:
                    return hit
                kept = [b for b in hit.boxes if b.conf + 1e-9 >= conf]
                if not kept:
                    # 阈值调高后一个框都不剩，语义等同"该阈值下没检测到"
                    raise NoDetection(conf, hit.conf_threshold < conf)
                return DetectionResult(
                    image_id=hit.image_id,
                    path=hit.path,
                    width=hit.width,
                    height=hit.height,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    boxes=kept,
                    elapsed_ms=0.0,  # 未推理，命中缓存
                    device=hit.device,
                )

        self._ensure_model()
        t0 = time.perf_counter()
        try:
            res = self._model.predict(
                str(p), conf=conf, iou=iou, imgsz=self.imgsz, device=self.device, verbose=False,
            )[0]
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            raise EngineError(f"推理失败: {e}", oom="out of memory" in msg or "cuda" in msg) from e
        elapsed = (time.perf_counter() - t0) * 1000

        boxes: list[Box] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.tolist()
            confs = res.boxes.conf.tolist()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                if c + 1e-9 >= conf:  # 缓存里可能含更低阈值的框，按本次阈值过滤
                    boxes.append(Box(x1, y1, x2, y2, c))

        if not boxes:
            retried = iid in self._cache and self._cache[iid].conf_threshold < conf
            raise NoDetection(conf, retried)

        result = DetectionResult(
            image_id=iid, path=str(p), width=w, height=h,
            conf_threshold=conf, iou_threshold=iou,
            boxes=boxes, elapsed_ms=elapsed, device=self.device,
        )
        self._remember(iid, result)
        return result

    def _remember(self, iid: str, result: DetectionResult) -> None:
        # 保留最低阈值版本，供后续 refine 比较"多检出多少"
        old = self._cache.get(iid)
        if old is None or result.conf_threshold <= old.conf_threshold:
            self._cache[iid] = result
            if iid not in self._cache_order:
                self._cache_order.append(iid)
            if len(self._cache_order) > 8:
                evict = self._cache_order.pop(0)
                self._cache.pop(evict, None)

    def get_cached(self, image_id: str) -> DetectionResult | None:
        return self._cache.get(image_id)

    def clear_cache(self) -> None:
        """清空检测缓存。

        评测必须每条任务都清一次：上一条任务把缓存留在 conf=0.15，
        下一条任务问"默认阈值下多少株"就会拿到 0.15 的结果——
        这是评测之间的状态泄漏，不是模型的问题。
        """
        self._cache.clear()
        self._cache_order.clear()

    def warmup(self) -> float:
        """预加载权重与 CUDA 上下文，返回耗时毫秒。

        单独拆出来的原因：首次调用包含 cuDNN 图编译，实测 20~30s，若不剥离
        会被算进第一个工具的耗时里，让人误以为检测本身很慢（实际 ~100ms）。
        这里刻意写一张临时 jpg 走完整链路，因为只喂 numpy 覆盖不到解码与
        letterbox 的首次开销。
        """
        import tempfile

        import numpy as np
        from PIL import Image

        t0 = time.perf_counter()
        self._ensure_model()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "warmup.jpg"
            Image.fromarray(np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)).save(p, quality=90)
            try:
                self.detect(str(p), conf=0.5, use_cache=False)
            except NoDetection:
                pass  # 纯灰图没有目标，属预期
        return (time.perf_counter() - t0) * 1000
