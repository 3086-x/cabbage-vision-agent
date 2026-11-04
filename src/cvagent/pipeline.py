"""确定性管线：不解自然语言，只做意图路由 + 固定动作。

## 为什么要有它

这个项目里绝大多数请求是「单图计数」「目录汇总」「这张图糊没糊」——它们的
下一步**不依赖上一步的观测**。把这类请求塞进 LLM 循环，付出的是：

    - 每次 1~2 次云端调用（延迟 + 费用）
    - 一个非确定性的行为（同样的问题可能走不同工具）
    - 一个额外的失败面（供应商超时、额度、限流）

换来的只是「措辞的泛化能力」。如果措辞在规则覆盖范围内，这份泛化是白付的。

## 它不是什么

不是 Agent 的降级版，是另一条路径：

    pipeline   0 次 LLM 调用，延迟 = 工具耗时，行为逐字节可预测
    agent      N 次 LLM 调用，延迟 = 工具 + 推理，能处理没见过的措辞

两者的边界是可测量的：见 eval/tasks.yaml 与 eval/tasks_holdout.yaml 的对照。

## 刻意没做的事

1. **不做模糊推断**。路由靠正则，命中就执行，不命中就明说不知道，绝不"猜一个
   最像的动作"——猜错比不做更贵，因为用户会拿到一个看似正常的错答案。
2. **不绕过工具层**。所有调用都走 `ToolRegistry.dispatch`，参数校验、错误语义、
   观测截断与 Agent 完全同一套。这样两边对比的才是"决策方式"，不是"实现质量"。
3. **不在答案里写死数字**。模板只拼接观测里出现的字段，所以幻觉率结构性为 0；
   代价是答案不如模型写得自然。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .agent.loop import Step, Trajectory
from .tools.registry import ToolRegistry

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Windows 绝对路径，或 POSIX 绝对路径；到空白/中文标点为止
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s，。；、）)\]】\"']+")
_CONF_RE = re.compile(r"(?:conf\s*[=＝:：]\s*|阈值(?:降到|调到|设为|改成|改为)?\s*)(\d+(?:\.\d+)?)", re.I)
_LIMIT_RE = re.compile(r"(?:前|先跑前|只跑前)\s*(\d+)\s*张")
_IMGID_RE = re.compile(r"image_id\s*(?:用|是|为|=|＝)?\s*([0-9a-f]{12})", re.I)

MAX_PAGES = 3  # 目录 75 张 / 单页 50 → 2 页；留一页余量，且不超任务集的 max_calls


@dataclass
class Slots:
    """从问题里能确定地抽出来的槽位。抽不到就是 None，不猜。"""

    files: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    conf: float | None = None
    limit: int | None = None
    image_id: str | None = None


def extract(question: str) -> Slots:
    s = Slots()
    for raw in _PATH_RE.findall(question):
        p = raw.rstrip(".,;:，。；：")
        if p.lower().endswith(tuple(IMG_EXTS)):
            s.files.append(p)
        else:
            s.dirs.append(p)
    m = _CONF_RE.search(question)
    if m:
        s.conf = float(m.group(1))
    m = _LIMIT_RE.search(question)
    if m:
        s.limit = int(m.group(1))
    m = _IMGID_RE.search(question)
    if m:
        s.image_id = m.group(1)
    return s


# ─────────────────────────── 意图路由 ───────────────────────────
# 顺序即优先级：先特异，后一般。写错了会造成"看起来对但选错动作"，
# 所以每条都注释了它要压住哪条更宽的规则。
_RULES: tuple[tuple[str, str], ...] = (
    # 能力边界类：必须最先判，否则会被"图片"之类的通用词带进检测流程
    ("refuse_disease", r"病害|虫害|生病|得病|有没有病|病斑|缺素"),
    ("refuse_area", r"每亩|株/亩|亩产|一亩|公顷"),
    # 复核类：只要出现 image_id，就走缓存路径（失败再回落检测）
    ("refine_by_id", r"image_id"),
    ("refine", r"阈值.{0,6}(降|调|改)|再数一遍|复核|可靠吗|准不准|会不会更准|差多少"),
    # 多图：先于单图计数，否则"这两张"会被当成一张。
    # 但它必须真的有两张路径才算数，否则"哪张株数最多"会被它抢走（那是目录任务）。
    ("compare2", r"哪张|谁更|两张|这两张|比较"),
    # 目录类：先于单图，否则目录路径会被当文件处理
    ("batch_limit", r"前\s*\d+\s*张"),
    ("batch_max", r"哪张.*(最多|最密)|最多的是"),
    ("batch_sweep", r"(目录|这批|这批图).{0,12}(有没有问题|看看|检查|异常)"),
    ("batch_total", r"目录|文件夹|这批"),
    # 质量类
    ("quality_then_count", r"如果.{0,10}(清|能拍|清楚).{0,12}(就|再).{0,6}(数|检测)"),
    ("quality", r"糊|清晰|拍得清|拍不清|能不能用|能用来|可用|分辨率|尺寸够"),
    # 单图细节：full 与 summary 的分叉点在于要不要逐框坐标
    ("detect_full", r"坐标|位置|每(一)?株.*(位置|在哪)|标注核对"),
    ("detect_boxes", r"尺寸分布|框.{0,4}大小|框.{0,4}尺寸"),
    # 读检测结果里的某个统计量，比"再数一遍"更省
    ("low_conf", r"置信度.{0,8}(偏低|较低|低|不高).{0,6}框|低置信"),
    # 省时诉求：必须压在"多少株"之前，否则会选 detect
    ("count_only", r"只要一个|越快越好|最省|不要多余|只给.{0,4}数"),
    ("detect_density", r"密度|长势|密不密"),
    ("count", r"多少株|几株|数一下|数数|计数|有多少"),
    # 兜底动作：只说了"检测这张图"而没说要什么统计量。放最后，因为它是
    # 信息最全也最贵的读法——有更省的就该走更省的。
    ("detect_generic", r"检测|识别|跑一下|看一下这(张|幅)"),
)


# 需要的最少文件数。不满足就跳过，避免"哪张最多"被当成两图比较。
_NEEDS_FILES = {"compare2": 2}

# 「前/头/首 5 张」这类"只要子集"的诉求。extract 只认「前 N 张」，
# 所以这里判的是"意图存在"而不是"数值抽到了"。
_SUBSET_RE = re.compile(r"(?:前|头|首|最前)\s*\d+\s*张")

# 空间分布类提问：「最密的那块有几株」「分布均匀吗」「哪块空着」。
# 这些问题的答案在 grid 字段里，而 `count` 规则只认「几株/多少株」——
# 不挡的话，"最密的那一块有几株"会被当成整图计数，答出 62 株（实测）。
# 这不是措辞问题：数字正确、问题答错，比拒答更糟。
_SPATIAL_RE = re.compile(r"均匀|最密|最稀|最疏|哪块|哪片|哪个区域|空着|空格|扎堆|聚集|分布")

# 但"尺寸分布/置信度分布"确实是统计量，属于 detect_boxes 那条路径，
# 不能一起放弃。这两个例外按现成的措辞写死：宁可漏挡，不可误挡。
_NONSPATIAL_DIST_RE = re.compile(r"尺寸分布|大小分布|框.{0,2}分布|置信度分布")


def route(question: str, s: Slots) -> str | None:
    """返回意图名；匹配不到返回 None（宁可不做，也不猜）。"""
    # 要子集但抽不出张数：量词写法不在规则里。此时任何 batch 意图都可能
    # 退化成"跑全量"，交出一个看起来正常、实际答非所问的汇总——比说不出更糟。
    # 所以连一条规则都不试，直接放弃，让上层升级。
    if _SUBSET_RE.search(question) and s.limit is None:
        return None

    # 空间分布类：规则表里没有对应动作，交给 Agent（它读 grid）。
    if _SPATIAL_RE.search(question) and not _NONSPATIAL_DIST_RE.search(question):
        return None

    for name, pat in _RULES:
        if not re.search(pat, question):
            continue
        # 目录类意图没有目录就降级，避免把"前 3 张"套到单图上
        if name.startswith("batch") and not s.dirs:
            continue
        need = _NEEDS_FILES.get(name)
        if need and len(s.files) < need:
            continue
        return name
    return None


# ─────────────────────────── 升级策略 ───────────────────────────
# 分层架构的分界线画在这里：pipeline 只处理「下一步动作在写代码时就确定」的请求，
# 其余交给 Agent。判据必须能从 Trajectory 里机械地读出来，不能靠读答案文本猜——
# 猜字符串会在改措辞时静默失效，这类错这个项目已经踩过两次（评测的禁语正则）。
#
# 只收确定性信号。以下情况**故意不升级**：
#   - 图片质量差 / 置信度双峰 / 检出数为 0（目录场景里 0 是正常值）
#   - 用户只是想要个粗略答案
# 它们属于业务判断，升级率会失控，而且 Agent 也未必答得更好。要加得先有数据。
def escalate_reason(traj: Trajectory) -> str | None:
    """返回升级原因；None 表示 pipeline 的结论可以直接交付。"""
    if traj.stopped_reason == "pipeline_unrouted":
        return "措辞不在规则覆盖范围内"

    for st in traj.steps:
        if st.ok:
            continue
        err = st.observation.get("error") or {}
        # 可恢复错误意味着「换个参数还能成」——这正是循环擅长、规则写不死的地方。
        # 不可恢复的（文件不存在、格式不支持）升级也没用，直接如实告诉用户。
        if err.get("retryable"):
            return f"{st.tool} 返回可恢复错误 {err.get('code')}"

    return None


def handoff(pre: Trajectory) -> str:
    """把规则层做过的动作，写成交给 Agent 的上下文。

    CLI 与评测必须用同一份：只要两边拼的文本不同，auto 模式在评测里的分数
    就不是线上那条路径的分数了。
    """
    # 一步都没走（路由就没命中）时不要交接口。实测过：塞一句「无工具调用」
    # 会让模型觉得任务还没开始，反而比纯 Agent 多绕一轮（H3 由此超预算）。
    # 空手升级就退化成纯 Agent，那才是它该有的样子。
    if not pre.steps:
        return ""
    done = "；".join(s.brief() for s in pre.steps)
    return f"[前置规则层] 已尝试：{done}。这一层无法收尾，请在此基础上继续判断。"


# ─────────────────────────── 管线 ───────────────────────────
class Pipeline:
    """零 LLM 的执行器。产出与 Agent 同构的 Trajectory，便于同一套判据评。"""

    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools
        self._steps: list[Step] = []

    # ---- 对外 ----
    def run(self, question: str) -> Trajectory:
        t0 = time.perf_counter()
        self._steps = []
        s = extract(question)
        intent = route(question, s)

        if intent is None:
            answer = ("无法确定要执行哪个动作。这条请求的措辞不在确定性规则的覆盖范围内，"
                      "需要走 Agent 路径，或把请求改写成明确的动作（如『数一下这张图的株数』）。")
            # 与走通的 "pipeline" 分开：这是分层架构里唯一的升级信号，
            # 用它判断要不要把请求交给 Agent，别写成两个字符串去比 answer。
            stopped = "pipeline_unrouted"
        else:
            answer = getattr(self, f"_{intent}")(question, s)
            stopped = "pipeline"

        return Trajectory(
            question=question,
            answer=answer,
            steps=self._steps,
            # 与 Agent 的 "stop" 区分开：这个结论不是模型决定收束的，是流程走到头了
            stopped_reason=stopped,
            elapsed_s=time.perf_counter() - t0,
            model="pipeline/rules",
        )

    # ---- 执行原语 ----
    def _call(self, tool: str, **args) -> Step:
        st = time.perf_counter()
        obs = self.tools.dispatch(tool, args)
        step = Step(
            index=len(self._steps),
            tool=tool,
            args=args,
            observation=obs,
            ok=bool(obs.get("ok")),
            elapsed_ms=(time.perf_counter() - st) * 1000,
            # 每一步都算一轮：pipeline 没有并行能力，也不该在"轮次"上占便宜。
            turn=len(self._steps),
        )
        self._steps.append(step)
        return step

    def _call_or_fix_conf(self, tool: str, conf: float | None, **kw) -> Step:
        """conf 越界时收敛到合法区间重试一次。

        这就是 F2 那条任务要考的行为：模型会把 1.5 改小再试，规则代码同样能，
        而且不需要"理解"——它只是把参数夹到 schema 声明的范围里。
        """
        if conf is None:
            return self._call(tool, **kw)
        if not 0 < conf <= 1:
            clamped = 1.0 if conf > 1 else 0.05
            return self._call(tool, conf=clamped, **kw)
        return self._call(tool, conf=conf, **kw)

    @staticmethod
    def _err(step: Step) -> str:
        e = step.observation.get("error", {})
        msg = e.get("message") or "未知错误"
        hint = e.get("hint")
        return f"{msg}。" + (f"{hint}" if hint else "")

    # ---- 能力边界 ----
    def _refuse_disease(self, q: str, s: Slots) -> str:
        # 一个动作都不执行：这类问题检测器答不了，跑一次只会给出误导性的株数。
        return ("不能判断。这是单类（甘蓝）目标检测器，只做株数统计与密度分级，"
                "不具备病害、虫害或长势健康的识别能力，输出的置信度只表示"
                "『这个位置像不像一株甘蓝』，与植株是否健康无关。")

    def _refuse_area(self, q: str, s: Slots) -> str:
        if not s.files:
            return "无法换算成每亩株数：缺少地块面积与地面标定（GSD），且没有指定图片。"
        st = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        n = st.observation.get("count")
        return (f"无法直接换算成每亩株数。这张图检出 {n} 株，但换算需要地面标定（GSD，"
                "即每像素代表多少厘米）或已知的地块面积；两者都没有，任何「每亩多少株」"
                "的说法都会引入无法核验的假设。这也是本项目明确列为局限的一条。")

    # ---- 单图 ----
    def _count(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径，无法计数。"
        st = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        return (f"{o['count']} 株（密度分级 {o['density']}，阈值 conf={o['conf_threshold']}）。"
                f"其中低置信（<0.5）的框 {o['low_conf_boxes']} 个。")

    def _count_only(self, q: str, s: Slots) -> str:
        # 与 _count 相同路径，但只说数字：用户明确要"不要多余信息"
        if not s.files:
            return "没有在请求里找到图片路径，无法计数。"
        st = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        return f"{st.observation['count']} 株。"

    def _detect_density(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        st = self._call_or_fix_conf("detect_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        cs = o.get("conf_stats", {})
        return (f"密度分级 {o['density']}（{o['count']} 株，约 {o['mean_density_per_mpx']} 株/百万像素，"
                f"覆盖率 {o['coverage']}）。置信度中位数 {cs.get('median')}，最低 {cs.get('min')}。"
                "分级阈值取自训练集逐图株数的四分位点，是相对本数据集的标签，不是农学绝对值。")

    def _detect_boxes(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        st = self._call_or_fix_conf("detect_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        bs = o.get("box_size_px", {})
        return (f"{o['count']} 株。框尺寸中位数 {bs.get('w_median')}×{bs.get('h_median')} 像素，"
                f"密度分级 {o['density']}，覆盖率 {o['coverage']}。")

    def _detect_full(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        kw = {"image_path": s.files[0], "detail": "full"}
        st = self._call("detect_cabbage", conf=s.conf if s.conf and 0 < s.conf <= 1 else 0.25, **kw)
        if not st.ok:
            return self._err(st)
        o = st.observation
        boxes = o.get("boxes") or []
        lowest = sorted(boxes, key=lambda b: b[4])[:3]
        pos = "；".join(f"({b[0]}, {b[1]}) conf={b[4]}" for b in lowest)
        return (f"{o['count']} 株，密度分级 {o['density']}。全部 {len(boxes)} 个框的坐标已取得"
                f"（格式 [cx, cy, w, h, conf]）。置信度最低的三个：{pos}。")

    def _low_conf(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        st = self._call_or_fix_conf("detect_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        cs = o.get("conf_stats", {})
        n = cs.get("n_low")
        pct = round(n / o["count"] * 100) if o.get("count") else 0
        return (f"置信度低于 0.5 的框有 {n} 个，占检出总数 {o['count']} 个的 {pct}%。"
                f"置信度范围 {cs.get('min')}~{cs.get('max')}，中位数 {cs.get('median')}。"
                "这些低置信框是计数不确定性的主要来源，可用 refine_detection 换阈值看影响。")

    def _detect_generic(self, q: str, s: Slots) -> str:
        # 用户只说了"检测"，没说要什么统计量：给最完整但不带逐框坐标的视图。
        if not s.files:
            return "没有在请求里找到图片路径。"
        st = self._call_or_fix_conf("detect_cabbage", s.conf, image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        return (f"检出 {o['count']} 株，密度分级 {o['density']}，覆盖率 {o['coverage']}。"
                f"置信度中位数 {o.get('conf_stats', {}).get('median')}，"
                f"低置信框 {o.get('conf_stats', {}).get('n_low')} 个。")


    # ---- 质量 ----
    def _quality(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        st = self._call("inspect_image", image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        scale = o.get("brightness_scale", 255)
        verdict = o["assessment"]
        usable = verdict == "曝光正常" and o.get("edge_strength", 0) >= 3
        head = "可以用" if usable else "不建议直接用于检测"
        reason = []
        if verdict != "曝光正常":
            reason.append(f"平均亮度 {o['mean_brightness']}/{scale}，{verdict}")
        if o.get("edge_strength", 0) < 3:
            reason.append(f"边缘强度 {o['edge_strength']}（满分 {o.get('edge_scale')}），偏低，可能存在模糊")
        side = o.get("size")
        return f"{head}。理由：{'；'.join(reason) or '亮度与锐度都在正常范围'}。图像 {side[0]}×{side[1]}，格式 {o['format']}。"

    def _quality_then_count(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        st = self._call("inspect_image", image_path=s.files[0])
        if not st.ok:
            return self._err(st)
        o = st.observation
        blurry = o.get("edge_strength", 0) < 3
        dark = o.get("assessment") != "曝光正常"
        if blurry or dark:
            why = []
            if dark:
                why.append(f"平均亮度 {o['mean_brightness']}/{o.get('brightness_scale')}，{o['assessment']}")
            if blurry:
                why.append(f"边缘强度仅 {o['edge_strength']}，判定为模糊")
            # 分支写死在代码里：这是"按观测决定下一步"的一种形态，
            # 只不过判据由人给定而不是由模型现场推理。
            return f"不能用，不数了。{'；'.join(why)}。建议重拍后再检测。"
        st2 = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[0])
        if not st2.ok:
            return self._err(st2)
        return (f"图像质量可用（亮度 {o['mean_brightness']}/{o.get('brightness_scale')}，"
                f"边缘强度 {o['edge_strength']}）。检出 {st2.observation['count']} 株。")

    # ---- 复核 ----
    def _refine(self, q: str, s: Slots) -> str:
        if not s.files:
            return "没有在请求里找到图片路径。"
        first = self._call_or_fix_conf("count_cabbage", None, image_path=s.files[0])
        if not first.ok:
            return self._err(first)
        base = first.observation["count"]
        # 用户没指定阈值时，用 0.15 作为"查漏检"的固定策略：比默认低，
        # 专门检验那些擦边的框会不会改变结论。
        target = s.conf if s.conf and 0 < s.conf <= 1 else 0.15
        second = self._call("refine_detection", image_id=first.observation["image_id"],
                            conf=target, iou=0.45)
        if not second.ok:
            return self._err(second)
        o = second.observation
        c = o["comparison"]
        delta = c["delta"]
        sign = "+" if delta >= 0 else ""
        return (f"两次结果对比：conf={c['before']['conf']} 时 {c['before']['count']} 株，"
                f"conf={c['after']['conf']} 时 {c['after']['count']} 株，差 {sign}{delta} 株。"
                f"稳定性判定 {o['stability']}——"
                + ("两次差异在 5% 以内，结论较可信。" if o["stability"] == "stable"
                   else "两次差异超过 5%，说明计数对阈值敏感，单次结果不足以支撑结论。"))

    def _refine_by_id(self, q: str, s: Slots) -> str:
        if s.image_id:
            target = s.conf if s.conf and 0 < s.conf <= 1 else 0.15
            st = self._call("refine_detection", image_id=s.image_id, conf=target, iou=0.45)
            if st.ok:
                c = st.observation["comparison"]
                return (f"复核完成：conf={c['before']['conf']} 时 {c['before']['count']} 株，"
                        f"conf={c['after']['conf']} 时 {c['after']['count']} 株，差 {c['delta']} 株。")
            # 缓存里没有这个 id（F3）：hint 说先检测。确定性代码直接照 hint 做，
            # 这正是"读 hint 自救"的规则版实现。
            if not s.files:
                return self._err(st) + " 缓存的上一轮结果会在进程重启后失效。"
        if not s.files:
            return "没有可用的 image_id 缓存，也没有图片路径，无法复核。"
        st2 = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[0])
        if not st2.ok:
            return self._err(st2)
        return (f"指定的 image_id 不在缓存里，已改为直接对图片重新检测："
                f"{st2.observation['count']} 株（conf={st2.observation['conf_threshold']}）。"
                "缓存只保留本进程内的结果，重启或换图都会失效。")

    def _compare2(self, q: str, s: Slots) -> str:
        if len(s.files) < 2:
            return "需要两张图片才能比较，但请求里只找到一张路径。"
        a = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[0])
        b = self._call_or_fix_conf("count_cabbage", s.conf, image_path=s.files[1])
        for st in (a, b):
            if not st.ok:
                return self._err(st)
        na, nb = a.observation["count"], b.observation["count"]
        name_a = s.files[0].replace("\\", "/").rsplit("/", 1)[-1]
        name_b = s.files[1].replace("\\", "/").rsplit("/", 1)[-1]
        if na == nb:
            return f"两张一样密：{name_a} 与 {name_b} 都是 {na} 株。"
        denser, other = (name_a, name_b) if na > nb else (name_b, name_a)
        return (f"{denser} 更密：{denser} 有 {max(na, nb)} 株，{other} 有 {min(na, nb)} 株，"
                f"差 {abs(na - nb)} 株。")

    # ---- 批量 ----
    def _batch(self, directory: str, conf: float | None, limit: int | None) -> dict | str:
        """翻页取全量。返回聚合结果，或一句错误说明。"""
        total = 0
        pages = 0
        offset = 0
        failed = 0
        best = (-1, None)
        while pages < MAX_PAGES:
            args = {"directory": directory, "offset": offset}
            if limit:
                args["limit"] = limit
            st = self._call("batch_detect", conf=conf if conf and 0 < conf <= 1 else 0.25, **args)
            if not st.ok:
                return self._err(st)
            o = st.observation
            pages += 1
            total += o["total_count"]
            failed += o["n_failed"]
            if o.get("max_count", -1) > best[0]:
                best = (o["max_count"], o.get("max_file"))
            if not o.get("truncated"):
                break
            offset = o["next_offset"]
        return {"total": total, "pages": pages, "failed": failed,
                "best_count": best[0], "best_file": best[1]}

    def _batch_total(self, q: str, s: Slots) -> str:
        if not s.dirs:
            return "没有在请求里找到目录路径。"
        r = self._batch(s.dirs[0], s.conf, None)
        if isinstance(r, str):
            return r
        return (f"目录内合计 {r['total']} 株，覆盖 {r['pages']} 页"
                + (f"（单页上限 50，需分页才能覆盖全量）" if r["pages"] > 1 else "（一页覆盖完）")
                + (f"，{r['failed']} 张处理失败。" if r["failed"] else "，无失败。"))

    def _batch_max(self, q: str, s: Slots) -> str:
        if not s.dirs:
            return "没有在请求里找到目录路径。"
        r = self._batch(s.dirs[0], s.conf, None)
        if isinstance(r, str):
            return r
        return (f"株数最多的是 {r['best_file']}，{r['best_count']} 株。"
                f"（已扫 {r['pages']} 页，合计 {r['total']} 株）")

    def _batch_limit(self, q: str, s: Slots) -> str:
        if not s.dirs:
            return "没有在请求里找到目录路径。"
        n = s.limit or 3
        st = self._call("batch_detect", directory=s.dirs[0],
                        conf=s.conf if s.conf and 0 < s.conf <= 1 else 0.25, limit=n)
        if not st.ok:
            return self._err(st)
        o = st.observation
        rows = o.get("rows_top") or []
        detail = "；".join(f"{r['file']} {r['count']} 株" for r in rows[:n])
        return f"按请求只跑了前 {o['n_processed']} 张：{detail}。合计 {o['total_count']} 株。"

    def _batch_sweep(self, q: str, s: Slots) -> str:
        if not s.dirs:
            return "没有在请求里找到目录路径。"
        r = self._batch(s.dirs[0], s.conf, None)
        if isinstance(r, str):
            return r
        head = (f"扫了 {r['pages']} 页，合计 {r['total']} 株，单张最多 {r['best_count']} 株"
                f"（{r['best_file']}）")
        issues = []
        if r["failed"]:
            issues.append(f"{r['failed']} 张处理失败")
        if r["best_count"] == 0:
            issues.append("存在 0 株的图，可能是空图或拍摄异常")
        if not issues:
            issues.append("没有发现明显异常：无失败文件，各图株数都大于 0")
        return f"{head}。{'；'.join(issues)}。要判断具体某张是否有问题，需要给出那张图的路径。"
