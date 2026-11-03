"""确定性管线的契约测试。不加载权重、不联网、不调模型。

这一层锁住的是**管线自己会不会答错**，与 Agent 无关：
    1. 槽位抽取（路径 / conf / limit / image_id）
    2. 意图路由，特别是规则顺序——写反了会造成"看起来对但选错动作"
    3. 参数越界时的确定性收敛
    4. 认不出来时**必须说不知道**，而不是猜一个动作
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvagent.agent.loop import Step, Trajectory  # noqa: E402
from cvagent.pipeline import (  # noqa: E402
    Pipeline,
    escalate_reason,
    extract,
    handoff,
    route,
)
from cvagent.tools.registry import ToolRegistry  # noqa: E402
from cvagent.tools.spec import build_tools  # noqa: E402
from cvagent.vision.types import Box, DetectionResult  # noqa: E402


class FakeEngine:
    """只实现工具层真正用到的方法。阈值越高检出越少，模拟真实行为。"""

    def __init__(self, count: int = 48, n_low: int = 2):
        self.base = count
        self.n_low = n_low
        self.calls: list[tuple] = []
        self._cache: dict[str, DetectionResult] = {}

    def validate(self, path):
        p = Path(path)
        if not p.exists():
            from cvagent.errors import ImageNotFound

            raise ImageNotFound(str(p))
        from PIL import Image

        with Image.open(p) as im:
            w, h = im.size
        if w < 32 or h < 32:
            from cvagent.errors import InvalidArgument

            raise InvalidArgument("image_path", str(p), f"图像过小（{w}x{h}），无法检测")
        return p, w, h, "png"

    def detect(self, image_path, conf=0.25, iou=0.45, use_cache=True):
        # 必须先 validate：真实的 YoloEngine.detect 第一件事就是校验路径与尺寸，
        # 替身省掉这一步的话，"文件不存在"这类任务的测试会假通过。
        self.validate(image_path)
        self.calls.append(("detect", str(image_path), conf))
        # 阈值越高留下越少：0.25 全留，每 +0.1 掉 1 个
        keep = max(1, self.base - int((conf - 0.25) * 10)) if conf > 0.25 else self.base
        boxes = [Box(10 * i, 10, 10 * i + 40, 50, 0.3 if i < self.n_low else 0.9) for i in range(keep)]
        res = DetectionResult(image_id="abc123def456", path=str(image_path), width=1024,
                              height=1024, conf_threshold=conf, iou_threshold=iou, boxes=boxes)
        self._cache[res.image_id] = res
        return res

    def get_cached(self, image_id):
        return self._cache.get(image_id)

    def get_last(self, image_id):
        return self._cache.get(image_id)

    def clear_cache(self):
        self._cache.clear()

    def warmup(self):
        return 0.0


def _png(dirpath: Path, name: str, size=(64, 64), gray=120) -> Path:
    from PIL import Image

    p = dirpath / name
    Image.new("RGB", size, (gray, gray, gray)).save(p)
    return p


class TestSlots(unittest.TestCase):
    def test_windows_paths_split_file_and_dir(self):
        s = extract(r"看这两张 D:\data\img_1.jpg 和 D:\data\img_2.png")
        self.assertEqual(len(s.files), 2)
        self.assertEqual(s.dirs, [])

    def test_dir_has_no_image_suffix(self):
        s = extract(r"统计这个目录 D:\data\test\images")
        self.assertEqual(s.dirs, [r"D:\data\test\images"])
        self.assertEqual(s.files, [])

    def test_conf_from_two_phrasings(self):
        self.assertEqual(extract("把阈值降到 0.15 再数").conf, 0.15)
        self.assertEqual(extract("用 conf=1.5 检测").conf, 1.5)

    def test_limit_and_image_id(self):
        s = extract("先跑前 3 张，image_id 用 abc123def456")
        self.assertEqual(s.limit, 3)
        self.assertEqual(s.image_id, "abc123def456")

    def test_trailing_punctuation_not_part_of_path(self):
        s = extract(r"图片：D:\a\b.jpg，谢谢")
        self.assertEqual(s.files, [r"D:\a\b.jpg"])


class TestRouting(unittest.TestCase):
    """路由的回归测试。每条都对应一个真实踩过的坑。"""

    def _r(self, q: str, dirs=None, files=None):
        s = extract(q)
        if dirs:
            s.dirs = dirs
        if files:
            s.files = files
        return route(q, s)

    def test_count_vs_density_vs_full(self):
        self.assertEqual(self._r("这张图有多少株？", files=["a.jpg"]), "count")
        self.assertEqual(self._r("密度是哪个级别？", files=["a.jpg"]), "detect_density")
        self.assertEqual(self._r("把每株坐标给我", files=["a.jpg"]), "detect_full")

    def test_cheap_beats_complete(self):
        # "只要一个株数"必须压在"多少株"之前，否则会选代价更高的 detect
        self.assertEqual(self._r("只要一个株数，越快越好", files=["a.jpg"]), "count_only")

    def test_batch_max_not_stolen_by_compare2(self):
        # 回归：compare2 的正则含"哪张"，曾经把「哪张图株数最多」抢走，
        # 然后在只有一张路径的情况下返回"需要两张图"。必须靠文件数门槛挡住。
        q = "这批里哪张图株数最多？目录：D:/x"
        self.assertEqual(self._r(q, dirs=["D:/x"]), "batch_max")

    def test_compare2_needs_two_files(self):
        self.assertNotEqual(self._r("这两张哪张更密？", files=["a.jpg"]), "compare2")
        self.assertEqual(self._r("这两张哪张更密？", files=["a.jpg", "b.jpg"]), "compare2")

    def test_batch_limit_before_batch_total(self):
        q = "只先跑前 3 张，目录：D:/x"
        self.assertEqual(self._r(q, dirs=["D:/x"]), "batch_limit")

    def test_refuse_before_everything(self):
        self.assertEqual(self._r("这图里的甘蓝有没有得病？", files=["a.jpg"]), "refuse_disease")
        self.assertEqual(self._r("换算成每亩多少株？", files=["a.jpg"]), "refuse_area")

    def test_generic_detect_is_last_resort(self):
        # 只说了"检测"：应当兜底到 detect，而不是被更宽的 count 抢走（count 规则
        # 里有"有多少"，这条没有，所以它只能落到 generic）
        self.assertEqual(self._r("检测这张图", files=["a.jpg"]), "detect_generic")

    def test_unknown_returns_none(self):
        self.assertIsNone(self._r("今天天气怎么样"))

    def test_subset_with_unknown_quantifier_gives_up(self):
        # 「头 5 张」里的量词规则不认识，抽不出 limit。这时必须放弃，
        # 不能让 batch_total 顶上去跑全量——那会交出一个看起来正常、
        # 实际答非所问的汇总（留出集 H6 实测跑成了全目录 4134 株）。
        self.assertIsNone(self._r("先跑头 5 张试试水，目录：D:/x", dirs=["D:/x"]))

    def test_subset_with_known_quantifier_still_routes(self):
        # 同一诉求用规则认识的写法，就该正常走 batch_limit，别一起放弃。
        self.assertEqual(self._r("先跑前 5 张，目录：D:/x", dirs=["D:/x"]), "batch_limit")


class TestEscalation(unittest.TestCase):
    """分层路由的判据。这些错了不会报错，只会静默少升级或多升级。"""

    def _traj(self, *, stopped="pipeline", steps=None):
        return Trajectory(question="q", answer="a", stopped_reason=stopped,
                          steps=steps or [])

    def _step(self, ok=True, tool="count_cabbage", obs=None):
        return Step(index=0, tool=tool, args={}, observation=obs or {}, ok=ok)

    def test_unrouted_escalates(self):
        self.assertIsNotNone(escalate_reason(self._traj(stopped="pipeline_unrouted")))

    def test_clean_run_does_not_escalate(self):
        t = self._traj(steps=[self._step(obs={"ok": True, "count": 48})])
        self.assertIsNone(escalate_reason(t))

    def test_retryable_error_escalates(self):
        # 换个参数还能成，正是循环擅长、规则写不死的地方
        obs = {"ok": False, "error": {"code": "Boom", "retryable": True}}
        t = self._traj(steps=[self._step(ok=False, obs=obs)])
        self.assertIn("可恢复", escalate_reason(t))

    def test_non_retryable_error_stays(self):
        # 文件不存在升级也没用，如实告诉用户比再跑一遍强
        obs = {"ok": False, "error": {"code": "ImageNotFound", "retryable": False}}
        t = self._traj(steps=[self._step(ok=False, obs=obs)])
        self.assertIsNone(escalate_reason(t))


class TestHandoff(unittest.TestCase):
    def test_empty_steps_produce_no_context(self):
        # 路由就没命中时 pipeline 一步没走。塞一句「无工具调用」会让模型
        # 以为任务还没开始，比纯 Agent 多绕一轮——空手升级就该退化成纯 Agent。
        self.assertEqual(handoff(Trajectory(question="q", answer="a", steps=[])), "")

    def test_steps_are_summarized_with_numbers(self):
        st = Step(index=0, tool="count_cabbage", args={},
                  observation={"ok": True, "count": 49}, ok=True)
        text = handoff(Trajectory(question="q", answer="a", steps=[st]))
        self.assertIn("count_cabbage", text)
        self.assertIn("49", text)


class TestPipelineRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.img = _png(self.dir, "a.png")
        self.img2 = _png(self.dir, "b.png")
        self.engine = FakeEngine(count=48, n_low=2)
        self.tools = ToolRegistry(build_tools(self.engine))
        self.pipe = Pipeline(self.tools)

    def tearDown(self):
        self.tmp.cleanup()

    def test_count_answer_carries_observation_numbers(self):
        traj = self.pipe.run(f"这张图有多少株？{self.img}")
        self.assertIn("48", traj.answer)
        self.assertEqual(traj.stopped_reason, "pipeline")
        self.assertEqual(len(traj.steps), 1)

    def test_conf_out_of_range_is_clamped_not_crashed(self):
        # F2 那条任务要的行为：1.5 越界，确定性代码把它夹到 1.0 再跑一次。
        traj = self.pipe.run(f"用 conf=1.5 检测这张图：{self.img}")
        self.assertTrue(traj.steps[0].ok)
        self.assertEqual(traj.steps[0].args["conf"], 1.0)

    def test_unknown_intent_calls_no_tool(self):
        traj = self.pipe.run("随便聊两句吧")
        self.assertEqual(traj.steps, [])
        self.assertIn("无法确定", traj.answer)

    def test_disease_is_refused_without_detection(self):
        # 拒答类必须先于检测：跑一次检测会给出一个误导性的株数
        traj = self.pipe.run(f"这些甘蓝有没有得病？{self.img}")
        self.assertEqual(traj.steps, [])
        self.assertIn("不具备", traj.answer)

    def test_missing_file_reports_reason_not_number(self):
        traj = self.pipe.run(f"检测这张图：{self.dir / 'nope.jpg'}")
        self.assertIn("不存在", traj.answer)
        self.assertNotIn("株", traj.answer.replace("株数", "").replace("每株", ""))

    def test_refine_produces_comparison(self):
        traj = self.pipe.run(f"把阈值降到 0.15 再数一遍，告诉我差多少：{self.img}")
        self.assertEqual([s.tool for s in traj.steps], ["count_cabbage", "refine_detection"])
        self.assertIn("差", traj.answer)

    def test_batch_pages_through_directory(self):
        # limit=1 模拟单页装不下，逼管线翻页
        for i in range(3):
            _png(self.dir, f"x{i}.png")
        traj = self.pipe.run(f"这个目录一共多少株？{self.dir}")
        self.assertGreaterEqual(len(traj.steps), 1)
        self.assertTrue(all(s.tool == "batch_detect" for s in traj.steps))
        self.assertIn("株", traj.answer)

    def test_compare_two_files_uses_both(self):
        traj = self.pipe.run(f"这两张哪张更密？{self.img} 和 {self.img2}")
        self.assertEqual(len(traj.steps), 2)
        self.assertTrue(all(s.tool == "count_cabbage" for s in traj.steps))


if __name__ == "__main__":
    unittest.main()
