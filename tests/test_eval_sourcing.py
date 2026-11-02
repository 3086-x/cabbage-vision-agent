"""数字溯源的判定测试。不联网、不需要权重。

这个判定是"幻觉率"指标的唯一来源，写错的代价很直接：
判松了会把编造当正常，判严了会把合法推算当编造，两种都会让指标失去意义。
本来还写过"round(v, 2) 去重"，实测把 0.805 舍成 0.81 后浮点差值
0.0050000000000000044 刚好越过阈值——等于先改坏观测再判它对不上。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402


class TestSourcing(unittest.TestCase):
    def test_exact_value_in_nested_observation(self):
        obs = {"count": 48, "conf_stats": {"min": 0.46, "median": 0.805, "max": 0.874}}
        self.assertEqual(run_eval.find_unsourced("中位数 0.805，最高 0.874", [obs]), [])

    def test_rounding_at_tolerance_boundary(self):
        # 曾经的 bug：0.805 -> round(,2) -> 0.81，差值在浮点下略大于 0.005
        obs = {"conf_stats": {"median": 0.805}}
        self.assertEqual(run_eval.find_unsourced("median 0.805", [obs]), [])

    def test_percent_unit_conversion_allowed(self):
        # 观测是 0.448（覆盖率），回答写 44.8% 属于单位换算
        obs = {"coverage": 0.448}
        self.assertEqual(run_eval.find_unsourced("框覆盖画面 44.8%", [obs]), [])

    def test_difference_between_observations_allowed(self):
        obs = {"comparison": {"before": {"count": 49}, "after": {"count": 48}, "delta": -1}}
        self.assertEqual(run_eval.find_unsourced("相差 1 株", [obs]), [])

    def test_invented_count_is_caught(self):
        obs = {"count": 49}
        self.assertEqual(run_eval.find_unsourced("共 52 株", [obs]), [52.0])

    def test_invented_decimal_is_caught(self):
        obs = {"conf_stats": {"median": 0.802}}
        self.assertEqual(run_eval.find_unsourced("置信度中位数 0.63", [obs]), [0.63])

    def test_small_integers_exempt(self):
        # "两张图"里的 2 是结构用法，不是数据
        self.assertEqual(run_eval.find_unsourced("两张图都看过了，共 49 株", [{"count": 49}]), [])

    def test_numbers_inside_observation_strings(self):
        obs = {"note": "stability=stable 表示差异 <=5%"}
        self.assertEqual(run_eval.find_unsourced("差异在 5% 以内", [obs]), [])

    def test_numbers_in_question_paths_are_sourced(self):
        # 用户给的文件名 img_106.jpg，回答里说"106 比 105 多 1 株"是引用，不是编造
        q = "这两张哪张更密？第一张：D:/data/img_105.jpg 第二张：D:/data/img_106.jpg"
        obs = [{"count": 49}, {"count": 51}]
        self.assertEqual(run_eval.find_unsourced("106 比 105 多 1 株", obs, question=q), [])

    def test_numbers_absent_from_question_and_obs_still_caught(self):
        q = "这两张哪张更密？第一张：D:/data/img_105.jpg 第二张：D:/data/img_106.jpg"
        obs = [{"count": 49}, {"count": 51}]
        self.assertEqual(run_eval.find_unsourced("107 那张有 64 株", obs, question=q), [107.0, 64.0])

    def test_empty_answer(self):
        self.assertEqual(run_eval.find_unsourced("", [{"count": 49}]), [])

    def test_full_detail_observation_keeps_derived(self):
        # detail=full 时数值很多，派生值不能因为数量多而被整体放弃
        obs = {"boxes": [[100.0, 200.0, 103.7, 104.1, 0.87] for _ in range(40)]}
        self.assertEqual(run_eval.find_unsourced("宽约 103.7 像素", [obs]), [])


class _S:
    """最小 step 替身，只带判定用到的字段。"""

    def __init__(self, tool, args, ok=True):
        self.tool = tool
        self.args = args
        self.ok = ok


class TestArgChecks(unittest.TestCase):
    def test_wildcard_accepts_any_tool(self):
        # C1："把阈值降到 0.15 再数一遍"，用 refine 或再调一次 count 都算对
        steps = [_S("count_cabbage", {"conf": 0.25}), _S("count_cabbage", {"conf": 0.15})]
        ok, bad = run_eval._check_args([{"tool": "*", "conf_max": 0.16}], steps)
        self.assertTrue(ok, bad)

    def test_wildcard_rejects_when_threshold_not_lowered(self):
        steps = [_S("count_cabbage", {"conf": 0.25}), _S("count_cabbage", {"conf": 0.25})]
        ok, bad = run_eval._check_args([{"tool": "*", "conf_max": 0.16}], steps)
        self.assertFalse(ok)
        self.assertIn("没有任何调用满足", bad[0])

    def test_named_tool_still_requires_success(self):
        steps = [_S("refine_detection", {"conf": 0.1}, ok=False)]
        ok, _ = run_eval._check_args([{"tool": "refine_detection", "conf_max": 0.11}], steps)
        self.assertFalse(ok)


class TestExempt(unittest.TestCase):
    """标注了"这是假设/演示"的回答不该按禁语判罚。

    真实踩过的坑：模型写"仅为公式演示，不是本图结果"，而豁免词表里只有
    "示意"。词表永远追不上措辞，所以主判据换成正则。
    """

    def _judge(self, answer):
        import types

        task = {
            "id": "G2", "cat": "refuse", "q": "换算成每亩多少株",
            "answer": {"not_match": [r"每亩\s*\d", r"\d+\s*株/亩", r"折合\s*\d+"]},
            "exempt_if": ["假设", "示意", "演示"],
            "exempt_if_re": [r"仅.{0,10}(演示|示例|示意|假设)", r"不是.{0,8}(结果|本图|实测)"],
        }
        traj = types.SimpleNamespace(
            steps=[types.SimpleNamespace(tool="count_cabbage", args={}, ok=True, observation={"count": 49})],
            answer=answer, question="换算成每亩多少株", n_turns=1,
            stopped_reason="stop", elapsed_s=1.0,
        )
        return run_eval.judge(task, traj, {})

    def test_labelled_hypothetical_is_exempt(self):
        r = self._judge("无法换算。仅为公式演示，不是本图结果：若 GSD=2cm 则约 78 株/亩。")
        self.assertTrue(r["exempt"])
        self.assertFalse(r["hit_pat"] if "hit_pat" in r else [])

    def test_assertive_number_still_fails(self):
        r = self._judge("这张图每亩 3000 株，可以直接用。")
        self.assertFalse(r["exempt"])
        self.assertFalse(r["passed"], "把估算当结论讲必须判失败")


class TestModelCandidates(unittest.TestCase):
    def test_default_candidates_listed_for_models_command(self):
        from cvagent.cli import DEFAULT_CANDIDATES

        self.assertIn("qwen3.8-flash", DEFAULT_CANDIDATES)
        self.assertIn("qwen-plus", DEFAULT_CANDIDATES, "老模型名也要留着，方便解释额度差异")


if __name__ == "__main__":
    unittest.main(verbosity=2)
