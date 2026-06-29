"""完整实验套件：知识库审计 + RAG 检索评测 + 改写对比，汇总 JSON 并生成报告片段。

用法:
  cd backend
  python scripts/run_experiment_suite.py              # 完整 60 题（约 40-60 分钟）
  python scripts/run_experiment_suite.py --quick        # 口语 20 题 + 关键题（约 15 分钟）
  python scripts/run_experiment_suite.py --report-only  # 仅从已有 JSON 生成报告片段
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

BACKEND = Path(__file__).resolve().parent.parent
DATA = BACKEND / "data" / "experiment"
PYTHON = sys.executable
sys.path.insert(0, str(BACKEND))

from config import INDEX_STATS_FILE, LAWS_YAML  # noqa: E402


def run_cmd(title: str, cmd: list[str]) -> int:
    print("\n" + "=" * 60, flush=True)
    print(title, flush=True)
    print("> " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=BACKEND).returncode


def audit_knowledge_base() -> dict:
    import yaml
    from fetcher import fetch_law_raw, parse_law, load_laws_config
    from parser import dedupe_articles

    laws_meta = []
    for law in load_laws_config():
        try:
            raw = fetch_law_raw(law)
            arts = dedupe_articles(parse_law(law, raw))
            laws_meta.append(
                {
                    "id": law["id"],
                    "name": law["name"],
                    "source_url": law.get("source_url", ""),
                    "raw_file": law["raw_file"],
                    "article_count": len(arts),
                    "last_article": arts[-1].article_no if arts else "",
                    "sample_first": arts[0].article_no if arts else "",
                }
            )
        except Exception as exc:
            laws_meta.append({"id": law["id"], "name": law["name"], "error": str(exc)})

    stats = {}
    if INDEX_STATS_FILE.exists():
        stats = json.loads(INDEX_STATS_FILE.read_text(encoding="utf-8"))

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "laws_yaml": str(LAWS_YAML),
        "index_stats": stats,
        "parsed": laws_meta,
        "total_articles": sum(x.get("article_count", 0) for x in laws_meta),
    }
    out = DATA / "kb_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"知识库审计已保存: {out}")
    return payload


def load_json(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_summary() -> dict:
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_base": load_json(DATA / "kb_audit.json"),
        "rewrite_rrf": load_json(DATA / "eval_rewrite_rrf.json"),
        "rewrite_modes": load_json(DATA / "rewrite_modes_compare.json"),
        "key_questions": load_json(DATA / "test_key_questions.json"),
        "oral_subset": load_json(DATA / "test_oral.json"),
    }

    rr = summary.get("rewrite_rrf") or {}
    if isinstance(rr, dict) and rr.get("summaries"):
        n = len([d for d in rr.get("details", []) if d.get("mode") == "retrieval_baseline"]) or 60
        table = {}
        for s in rr["summaries"]:
            mode = s.get("mode", "")
            key = {
                "retrieval_baseline": "baseline",
                "retrieval_rewrite": "rewrite",
                "retrieval_hybrid": "hybrid",
                "retrieval_dual_rrf": "hybrid",  # 兼容旧 JSON
            }.get(mode)
            if key:
                cnt = s.get("count", n)
                table[key] = {
                    "n": cnt,
                    "hits": int(round((s.get("recall_at_k") or 0) * cnt)),
                    "recall": s.get("recall_at_k"),
                    "latency_ms": s.get("avg_latency_ms"),
                }
        summary["recall_table"] = table

    rm = summary.get("rewrite_modes") or {}
    if isinstance(rm, dict):
        summary["two_stage_gain"] = {
            "only_two_stage_hit": rm.get("only_two_stage_hit", []),
            "only_single_hit": rm.get("only_single_hit", []),
            "single_stats": rm.get("single_stats"),
            "two_stage_stats": rm.get("two_stage_stats"),
        }

    out = DATA / "experiment_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"实验汇总已保存: {out}")
    return summary


def render_report_section(summary: dict) -> str:
    kb = summary.get("knowledge_base") or {}
    idx = kb.get("index_stats") or {}
    parsed = {x["id"]: x for x in kb.get("parsed", []) if "article_count" in x}

    lines = [
        "### 知识库构建与条数核验实验",
        "",
        f"**实验时间：** {summary.get('generated_at', '')[:10]}",
        "",
        "在完成刑法来源切换（最高法院公报不完整版 → 中国人大网 1997 修订版全文）及解析器改进（按行首「第X条」切分，避免正文引用误切）后，对四部法律重新构建向量索引并核验条数。",
        "",
        "| 法律 | 数据来源 | 索引条数 | 末条 |",
        "|------|----------|----------|------|",
    ]
    names = {
        "constitution": "中华人民共和国宪法",
        "civil_code": "中华人民共和国民法典",
        "criminal_law": "中华人民共和国刑法",
        "labor_law": "中华人民共和国劳动法",
    }
    sources = {
        "constitution": "中国人大网 2024 版",
        "civil_code": "国家法律法规数据库 PDF 转文本（1260 条）",
        "criminal_law": "中国人大网 1997 修订版",
        "labor_law": "中国人大网官方全文",
    }
    for lid in ("constitution", "civil_code", "criminal_law", "labor_law"):
        p = parsed.get(lid, {})
        lines.append(
            f"| {names.get(lid, lid)} | {sources.get(lid, p.get('source_url', ''))} "
            f"| **{idx.get(lid, p.get('article_count', '?'))}** | {p.get('last_article', '—')} |"
        )
    lines.extend(
        [
            "",
            f"**合计：** {kb.get('total_articles', sum(idx.values()))} 条法条入库。",
            "",
            "**说明：** 刑法由原先公报来源误切出的 273 条修正为完整 **452 条**；民法典按行首切分后稳定为 **1260 条**（修正 PDF 全文切分产生的 1262 条冗余）。",
            "",
        ]
    )

    rr = summary.get("recall_table") or {}
    if rr:
        def pct(d: dict) -> str:
            if not d:
                return "—"
            n, hit = d.get("n", 60), d.get("hits", 0)
            return f"**{hit / n:.1%}**（{hit}/{n}）" if n else "—"

        lines.extend(
            [
                "### 更新知识库后的检索 Recall 实验（60 题）",
                "",
                "评测集：`backend/data/eval_questions_verified.yaml`（68 题，含指导案例/典型案例案情；旧版 AI 集见 eval_questions.yaml）。"
                "指标：Top-5 检索结果中任一条 `article_no` 与标注期望法条匹配即记为命中。",
                "",
                "| 模式 | Recall@5 | 说明 |",
                "|------|----------|------|",
                f"| 不改写（baseline） | {pct(rr.get('baseline', {}))} | 原问题直接向量检索 |",
                f"| Query 改写单路 | {pct(rr.get('rewrite', {}))} | LLM 压缩为法律检索词 |",
                f"| 混合融合（hybrid） | {pct(rr.get('hybrid', rr.get('dual_rrf', {})))} | "
                "改写双路向量 + BM25；RRF池或 RRF池+Rerank 定 Top-5 |",
                "",
            ]
        )
        out_path = DATA / "eval_rewrite_rrf.json"
        lines.append(f"详细逐题结果：`backend/data/experiment/eval_rewrite_rrf.json`。")
        lines.append("")

    tg = summary.get("two_stage_gain") or {}
    if tg.get("single_stats") and tg.get("two_stage_stats"):
        ss, ts = tg["single_stats"], tg["two_stage_stats"]
        n = summary.get("rewrite_modes", {}).get("n", 60)
        lines.extend(
            [
                "### 方案七：一阶段 vs 两阶段 Query 改写（历史，脚本已移除）",
                "",
                "> `compare_rewrite_modes.py` 已删除；生产固定使用 two_stage。",
                "",
                "| 改写模式 | baseline | rewrite | hybrid |",
                "|----------|----------|---------|--------|",
                f"| single（一阶段） | {ss['baseline']/n:.1%} | {ss['rewrite']/n:.1%} | {ss['dual_rrf']/n:.1%} |",
                f"| two_stage（两阶段） | {ts['baseline']/n:.1%} | {ts['rewrite']/n:.1%} | {ts['dual_rrf']/n:.1%} |",
                "",
            ]
        )

    lines.extend(
        [
            "**复现命令：**",
            "",
            "```bash",
            "cd backend",
            "python scripts/build_index.py",
            "python scripts/compare_rag.py --compare-rewrite --retrieval-only",
            "```",
            "",
            "**Cascade 混合检索实验（2026-06-16）：** 见 `backend/data/experiment/cascade_experiment_20260616.md` 与报告章节「Cascade 混合检索实验」。",
            "",
        ]
    )
    return "\n".join(lines)


def patch_report(section: str) -> None:
    report = BACKEND.parent / "课程设计报告.md"
    if not report.exists():
        print(f"报告不存在: {report}")
        fragment = DATA / "report_section.md"
        fragment.write_text(section, encoding="utf-8")
        print(f"已写入片段: {fragment}")
        return

    text = report.read_text(encoding="utf-8")
    start = "### 知识库构建与条数核验实验"
    end = "### 检索策略讨论：级联改写 vs 双路 RRF 融合"
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end, 1)[1]
        new_text = before + section + end + after
    else:
        anchor = "### Query 改写与双路 RRF 检索实验（60 题）"
        if anchor in text:
            parts = text.split(anchor, 1)
            new_text = parts[0] + section + anchor + parts[1]
        else:
            new_text = text.rstrip() + "\n\n---\n\n" + section

    report.write_text(new_text, encoding="utf-8")
    print(f"已更新报告: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="完整实验套件")
    parser.add_argument("--quick", action="store_true", help="仅口语子集 + 关键题")
    parser.add_argument("--report-only", action="store_true", help="仅从已有 JSON 更新报告")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        summary = build_summary()
        patch_report(render_report_section(summary))
        return

    failed = 0
    audit_knowledge_base()

    failed += run_cmd(
        "[1] 冒烟测试",
        [PYTHON, "scripts/test_rag_pipeline.py", "--smoke"],
    )

    failed += run_cmd(
        "[2] 关键题 v05,v26,v41",
        [
            PYTHON,
            "scripts/test_rag_pipeline.py",
            "--ids",
            "v05,v26,v41",
            "--output",
            "data/experiment/test_key_questions.json",
        ],
    )

    if args.quick:
        failed += run_cmd(
            "[3] 口语题 q41-q60",
            [
                PYTHON,
                "scripts/test_rag_pipeline.py",
                "--oral",
                "--output",
                "data/experiment/test_oral.json",
            ],
        )
        failed += run_cmd(
            "[4] 混合诊断 v25,v68",
            [
                PYTHON,
                "scripts/diagnose_hybrid_gap.py",
                "--ids",
                "v25,v68",
            ],
        )
    else:
        failed += run_cmd(
            "[3] 60 题 baseline / 改写 / 双路 RRF",
            [
                PYTHON,
                "scripts/compare_rag.py",
                "--compare-rewrite",
                "--retrieval-only",
                "--output",
                "data/experiment/eval_rewrite_rrf.json",
            ],
        )
        failed += run_cmd(
            "[4] Cascade 混合诊断",
            [
                PYTHON,
                "scripts/diagnose_hybrid_gap.py",
                "--ids",
                "v25,v68",
            ],
        )

    summary = build_summary()
    patch_report(render_report_section(summary))

    if failed:
        print(f"\n完成，{failed} 个步骤返回非零退出码")
        sys.exit(1)
    print("\n全部实验完成，数据与报告已更新。")


if __name__ == "__main__":
    main()
