"""Assemble cleaned research views from completed, explicit semantic reviews.

This script validates and applies recorded decisions. It does not infer labels,
rewrite queries, correct gold sets, or shrink candidate pools.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from validate_research_audit import ROOT, SOURCE, OUTPUT, read_jsonl, validate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha256(source) == sha256(target), f"Copy mismatch: {target}"


def main():
    summary, decisions = validate()
    if summary["errors"]:
        raise ValueError(json.dumps(summary, ensure_ascii=False))
    independent_path = OUTPUT / "audit/independent_test_second_review.jsonl"
    independent_notes = read_jsonl(independent_path)
    if len(independent_notes) != 12 or len({r["query_id"] for r in independent_notes}) != 12:
        raise ValueError("Expected 12 distinct targeted independent second-review notes.")
    if any(("test", r["query_id"]) not in decisions for r in independent_notes):
        raise ValueError("Unknown query in independent second-review notes.")
    if (OUTPUT / "MANIFEST.json").exists():
        raise ValueError("Cleaned dataset already built; refusing to overwrite it.")
    source_manifest = json.loads((SOURCE / "MANIFEST.json").read_text(encoding="utf-8"))
    for entry in source_manifest["files"]:
        if sha256(SOURCE / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source dataset modified: {entry['path']}")
    stats = {}
    kept_by_split = {}
    all_decisions = []
    dropped_rows = []
    original_annotations = {}
    for split in ("train", "test"):
        original_annotations[split] = {a["query_id"]: a for a in read_jsonl(SOURCE / f"annotations/queries/{split}.jsonl")}
        kept_by_split[split] = set()
        for query in read_jsonl(SOURCE / f"data/queries/{split}.jsonl"):
            review = decisions[(split, query["id"])]
            review = {**review, "original_scope": original_annotations[split][query["id"]]["scope"]}
            all_decisions.append(review)
            if review["decision"] == "keep":
                kept_by_split[split].add(query["id"])
            else:
                dropped_rows.append({"split": split, "query": query,
                                     "original_views": ["extended", "research_only"] if review["original_scope"] == "research_only" else ["extended"],
                                     "review": review})

    for view, prefix in (("extended", Path()), ("research_only", Path("research_only"))):
        stats[view] = {}
        for split in ("train", "test"):
            input_queries = read_jsonl(SOURCE / prefix / f"data/queries/{split}.jsonl")
            kept_queries = [q for q in input_queries if q["id"] in kept_by_split[split]]
            ids = {q["id"] for q in kept_queries}
            original_qrels = read_jsonl(SOURCE / prefix / f"data/qrels/{split}.jsonl")
            kept_qrels = [r for r in original_qrels if r["query_id"] in ids]
            write_jsonl(OUTPUT / prefix / f"data/queries/{split}.jsonl", kept_queries)
            write_jsonl(OUTPUT / prefix / f"data/qrels/{split}.jsonl", kept_qrels)
            copy_file(SOURCE / prefix / f"data/skills/{split}.jsonl", OUTPUT / prefix / f"data/skills/{split}.jsonl")
            annotations = read_jsonl(SOURCE / prefix / f"annotations/queries/{split}.jsonl")
            write_jsonl(OUTPUT / prefix / f"annotations/queries/{split}.jsonl", (a for a in annotations if a["query_id"] in ids))
            skills = read_jsonl(OUTPUT / prefix / f"data/skills/{split}.jsonl")
            pool = {s["id"] for s in skills}
            relations = {}
            for r in kept_qrels:
                assert r["skill_id"] in pool
                relations.setdefault(r["query_id"], set()).add(r["skill_id"])
            for query in kept_queries:
                assert set(query["skill_ids"]) == relations[query["id"]]
                assert query["k"] == len(set(query["skill_ids"]))
            assert len(ids) == len(kept_queries)
            assert kept_qrels == [r for r in original_qrels if r["query_id"] in ids]
            stats[view][split] = {
                "input_queries": len(input_queries), "kept_queries": len(kept_queries),
                "removed_queries": len(input_queries) - len(kept_queries),
                "qrels": len(kept_qrels), "candidate_skills": len(pool),
                "gold_skills_used": len({sid for q in kept_queries for sid in q["skill_ids"]}),
                "query_skill_counts": dict(sorted(Counter(q["k"] for q in kept_queries).items())),
                "source_scopes": dict(Counter(original_annotations[split][q["id"]]["scope"] for q in kept_queries)),
            }
        copy_file(SOURCE / prefix / "data/skills.jsonl", OUTPUT / prefix / "data/skills.jsonl")

    for name in ("annotations/skills.jsonl", "data/source_taxonomy.json", "SOURCE_DATASET_CARD.md", "selection_policy.json"):
        copy_file(SOURCE / name, OUTPUT / name)
    copy_file(ROOT / "scripts/research_review_protocol.json", OUTPUT / "review_protocol.json")
    write_jsonl(OUTPUT / "audit/decisions.jsonl", all_decisions)
    write_jsonl(OUTPUT / "audit/removed.jsonl", dropped_rows)
    write_jsonl(OUTPUT / "audit/retained.jsonl", (r for r in all_decisions if r["decision"] == "keep"))
    codes = Counter(code for r in all_decisions if r["decision"] == "drop" for code in r["reason_codes"])
    label_statuses = Counter(l["status"] for r in all_decisions for l in r["labels"])
    protocol = json.loads((OUTPUT / "review_protocol.json").read_text(encoding="utf-8"))
    write_json(OUTPUT / "audit/VALIDATION.json", summary)
    write_json(OUTPUT / "research_only/SUBSET_INFO.json", {
        "scope": "Cleaned strict research-only view; all original gold labels are research-core and individually reviewed as matched.",
        "stats": stats["research_only"], "candidate_pools": "Byte-identical to original research-only view.",
        "audit": "../audit/decisions.jsonl", "source_revision": source_manifest["source"]["revision"]})

    def table_escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    removal_md = ["# 剔除记录", "", "完整原始请求、每个标签状态和原文证据见同目录 `removed.jsonl`。原标签未修补；含任一可疑标签即整条剔除。", "",
                  "| Split | Query ID | 原始范围 | 原始技能 | 原因代码 | 具体原因 |", "|---|---|---|---|---|---|"]
    for row in dropped_rows:
        r = row["review"]
        removal_md.append("| " + " | ".join(table_escape(v) for v in [row["split"], row["query"]["id"], r["original_scope"], ", ".join(row["query"]["skill_names"]), ", ".join(r["reason_codes"]), r["reason_zh"]]) + " |")
    (OUTPUT / "audit/removed.md").write_text("\n".join(removal_md) + "\n", encoding="utf-8")

    report = ["# Research 子集清洗报告", "", "范围：学术研究与通用深度调研。两份源数据有包含关系，去重后逐条审查 1,328 条请求；每个原始 gold skill 均有语义判定和原文证据。", "",
              "方法：由模型辅助逐条语义复核，读取技能描述并针对专用或模糊技能核查正文；主审汇总、证据校验和定向二审。此过程不是独立人类专家标注，也没有执行请求中的外部工作流。", "",
              "| 数据视图 | Split | 输入请求 | 保留 | 剔除 | 候选技能（未缩减） |", "|---|---|---:|---:|---:|---:|"]
    for view in stats:
        for split, s in stats[view].items():
            report.append(f"| {view} | {split} | {s['input_queries']} | {s['kept_queries']} | {s['removed_queries']} | {s['candidate_skills']} |")
    report += ["", "扩展集包含纯研究子集，两个视图的数量不能相加。同一请求在两个视图使用同一复核决定。", "", "## 剔除原因", "", "原因可多选，次数之和可能大于剔除请求数。", "", "| 原因 | 次数 | 说明 |", "|---|---:|---|"]
    for code, count in codes.most_common():
        report.append(f"| {code} | {count} | {protocol['reason_codes'][code]} |")
    report += ["", "## 文件", "",
               "- `data/`：清洗后的扩展集，沿用原 train/test。",
               "- `research_only/data/`：清洗后的纯研究集，沿用原 train/test。",
               "- `audit/decisions.jsonl`：全部 1,328 条最终决定，每标签状态、原因及精确原文证据。",
               "- `audit/removed.jsonl`：所有剔除请求的完整原始记录与审计决定。",
               "- `audit/removed.md`：便于人工浏览的逐条剔除原因表。",
               "- `audit/retained.jsonl`：保留请求的审计记录。",
               "- `audit/parts/`、`audit/second_reviews.jsonl`（若存在）：分批首审和定向二审历史。",
               "- `audit/independent_test_second_review.jsonl`：另一审查代理对12条测试边界案例的独立建议；最终决定仍以decisions.jsonl为准。",
               "- `review_protocol.json`：判定口径与原因代码；`MANIFEST.json`：来源、计数与哈希。", "",
               "## 评测约束", "",
               "只删除请求及其全部 qrels，不改写请求、不修补 skill 标签。每个视图的候选池完整保留，已被判为错配的技能仍可作为其他请求的候选。",
               "保留原始 train/test 划分，不能把 train 重新命名为 test。正式报告优先使用 test；保留的训练请求可用于开发与校准。",
               "纯研究测试量较小，不能据此声称覆盖广泛多技能性能。清洗不证明未标注替代技能都不相关，也未重新审计语义近重复。",
               "继承上游合成请求、编码瑕疵和来源许可；清洗后的成绩不可与未清洗全集的成绩直接比较。", "",
               "## 复现", "", "在完整审计分片到位后运行 `python scripts/build_clean_research_dataset.py`。该脚本仅验证和应用显式审查决定，不自动推断语义判定。已有最终输出不会被覆盖。"]
    (OUTPUT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Verify no cross-split ID overlap in either retained view.
    for prefix in (Path(), Path("research_only")):
        train = {q["id"] for q in read_jsonl(OUTPUT / prefix / "data/queries/train.jsonl")}
        test = {q["id"] for q in read_jsonl(OUTPUT / prefix / "data/queries/test.jsonl")}
        assert not train & test
    files = [{"path": p.relative_to(OUTPUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"]
    manifest = {"name": "skillret-research-clean-v1.1", "review_protocol_version": protocol["version"],
                "source": {"path": str(SOURCE), "research_manifest_sha256": sha256(SOURCE / "MANIFEST.json"),
                           "upstream": source_manifest["source"]},
                "distinct_reviewed_queries": len(all_decisions),
                "distinct_kept_queries": sum(r["decision"] == "keep" for r in all_decisions),
                "distinct_removed_queries": len(dropped_rows),
                "targeted_independent_test_second_reviews": len(independent_notes),
                "label_statuses": dict(label_statuses), "reason_counts": dict(codes), "stats": stats,
                "validation": {"complete_review_coverage": True, "source_hashes_unchanged": True,
                               "verbatim_evidence_valid": True, "all_original_gold_sets_preserved": True,
                               "candidate_pools_byte_identical": True, "original_split_membership_preserved": True},
                "builder_sha256": sha256(Path(__file__)), "files": files}
    write_json(OUTPUT / "MANIFEST.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["distinct_reviewed_queries", "distinct_kept_queries", "distinct_removed_queries", "stats", "reason_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
