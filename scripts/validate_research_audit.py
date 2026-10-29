"""Validate review coverage, original labels and verbatim evidence. No relevance inference."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/skillret-research-v1.1"
OUTPUT = ROOT / "data/skillret-research-clean-v1.1"

def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def load_reviews():
    reviews = []
    for path in sorted((OUTPUT / "audit/parts").glob("*.jsonl")):
        reviews.extend(read_jsonl(path))
    override_path = OUTPUT / "audit/second_reviews.jsonl"
    overrides = read_jsonl(override_path) if override_path.exists() else []
    return reviews, overrides

def validate(allow_incomplete=False):
    protocol = json.loads((ROOT / "scripts/research_review_protocol.json").read_text(encoding="utf-8"))
    allowed_codes = set(protocol["reason_codes"])
    queries = {}
    skills = {}
    for split in ("train", "test"):
        for query in read_jsonl(SOURCE / f"data/queries/{split}.jsonl"):
            queries[(split, query["id"])] = query
        for skill in read_jsonl(SOURCE / f"data/skills/{split}.jsonl"):
            skills[skill["id"]] = skill
    reviews, overrides = load_reviews()
    errors = []
    seen = set()
    for review in reviews:
        key = (review.get("split"), review.get("query_id"))
        if key in seen:
            errors.append({"query_id": key[1], "error": "Duplicate primary review"})
        seen.add(key)
    final = {(r.get("split"), r.get("query_id")): r for r in reviews}
    for override in overrides:
        key = (override.get("split"), override.get("query_id"))
        if key not in final:
            errors.append({"query_id": key[1], "error": "Override has no primary review"})
        final[key] = override
    # Validate raw parts too, even if a second review supersedes their verdict.
    for review in reviews + overrides:
        key = (review.get("split"), review.get("query_id"))
        query = queries.get(key)
        def error(message, **extra):
            errors.append({"query_id": key[1], "split": key[0], "error": message, **extra})
        if query is None:
            error("Unknown source query")
            continue
        for field in protocol["required_fields_per_query"]:
            if field not in review:
                error("Missing query field", field=field)
        if review.get("decision") not in ("keep", "drop"):
            error("Invalid decision")
        if not isinstance(review.get("reason_zh"), str) or not review["reason_zh"].strip():
            error("Missing query-specific reason")
        labels = review.get("labels", [])
        if [label.get("skill_id") for label in labels] != query["skill_ids"]:
            error("Original label IDs/order not preserved")
        if set(review.get("reason_codes", [])) - allowed_codes:
            error("Unknown reason codes", codes=review["reason_codes"])
        if review.get("decision") == "keep":
            if review.get("reason_codes") or any(label.get("status") != "matched" for label in labels):
                error("Keep requires every label matched and no rejection code")
        elif not review.get("reason_codes"):
            error("Drop requires rejection code")
        for label in labels:
            sid = label.get("skill_id")
            skill = skills.get(sid)
            if skill is None:
                error("Unknown skill", skill_id=sid)
                continue
            for field in protocol["required_fields_per_label"]:
                if field not in label:
                    error("Missing label field", skill_id=sid, field=field)
            if label.get("skill_name") != skill["name"]:
                error("Skill name mismatch", skill_id=sid)
            if label.get("status") not in protocol["label_statuses"]:
                error("Invalid label status", skill_id=sid)
            if not label.get("reason_zh", "").strip():
                error("Missing label-specific reason", skill_id=sid)
            evidence = label.get("query_evidence")
            if not isinstance(evidence, str) or evidence not in query["query"]:
                error("Query evidence is not verbatim", skill_id=sid, evidence=evidence)
            evidence = label.get("skill_evidence")
            texts = [skill.get("description", ""), skill.get("body", ""), skill.get("skill_md", "")]
            if not isinstance(evidence, str) or not evidence or not any(evidence in t for t in texts):
                error("Skill evidence is not verbatim", skill_id=sid, evidence=evidence)
    missing = sorted(set(queries) - set(final))
    if missing and not allow_incomplete:
        errors.append({"error": "Missing reviews", "count": len(missing), "examples": missing[:10]})
    summary = {"source_queries": len(queries), "reviewed_queries": len(final),
               "missing_queries": len(missing), "second_reviews": len(overrides),
               "decisions": dict(Counter(r["decision"] for r in final.values())),
               "error_count": len(errors), "errors": errors}
    return summary, final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    summary, _ = validate(args.allow_incomplete)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(bool(summary["errors"]))
