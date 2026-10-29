"""Record explicit per-row reviewer decisions supplied as JSON on stdin.

Input rows: [test_index, decision, reason_codes, reason_zh,
             [[label_status, label_reason_zh, exact_query_quote, optional_skill_quote], ...]]
This helper validates evidence and IDs; it does not decide semantic relevance.
"""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
source = root / "data/skillret-research-v1.1"
output = root / "data/skillret-research-clean-v1.1/audit/parts/test.jsonl"
def load(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
queries = load(source / "data/queries/test.jsonl")
skills = {s["id"]: s for s in load(source / "data/skills/test.jsonl")}
seen = {r["query_id"] for r in load(output)} if output.exists() else set()
rows = json.load(sys.stdin)
records = []
for index, decision, codes, reason, choices in rows:
    query = queries[index]
    assert query["id"] not in seen, "Duplicate review"
    assert decision in ("keep", "drop")
    assert len(choices) == len(query["skill_ids"])
    labels = []
    for sid, choice in zip(query["skill_ids"], choices):
        skill = skills[sid]
        status, label_reason, query_quote, *skill_quotes = choice
        skill_quote = skill_quotes[0] if skill_quotes else skill["description"][:240]
        assert status in ("matched", "mismatched", "uncertain")
        assert query_quote in query["query"], (query["id"], query_quote)
        assert skill_quote in skill["description"] or skill_quote in skill.get("body", "") or skill_quote in skill.get("skill_md", ""), (skill["name"], skill_quote)
        labels.append({"skill_id": sid, "skill_name": skill["name"], "status": status,
                       "reason_zh": label_reason, "query_evidence": query_quote,
                       "skill_evidence": skill_quote})
    assert decision != "keep" or (all(l["status"] == "matched" for l in labels) and not codes)
    assert decision != "drop" or codes
    records.append({"query_id": query["id"], "split": "test", "decision": decision,
                    "reason_codes": codes, "reason_zh": reason, "labels": labels,
                    "reviewer": "primary_agent"})
    seen.add(query["id"])
with output.open("a", encoding="utf-8", newline="\n") as stream:
    for record in records:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps({"recorded": len(records), "total": len(seen)}, ensure_ascii=False))
