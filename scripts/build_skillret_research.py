"""Build an auditable research subset without changing SKILLRET labels or splits.

Uses only the Python standard library. Run from the repository root:
    python scripts/build_skillret_research.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil


SOURCE_REVISION = "6583d7d2ed07644d0fb8938ed8178f3a7dc42a12"
POLICY_VERSION = "1.0"

# These are curated skill names, not substring matches against arbitrary bodies.
# Identically named skills are disambiguated below by their original UUID.
CATEGORY_NAMES = {
    "literature_discovery_and_review": """
Academic Researcher
Answering Research Questions
Getting Started with Research Superpowers
Searching Scientific Literature
Traversing Citation Networks
Subagent-Driven Literature Review
academic-search
arxiv-analysis
arxiv-research
arxiv-search
literature-gap
literature-gap-finder
literature-research
literature-review
neurosurgery-literature
paper-finder
paper-search-usage
paper-to-intuition
pubmed-search
pubmed-field-validator
scholarag
systematic-review
inclusion-criteria
synthesis-matrix
data-extraction
extract-from-pdfs
zotero-mcp
hatchery-papers
gpu-graphics-research
""",
    "research_methods_and_analysis": """
Statistical Hypothesis Testing
bayesian-meta-analysis
designing-experiments
diagnostic-meta-analysis
effect-size
forest-plot-creation
gioia-analysis
hypothesis-generation
hypothesis-test
hypothesis-testing
hypothesis-testing-engine
hypogenic
ipd-meta-analysis
meta-analysis
meta-analysis-fundamentals
nma-methodology
pairwise-ma-methodology
pre-registration
irb-protocol
research-assessor
research-question-refiner
research-questions
research-taste-developer
scholar-evaluation
scientific-reviewer
statistical-analysis
subgroup-analysis
experiment-design-checklist
implement-paper-from-scratch
paper-to-code
transformer-lens-interpretability
""",
    "academic_writing_and_publication": """
Peer Reviewer
academic
academic-research
academic-writing-style
assembler
bib-managing
bibliography
citation-format
citations-retrieval
enrich-citations
grant
grant-application-setup
grant-compliance-checking
ieee-pes-paper-reviewer
manuscript-writing
ml-paper-writing
peer-review
publication-prep
research-grants
reviewer-2-simulator
scientific-writing
scientific-slides
scientific-schematics
latex-posters
pptx-posters
scoping
venue-templates
""",
    "deep_research_and_evidence_synthesis": """
Matrix Deep Research
Research Synthesizer
analysis-report
analyzing-research-documents
analyzing-source
cc2-research-framework
comprehensive-research-agent
data-synthesis
deep-research
deep-researcher
deep-research-synthesizer
deepresearch-integrator
discover-research
exa-research
fact-checker
gpt-researcher
internet-deep-orchestrator
internet-light-orchestrator
internet-research-orchestrator
langchain-deep-research
multi-agent-researcher
multi-model-research
multi-source-investigation
obsidian-research-synthesis
openai-deep-research
perplexity-research
perplexity-researcher-reasoning-pro
Querying Web Search Agent
reference
research
research-analyst
research-and-incorporate
research-and-report
research-assistant
research-documentation
research-executor
research-external
research-framework
research-lookup
research-methodology
research-methods
research-notebook
research-reviewer
research-synthesis
research-topic
research-validation
research-web
researching-web
researching-web-sources
source-verification
stanford-storm
technical-research
ultimate-research
updating-knowledge
verify-content
web-research
web-research-documenter
web-search-researcher
webresearcher
websearch-deep
websearch-standard
""",
    "research_workflow_and_tools": """
Cleaning Up Research Sessions
eln
notebooklm
research-storage
scientific-workflows
sync-paper
""",
}

EXCLUDE_IDS = {
    "63540641-7eec-4180-abae-8677a1f87024": "Hypothesis Python property-based software testing, not research hypothesis testing.",
    "92f7565b-c9e6-4ad3-a20c-a1bc7cde0e9a": "Codebase documentation/history workflow, not external research.",
    "95050f08-72f4-4f5e-84d5-965c963e6d3a": "Feature implementation planning rather than research as the primary deliverable.",
    "e9e10646-a090-4dd1-96c3-e38062c8c2ef": "Routine framework/API documentation lookup.",
}

DISCOVERY_PATTERN = re.compile(
    r"research|scholar|academic|scientific|literature|arxiv|pubmed|citation|"
    r"bibliograph|systematic.review|meta.analysis|hypothesis|statistical|"
    r"experiment.design|peer.review", re.I
)


def normalize(name):
    return re.sub(r"[\s_]+", "-", name.strip().casefold())


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build(source, output, refresh_generated=False):
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("Output must be separate from the original dataset directory.")
    if output.exists() and any(output.iterdir()):
        existing_manifest = output / "MANIFEST.json"
        if not refresh_generated or not existing_manifest.is_file():
            raise ValueError(f"Output already contains files; choose a new directory: {output}")
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (existing.get("name") != "skillret-research-v1.1"
                or existing.get("source", {}).get("revision") != SOURCE_REVISION):
            raise ValueError("Refusing to refresh an unrelated output directory.")
        for entry in existing["files"]:
            path = output / entry["path"]
            if path.resolve().is_relative_to(output.resolve()) is False:
                raise ValueError("Invalid path in the previous generated manifest.")
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Refusing to overwrite a modified generated file: {path}")
    provenance = json.loads((source / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    if provenance["revision"] != SOURCE_REVISION:
        raise ValueError("This selection policy requires the pinned SKILLRET v1.1 revision.")
    for entry in provenance["files"]:
        if sha256(source / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {entry['path']}")

    categories = {key: {normalize(n) for n in names.splitlines() if n.strip()}
                  for key, names in CATEGORY_NAMES.items()}
    all_names = set().union(*categories.values())
    seen_names = set()
    core = {}
    audit = []
    for skill in read_jsonl(source / "data/skills.jsonl"):
        name = normalize(skill["name"])
        seen_names.add(name)
        tags = sorted(key for key, names in categories.items() if name in names)
        # This subject-specific literature scan has an upstream mojibake name.
        if name.startswith("petase-") and "literature" in name:
            tags = ["literature_discovery_and_review"]
        excluded = EXCLUDE_IDS.get(skill["id"])
        if tags and not excluded:
            core[skill["id"]] = {"skill": skill, "categories": tags}
        if tags or excluded or DISCOVERY_PATTERN.search(skill["name"] + " " + skill["description"]):
            audit.append({"skill_id": skill["id"], "name": skill["name"],
                          "description": skill["description"], "source_sub": skill["sub"],
                          "selected_as_core": bool(tags and not excluded),
                          "categories": tags if not excluded else [],
                          "reason": excluded or ("Curated research capability allowlist." if tags else
                                    "Outside the curated academic/deep-research scope or not selected conservatively.")})
    missing_names = sorted(all_names - seen_names)
    if missing_names:
        raise ValueError(f"Unknown curated skill names: {missing_names}")
    print(f"Selected {len(core)} core research skills; checked {len(audit)} metadata candidates.", flush=True)

    split_data = {}
    selected_skills = {key: value["skill"] for key, value in core.items()}
    skill_splits = {}
    for split in ("train", "test"):
        # Original IDs and records are retained, including every multi-skill label.
        queries = [q for q in read_jsonl(source / f"data/queries/{split}.jsonl")
                   if set(q["skill_ids"]) & core.keys()]
        query_ids = {q["id"] for q in queries}
        gold_ids = {sid for q in queries for sid in q["skill_ids"]}
        split_skills = [s for s in read_jsonl(source / f"data/skills/{split}.jsonl")
                        if s["id"] in core or s["id"] in gold_ids]
        pool_ids = {s["id"] for s in split_skills}
        if not gold_ids <= pool_ids:
            raise ValueError(f"Missing gold skills in {split}: {gold_ids - pool_ids}")
        for skill in split_skills:
            selected_skills[skill["id"]] = skill
            skill_splits.setdefault(skill["id"], []).append(split)
        qrels = [r for r in read_jsonl(source / f"data/qrels/{split}.jsonl")
                 if r["query_id"] in query_ids]
        relations = {}
        for relation in qrels:
            if relation["skill_id"] not in pool_ids:
                raise ValueError("Dangling qrel skill reference.")
            relations.setdefault(relation["query_id"], set()).add(relation["skill_id"])
        annotations = []
        for query in queries:
            gold = set(query["skill_ids"])
            if gold != relations.get(query["id"], set()) or query["k"] != len(gold):
                raise ValueError("Query/qrel completeness mismatch.")
            research_ids = sorted(gold & core.keys())
            support_ids = sorted(gold - core.keys())
            annotations.append({"query_id": query["id"], "split": split,
                                "scope": "mixed" if support_ids else "research_only",
                                "research_skill_ids": research_ids,
                                "support_skill_ids": support_ids,
                                "categories": sorted({c for sid in research_ids for c in core[sid]["categories"]})})
        split_data[split] = {"skills": split_skills, "queries": queries,
                             "qrels": qrels, "annotations": annotations}
    train_ids = {s["id"] for s in split_data["train"]["skills"]}
    test_ids = {s["id"] for s in split_data["test"]["skills"]}
    if train_ids & test_ids:
        raise ValueError("Train/test skill ID overlap.")
    if {q["id"] for q in split_data["train"]["queries"]} & {q["id"] for q in split_data["test"]["queries"]}:
        raise ValueError("Train/test query ID overlap.")

    stats = {}
    research_only_stats = {}
    for split, data in split_data.items():
        for key in ("skills", "queries", "qrels"):
            write_jsonl(output / f"data/{key}/{split}.jsonl", data[key])
        write_jsonl(output / f"annotations/queries/{split}.jsonl", data["annotations"])
        stats[split] = {"skills": len(data["skills"]),
                        "core_research_skills": sum(s["id"] in core for s in data["skills"]),
                        "support_skills": sum(s["id"] not in core for s in data["skills"]),
                        "queries": len(data["queries"]), "qrels": len(data["qrels"]),
                        "research_only_queries": sum(a["scope"] == "research_only" for a in data["annotations"]),
                        "mixed_queries": sum(a["scope"] == "mixed" for a in data["annotations"]),
                        "query_skill_counts": dict(sorted(Counter(q["k"] for q in data["queries"]).items())),
                        "query_categories": dict(sorted(Counter(c for a in data["annotations"] for c in a["categories"]).items()))}
        pure_annotations = [a for a in data["annotations"] if a["scope"] == "research_only"]
        pure_ids = {a["query_id"] for a in pure_annotations}
        pure_data = {
            "queries": [q for q in data["queries"] if q["id"] in pure_ids],
            "skills": [s for s in data["skills"] if s["id"] in core],
            "qrels": [r for r in data["qrels"] if r["query_id"] in pure_ids],
        }
        for key, records in pure_data.items():
            write_jsonl(output / f"research_only/data/{key}/{split}.jsonl", records)
        write_jsonl(output / f"research_only/annotations/queries/{split}.jsonl", pure_annotations)
        research_only_stats[split] = {key: len(records) for key, records in pure_data.items()}
        research_only_stats[split]["query_skill_counts"] = dict(sorted(Counter(q["k"] for q in pure_data["queries"]).items()))
    write_jsonl(output / "research_only/data/skills.jsonl", (core[sid]["skill"] for sid in sorted(core)))
    write_json(output / "research_only/SUBSET_INFO.json", {
        "scope": "Strict research-only view: every original gold skill is research_core.",
        "core_research_skills_total": len(core), "stats": research_only_stats,
        "source": "Parent directory MANIFEST.json and selection_policy.json.",
        "evaluation": "Use matching train/test pools. The all-skills union also contains out-of-split skills.",
        "limitations": "Small held-out test set; inherited upstream synthetic queries and labels. Not a new expert-annotated benchmark."})
    write_jsonl(output / "data/skills.jsonl", (selected_skills[sid] for sid in sorted(selected_skills)))
    write_jsonl(output / "annotations/skills.jsonl", (
        {"skill_id": sid, "name": selected_skills[sid]["name"],
         "role": "research_core" if sid in core else "label_support",
         "categories": core[sid]["categories"] if sid in core else [],
         "source_splits": skill_splits.get(sid, []),
         "reason": "Curated research capability." if sid in core else "Required to preserve all gold labels of a mixed research query."}
        for sid in sorted(selected_skills)))
    write_jsonl(output / "annotations/selection_decisions.jsonl", sorted(audit, key=lambda a: (a["name"], a["skill_id"])))
    shutil.copyfile(source / "data/taxonomy.json", output / "data/source_taxonomy.json")
    shutil.copyfile(source / "README.md", output / "SOURCE_DATASET_CARD.md")
    write_json(output / "selection_policy.json", {
        "version": POLICY_VERSION,
        "scope": "Academic research plus general-purpose deep research and evidence synthesis.",
        "categories": {key: sorted(names) for key, names in categories.items()},
        "id_exclusions": EXCLUDE_IDS,
        "special_inclusion": "PETase literature scan, identified by name prefix and literature keyword.",
        "query_policy": "Keep a query iff at least one original gold skill is research_core. Preserve ALL original gold skills and qrels; add non-core gold skills as label_support.",
        "exclusions": ["Dedicated market, competitor, sales, SEO, and UX research workflows.",
                       "Routine codebase analysis, API lookup, debugging, feature planning, and generic ML development.",
                       "Generic tools merely mentioning research in passing; no whole-body keyword classification."],
        "review_method": "Curated names based on metadata review, with selected body excerpts checked for ambiguity. Not an exhaustive independent expert annotation.",
        "limitations": ["Inherited synthetic English-dominant queries; no new requests or negative examples generated.",
                        "Broad research skills can have original queries in other domains; mixed queries are explicitly annotated.",
                        "Existing upstream encoding artifacts and labels are retained unchanged.",
                        "A smaller candidate pool can inflate routing scores; compare on the original candidate pool as well.",
                        "Original train/test split retained; semantic near-duplicate leakage has not been independently re-audited."]})

    readme = f"""# SKILLRET v1.1 Research Subset

Scope: academic research plus general-purpose deep research. Dedicated market,
competitor, sales, SEO, and UX research workflows are excluded as core skills.

## Recommended research-only view

Use `research_only/data/` for the strict subset. All gold skills in each query are
research core; no labels are removed. This view contains
{research_only_stats['train']['queries']} training queries with {research_only_stats['train']['skills']} train-pool skills,
and {research_only_stats['test']['queries']} test queries with {research_only_stats['test']['skills']} test-pool skills.
`research_only/data/skills.jsonl` contains all {len(core)} research core skills,
including skills not assigned to upstream train/test splits.
See `research_only/SUBSET_INFO.json` for counts.

## Extended view including mixed tasks

The root `data/` view also keeps queries that combine research with other tasks.
It is a superset of the research-only view, not a disjoint extra dataset.
Some upstream mixed-task labels may be weakly aligned with query text; they are
preserved for traceability, not independently endorsed as correct.

| Split | Queries | Skills | Research core | Label support | Research-only queries | Mixed queries |
|---|---:|---:|---:|---:|---:|---:|
"""
    for split, values in stats.items():
        readme += f"| {split} | {values['queries']} | {values['skills']} | {values['core_research_skills']} | {values['support_skills']} | {values['research_only_queries']} | {values['mixed_queries']} |\n"
    readme += f"""
Source: https://huggingface.co/datasets/ThakiCloud/SKILLRET
Source revision: `{SOURCE_REVISION}`. Selection policy version: {POLICY_VERSION}.

## Files and evaluation

- `data/skills/{{train,test}}.jsonl`: split candidate pools, with original complete skill records.
- `data/queries/{{train,test}}.jsonl`: original queries; IDs, text, names, and all gold labels unchanged.
- `data/qrels/{{train,test}}.jsonl`: all original relevance rows for the selected queries.
- `data/skills.jsonl`: union of research skills (including out-of-split skills) and required support skills. Do not use this union in place of the split pools for standard evaluation.
- `annotations/queries/{{train,test}}.jsonl`: join on `query_id`; `scope=research_only` means every gold skill is research core, while `scope=mixed` also needs support skills.
- `annotations/skills.jsonl`: join on `skill_id`; separates `research_core` from `label_support` and records topical categories.
- `annotations/selection_decisions.jsonl`: candidate metadata and inclusion/exclusion decisions.
- `selection_policy.json`: exact curated names, ID-specific exclusions, scope, and limitations.
- `MANIFEST.json`: source provenance, checksums, counts, and validation results.
- `SOURCE_DATASET_CARD.md`, `data/source_taxonomy.json`: original upstream documentation and taxonomy.

For strict research-only evaluation, filter queries by annotation `scope=research_only`
and filter qrels by the retained query IDs. Never remove individual labels from a mixed query.
Use the matching split pool. Also report results against the original SKILLRET split pool
to measure interference from non-research skills; scores on the smaller pool are not directly
comparable with the original benchmark. Categories overlap and should not be summed.

Selection is skill-led and conservatively curated, not an independent human annotation
of every query. Original synthetic requests, encoding artifacts, and label limitations remain.
No new no-match queries are introduced. Original split membership is retained.

## Reproduce

From the project root, with the original downloaded dataset present:

```text
python scripts/build_skillret_research.py --output data/skillret-research-v1.1-rebuilt
```

The output directory must normally be empty or absent. `--refresh-generated` can
refresh an earlier generated output only if its source identity and every recorded
file checksum still match; edited files and unrelated outputs are refused.

## License and attribution

Benchmark queries/metadata/taxonomy: upstream Apache-2.0. Skill contents retain their
individual upstream MIT or Apache-2.0 licenses and source attribution fields.
See `SOURCE_DATASET_CARD.md`; retain upstream notices and cite the SkillRet paper.
This subset adds selection annotations and does not relicense source skill contents.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
              "sha256": sha256(path)} for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "MANIFEST.json"]
    manifest = {"name": "skillret-research-v1.1", "selection_policy_version": POLICY_VERSION,
                "source": {"dataset": provenance["dataset"], "version": provenance["version"],
                           "revision": SOURCE_REVISION, "source": provenance["source"],
                           "files": provenance["files"]},
                "core_research_skills_total": len(core),
                "skills_union_total": len(selected_skills),
                "core_skills_outside_source_splits": len(set(core) - (train_ids | test_ids)),
                "core_categories": dict(sorted(Counter(c for item in core.values() for c in item["categories"]).items())),
                "stats": stats,
                "research_only_stats": research_only_stats,
                "validation": {"source_hashes_verified": True, "all_gold_labels_preserved": True,
                               "all_qrel_references_valid": True, "no_skill_id_overlap_between_splits": True,
                               "no_query_id_overlap_between_splits": True},
                "files": files,
                "builder_sha256": sha256(Path(__file__))}
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps({"output": str(output.resolve()), "core_skills": len(core), "union_skills": len(selected_skills),
                      "stats": stats}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/skillret-v1.1"))
    parser.add_argument("--output", type=Path, default=Path("data/skillret-research-v1.1"))
    parser.add_argument("--refresh-generated", action="store_true", help="Refresh only an unmodified output previously built by this script.")
    args = parser.parse_args()
    build(args.source, args.output, args.refresh_generated)
