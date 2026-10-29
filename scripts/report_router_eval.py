"""Build a Chinese report from actual per-query local evaluation artifacts."""
import argparse
import json
from pathlib import Path
from evaluate_local_routers import ROOT, load_views, dump, metrics, aggregate

NAMES = {'bm25_embedding': 'BM25', 'qwen_embedding': 'Qwen3 0.6B', 'qwen_reranked': 'Qwen3 0.6B + reranker', 'skillret_embedding': 'SkillRet 0.6B', 'skillret_reranked': 'SkillRet 0.6B + reranker', 'skillrouter_embedding': 'SkillRouter 0.6B', 'skillrouter_reranked': 'SkillRouter 0.6B + reranker', 'r3_embedding': 'R3 0.6B', 'r3_reranked': 'R3 0.6B + reranker', 'skillret8b_embedding': 'SkillRet 8B (NF4)', 'skillret8b_reranked': 'SkillRet 8B (NF4) + 0.6B reranker', 'skillsight_embedding': 'SkillSight + Qwen3 0.6B'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/research-test-20261029')
    args = parser.parse_args()
    directory = ROOT / args.run
    views = load_views()
    artifacts = {name: json.loads((directory / (name + '.json')).read_text(encoding='utf-8')) for name in NAMES if (directory / (name + '.json')).exists()}
    assert artifacts, 'No completed evaluation results'
    for name, artifact in artifacts.items():
        for viewname, view in views.items():
            rows = artifact[viewname]['rows']
            assert [r['query_id'] for r in rows] == [q['id'] for q in view['queries']]
            pool = {s['id'] for s in view['skills']}
            assert all(set(r['ranking']) == pool and len(r['ranking']) == len(pool) for r in rows)
            for row in rows:
                assert row['metrics'] == metrics(row['ranking'], view['gold'][row['query_id']])
            assert artifact[viewname]['metrics'] == aggregate(rows)
            if name.endswith('_reranked'):
                baseline = artifacts[name.replace('_reranked', '_embedding')][viewname]['rows']
                assert all(set(row['ranking'][:20]) == set(base['ranking'][:20]) for row, base in zip(rows, baseline))
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    for snapshot in protocol.get('measurement_source_snapshots', []):
        import hashlib
        assert hashlib.sha256((ROOT / snapshot['path']).read_bytes()).hexdigest() == snapshot['sha256']
    artifact_sources = protocol.setdefault('artifact_evaluator_sha256', {})
    for name in artifacts:
        artifact_sources.setdefault(name, protocol['script_sha256'])
    dump(directory / 'protocol.json', protocol)
    summary = {name: {view: artifact[view]['metrics'] for view in views} for name, artifact in artifacts.items()}
    mixed_ids = {q['id'] for q in views['extended']['queries'] if len(q['skill_ids']) > 1}
    for name, artifact in artifacts.items():
        summary[name]['mixed_tasks'] = aggregate([r for r in artifact['extended']['rows'] if r['query_id'] in mixed_ids])
    dump(directory / 'summary.json', summary)
    # Human-readable audit: join IDs back to requests and skill names, retaining
    # each method's gold ranks and top ten without requiring a manual join.
    with (directory / 'prediction_audit.jsonl').open('w', encoding='utf-8') as audit:
        for viewname, view in views.items():
            skills = {s['id']: s for s in view['skills']}
            for i, q in enumerate(view['queries']):
                row = {'view': viewname, 'query_id': q['id'], 'query': q['query'], 'gold': [{'id': sid, 'name': skills[sid]['name']} for sid in q['skill_ids']], 'methods': {}}
                for method, artifact in artifacts.items():
                    result = artifact[viewname]['rows'][i]
                    row['methods'][method] = {'gold_ranks': {sid: result['ranking'].index(sid)+1 for sid in q['skill_ids']}, 'top10': [{'id': sid, 'name': skills[sid]['name']} for sid in result['ranking'][:10]], 'metrics': result['metrics']}
                audit.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# Research 子集本地评测报告', '', f'完成方法配置：{len(artifacts)} / {len(NAMES)}。全部数字来自本机运行产生的逐请求预测，不引用论文成绩充当本地结果。', '', '## 评测范围与解释', '', '- 纯研究：22 条 test 请求、64 个候选技能；扩展集：69 条 test 请求、125 个候选技能、125 个 gold 标签。纯研究请求包含在扩展集中，不是 91 条独立请求。', '- 保留每个视图的原候选池与 gold 集合；不执行请求任务、不调用外部模型 API、不使用 train 调参。', '- 所有神经模型使用 name | description | skill_md，4096 token 上限、左侧 padding、末 token 池化与 L2 归一化。重排各自检索器的 top20，使用 yes/no 最末 token 的 logit 差排序，尾部保持原序。', '- 0.6B 使用 BF16；8B 使用 bitsandbytes NF4 双重量化、BF16 计算。8B 结果不代表未量化论文结果。模型逐个加载，本轮是统一硬件预算下的适配评测，不是论文原始配置的逐项复现。', '- SkillRouter 保留作者发布的 coding task 指令，但文档输入统一为上述格式；R3 保留作者指令和 tokenizer chat template。Qwen 与 SkillRet 使用 SKILLRET 官方评测的 skill search 指令。', '- SkillSight 直接调用所下载官方代码，复用本轮 Qwen 嵌入，candidate_k=300（受本地候选数量上限限制），beta=1。其当前官方 retriever 调用 lam=None 的融合分支，与 fusion.py 注释所述论文 Eq.15 的 lam=1 分支不同；本报告明确评测代码版本行为。', '- Recall 为逐请求 gold 覆盖率的宏平均；Completeness 是 top-k 包含该请求全部 gold 的请求比例。NDCG 使用二元 qrels，MRR 截止 10。', '- 在线耗时 = 单条请求编码 + 搜索排序 + 重排；不含权重加载与离线技能索引。单次运行、先 warmup，无并发吞吐或服务 SLA 含义。SkillSight 的查询编码耗时复用 Qwen 实测值。', '', f"硬件：{protocol['gpu']}；PyTorch {protocol['torch']}；随机种子 42。", '']
    for viewname, title in [('extended', '扩展测试集'), ('research_only', '纯研究测试集')]:
        lines += [f'## {title}', '', '| 方法 | Hit@1 | Recall@5 | Recall@10 | Completeness@10 | NDCG@10 | MRR@10 | 在线中位数 ms | P95 ms |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        ordered = sorted(artifacts, key=lambda name: summary[name][viewname]['ndcg@10'], reverse=True)
        for name in ordered:
            m = summary[name][viewname]
            vals = [f'{m[k]*100:.2f}%' for k in ['hit@1', 'recall@5', 'recall@10', 'completeness@10', 'ndcg@10', 'mrr@10']]
            lines.append('| ' + NAMES[name] + ' | ' + ' | '.join(vals + [f"{m['online_ms_median']:.1f}", f"{m['online_ms_p95']:.1f}"]) + ' |')
        best = ordered[0]
        lo, hi = summary[best][viewname]['ndcg@10_bootstrap95']
        lines += ['', f"本次 NDCG@10 最高：{NAMES[best]}，{summary[best][viewname]['ndcg@10']:.4f}；按请求自助采样 2000 次的 95% 区间 [{lo:.4f}, {hi:.4f}]。该区间不校正多方法选择，也不能据此断言方法差异显著。", '']
    best_quality = max(artifacts, key=lambda n: summary[n]['extended']['ndcg@10'])
    fast_methods = [n for n in artifacts if summary[n]['extended']['online_ms_median'] < 100]
    best_fast = max(fast_methods, key=lambda n: summary[n]['extended']['ndcg@10']) if fast_methods else best_quality
    best_quality_metrics = summary[best_quality]['extended']
    best_fast_metrics = summary[best_fast]['extended']
    lines += ['## 本轮选型判断', '', f"以扩展集 NDCG@10 排序，最高配置为 {NAMES[best_quality]}（{best_quality_metrics['ndcg@10']*100:.2f}%，完整覆盖率 {best_quality_metrics['completeness@10']*100:.2f}%，单请求在线中位耗时 {best_quality_metrics['online_ms_median']:.1f}ms）。", '', f"若本机单次测量的中位耗时预算约 100ms，可优先参考 {NAMES[best_fast]}（NDCG@10 {best_fast_metrics['ndcg@10']*100:.2f}%，完整覆盖率 {best_fast_metrics['completeness@10']*100:.2f}%，中位耗时 {best_fast_metrics['online_ms_median']:.1f}ms）。此门槛仅用于解读本轮结果，不是生产延迟承诺。", '', '这两项是样本内观察，不等于统计显著或生产选型结论；在进入实现前，应再用全量测试候选池验证路由质量、长文档策略及拒识行为。', '']
    lines += ['## 混合任务与重排收益', '', '仅取扩展集中 47 条多技能请求，候选池仍为 125；该切片不额外调用模型。', '', '| 方法 | Recall@10 | Completeness@10 | NDCG@10 |', '|---|---:|---:|---:|']
    for name in sorted(artifacts, key=lambda n: summary[n]['mixed_tasks']['completeness@10'], reverse=True):
        m = summary[name]['mixed_tasks']
        lines.append(f"| {NAMES[name]} | {m['recall@10']*100:.2f}% | {m['completeness@10']*100:.2f}% | {m['ndcg@10']*100:.2f}% |")
    lines += ['', '| 配对方法（扩展集） | 重排后 NDCG@10 变化 | 完整覆盖率变化 | 在线中位耗时倍数 |', '|---|---:|---:|---:|']
    for family in ['qwen', 'skillret', 'skillrouter', 'r3', 'skillret8b']:
        e, r = family + '_embedding', family + '_reranked'
        if e in artifacts and r in artifacts:
            a, b = summary[e]['extended'], summary[r]['extended']
            lines.append(f"| {NAMES[e]} | {(b['ndcg@10']-a['ndcg@10'])*100:+.2f} 个百分点 | {(b['completeness@10']-a['completeness@10'])*100:+.2f} 个百分点 | {b['online_ms_median']/a['online_ms_median']:.1f}x |")
    lines += ['', '## 输入截断与离线建库', '', '| 方法 | 扩展文档截断 | 扩展查询截断 | 纯研究文档截断 | 纯研究查询截断 | 扩展建库秒 |', '|---|---:|---:|---:|---:|---:|']
    for name, artifact in artifacts.items():
        if 'document_tokens' not in artifact['extended']:
            continue
        e, r = artifact['extended'], artifact['research_only']
        lines.append(f"| {NAMES[name]} | {e['document_tokens']['truncated']}/{e['document_tokens']['texts']} | {e['query_truncated']}/69 | {r['document_tokens']['truncated']}/{r['document_tokens']['texts']} | {r['query_truncated']}/22 | {e['index_s']:.2f} |")
    lines += ['', '重排输入截断统计（两个视图合计，分母包含 1 对未截断 warmup；每次最多输入 4096 token，保留 assistant 后缀）：', '', '| 方法 | 被截断的请求—技能对 |', '|---|---:|']
    for name, artifact in artifacts.items():
        if '_reranker_token_stats' in artifact:
            stats = artifact['_reranker_token_stats']
            lines.append(f"| {NAMES[name]} | {stats['truncated']}/{stats['pairs_including_warmup']} |")
    lines += ['', '## 典型错误（扩展集）', '']
    dense_name = 'skillret_embedding' if 'skillret_embedding' in artifacts else next(iter(artifacts))
    compared = 'skillret_reranked' if 'skillret_reranked' in artifacts else dense_name
    base_rows = {r['query_id']: r for r in artifacts[dense_name]['extended']['rows']}
    rows = artifacts[compared]['extended']['rows']
    queries = {q['id']: q for q in views['extended']['queries']}
    skills = {s['id']: s for s in views['extended']['skills']}
    cases = sorted(rows, key=lambda r: r['metrics']['ndcg@10'])[:5]
    for row in cases:
        q = queries[row['query_id']]
        ranks = {sid: row['ranking'].index(sid) + 1 for sid in q['skill_ids']}
        lines += [f"### {q['id']}", '', f"用户请求：{q['query']}", '', 'Gold：' + '；'.join(f"{skills[sid]['name']}（第 {ranks[sid]} 名）" for sid in q['skill_ids']), '', 'Top3：' + '；'.join(skills[sid]['name'] for sid in row['ranking'][:3]), '', f"{NAMES[dense_name]} NDCG@10={base_rows[q['id']]['metrics']['ndcg@10']:.4f}；{NAMES[compared]}={row['metrics']['ndcg@10']:.4f}。", '']
    if 'skillret_embedding' in artifacts and 'skillret_reranked' in artifacts:
        pure = views['research_only']
        qmap = {q['id']: q for q in pure['queries']}
        smap = {s['id']: s for s in pure['skills']}
        base = {r['query_id']: r for r in artifacts['skillret_embedding']['research_only']['rows']}
        reranked = artifacts['skillret_reranked']['research_only']['rows']
        degraded = sorted(reranked, key=lambda r: r['metrics']['ndcg@10']-base[r['query_id']]['metrics']['ndcg@10'])[:2]
        lines += ['## 纯研究重排复核案例', '', '以下为本轮 SkillRet 重排相对检索器退步最大的两个纯研究案例。只描述实际排名变化，不据此认定 gold 错误或确定退步原因。', '']
        for row in degraded:
            q = qmap[row['query_id']]
            sid = q['skill_ids'][0]
            lines += [f"### {q['id']}", '', q['query'], '', f"Gold：{smap[sid]['name']}；检索第 {base[q['id']]['ranking'].index(sid)+1} 名 → 重排第 {row['ranking'].index(sid)+1} 名。", '', '重排 Top3：' + '；'.join(smap[s]['name'] for s in row['ranking'][:3]), '']
    lines += ['## 限制与结果文件', '', '- 这是经模型辅助清洗的合成请求小样本；保留标签不构成独立专家金标准。未标注替代技能可能同样可用。', '- 候选池只有 64/125，不能外推到 SKILLRET 全集 6006 个测试技能，更不能与论文全量成绩直接比较。', '- SkillRet 使用原始 train 训练；本轮只评原始 test。R3 等模型训练来源可能包含 SKILLRET，未完成跨项目训练样本重叠审计，不将其称为严格零样本评测。', '- 4096 token 会截断部分长技能；截断数见上表。没有验证路由后技能执行成功率、拒识能力、训练/测试语义近重复或跨语言泛化。', f'- 逐请求排名与指标：`{args.run}/*_embedding.json`、`*_reranked.json`；统一汇总：`summary.json`；协议、输入哈希和权重 revision：`protocol.json`。', '- 权重文件完整性：`models/MANIFEST.json`；代码来源：`config/deployment_sources.json`；环境：`config/requirements-local.lock.txt`。', '', '## 上游来源', '', '- [SKILLRET](https://github.com/ThakiCloud/SKILLRET)', '- [SkillRouter](https://github.com/zhengyanzhao1997/SkillRouter)', '- [R3-Skill](https://github.com/Tencent/R3-Skill)', '- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)', '- [SkillSight](https://github.com/xiaojinying/SkillSight)', '']
    lines = [line.replace('所有神经模型使用', '所有嵌入模型使用') for line in lines]
    lines += ['本轮嵌入批量为 2（8B 为 1），重排批量为 8，并在每条请求内按文本 token 长度分组后恢复候选对应关系。部分配置评测期间有并行权重下载，桌面进程也可能带来系统负载，耗时仅作本机单次测量。', '']
    if protocol.get('compatible_control_fixes'):
        lines += ['R3 重排适配修正了 Transformers 5.5.4 的模板返回类型，并从实际渲染结果提取长文本截断后保留的 assistant 后缀；在修复前没有接受任何完整 R3 重排结果。此前九个已完成配置的评分函数未改动、预测予以保留。`protocol.json` 的 `artifact_evaluator_sha256` 为每个配置记录实测代码版本，`measurement_source_snapshots` 保留相应完整源码快照。', '']
    lines += ['BM25 使用 rank-bm25 的 BM25Okapi 默认参数（k1=1.5、b=0.75），以小写 Unicode \\w+ 分词、不去停用词、不截断全文；该实现不同于 SKILLRET 当前官方脚本的 bm25s 基线。SkillSight 的词汇统计采用官方默认每文档前 200 个不同 token。', '']
    model_manifest = json.loads((ROOT / 'models/MANIFEST.json').read_text())
    lines += ['## 本地权重清单', '', '下列为官方原始快照；8B 的 NF4 量化在加载时完成。所有文件下载时均核对上游 LFS SHA256 或 Git blob SHA1，完整 SHA256 保存在模型清单中。', '', '| Hugging Face 模型 | Revision（前12位） | 本地快照 GiB |', '|---|---|---:|']
    for repo, item in model_manifest.items():
        lines.append(f"| {repo} | {item['revision'][:12]} | {sum(f['size'] for f in item['files'])/1024**3:.2f} |")
    lines += ['', f"已校验快照共 {len(model_manifest)} 份，占 {sum(f['size'] for item in model_manifest.values() for f in item['files'])/1024**3:.2f} GiB（不含代码、Python 环境和下载缓存）。", '']
    attention_check = ROOT / 'runs/efficient-attention-validation.json'
    if attention_check.exists():
        check = json.loads(attention_check.read_text())
        lines += ['## 本机注意力后端验证', '', f"PyTorch 2.7.1 的 Windows wheel 不含 Flash Attention。本轮统一采用 memory-efficient SDPA，并使用 Transformers 现有的显式 KV 重复分支处理 GQA，禁用默认 math/cuDNN 后备内核。固定两对 4096-token 文本的检查中，默认路径 {check['default_seconds']:.3f}s，优化路径 {check['efficient_seconds']:.3f}s；yes/no logit 差的最大绝对变化 {check['max_abs_difference']:.4f}，该固定样例的排序一致。此检查不证明全语料数值完全相同，正式表格全部在统一优化后端下重新计算。", '', '[PyTorch SDPA 文档](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)；验证明细：`runs/efficient-attention-validation.json`。', '']
    report = ROOT / 'reports/research-test-20261029.md'
    lines[4:4] = ['![Quality versus local latency](research-test-20261029-quality-latency.png)', '', '图中越靠左上，表示本轮测试中质量越高、耗时越低；虚线只表示本轮观测的非支配配置。', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    dump(report.with_suffix('.json'), {'completed_methods': len(artifacts), 'expected_methods': len(NAMES), 'run_directory': args.run, 'protocol': protocol, 'summary': summary, 'code_sources': json.loads((ROOT / 'config/deployment_sources.json').read_text(encoding='utf-8-sig'))})
    print(report)

if __name__ == '__main__':
    main()
