"""Standalone scientific plot of measured retrieval quality versus latency."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'bm25_embedding': 'BM25', 'qwen_embedding': 'Qwen 0.6B', 'qwen_reranked': 'Qwen + RR', 'skillret_embedding': 'SkillRet 0.6B', 'skillret_reranked': 'SkillRet 0.6B + RR', 'skillrouter_embedding': 'SkillRouter 0.6B', 'skillrouter_reranked': 'SkillRouter + RR', 'r3_embedding': 'R3 0.6B', 'r3_reranked': 'R3 + RR', 'skillret8b_embedding': 'SkillRet 8B NF4', 'skillret8b_reranked': 'SkillRet 8B NF4 + RR', 'skillsight_embedding': 'SkillSight + Qwen'}
COLORS = {'bm25': '#6b7280', 'qwen': '#2563eb', 'skillret': '#059669', 'skillrouter': '#9333ea', 'r3': '#ea580c', 'skillret8b': '#0d9488', 'skillsight': '#be185d'}

def main():
    data = json.loads((ROOT / 'reports/research-test-20261029.json').read_text(encoding='utf-8'))
    metrics = {name: values['extended'] for name, values in data['summary'].items()}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(12, 7.5), layout='constrained')
    for name, m in metrics.items():
        family = name.rsplit('_', 1)[0]
        x, y = m['online_ms_median'], m['ndcg@10'] * 100
        reranked = name.endswith('_reranked')
        ax.scatter(x, y, s=95, marker='^' if reranked else 'o', color=COLORS[family], edgecolors='white', linewidths=.8, zorder=3)
    # Spread labels within each latency region while keeping the measured point exact.
    groups = [[n for n in metrics if metrics[n]['online_ms_median'] < 500], [n for n in metrics if metrics[n]['online_ms_median'] >= 500]]
    offsets = {}
    for group in groups:
        last_y = -100
        for name in sorted(group, key=lambda n: metrics[n]['ndcg@10']):
            y = metrics[name]['ndcg@10'] * 100
            label_y = max(y, last_y + 2.4)
            offsets[name] = label_y
            last_y = label_y
    for name, m in metrics.items():
        x, y = m['online_ms_median'], m['ndcg@10'] * 100
        ax.annotate(LABELS[name], xy=(x, y), xytext=(x * (1.18 if x < 500 else .87), offsets[name]), ha='left' if x < 500 else 'right', va='center', fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#9ca3af', 'lw': .7})
    frontier = []
    best = float('-inf')
    for name in sorted(metrics, key=lambda n: metrics[n]['online_ms_median']):
        m = metrics[name]
        if m['ndcg@10'] > best:
            frontier.append((m['online_ms_median'], m['ndcg@10'] * 100))
            best = m['ndcg@10']
    if frontier:
        ax.plot(*zip(*frontier), '--', color='#9ca3af', linewidth=1, zorder=1)
    ax.set_xscale('log')
    ax.set_xlabel('Median online latency per query (ms, logarithmic scale)')
    ax.set_ylabel('NDCG@10 (%)')
    ax.set_title('Cleaned research extended test: quality vs. local latency', loc='left', fontweight='bold', pad=18)
    ax.text(0, 1.015, '69 queries / 125 candidate skills | RTX 4070 Laptop 8GB | 4096-token cap | RR: top 20', transform=ax.transAxes, fontsize=9, color='#4b5563')
    ax.set_ylim(45, max(102, max(offsets.values()) + 3))
    xs = [m['online_ms_median'] for m in metrics.values()]
    ax.set_xlim(min(xs)*.55, max(xs)*1.65)
    ax.grid(axis='both', color='#e5e7eb', linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#374151', label='Retrieval / calibration', markersize=8), Line2D([0], [0], marker='^', color='none', markerfacecolor='#374151', label='Retrieval + reranking', markersize=8), Line2D([0], [0], linestyle='--', color='#9ca3af', label='Observed Pareto frontier')], loc='lower right', frameon=True)
    fig.savefig(ROOT / 'reports/research-test-20261029-quality-latency.png', dpi=180)
    svg_path = ROOT / 'reports/research-test-20261029-quality-latency.svg'
    fig.savefig(svg_path)
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    print('Saved standalone PNG and SVG plots')

if __name__ == '__main__':
    main()
