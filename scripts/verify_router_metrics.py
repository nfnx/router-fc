"""Independent sanity checks for multi-label retrieval metrics and test pools."""
import math
from evaluate_local_routers import load_views, metrics

def main():
    assert metrics(['a', 'b', 'c'], {'a', 'b'})['ndcg@10'] == 1
    assert metrics(['a', 'b', 'c'], {'a', 'b'})['recall@1'] == .5
    assert metrics(['a', 'b', 'c'], {'a', 'b'})['hit@1'] == 1
    assert metrics(['a', 'b', 'c'], {'a', 'b'})['completeness@1'] == 0
    assert metrics(['a', 'b', 'c'], {'a', 'b'})['completeness@5'] == 1
    assert metrics(['c', 'a', 'b'], {'a'})['mrr@10'] == .5
    assert math.isclose(metrics(['c', 'a', 'b'], {'a'})['ndcg@10'], 1/math.log2(3))
    wrong = metrics(list('abcdefghijk'), {'k'})
    assert wrong['ndcg@10'] == wrong['recall@10'] == wrong['mrr@10'] == 0
    assert wrong['recall@20'] == 1
    views = load_views()
    assert len(views['extended']['queries']) == 69 and len(views['extended']['skills']) == 125
    assert len(views['research_only']['queries']) == 22 and len(views['research_only']['skills']) == 64
    assert {q['id'] for q in views['research_only']['queries']} <= {q['id'] for q in views['extended']['queries']}
    print('PASS: hand-computed metrics, multi-label coverage, exact test pools, subset relationship')

if __name__ == '__main__':
    main()
