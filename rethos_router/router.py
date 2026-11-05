"""Public route results contain exactly skills, harness and model."""
from .common import RequestError, load_config, resolve_path
from .policy import BenchmarkPolicy
from .retrieval import SkillIndex, SkillRetEncoder, load_pool
from .review import HTTPReviewer, review_messages, validate_decision


def load_index(config, *, rebuild=False):
    root = config['_root']
    skills, digest = load_pool(resolve_path(config['skill_pool_registry'], root), root, config['skill_scope'])
    encoder = SkillRetEncoder(config['embedding'], root)
    return SkillIndex.load_or_build(skills, digest, encoder,
        resolve_path(config['embedding']['index_path'], root), rebuild=rebuild)


class Router:
    """Reusable instance: index, encoder and policy are loaded once.

    skills contains stable registry IDs, preserving distinct same-name implementations.
    One valid request invokes reviewer.review once, even when no skill is selected.
    """
    def __init__(self, index, reviewer, policy, retrieval, max_request_chars=32_000):
        self.index, self.reviewer, self.policy = index, reviewer, policy
        self.retrieval, self.max_request_chars = retrieval, max_request_chars

    @classmethod
    def from_config(cls, path=None, *, root=None):
        config = load_config(path, **({'root': root} if root is not None else {}))
        # Validate credentials/config before doing expensive model/index work.
        reviewer = HTTPReviewer(config['review'])
        policy = BenchmarkPolicy.load(resolve_path(config['agent_registry'], config['_root']), config['fallback'])
        return cls(load_index(config), reviewer, policy, config['retrieval'], config['max_request_chars'])

    def route(self, request):
        if not isinstance(request, str) or not request.strip():
            raise RequestError('Request must be nonempty text')
        if len(request) > self.max_request_chars:
            raise RequestError(f'Request exceeds {self.max_request_chars} characters')
        candidates = self.index.search(request, self.retrieval['top_k'])
        result = self.reviewer.review(review_messages(request, candidates, self.retrieval))
        decision = validate_decision(result, candidates, self.retrieval['max_selected_skills'])
        selected = self.policy.select(decision)
        return {'skills': list(decision.selected_skill_ids), 'harness': selected['harness'], 'model': selected['model']}
