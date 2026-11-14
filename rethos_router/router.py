"""Public route results contain exactly skills, harness and model."""
import json
import logging
from .common import RequestError, load_config, resolve_path
from .policy import BenchmarkPolicy
from .retrieval import SkillIndex, SkillRetEncoder, load_pool
from .review import HTTPReviewer, review_messages, validate_decision
from .common import ReviewError
LOG=logging.getLogger(__name__)


def load_index(config, *, rebuild=False):
    root = config['_root']
    skills, digest = load_pool(resolve_path(config['skill_pool_registry'], root), root, config['skill_scope'])
    encoder = SkillRetEncoder(config['embedding'], root)
    return SkillIndex.load_or_build(skills, digest, encoder,
        resolve_path(config['embedding']['index_path'], root), rebuild=rebuild)


class Router:
    """Reusable instance: index, encoder and policy are loaded once.

    skills contains stable registry IDs, preserving distinct same-name implementations.
    A valid first decision uses one review. Invalid decisions get at most three
    further attempts; transport/API failures are already handled by the gateway.
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
        messages = review_messages(request, candidates, self.retrieval)
        for attempt in range(4):
            # Do not retry HTTP/auth/transport errors here: that would multiply
            # the model gateway's independent three-retry transport budget.
            result = self.reviewer.review(messages)
            try:
                decision = validate_decision(result, candidates, self.retrieval['max_selected_skills'])
                break
            except ReviewError as error:
                if attempt == 3:raise
                LOG.warning(json.dumps({'level':'WARN','event':'router.decision_retry',
                                        'attempt':attempt+1,'reason':str(error)}))
                # Server-owned correction only; do not echo a hallucinated or
                # malicious model reply back into privileged instructions.
                messages = [*messages[:2], {'role':'user','content':
                    'Your previous decision failed validation: '+str(error)+'. '
                    'Return exactly the five specified fields. selected_skill_ids must contain only '
                    'exact candidate id values and at most '+str(self.retrieval['max_selected_skills'])+
                    ' item(s). Use [] when no candidate applies. Do not invent, shorten or rename IDs.'}]
        selected = self.policy.select(decision)
        return {'skills': list(decision.selected_skill_ids), 'harness': selected['harness'], 'model': selected['model']}
