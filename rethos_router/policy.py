"""Deterministic selection from frozen evaluation priorities, never LLM guesses."""
import logging
import math

from .common import ConfigurationError, HARNESSES, MODELS, TASK_TYPES, read_json

LOG = logging.getLogger(__name__)


def pair(value):
    if not isinstance(value, dict) or set(value) != {'harness', 'model'}:
        raise ConfigurationError('A fallback pair must contain harness and model')
    if (not isinstance(value['harness'], str) or not isinstance(value['model'], str) or
            value['harness'] not in HARNESSES or value['model'] not in MODELS):
        raise ConfigurationError('Fallback pair is outside the six supported configurations')
    return dict(value)


class BenchmarkPolicy:
    def __init__(self, registry, fallback):
        self.registry_id = registry.get('registry_id')
        self.default = pair(fallback['default'])
        self.provider_defaults = {provider: pair(value) for provider, value in fallback['by_model'].items()}
        if set(self.provider_defaults) != MODELS or any(value['model'] != provider for provider, value in self.provider_defaults.items()):
            raise ConfigurationError('Fallbacks must preserve each supported model constraint')
        self.agents = {}
        for agent in registry.get('agents', []):
            aid = agent.get('agent_id')
            if not isinstance(aid, str) or aid in self.agents:
                raise ConfigurationError('Agent registry IDs must be unique')
            self.agents[aid] = agent
        rules = registry.get('routing_policy', {}).get('rules', [])
        self.rules = {}
        for rule in rules:
            task = rule.get('task_type')
            if task not in TASK_TYPES or task in self.rules:
                raise ConfigurationError('Invalid or duplicate benchmark routing task type')
            self.rules[task] = rule
            referenced = list(rule.get('quality_first', [])) + list(rule.get('latency_first', []))
            for provider, ids in rule.get('provider_constrained_quality', {}).items():
                if provider not in MODELS:
                    raise ConfigurationError('Unsupported model in policy')
                referenced.extend(ids)
            for aid in referenced:
                agent = self.agents.get(aid)
                if (not agent or agent.get('study') != 'baseline' or agent.get('task_type') != task or agent.get('skills')):
                    raise ConfigurationError('Default policy must reference same-task no-skill baseline evidence')
                self._pair(agent)
            for provider, ids in rule.get('provider_constrained_quality', {}).items():
                if any(self._pair(self.agents[aid])['model'] != provider for aid in ids):
                    raise ConfigurationError('Provider-constrained policy references a different provider')
        if set(self.rules) != TASK_TYPES:
            raise ConfigurationError('Registry must contain the four supported task policies')

    @classmethod
    def load(cls, path, fallback):
        try:
            return cls(read_json(path), fallback)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ConfigurationError('Malformed benchmark registry or fallback configuration') from exc

    @staticmethod
    def _pair(agent):
        return pair({'harness': agent['harness']['id'], 'model': agent['model']['provider']})

    @staticmethod
    def _quality_available(agent):
        cohort = agent.get('quality', {}).get('common_cohort', {})
        value = cohort.get('value')
        return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 and
                cohort.get('unique_task_n', 0) > 0)

    @staticmethod
    def _latency(agent):
        cohort = agent.get('latency', {}).get('common_completed_cohort', {})
        value = cohort.get('value')
        return value if (type(value) in (int, float) and math.isfinite(value) and value > 0 and
                         cohort.get('unique_task_n', 0) > 0) else None

    def select(self, decision):
        task = decision.primary_task_type if decision.task_type == 'mixed' else decision.task_type
        rule = self.rules.get(task)
        constraint = decision.model_constraint
        if rule:
            ids = rule[decision.preference]
            if constraint and decision.preference == 'quality_first':
                ids = rule.get('provider_constrained_quality', {}).get(constraint, [])
            for aid in ids:
                agent = self.agents[aid]
                if not self._quality_available(agent) or (decision.preference == 'latency_first' and self._latency(agent) is None):
                    continue
                result = self._pair(agent)
                if constraint is None or result['model'] == constraint:
                    LOG.debug('Benchmark selection: task=%s policy=%s agent=%s', task, decision.preference, aid)
                    return result
            if constraint and decision.preference == 'latency_first':
                # The precomputed latency shortlist may only include another provider.
                # Compare same-task baseline common-cohort timing within the requested provider.
                timed = []
                for aid, agent in self.agents.items():
                    if (agent.get('study') != 'baseline' or agent.get('task_type') != task or agent.get('skills') or
                            agent['model']['provider'] != constraint):
                        continue
                    value = self._latency(agent)
                    if value is not None and self._quality_available(agent):
                        timed.append((value, aid, self._pair(agent)))
                if timed:
                    return min(timed, key=lambda item: (item[0], item[1]))[2]
        LOG.info('Using configured default pair: task=%s; historical coverage is insufficient', task)
        return dict(self.provider_defaults[constraint] if constraint else self.default)
