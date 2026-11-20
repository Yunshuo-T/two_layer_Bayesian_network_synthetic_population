from collections import Counter
from typing import Any
from synpop.threshold.threshold import lookup

class CandidatePool:
    """A dictionary of individuals synthesized by Person Bayesian model. {set(combination of household values): pd.Dataframe(Individuals)}"""

    def __init__(self, candidate_pool, age_attr) -> None:
        self._age_attr = age_attr
        self._pool = self._sort_age(candidate_pool)

    def take_candidate(
        self,
        h_dict,
        threshold:float,
        total_population,
        evidence_key,
        required_group,
        previous_age,
        probabilities,
        attributes,
        age_states_by_group,  
        p_data_include_kid=True
    ) -> dict[str, Any] | None:
        """
        Take one individual candidate from the pool, according to the evidence key requirement.
        """
        candidates = self._pool.get(evidence_key, [])
        for i, candidate in enumerate(candidates):
            person_age = int(candidate[self._age_attr])
            if self._matches(
                required_group=required_group,
                age=person_age,
                previous_age=previous_age,
                age_states_by_group=age_states_by_group,
                p_data_include_kid=p_data_include_kid,
            ):
                prob = lookup(probabilities, attributes, h_dict, candidate)
                if prob < threshold * (1/total_population):
                    continue
                return candidates.pop(i)
        return None

    def roll_back(self, used_candidates) -> None:
        """
        Roll back the taken candidates to the pool if the household does not have full household members.
        """
        for key, candidate, h_id in used_candidates:
            candidate.pop(h_id, None)
            self._pool[key].append(candidate)

    def _sort_age(self, candidate_pool) -> dict[Any, list[dict[str, Any]]]:
        return {
            k: sorted(v, key=lambda s: int(s[self._age_attr]), reverse=True)
            for k, v in candidate_pool.items()
        }

    @staticmethod
    def _matches(
        required_group: str,
        age: int,
        previous_age: int | None,
        age_states_by_group,
        p_data_include_kid: bool = True,
    ) -> bool:
        # Members are assigned in descending age order.
        if previous_age is not None and age > previous_age:
            return False

        if required_group == "kid" and not p_data_include_kid:
            return False

        # Accept any age state belonging to the required group.
        return age in age_states_by_group[required_group]

    def _age_distribution(self, evidence_key) -> dict[int, int]:
        candidates = self._pool.get(evidence_key, [])
        return dict(Counter(int(c[self._age_attr]) for c in candidates))
