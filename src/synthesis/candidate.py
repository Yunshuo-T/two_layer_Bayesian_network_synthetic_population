from collections import Counter


class CandidatePool:
    def __init__(self, candidate_pool, age_attr) -> None:
        self._age_attr = age_attr
        self._pool = self._sort_age(candidate_pool)

    def take_candidate(self, evidence_key, required_group, previous_age):
        candidates = self._pool.get(evidence_key, [])
        for i, candidate in enumerate(candidates):
            person_age = int(candidate[self._age_attr])
            if self._matches(required_group, person_age, previous_age):
                return candidates.pop(i)
        return None

    def roll_back(self, used_candidates):
        for key, candidate, h_id in used_candidates:
            candidate.pop(h_id, None)
            self._pool[key].append(candidate)

    def _sort_age(self, candidate_pool):
        return {
            k: sorted(v, key=lambda s: int(s[self._age_attr]), reverse=True)
            for k, v in candidate_pool.items()
        }

    @staticmethod
    def _matches(required_group: str, age: int, previous_age: int | None) -> bool:
        if (
            previous_age is not None and age > previous_age
        ):  # Its age should not older than the age of previous rank
            return False
        if required_group == "adult":
            return age > 1
        if required_group == "minor":
            return age == 1
        return age == 0

    def _age_distribution(self, evidence_key):
        candidates = self._pool.get(evidence_key, [])
        return dict(Counter(int(c[self._age_attr]) for c in candidates))
