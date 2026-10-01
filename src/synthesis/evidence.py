from typing import Any, NamedTuple


class Evidence(NamedTuple):
    h_type: Any
    rank: int
    head_age: Any
    head_gender: Any
    p_age: int | None
    p_gender: Any

    @classmethod
    def make_evidence_key(
        cls,
        h_type: Any,
        rank: int,
        head_age: Any,
        head_gender: Any,
        required_group: str,
        head_age_to_person_age: dict,
        head_gender_to_person_gender: dict,
    ):
        """Construct an EvidenceKey for simulating individuals"""
        p_age = None
        p_gender = None

        if rank == 0:
            p_age = head_age_to_person_age[head_age]
            p_gender = head_gender_to_person_gender[head_gender]

        elif required_group == "minor":
            p_age = 1

        elif required_group == "kid":
            p_age = 0

        return cls(
            h_type,
            rank,
            head_age,
            head_gender,
            p_age,
            p_gender,
        )
