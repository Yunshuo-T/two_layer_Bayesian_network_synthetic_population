from typing import Any, NamedTuple


class PersonSimulationKey(NamedTuple):
    """Conditions defining a person-simulation evidence group."""

    h_type: Any
    rank: int
    head_age: Any
    head_gender: Any
    p_age_states: tuple[int, ...]
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
        age_states_by_group: dict[str, list[int] | tuple[int, ...]],
    ) -> "PersonSimulationKey":
        """Keep the head's exact age, or allow every state in a member group."""
        if rank == 0:
            p_age_states = (head_age_to_person_age[head_age],)
            p_gender = head_gender_to_person_gender[head_gender]
        else:
            p_age_states = tuple(sorted(set(age_states_by_group[required_group])))
            p_gender = None

        if not p_age_states:
            raise ValueError(f"No supported age states for group {required_group!r}.")

        return cls(
            h_type,
            rank,
            head_age,
            head_gender,
            p_age_states,
            p_gender,
        )
