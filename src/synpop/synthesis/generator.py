from typing import Any
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd
from collections import Counter
from synpop.schema import ColumnNames,ModelSchema
from synpop.synthesis.evidence import PersonSimulationKey
from synpop.synthesis.candidate import CandidatePool
from concurrent.futures import ProcessPoolExecutor, as_completed
from synpop.utils import encoding,seeding
import logging

logger = logging.getLogger(__name__)

class Generator:
    def __init__(
        self,
        h_model: DiscreteBayesianNetwork,
        p_model: DiscreteBayesianNetwork,
        config: ModelSchema,
        p_data_include_kid: bool = True,
    ):
        """
        Initialize the Bayesian-network workflow.

        Args:
            config: Column-name configuration used by all methods.
        """
        self.h_model = h_model
        self.p_model = p_model
        self.config = config
        self.head_age_to_person_age = encoding.make_code_crosswalk(
            ColumnNames.HEAD_AGE,
            ColumnNames.AGE_CATE,
            self.config.code_to_attr,
            self.config.attr_to_code,
        )
        self.head_gender_to_person_gender = encoding.make_code_crosswalk(
            ColumnNames.HEAD_GENDER,
            self.config.gender_col,
            self.config.code_to_attr,
            self.config.attr_to_code,
        )
        self.p_data_include_kid = p_data_include_kid
        self.h_sampler = BayesianModelSampling(h_model)
        self.p_sampler = BayesianModelSampling(p_model)

    def synthesize(
        self,
        num_households: int,
        pool_multiplier: int = 50,
        max_retries: int = 10,
        random_seed: int | None = None,
        max_workers: int | None = None,
    ):
        h_blueprint = self.simulate_household(num_households, random_seed=random_seed)
        demands, valid_blueprint = self.generate_evidence_batches(h_blueprint)
        candidate_pool = self.simulate_people(
            demands, pool_multiplier, random_seed, max_workers
        )
        realizations, failed_h_ids = self.allocate_households(
            CandidatePool(candidate_pool, ColumnNames.AGE_CATE), valid_blueprint
        )
        remaining = valid_blueprint[
            valid_blueprint[self.config.h_id_col].isin(failed_h_ids)
        ]
        for _ in range(max_retries):
            if remaining.empty:
                break
            random_seed = seeding.random_seeds(random_seed)
            demands, remaining_bp = self.generate_evidence_batches(remaining)
            retry_pool = self.simulate_people(
                demands, pool_multiplier, random_seed, max_workers
            )
            retry_realizations, failed_h_ids = self.allocate_households(
                CandidatePool(retry_pool, ColumnNames.AGE_CATE), remaining_bp
            )
            realizations.extend(retry_realizations)
            remaining = remaining_bp[
                remaining_bp[self.config.h_id_col].isin(failed_h_ids)
            ]
        p_df = pd.DataFrame(realizations).reset_index(drop=True)
        h_df = valid_blueprint[
            ~valid_blueprint[self.config.h_id_col].isin(failed_h_ids)
        ].reset_index(drop=True)
        return h_df, p_df

    def simulate_household(
        self,
        num_samples: int,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Sample synthetic household records from a fitted household model.

        Args:
            model: Fitted household Bayesian network.
            num_samples: Number of households to generate.
            random_seed: Optional reproducibility seed.

        Returns:
            A data frame of sampled households with generated household IDs.
        """
        h_blueprint = self.h_sampler.forward_sample(
            size=num_samples, seed=random_seed, show_progress=False
        )
        h_blueprint[self.config.h_id_col] = h_blueprint.index

        return h_blueprint

    def generate_evidence_batches(self, h_blueprint: pd.DataFrame):
        """
        Aggregate person-sampling requirements from simulated household records.

        Args:
            h_blueprint: Sampled household records.

        Returns:
            A tuple of evidence-demand counts, rank-to-model-state mapping, and
            the maximum supported rank.
        """
        max_rank = max(self._rank_states(self.p_model))
        demands = Counter()

        # Remove household with h_size exceeding max_rank
        h_blueprint = self._check_blueprint(h_blueprint, max_rank)
        # count the unique combinations of household elementary attributes.
        unique_blueprints = (
            h_blueprint[list(self.config.elementary_attributes["H"])]
            .value_counts()
            .reset_index(name="demand_count")
        )

        for h in unique_blueprints.itertuples():
            h_dict = h._asdict()  # type: ignore
            h_type, head_age, head_gender, adults, minors, _, h_size = (
                self._get_values_from_h_blueprint(h_dict)
            )

            demand_count = int(h_dict["demand_count"])

            # For each rank in the household
            for rank in range(h_size):
                key = PersonSimulationKey.make_evidence_key(
                    h_type,
                    rank,
                    head_age,
                    head_gender,
                    self._required_group(rank, adults, minors),
                    self.head_age_to_person_age,
                    self.head_gender_to_person_gender,
                )

                demands[key] += demand_count

        return demands, h_blueprint

    def simulate_people(
        self,
        demands: dict,
        pool_multiplier: int,
        random_seed: int | None = None,
        max_workers: int | None = None,
    ) -> dict[tuple, list[dict[str, Any]]]:
        """
        Generate conditional pools of candidate persons for each evidence key.

        Args:
            p_model: Fitted person Bayesian network.
            demands: Required people per evidence combination.
            pool_multiplier: Candidate draws per required person.
            random_seed: Optional master seed.
            max_workers: Maximum process-worker count.

        Returns:
            Candidate person records grouped by evidence key.
        """

        candidate_pool = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _batch_sample_persons,
                    key,
                    count,
                    pool_multiplier,
                    self.p_model,
                    self.config,
                    seeding.random_seeds(random_seed),
                )
                for key, count in demands.items()
            ]
            for future in as_completed(futures):
                key, records = future.result()
                candidate_pool[key] = records

        return candidate_pool

    def allocate_households(
        self,
        candidate_pool: CandidatePool,
        h_blueprint: pd.DataFrame,
    ):
        """
        Allocate compatible sampled people to households.
        Candidate people are removed from pools after assigned.
        """
        realizations = []
        invalid_h_ids = set()
        # For each household in the household blueprint
        for h in h_blueprint.itertuples():
            h_dict = h._asdict()  # type: ignore
            adults, minors, _, h_size = self._count_member(h_dict)
            h_id = h_dict[self.config.h_id_col]

            temp_persons = []
            used_candidates = []
            valid_household = True
            previous_assigned_age = None  # Age of previously assigned member

            # For each rank in the household
            for rank in range(h_size):
                required_group = self._required_group(rank, adults, minors)
                evidence_key = PersonSimulationKey.make_evidence_key(
                    h_type=h_dict[ColumnNames.H_TYPE],
                    rank=rank,
                    head_age=h_dict[ColumnNames.HEAD_AGE],
                    head_gender=h_dict[ColumnNames.HEAD_GENDER],
                    required_group=required_group,
                    head_age_to_person_age=self.head_age_to_person_age,
                    head_gender_to_person_gender=self.head_gender_to_person_gender,
                )

                candidate = candidate_pool.take_candidate(
                    evidence_key, required_group, previous_assigned_age
                )
                if candidate is None:
                    invalid_h_ids.add(h_id)
                    valid_household = False
                    logger.warning(
                        "Allocation failed for household type %s at rank %d, required_group %s. Pool state: %s",
                        h_dict[ColumnNames.H_TYPE],
                        rank,
                        required_group,
                        candidate_pool._age_distribution(evidence_key),
                    )
                    break
                candidate[self.config.h_id_col] = h_id
                temp_persons.append(candidate)
                used_candidates.append((evidence_key, candidate, self.config.h_id_col))
                previous_assigned_age = int(candidate[ColumnNames.AGE_CATE])

            if valid_household:
                realizations.extend(temp_persons)
            else:
                # Restore all failed candidates to the pool.
                candidate_pool.roll_back(used_candidates)

        return realizations, invalid_h_ids

    def _get_values_from_h_blueprint(self, h_dict):
        h_type = h_dict[ColumnNames.H_TYPE]
        head_age = h_dict[ColumnNames.HEAD_AGE]
        head_gender = h_dict[ColumnNames.HEAD_GENDER]
        adults, minors, kids, h_size = self._count_member(h_dict)
        return (h_type, head_age, head_gender, adults, minors, kids, h_size)

    def _rank_states(self, p_model):
        return list(
            map(
                int,
                p_model.get_cpds(ColumnNames.MEMBER_RANK).state_names[
                    ColumnNames.MEMBER_RANK
                ],  # type: ignore
            )  # type ignore
        )

    def _count_member(self, h_dict):
        """
        Return the number of adults, minor, kids within the household
        and the household size.
        """
        adults = int(h_dict[ColumnNames.ADULTS])
        minors = int(h_dict[ColumnNames.MINORS])
        kids = int(h_dict[ColumnNames.KIDS])
        if not self.p_data_include_kid:
            h_size = adults + minors
        else:
            h_size = adults + minors + kids
        return adults, minors, kids, h_size

    def _check_blueprint(self, h_blueprint, max_rank) -> pd.DataFrame:
        """
        Check if the household size exceeds the maximum member rank from
        learned person Bayesian network.
        """
        max_size = max_rank + 1

        h_size = (
            h_blueprint[ColumnNames.ADULTS]
            + h_blueprint[ColumnNames.MINORS]
            + h_blueprint[ColumnNames.KIDS]
        )

        valid_mask = h_size <= max_size
        invalid_count = (~valid_mask).sum()
        if invalid_count:
            logger.warning(
                f"{invalid_count} households exceed "
                f"maximum supported size {max_size}."
            )

        return h_blueprint.loc[valid_mask]

    @staticmethod
    def _required_group(rank, adults, minors) -> str:
        """
        return the type (adults, minor, kids) of household member
        based on its rank.
        """
        if rank < adults:
            return "adult"
        if rank < adults + minors:
            return "minor"
        return "kid"


def _batch_sample_persons(key, count, pool_multiplier, p_model, config, random_seed):
    sampler = BayesianModelSampling(p_model)
    h_type, rank, head_age, head_gender, p_age, p_gender = key
    # Build evidence
    partial = {
        ColumnNames.H_TYPE: h_type,
        ColumnNames.MEMBER_RANK: rank,
        ColumnNames.HEAD_AGE: head_age,
        ColumnNames.HEAD_GENDER: head_gender,
    }
    batch_size = count * pool_multiplier

    if p_gender is not None:
        partial[config.gender_col] = p_gender

    if p_age is not None:
        partial[ColumnNames.AGE_CATE] = p_age
        # Create a batch

    evidence_df = pd.DataFrame([partial] * batch_size)
    # Sample
    batch_samples = sampler.forward_sample(
        size=batch_size,
        partial_samples=evidence_df,
        show_progress=False,
        seed=random_seed,
    )

    return key, batch_samples.to_dict("records")  # Return (key, list_of_records)
