"""
Defines and operates on tasks in the style of mt-metrics-eval (https://github.com/google-research/mt-metrics-eval).
"""
import collections
import dataclasses
from typing import List, Literal, Dict, Optional, Iterable, Tuple, Iterator

from mt_metrics_eval.tasks import TaskResults, CORRELATION_FUNCTIONS, TaskSetResults
from mt_metrics_eval.data import CompareMetrics
from mt_metrics_eval import stats

from difficulty_sampling.data import Data, SrcData


wmt24_domains = {"news", "social", "literary", "speech"}


def correlation_for_difficulty(
    gold_scores: Dict[str, List[Optional[float]]],
    scores_to_evaluate: Dict[str, List[Optional[float]]],
    sys_names: Iterable[str] = None,
):
    """Get correlation statistics for given metric scores.

    Args:
      gold_scores: Gold scores to use. Same format as `scores`.
      scores_to_evaluate: Output scores to evaluate, a map from system names to lists of float scores.
      sys_names: Names of systems to use in comparison, must exist in both metric_scores and gold_scores.

    Returns:
      A stats.Correlation object for computing correlation statistics.
    """
    if sys_names is None:
        sys_names = gold_scores
    sys_names = set(sys_names)
    if not sys_names.issubset(scores_to_evaluate):
        raise ValueError(
            f"Missing metric scores: {sys_names - set(scores_to_evaluate)}"
        )
    if not sys_names.issubset(gold_scores):
        raise ValueError(f"Missing gold scores: {sys_names - set(gold_scores)}")

    all_gold_scores, all_scores_to_evaluate = [], []
    for sys_name in sys_names:
        gscores, scores = gold_scores[sys_name], scores_to_evaluate[sys_name]
        if len(gscores) != len(scores):
            raise ValueError(
                "Wrong number of scores for system %s: %d vs %d"
                % (sys_name, len(gscores), len(scores))
            )
        all_gold_scores.extend(gscores)
        all_scores_to_evaluate.extend(scores)
    return stats.Correlation(len(sys_names), all_gold_scores, all_scores_to_evaluate)


def get_correlations_for_difficulty(
    src_data_list: List[SrcData],
    scorer_names: List[str],
    gold_name: str = "human",
    domain: str = None,
    scorer_names_mapping: Dict[str, str] = None,
) -> Dict[str, stats.Correlation]:
    """
    Convenience function to generate stats for given parameters.

    Args:
        src_data_list: List of SrcData objects containing the outputs of the scorers to be evaluated.
        scorer_names: List of scorer names to be evaluated.
        gold_name: Name of the gold scorer. Default: "human".
        domain: If not None, it indicates that only the scores pertaining to that domain should be used.
        scorer_names_mapping: Mapping from scorer names in the input data to their desired names in the output.

    Returns:
         Map from metric names to stats.Correlation objects from which correlation and stat sign can be computed.
    """
    if domain is not None and domain not in wmt24_domains:
        raise ValueError(
            f"Invalid domain: {domain}. Allowed values are: {wmt24_domains}."
        )

    sys_names = {sys for src_data in src_data_list for sys in src_data["scores"]}

    def get_scores_from_src_data_list(
        scorer_name: str,
    ) -> Dict[str, List[Optional[float]]]:
        """
        Get scores for a specific scorer from the source data list.

        Args:
            scorer_name: Name of the scorer to extract scores for.

        Returns:
            Dictionary containing the segment-level scores for the specified scorer.
        """
        sys2scores = collections.defaultdict(list)
        for src_data in src_data_list:
            if domain is not None and src_data["domain"] != domain:
                continue

            for sys in sys_names:
                sys_scores_dict = src_data["scores"].get(sys)
                sys2scores[sys].append(
                    None
                    if sys_scores_dict is None
                    else sys_scores_dict.get(scorer_name)
                )

        return sys2scores

    # Get gold scores and filter outputs to those for which we have gold scores.
    gold_scores = get_scores_from_src_data_list(gold_name)

    # Gold_scores may contain systems that don't have any gold scores. Select just the subset of systems that does.
    gold_scores = {
        system: scores
        for system, scores in gold_scores.items()
        if scores and any(score is not None for score in scores)
    }

    sys_names = sys_names.intersection(gold_scores)

    # Generate 'Correlation' objects for all specified scorers.
    correlations = dict()  # scorer -> Correlation
    for scorer in scorer_names:
        curr_seg_scores = get_scores_from_src_data_list(scorer)
        display_name = (
            scorer_names_mapping[scorer] if scorer_names_mapping is not None else scorer
        )
        correlations[display_name] = correlation_for_difficulty(
            gold_scores, curr_seg_scores, sys_names
        )

    return correlations


@dataclasses.dataclass()
class DiffCorrTask:
    """Parameters for mt_metrics_eval.data.CompareMetrics."""

    lang: str = "en-es"
    domain: str | None = None
    corr_fcn: Literal["kendall", "pearson"] = "kendall"
    k: int = 1000
    gold: str = "human"
    pval: float = 0.05
    block_size: int = 100
    early_min: float = 0.02
    early_max: float = 0.50
    replace_nans_with_zeros: bool = False

    def __post_init__(self) -> None:
        """Check the validity of the `corr_fcn` attribute value."""
        if self.corr_fcn != "kendall" and self.corr_fcn != "pearson":
            raise ValueError(
                f"Invalid value for corr_fcn: {self.corr_fcn}. Allowed values are 'kendall' and 'pearson'."
            )

    @property
    def name(self) -> str:
        """Single string attr=value representation."""
        return " ".join(
            f"{a}={self.StrVal(a)}" for a in list(DiffCorrTask.__annotations__.keys())
        )

    def StrVal(self, attr: str) -> str:
        """
        Get the string representation of an attribute value.

        Args:
            attr: Attribute name to get the string representation for.

        Returns:
            String representation of the attribute value.
        """
        return f"{getattr(self, attr)}".replace(" ", "")

    def Run(
        self,
        data: Data,
        scorer_names: List[str],
        scorer_names_mapping: Dict[str, str] = None,
        parallel_file=None,
    ) -> TaskResults:
        """
        Generate metric correlations and pairwise significance results.

        Args:
            data: Data object containing the scorers to be evaluated.
            scorer_names: List of scorer names to be evaluated.
            scorer_names_mapping: Mapping from scorer names in the input data to their desired names in the output.
            parallel_file: File to use for statistical significance parallel processing. Default: None.

        Returns:
            TaskResults: Results of the task.
        """
        psd = stats.PermutationSigDiffParams(
            self.block_size, self.early_min, self.early_max
        )

        corr_fcn = CORRELATION_FUNCTIONS[self.corr_fcn]
        corrs = get_correlations_for_difficulty(
            data.lp2src_data_list[self.lang],
            scorer_names,
            self.gold,
            self.domain,
            scorer_names_mapping,
        )
        res = CompareMetrics(
            corrs,
            corr_fcn,
            "sys",
            self.k,
            psd,
            self.pval,
            self.replace_nans_with_zeros,
            "scores",
            parallel_file=parallel_file,
        )
        return TaskResults(self, res)


class DiffCorrTaskSet:
    """Convenience class to create and operate on sets of DiffCorr tasks."""

    def __init__(self):
        self.tasks = []
        self.data = dict()  # Lazily set by Run.

    def __len__(self) -> int:
        """Return the number of tasks in the set."""
        return len(self.tasks)

    def __add__(self, other):
        """
        Combine DiffCorr tasks sets. Any duplicate tasks will get repeated.

        Args:
            other: Another DiffCorrTaskSet to combine with this one.

        Returns:
            A new DiffCorrTaskSet containing all tasks from both sets.
        """
        res = DiffCorrTaskSet()
        res.tasks = self.tasks + other.tasks
        return res

    def __iter__(self) -> Iterator[DiffCorrTask]:
        """Return an iterator over the tasks in the set."""
        return iter(self.tasks)

    def Append(self, task: DiffCorrTask) -> None:
        """
        Append a DiffCorr task to the set.

        Args:
            task: DiffCorrTask to be added to the set.
        """
        self.tasks.append(task)

    def Run(
        self,
        data: Data,
        scorer_names: List[str],
        scorer_names_mapping: Dict[str, str] = None,
    ) -> TaskSetResults:
        """
        Run all tasks in the set.

        Args:
            data: Data object containing the scorers to be evaluated.
            scorer_names: List of scorer names to be evaluated.
            scorer_names_mapping: Mapping from scorer names in the input data to their desired names in the output.

        Returns:
            Results of the tasks in the set.
        """
        self.data = data
        return TaskSetResults(
            [
                task.Run(self.data, scorer_names, scorer_names_mapping)
                for task in self.tasks
            ]
        )


def diff_correlations_on_wmt24(
    lps: List[str], k=0
) -> Tuple[DiffCorrTaskSet, List[float]]:
    """
    Generate the DiffCorr tasks on the WMT24 test set, together with the associated weight vector.

    Args:
        lps: List of language pairs to be used for the tasks.
        k: Number of resampling runs to be used for statistical significance. Default: 0.

    Returns:
        A tuple containing the DiffCorrTaskSet and a list of weights for each task.
    """

    def add(language_pair: str, corr_fcn: Literal["kendall", "pearson"]) -> None:
        """
        Add a DiffCorr task to the set.

        Args:
            language_pair: Language pair to be used for the task.
            corr_fcn: Correlation function to be used for the task. Allowed values are 'kendall' and 'pearson'.
        """
        if corr_fcn != "kendall" and corr_fcn != "pearson":
            raise ValueError(
                f"Invalid value for corr_fcn: {corr_fcn}. Allowed values are 'kendall' and 'pearson'."
            )

        tasks.Append(DiffCorrTask(language_pair, corr_fcn=corr_fcn, k=k))

    lps = sorted(lps)

    tasks = DiffCorrTaskSet()

    # For each language pair, compute DiffCorr.
    for lp in lps:
        add(lp, "kendall")

    weights = [1] * len(tasks)
    weights = [w / sum(weights) for w in weights]

    return tasks, weights
