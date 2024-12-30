from typing import Iterator, List

from src.apriori_utils import AprioriUtils
from src.dataclasses import Item, Rule, RuleMetrics


class Metrics:
    """
    Calculate the metrics for the Apriori algorithm.
    """
    def __init__(self, dataset: Iterator, items: List[Item]):
        self._init_dataset(dataset)
        self._supports = {item.item: item.support for item in items}

    def _init_dataset(self, dataset_iter: Iterator):
        """
        Initialize the dataset.
        """
        item_set, transaction_list = \
            AprioriUtils.get_item_set_transaction_list(
                dataset_iter
            )
        self._item_set = item_set
        self._transaction_list = transaction_list

    # =========== Private methods =========== #

    def _get_support(self, item: frozenset) -> float:
        """
        Get the support of an item.
        """
        if item not in self._supports:
            support = sum(1 for transaction in self._transaction_list if item.issubset(transaction))
            self._supports[item] = support / len(self._transaction_list)

        return self._supports[item]

    def _get_lift_factor(self, rule: Rule) -> float:
        """
        Get the lift factor of a rule.
        """
        return rule.confidence / self._get_support(rule.post)

    def _get_jaacard(self, rule: Rule) -> float:
        """
        Get the Jaaccard of a rule.
        """
        try:
            return rule.confidence / (self._get_support(rule.pre) + self._get_support(rule.post) - rule.confidence)
        except ZeroDivisionError:
            return 0.0

    def _get_odds_ratio(self, rule: Rule) -> float:
        """
        Get the odds ratio of a rule.
        """
        return rule.confidence / (self._get_support(rule.pre) * self._get_support(rule.post))

    # =========== Public methods =========== #

    def get_metrics(self, rules: List[Rule]) -> List[RuleMetrics]:
        """
        Get the metrics for the rules.
        """
        return [
            RuleMetrics(
                rule=rule,
                relative_support=self._get_support(rule.pre | rule.post),
                lift_factor=self._get_lift_factor(rule),
                certainty_factor=rule.confidence / self._get_support(rule.pre),
                jaaccard=self._get_jaacard(rule),
                odds_ratio=self._get_odds_ratio(rule)
            ) for rule in rules
        ]
