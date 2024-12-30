
from collections import defaultdict
from itertools import chain, combinations
from typing import Iterator, List, Tuple

from src.dataclasses import RuleMetrics


class AprioriUtils:
    """
    Utility class for Apriori algorithm.
    """

    @staticmethod
    def get_subsets(arr: List) -> chain:
        """
        Returns non empty subsets of arr.
        """
        return chain(
            *[combinations(arr, i + 1) for i, _ in enumerate(arr)]
        )

    @staticmethod
    def get_items_with_min_support(
            item_set: set,
            transaction_list: List,
            min_support: float,
            freq_set: set,
            ) -> set:
        """
        Calculate the support of itemsets in the itemSet and return a subset.
        """
        output = set()
        item_count_dict = defaultdict(int)

        for item in item_set:
            for transaction in transaction_list:
                if item.issubset(transaction):
                    freq_set[item] += 1
                    item_count_dict[item] += 1

        for item, count in item_count_dict.items():
            support = count / len(transaction_list)

            if support >= min_support:
                output.add(item)

        return output

    @staticmethod
    def join_set(item_set: set, length: int) -> set:
        """
        Join a set with itself and returns the n-element itemsets.
        """
        return set(
            [
                i.union(j) for i in item_set for j in item_set
                if len(i.union(j)) == length
            ]
        )

    @staticmethod
    def get_item_set_transaction_list(
            data_iterator: Iterator
            ) -> Tuple[set, List]:
        """
        Returns a list of transactions and a set of items.
        """
        transaction_list = list()
        item_set = set()
        for record in data_iterator:
            transaction = frozenset(record)
            transaction_list.append(transaction)
            for item in transaction:
                item_set.add(frozenset([item]))
        return item_set, transaction_list

    @staticmethod
    def print_results(items: List, rules: List) -> None:
        """
        Prints the generated itemsets sorted by support and the
        confidence rules sorted by confidence.
        """
        print(f"\n {'-' * 25}< Supports >{'-' * 25}")
        for item in sorted(items):
            print(item)
        print(f"\n {'-' * 25}< Rules >{'-' * 25}")
        for rule in sorted(rules):
            print(rule)

    @staticmethod
    def print_metrics(rules_metrics: List[RuleMetrics]) -> None:
        """
        Prints the metrics for the rules.
        """
        print(f"\n {'-' * 25}< Metrics >{'-' * 25}")
        for rule_metric in rules_metrics:
            print(rule_metric)
