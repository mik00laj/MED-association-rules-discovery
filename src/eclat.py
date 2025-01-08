from collections import defaultdict
from typing import Iterator, List, Tuple, Set
from src.dataclasses import Item, Rule
from src.apriori_utils import AprioriUtils


class Eclat:
    """
    Eclat algorithm implementation.
    """
    def __init__(self, min_support: float, min_confidence: float):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def _get_tid_sets(self, transactions: List[Set]) -> dict:
        """
        Create TID-sets for each individual item.
        """
        tid_sets = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                tid_sets[frozenset([item])].add(tid)

        return tid_sets

    def _eclat_recursive(self, prefix: frozenset, items: List[Tuple[frozenset, Set[int]]], freq_sets: List[Item]):
        """
        Recursive function to find frequent itemsets.
        """
        while items:
            item, tids = items.pop()
            new_prefix = prefix.union(item)
            support = len(tids) / self.num_transactions

            if support >= self.min_support:
                freq_sets.append(Item(item=new_prefix, support=support))
                new_items = []
                for other_item, other_tids in items:
                    intersect_tids = tids & other_tids
                    if len(intersect_tids) / self.num_transactions >= self.min_support:
                        new_items.append((other_item, intersect_tids))
                self._eclat_recursive(new_prefix, new_items, freq_sets)

    def run(self, data_iter: Iterator):
        """
        Run the Eclat algorithm.
        """
        transactions = [frozenset(record) for record in data_iter]
        self.num_transactions = len(transactions)

        tid_sets = self._get_tid_sets(transactions)
        freq_sets = []

        self._eclat_recursive(frozenset(), list(tid_sets.items()), freq_sets)

        # Generate rules
        rules_output = []
        for item in freq_sets:
            subsets = [frozenset(x) for x in AprioriUtils.get_subsets(item.item) if x]
            for subset in subsets:
                remain = item.item - subset
                if remain:
                    subset_support = sum(1 for t in transactions if subset.issubset(t)) / self.num_transactions
                    confidence = item.support / subset_support if subset_support > 0 else 0

                    # print(f"Subset: {subset}, Remain: {remain}, Support: {item.support}, Confidence: {confidence}")

                    if confidence >= self.min_confidence:
                        rules_output.append(Rule(pre=subset, post=remain, confidence=confidence))

        return freq_sets, rules_output
