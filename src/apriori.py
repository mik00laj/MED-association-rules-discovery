from src.apriori_utils import AprioriUtils
from src.dataclasses import Rule, Item
from collections import defaultdict
from typing import Iterator


class Apriori:
    """
    Apriori algorithm implementation.
    """
    def __init__(self, min_support: float, min_confidence: float):
        self._min_support = min_support
        self._min_confidence = min_confidence

    # =========== Public methods =========== #

    def run(self, data_iter: Iterator):
        """
        Run the Apriori algorithm.
        """
        item_set, transaction_list = \
            AprioriUtils.get_item_set_transaction_list(
                data_iter
            )

        freq_set = defaultdict(int)
        total_set = dict()

        current_L_set = AprioriUtils.get_items_with_min_support(
            item_set, transaction_list, self._min_support, freq_set
        )
        k = 2
        while not len(current_L_set) == 0:
            total_set[k - 1] = current_L_set
            current_L_set = AprioriUtils.join_set(current_L_set, k)
            current_L_set = AprioriUtils.get_items_with_min_support(
                current_L_set, transaction_list, self._min_support, freq_set
            )
            k = k + 1

        # Generate the rules
        items_output = []
        for value in total_set.values():
            items_output.extend(
                [
                    Item(
                        item=(item),
                        support=freq_set[item] / len(transaction_list)
                    ) for item in value
                ]
            )

        rules_output = []
        for value in list(total_set.values())[1:]:
            for item in value:
                _subsets = map(
                    frozenset,
                    [x for x in AprioriUtils.get_subsets(item)]
                )
                for elem in _subsets:
                    remain = item.difference(elem)
                    if len(remain) > 0:
                        item_support = freq_set[item] / len(transaction_list)
                        elem_support = freq_set[elem] / len(transaction_list)
                        confidence = item_support / elem_support
                        if confidence >= self._min_confidence:
                            rules_output.append(
                                Rule(
                                    pre=elem,
                                    post=remain,
                                    confidence=confidence
                                )
                            )

        return items_output, rules_output


def get_data_from_file(fname: str) -> Iterator[str]:
    """
    Function to read the input file and return the data.
    """
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record
