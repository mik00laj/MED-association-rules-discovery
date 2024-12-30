import itertools
import timeit
from prettytable import PrettyTable
from apyori import apriori, RelationRecord
from src import Apriori, DataManager, Rule
from itertools import tee
from typing import List, Tuple

TIMEIT_NUMBER = 1000000
DATASETS = [
    "car_evaluation",
    "tic_tac_toe_endgame",
    "nursery"
]
SUPPORTS = [0.15, 0.50, 0.8]
CONFIDENCES = [0.15, 0.50, 0.8]


def map_apyori_results(results: List[RelationRecord]) -> List[Rule]:
    """
    Map the results from the apyori library to the custom Rule class.
    """
    return [
        Rule(
            pre=ordered_statistic.items_base,
            post=ordered_statistic.items_add,
            confidence=ordered_statistic.confidence
        )
        for rule in results
        for ordered_statistic in rule.ordered_statistics
        if len(ordered_statistic.items_base) > 0
    ]


def compare_results(
        my_rules: List[Rule], apyori_rules: List[Rule]
        ) -> Tuple[float, float]:
    """
    Compare the results from the custom Apriori algorithm with
    the results from the apyori library.

    :param my_rules: List of rules from the custom Apriori algorithm.\n
    :param apyori_rules: List of rules from the apyori library.\n

    :return: Tuple with the percentage of rules matched and the percentage of
             confidences matched.
    """
    rules_matched = 0
    confidenced_matched = 0

    # === Compare the rules ===
    for apyori_rule in apyori_rules:
        for my_rule in my_rules:
            if my_rule.pre == apyori_rule.pre \
                    and my_rule.post == apyori_rule.post:
                rules_matched += 1
                if my_rule.confidence == apyori_rule.confidence:
                    confidenced_matched += 1
                break

    rules_matched = \
        (
            rules_matched / len(my_rules)
            if len(my_rules) > 0 else 1
        )
    confidenced_matched = \
        (
            confidenced_matched / len(my_rules)
            if len(my_rules) > 0 else 1
        )

    return rules_matched, confidenced_matched


def format_performance_stats(performance_stats: dict) -> str:
    """
    Format the performance statistics to a string in a table format.
    """
    table = PrettyTable()
    table.field_names = [
        "Dataset", "Min Support", "Min Confidence",
        "Rules Match", "Confidence Match",
        "My Exec Time", "Apyori Exec Time"
    ]

    for (dataset, min_support, min_confidence), metrics \
            in performance_stats.items():
        table.add_row([
            dataset, min_support, min_confidence,
            metrics['rules_match'], metrics['confidence_match'],
            metrics['my_exec_time'], metrics['apyori_exec_time']
        ])

    return str(table)


def main():
    data_manager = DataManager()
    performance_stats = {}

    for dataset, support, confidence in itertools.product(
                DATASETS, SUPPORTS, CONFIDENCES
            ):
        print(f"Running for: d: {dataset}, s: {support}, c: {confidence}...")

        # === Prepare the input data ===
        data_x, data_y = data_manager.fetch_data_from_UCI(dataset)
        input = DataManager.combine_data(data_x, data_y)
        my_apriori_input, apyori_input = tee(input)

        # === Run the Custom Apriori algorithm ===
        def run_my_apriori():
            my_apriori = Apriori(
                min_support=support,
                min_confidence=confidence,
            )
            _, my_rules = my_apriori.run(my_apriori_input)
            return my_rules

        my_exec_time = timeit.timeit(
            run_my_apriori,
            number=TIMEIT_NUMBER
        )
        my_rules = run_my_apriori()

        # === Run the Apriori algorithm from the apyori library ===
        def run_apyori():
            return apriori(
                apyori_input,
                min_support=support,
                min_confidence=confidence
            )

        apyori_exec_time = timeit.timeit(
            run_apyori,
            number=TIMEIT_NUMBER
        )
        raw_apyori_rules = run_apyori()

        # === Compare the results ===
        rules_match, confidence_match = compare_results(
            my_rules=my_rules,
            apyori_rules=map_apyori_results(
                list(raw_apyori_rules)
            )
        )

        performance_stats[(dataset, support, confidence)] = {
            "rules_match": f"{rules_match:.2%}",
            "confidence_match": f"{confidence_match:.2%}",
            "my_exec_time": f"{my_exec_time:.3f} ms",
            "apyori_exec_time": f"{apyori_exec_time:.3f} ms",
        }

    print(format_performance_stats(performance_stats))


if __name__ == "__main__":
    main()
