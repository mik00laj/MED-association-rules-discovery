import itertools
import timeit
from prettytable import PrettyTable
from src import Apriori, Eclat, DataManager, Rule, Metrics
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


def compare_results(apriori_rules: List[Rule], eclat_rules: List[Rule]) -> Tuple[float, float]:
    """
    Porównanie wyników algorytmów Apriori i Eclat.
    """
    rules_matched = 0
    confidenced_matched = 0

    for apriori_rule in apriori_rules:
        for eclat_rule in eclat_rules:
            if apriori_rule.pre == eclat_rule.pre and apriori_rule.post == eclat_rule.post:
                rules_matched += 1
                if apriori_rule.confidence == eclat_rule.confidence:
                    confidenced_matched += 1
                break

    rules_matched = (rules_matched / len(apriori_rules)) if len(apriori_rules) > 0 else 1
    confidenced_matched = (confidenced_matched / len(apriori_rules)) if len(apriori_rules) > 0 else 1

    return rules_matched, confidenced_matched


def format_performance_stats(performance_stats: dict) -> str:
    """
    Formatuje statystyki porównawcze jako tabelę.
    """
    table = PrettyTable()
    table.field_names = [
        "Dataset", "Min Support", "Min Confidence",
        "Rules Match", "Confidence Match",
        "Apriori Exec Time", "Eclat Exec Time"
    ]

    for (dataset, min_support, min_confidence), metrics in performance_stats.items():
        table.add_row([
            dataset, min_support, min_confidence,
            metrics['rules_match'], metrics['confidence_match'],
            metrics['apriori_exec_time'], metrics['eclat_exec_time']
        ])

    return str(table)


def main():
    data_manager = DataManager()
    performance_stats = {}

    for dataset, support, confidence in itertools.product(DATASETS, SUPPORTS, CONFIDENCES):
        print(f"Running for: Dataset: {dataset}, Support: {support}, Confidence: {confidence}...")

        # Przygotowanie danych
        data_x, data_y = data_manager.fetch_data_from_UCI(dataset)
        input_data = DataManager.combine_data(data_x, data_y)
        apriori_input, eclat_input = tee(input_data)

        # === Uruchomienie algorytmu Apriori ===
        def run_apriori():
            apriori = Apriori(min_support=support, min_confidence=confidence)
            _, apriori_rules = apriori.run(apriori_input)
            return apriori_rules

        apriori_exec_time = timeit.timeit(run_apriori, number=TIMEIT_NUMBER)
        apriori_rules = run_apriori()

        # === Uruchomienie algorytmu Eclat ===
        def run_eclat():
            eclat = Eclat(min_support=support, min_confidence=confidence)
            _, eclat_rules = eclat.run(eclat_input)
            return eclat_rules

        eclat_exec_time = timeit.timeit(run_eclat, number=TIMEIT_NUMBER)
        eclat_rules = run_eclat()

        # === Porównanie wyników ===
        rules_match, confidence_match = compare_results(
            apriori_rules=apriori_rules,
            eclat_rules=eclat_rules
        )

        performance_stats[(dataset, support, confidence)] = {
            "rules_match": f"{rules_match:.2%}",
            "confidence_match": f"{confidence_match:.2%}",
            "apriori_exec_time": f"{apriori_exec_time:.3f} s",
            "eclat_exec_time": f"{eclat_exec_time:.3f} s"
        }

    print(format_performance_stats(performance_stats))


if __name__ == "__main__":
    main()
