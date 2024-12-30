from prettytable import PrettyTable
from collections import defaultdict
from typing import Dict, List
from src import Apriori, DataManager, Metrics, RuleMetrics
from itertools import tee


DATASETS = [
    "car_evaluation",
    "tic_tac_toe_endgame",
    "nursery"
]
MIN_SUPPORT = 0.15
MIN_CONFIDENCE = 0.3


def format_grouped_metrics_average(
    grouped_metrics: Dict[str, List[RuleMetrics]]
        ) -> str:
    """
    Format the grouped metrics into a table with avg values
    for each bucket.
    """
    table = PrettyTable()
    table.field_names = [
        "Lift Range", "No. Rules", "Relative Support [Avg. / Std. dev.]",
        "Certainty [Avg. / Std. dev.]", "Jaccard [Avg. / Std. dev.]",
        "Odds Ratio [Avg. / Std. dev.]"
    ]

    for lift_range, metrics in sorted(
            grouped_metrics.items(),
            key=lambda x: x[0]
            ):
        avg_relative_support = sum(
            metric.relative_support for metric in metrics
        ) / len(metrics)
        std_dev_relative_support = sum(
            (metric.relative_support - avg_relative_support) ** 2
            for metric in metrics
        ) / len(metrics)

        avg_certainty = sum(
            metric.certainty_factor for metric in metrics
        ) / len(metrics)
        std_dev_certainty = sum(
            (metric.certainty_factor - avg_certainty) ** 2
            for metric in metrics
        ) / len(metrics)

        avg_jaccard = sum(
            metric.jaaccard for metric in metrics
        ) / len(metrics)
        std_dev_jacaaard = sum(
            (metric.jaaccard - avg_jaccard) ** 2
            for metric in metrics
        ) / len(metrics)

        avg_odds_ratio = sum(
            metric.odds_ratio for metric in metrics
        ) / len(metrics)
        std_dev_odd_ratio = sum(
            (metric.odds_ratio - avg_odds_ratio) ** 2
            for metric in metrics
        ) / len(metrics)

        table.add_row([
            lift_range,
            len(metrics),
            f"[{avg_relative_support:.3f} / {std_dev_relative_support:.3f}]",
            f"[{avg_certainty:.3f} / {std_dev_certainty:.3f}]",
            f"[{avg_jaccard:.3f} / {std_dev_jacaaard:.3f}]",
            f"[{avg_odds_ratio:.3f} / {std_dev_odd_ratio:.3f}]"
        ])

    return table.get_string()


def format_grouped_metrics_full(
        grouped_metrics: Dict[str, List[RuleMetrics]]
        ) -> str:
    """
    Format the grouped metrics into a table with full results.
    """
    table = PrettyTable()
    table.field_names = [
        "Lift Range", "Rule", "Relative Support",
        "Lift", "Certainty", "Jaccard", "Odds Ratio"
    ]

    for lift_range, metrics in sorted(
            grouped_metrics.items(),
            key=lambda x: x[0]
            ):
        for metric in metrics:
            table.add_row([
                lift_range,
                f"{metric.rule.raw_str()}",
                f"{metric.relative_support:.2f}",
                f"{metric.lift_factor:.2f}",
                f"{metric.certainty_factor:.2f}",
                f"{metric.jaaccard:.2f}",
                f"{metric.odds_ratio:.2f}"
            ])

    return table.get_string()


def group_by_lift(
        rules_metrics: List[RuleMetrics],
        bucket_size: float = 0.1) -> Dict[str, List[RuleMetrics]]:
    """
    Groups association rules into buckets by similar lift values.

    :param rules: List of AssociationRule instances.
    :param bucket_size: Size of each bucket for grouping lift values.
    :return: Dictionary with lift range as keys and list of rules as values.
    """
    buckets = defaultdict(list)

    for metric in rules_metrics:
        # Determine the bucket key based on the lift value
        bucket_low_val = float(
            metric.lift_factor // bucket_size * bucket_size
        )
        bucket_high_val = float(
            (metric.lift_factor // bucket_size + 1) * bucket_size
        )
        bucket_key = f"{bucket_low_val:.1f}-{bucket_high_val:.1f}"
        buckets[bucket_key].append(metric)

    return buckets


def main():
    data_manager = DataManager()
    apriori = Apriori(
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE
    )

    for dataset in DATASETS:
        # === Prepare the data ===
        data_x, data_y = data_manager.fetch_data_from_UCI(dataset)
        input = DataManager.combine_data(data_x, data_y)
        apriori_input, metrics_input = tee(input)

        # === Run the Apriori algorithm ===
        items, rules = apriori.run(apriori_input)
        metrics = Metrics(metrics_input, items)

        # === Get the metrics ===
        rules_metrics = metrics.get_metrics(rules)

        # === Group the metrics by simmilar lift factor value ===
        grouped_metrics = group_by_lift(rules_metrics)

        # === Print the results ===
        print(f"==============< Dataset: {dataset} >===============")
        print(format_grouped_metrics_full(grouped_metrics))
        print(format_grouped_metrics_average(grouped_metrics))


if __name__ == "__main__":
    main()
