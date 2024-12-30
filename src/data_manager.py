"""
This module is responsible for managing the data of the application.
"""
import pandas as pd
from typing import Generator, Iterator, Tuple
from ucimlrepo import fetch_ucirepo


class DataManager:
    _DATABASES = {
        "car_evaluation": 19,
        "tic_tac_toe_endgame": 101,
        "nursery": 76,
    }

    def __init__(self):
        self._cached_datasets = {}

    # =========== Static methods =========== #

    @staticmethod
    def get_data_from_file(fname: str) -> Iterator[str]:
        """
        Function to read the input file and return the data.
        """
        with open(fname, "r") as file_iter:
            for line in file_iter:
                line = line.strip().rstrip(",")  # Remove trailing comma
                record = frozenset(line.split(","))
                yield record

    @staticmethod
    def combine_data(
            data_x: pd.DataFrame,
            data_y: pd.Series
            ) -> Generator[Tuple[str], None, None]:
        """
        Combine the features and target data.
        """
        data_x['target'] = data_y
        tuples_list = [
            tuple(
                f"{col}_{val}" for col, val in zip(data_x.columns, row)
            ) for row in data_x.values
        ]

        yield from tuples_list

    # =========== Public methods =========== #

    def fetch_data_from_UCI(self, dataset: str):
        """
        Fetch the dataset from the UCI repository.
        """
        if dataset not in self._DATABASES.keys():
            raise ValueError(f"Dataset {dataset} not found.")

        if self._cached_datasets.get(dataset) is None:
            self._cached_datasets[dataset] = fetch_ucirepo(
                id=self._DATABASES[dataset]
            )

        return (
            self._cached_datasets[dataset].data.features,
            self._cached_datasets[dataset].data.targets
        )
