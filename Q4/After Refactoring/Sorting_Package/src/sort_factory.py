"""
Factory that provides sorting algorithm implementations and a unified `sort` API.
"""
from __future__ import annotations

from .bubble_sort import BubbleSort
from .selection_sort import SelectionSort
from .quick_sort import QuickSort
from .merge_sort import MergeSort
from .shell_sort import ShellSort


class SortFactory:
    """Map string names to sorting algorithm instances and expose a simple API."""

    def __init__(self) -> None:
        self.algorithms = {
            "bubble": BubbleSort(),
            "selection": SelectionSort(),
            "quick": QuickSort(),
            "merge": MergeSort(),
            "shell": ShellSort(),
        }

    def list_algorithms(self) -> list[str]:
        """Return a list of available algorithm names."""
        return list(self.algorithms.keys())

    def sort(
        self,
        ascending: bool,
        algorithm: str,
        data_size: int,
        data: list[int],
    ) -> list[int]:
        """Run the chosen algorithm on `data` and return the sorted list.

        Raises:
            ValueError: if `algorithm` is not a known algorithm name.
        """
        if algorithm not in self.algorithms:
            raise ValueError(
                "Unknown algorithm %r. Choose from: %s"
                % (algorithm, ", ".join(self.list_algorithms()))
            )

        return self.algorithms[algorithm].sort(ascending, data_size, data)
