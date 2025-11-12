"""
Implementation of the Selection Sort algorithm inheriting from BaseSort.
"""

from .base_sort import BaseSort


class SelectionSort(BaseSort):
    """Selection Sort algorithm implementation supporting ascending and descending order."""

    def sort(self, ascending: bool, data_size: int, data: list[int]) -> list[int]:
        """Public interface for sorting data using selection sort."""
        self._validate_input(data)
        arr = data.copy()

        for i in range(data_size):
            idx = i
            for j in range(i + 1, data_size):
                if (ascending and arr[j] < arr[idx]) or (not ascending and arr[j] > arr[idx]):
                    idx = j
            arr[i], arr[idx] = arr[idx], arr[i]

        return arr
