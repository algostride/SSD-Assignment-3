"""
Implementation of the Bubble Sort algorithm inheriting from BaseSort.
"""

from .base_sort import BaseSort


class BubbleSort(BaseSort):
    """Bubble Sort algorithm implementation supporting ascending and descending order."""

    def sort(self, ascending: bool, data_size: int, data: list[int]) -> list[int]:
        """Public interface for sorting data using bubble sort."""
        self._validate_input(data)
        arr = data.copy()

        for i in range(data_size):
            for j in range(0, data_size - i - 1):
                if (ascending and arr[j] > arr[j + 1]) or (not ascending and arr[j] < arr[j + 1]):
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

        return arr
