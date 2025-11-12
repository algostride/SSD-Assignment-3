"""
Implementation of the Quick Sort algorithm inheriting from BaseSort.
"""

from .base_sort import BaseSort


class QuickSort(BaseSort):
    """Quick Sort algorithm implementation supporting ascending and descending order."""

    def sort(self, ascending: bool, data_size: int, data: list[int]) -> list[int]:
        """Public interface for sorting data using quick sort."""
        self._validate_input(data)
        return self._quick_sort(data.copy(), ascending)

    def _quick_sort(self, arr: list[int], ascending: bool) -> list[int]:
        """Recursively partition and sort the array using the quick sort algorithm."""
        if len(arr) <= 1:
            return arr

        pivot = arr[len(arr) // 2]

        if ascending:
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
        else:
            left = [x for x in arr if x > pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x < pivot]

        return (
            self._quick_sort(left, ascending)
            + middle
            + self._quick_sort(right, ascending)
        )
