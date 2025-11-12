"""
Implementation of the Merge Sort algorithm inheriting from BaseSort.
"""

from .base_sort import BaseSort


class MergeSort(BaseSort):
    """Merge Sort algorithm implementation supporting ascending and descending order."""

    def sort(self, ascending: bool, data_size: int, data: list[int]) -> list[int]:
        """Public interface for sorting data using merge sort."""
        self._validate_input(data)
        return self._merge_sort(data.copy(), ascending)

    def _merge_sort(self, arr: list[int], ascending: bool) -> list[int]:
        """Recursively split and merge the array."""
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self._merge_sort(arr[:mid], ascending)
        right = self._merge_sort(arr[mid:], ascending)
        return self._merge(left, right, ascending)

    def _merge(self, left: list[int], right: list[int], ascending: bool) -> list[int]:
        """Merge two sorted lists into one sorted list."""
        result = []
        i, j = 0, 0

        while i < len(left) and j < len(right):
            if (ascending and left[i] <= right[j]) or (not ascending and left[i] >= right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result
