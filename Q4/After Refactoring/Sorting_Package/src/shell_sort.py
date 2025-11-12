"""
Implementation of the Shell Sort algorithm inheriting from BaseSort.
"""

from .base_sort import BaseSort


class ShellSort(BaseSort):
    """Shell Sort algorithm implementation supporting ascending and descending order."""

    def sort(self, ascending: bool, data_size: int, data: list[int]) -> list[int]:
        """Public interface for sorting data using shell sort."""
        self._validate_input(data)
        arr = data.copy()
        n = len(arr)
        gap = n // 2

        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                if ascending:
                    while j >= gap and arr[j - gap] > temp:
                        arr[j] = arr[j - gap]
                        j -= gap
                else:
                    while j >= gap and arr[j - gap] < temp:
                        arr[j] = arr[j - gap]
                        j -= gap
                arr[j] = temp
            gap //= 2

        return arr
