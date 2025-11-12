from .base_sort import BaseSort

class QuickSort(BaseSort):
    def sort(self, ascending: bool, dataSize: int, data: list[int]) -> list[int]:
        self._validate_input(data)
        arr = data.copy()
        return self._quick_sort(arr, ascending)

    def _quick_sort(self, arr, ascending):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if (x < pivot if ascending else x > pivot)]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if (x > pivot if ascending else x < pivot)]
        return self._quick_sort(left, ascending) + middle + self._quick_sort(right, ascending)