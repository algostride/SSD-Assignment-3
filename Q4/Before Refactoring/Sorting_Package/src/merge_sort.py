from .base_sort import BaseSort

class MergeSort(BaseSort):
    def sort(self, ascending: bool, dataSize: int, data: list[int]) -> list[int]:
        self._validate_input(data)
        arr = data.copy()
        return self._merge_sort(arr, ascending)

    def _merge_sort(self, arr, ascending):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self._merge_sort(arr[:mid], ascending)
        right = self._merge_sort(arr[mid:], ascending)
        return self._merge(left, right, ascending)

    def _merge(self, left, right, ascending):
        result = []
        while left and right:
            if (ascending and left[0] <= right[0]) or (not ascending and left[0] >= right[0]):
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        result.extend(left or right)
        return result