from .base_sort import BaseSort

class SelectionSort(BaseSort):
    def sort(self, ascending: bool, dataSize: int, data: list[int]) -> list[int]:
        self._validate_input(data)
        arr = data.copy()
        for i in range(dataSize):
            idx = i
            for j in range(i + 1, dataSize):
                if (ascending and arr[j] < arr[idx]) or (not ascending and arr[j] > arr[idx]):
                    idx = j
            arr[i], arr[idx] = arr[idx], arr[i]
        return arr