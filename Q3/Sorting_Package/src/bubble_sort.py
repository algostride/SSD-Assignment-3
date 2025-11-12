from .base_sort import BaseSort

class BubbleSort(BaseSort):
    def sort(self, ascending: bool, dataSize: int, data: list[int]) -> list[int]:
        self._validate_input(data)
        arr = data.copy()
        for i in range(dataSize):
            for j in range(0, dataSize - i - 1):
                if (ascending and arr[j] > arr[j + 1]) or (not ascending and arr[j] < arr[j + 1]):
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr