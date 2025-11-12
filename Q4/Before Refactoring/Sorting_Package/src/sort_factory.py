from .bubble_sort import BubbleSort
from .selection_sort import SelectionSort
from .quick_sort import QuickSort
from .merge_sort import MergeSort

class SortFactory:
    def __init__(self):
        self.algorithms = {
            "bubble": BubbleSort(),
            "selection": SelectionSort(),
            "quick": QuickSort(),
            "merge": MergeSort(),
        }

    def sort(self, ascending: bool, algorithm: str, dataSize: int, data: list[int]) -> list[int]:
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(self.algorithms.keys())}")
        return self.algorithms[algorithm].sort(ascending, dataSize, data)