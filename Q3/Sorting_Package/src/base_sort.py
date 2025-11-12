from abc import ABC, abstractmethod

class BaseSort(ABC):
    @abstractmethod
    def sort(self, ascending: bool, dataSize: int, data: list[int]) -> list[int]:
        """Sorts the data and returns a new sorted list."""
        pass

    def _validate_input(self, data: list[int]):
        if not all(isinstance(x, int) for x in data):
            raise TypeError("All elements must be integers.")
        if len(data) > 2 * 1e5:
            raise ValueError("Input list exceeds 2 * 1e5 elements.")