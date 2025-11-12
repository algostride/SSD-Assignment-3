"""
Abstract base class for all sorting algorithm implementations.
"""
from abc import ABC, abstractmethod

MAX_INPUT_SIZE = 2 * 10**5


class BaseSort(ABC):
    """Base class that defines the sorting interface and provides input validation."""

    @abstractmethod
    def sort(self, ascending: bool, data_size: int, data: list[int]) -> list[int]:
        """Sorts `data` and returns a new sorted list.

        Implementations must not modify the original list unless documented.
        """
        raise NotImplementedError

    def _validate_input(self, data: list[int]) -> None:
        """Validate that `data` is a list of integers and within allowed size."""
        if not isinstance(data, list):
            raise TypeError("Input must be a list of integers.")
        if not all(isinstance(x, int) for x in data):
            raise TypeError("All elements must be integers.")
        if len(data) > MAX_INPUT_SIZE:
            raise ValueError(f"Input list exceeds {MAX_INPUT_SIZE} elements.")
