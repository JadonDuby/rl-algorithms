from abc import ABC, abstractmethod

class BaseSchedule(ABC):
    @abstractmethod
    def get_epsilon(self, step: int) -> float:
        """
        Returns the epsilon value at a given step or episode.
        """
        pass

    @abstractmethod
    def update_epsilon(self, step: int) -> float:
        """
        update epsilon value at a given step or episode.
        """
        pass
