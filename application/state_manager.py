from abc import ABC, abstractmethod
from typing import Sequence, Any

class StateManager(ABC):
    """Abstract interface for a simulation‐state manager."""

    @abstractmethod
    def get_tags(self) -> Any:
        """Return an array of particle/object tags."""
        pass

    @abstractmethod
    def get_positions(self) -> Any:
        """Return an array of current positions."""
        pass

    @abstractmethod
    def get_velocities(self) -> Any:
        """Return an array of current velocities."""
        pass

    @abstractmethod
    def cases_names(self) -> Sequence[str]:
        """Return the list of available case names."""
        pass

    @abstractmethod
    def solvers_names(self) -> Sequence[str]:
        """Return the list of available solver names."""
        pass

    @abstractmethod
    def select_case(self, case_name: str, state=None) -> None:
        """Switch to a different simulation case."""
        pass

    @abstractmethod
    def select_solver(self, solver_name: str) -> None:
        """Switch to a different solver backend."""
        pass

    @abstractmethod
    def reset_scene(self) -> None:
        """Reset the simulation to its initial state."""
        pass

    @abstractmethod
    def advance(self) -> None:
        """Advance the simulation by one time‐step."""
        pass

    @abstractmethod
    def get_timestamp(self) -> float:
        """Return the current simulation time (or step count)."""
        pass

    @abstractmethod
    def get_current_save_directory(self) -> str:
        """Serialize and save the current state to disk."""
        pass