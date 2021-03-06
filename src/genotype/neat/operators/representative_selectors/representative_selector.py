from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from src.genotype.neat.genome import Genome


class RepresentativeSelector(ABC):
    """Finds a representative for a species"""

    @abstractmethod
    def select_representative(self, genomes: Dict[int:Genome]) -> Genome:
        pass
