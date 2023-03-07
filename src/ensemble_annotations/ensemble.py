from abc import ABC, abstractmethod


class EnsembleAnnotation(ABC):
    @abstractmethod
    def annotate(self, annotations: dict):
        pass
