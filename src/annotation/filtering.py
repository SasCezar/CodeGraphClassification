from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import jensenshannon


class Filtering(ABC):
    @abstractmethod
    def filter(self, distribution):
        pass


class JSDivergence(Filtering):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, distribution):
        n = len(distribution)
        uniform_vec = np.ones(n) / n

        if jensenshannon(distribution, uniform_vec) >= self.threshold:
            return 0

        return 1


class Threshold(Filtering):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, distribution):
        if np.max(distribution) >= self.threshold:
            return 1

        return 0


class Transformation(ABC):
    @abstractmethod
    def transform(self, distribution):
        pass


class SingleLabel(Transformation):
    def transform(self, distribution):
        return distribution


class SoftSingleLabel(Transformation):
    def transform(self, distribution):
        return distribution
