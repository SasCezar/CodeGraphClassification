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
        norm = np.sum(distribution)
        node_vec = np.zeros(n)

        if jensenshannon(distribution, uniform_vec) >= self.threshold and norm > 0:
            node_vec = distribution / norm

        return node_vec


class SingleLabel(Filtering):
    def filter(self, distribution):
        return distribution


class SoftSingleLabel(Filtering):
    def filter(self, distribution):
        return distribution
