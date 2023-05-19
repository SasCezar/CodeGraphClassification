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

        if np.linalg.norm(distribution) == 0 or jensenshannon(distribution, uniform_vec) <= self.threshold:
            return 1

        return 0


class Threshold(Filtering):
    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, distribution):
        if np.linalg.norm(distribution) == 0 or np.max(distribution) <= self.threshold:
            return 0

        return 1


class Transformation(ABC):
    @abstractmethod
    def transform(self, distribution):
        pass


class SingleLabel(Transformation):
    def transform(self, distribution):
        argmax = np.argmax(distribution)
        distribution = np.zeros(len(distribution))
        distribution[argmax] = 1
        return distribution


class SoftLabel(Transformation):
    def __init__(self, top_k, min_threshold=0.05):
        self.top_k = top_k
        self.min_threshold = min_threshold

    def transform(self, distribution):
        sorted_distribution = np.argsort(distribution)[::-1]
        res_distribution = np.zeros(len(distribution))

        for i in sorted_distribution[:self.top_k]:
            res_distribution[i] = distribution[i] if distribution[i] > self.min_threshold else 0

        norm = np.linalg.norm(res_distribution)
        if norm != 0:
            res_distribution = res_distribution / norm
        return res_distribution
