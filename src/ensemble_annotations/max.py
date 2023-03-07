import numpy as np

from ensemble_annotations.ensemble import EnsembleAnnotation


class MaxEnsemble(EnsembleAnnotation):
    def annotate(self, annotations: dict):
        distributions = []
        n = 0
        for lf in annotations:
            distributions.append(annotations[lf]['distribution']) if not annotations[lf]['unannotated'] else None
            n = len(annotations[lf]['distribution'])

        if distributions:
            return np.max(distributions, axis=0), 0
        else:
            return np.zeros(n), 1
