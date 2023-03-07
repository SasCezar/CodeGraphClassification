import numpy as np

from ensemble_annotations.ensemble import EnsembleAnnotation


class CascadeEnsemble(EnsembleAnnotation):
    def annotate(self, annotations: dict):
        n = 0
        for lf in annotations:
            if not annotations[lf]['unannotated']:
                return annotations[lf]['distribution'], 0
            n = len(annotations[lf]['distribution'])

        return np.zeros(n), 1
