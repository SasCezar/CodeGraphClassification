import numpy as np
from scipy.spatial.distance import jensenshannon

from ensemble_annotations.ensemble import EnsembleAnnotation


class JSDEnsemble(EnsembleAnnotation):
    def annotate(self, annotations: dict):
        n = 0
        best = -1
        max_jsd = 0
        for i, lf in enumerate(annotations):
            n = len(annotations[lf]['distribution'])
            if annotations[lf]['unannotated']:
                jsd_score = jensenshannon(annotations[lf]['distribution'], np.ones(n) / n)
                if jsd_score > max_jsd:
                    max_jsd = jsd_score
                    best = i

        if best != -1:
            return annotations[best]['distribution'], 0
        else:
            return np.zeros(n), 1
