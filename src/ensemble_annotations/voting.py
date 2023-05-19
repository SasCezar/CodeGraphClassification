import numpy as np

from ensemble_annotations.ensemble import EnsembleAnnotation


class VotingEnsemble(EnsembleAnnotation):
    def __int__(self, k=10):
        self.k = 10

    def annotate(self, annotations: dict):
        best, n = self.extract_best(annotations)

        if best:
            distributions = np.zeros(n)
            for lf_top in best:
                for i, p in enumerate(lf_top):
                    distributions[p] = distributions[p] + self.vote_weight(i)
            distributions = distributions / np.linalg.norm(distributions)
            return distributions, 0
        else:
            return np.zeros(n), 1

    def extract_best(self, annotations):
        best = []
        n = 0
        for lf in annotations:
            n = len(annotations[lf]['distribution'])
            if not annotations[lf]['unannotated']:
                top = np.argsort(annotations[lf]['distribution'])[::-1][:self.k]
                best.append(top)
        return best, n

    def vote_weight(self, i):
        return self.k - i


class ExpVotingEnsemble(VotingEnsemble):
    def vote_weight(self, i):
        return (1 / self.k) ** i
