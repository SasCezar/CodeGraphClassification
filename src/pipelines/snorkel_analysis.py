import glob
import json
from collections import defaultdict
from itertools import combinations
from os.path import join

import hydra
import numpy as np
import pandas
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from tqdm import tqdm

from utils import parse_settings


def load_annot(files):
    annot = defaultdict(dict)
    projects = set()
    true = {}
    for file_path, filename in files:
        with open(file_path, 'rt') as inf:
            for line in inf:
                proj = json.loads(line)
                pred = proj['predicted_labels'][0]
                annot[filename][proj['project']] = pred
                projects.add(proj['project'])
                labels = proj['true_labels']

                if proj['project'] not in true:
                    true[proj['project']] = labels

    return projects, annot, true


class LF:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


def make_matrix(projects, lf_annot, mapping):
    annotations = []

    names = [LF(x) for x in lf_annot.keys()]
    for p in projects:
        annot = []
        for lf in lf_annot:
            label = mapping[lf_annot[lf][p]] if p in lf_annot[lf] else -1
            annot.append(label)

        annotations.append(annot)

    return names, np.array(annotations)


def load_files(path):
    files = glob.glob(path + '/**/*.json', recursive=True)
    files = [(x, x.replace(path, '').replace('/annotations.json', '')) for x in files if
             'single_label' not in x and 'soft_label' not in x and
             'ensemble' not in x]  # and ('w2v-so' in x or 'keyword' in x)]
    return files


def load_mapping(annotations_dir):
    with open(join(annotations_dir, 'label_mapping.json'), 'rt') as inf:
        return json.load(inf)


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.3")
def analyze(cfg: DictConfig):
    files = load_files(cfg.project_labels_dir)

    mapping = load_mapping(cfg.annotations_dir)
    projects, lf_annot, true = load_annot(files)

    true = [[mapping[l] for l in true[p]] for p in projects]

    names, lf_annot_matrix = make_matrix(projects, lf_annot, mapping)

    analysis_summary = LFAnalysis(L=lf_annot_matrix, lfs=names).lf_summary()

    analysis_summary['Polarity'] = analysis_summary['Polarity'].apply(lambda x: len(x))

    analysis_summary.index.names = ['LF']
    analysis_summary.reset_index(inplace=True)

    analysis_summary['LF'] = analysis_summary['LF'].apply(lambda x: parse_settings(x))

    split = pd.DataFrame(analysis_summary['LF'].to_list(),
                         columns=['annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold'])

    # join split columns back to original DataFrame
    # analysis_summary = pd.concat([split, analysis_summary], axis=1)
    analysis_summary.drop(columns=['j', 'LF'], inplace=True)
    analysis_summary = analysis_summary.transpose()
    multi_index = pd.MultiIndex.from_frame(split)
    analysis_summary = pd.DataFrame(analysis_summary.to_numpy(), columns=multi_index, index=analysis_summary.index)
    analysis_summary.reset_index(inplace=True)
    with pd.option_context("max_colwidth", 100):
        print(analysis_summary.to_latex(float_format="{:0.2f}".format,
                                        index=False,
                                        escape=True,
                                        sparsify=True,
                                        multirow=True,
                                        multicolumn=True,
                                        multicolumn_format='c',
                                        position='htbp'))

    label_model = LabelModel(cardinality=len(mapping), verbose=False)
    label_model.fit(lf_annot_matrix, n_epochs=100, seed=123)
    pred = []
    _, pred_prob = label_model.predict(lf_annot_matrix, tie_break_policy="random", return_probs=True)

    for i in range(len(pred_prob)):
        sorted_labels = np.argsort(pred_prob[i])[::-1]
        top_k = sorted_labels[:10]
        pred.append(top_k)

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(true)
    y_pred = mlb.transform(pred)

    recall = recall_score(y_true, y_pred, average='samples')
    print(recall)
    # lf_weights = label_model.get_weights()
    # print(lf_weights)
    # aggregated_prob = label_model.get_conditional_probs()
    # print(aggregated_prob)

    majority_model = MajorityLabelVoter(cardinality=len(mapping))
    preds_train = majority_model.predict(L=lf_annot_matrix)

    pred = []

    for i in range(len(preds_train)):
        sorted_labels = np.argsort(preds_train[i])[::-1]
        top_k = sorted_labels[:10]
        pred.append(top_k)

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(true)
    y_pred = mlb.transform(pred)
    recall = recall_score(y_true, y_pred, average='samples')
    print(recall)
    # majority_acc = majority_model.score(L=lf_annot_matrix, Y=true, tie_break_policy="random")[
    #     "accuracy"
    # ]
    # print(majority_acc)
    #
    # print(recall_score(preds_train, true, average='samples'))


def load_annot_all(files):
    annot = defaultdict(dict)
    projects = set()
    polarity = defaultdict(set)
    num_labels = set()
    for file_path, filename in files:
        with open(file_path, 'rt') as inf:
            for line in inf:
                proj = json.loads(line)
                pred = proj['predicted_labels'][:10]
                num_labels.update(proj['true_labels'])
                annot[filename][proj['project']] = pred
                projects.add(proj['project'])
                polarity[filename].update(pred)

    print(len(num_labels))
    return projects, annot, polarity


def polarity_latex(polarity):
    index = [parse_settings(k) for k in polarity]
    polarity = [(*parse_settings(k), len(polarity[k])) for k in polarity]

    index = pandas.MultiIndex.from_frame(
        pd.DataFrame(index, columns=['annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold']))
    polarity = pd.DataFrame(polarity,
                            columns=['annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold',
                                     'polarity'])
    polarity.to_csv('polarity.csv', index=False)
    tex = pd.DataFrame(data=polarity['polarity'].tolist(), index=index).transpose()
    tex.reset_index(inplace=True)
    tex = tex.to_latex(float_format="{:0.2f}".format,
                       index=False,
                       escape=True,
                       sparsify=True,
                       multirow=True,
                       multicolumn=True,
                       multicolumn_format='c',
                       position='htbp')

    print(tex)


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.3")
def pairwise_stats(cfg: DictConfig):
    files = load_files(cfg.project_labels_dir)
    pairs = list(combinations(files, 2))
    print(len(pairs))

    projects, lf_annot, polarity = load_annot_all(files)
    polarity_latex(polarity)

    res = Parallel(n_jobs=10)(delayed(lf_agreement)(a, b, lf_annot) for a, b in tqdm(pairs))
    conflicts = pd.concat(res)

    conflicts.to_csv('agreement_all.csv', index=False)
    aggregated = \
        conflicts.groupby(['x_annotation', 'x_content', 'x_algorithm', 'x_transformation', 'x_filtering', 'x_threshold',
                           'y_annotation', 'y_content', 'y_algorithm', 'y_transformation', 'y_filtering', 'y_threshold',
                           'k'])['agreement', 'agreement_percent'].mean()
    aggregated.reset_index(inplace=True)
    aggregated.to_csv('aggregated_agreement_all.csv', index=False)


def lf_agreement(a, b, lf_annot):
    agreements = pd.DataFrame(
        columns=['x_annotation', 'x_content', 'x_algorithm', 'x_transformation', 'x_filtering', 'x_threshold',
                 'y_annotation', 'y_content', 'y_algorithm', 'y_transformation', 'y_filtering', 'y_threshold',
                 'project', 'k', 'agreement', 'agreement_percent'])

    projs = set(lf_annot[a[1]].keys()).union(lf_annot[b[1]].keys())
    sett_a = parse_settings(a[1])
    sett_b = parse_settings(b[1])
    for p in tqdm(projs, position=0, leave=True):
        for k in [3, 5, 10]:
            proj_a = set(lf_annot[a[1]][p][:k]) if p in lf_annot[a[1]] else set()
            proj_b = set(lf_annot[b[1]][p][:k]) if p in lf_annot[b[1]] else set()
            agreement = len(proj_a.intersection(proj_b))
            row = sett_a + sett_b + [p, k, agreement, agreement / k]
            agreements.loc[len(agreements)] = row
            row = sett_b + sett_a + [p, k, agreement, agreement / k]
            agreements.loc[len(agreements)] = row

    return agreements


if __name__ == '__main__':
    # analyze()
    pairwise_stats()
