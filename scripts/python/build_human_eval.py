import glob
import json
import string
from collections import Counter, deque
from os.path import join
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def select_projects(file):
    projs = Counter()
    with open(file, 'rt') as inf:
        for line in inf:
            proj = json.loads(line)
            projs[proj['project']] += int(proj['total_nodes'])

    return [x[0] for x in projs.most_common(100)]


def load_annot(path, projects):
    projs = {}
    best_10 = {}
    with open(path, 'rt') as inf:
        for line in inf:
            proj = json.loads(line)
            if proj['project'] in projects:
                projs[proj['project']] = {'true': proj['true_labels'], 'pred': proj['predicted_labels']}

                best_10[proj['project']] = proj['predicted_labels'][:10]

    return projs, best_10


def load_package_annot(path, projects, best, mapping):
    annot = {}
    with open(path, 'rt') as inf:
        for line in inf:
            proj = json.loads(line)
            if proj['project'] in projects:
                packages_annot = {}
                for package in proj['packages']:
                    if proj['packages'][package]['unannotated']:
                        continue
                    distr = proj['packages'][package]['clean_distribution']
                    sorted_labels = np.argsort(distr)[::-1]

                    y_labels = []
                    for i in sorted_labels[:10]:
                        y_labels.append(mapping[i]) if mapping[i] in best[proj['project']] else None

                    packages_annot[package] = y_labels

                annot[proj['project']] = packages_annot

    return annot


def load_file_annot(path, projects, best, mapping):
    all_projects = [(Path(x).name.rsplit('-', maxsplit=2)[0], x) for x in glob.glob(join(path, '*.json'))]
    annotations = {}
    for project, annot_path in all_projects:
        if project in projects:
            with open(annot_path, 'rt') as inf:
                for line in inf:
                    proj = json.loads(line)
                    file_annot = {}
                    for file in proj:
                        if proj[file]['unannotated']:
                            continue
                        distr = proj[file]['distribution']
                        sorted_labels = np.argsort(distr)[::-1]

                        y_labels = []
                        for i in sorted_labels[:10]:
                            y_labels.append(mapping[i]) if mapping[i] in best[project] else None

                        if not y_labels:
                            y_labels = ['Other']
                        file_annot[file] = y_labels

            annotations[project] = file_annot

    return annotations


def build_df_other(annotations, level):
    tuples = []
    top = 3
    for project in annotations:
        for unit in annotations[project]:
            if unit == '.':
                continue
            labels = annotations[project][unit][:top]
            if len(labels) < top:
                labels.extend([''] * (top - len(labels)))
            url = f'https://github.com/{project.replace("|", "/")}/tree/master/{unit}'
            row = [project, unit, url] + labels
            tpl = tuple(row)

            tuples.append(tpl)

    lheader = [f'label_{i + 1}' for i in range(top)]
    header = ['project', level, 'url'] + lheader
    df = pd.DataFrame(tuples, columns=header)
    return df


def build_df_project(annotations):
    tuples = []
    for project in annotations:
        labels = annotations[project]['pred'][:10]
        for label in labels:
            if label not in annotations[project]['true']:
                tpl = (project, label)
                tuples.append(tpl)

    df = pd.DataFrame(tuples, columns=['project', 'label'])
    return df


def assign_annotator(annotators, elements):
    res = []
    num = elements // annotators
    last = None
    for a in string.ascii_lowercase[:annotators]:
        res.extend([a] * num)
        last = a

    if len(res) < elements:
        res.append([last] * abs(len(res) - elements))

    return res


@hydra.main(config_path="../../src/conf", config_name="annotation", version_base="1.2")
def build(cfg: DictConfig):
    annotators = 8

    proj_paths = join(cfg.project_labels_path, 'annotations.json')
    projects = select_projects(proj_paths)

    projects_annot, best = load_annot(proj_paths, projects)
    project_level_pred = build_df_project(projects_annot)

    annot_proj = assign_annotator(annotators, len(project_level_pred))
    project_level_pred['annotator'] = annot_proj
    annot_proj_2 = deque(annot_proj)
    annot_proj_2.rotate(len(project_level_pred) // annotators)
    project_level_pred['annotator_2'] = list(annot_proj_2)

    project_level_pred.to_csv('project_human_eval.csv')

    label_mapping = join(cfg.annotations_dir, f"label_mapping.json")

    with open(label_mapping, 'rt') as outf:
        label_map = json.load(outf)

    label_map = {v: k for k, v in label_map.items()}

    projects = projects[:10]
    package_annot = load_package_annot(join(cfg.package_labels_path, 'annotations.json'), projects, best, label_map)
    package_level_pred = build_df_other(package_annot, 'package')
    package_level_pred = package_level_pred.groupby('project').sample(n=100, random_state=1)
    package_level_pred.reset_index(drop=True, inplace=True)

    annot_pack = assign_annotator(annotators, len(package_level_pred))
    package_level_pred['annotator'] = annot_pack

    annot_pack_2 = deque(annot_pack)
    annot_pack_2.rotate(len(package_level_pred) // annotators)
    package_level_pred['annotator_2'] = list(annot_pack_2)

    package_level_pred.to_csv('package_human_eval.csv')

    file_annot = load_file_annot(cfg.annotations_path, projects, best, label_map)
    file_level_pred = build_df_other(file_annot, 'file')
    file_level_pred = file_level_pred.groupby('project').sample(n=100, random_state=1)
    file_level_pred.reset_index(drop=True, inplace=True)

    annot_file = assign_annotator(annotators, len(file_level_pred))
    file_level_pred['annotator'] = annot_file

    annot_file_2 = deque(annot_pack)
    annot_file_2.rotate(len(file_level_pred) // annotators)
    file_level_pred['annotator_2'] = list(annot_file_2)

    file_level_pred.to_csv('file_human_eval.csv')


if __name__ == '__main__':
    build()
