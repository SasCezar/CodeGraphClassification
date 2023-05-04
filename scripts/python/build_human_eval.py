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
from pandas import DataFrame


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

                    if y_labels:
                        packages_annot[package] = y_labels

                if packages_annot:
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
            url = f'https://github.com/{project.replace("|", "/")}'
            row = [project, url, unit] + labels
            tpl = tuple(row)

            tuples.append(tpl)

    lheader = [f'label_{i + 1}' for i in range(top)]
    header = ['project', 'url', level] + lheader
    df = pd.DataFrame(tuples, columns=header)
    return df


def build_df_project(annotations):
    tuples = []
    for project in annotations:
        labels = annotations[project]['pred'][:10]
        for label in labels:
            if label not in annotations[project]['true']:
                url = f'https://github.com/{project.replace("|", "/")}'
                tpl = (project, url, label)
                tuples.append(tpl)

    df = pd.DataFrame(tuples, columns=['project', 'url', 'label'])
    return df


def assign_annotator(annotators, elements):
    res = []
    num = elements // annotators
    last = None
    for a in string.ascii_uppercase[:annotators]:
        res.extend([a] * num)
        last = a

    if len(res) < elements:
        res.append([last] * abs(len(res) - elements))

    return res


def assign_annotators(annotators, package_level_pred, k=2):
    pred_w_annot = []
    annot_pack = assign_annotator(annotators, len(package_level_pred))
    for i in range(k):
        packag_pred = package_level_pred.copy(deep=True)
        annot_pack = deque(annot_pack)
        annot_pack.rotate(len(package_level_pred) // annotators)
        packag_pred['annotator'] = list(annot_pack)
        pred_w_annot.append(packag_pred)

    pred_w_annot = pd.concat(pred_w_annot)
    return pred_w_annot


def to_xlsl(df, filename):
    writer = pd.ExcelWriter(filename)
    df['url'] = df['url'].apply(lambda x: make_hyperlink(x))

    for group, data in df.groupby('annotator'):
        data.to_excel(writer, sheet_name=group)
    writer.save()


def make_hyperlink(url):
    return f'=HYPERLINK("{url}", "{url}")'


@hydra.main(config_path="../../src/conf", config_name="annotation", version_base="1.2")
def build(cfg: DictConfig):
    annotators = 8

    proj_paths = join(cfg.project_labels_path, 'annotations.json')
    projects = select_projects(proj_paths)

    projects_annot, best = load_annot(proj_paths, projects)
    project_level_pred = build_df_project(projects_annot)

    project_level_pred_annot = assign_annotators(annotators, project_level_pred)

    project_filename = f'project_human_eval_{cfg.ensemble.name.replace("/", "-")}.csv'
    project_level_pred_annot.to_csv(project_filename)

    to_xlsl(project_level_pred_annot, project_filename.replace('csv', 'xlsx'))

    label_mapping = join(cfg.annotations_dir, f"label_mapping.json")

    with open(label_mapping, 'rt') as outf:
        label_map = json.load(outf)

    label_map = {v: k for k, v in label_map.items()}

    # projects = projects[:10]
    package_annot = load_package_annot(join(cfg.package_labels_path, 'annotations.json'), projects, best, label_map)
    package_level_pred: DataFrame = build_df_other(package_annot, 'package')
    package_level_pred = package_level_pred.groupby('project').sample(n=10, random_state=1)
    package_level_pred.reset_index(drop=True, inplace=True)

    package_level_pred_annot = assign_annotators(annotators, package_level_pred)

    package_filename = f'package_human_eval_{cfg.ensemble.name.replace("/", "-")}.csv'
    package_level_pred_annot.to_csv(package_filename)

    to_xlsl(package_level_pred_annot, package_filename.replace('csv', 'xlsx'))

    file_annot = load_file_annot(cfg.annotations_path, projects, best, label_map)
    file_level_pred = build_df_other(file_annot, 'file')
    file_level_pred = file_level_pred.groupby('project').sample(n=10, random_state=1)
    file_level_pred.reset_index(drop=True, inplace=True)

    file_level_pred_annot = assign_annotators(annotators, file_level_pred)
    file_filename = f'file_human_eval_{cfg.ensemble.name.replace("/", "-")}.csv'
    file_level_pred_annot.to_csv(file_filename)
    to_xlsl(file_level_pred_annot, file_filename.replace('csv', 'xlsx'))


if __name__ == '__main__':
    build()
