import json
import os

import numpy as np
import pandas
from hydra import initialize, compose

with initialize(version_base=None, config_path="../../src/conf/"):
    cfg = compose(config_name='node_annotation.yaml', overrides=["local=default"])

projects = ['Waikato|weka-3.8-903-04804ccd6dff03534cbf3f2a71a35c73eef24fe8.csv',
            'SonarSource|sonar-java-8464-6400749499be832a3e37fa2f6beed47f47c04f36.csv',
            'GenomicParisCentre|eoulsan-5391-0fac2ac7fa5da73c794824de1dbf216099ebefc2.csv']

projects_map = {'Waikato|weka-3.8-903-04804ccd6dff03534cbf3f2a71a35c73eef24fe8.csv': 'Weka',
                'SonarSource|sonar-java-8464-6400749499be832a3e37fa2f6beed47f47c04f36.csv': 'Sonar',
                'GenomicParisCentre|eoulsan-5391-0fac2ac7fa5da73c794824de1dbf216099ebefc2.csv': 'Eoulsan'}

method = 'name'

with open(f"/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/annotations/{method}/label_mapping.json",
          'rt') as inf:
    label_map = json.load(inf)

proj_mean = []
for project in projects:
    path = os.path.join(cfg.annotations_path, method, project)

    df = pandas.read_csv(path)
    df.drop('node', axis=1, inplace=True)
    agg = df.mean(axis=0)
    agg = np.array(agg) / np.linalg.norm(agg)
    proj_mean.extend([(projects_map[project], i, x) for i, x in enumerate(agg)])

proj_labs = []
for p, i, m in proj_mean:
    proj_labs.extend([[p, i]] * int(m * 100))
mean_df = pandas.DataFrame(proj_labs, columns=['project', 'label'])
mean_df.to_csv(
    f"/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/annotations/{method}/projects_mean.csv",
    index=False, header=True)
