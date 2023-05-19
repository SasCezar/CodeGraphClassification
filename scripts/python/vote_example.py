import json
from os.path import join

import numpy as np
import pandas as pd
from hydra import initialize, compose


def create_example():
    with initialize(version_base=None, config_path="../../src/conf/"):
        cfg = compose(config_name='annotation.yaml', overrides=["local=default"])
    project = 'Waikato|weka-3.8-903-04804ccd6dff03534cbf3f2a71a35c73eef24fe8.json'
    with open(f"{cfg.base_path}/data/processed/annotations/label_mapping.json",
              'rt') as inf:
        label_map = json.load(inf)
        inverse_map = {v: k for k, v in label_map.items()}

    annot_path = f"{cfg.base_path}/data/processed/annotations"
    methods = ['keyword/name/yake/none/none', 'similarity/name/w2v-so/none/none',
               'ensemble/best/voting/none/none']

    # file = 'weka/src/main/java/weka/classifiers/functions/SimpleLogistic.java'
    file = 'packages/internal/classificationViaClustering/src/main/java/weka/classifiers/meta/ClassificationViaClustering.java'
    for method in methods:
        table = []
        annot_file_path = join(annot_path, method, project)
        with open(annot_file_path) as inf:
            obj = json.load(inf)

        file_annot = obj[file]['distribution']
        print(obj[file]['unannotated'])

        sorted_labels = np.argsort(file_annot)[::-1]

        for x, i in enumerate(sorted_labels[:10]):
            row = [inverse_map[i].title(), 10 - x, str(file_annot[i])[:6]]
            table.append(row)

        df = pd.DataFrame(table, columns=['Label', 'Weight', 'Prob'])

        print(df.to_latex(index=False))


if __name__ == '__main__':
    create_example()
