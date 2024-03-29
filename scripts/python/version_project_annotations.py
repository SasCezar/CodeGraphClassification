import glob
import json

import numpy as np
import pandas
from hydra import initialize, compose

with initialize(version_base=None, config_path="../../src/conf/"):
    cfg = compose(config_name='annotation.yaml', overrides=["local=default"])

projects = sorted(glob.glob(f"{cfg.base_path}/processed/annotations/name/*weka*.csv"),
                  key=lambda x: int(x.split('-')[-2]))
method = 'name'

with open(f"{cfg.base_path}/data/processed/annotations/{method}/label_mapping.json",
          'rt') as inf:
    label_map = json.load(inf)

proj_mean = []
for p, path in enumerate(projects):
    df = pandas.read_csv(path)
    df.drop('node', axis=1, inplace=True)
    agg = df.sum(axis=0)
    agg = np.array(agg) #/ np.linalg.norm(agg)
    proj_mean.extend([(p, i, x) for i, x in enumerate(agg)])

proj_labs = []
for p, i, m in proj_mean:
    proj_labs.extend([[p, i]] * int(m * 100))
mean_df = pandas.DataFrame(proj_labs, columns=['project', 'label', ])
mean_df.to_csv(
    f"{cfg.base_path}/data/processed/weka-java_versions.csv",
    index=False, header=True)
