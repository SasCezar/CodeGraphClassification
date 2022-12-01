import ast
import csv
import json
import os
from collections import Counter, defaultdict
from os.path import join

import hydra
import pandas as pd
from hydra.utils import instantiate
from more_itertools import flatten
from omegaconf import DictConfig
from tqdm import tqdm

from feature.keyword_extraction import AbstractKeywordExtraction
from utils import filter_by_label


def extract_keywords(projects, cfg):
    project_keywords = {}
    kw_extractor: AbstractKeywordExtraction = instantiate(cfg.keyword.cls)
    term_freq = defaultdict(Counter)
    for project_name in tqdm(projects):
        content = load_content(join(cfg.content_dir, f"{project_name}.json"))
        term_freq[project_name].update(content)
        text = [" ".join(x).replace(".", " ") for x in content]
        keywords = kw_extractor.get_keywords(" ".join(text))
        project_keywords[project_name] = keywords

    return project_keywords, term_freq


def load_content(content_path):
    content = []
    with open(content_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            content.extend(obj["content"])
    return content


@hydra.main(config_path="../conf", config_name="keyword_extraction", version_base="1.2")
def extract_label_keyword(cfg: DictConfig):
    """
    Extracts the keywords of the labels using the projects of that label and the git history (augmented data).
    :param cfg:
    :return:
    """

    projects = pd.read_csv(cfg.project_path)
    #skip_projects = ['Waikato|weka-3.8', 'SonarSource|sonar-java', 'GenomicParisCentre|eoulsan']
    skip_projects = []

    projects_keyword, term_freq = extract_keywords(set(projects['name'].tolist()), cfg)

    labels = projects['label'].apply(ast.literal_eval).apply(tuple).tolist()
    labels = list(set(flatten(labels)))

    label_keywords = defaultdict(list)
    for label in tqdm(labels):
        projects_label = filter_by_label(projects.copy(deep=True), [label])
        projects_label = set(projects_label['name'].tolist())
        counter = Counter()
        term_count = Counter()

        for proj in projects_label:
            if proj in skip_projects:
                continue

            counter.update(projects_keyword[proj])
            term_count.update(term_freq[proj])
            label_keywords[label].append(projects_keyword[proj])

        triple = [(x[0], counter[x[0]], term_count[x[0]]) for x in counter.most_common()]
        df = pd.DataFrame(triple, columns=['keyword', 'doc_freq', 'total_freq'])

        os.makedirs(cfg.keywords_path, exist_ok=True)
        df.to_csv(os.path.join(cfg.keywords_path, f"{label}.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    extract_label_keyword()
