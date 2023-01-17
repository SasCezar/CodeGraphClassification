import ast
import csv
import json
from collections import Counter, defaultdict
from os import makedirs
from os.path import join

import hydra
import pandas as pd
from hydra.utils import instantiate
from joblib import Parallel, delayed
from more_itertools import flatten
from omegaconf import DictConfig
from tqdm import tqdm

from feature.keyword_extraction import AbstractKeywordExtraction
from utils import filter_by_label


def extract_keywords(projects, cfg):
    project_keywords = {}

    term_freq = defaultdict(Counter)
    if cfg.num_workers > 1:
        with Parallel(n_jobs=cfg.num_workers) as parallel:
            results = parallel(delayed(extract_kw)(cfg, project) for project in tqdm(projects))
            for project, content_count, keywords in results:
                term_freq[project].update(content_count)
                project_keywords[project] = keywords
    else:
        for project_name in tqdm(projects):
            project, content_count, keywords = extract_kw(cfg, project_name)
            term_freq[project].update(content_count)
            project_keywords[project] = keywords

    return project_keywords, term_freq


def extract_kw(cfg, project_name):
    kw_extractor: AbstractKeywordExtraction = instantiate(cfg.keyword.cls)
    content = load_content(join(cfg.content_dir, f"{project_name}.json"))
    text = " ".join(content)
    tokens = text.split(" ")
    keywords = [x[0] for x in kw_extractor.get_keywords(text)]
    content_count = Counter(tokens)
    return project_name, content_count, keywords


def load_content(content_path):
    content = []
    with open(content_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            for filename in obj['content']:
                content.append(obj["content"][filename])
    return content


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def extract_label_keyword(cfg: DictConfig):
    """
    Extracts the keywords of the labels using the projects of that label and the git history (augmented data).
    :param cfg:
    :return:
    """

    projects = pd.read_csv(cfg.project_path)
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

        makedirs(cfg.keywords_dir, exist_ok=True)
        df.to_csv(join(cfg.keywords_dir, f"{label}.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    extract_label_keyword()
