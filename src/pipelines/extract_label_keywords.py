import ast
import csv
import os
from collections import Counter
from os.path import join

import hydra
import pandas as pd
from hydra.utils import instantiate
from more_itertools import flatten
from omegaconf import DictConfig
from tqdm import tqdm

from feature.content import ContentExtraction
from feature.keyword_extraction import AbstractKeywordExtraction
from utils import git_clone, get_versions, git_checkout, filter_by_label


@hydra.main(config_path="../conf", config_name="keyword_extraction", version_base="1.2")
def extract_label_keyword(cfg: DictConfig):
    """
    Extracts the keywords of the labels using the projects of that label and the git history (augmented data).
    :param cfg:
    :return:
    """

    projects = pd.read_csv(cfg.project_path)

    labels = projects['label'].apply(ast.literal_eval).apply(tuple).tolist()
    labels = list(set(flatten(labels)))

    content_extractor: ContentExtraction = instantiate(cfg.content)

    kw_extractor: AbstractKeywordExtraction = instantiate(cfg.keyword)

    for label in tqdm(labels):
        projects_label = filter_by_label(projects.copy(deep=True), [label])
        projects_label = list(set(projects_label['name'].tolist()))
        term_count = Counter()
        identifiers = []
        content = {}
        for project_name in projects_label:
            project = project_name.replace('|', '/')

            project_url = f'https://github.com/{project}'
            if content_extractor.clone:
                git_clone(project_url, project_name, cfg.repositories_path)

            versions = get_versions(project_name, cfg.arcan_graphs)

            for num, sha in versions:
                try:
                    if content_extractor.clone:
                        git_checkout(join(cfg.repositories_path, project_name), sha)
                    res = [x[1] for x in content_extractor.extract(project_name, sha, num)]
                    identifiers.append(res)
                except:
                    continue

            text = [" ".join(x).replace(".", " ") for x in identifiers]
            content[project_name] = text
            term_count.update(flatten([x.split() for x in text]))
            # repo_path = join(cfg.repository_path, project_name)
            # shutil.rmtree(repo_path, ignore_errors=True)

        extracted_kw = {}
        counter = Counter()
        for project in content:
            extracted_kw[project] = kw_extractor.get_keywords(" ".join(content[project]).lower().strip())
            terms = [x[0] for x in extracted_kw[project]]
            counter.update(terms)

        triple = [(x[0], counter[x[0]], term_count[x[0]]) for x in counter.most_common()]
        df = pd.DataFrame(triple, columns=['keyword', 'doc_freq', 'total_freq'])

        out_path = f"{cfg.keywords_out}/{kw_extractor.name}/all/"
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(os.path.join(out_path, f"{label}.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    extract_label_keyword()
