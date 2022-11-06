import ast
from collections import Counter
from os.path import join

import hydra
import pandas as pd
import yake
from RAKE import Rake
from more_itertools import flatten
from omegaconf import DictConfig
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from feature.content import NameContentExtraction
from feature.embedding import FastTextEmbedding, W2VEmbedding
from utils import git_clone, get_versions, git_checkout, filter_by_label


@hydra.main(config_path="../conf", config_name="main", version_base="1.2")
def extract_label_keyword(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """

    projects = pd.read_csv("/home/sasce/PycharmProjects/CodeGraphClassification/data/raw/dataset_with_graphs.csv")

    labels = projects['label'].apply(ast.literal_eval).apply(tuple).tolist()
    labels = list(set(flatten(labels)))

    content_extractor = NameContentExtraction(graph_path=cfg.arcan_graphs)

    #ft = FastTextEmbedding(path='/home/sasce/PycharmProjects/CodeGraphClassification/data/models/wiki.en.bin',
    #                       model='fastText')

    ft = W2VEmbedding(path='/home/sasce/PycharmProjects/CodeGraphClassification/data/models/SO_vectors_200.bin',
                        model='w2v')
    with open("/home/sasce/PycharmProjects/CodeGraphClassification/data/raw/java_stopwords", 'rt') as inf:
        java_stopwords = {x.strip() for x in inf.readlines()}

    # java_stopwords.update(['get', 'set', 'org', 'com', 'exception', 'override', 'java', 'string',
    #                        'list', 'util', 'value', 'length', 'println'])

    kw_extractor = yake.KeywordExtractor(n=1, stopwords=java_stopwords, top=50)

    #rake = Rake("/home/sasce/PycharmProjects/CodeGraphClassification/data/raw/java_stopwords")

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
                    res = content_extractor.extract(project_name, sha, num)
                    identifiers.append(res)
                except:
                    continue

            text = [" ".join(x).replace(".", " ") for x in identifiers]
            content[project_name] = text
            term_count.update(flatten([x.split() for x in text]))
            # repo_path = join(cfg.repository_path, project_name)
            # shutil.rmtree(repo_path, ignore_errors=True)

        extracted_kw = {}

        for project in content:
            extracted_kw[project] = kw_extractor.extract_keywords(" ".join(content[project]).lower().strip())
            #extracted_kw[project] = rake.run(" ".join(content[project]).lower().strip(), minCharacters=3, maxWords=1,
            #                                 minFrequency=1)

        counter = Counter()
        all_terms = set()
        for p in extracted_kw:
            terms = [x[0] for x in extracted_kw[p]]
            all_terms.update(terms)
            counter.update(terms)

        label_emb = [ft.get_embedding(label)]
        keywords = [x[0] for x in counter.most_common()]

        similarities = []
        for i, key in tqdm(enumerate(keywords)):
            tokvecs = [ft.get_embedding(key)]
            sim = cosine_similarity(tokvecs, label_emb)[0][0]
            similarities.append((keywords[i], sim))

        similarities.sort(key=lambda x: -x[1])

        triples = [(x[0], x[1], term_count[x[0]]) for x in similarities]

        df = pd.DataFrame(triples, columns=['keyword', 'similarity', 'frequency'])

        df.to_csv(f"/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/keywords/SO/rake/all/{label}.csv",
                  index=False)


if __name__ == '__main__':
    extract_label_keyword()
