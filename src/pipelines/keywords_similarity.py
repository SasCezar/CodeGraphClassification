import ast
import os
from os.path import join

import hydra
import pandas as pd
from hydra.utils import instantiate
from more_itertools import flatten
from omegaconf import DictConfig
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from feature.embedding import AbstractEmbeddingModel


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def keyword_similarity(cfg: DictConfig):
    """
    Extracts the keywords of the labels using the projects of that label and the git history (augmented data).
    :param cfg:
    :return:
    """

    projects = pd.read_csv(cfg.project_path)

    labels = projects['label'].apply(ast.literal_eval).apply(tuple).tolist()
    labels = list(set(flatten(labels)))
    embedding_model: AbstractEmbeddingModel = instantiate(cfg.embedding.cls)

    for label in tqdm(labels):
        try:
            path = join(cfg.keywords_dir, "similarity", f"{label}.csv")
            if not os.path.exists(path) or cfg.force_new:
                path = join(cfg.keywords_dir, f"{label}.csv")
            keywords_df = pd.read_csv(path, dtype={0: str, 1: int})
        except:
            continue

        similarities = []
        label_emb = [embedding_model.get_embedding(label)]
        for i, key in enumerate(keywords_df['keyword']):
            tokvecs = [embedding_model.get_embedding(str(key))]
            sim = cosine_similarity(tokvecs, label_emb)[0][0]
            similarities.append(sim)

        keywords_df[f'{embedding_model.name}'] = similarities
        out_path = f"{cfg.keywords_dir}/similarity/"
        os.makedirs(out_path, exist_ok=True)
        keywords_df.to_csv(os.path.join(out_path, f"{label}.csv"), index=False)


if __name__ == '__main__':
    keyword_similarity()
