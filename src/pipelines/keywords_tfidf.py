from os.path import join
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from gensim import corpora, models
from loguru import logger
from more_itertools import flatten
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def keywords_tfidf(cfg: DictConfig):
    keywords_path = Path(cfg.keywords_dir)

    keywords_files = list(keywords_path.glob("*.csv"))
    labels = []
    doc_tokenized = []
    for keywords_file in keywords_files:
        label = keywords_file.stem
        labels.append(label)
        df = pd.read_csv(keywords_file)
        try:
            df.drop('similarity', axis=1, inplace=True)
        except:
            pass
        terms = df.values.tolist()

        doc_tokenized.append(list(flatten([[str(term)] * n for term, _, n in terms])))

    logger.info(f"Creating dictionary")
    dictionary = corpora.Dictionary()
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]

    logger.info(f"Creating TF-IDF model")
    tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')

    label_terms_tfidf = []
    for doc in tfidf[BoW_corpus]:
        label_terms_tfidf.append({dictionary[id]: np.around(freq, decimals=2) for id, freq in doc})

    for i, keywords_file in enumerate(keywords_files):
        label = keywords_file.stem
        df = pd.read_csv(keywords_file)
        df['tfidf'] = df['keyword'].map(label_terms_tfidf[i])
        df.to_csv(join(cfg.keywords_dir, "similarity", f"{label}.csv"),
                  index=False)


if __name__ == '__main__':
    keywords_tfidf()
