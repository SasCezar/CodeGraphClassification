from pathlib import Path

import numpy as np
import pandas
from gensim import corpora, models
from loguru import logger
from more_itertools import flatten


def compute_keywords_tfidf():
    keywords_path = Path("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/keywords/yake/paper")

    keywords_files = list(keywords_path.glob("*.csv"))
    labels = []
    doc_tokenized = []
    for keywords_file in keywords_files:
        label = keywords_file.stem
        labels.append(label)
        df = pandas.read_csv(keywords_file)
        try:
            df.drop('similarity', axis=1, inplace=True)
        except:
            pass
        terms = df.values.tolist()

        doc_tokenized.append(list(flatten([[str(term)] * n for term, _, n in terms])))

    # logger.info(f"Tokeinzing {len(docs)} documents")
    # doc_tokenized = [simple_preprocess(doc) for doc in docs]
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
        df = pandas.read_csv(keywords_file)
        df['tfidf'] = df['keyword'].map(label_terms_tfidf[i])
        df.to_csv(
            f"/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/keywords/yake/similarity/{label}.csv",
            index=False)


if __name__ == '__main__':
    compute_keywords_tfidf()
