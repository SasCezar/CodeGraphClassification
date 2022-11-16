import ast
import glob
import os
from collections import Counter
from os.path import basename, join
from subprocess import call
from typing import Tuple, List, Dict

import igraph
import pandas as pd
from loguru import logger
from more_itertools import flatten
from sklearn import preprocessing


def check_dir(path: str) -> None:
    """
    Checks if the directory exists and creates it if not.
    :param path:
    :return:
    """
    project_path = os.path.join(path)
    os.makedirs(project_path, exist_ok=True)
    return


def git_clone(repo_url: str, name: str, out_path: str, force: bool = False) -> None:
    """
    Clones a repository into the out_path.
    :param repo_url: URL of the repository
    :param name: Project name
    :param out_path: Output folder
    :param force: If true, the repository will be cloned even if it already exists
    :return:
    """
    out_folder = os.path.join(out_path, name)
    if os.path.exists(out_folder):
        if not force:
            return
        os.rmdir(out_folder)

    call(["git", "clone", repo_url, out_folder])
    return


def git_checkout(repo_path: str, sha: str) -> None:
    """
    Checks out a specific commit in a repository.
    :param repo_path: Path of the repository
    :param sha: Version to checkout
    :return:
    """
    call(["git", "checkout", sha], cwd=repo_path)
    return


def get_versions(project: str, arcan_out: str) -> List[Tuple[str, str]]:
    """
    Returns a list of tuples (version, sha) for a project. The version, is the number of the commit in the git history.
    :param project: Project name
    :param arcan_out: Arcan output folder
    :return:
    """
    files = [basename(x) for x in glob.glob(join(arcan_out, project, "*"))]
    res = []
    for file in files:
        num, sha = file.replace('.graphml', '').split("-")[-1].split("_")
        res.append((num, sha))
    res.sort(key=lambda x: int(x[0]))
    return res


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the labels of a dataframe.
    :param df:
    :return:
    """
    if df.label.dtype == str:
        df['label'] = df['label'].apply(ast.literal_eval).apply(tuple)
    labels = list(set(flatten(df['label'].tolist())))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    res = df['label'].apply(lambda x: label_encoder.transform(x))
    df['labels_id'] = res
    return df.copy(deep=True)


def only_top_labels(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """
    Keeps only the top labels.
    :param df:
    :param top:
    :return:
    """
    df.drop('level', axis=1, inplace=True)
    labels = df['label'].apply(ast.literal_eval).apply(tuple).tolist()
    labels = list(flatten(labels))
    count = Counter(labels)
    most_common = [x[0] for x in count.most_common(top)]
    df = filter_by_label(df, most_common)
    return df


def filter_by_label(df, labels):
    logger.info(f"Labels {labels}")
    df['label'] = df['label'].apply(ast.literal_eval).apply(tuple)
    df['label'] = df['label'].apply(lambda x: [y for y in x if y in labels])
    df = df[df['label'].apply(lambda x: len(x) > 0)]
    df['label'] = df['label'].apply(tuple)
    return df


def node_package_mapping(graph: igraph.Graph) -> Dict[str, str]:
    """
    Returns a dictionary mapping node ids to package names.
    :param graph:
    :return:
    """
    res = {}
    for edge in graph.es:
        v1, v2 = edge.source, edge.target
        if v1['labelV'] == 'container':
            res[v1['filePathRelative']] = v2['filePathRelative']
    return res