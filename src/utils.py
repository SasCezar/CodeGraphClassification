import ast
import glob
import os
import random
from os.path import basename, join
from subprocess import call
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
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
    return res


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the labels of a dataframe.
    :param df:
    :return:
    """
    df['label'] = df['label'].apply(ast.literal_eval).apply(tuple)
    df['level'] = df['level'].apply(ast.literal_eval).apply(tuple)
    labels = list(set(flatten(df['label'].tolist())))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    res = df['label'].apply(lambda x: label_encoder.transform(x))
    df['labels_id'] = res
    return df.copy(deep=True)
