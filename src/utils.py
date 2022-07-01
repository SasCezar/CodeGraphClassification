import glob
import os
from os.path import basename, join
from subprocess import call
from typing import Tuple, List


def check_dir(path):
    """
    Checks if the directory exists and creates it if not.
    :param path:
    :return:
    """
    project_path = os.path.join(path)
    os.makedirs(project_path, exist_ok=True)


def git_clone(repo_url, name, out_path, force=False):
    out_folder = os.path.join(out_path, name)
    if os.path.exists(out_folder):
        if not force:
            return
        os.rmdir(out_folder)

    call(["git", "clone", repo_url, out_folder])
    return


def git_checkout(repo_path, sha):
    call(["git", "checkout", sha], cwd=repo_path)
    return


def get_versions(project, arcan_out) -> List[Tuple[str, str]]:
    files = [basename(x) for x in glob.glob(join(arcan_out, project, "*"))]
    res = []
    for file in files:
        num, sha = file.replace('.graphml', '').split("-")[-1].split("_")
        res.append((num, sha))
    return res
