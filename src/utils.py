import os
from subprocess import call


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
