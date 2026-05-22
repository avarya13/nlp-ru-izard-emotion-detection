from dvc.repo import Repo


def dvc_pull(remote: str = None, target: str = None) -> None:
    if target is None:
        raise ValueError("Target must be specified")
    try:
        with Repo() as repo:
            repo.pull(targets=[target], remote=remote)
    except Exception as e:
        print(f"Failed to pull from DVC: {e}")
        raise
