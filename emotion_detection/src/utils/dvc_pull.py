from dvc.repo import Repo


def dvc_pull(remote="ruizard-emotions-data", target="data/ru-izard-emotions.dvc"):
    """Pull data or model from DVC remote"""
    try:
        with Repo() as repo:
            repo.pull(targets=[target], remote=remote)
    except Exception as e:
        print(f"Failed to pull from DVC: {e}")
        raise
