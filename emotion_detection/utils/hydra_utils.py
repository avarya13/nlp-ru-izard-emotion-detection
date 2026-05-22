from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_cfg(
    config_name: str,
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Path] = None,
) -> DictConfig:
    if config_dir is None:
        config_dir = Path(__file__).parents[2] / "configs"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir.resolve())
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg
