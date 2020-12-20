import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR_VAR = "PROJECT_DIR"
PROJECT_DATA_DIR_VAR = "PROJECT_DATA_DIR"
PROJECT_MEDIA_DIR_VAR = "PROJECT_MEDIA_DIR"


@dataclass(frozen=True)
class Environ:
    """Groups the main environment variables of the project."""
    PROJECT_DIR: Path
    DATA_DIR: Path = None
    MEDIA_DIR: Path = None

    def __post_init__(self):
        """"""
        if self.DATA_DIR is None:
            data_dir = self.PROJECT_DIR / "data"
            object.__setattr__(self, "DATA_DIR", data_dir)
        if self.MEDIA_DIR is None:
            media_dir = self.PROJECT_DIR / "media"
            object.__setattr__(self, "MEDIA_DIR", media_dir)

    @classmethod
    def from_environ(cls):
        """"""
        project_dir_var = os.getenv(PROJECT_DIR_VAR)
        if project_dir_var is None:
            # NOTICE: When there is no PROJECT_DIR environment variable
            #  defined,  the
            #  current working directory will be used as the project
            #  directory.
            project_dir = Path.cwd().absolute()
        else:
            project_dir = Path(project_dir_var).resolve()
        data_dir_env = os.getenv(PROJECT_DATA_DIR_VAR)
        media_dir_env = os.getenv(PROJECT_DATA_DIR_VAR)
        data_dir = None if data_dir_env is None \
            else Path(data_dir_env).absolute()
        media_dir = None if media_dir_env is None \
            else Path(media_dir_env).absolute()
        return cls(project_dir, data_dir, media_dir)
