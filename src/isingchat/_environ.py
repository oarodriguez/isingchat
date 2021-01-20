import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR_VAR = "PROJECT_DIR"
PROJECT_DATA_DIR_VAR = "PROJECT_DATA_DIR"
PROJECT_MEDIA_DIR_VAR = "PROJECT_MEDIA_DIR"
PROJECT_EDITABLE_MODE_VAR = "PROJECT_EDITABLE_MODE"

# NOTICE: The project_dir variable refers to our project base directory
#  only when we install our project package in editable mode, i.e.,
#  when we execute
#  -
#       poetry install
#  -
#  in the current python environment. Otherwise, if we should install
#  this package in non-editable mode, the built-in variable __file__ would
#  point to a different place.
file_dir = Path(__file__).parent
project_dir_edit = file_dir.parent.parent


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
            if bool(os.getenv(PROJECT_EDITABLE_MODE_VAR)):
                project_dir = project_dir_edit.absolute()
            else:
                # If the library is not installed in editable mode, the
                # current working directory will be used as the project
                # directory.
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
