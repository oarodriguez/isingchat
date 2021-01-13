from dataclasses import dataclass

from click import ClickException


@dataclass
class CLIError(ClickException):
    """Represent a CLI Exception."""
    message: str
    label: str = "[red]Error[/]"

    def __post_init__(self):
        """"""
        super().__init__(self.message)

    def show(self, file=None):
        """"""
        message = self.format_message()
        print(f"{self.label}: {message}")
