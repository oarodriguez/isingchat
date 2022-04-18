import pathlib

import click

from .commands.run import run
from .commands.eigens import eigens
from .commands.correlation_function import correlation_function
from .commands.correlation_length import correlation_length


COMMANDS = {
    "run": run,
    "eigens": eigens,
    "corr_length": correlation_length,
    "corr_function": correlation_function
}

file_dir = pathlib.Path(__file__).parent


class CLI(click.MultiCommand):
    """Gather the commands for different tasks."""

    def list_commands(self, ctx):
        """Return the list of commands."""
        return sorted(COMMANDS)

    def get_command(self, ctx, cmd_name: str):
        """Return the commands."""
        return COMMANDS.get(cmd_name, None)


@click.command(cls=CLI)
def main():
    """CLI of spinchains-manybody library."""
    pass
