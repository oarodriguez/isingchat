import pathlib

import click

from .ising import ising

COMMANDS = {
    "ising": ising
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
