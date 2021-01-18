import sys
import typing as t

from dask.callbacks import Callback
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.progress import BarColumn, Progress, TextColumn

# Rich output console instances.
console = Console()
err_console = Console(file=sys.stderr)


class RichProgressBar(Progress):
    """A slightly modified rich progress bar."""

    def get_renderables(self) -> t.Iterable[RenderableType]:
        """"""
        yield Padding(self.make_tasks_table(self.tasks), (1, 1))


# ProgressBar columns.
columns = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("|"),
    TextColumn("[progress.remaining]Elapsed time: {task.elapsed:>.2f}s"),
)


class DaskProgressBar(Callback):
    """Progress bar for dask computations."""

    def _start_state(self, dsk, state):
        """"""
        total = sum(len(state[k]) for k in ["ready", "waiting", "running"])
        progress = RichProgressBar(*columns, console=console,
                                   auto_refresh=False)
        self.rich_progress = progress
        self.main_task = progress.add_task("[red]Progress", total=total)
        progress.start()

    def _posttask(self, key, result, dsk, state, worker_id):
        """"""
        progress_task = self.main_task
        self.rich_progress.update(progress_task, advance=1)
        self.rich_progress.refresh()

    def _finish(self, dsk, state, errored):
        self.rich_progress.stop()


def bin_digits(value: int, length: int = None):
    """Get a list with the binary digits of an integer.

    Fill the list with zeros to the left to get a list of length ``length``.
    If ``length`` is ``None``, return only the value digits.
    """
    length = length or 0
    fmt_str = f"{{0:0{length}b}}"
    return [int(v) for v in fmt_str.format(value)]
