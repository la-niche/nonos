import sys
from typing import Union

from loguru import logger
from rich import print as rprint
from rich.logging import RichHandler


def configure_logger(level: Union[int, str] = 30, **kwargs) -> None:
    logger.remove()  # remove pre-existing handler
    logger.add(
        RichHandler(
            log_time_format="[%X] nonos",
            omit_repeated_times=False,
        ),
        format="{message}",
        level=level,
        **kwargs,
    )


def print_warn(message) -> None:
    """
    adapted from idefix_cli (cmt robert)
    https://github.com/neutrinoceros/idefix_cli
    """
    rprint(f":bone: [bold red]Warning[/] {message}", file=sys.stderr)


def print_err(message) -> None:
    """
    adapted from idefix_cli (cmt robert)
    https://github.com/neutrinoceros/idefix_cli
    """
    rprint(f":bone: [bold white on red]Error[/] {message}", file=sys.stderr)


def parse_verbose_level(verbose: int) -> str:
    levels = ["WARNING", "INFO", "DEBUG"]
    level = levels[min(len(levels) - 1, verbose)]  # capped to number of levels
    return level


configure_logger(level=30)
